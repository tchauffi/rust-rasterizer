struct SceneUniform {
    resolution: vec4<u32>, // x: width, y: height, z: triangle count, w: sphere count
    camera_position: vec4<f32>,
    lower_left_corner: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    light_direction: vec4<f32>, // xyz: direction, w: strength
    light_color: vec4<f32>,
    ambient_color: vec4<f32>,
    mesh_color: vec4<f32>,
    mesh_material: vec4<f32>, // x: roughness, y: metallic, z: material_type, w: unused
    render_config: vec4<u32>,
    accel_info: vec4<u32>,
};

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
};

struct Sphere {
    center_radius: vec4<f32>, // xyz: center, w: radius
    color: vec4<f32>,          // xyz: color, w: unused
    material: vec4<f32>,       // x: roughness, y: metallic, z: material_type, w: unused
};

struct BvhNode {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    left_first: u32,
    prim_count: u32,
    right_child: u32,
    _padding: u32,
};

struct HitInfo {
    dist: f32,
    normal: vec3<f32>,
    color: vec3<f32>,
    hit: bool,
    roughness: f32,
    metallic: f32,
    material_type: f32,
};

const PI: f32 = 3.141592653589793;

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> triangles: array<Triangle>;

@group(0) @binding(2)
var<storage, read> spheres: array<Sphere>;

@group(0) @binding(3)
var<storage, read_write> image: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> bvh_nodes: array<BvhNode>;

@group(0) @binding(5)
var environment_map: texture_2d<f32>;

@group(0) @binding(6)
var environment_sampler: sampler;

const BVH_STACK_SIZE: u32 = 64u;
const LARGE_DISTANCE: f32 = 1e30;

// PCG3D: a high-quality integer hash (Mark Jarzynski & Marc Olano, JCGT 2020).
// Replaces the old fract(sin(...)) hash, which had poor decorrelation and
// relied on sin() precision that varies between GPUs.
fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return v;
}

fn hash_float3(value: vec3<u32>) -> f32 {
    // Map the top word of the PCG3D output to [0, 1).
    return f32(pcg3d(value).x) * (1.0 / 4294967296.0);
}

fn random2(pixel: vec2<u32>, sample: u32, frame_seed: u32) -> vec2<f32> {
    let seed0 = vec3<u32>(pixel.x, pixel.y, sample + frame_seed * 1000u);
    let seed1 = seed0 + vec3<u32>(17u, 59u, 83u);
    return vec2<f32>(hash_float3(seed0), hash_float3(seed1));
}

fn random_float(seed: vec3<u32>) -> f32 {
    return hash_float3(seed);
}

fn random_unit_vector(seed: vec3<u32>) -> vec3<f32> {
    let u = random_float(seed);
    let v = random_float(seed + vec3<u32>(1u, 1u, 1u));
    let theta = 2.0 * PI * u;
    let z = v * 2.0 - 1.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(cos(theta) * r, sin(theta) * r, z);
}

fn random_pair(seed: vec3<u32>) -> vec2<f32> {
    let r = pcg3d(seed);
    return vec2<f32>(f32(r.x), f32(r.y)) * (1.0 / 4294967296.0);
}

fn reflect(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Cosine-weighted hemisphere sampling around `normal`. The cosine term in the
// diffuse rendering equation cancels the pdf (cos/PI), so paths sampled this way
// need only weight throughput by the surface albedo.
fn cosine_sample_hemisphere(normal: vec3<f32>, seed: vec3<u32>) -> vec3<f32> {
    let xi = random_pair(seed);
    let r = sqrt(xi.x);
    let theta = 2.0 * PI * xi.y;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - xi.x));

    // Build an orthonormal basis around the normal.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

// Material type tags (must match gpu_scene.rs / sphere_to_gpu): 0 diffuse,
// 1 metallic, 2 dielectric (glass).
const MATERIAL_DIELECTRIC: f32 = 1.5; // threshold: material_type > this == glass
const GLASS_IOR: f32 = 1.5;

// Choose an outgoing direction for the surface interaction.
// - Dielectric (glass): stochastically reflect or refract using the Fresnel
//   (Schlick) reflectance, handling total internal reflection.
// - Otherwise: specular and diffuse lobes are selected stochastically with
//   probability `metallic`; because each lobe's BRDF weight is cancelled by its
//   selection probability, the caller updates throughput simply with *= albedo,
//   keeping the path tracer energy conserving (no ad-hoc loss factors).
fn scatter_direction(
    incident_dir: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    material_type: f32,
    seed: vec3<u32>
) -> vec3<f32> {
    if (material_type > MATERIAL_DIELECTRIC) {
        let unit_in = normalize(incident_dir);
        // Outward normal points away from the surface; orient it against the
        // incoming ray and pick the index ratio for entering vs exiting glass.
        let front_face = dot(unit_in, normal) < 0.0;
        let oriented_n = select(-normal, normal, front_face);
        let eta = select(GLASS_IOR, 1.0 / GLASS_IOR, front_face);

        let cos_theta = min(dot(-unit_in, oriented_n), 1.0);
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

        // Schlick approximation of the Fresnel reflectance.
        let r0_root = (1.0 - eta) / (1.0 + eta);
        let r0 = r0_root * r0_root;
        let reflectance = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

        let xi = random_float(seed + vec3<u32>(211u, 97u, 41u));
        // Total internal reflection, or a Fresnel-weighted reflection event.
        if (eta * sin_theta > 1.0 || reflectance > xi) {
            return reflect(unit_in, oriented_n);
        }
        return refract(unit_in, oriented_n, eta);
    }

    let select_specular = random_float(seed + vec3<u32>(101u, 53u, 29u)) < metallic;
    if (select_specular) {
        let reflected = reflect(incident_dir, normal);
        var dir = reflected;
        if (roughness > 0.01) {
            let perturb = random_unit_vector(seed + vec3<u32>(7u, 11u, 13u)) * roughness;
            dir = normalize(reflected + perturb);
        }
        // Keep glossy reflections in the upper hemisphere.
        if (dot(dir, normal) < 0.0) {
            dir = reflected;
        }
        return dir;
    }
    return cosine_sample_hemisphere(normal, seed);
}

fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    // Convert ray direction to spherical UV coordinates for environment mapping
    let dir = normalize(ray_dir);
    let u = 0.5 + atan2(dir.z, dir.x) / (2.0 * PI);
    let v = 0.5 - asin(dir.y) / PI;

    // Wrap u and clamp v
    let uv = vec2<f32>(fract(u), clamp(v, 0.0, 1.0));

    // Sample the environment map using the sampler (this handles filtering properly)
    let env_color = textureSampleLevel(environment_map, environment_sampler, uv, 0.0);

    // Apply environment strength (stored in ambient_color.w)
    let environment_strength = scene.ambient_color.w;

    // Return the environment color scaled by strength (exposure already applied during texture loading)
    return env_color.rgb * environment_strength;
}

fn intersect_triangle(origin: vec3<f32>, dir: vec3<f32>, tri: Triangle) -> HitInfo {
    var info = HitInfo(LARGE_DISTANCE, vec3<f32>(0.0), vec3<f32>(0.0), false, 0.5, 0.0, 0.0);

    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);

    if (abs(a) < 1e-6) {
        return info;
    }

    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return info;
    }

    let q = cross(s, edge1);
    let v = f * dot(dir, q);

    if (v < 0.0 || (u + v) > 1.0) {
        return info;
    }

    let t = f * dot(edge2, q);

    if (t > 1e-4) {
        info.dist = t;
        let w = 1.0 - u - v;
        let interpolated = tri.n0.xyz * w + tri.n1.xyz * u + tri.n2.xyz * v;
        info.normal = normalize(interpolated);
        info.color = scene.mesh_color.xyz;
        info.hit = true;
        info.roughness = scene.mesh_material.x;
        info.metallic = scene.mesh_material.y;
        info.material_type = scene.mesh_material.z;
    }

    return info;
}

fn intersect_sphere(origin: vec3<f32>, dir: vec3<f32>, sph: Sphere) -> HitInfo {
    var info = HitInfo(LARGE_DISTANCE, vec3<f32>(0.0), vec3<f32>(0.0), false, 0.5, 0.0, 0.0);

    let center = sph.center_radius.xyz;
    let radius = sph.center_radius.w;
    let oc = origin - center;
    let a = dot(dir, dir);
    let half_b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0) {
        return info;
    }

    let sqrt_disc = sqrt(discriminant);
    var t = (-half_b - sqrt_disc) / a;

    if (t < 1e-4) {
        t = (-half_b + sqrt_disc) / a;
    }

    if (t > 1e-4) {
        let hit_pos = origin + dir * t;
        info.dist = t;
        info.normal = normalize(hit_pos - center);
        info.color = sph.color.xyz;
        info.hit = true;
        info.roughness = sph.material.x;
        info.metallic = sph.material.y;
        info.material_type = sph.material.z;
    }

    return info;
}

fn make_safe_dir(dir: vec3<f32>) -> vec3<f32> {
    let eps = 1e-6;
    let sign_x = select(1.0, -1.0, dir.x < 0.0);
    let sign_y = select(1.0, -1.0, dir.y < 0.0);
    let sign_z = select(1.0, -1.0, dir.z < 0.0);
    let adjust_x = select(0.0, sign_x * eps, abs(dir.x) < eps);
    let adjust_y = select(0.0, sign_y * eps, abs(dir.y) < eps);
    let adjust_z = select(0.0, sign_z * eps, abs(dir.z) < eps);
    return vec3<f32>(dir.x + adjust_x, dir.y + adjust_y, dir.z + adjust_z);
}

fn aabb_entry_distance(origin: vec3<f32>, inv_dir: vec3<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> f32 {
    let t1 = (bounds_min - origin) * inv_dir;
    let t2 = (bounds_max - origin) * inv_dir;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmax >= max(tmin, 0.0)) {
        return tmin;
    }
    return LARGE_DISTANCE;
}

fn traverse_triangles(origin: vec3<f32>, dir: vec3<f32>) -> HitInfo {
    var closest_hit = HitInfo(LARGE_DISTANCE, vec3<f32>(0.0), vec3<f32>(0.0), false, 0.5, 0.0, 0.0);
    let triangle_count = scene.resolution.z;
    if (triangle_count == 0u) {
        return closest_hit;
    }

    let node_count = scene.accel_info.x;
    if (node_count == 0u) {
        for (var i: u32 = 0u; i < triangle_count; i = i + 1u) {
            let hit = intersect_triangle(origin, dir, triangles[i]);
            if (hit.hit && hit.dist < closest_hit.dist) {
                closest_hit = hit;
            }
        }
        return closest_hit;
    }

    let safe_dir = make_safe_dir(dir);
    let inv_dir = 1.0 / safe_dir;

    var stack: array<u32, BVH_STACK_SIZE>;
    var stack_size: u32 = 1u;
    stack[0u] = 0u;

    loop {
        if (stack_size == 0u) {
            break;
        }
        stack_size = stack_size - 1u;
        let node_index = stack[stack_size];
        if (node_index >= node_count) {
            continue;
        }
        let node = bvh_nodes[node_index];
        let entry = aabb_entry_distance(origin, inv_dir, node.bounds_min.xyz, node.bounds_max.xyz);
        if (entry >= closest_hit.dist) {
            continue;
        }

        if (node.prim_count > 0u) {
            let start = node.left_first;
            let end = start + node.prim_count;
            for (var i = start; i < end; i = i + 1u) {
                if (i >= triangle_count) {
                    break;
                }
                let hit = intersect_triangle(origin, dir, triangles[i]);
                if (hit.hit && hit.dist < closest_hit.dist) {
                    closest_hit = hit;
                }
            }
        } else {
            let left = node.left_first;
            let right = node.right_child;

            var left_entry = LARGE_DISTANCE;
            var right_entry = LARGE_DISTANCE;

            if (left < node_count) {
                let left_node = bvh_nodes[left];
                left_entry = aabb_entry_distance(origin, inv_dir, left_node.bounds_min.xyz, left_node.bounds_max.xyz);
            }
            if (right < node_count) {
                let right_node = bvh_nodes[right];
                right_entry = aabb_entry_distance(origin, inv_dir, right_node.bounds_min.xyz, right_node.bounds_max.xyz);
            }

            if (left_entry < right_entry) {
                if (right_entry < closest_hit.dist && stack_size < BVH_STACK_SIZE) {
                    stack[stack_size] = right;
                    stack_size = stack_size + 1u;
                }
                if (left_entry < closest_hit.dist && stack_size < BVH_STACK_SIZE) {
                    stack[stack_size] = left;
                    stack_size = stack_size + 1u;
                }
            } else {
                if (left_entry < closest_hit.dist && stack_size < BVH_STACK_SIZE) {
                    stack[stack_size] = left;
                    stack_size = stack_size + 1u;
                }
                if (right_entry < closest_hit.dist && stack_size < BVH_STACK_SIZE) {
                    stack[stack_size] = right;
                    stack_size = stack_size + 1u;
                }
            }
        }
    }

    return closest_hit;
}

fn trace_ray(origin: vec3<f32>, dir: vec3<f32>) -> HitInfo {
    var closest_hit = HitInfo(LARGE_DISTANCE, vec3<f32>(0.0), vec3<f32>(0.0), false, 0.5, 0.0, 0.0);

    let triangle_hit = traverse_triangles(origin, dir);
    if (triangle_hit.hit) {
        closest_hit = triangle_hit;
    }

    for (var i: u32 = 0u; i < scene.resolution.w; i = i + 1u) {
        let hit = intersect_sphere(origin, dir, spheres[i]);
        if (hit.hit && hit.dist < closest_hit.dist) {
            closest_hit = hit;
        }
    }

    return closest_hit;
}

// Clamp the luminance of an indirect contribution to suppress fireflies: rare
// very bright paths (e.g. a glossy/refractive bounce that catches the sun)
// otherwise leave speckles that persist through temporal accumulation. Applied
// only to bounced light, never to the directly-visible background.
const FIREFLY_MAX_LUMINANCE: f32 = 8.0;

fn clamp_firefly(contribution: vec3<f32>) -> vec3<f32> {
    let luminance = dot(contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (luminance > FIREFLY_MAX_LUMINANCE) {
        return contribution * (FIREFLY_MAX_LUMINANCE / luminance);
    }
    return contribution;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.resolution.x || global_id.y >= scene.resolution.y) {
        return;
    }

    let width = scene.resolution.x;
    let height = scene.resolution.y;
    let padded_width = scene.render_config.z; // Padded width for buffer alignment
    let pixel_index = global_id.y * padded_width + global_id.x;

    let samples = max(scene.render_config.x, 1u);
    let max_bounces = max(scene.render_config.y, 1u);
    let frame_seed = scene.render_config.w; // Frame counter for temporal variation
    let origin = scene.camera_position.xyz;
    let pixel_coords = vec2<u32>(global_id.x, global_id.y);
    var color_accum = vec3<f32>(0.0);

    for (var sample: u32 = 0u; sample < samples; sample = sample + 1u) {
        let jitter = random2(pixel_coords, sample, frame_seed);
        let u = (f32(global_id.x) + jitter.x) / f32(width);
        let v = (f32(global_id.y) + jitter.y) / f32(height);

        var ray_origin = origin;
        var ray_dir = normalize(
            scene.lower_left_corner.xyz + scene.horizontal.xyz * u + scene.vertical.xyz * v - origin,
        );
        var throughput = vec3<f32>(1.0, 1.0, 1.0);
        var radiance = vec3<f32>(0.0);
        var path_active = true;

        for (var bounce: u32 = 0u; bounce < max_bounces; bounce = bounce + 1u) {
            let hit = trace_ray(ray_origin, ray_dir);
            if (!hit.hit) {
                // Ray escaped to the environment: collect sky radiance. This is
                // the only place sky light enters the path, so each surface is
                // lit exactly once (no per-bounce double counting). The primary
                // ray (bounce 0) is the directly-visible background and is kept
                // unclamped; bounced contributions are firefly-clamped.
                let contribution = throughput * sky_color(ray_dir);
                if (bounce == 0u) {
                    radiance = radiance + contribution;
                } else {
                    radiance = radiance + clamp_firefly(contribution);
                }
                path_active = false;
                break;
            }

            let hit_pos = ray_origin + ray_dir * hit.dist;

            let bounce_seed = vec3<u32>(
                pixel_coords.x + 17u * bounce + 13u * sample,
                pixel_coords.y + 31u * bounce + 7u * sample,
                sample * max_bounces + bounce + frame_seed * 1000u,
            );

            // Surfaces are not emitters: lighting is gathered by tracing the
            // scattered ray. With cosine-weighted diffuse sampling and stochastic
            // lobe selection, the throughput update is just *= albedo (the BRDF
            // weights and pdfs cancel), so the estimator stays energy conserving.
            throughput = throughput * hit.color;

            ray_dir = scatter_direction(ray_dir, hit.normal, hit.roughness, hit.metallic, hit.material_type, bounce_seed);
            // Offset along whichever side of the surface the new ray leaves on,
            // so transmitted (refracted) rays are pushed inside, not back out.
            let offset_normal = select(-hit.normal, hit.normal, dot(ray_dir, hit.normal) >= 0.0);
            ray_origin = hit_pos + offset_normal * 1e-3;
        }

        // Paths that exhaust the bounce budget approximate their remaining
        // contribution with the environment along the last scattered direction.
        // This is a bounded approximation, not a re-add of already-counted light.
        if (path_active) {
            radiance = radiance + clamp_firefly(throughput * sky_color(ray_dir));
        }

        color_accum = color_accum + radiance;
    }

    let color = color_accum / f32(samples);
    image[pixel_index] = vec4<f32>(color, 1.0);
}
