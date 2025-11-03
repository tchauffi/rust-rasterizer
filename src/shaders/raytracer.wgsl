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

fn hash_float3(value: vec3<u32>) -> f32 {
    let value_f = vec3<f32>(f32(value.x), f32(value.y), f32(value.z));
    let dot_product = dot(value_f, vec3<f32>(0.1031, 0.11369, 0.13787));
    return fract(sin(dot_product) * 43758.5453);
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

fn random_in_hemisphere(normal: vec3<f32>, seed: vec3<u32>) -> vec3<f32> {
    var dir = random_unit_vector(seed);
    if (dot(dir, normal) < 0.0) {
        dir = -dir;
    }
    return dir;
}

fn reflect(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Sample a direction based on material type
fn sample_material_direction(
    incident_dir: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    seed: vec3<u32>
) -> vec3<f32> {
    // Perfect reflection for metallic
    let reflected = reflect(incident_dir, normal);

    // Diffuse scatter
    let diffuse = random_in_hemisphere(normal, seed);

    // Blend between diffuse and reflection based on metallic value
    let dir = mix(diffuse, reflected, metallic);

    // Add roughness by perturbing the direction slightly
    if (roughness > 0.01) {
        let perturb = random_unit_vector(seed + vec3<u32>(7u, 11u, 13u)) * roughness;
        return normalize(dir + perturb);
    }

    return normalize(dir);
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

fn evaluate_environment(
    normal: vec3<f32>,
    albedo: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    metallic: f32
) -> vec3<f32> {
    let n = normalize(normal);

    // Approximate diffuse contribution by sampling the environment along the surface normal.
    let env_diffuse = sky_color(n);
    let diffuse = albedo * env_diffuse * (1.0 - metallic) * (1.0 / PI);

    // Approximate specular contribution by sampling the environment in the reflection direction.
    let reflection_dir = reflect(-view_dir, n);
    let raw_specular = sky_color(reflection_dir);

    // Rough surfaces see a blurrier environment; approximate by blending with the diffuse sample.
    let reflectivity = pow(1.0 - roughness, 4.0);
    let env_specular = mix(env_diffuse, raw_specular, reflectivity);

    // Fresnel term (Schlick approximation) to mix specular color.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let cos_theta = clamp(dot(normalize(view_dir), n), 0.0, 1.0);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);

    let specular = env_specular * fresnel * reflectivity;

    return diffuse + specular;
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
                radiance = radiance + throughput * sky_color(ray_dir);
                path_active = false;
                break;
            }

            let hit_pos = ray_origin + ray_dir * hit.dist;

            // Compute view direction for specular
            let view_dir = normalize(-ray_dir);

            // Evaluate direct lighting (includes diffuse and specular)
            let direct = evaluate_environment(hit.normal, hit.color, view_dir, hit.roughness, hit.metallic);
            radiance = radiance + throughput * direct;

            let bounce_albedo = mix(hit.color, vec3<f32>(1.0), hit.metallic);
            let scatter_loss = mix(0.85, 0.98, hit.metallic);
            throughput = throughput * bounce_albedo * scatter_loss;

            let bounce_seed = vec3<u32>(
                pixel_coords.x + 17u * bounce + 13u * sample,
                pixel_coords.y + 31u * bounce + 7u * sample,
                sample * max_bounces + bounce + frame_seed * 1000u,
            );
            ray_origin = hit_pos + hit.normal * 1e-3;

            // Use material-aware direction sampling
            ray_dir = sample_material_direction(ray_dir, hit.normal, hit.roughness, hit.metallic, bounce_seed);
        }

        // If path is still active after max bounces, add environment contribution
        if (path_active) {
            radiance = radiance + throughput * sky_color(ray_dir);
        }

        color_accum = color_accum + radiance;
    }

    let color = color_accum / f32(samples);
    image[pixel_index] = vec4<f32>(color, 1.0);
}
