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
    render_config: vec4<u32>,
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
    color: vec4<f32>,
};

struct HitInfo {
    dist: f32,
    normal: vec3<f32>,
    color: vec3<f32>,
    hit: bool,
};

const PI: f32 = 3.141592653589793;
const INDIRECT_ATTENUATION: f32 = 0.05;

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> triangles: array<Triangle>;

@group(0) @binding(2)
var<storage, read> spheres: array<Sphere>;

@group(0) @binding(3)
var<storage, read_write> image: array<vec4<f32>>;

fn hash_float3(value: vec3<u32>) -> f32 {
    let value_f = vec3<f32>(f32(value.x), f32(value.y), f32(value.z));
    let dot_product = dot(value_f, vec3<f32>(0.1031, 0.11369, 0.13787));
    return fract(sin(dot_product) * 43758.5453);
}

fn random2(pixel: vec2<u32>, sample: u32) -> vec2<f32> {
    let seed0 = vec3<u32>(pixel.x, pixel.y, sample);
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

fn sky_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (ray_dir.y + 1.0);
    let white = vec3<f32>(1.0, 1.0, 1.0);
    let blue = vec3<f32>(0.2, 0.3, 0.8);
    return mix(white, blue, t);
}

fn intersect_triangle(origin: vec3<f32>, dir: vec3<f32>, tri: Triangle) -> HitInfo {
    var info = HitInfo(1e30, vec3<f32>(0.0), vec3<f32>(0.0), false);

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
    }

    return info;
}

fn intersect_sphere(origin: vec3<f32>, dir: vec3<f32>, sph: Sphere) -> HitInfo {
    var info = HitInfo(1e30, vec3<f32>(0.0), vec3<f32>(0.0), false);

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
    }

    return info;
}

fn trace_ray(origin: vec3<f32>, dir: vec3<f32>) -> HitInfo {
    var closest_hit = HitInfo(1e30, vec3<f32>(0.0), vec3<f32>(0.0), false);

    for (var i: u32 = 0u; i < scene.resolution.z; i = i + 1u) {
        let hit = intersect_triangle(origin, dir, triangles[i]);
        if (hit.hit && hit.dist < closest_hit.dist) {
            closest_hit = hit;
        }
    }

    for (var i: u32 = 0u; i < scene.resolution.w; i = i + 1u) {
        let hit = intersect_sphere(origin, dir, spheres[i]);
        if (hit.hit && hit.dist < closest_hit.dist) {
            closest_hit = hit;
        }
    }

    return closest_hit;
}

fn is_shadowed(hit_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> bool {
    let shadow_origin = hit_pos + normal * 1e-3;
    let shadow_hit = trace_ray(shadow_origin, light_dir);
    return shadow_hit.hit;
}

fn evaluate_direct(hit_pos: vec3<f32>, normal: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(-scene.light_direction.xyz);
    let light_strength = scene.light_direction.w;
    var diffuse = max(dot(normal, light_dir), 0.0) * light_strength;

    if (diffuse > 0.0 && is_shadowed(hit_pos, normal, light_dir)) {
        diffuse = 0.0;
    }

    let ambient = scene.ambient_color.xyz;
    let light_color = scene.light_color.xyz;
    let lighting = ambient + light_color * diffuse;
    return albedo * lighting;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.resolution.x || global_id.y >= scene.resolution.y) {
        return;
    }

    let width = scene.resolution.x;
    let height = scene.resolution.y;
    let pixel_index = global_id.y * width + global_id.x;

    let samples = max(scene.render_config.x, 1u);
    let max_bounces = max(scene.render_config.y, 1u);
    let origin = scene.camera_position.xyz;
    let pixel_coords = vec2<u32>(global_id.x, global_id.y);
    var color_accum = vec3<f32>(0.0);

    for (var sample: u32 = 0u; sample < samples; sample = sample + 1u) {
        let jitter = random2(pixel_coords, sample);
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
            let direct = evaluate_direct(hit_pos, hit.normal, hit.color);
            radiance = radiance + throughput * direct;

            throughput = throughput * hit.color * INDIRECT_ATTENUATION;

            let bounce_seed = vec3<u32>(
                pixel_coords.x + 17u * bounce + 13u * sample,
                pixel_coords.y + 31u * bounce + 7u * sample,
                sample * max_bounces + bounce,
            );
            ray_origin = hit_pos + hit.normal * 1e-3;
            ray_dir = random_in_hemisphere(hit.normal, bounce_seed);
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
