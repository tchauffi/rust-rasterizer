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

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> triangles: array<Triangle>;

@group(0) @binding(2)
var<storage, read> spheres: array<Sphere>;

@group(0) @binding(3)
var<storage, read_write> image: array<vec4<f32>>;

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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= scene.resolution.x || global_id.y >= scene.resolution.y) {
        return;
    }

    let width = scene.resolution.x;
    let height = scene.resolution.y;
    let padded_width = scene.render_config.z; // Padded width for buffer alignment
    let pixel_index = global_id.y * padded_width + global_id.x;

    let origin = scene.camera_position.xyz;
    let u = (f32(global_id.x) + 0.5) / f32(width);
    let v = (f32(global_id.y) + 0.5) / f32(height);

    let ray_dir = normalize(
        scene.lower_left_corner.xyz + scene.horizontal.xyz * u + scene.vertical.xyz * v - origin,
    );

    let hit = trace_ray(origin, ray_dir);

    var color: vec3<f32>;
    if (hit.hit) {
        // Display normals as colors: convert from [-1, 1] to [0, 1]
        color = (hit.normal + vec3<f32>(1.0)) * 0.5;
    } else {
        // Sky color
        let t = 0.5 * (ray_dir.y + 1.0);
        let white = vec3<f32>(1.0, 1.0, 1.0);
        let blue = vec3<f32>(0.5, 0.7, 1.0);
        color = vec3<f32>(0.0, 0.0, 0.0);
    }

    image[pixel_index] = vec4<f32>(color, 1.0);
}
