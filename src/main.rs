mod camera;
mod hit;
mod lights;
mod ray;
mod sphere;
mod vec3;
use camera::Camera;
use hit::HitRecord;
use lights::Light;
use ray::Ray;
use sphere::Material;
use sphere::Sphere;
use vec3::Vec3;

fn ray_color(ray: &Ray, objects: &[Sphere], lights: &[Light], depth: u32) -> Vec3 {
    if depth == 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut closest_t = f64::INFINITY;
    let mut hit_record: Option<HitRecord> = None;
    let mut hit_idx: Option<usize> = None;

    // Find closest hit
    for (idx, obj) in objects.iter().enumerate() {
        let rec = obj.hit(ray);
        if rec.is_hit && rec.dst < closest_t {
            closest_t = rec.dst;
            hit_record = Some(rec);
            hit_idx = Some(idx);
        }
    }

    if let Some(rec) = hit_record {
        let sphere = &objects[hit_idx.unwrap()];

        // Calculate direct lighting from all lights
        let mut direct_light = Vec3::new(0.0, 0.0, 0.0);

        for light in lights {
            match light {
                Light::Ambient(ambient) => {
                    // Ambient light affects everything equally
                    direct_light = direct_light + ambient.color * ambient.strength;
                }
                Light::Point(point_light) => {
                    // Direction from hit point to light
                    let to_light = point_light.position - rec.hit_point;
                    let light_distance = to_light.length();
                    let light_dir = to_light.normalize();

                    // Shadow ray - check if light is blocked
                    let shadow_origin = rec.hit_point + rec.normal * 1e-4;
                    let shadow_ray = Ray::new(shadow_origin, light_dir);
                    let mut in_shadow = false;

                    for obj in objects {
                        let shadow_hit = obj.hit(&shadow_ray);
                        if shadow_hit.is_hit && shadow_hit.dst < light_distance {
                            in_shadow = true;
                            break;
                        }
                    }

                    if !in_shadow {
                        // Diffuse lighting (Lambertian)
                        let diffuse = rec.normal.dot(&light_dir).max(0.0);

                        // Light intensity falls off with distance squared
                        let attenuation = point_light.strength / (light_distance * light_distance);

                        direct_light = direct_light + point_light.color * diffuse * attenuation;
                    }
                }
                Light::Directional(dir_light) => {
                    // Directional light comes from infinity in a direction
                    let light_dir = (dir_light.direction * -1.0).normalize();

                    // Shadow ray for directional light (goes to infinity)
                    let shadow_origin = rec.hit_point + rec.normal * 1e-4;
                    let shadow_ray = Ray::new(shadow_origin, light_dir);
                    let mut in_shadow = false;

                    for obj in objects {
                        let shadow_hit = obj.hit(&shadow_ray);
                        if shadow_hit.is_hit {
                            in_shadow = true;
                            break;
                        }
                    }

                    if !in_shadow {
                        let diffuse = rec.normal.dot(&light_dir).max(0.0);
                        direct_light =
                            direct_light + dir_light.color * dir_light.strength * diffuse;
                    }
                }
            }
        }

        // Indirect lighting - random bounce for global illumination
        let bounce_direction = Vec3::random_in_hemisphere(&rec.normal);
        let bounce_origin = rec.hit_point + rec.normal * 1e-4;
        let bounced_ray = Ray::new(bounce_origin, bounce_direction);

        // Recursively trace the bounced ray
        let incoming_light = ray_color(&bounced_ray, objects, lights, depth - 1);

        // Get sphere's albedo (color)
        let albedo = sphere.material.color;

        // Combine direct and indirect lighting
        // Scale down indirect to balance with direct lighting
        let total_light = direct_light + incoming_light * 0.2;

        // Multiply light by material color (albedo)
        Vec3::new(
            albedo.x * total_light.x,
            albedo.y * total_light.y,
            albedo.z * total_light.z,
        )
    } else {
        // No hit - return sky gradient
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);

        // Blue-to-white gradient
        let white = Vec3::new(1.0, 1.0, 1.0);
        let blue = Vec3::new(0.5, 0.7, 1.0);

        white * (1.0 - t) + blue * t
    }
}

fn main() {
    // Setup
    let width = 800;
    let height = 600;

    // Create camera
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        width as f64,
        height as f64,
    );

    // Create sphere in front of camera
    let spheres = [
        Sphere::new(
            Vec3::new(0.0, 0.0, 5.0),
            1.0,
            Material::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.0),
        ),
        Sphere::new(
            Vec3::new(2.0, 0.0, 6.0),
            1.0,
            Material::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.0),
        ),
        Sphere::new(
            Vec3::new(-2.0, 0.0, 6.0),
            1.0,
            Material::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0), 0.0),
        ),
    ];

    let lights = [
        Light::new_directional_light(Vec3::new(3.0, -3.0, 3.0), Vec3::new(1.0, 1.0, 1.0), 1.5),
        Light::new_ambient_light(Vec3::new(0.1, 0.1, 0.1), 0.1),
    ];

    // Start PPM file
    println!("P3");
    println!("{} {}", width, height);
    println!("255");

    // Loop through each pixel
    for j in (0..height).rev() {
        for i in 0..width {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            let samples_per_pixel = 50;

            for _ in 0..samples_per_pixel {
                let u = (i as f64 + rand::random::<f64>()) / (width - 1) as f64;
                let v = (j as f64 + rand::random::<f64>()) / (height - 1) as f64;

                let ray = camera.get_ray(u, v);
                let sample_color = ray_color(&ray, &spheres, &lights, 5);

                pixel_color.x += sample_color.x;
                pixel_color.y += sample_color.y;
                pixel_color.z += sample_color.z;
            }

            // Averaging + Gamma correction + Color clamping
            let scale = 1.0 / samples_per_pixel as f64;
            let r = (pixel_color.x * scale).sqrt();
            let g = (pixel_color.y * scale).sqrt();
            let b = (pixel_color.z * scale).sqrt();

            let ir = (255.0 * r.clamp(0.0, 1.0)) as u8;
            let ig = (255.0 * g.clamp(0.0, 1.0)) as u8;
            let ib = (255.0 * b.clamp(0.0, 1.0)) as u8;

            println!("{} {} {}", ir, ig, ib);
        }
    }
}
