mod camera;
mod hit;
mod ray;
mod sphere;
mod vec3;
use camera::Camera;
use hit::HitRecord;
use ray::Ray;
use sphere::Color;
use sphere::Material;
use sphere::Sphere;
use vec3::Vec3;

fn ray_color(ray: &Ray, objects: &[Sphere], depth: u32) -> Color {
    if depth == 0 {
        return Color::new(0, 0, 0);
    }

    let mut closest_t = f64::INFINITY;
    let mut hit_record: Option<HitRecord> = None;
    let mut hit_idx: Option<usize> = None;

    for (idx, object) in objects.iter().enumerate() {
        let temp_rec = object.hit(ray);
        if temp_rec.is_hit && temp_rec.dst < closest_t {
            closest_t = temp_rec.dst;
            hit_record = Some(temp_rec);
            hit_idx = Some(idx);
        }
    }

    if let Some(rec) = hit_record {
        let sphere = &objects[hit_idx.unwrap()];

        let bounce_direction = Vec3::random_in_hemisphere(&rec.normal);
        let bounce_origin = rec.hit_point;
        let bounced_ray = Ray::new(bounce_origin, bounce_direction);

        let incoming_light = ray_color(&bounced_ray, objects, depth - 1);

        let albedo = Vec3::new(
            sphere.material.color.r as f64 / 255.0,
            sphere.material.color.g as f64 / 255.0,
            sphere.material.color.b as f64 / 255.0,
        );

        Color::new(
            (incoming_light.r as f64 * albedo.x * 0.5) as u8,
            (incoming_light.g as f64 * albedo.y * 0.5) as u8,
            (incoming_light.b as f64 * albedo.z * 0.5) as u8,
        )
    } else {
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        let r = (1.0 - t) * 1.0 + t * 0.5;
        let g = (1.0 - t) * 1.0 + t * 0.7;
        let b = (1.0 - t) * 1.0 + t * 1.0;

        Color::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
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
            Material::new(Color::new(255, 0, 0), Color::new(0, 0, 0), 0.0),
        ),
        Sphere::new(
            Vec3::new(2.0, 0.0, 6.0),
            1.0,
            Material::new(Color::new(0, 255, 0), Color::new(0, 0, 0), 0.0),
        ),
        Sphere::new(
            Vec3::new(-1.0, 0.0, 3.0),
            1.0,
            Material::new(Color::new(0, 0, 255), Color::new(0, 0, 0), 0.0),
        ),
    ];

    // Start PPM file
    println!("P3");
    println!("{} {}", width, height);
    println!("255");

    // Loop through each pixel
    for j in (0..height).rev() {
        for i in 0..width {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            let samples_per_pixel = 10;

            for _ in 0..samples_per_pixel {
                let u = (i as f64 + rand::random::<f64>()) / (width - 1) as f64;
                let v = (j as f64 + rand::random::<f64>()) / (height - 1) as f64;

                let ray = camera.get_ray(u, v);
                let sample_color = ray_color(&ray, &spheres, 5);

                pixel_color.x += sample_color.r as f64;
                pixel_color.y += sample_color.g as f64;
                pixel_color.z += sample_color.b as f64;
            }

            // Averaging + Gamma correction + Color clamping
            let scale = 1.0 / samples_per_pixel as f64;
            let r = (pixel_color.x * scale).sqrt();
            let g = (pixel_color.y * scale).sqrt();
            let b = (pixel_color.z * scale).sqrt();

            let ir = (256.0 * r.clamp(0.0, 0.999)) as u8;
            let ig = (256.0 * g.clamp(0.0, 0.999)) as u8;
            let ib = (256.0 * b.clamp(0.0, 0.999)) as u8;

            println!("{} {} {}", ir, ig, ib);
        }
    }
}
