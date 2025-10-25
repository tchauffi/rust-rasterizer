mod camera;
mod hit;
mod ray;
mod sphere;
mod vec3;
use camera::Camera;
use hit::HitRecord;
use rand::Rng;
use ray::Ray;
use sphere::Material;
use sphere::Sphere;
use vec3::Vec3;

fn ray_color(ray: &Ray, objects: &[Sphere], depth: u32) -> Vec3 {
    if depth == 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut closest_t = f64::INFINITY;
    let mut hit_record: Option<HitRecord> = None;
    let mut hit_idx: Option<usize> = None;

    let mut rng = rand::rng();

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
        let bounce_origin = rec.hit_point
            + Vec3::new(
                rng.random::<f64>(),
                rng.random::<f64>(),
                rng.random::<f64>(),
            ) * 1e-3;
        let bounced_ray = Ray::new(bounce_origin, bounce_direction);

        let incoming_light = ray_color(&bounced_ray, objects, depth - 1);

        let albedo = Vec3::new(
            sphere.material.color.x,
            sphere.material.color.y,
            sphere.material.color.z,
        );

        Vec3::new(
            albedo.x * incoming_light.x,
            albedo.y * incoming_light.y,
            albedo.z * incoming_light.z,
        )
    } else {
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        let r = (1.0 - t) * 1.0 + t * 0.5;
        let g = (1.0 - t) * 1.0 + t * 0.7;
        let b = (1.0 - t) * 1.0 + t * 1.0;

        Vec3::new(r, g, b)
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
            Vec3::new(-1.0, 0.0, 3.0),
            1.0,
            Material::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0), 0.0),
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
            let samples_per_pixel = 20;

            for _ in 0..samples_per_pixel {
                let u = (i as f64 + rand::random::<f64>()) / (width - 1) as f64;
                let v = (j as f64 + rand::random::<f64>()) / (height - 1) as f64;

                let ray = camera.get_ray(u, v);
                let sample_color = ray_color(&ray, &spheres, 10);

                pixel_color.x += sample_color.x;
                pixel_color.y += sample_color.y;
                pixel_color.z += sample_color.z;
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
