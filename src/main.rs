mod camera;
mod hit;
mod ray;
mod sphere;
mod vec3;
use camera::Camera;
use hit::HitRecord;
use sphere::Color;
use sphere::Material;
use sphere::Sphere;
use vec3::Vec3;

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

    // Add light sources
    let light_direction = Vec3::new(-0.5, -1.0, -1.0).normalize();
    let ambiant_strength = 0.4;

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
            let u = i as f64 / (width - 1) as f64;
            let v = j as f64 / (height - 1) as f64;

            let ray = camera.get_ray(u, v);

            let mut closest_t = f64::INFINITY;
            let mut hit_idx: Option<usize> = None;
            let mut hit_rec: Option<HitRecord> = None;

            for (idx, sphere) in spheres.iter().enumerate() {
                let hit_record: HitRecord = sphere.hit(&ray);
                if hit_record.is_hit && hit_record.dst < closest_t {
                    closest_t = hit_record.dst;
                    hit_rec = Some(hit_record);
                    hit_idx = Some(idx);
                }
            }

            if let Some(rec) = hit_rec {
                let diffuse = (-light_direction).dot(&rec.normal).max(0.0);

                let ligthing = ambiant_strength + (1.0 - ambiant_strength) * diffuse.powf(2.0);

                let sphere = &spheres[hit_idx.unwrap()];

                let r = (sphere.material.color.r as f64 * ligthing) as u8;
                let g = (sphere.material.color.g as f64 * ligthing) as u8;
                let b = (sphere.material.color.b as f64 * ligthing) as u8;

                println!("{} {} {}", r, g, b);
            } else {
                // Background color
                println!("135 206 235"); // Sky blue
            }
        }
    }
}
