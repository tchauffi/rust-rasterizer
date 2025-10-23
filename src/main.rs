mod camera;
mod ray;
mod sphere;
mod vec3;
use camera::Camera;
use sphere::Color;
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
        90.0,
        width as f64,
        height as f64,
    );

    // Create sphere in front of camera
    let spheres = vec![
        Sphere::new(Vec3::new(0.0, 0.0, 5.0), 1.0, Color::new(255, 0, 0)),
        Sphere::new(Vec3::new(2.0, 0.0, 6.0), 1.0, Color::new(0, 255, 0)),
        Sphere::new(Vec3::new(-1.0, 0.0, 2.0), 1.0, Color::new(0, 0, 255)),
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

            for (idx, sphere) in spheres.iter().enumerate() {
                if let Some(t) = sphere.hit(&ray) {
                    if t < closest_t {
                        closest_t = t;
                        hit_idx = Some(idx);
                    }
                }
            }

            if hit_idx.is_some() {
                let sphere = &spheres[hit_idx.unwrap()];
                println!("{} {} {}", sphere.color.r, sphere.color.g, sphere.color.b);
            } else {
                // Background color
                println!("135 206 235"); // Sky blue
            }



        }
    }
}
