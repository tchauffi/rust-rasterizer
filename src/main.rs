mod vec3;
mod sphere;
mod camera;
mod ray;
use vec3::Vec3;
use sphere::Sphere;
use sphere::Color;
use camera::Camera;

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
        height as f64 
    );
    
    // Create sphere in front of camera
    let spheres = vec![
        Sphere::new(Vec3::new(0.0, 0.0, -5.0), 1.0, Color::new(255, 0, 0)),
        Sphere::new(Vec3::new(2.0, 0.0, -6.0), 0.5, Color::new(0, 255, 0)),
        Sphere::new(Vec3::new(-2.0, 0.0, -4.0), 1.5, Color::new(0, 0, 255)),
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
            
            let color = if spheres.iter().any(|sphere| sphere.hit(&ray)) {
                // Hit: red color
                spheres.iter().find(|sphere| sphere.hit(&ray)).map_or(Vec3::new(255.0, 0.0, 0.0), |sphere| {
                    Vec3::new(sphere.color.r as f64, sphere.color.g as f64, sphere.color.b as f64)
                })
            } else {
                // Miss: blue gradient background
                let t = 0.5 * (ray.direction.y + 1.0);
                let white = Vec3::new(255.0, 255.0, 255.0);
                let blue = Vec3::new(128.0, 178.0, 255.0);
                white * (1.0 - t) + blue * t
            };
            
            println!("{} {} {}", color.x as i32, color.y as i32, color.z as i32);
        }
    }
}