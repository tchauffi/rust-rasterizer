mod aabb;
mod camera;
mod hittable;
mod lights;
mod material;
mod mesh;
mod ray;
mod scene;
mod sphere;
mod vec3;
use camera::Camera;
use hittable::{HitRecord, Hittable};
use lights::Light;
use material::Material;
use mesh::Mesh;
use scene::Scene;
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

    // Load the bunny mesh
    let bunny_material = Material::new(Vec3::new(1.0, 0.0, 0.0), 0.5);

    let mut bunny =
        Mesh::from_obj_file("data/bunny.obj", bunny_material).expect("Failed to load bunny.obj");

    eprintln!(
        "Loaded bunny mesh: {} vertices, {} faces",
        bunny.vertices.len(),
        bunny.faces.len() / 3
    );

    eprintln!(
        "Original bunny bounding box: min=({:.3}, {:.3}, {:.3}), max=({:.3}, {:.3}, {:.3})",
        bunny.bounding_box.min.x,
        bunny.bounding_box.min.y,
        bunny.bounding_box.min.z,
        bunny.bounding_box.max.x,
        bunny.bounding_box.max.y,
        bunny.bounding_box.max.z
    );

    // Rotate 180 degrees around Y axis (vertical)
    bunny.rotate_y(180.0);

    // Scale bunny up 10x and move it in front of camera
    bunny.transform(10.0, Vec3::new(0.0, -1.2, 4.0));

    eprintln!(
        "Transformed bunny bounding box: min=({:.3}, {:.3}, {:.3}), max=({:.3}, {:.3}, {:.3})",
        bunny.bounding_box.min.x,
        bunny.bounding_box.min.y,
        bunny.bounding_box.min.z,
        bunny.bounding_box.max.x,
        bunny.bounding_box.max.y,
        bunny.bounding_box.max.z
    );

    let sphere1 = Sphere::new(
        Vec3::new(0.0, -1001.0, 0.0),
        1000.0,
        Material::new(Vec3::new(1.0, 1.0, 1.0), 0.5),
    );

    // Create spheres for background
    let sphere2 = Sphere::new(
        Vec3::new(2.0, 0.0, 6.0),
        1.0,
        Material::new(Vec3::new(0.0, 1.0, 0.0), 0.5),
    );
    let sphere3 = Sphere::new(
        Vec3::new(-2.0, 0.0, 6.0),
        1.0,
        Material::new(Vec3::new(0.0, 0.0, 1.0), 0.5),
    );

    // Create array of hittable objects including the bunny
    let objects: Vec<Box<dyn Hittable>> = vec![
        Box::new(bunny),
        Box::new(sphere1),
        Box::new(sphere2),
        Box::new(sphere3),
    ];

    let lights = [
        Light::new_directional_light(Vec3::new(3.0, -3.0, 3.0), Vec3::new(1.0, 1.0, 1.0), 0.6),
        Light::new_ambient_light(Vec3::new(1.0, 1.0, 1.0), 0.2),
    ];

    let scene = Scene::new(camera, objects, lights.to_vec());
    eprintln!("Starting render...");
    let image = scene.render();

    println!("P3");
    println!("{} {}", width, height);
    println!("255");
    for j in (0..height).rev() {
        for i in 0..width {
            let pixel_color = image[j][i];
            let ir = (255.999 * pixel_color.x.clamp(0.0, 1.0)) as u32;
            let ig = (255.999 * pixel_color.y.clamp(0.0, 1.0)) as u32;
            let ib = (255.999 * pixel_color.z.clamp(0.0, 1.0)) as u32;
            println!("{} {} {}", ir, ig, ib);
        }
    }
    eprintln!("Render complete.");
}
