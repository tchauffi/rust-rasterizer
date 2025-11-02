use crate::HitRecord;
use crate::camera::Camera;
use crate::hittable::Hittable;
use crate::lights::Light;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Box<dyn Hittable>>,
    pub lights: Vec<Light>,
}

#[allow(dead_code)]
impl Scene {
    pub fn new(camera: Camera, objects: Vec<Box<dyn Hittable>>, lights: Vec<Light>) -> Self {
        Scene {
            camera,
            objects,
            lights,
        }
    }
    pub fn add_object(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }
    pub fn clear_objects(&mut self) {
        self.objects.clear();
    }
    pub fn render(&self) -> Vec<Vec<Vec3>> {
        let width = self.camera.width as usize;
        let height = self.camera.height as usize;
        let samples_per_pixel = 20;
        let max_depth = 2;

        let mut image = vec![vec![Vec3::new(0.0, 0.0, 0.0); width]; height];

        #[allow(clippy::needless_range_loop)]
        for j in 0..height {
            for i in 0..width {
                let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

                for _s in 0..samples_per_pixel {
                    let u = (i as f64 + rand::random::<f64>()) / (width - 1) as f64;
                    let v = (j as f64 + rand::random::<f64>()) / (height - 1) as f64;

                    let ray = self.camera.get_ray(u, v);
                    let sample_color = self.trace_ray(&ray, max_depth);

                    pixel_color = pixel_color + sample_color;
                }

                // Average the color and store it
                pixel_color = pixel_color / samples_per_pixel as f64;

                pixel_color = Vec3::new(
                    pixel_color.x.sqrt().clamp(0.0, 1.0),
                    pixel_color.y.sqrt().clamp(0.0, 1.0),
                    pixel_color.z.sqrt().clamp(0.0, 1.0),
                );

                image[j][i] = pixel_color;
            }
        }
        image
    }

    pub fn trace_ray(&self, ray: &Ray, depth: u32) -> Vec3 {
        if depth == 0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }

        const EPSILON: f64 = 1e-3;

        // Find closest hit and track which object was hit
        if let Some((hit, obj_idx)) = self.find_closest_hit(ray) {
            // Get material from the hit object
            let material = *self.objects[obj_idx].get_material();

            // Compute direct lighting using the lights module
            let direct_light = self.compute_direct_lighting(&hit);

            // Compute indirect lighting (global illumination)
            let indirect_light = self.compute_indirect_lighting(&hit, depth);

            let albedo = material.color;

            let direct_contribution = direct_light * albedo;
            let indirect_contribution = indirect_light * albedo * 0.5;

            direct_contribution + indirect_contribution
        } else {
            self.background_color(ray)
        }
    }

    fn find_closest_hit(&self, ray: &Ray) -> Option<(HitRecord, usize)> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_idx: Option<usize> = None;

        for (idx, object) in self.objects.iter().enumerate() {
            let hit_record = object.hit(ray);
            if hit_record.is_hit
                && (closest_hit.is_none() || hit_record.dst < closest_hit.as_ref().unwrap().dst)
            {
                closest_hit = Some(hit_record);
                closest_idx = Some(idx);
            }
        }

        match (closest_hit, closest_idx) {
            (Some(hit), Some(idx)) => Some((hit, idx)),
            _ => None,
        }
    }

    fn compute_direct_lighting(&self, hit: &HitRecord) -> Vec3 {
        self.lights
            .iter()
            .map(|light| light.compute_contribution(hit, &self.objects))
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, contrib| acc + contrib)
    }

    fn compute_indirect_lighting(&self, hit: &HitRecord, depth: u32) -> Vec3 {
        const EPSILON: f64 = 1e-3;

        let bounce_direction = hit.normal + Vec3::random_in_hemisphere(&hit.normal);
        let bounce_origin = hit.hit_point + hit.normal * EPSILON;
        let bounce_ray = Ray::new(bounce_origin, bounce_direction.normalize());

        self.trace_ray(&bounce_ray, depth - 1)
    }

    fn background_color(&self, ray: &Ray) -> Vec3 {
        // Check if there's an environment light in the scene
        for light in &self.lights {
            if let crate::lights::Light::Environment(env_light) = light {
                return env_light.compute(&ray.direction);
            }
        }

        // Default sky gradient if no environment light
        let t = 0.5 * (ray.direction.normalize().y + 1.0);
        Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}
