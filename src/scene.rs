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
        let samples_per_pixel = 5;
        let max_depth = 3;

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
            return Vec3::new(0.0, 0.0, 0.0); // Black if max depth reached
        }

        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_obj_idx: Option<usize> = None;

        const EPSILON: f64 = 1e-3;

        // Find closest hit
        for (idx, object) in self.objects.iter().enumerate() {
            let hit_record = object.hit(ray);
            if hit_record.is_hit
                && (closest_hit.is_none() || hit_record.dst < closest_hit.as_ref().unwrap().dst)
            {
                closest_hit = Some(hit_record);
                closest_obj_idx = Some(idx);
            }
        }

        // If we hit something
        if let Some(hit) = closest_hit {
            let material = self.objects[closest_obj_idx.unwrap()].get_material();

            // Accumulate light contributions
            let mut direct_light = Vec3::new(0.0, 0.0, 0.0);

            for light in &self.lights {
                match light {
                    Light::Ambient(ambient) => {
                        // Ambient light affects everything equally
                        direct_light = direct_light + ambient.color * ambient.intensity;
                    }
                    Light::Point(point_light) => {
                        // Direction from hit point to light
                        let to_light = point_light.position - hit.hit_point;
                        let light_distance = to_light.length();
                        let light_dir = to_light.normalize();

                        // Shadow check
                        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
                        let shadow_ray = Ray::new(shadow_origin, light_dir);
                        let mut in_shadow = false;

                        for obj in &self.objects {
                            let shadow_hit = obj.hit(&shadow_ray);
                            if shadow_hit.is_hit && shadow_hit.dst < light_distance {
                                in_shadow = true;
                                break;
                            }
                        }

                        if !in_shadow {
                            // Diffuse lighting
                            let diffuse = hit.normal.dot(&light_dir).max(0.0);
                            let attenuation =
                                point_light.intensity / (light_distance * light_distance);
                            direct_light = direct_light + point_light.color * diffuse * attenuation;
                        }
                    }
                    Light::Directional(dir_light) => {
                        // Directional light comes from infinity
                        let light_dir = (dir_light.direction * -1.0).normalize();

                        // Shadow check
                        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
                        let shadow_ray = Ray::new(shadow_origin, light_dir);
                        let mut in_shadow = false;

                        for obj in &self.objects {
                            let shadow_hit = obj.hit(&shadow_ray);
                            if shadow_hit.is_hit {
                                in_shadow = true;
                                break;
                            }
                        }

                        if !in_shadow {
                            let diffuse = hit.normal.dot(&light_dir).max(0.0);
                            direct_light =
                                direct_light + dir_light.color * dir_light.intensity * diffuse;
                        }
                    }
                }
            }

            let bounce_direction = hit.normal + Vec3::random_in_hemisphere(&hit.normal);
            let bounce_origin = hit.hit_point + hit.normal * EPSILON;
            let bounce_ray = Ray::new(bounce_origin, bounce_direction.normalize());

            let indirect_light = self.trace_ray(&bounce_ray, depth - 1);

            let albedo = material.color;
            direct_light * albedo + indirect_light * albedo * 0.1
        } else {
            // Background gradient
            let t = 0.5 * (ray.direction.normalize().y + 1.0);
            Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
        }
    }
}
