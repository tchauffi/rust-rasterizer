use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::vec3::Vec3;

const EPSILON: f64 = 1e-3;

#[allow(dead_code)]
#[derive(Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub radius: f64,
    pub color: Vec3,
    pub intensity: f64,
}

#[allow(dead_code)]
impl PointLight {
    pub fn new(position: Vec3, radius: f64, color: Vec3, intensity: f64) -> Self {
        PointLight {
            position,
            radius,
            color,
            intensity,
        }
    }
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Direction from hit point to light
        let to_light = self.position - hit.hit_point;
        let light_distance = to_light.length();
        let light_dir = to_light.normalize();

        // Shadow check
        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
        let shadow_ray = Ray::new(shadow_origin, light_dir);
        let mut in_shadow = false;

        for obj in objects {
            let shadow_hit = obj.hit(&shadow_ray);
            if shadow_hit.is_hit && shadow_hit.dst < light_distance - EPSILON {
                in_shadow = true;
                break;
            }
        }

        if !in_shadow {
            // Diffuse lighting
            let diffuse = hit.normal.dot(&light_dir).max(0.0);
            let attenuation = self.intensity / (light_distance * light_distance);
            self.color * diffuse * attenuation
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }
}
#[derive(Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f64,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f64) -> Self {
        DirectionalLight {
            direction,
            color,
            intensity,
        }
    }
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        let light_dir = (self.direction * -1.0).normalize();

        // Shadow check
        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
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
            let diffuse = hit.normal.dot(&light_dir).max(0.0);
            self.color * self.intensity * diffuse
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }
}
#[derive(Clone)]
pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f64,
}

impl AmbientLight {
    pub fn new(color: Vec3, intensity: f64) -> Self {
        AmbientLight { color, intensity }
    }
    pub fn compute(&self, _hit: &HitRecord, _objects: &[Box<dyn Hittable>]) -> Vec3 {
        self.color * self.intensity
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum Light {
    Point(PointLight),
    Directional(DirectionalLight),
    Ambient(AmbientLight),
}

#[allow(dead_code)]
impl Light {
    pub fn new_point_light(position: Vec3, radius: f64, color: Vec3, intensity: f64) -> Self {
        Light::Point(PointLight::new(position, radius, color, intensity))
    }

    pub fn new_directional_light(direction: Vec3, color: Vec3, intensity: f64) -> Self {
        Light::Directional(DirectionalLight::new(direction, color, intensity))
    }

    pub fn new_ambient_light(color: Vec3, intensity: f64) -> Self {
        Light::Ambient(AmbientLight::new(color, intensity))
    }
    pub fn compute_contribution(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        match self {
            Light::Ambient(ambient) => ambient.compute(hit, objects),
            Light::Point(point) => point.compute(hit, objects),
            Light::Directional(dir) => dir.compute(hit, objects),
        }
    }
}

// tests for lights
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_creation() {
        let light = PointLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            1.0,
            Vec3::new(1.0, 1.0, 1.0),
            10.0,
        );
        assert_eq!(light.position, Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(light.radius, 1.0);
        assert_eq!(light.intensity, 10.0);
    }

    #[test]
    fn test_directional_light_creation() {
        let light = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert_eq!(light.direction, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(light.intensity, 1.0);
    }

    #[test]
    fn test_ambient_light_creation() {
        let light = AmbientLight::new(Vec3::new(0.2, 0.2, 0.2), 0.5);
        assert_eq!(light.color, Vec3::new(0.2, 0.2, 0.2));
        assert_eq!(light.intensity, 0.5);
    }

    #[test]
    fn test_ambient_light_compute() {
        let light = AmbientLight::new(Vec3::new(1.0, 1.0, 1.0), 0.3);
        let hit = HitRecord {
            is_hit: true,
            dst: 1.0,
            hit_point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        };
        let result = light.compute(&hit, &[]);
        assert_eq!(result, Vec3::new(0.3, 0.3, 0.3));
    }

    #[test]
    fn test_light_enum_constructors() {
        let point =
            Light::new_point_light(Vec3::new(0.0, 0.0, 0.0), 1.0, Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert!(matches!(point, Light::Point(_)));

        let directional =
            Light::new_directional_light(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert!(matches!(directional, Light::Directional(_)));

        let ambient = Light::new_ambient_light(Vec3::new(0.5, 0.5, 0.5), 0.2);
        assert!(matches!(ambient, Light::Ambient(_)));
    }
}
