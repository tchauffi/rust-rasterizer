use crate::hit::HitRecord;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b }
    }
}

#[allow(dead_code)]
pub struct Material {
    pub color: Color,
    pub emissive_color: Color,
    pub emissive_strength: f64,
}

impl Material {
    pub fn new(color: Color, emissive_color: Color, emissive_strength: f64) -> Self {
        Material {
            color,
            emissive_color,
            emissive_strength,
        }
    }
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material: Material) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Sphere {
    pub fn hit(&self, ray: &Ray) -> HitRecord {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * ray.direction.dot(&oc);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant >= 0.0 {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            if t > 0.0 {
                let hit_point = ray.at(t);
                let normal = (hit_point - self.center).normalize();
                HitRecord::new(true, t, hit_point, normal)
            } else {
                HitRecord::new(
                    false,
                    f64::INFINITY,
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 0.0),
                )
            }
        } else {
            HitRecord::new(
                false,
                f64::INFINITY,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
            )
        }
    }
}
