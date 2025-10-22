use crate::vec3::Vec3; 
use crate::ray::Ray;

pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center:Vec3, radius: f64) -> Self {
        Sphere { center, radius}
    }
}

impl Sphere {
    pub fn hit(&self, ray: &Ray) -> bool {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(&ray.direction);
        let b = 2.0 * ray.direction.dot(&oc);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;
        
        discriminant >= 0.0
    }
}