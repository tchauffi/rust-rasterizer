use crate::hit::HitRecord;
use crate::ray::Ray;
use crate::sphere::Material;

pub trait Hittable {
    fn hit(&self, ray: &Ray) -> HitRecord;
    fn get_material(&self) -> &Material;
}
