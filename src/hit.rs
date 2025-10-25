use crate::vec3::Vec3;

pub struct HitRecord {
    pub is_hit: bool,
    pub dst: f64,
    pub hit_point: Vec3,
    pub normal: Vec3,
}

impl HitRecord {
    pub fn new(is_hit: bool, dst: f64, hit_point: Vec3, normal: Vec3) -> Self {
        HitRecord {
            is_hit,
            dst,
            hit_point,
            normal,
        }
    }
}
