use crate::vec3::Vec3;

#[allow(dead_code)]
pub struct Material {
    pub color: Vec3,
    pub roughness: f64,
}

impl Material {
    pub fn new(color: Vec3, roughness: f64) -> Self {
        Material { color, roughness }
    }
}
