use crate::vec3::Vec3;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialType {
    Diffuse = 0,
    Metallic = 1,
    Dielectric = 2, // For future glass/transparent materials
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct Material {
    pub color: Vec3,
    pub roughness: f64,
    pub material_type: MaterialType,
    pub metallic: f64, // 0.0 = diffuse, 1.0 = fully metallic
}

#[allow(dead_code)]
impl Material {
    pub fn new(color: Vec3, roughness: f64) -> Self {
        Material {
            color,
            roughness,
            material_type: MaterialType::Diffuse,
            metallic: 0.0,
        }
    }

    pub fn new_metallic(color: Vec3, roughness: f64) -> Self {
        Material {
            color,
            roughness,
            material_type: MaterialType::Metallic,
            metallic: 1.0,
        }
    }

    pub fn new_mixed(color: Vec3, roughness: f64, metallic: f64) -> Self {
        let material_type = if metallic > 0.5 {
            MaterialType::Metallic
        } else {
            MaterialType::Diffuse
        };
        Material {
            color,
            roughness,
            material_type,
            metallic: metallic.clamp(0.0, 1.0),
        }
    }
}
