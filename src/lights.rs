use crate::vec3::Vec3;

#[allow(dead_code)]
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
}

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
}

pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f64,
}

impl AmbientLight {
    pub fn new(color: Vec3, intensity: f64) -> Self {
        AmbientLight { color, intensity }
    }
}

#[allow(dead_code)]
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
}
