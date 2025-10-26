use crate::vec3::Vec3;

pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub strength: f64,
}

#[allow(dead_code)]
impl PointLight {
    pub fn new(position: Vec3, color: Vec3, strength: f64) -> Self {
        PointLight {
            position,
            color,
            strength,
        }
    }
}

pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub strength: f64,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, strength: f64) -> Self {
        DirectionalLight {
            direction,
            color,
            strength,
        }
    }
}

pub struct AmbientLight {
    pub color: Vec3,
    pub strength: f64,
}

impl AmbientLight {
    pub fn new(color: Vec3, strength: f64) -> Self {
        AmbientLight { color, strength }
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
    pub fn new_point_light(position: Vec3, color: Vec3, strength: f64) -> Self {
        Light::Point(PointLight::new(position, color, strength))
    }

    pub fn new_directional_light(direction: Vec3, color: Vec3, strength: f64) -> Self {
        Light::Directional(DirectionalLight::new(direction, color, strength))
    }

    pub fn new_ambient_light(color: Vec3, strength: f64) -> Self {
        Light::Ambient(AmbientLight::new(color, strength))
    }
}
