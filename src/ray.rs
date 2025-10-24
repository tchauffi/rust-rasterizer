use crate::vec3::Vec3;

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }
}

impl Ray {
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

impl std::fmt::Display for Ray {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Ray(origin: {}, direction: {})",
            self.origin, self.direction
        )
    }
}
