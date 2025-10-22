use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct Camera{
    pub position: Vec3,
    pub look_direction: Vec3,
    pub up: Vec3,
    pub fov: f64,
    pub width: f64,
    pub height: f64,
}

impl Camera {
    pub fn new(position: Vec3, look_direction: Vec3, up: Vec3, fov: f64, width: f64, height: f64) -> Self {
        Camera {
            position,
            look_direction,
            up,
            fov,
            width,
            height,
        }
    }
}

impl Camera {
    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        let aspect_ratio = self.width / self.height;
        let theta = self.fov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;

        let w = (-self.look_direction).normalize();
        let u_vec = self.up.cross(&w).normalize();
        let v_vec = w.cross(&u_vec);

        let lower_left_corner = self.position - u_vec * half_width - v_vec * half_height + w;
        let horizontal = u_vec * (2.0 * half_width);
        let vertical = v_vec * (2.0 * half_height);

        let direction = lower_left_corner + horizontal * u + vertical * v - self.position;

        Ray::new(self.position, direction)
    }
}

impl std::fmt::Display for Camera {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Camera(position: {}, look_direction: {}, up: {}, fov: {}, width: {}, height: {})",
               self.position, self.look_direction, self.up, self.fov, self.width, self.height)
    }
}
