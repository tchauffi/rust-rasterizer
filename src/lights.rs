use crate::hittable::HitRecord;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::texture::Texture;
use crate::vec3::Vec3;

const EPSILON: f64 = 1e-3;

#[allow(dead_code)]
#[derive(Clone)]
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
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Direction from hit point to light
        let to_light = self.position - hit.hit_point;
        let light_distance = to_light.length();
        let light_dir = to_light.normalize();

        // Shadow check
        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
        let shadow_ray = Ray::new(shadow_origin, light_dir);
        let mut in_shadow = false;

        for obj in objects {
            let shadow_hit = obj.hit(&shadow_ray);
            if shadow_hit.is_hit && shadow_hit.dst < light_distance - EPSILON {
                in_shadow = true;
                break;
            }
        }

        if !in_shadow {
            // Diffuse lighting
            let diffuse = hit.normal.dot(&light_dir).max(0.0);
            let attenuation = self.intensity / (light_distance * light_distance);
            self.color * diffuse * attenuation
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }
}
#[derive(Clone)]
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
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        let light_dir = (self.direction * -1.0).normalize();

        // Shadow check
        let shadow_origin = hit.hit_point + hit.normal * EPSILON;
        let shadow_ray = Ray::new(shadow_origin, light_dir);
        let mut in_shadow = false;

        for obj in objects {
            let shadow_hit = obj.hit(&shadow_ray);
            if shadow_hit.is_hit {
                in_shadow = true;
                break;
            }
        }

        if !in_shadow {
            let diffuse = hit.normal.dot(&light_dir).max(0.0);
            self.color * self.intensity * diffuse
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }
}
#[derive(Clone)]
pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f64,
}

impl AmbientLight {
    pub fn new(color: Vec3, intensity: f64) -> Self {
        AmbientLight { color, intensity }
    }
    pub fn compute(&self, _hit: &HitRecord, _objects: &[Box<dyn Hittable>]) -> Vec3 {
        self.color * self.intensity
    }
}

#[derive(Clone)]
pub struct TextureLight {
    pub texture: Texture,
    pub intensity: f64,
}

impl TextureLight {
    pub fn new(texture: Texture, intensity: f64) -> Self {
        TextureLight { texture, intensity }
    }

    pub fn compute(&self, hit: &HitRecord, _objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Convert hit point to UV coordinates
        // This is a simple spherical mapping - can be extended for other mappings
        let (u, v) = self.world_to_uv(&hit.hit_point, &hit.normal);

        // Sample the texture with bilinear interpolation for smooth results
        let texture_color = self.texture.get_color_with_interpolation(
            u,
            v,
            crate::texture::InterpolationMethod::Bilinear,
        );

        texture_color * self.intensity
    }

    /// Converts world space coordinates to UV coordinates
    /// Uses spherical mapping based on the normal direction
    fn world_to_uv(&self, _point: &Vec3, normal: &Vec3) -> (f64, f64) {
        let u = 0.5 + normal.z.atan2(normal.x) / (2.0 * std::f64::consts::PI);
        let v = 0.5 - normal.y.asin() / std::f64::consts::PI;
        (u, v)
    }
}

#[derive(Clone)]
pub struct EnvironmentLight {
    pub texture: Texture,
    pub intensity: f64,
}

impl EnvironmentLight {
    pub fn new(texture: Texture, intensity: f64) -> Self {
        EnvironmentLight { texture, intensity }
    }

    pub fn compute(&self, ray_direction: &Vec3) -> Vec3 {
        // Convert ray direction to spherical UV coordinates
        let (u, v) = self.direction_to_uv(ray_direction);

        // Sample the environment map with bilinear interpolation
        let env_color = self.texture.get_color_with_interpolation(
            u,
            v,
            crate::texture::InterpolationMethod::Bilinear,
        );

        env_color * self.intensity
    }

    /// Converts a ray direction to UV coordinates for environment mapping
    /// Uses equirectangular/spherical mapping
    fn direction_to_uv(&self, direction: &Vec3) -> (f64, f64) {
        let dir = direction.normalize();
        let u = 0.5 + dir.z.atan2(dir.x) / (2.0 * std::f64::consts::PI);
        let v = 0.5 - dir.y.asin() / std::f64::consts::PI;
        (u, v)
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum Light {
    Point(PointLight),
    Directional(DirectionalLight),
    Ambient(AmbientLight),
    Texture(TextureLight),
    Environment(EnvironmentLight),
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

    pub fn new_texture_light(texture: Texture, intensity: f64) -> Self {
        Light::Texture(TextureLight::new(texture, intensity))
    }

    pub fn new_environment_light(texture: Texture, intensity: f64) -> Self {
        Light::Environment(EnvironmentLight::new(texture, intensity))
    }

    pub fn compute_contribution(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        match self {
            Light::Ambient(ambient) => ambient.compute(hit, objects),
            Light::Point(point) => point.compute(hit, objects),
            Light::Directional(dir) => dir.compute(hit, objects),
            Light::Texture(texture) => texture.compute(hit, objects),
            Light::Environment(_) => Vec3::new(0.0, 0.0, 0.0), // Environment light is sampled during ray misses
        }
    }
}

// tests for lights
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_creation() {
        let light = PointLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            1.0,
            Vec3::new(1.0, 1.0, 1.0),
            10.0,
        );
        assert_eq!(light.position, Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(light.radius, 1.0);
        assert_eq!(light.intensity, 10.0);
    }

    #[test]
    fn test_directional_light_creation() {
        let light = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert_eq!(light.direction, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(light.intensity, 1.0);
    }

    #[test]
    fn test_ambient_light_creation() {
        let light = AmbientLight::new(Vec3::new(0.2, 0.2, 0.2), 0.5);
        assert_eq!(light.color, Vec3::new(0.2, 0.2, 0.2));
        assert_eq!(light.intensity, 0.5);
    }

    #[test]
    fn test_ambient_light_compute() {
        let light = AmbientLight::new(Vec3::new(1.0, 1.0, 1.0), 0.3);
        let hit = HitRecord {
            is_hit: true,
            dst: 1.0,
            hit_point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
        };
        let result = light.compute(&hit, &[]);
        assert_eq!(result, Vec3::new(0.3, 0.3, 0.3));
    }

    #[test]
    fn test_light_enum_constructors() {
        let point =
            Light::new_point_light(Vec3::new(0.0, 0.0, 0.0), 1.0, Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert!(matches!(point, Light::Point(_)));

        let directional =
            Light::new_directional_light(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert!(matches!(directional, Light::Directional(_)));

        let ambient = Light::new_ambient_light(Vec3::new(0.5, 0.5, 0.5), 0.2);
        assert!(matches!(ambient, Light::Ambient(_)));
    }

    #[test]
    fn test_environment_light_creation() {
        let texture = Texture::default();
        let light = EnvironmentLight::new(texture, 1.0);
        assert_eq!(light.intensity, 1.0);
    }

    #[test]
    fn test_environment_light_direction_to_uv() {
        let texture = Texture::default();
        let light = EnvironmentLight::new(texture, 1.0);

        // Test positive Z direction (0, 0, 1)
        // atan2(1.0, 0.0) = π/2, so u = 0.5 + 0.25 = 0.75
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = light.direction_to_uv(&dir);
        assert!((u - 0.75).abs() < 0.01, "Expected u ≈ 0.75, got {}", u);
        assert!((v - 0.5).abs() < 0.01, "Expected v ≈ 0.5, got {}", v);

        // Test positive X direction (1, 0, 0)
        // atan2(0.0, 1.0) = 0, so u = 0.5 + 0.0 = 0.5
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let (u, v) = light.direction_to_uv(&dir);
        assert!((u - 0.5).abs() < 0.01, "Expected u ≈ 0.5, got {}", u);
        assert!((v - 0.5).abs() < 0.01, "Expected v ≈ 0.5, got {}", v);

        // Test positive Y direction (0, 1, 0) - straight up
        // asin(1.0) = π/2, so v = 0.5 - 0.5 = 0.0
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let (_u, v) = light.direction_to_uv(&dir);
        assert!((v - 0.0).abs() < 0.01, "Expected v ≈ 0.0, got {}", v);

        // Test negative Y direction (0, -1, 0) - straight down
        // asin(-1.0) = -π/2, so v = 0.5 - (-0.5) = 1.0
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let (_u, v) = light.direction_to_uv(&dir);
        assert!((v - 1.0).abs() < 0.01, "Expected v ≈ 1.0, got {}", v);
    }

    #[test]
    fn test_environment_light_enum() {
        let texture = Texture::default();
        let env_light = Light::new_environment_light(texture, 1.5);
        assert!(matches!(env_light, Light::Environment(_)));
    }
}
