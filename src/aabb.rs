use crate::ray::Ray;
use crate::vec3::Vec3;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        AABB { min, max }
    }

    /// Create an AABB that encompasses all vertices
    pub fn from_vertices(vertices: &[Vec3]) -> Self {
        if vertices.is_empty() {
            return AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        }

        let mut min = vertices[0];
        let mut max = vertices[0];

        for v in vertices.iter().skip(1) {
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
        }

        AABB::new(min, max)
    }

    /// Fast ray-AABB intersection test using slab method
    /// Returns true if the ray intersects the AABB
    pub fn intersect(&self, ray: &Ray) -> bool {
        let mut tmin = f64::NEG_INFINITY;
        let mut tmax = f64::INFINITY;

        // X axis
        let inv_d = 1.0 / ray.direction.x;
        let mut t0 = (self.min.x - ray.origin.x) * inv_d;
        let mut t1 = (self.max.x - ray.origin.x) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmax <= tmin {
            return false;
        }

        // Y axis
        let inv_d = 1.0 / ray.direction.y;
        let mut t0 = (self.min.y - ray.origin.y) * inv_d;
        let mut t1 = (self.max.y - ray.origin.y) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmax <= tmin {
            return false;
        }

        // Z axis
        let inv_d = 1.0 / ray.direction.z;
        let mut t0 = (self.min.z - ray.origin.z) * inv_d;
        let mut t1 = (self.max.z - ray.origin.z) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmax <= tmin {
            return false;
        }

        tmax > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersect_hit() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        assert!(aabb.intersect(&ray));
    }

    #[test]
    fn test_aabb_intersect_miss() {
        let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

        let ray = Ray::new(Vec3::new(0.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        assert!(!aabb.intersect(&ray));
    }
}
