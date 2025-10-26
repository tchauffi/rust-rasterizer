use crate::aabb::AABB;
use crate::hit::HitRecord;
use crate::hittable::Hittable;
use crate::ray::Ray;
use crate::sphere::Material;
use crate::vec3::Vec3;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[allow(dead_code)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub faces: Vec<usize>,
    pub normals: Vec<Vec3>,
    pub texture_coords: Vec<(f64, f64)>,
    pub material: Material,
    pub bounding_box: AABB,
}

impl Mesh {
    pub fn new(
        vertices: Vec<Vec3>,
        faces: Vec<usize>,
        normals: Vec<Vec3>,
        texture_coords: Vec<(f64, f64)>,
        material: Material,
    ) -> Self {
        let bounding_box = AABB::from_vertices(&vertices);
        Mesh {
            vertices,
            faces,
            normals,
            texture_coords,
            material,
            bounding_box,
        }
    }
    pub fn from_obj_file(
        file_path: &str,
        material: Material,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut normals = Vec::new();
        let mut texture_coords = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "v" => {
                    if parts.len() >= 4 {
                        let x = parts[1].parse::<f64>()?;
                        let y = parts[2].parse::<f64>()?;
                        let z = parts[3].parse::<f64>()?;
                        vertices.push(Vec3::new(x, y, z));
                    }
                }
                "f" => {
                    for part in parts.iter().skip(1) {
                        let vertex_index = part.split('/').next().unwrap().parse::<usize>()? - 1;
                        faces.push(vertex_index);
                    }
                }
                "vn" => {
                    if parts.len() >= 4 {
                        let x = parts[1].parse::<f64>()?;
                        let y = parts[2].parse::<f64>()?;
                        let z = parts[3].parse::<f64>()?;
                        normals.push(Vec3::new(x, y, z));
                    }
                }
                "vt" => {
                    if parts.len() >= 3 {
                        let u = parts[1].parse::<f64>()?;
                        let v = parts[2].parse::<f64>()?;
                        texture_coords.push((u, v));
                    }
                }
                _ => {}
            }
        }

        let mut mesh = Mesh::new(vertices, faces, normals, texture_coords, material);
        if mesh.normals.is_empty() {
            mesh.compute_normals();
        }
        eprintln!("Computed normals for mesh: {} normals", mesh.normals.len());
        Ok(mesh)
    }

    /// Transform the mesh by scaling and translating
    pub fn transform(&mut self, scale: f64, translation: Vec3) {
        for vertex in &mut self.vertices {
            *vertex = (*vertex * scale) + translation;
        }
        // Recompute bounding box after transformation
        self.bounding_box = AABB::from_vertices(&self.vertices);
    }

    /// Rotate the mesh around the Y axis (vertical)
    /// angle is in degrees
    pub fn rotate_y(&mut self, angle_degrees: f64) {
        let angle_radians = angle_degrees.to_radians();
        let cos_theta = angle_radians.cos();
        let sin_theta = angle_radians.sin();

        // Rotate vertices
        for vertex in &mut self.vertices {
            let x = vertex.x;
            let z = vertex.z;

            // Rotation matrix around Y axis:
            // [cos  0  sin]   [x]
            // [0    1  0  ] * [y]
            // [-sin 0  cos]   [z]
            vertex.x = cos_theta * x + sin_theta * z;
            vertex.z = -sin_theta * x + cos_theta * z;
            // y stays the same
        }

        // Rotate normals as well!
        for normal in &mut self.normals {
            let x = normal.x;
            let z = normal.z;

            normal.x = cos_theta * x + sin_theta * z;
            normal.z = -sin_theta * x + cos_theta * z;
            // y stays the same
        }

        // Recompute bounding box after rotation
        self.bounding_box = AABB::from_vertices(&self.vertices);
    }
    pub fn compute_normals(&mut self) {
        self.normals.clear();
        self.normals
            .resize(self.vertices.len(), Vec3::new(0.0, 0.0, 0.0));
        let mut counts = vec![0; self.vertices.len()];

        for i in (0..self.faces.len()).step_by(3) {
            if i + 2 >= self.faces.len() {
                break;
            }

            let v0 = self.vertices[self.faces[i]];
            let v1 = self.vertices[self.faces[i + 1]];
            let v2 = self.vertices[self.faces[i + 2]];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let face_normal = edge1.cross(&edge2).normalize();

            for j in 0..3 {
                let vertex_index = self.faces[i + j];
                self.normals[vertex_index] = self.normals[vertex_index] + face_normal;
                counts[vertex_index] += 1;
            }
        }

        for (i, _) in counts.iter().enumerate().take(self.normals.len()) {
            if counts[i] > 0 {
                self.normals[i] = (self.normals[i] / counts[i] as f64).normalize();
            }
        }
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray) -> HitRecord {
        // Early rejection: test against bounding box first
        if !self.bounding_box.intersect(ray) {
            return HitRecord {
                is_hit: false,
                dst: f64::INFINITY,
                hit_point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 0.0),
            };
        }

        let mut closest_hit = HitRecord {
            is_hit: false,
            dst: f64::INFINITY,
            hit_point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 0.0),
        };
        let mut closest_t = f64::INFINITY;

        // Iterate through triangles (assuming faces are stored as consecutive triplets)
        for i in (0..self.faces.len()).step_by(3) {
            if i + 2 >= self.faces.len() {
                break;
            }

            let v0 = self.vertices[self.faces[i]];
            let v1 = self.vertices[self.faces[i + 1]];
            let v2 = self.vertices[self.faces[i + 2]];

            // Möller-Trumbore ray-triangle intersection algorithm
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let h = ray.direction.cross(&edge2);
            let a = edge1.dot(&h);

            if a.abs() < 1e-8 {
                continue; // Ray is parallel to triangle
            }

            let f = 1.0 / a;
            let s = ray.origin - v0;
            let u = f * s.dot(&h);

            if !(0.0..=1.0).contains(&u) {
                continue;
            }

            let q = s.cross(&edge1);
            let v = f * ray.direction.dot(&q);

            if v < 0.0 || u + v > 1.0 {
                continue;
            }

            let t = f * edge2.dot(&q);

            if t > 1e-8 && t < closest_t {
                closest_t = t;
                let hit_point = ray.at(t);
                let i0 = self.faces[i];
                let i1 = self.faces[i + 1];
                let i2 = self.faces[i + 2];
                let normal = (self.normals[i0] * (1.0 - u - v)
                    + self.normals[i1] * u
                    + self.normals[i2] * v)
                    .normalize();

                closest_hit = HitRecord {
                    dst: t,
                    hit_point,
                    normal,
                    is_hit: true,
                };
            }
        }

        closest_hit
    }

    fn get_material(&self) -> &Material {
        &self.material
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    #[test]
    fn test_mesh_from_obj_file() {
        let obj_data = "# Test OBJ file
                                        v 0.0 0.0 0.0
                                        v 1.0 0.0 0.0
                                        v 0.0 1.0 0.0
                                        f 1 2 3
                                        ";
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", obj_data).unwrap();

        let material = Material::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.0);
        let mesh = Mesh::from_obj_file(temp_file.path().to_str().unwrap(), material).unwrap();
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.faces.len(), 3);
        assert_eq!(mesh.vertices[0], Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(mesh.faces[0], 0);
    }
}
