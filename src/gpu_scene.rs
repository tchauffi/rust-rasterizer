// GPU Scene Setup Module
// This module provides utilities for setting up scenes for GPU raytracing.
// Note: The GPU uses its own WGSL shader implementation of lighting,
// separate from the CPU Scene and Light structs.

use crate::camera::Camera;
use crate::mesh::Mesh;
use crate::sphere::Sphere;
use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuTriangle {
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub v2: [f32; 4],
    pub n0: [f32; 4],
    pub n1: [f32; 4],
    pub n2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSphere {
    pub center_radius: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SceneUniform {
    pub resolution: [u32; 4], // width, height, triangle_count, sphere_count
    pub camera_position: [f32; 4],
    pub lower_left_corner: [f32; 4],
    pub horizontal: [f32; 4],
    pub vertical: [f32; 4],
    pub light_direction: [f32; 4], // xyz: direction, w: intensity
    pub light_color: [f32; 4],
    pub ambient_color: [f32; 4],
    pub mesh_color: [f32; 4],
    pub render_config: [u32; 4], // samples_per_pixel, max_bounces, padded_width_in_pixels, frame_seed
}

pub fn vec3_to_array(vec: Vec3, w: f32) -> [f32; 4] {
    [vec.x as f32, vec.y as f32, vec.z as f32, w]
}

pub fn mesh_to_gpu_triangles(mesh: &Mesh) -> Vec<GpuTriangle> {
    let mut triangles = Vec::with_capacity(mesh.faces.len() / 3);
    for face in mesh.faces.chunks_exact(3) {
        let i0 = face[0];
        let i1 = face[1];
        let i2 = face[2];

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let face_normal = (v1 - v0).cross(&(v2 - v0)).normalize();

        let n0 = mesh.normals.get(i0).copied().unwrap_or(face_normal);
        let n1 = mesh.normals.get(i1).copied().unwrap_or(face_normal);
        let n2 = mesh.normals.get(i2).copied().unwrap_or(face_normal);

        triangles.push(GpuTriangle {
            v0: vec3_to_array(v0, 0.0),
            v1: vec3_to_array(v1, 0.0),
            v2: vec3_to_array(v2, 0.0),
            n0: vec3_to_array(n0.normalize(), 0.0),
            n1: vec3_to_array(n1.normalize(), 0.0),
            n2: vec3_to_array(n2.normalize(), 0.0),
        });
    }
    triangles
}

pub fn sphere_to_gpu(sphere: &Sphere) -> GpuSphere {
    GpuSphere {
        center_radius: [
            sphere.center.x as f32,
            sphere.center.y as f32,
            sphere.center.z as f32,
            sphere.radius as f32,
        ],
        color: [
            sphere.material.color.x as f32,
            sphere.material.color.y as f32,
            sphere.material.color.z as f32,
            1.0,
        ],
    }
}

pub fn camera_frame(camera: &Camera) -> (Vec3, Vec3, Vec3) {
    let aspect_ratio = camera.width / camera.height;
    let theta = camera.fov.to_radians();
    let half_height = (theta / 2.0).tan();
    let half_width = aspect_ratio * half_height;

    let w = (-camera.look_direction).normalize();
    let u = camera.up.cross(&w).normalize();
    let v = w.cross(&u);

    let lower_left_corner = camera.position - u * half_width - v * half_height + w;
    let horizontal = u * (2.0 * half_width);
    let vertical = v * (2.0 * half_height);

    (lower_left_corner, horizontal, vertical)
}
