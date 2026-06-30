// GPU Scene Setup Module
// This module provides utilities for setting up scenes for GPU raytracing.
// Note: The GPU uses its own WGSL shader implementation of lighting,
// separate from the CPU Scene and Light structs.

use crate::camera::Camera;
use crate::mesh::Mesh;
use crate::sphere::Sphere;
use crate::vec3::Vec3;
use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;

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
    pub center_radius: [f32; 4], // xyz: center, w: radius
    pub color: [f32; 4],         // xyz: color, w: unused
    pub material: [f32; 4],      // x: roughness, y: metallic, z: material_type, w: unused
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
    pub mesh_material: [f32; 4], // x: roughness, y: metallic, z: material_type (0.0=diffuse, 1.0=metallic), w: unused
    pub render_config: [u32; 4], // samples_per_pixel, max_bounces, padded_width_in_pixels, frame_seed
    pub accel_info: [u32; 4],    // bvh_node_count, reserved
}

pub fn vec3_to_array(vec: Vec3, w: f32) -> [f32; 4] {
    [vec.x as f32, vec.y as f32, vec.z as f32, w]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub left_first: u32,
    pub prim_count: u32,
    pub right_child: u32,
    pub _padding: u32,
}

#[derive(Clone, Copy)]
struct TrianglePrimitive {
    triangle: GpuTriangle,
    bounds_min: Vec3,
    bounds_max: Vec3,
    centroid: Vec3,
}

fn triangle_bounds(v0: Vec3, v1: Vec3, v2: Vec3) -> (Vec3, Vec3) {
    let min = Vec3::new(
        v0.x.min(v1.x).min(v2.x),
        v0.y.min(v1.y).min(v2.y),
        v0.z.min(v1.z).min(v2.z),
    );
    let max = Vec3::new(
        v0.x.max(v1.x).max(v2.x),
        v0.y.max(v1.y).max(v2.y),
        v0.z.max(v1.z).max(v2.z),
    );
    (min, max)
}

fn centroid_component(primitive: &TrianglePrimitive, axis: usize) -> f64 {
    match axis {
        0 => primitive.centroid.x,
        1 => primitive.centroid.y,
        _ => primitive.centroid.z,
    }
}

fn axis_value(vec: Vec3, axis: usize) -> f64 {
    match axis {
        0 => vec.x,
        1 => vec.y,
        _ => vec.z,
    }
}

fn surface_area(min: Vec3, max: Vec3) -> f64 {
    let dx = (max.x - min.x).max(0.0);
    let dy = (max.y - min.y).max(0.0);
    let dz = (max.z - min.z).max(0.0);
    2.0 * (dx * dy + dy * dz + dz * dx)
}

/// A bucket used by the binned Surface Area Heuristic builder.
#[derive(Clone, Copy)]
struct SahBin {
    count: usize,
    bounds_min: Vec3,
    bounds_max: Vec3,
}

impl SahBin {
    fn empty() -> Self {
        SahBin {
            count: 0,
            bounds_min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            bounds_max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    fn expand(&mut self, min: Vec3, max: Vec3) {
        self.bounds_min.x = self.bounds_min.x.min(min.x);
        self.bounds_min.y = self.bounds_min.y.min(min.y);
        self.bounds_min.z = self.bounds_min.z.min(min.z);
        self.bounds_max.x = self.bounds_max.x.max(max.x);
        self.bounds_max.y = self.bounds_max.y.max(max.y);
        self.bounds_max.z = self.bounds_max.z.max(max.z);
    }

    fn merge(&mut self, other: &SahBin) {
        self.count += other.count;
        if other.count > 0 {
            self.expand(other.bounds_min, other.bounds_max);
        }
    }

    fn area(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            surface_area(self.bounds_min, self.bounds_max)
        }
    }
}

/// Reorder `indices` so that primitives whose centroid is below `split` along
/// `axis` come first. Returns the number of primitives on the left side.
fn partition_indices(
    indices: &mut [usize],
    primitives: &[TrianglePrimitive],
    axis: usize,
    split: f64,
) -> usize {
    let mut left = 0usize;
    for i in 0..indices.len() {
        if centroid_component(&primitives[indices[i]], axis) < split {
            indices.swap(left, i);
            left += 1;
        }
    }
    left
}

/// Sort `indices` along `axis` and return the median offset (always a valid,
/// non-empty split for slices of length >= 2). Used as a fallback when SAH
/// cannot find a useful split (e.g. coincident centroids).
fn median_split(indices: &mut [usize], primitives: &[TrianglePrimitive], axis: usize) -> usize {
    indices.sort_by(|&a, &b| {
        centroid_component(&primitives[a], axis)
            .partial_cmp(&centroid_component(&primitives[b], axis))
            .unwrap_or(Ordering::Equal)
    });
    indices.len() / 2
}

fn merge_bounds(min_a: Vec3, max_a: Vec3, min_b: Vec3, max_b: Vec3) -> (Vec3, Vec3) {
    let min = Vec3::new(
        min_a.x.min(min_b.x),
        min_a.y.min(min_b.y),
        min_a.z.min(min_b.z),
    );
    let max = Vec3::new(
        max_a.x.max(max_b.x),
        max_a.y.max(max_b.y),
        max_a.z.max(max_b.z),
    );
    (min, max)
}

fn build_bvh_nodes(
    primitives: &[TrianglePrimitive],
    indices: &mut [usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<GpuBvhNode>,
    output_triangles: &mut Vec<GpuTriangle>,
) -> (u32, Vec3, Vec3) {
    const MAX_LEAF_SIZE: usize = 4;

    let node_index = nodes.len() as u32;
    nodes.push(GpuBvhNode::zeroed());

    let count = end - start;

    if count <= MAX_LEAF_SIZE {
        let first_prim = output_triangles.len() as u32;
        let mut bounds_min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut bounds_max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

        for &idx in &indices[start..end] {
            let primitive = &primitives[idx];
            output_triangles.push(primitive.triangle);
            bounds_min.x = bounds_min.x.min(primitive.bounds_min.x);
            bounds_min.y = bounds_min.y.min(primitive.bounds_min.y);
            bounds_min.z = bounds_min.z.min(primitive.bounds_min.z);
            bounds_max.x = bounds_max.x.max(primitive.bounds_max.x);
            bounds_max.y = bounds_max.y.max(primitive.bounds_max.y);
            bounds_max.z = bounds_max.z.max(primitive.bounds_max.z);
        }

        nodes[node_index as usize] = GpuBvhNode {
            bounds_min: vec3_to_array(bounds_min, 0.0),
            bounds_max: vec3_to_array(bounds_max, 0.0),
            left_first: first_prim,
            prim_count: count as u32,
            right_child: 0,
            _padding: 0,
        };

        return (node_index, bounds_min, bounds_max);
    }

    // Compute centroid bounds to determine split axis
    let mut centroid_min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut centroid_max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for &idx in &indices[start..end] {
        let c = primitives[idx].centroid;
        centroid_min.x = centroid_min.x.min(c.x);
        centroid_min.y = centroid_min.y.min(c.y);
        centroid_min.z = centroid_min.z.min(c.z);
        centroid_max.x = centroid_max.x.max(c.x);
        centroid_max.y = centroid_max.y.max(c.y);
        centroid_max.z = centroid_max.z.max(c.z);
    }

    let extent = Vec3::new(
        centroid_max.x - centroid_min.x,
        centroid_max.y - centroid_min.y,
        centroid_max.z - centroid_min.z,
    );

    // Largest-extent axis, used as the SAH fallback when no good split exists.
    let mut largest_axis = 0;
    if extent.y > extent.x {
        largest_axis = 1;
    }
    if extent.z > axis_value(extent, largest_axis) {
        largest_axis = 2;
    }

    // Binned Surface Area Heuristic: evaluate candidate split planes across all
    // three axes and pick the one minimising expected traversal cost. This
    // produces tighter, faster-to-traverse trees than a plain median split.
    const NUM_BINS: usize = 12;
    let mut best_cost = f64::INFINITY;
    let mut best_axis = largest_axis;
    let mut best_split = 0.0f64;
    let mut found_split = false;

    for axis in 0..3 {
        let axis_min = axis_value(centroid_min, axis);
        let axis_extent = axis_value(extent, axis);
        if axis_extent <= 0.0 {
            continue;
        }

        let mut bins = [SahBin::empty(); NUM_BINS];
        let scale = NUM_BINS as f64 / axis_extent;
        for &idx in &indices[start..end] {
            let c = centroid_component(&primitives[idx], axis);
            let mut bin = ((c - axis_min) * scale) as usize;
            if bin >= NUM_BINS {
                bin = NUM_BINS - 1;
            }
            bins[bin].count += 1;
            bins[bin].expand(primitives[idx].bounds_min, primitives[idx].bounds_max);
        }

        // Prefix sweep: bounds/count of everything left of each split plane.
        let mut left_area = [0.0f64; NUM_BINS - 1];
        let mut left_count = [0usize; NUM_BINS - 1];
        let mut acc = SahBin::empty();
        for i in 0..(NUM_BINS - 1) {
            acc.merge(&bins[i]);
            left_area[i] = acc.area();
            left_count[i] = acc.count;
        }

        // Suffix sweep: combine with the right side to score each plane.
        let mut acc_right = SahBin::empty();
        for i in (0..(NUM_BINS - 1)).rev() {
            acc_right.merge(&bins[i + 1]);
            if left_count[i] == 0 || acc_right.count == 0 {
                continue;
            }
            let cost =
                left_area[i] * left_count[i] as f64 + acc_right.area() * acc_right.count as f64;
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = axis_min + (i as f64 + 1.0) * (axis_extent / NUM_BINS as f64);
                found_split = true;
            }
        }
    }

    let left_count = {
        let slice = &mut indices[start..end];
        let mut lc = if found_split {
            partition_indices(slice, primitives, best_axis, best_split)
        } else {
            median_split(slice, primitives, best_axis)
        };
        // Guard against a degenerate partition (all primitives on one side).
        if lc == 0 || lc == count {
            lc = median_split(slice, primitives, best_axis);
        }
        lc
    };
    let mid = start + left_count;

    let (left_child, left_min, left_max) =
        build_bvh_nodes(primitives, indices, start, mid, nodes, output_triangles);
    let (right_child, right_min, right_max) =
        build_bvh_nodes(primitives, indices, mid, end, nodes, output_triangles);

    let (bounds_min, bounds_max) = merge_bounds(left_min, left_max, right_min, right_max);

    nodes[node_index as usize] = GpuBvhNode {
        bounds_min: vec3_to_array(bounds_min, 0.0),
        bounds_max: vec3_to_array(bounds_max, 0.0),
        left_first: left_child,
        prim_count: 0,
        right_child,
        _padding: 0,
    };

    (node_index, bounds_min, bounds_max)
}

pub fn mesh_to_gpu_data(mesh: &Mesh) -> (Vec<GpuTriangle>, Vec<GpuBvhNode>) {
    let mut primitives = Vec::with_capacity(mesh.faces.len() / 3);

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

        let triangle = GpuTriangle {
            v0: vec3_to_array(v0, 0.0),
            v1: vec3_to_array(v1, 0.0),
            v2: vec3_to_array(v2, 0.0),
            n0: vec3_to_array(n0.normalize(), 0.0),
            n1: vec3_to_array(n1.normalize(), 0.0),
            n2: vec3_to_array(n2.normalize(), 0.0),
        };

        let centroid = Vec3::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        );

        let (bounds_min, bounds_max) = triangle_bounds(v0, v1, v2);

        primitives.push(TrianglePrimitive {
            triangle,
            bounds_min,
            bounds_max,
            centroid,
        });
    }

    if primitives.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut indices: Vec<usize> = (0..primitives.len()).collect();
    let mut nodes = Vec::new();
    let mut triangles = Vec::with_capacity(primitives.len());
    build_bvh_nodes(
        &primitives,
        &mut indices,
        0,
        primitives.len(),
        &mut nodes,
        &mut triangles,
    );

    (triangles, nodes)
}

pub fn sphere_to_gpu(sphere: &Sphere) -> GpuSphere {
    use crate::material::MaterialType;

    let material_type = match sphere.material.material_type {
        MaterialType::Diffuse => 0.0,
        MaterialType::Metallic => 1.0,
        MaterialType::Dielectric => 2.0,
    };

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
        material: [
            sphere.material.roughness as f32,
            sphere.material.metallic as f32,
            material_type,
            0.0,
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
