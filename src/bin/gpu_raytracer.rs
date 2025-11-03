use std::borrow::Cow;
use std::sync::mpsc;

use anyhow::{Context, Result, anyhow};
use bytemuck::Zeroable;
use rust_raytracer::camera::Camera;
use rust_raytracer::gpu_scene::*;
use rust_raytracer::material::Material;
use rust_raytracer::mesh::Mesh;
use rust_raytracer::sphere::Sphere;
use rust_raytracer::vec3::Vec3;
use wgpu::util::DeviceExt;

// Material constants
const METALLIC_THRESHOLD: f32 = 0.5;

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let width: u32 = 800;
    let height: u32 = 600;

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
        60.0,
        width as f64,
        height as f64,
    );

    let bunny_material = Material::new(Vec3::new(1.0, 0.0, 0.0), 0.2);
    let mut bunny = Mesh::from_obj_file("data/bunny.obj", bunny_material)
        .map_err(|err| anyhow!("failed to load bunny OBJ: {err}"))?;
    bunny.rotate_y(180.0);
    bunny.transform(10.0, Vec3::new(0.0, -1.0, 4.0));

    let sphere1 = Sphere::new(
        Vec3::new(0.0, -1001.0, 5.0),
        1000.0,
        Material::new_metallic(Vec3::new(1.0, 1.0, 1.0), 0.07),
    );

    let sphere2 = Sphere::new(
        Vec3::new(2.0, 0.0, 5.0),
        1.0,
        Material::new(Vec3::new(0.0, 1.0, 0.0), 0.5),
    );
    let sphere3 = Sphere::new(
        Vec3::new(-1.6, 0.0, 5.0),
        1.0,
        Material::new(Vec3::new(0.0, 0.0, 1.0), 0.5),
    );
    let spheres = [sphere1, sphere2, sphere3];

    let (triangles, mut bvh_nodes) = mesh_to_gpu_data(&bunny);
    let triangle_count = triangles.len() as u32;
    if bvh_nodes.is_empty() {
        bvh_nodes.push(GpuBvhNode::zeroed());
    }
    let bvh_node_count = if triangle_count == 0 {
        0
    } else {
        bvh_nodes.len() as u32
    };
    let triangles_storage: Cow<[GpuTriangle]> = if triangles.is_empty() {
        Cow::Owned(vec![GpuTriangle::zeroed()])
    } else {
        Cow::Owned(triangles)
    };
    let bvh_storage: Cow<[GpuBvhNode]> = Cow::Owned(bvh_nodes);

    let gpu_spheres: Vec<GpuSphere> = spheres.iter().map(sphere_to_gpu).collect();
    let sphere_count = gpu_spheres.len() as u32;
    let spheres_storage: Cow<[GpuSphere]> = if gpu_spheres.is_empty() {
        Cow::Owned(vec![GpuSphere::zeroed()])
    } else {
        Cow::Owned(gpu_spheres)
    };

    let (lower_left_corner, horizontal, vertical) = camera_frame(&camera);

    let directional_dir = Vec3::new(3.0, -3.0, 3.0).normalize();
    let directional_strength = 0.8_f64;
    let directional_color = Vec3::new(1.0, 1.0, 1.0);
    let ambient_color = Vec3::new(0.1, 0.1, 0.1) * 0.2;
    let samples_per_pixel = 256u32; // Increased from 10 to 100
    let max_bounces = 3u32;

    // Storage buffer is laid out in tightly packed rows, so padded width just equals image width.
    let padded_width = width;

    let mesh_material_type = if bunny.material.metallic as f32 > METALLIC_THRESHOLD { 1.0 } else { 0.0 };
    let scene_uniform = SceneUniform {
        resolution: [width, height, triangle_count, sphere_count],
        camera_position: vec3_to_array(camera.position, 0.0),
        lower_left_corner: vec3_to_array(lower_left_corner, 0.0),
        horizontal: vec3_to_array(horizontal, 0.0),
        vertical: vec3_to_array(vertical, 0.0),
        light_direction: vec3_to_array(directional_dir, directional_strength as f32),
        light_color: vec3_to_array(directional_color, 0.0),
        ambient_color: vec3_to_array(ambient_color, 1.0), // w: environment_strength
        mesh_color: vec3_to_array(bunny.material.color, 1.0),
        mesh_material: [bunny.material.roughness as f32, bunny.material.metallic as f32, mesh_material_type, 0.0],
        render_config: [samples_per_pixel, max_bounces, padded_width, 0],
        accel_info: [bvh_node_count, 0, 0, 0],
    };

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("failed to find GPU adapter")?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-raytracer-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .context("failed to create device")?;

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene-uniform"),
        contents: bytemuck::bytes_of(&scene_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("triangle-buffer"),
        contents: bytemuck::cast_slice(triangles_storage.as_ref()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bvh-buffer"),
        contents: bytemuck::cast_slice(bvh_storage.as_ref()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sphere-buffer"),
        contents: bytemuck::cast_slice(spheres_storage.as_ref()),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pixel_stride = (std::mem::size_of::<f32>() * 4) as u64;
    let output_buffer_size = pixel_stride * width as u64 * height as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output-buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raytracer-bind-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raytracer-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: triangle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sphere_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bvh_buffer.as_entire_binding(),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raytracer-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/raytracer.wgsl"))),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raytracer-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raytracer-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("raytracer-encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raytracer-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = [8u32, 8u32];
        let dispatch_x = (width + workgroup_size[0] - 1).div_ceil(workgroup_size[0]);
        let dispatch_y = (height + workgroup_size[1] - 1).div_ceil(workgroup_size[1]);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .context("failed to receive staging buffer map result")?
        .context("failed to map staging buffer")?;

    {
        let data = buffer_slice.get_mapped_range();
        let pixels: &[f32] = bytemuck::cast_slice(&data);

        println!("P3");
        println!("{} {}", width, height);
        println!("255");

        for j in (0..height).rev() {
            for i in 0..width {
                let idx = ((j * width) + i) as usize * 4;
                let r_linear = pixels[idx];
                let g_linear = pixels[idx + 1];
                let b_linear = pixels[idx + 2];

                let r = r_linear.clamp(0.0, 1.0).sqrt();
                let g = g_linear.clamp(0.0, 1.0).sqrt();
                let b = b_linear.clamp(0.0, 1.0).sqrt();

                let ir = (255.0 * r) as u8;
                let ig = (255.0 * g) as u8;
                let ib = (255.0 * b) as u8;

                println!("{} {} {}", ir, ig, ib);
            }
        }
    }

    staging_buffer.unmap();

    Ok(())
}
