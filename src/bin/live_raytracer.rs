use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use bytemuck::Zeroable;
use rust_raytracer::camera::Camera;

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

// WASM support
use rust_raytracer::gpu_scene::*;
use rust_raytracer::material::Material;
use rust_raytracer::mesh::Mesh;
use rust_raytracer::sphere::Sphere;
use rust_raytracer::vec3::Vec3;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

const DEFAULT_ENV_MAP: &str = "data/rogland_sunset_4k.exr";
const MAX_ENV_DIM: u32 = 2048;

// UI Scene editing state
#[derive(Clone)]
struct SceneObject {
    name: String,
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
    roughness: f32,
    metallic: f32,
    enabled: bool,
}

struct UIState {
    // Environment controls
    environment_maps: Vec<String>,
    selected_environment: usize,
    environment_strength: f32,

    // Render settings
    samples_per_pixel: u32,
    max_bounces: u32,

    // Mesh properties
    mesh_color: [f32; 3],
    mesh_position: [f32; 3],
    mesh_roughness: f32,
    mesh_metallic: f32,

    // Scene objects
    objects: Vec<SceneObject>,

    // Add object dialog
    show_add_object_dialog: bool,
    new_object_position: [f32; 3],
    new_object_radius: f32,
    new_object_color: [f32; 3],

    // File upload dialog (WASM only)
    #[allow(dead_code)]
    show_file_upload_dialog: bool,

    // UI visibility
    show_ui: bool,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<winit::window::Window>,

    // Raytracing resources
    uniform_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    compute_pipeline_normals: wgpu::ComputePipeline,

    // Accumulation resources
    accumulation_pipeline: wgpu::ComputePipeline,
    accumulation_bind_group: wgpu::BindGroup,
    accumulation_uniform_buffer: wgpu::Buffer,

    // Display resources
    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,
    display_bind_group_layout: wgpu::BindGroupLayout,
    output_texture: wgpu::Texture,
    sampler: wgpu::Sampler,

    // Environment map resources
    environment_texture: wgpu::Texture,
    environment_sampler: wgpu::Sampler,
    environment_path: String,
    pending_environment_change: Option<String>,

    // Scene data
    camera: Camera,
    scene_uniform: SceneUniform,

    // Dynamic scene buffers
    triangle_buffer: wgpu::Buffer,
    bvh_buffer: wgpu::Buffer,
    sphere_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    bunny_mesh: Mesh,
    original_mesh: Mesh,

    // Camera control
    camera_position: Vec3,
    camera_yaw: f64,   // Rotation around Y axis
    camera_pitch: f64, // Rotation up/down

    // Mouse state
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pressed: bool,

    // Render mode
    use_raytracing: bool, // true = raytracing, false = normals view

    // Temporal accumulation
    accumulation_buffer: wgpu::Buffer,
    accumulation_texture: wgpu::Texture,
    accumulated_frames: u32,
    camera_moved: bool,

    // FPS tracking
    frame_count: u32,
    fps_timer: Instant,
    current_fps: f32,

    // UI
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_ctx: egui::Context,
    ui_state: UIState,
}

impl State {
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_environment_maps() -> Vec<String> {
        let mut maps = Vec::new();
        if let Ok(entries) = fs::read_dir("data") {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
                    let ext_lower = ext.to_ascii_lowercase();
                    if ext_lower == "exr" {
                        maps.push(path.to_string_lossy().into_owned());
                    }
                }
            }
        }

        if !maps.iter().any(|p| p == DEFAULT_ENV_MAP) {
            maps.push(DEFAULT_ENV_MAP.to_string());
        }

        maps.sort();
        maps
    }

    #[cfg(target_arch = "wasm32")]
    fn discover_environment_maps() -> Vec<String> {
        vec![DEFAULT_ENV_MAP.to_string()]
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_environment_rgb(path: &str) -> Result<(u32, u32, Vec<u8>)> {
        use rust_raytracer::texture::Texture;

        eprintln!("🔍 Attempting to load environment map from: {}", path);
        let mut texture = Texture::from_exr(path).map_err(|err| anyhow!(err.to_string()))?;
        eprintln!(
            "✅ Loaded environment map: {}x{}",
            texture.width, texture.height
        );

        if texture.width > MAX_ENV_DIM || texture.height > MAX_ENV_DIM {
            let scale = (MAX_ENV_DIM as f32 / texture.width.max(texture.height) as f32).min(1.0);
            let new_width = (texture.width as f32 * scale) as u32;
            let new_height = (texture.height as f32 * scale) as u32;

            eprintln!(
                "⚠️ Downsampling environment map from {}x{} to {}x{}",
                texture.width, texture.height, new_width, new_height
            );

            let mut new_data = Vec::with_capacity((new_width * new_height * 3) as usize);
            for y in 0..new_height {
                for x in 0..new_width {
                    let src_x = (x as f32 / new_width as f32) * texture.width as f32;
                    let src_y = (y as f32 / new_height as f32) * texture.height as f32;

                    let x0 = src_x.floor() as u32;
                    let y0 = src_y.floor() as u32;
                    let x1 = (x0 + 1).min(texture.width - 1);
                    let y1 = (y0 + 1).min(texture.height - 1);

                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;

                    let get_pixel = |px: u32, py: u32| -> (f32, f32, f32) {
                        let idx = ((py * texture.width + px) * 3) as usize;
                        (
                            texture.data[idx] as f32,
                            texture.data[idx + 1] as f32,
                            texture.data[idx + 2] as f32,
                        )
                    };

                    let p00 = get_pixel(x0, y0);
                    let p10 = get_pixel(x1, y0);
                    let p01 = get_pixel(x0, y1);
                    let p11 = get_pixel(x1, y1);

                    let interpolate = |v00: f32, v10: f32, v01: f32, v11: f32| -> u8 {
                        let top = v00 * (1.0 - fx) + v10 * fx;
                        let bottom = v01 * (1.0 - fx) + v11 * fx;
                        let result = top * (1.0 - fy) + bottom * fy;
                        result.clamp(0.0, 255.0) as u8
                    };

                    new_data.push(interpolate(p00.0, p10.0, p01.0, p11.0));
                    new_data.push(interpolate(p00.1, p10.1, p01.1, p11.1));
                    new_data.push(interpolate(p00.2, p10.2, p01.2, p11.2));
                }
            }

            texture.width = new_width;
            texture.height = new_height;
            texture.data = new_data;
        }

        let expected_size = (texture.width * texture.height * 3) as usize;
        if texture.data.len() != expected_size {
            eprintln!(
                "⚠️ WARNING: Texture data size mismatch! Expected {} bytes, got {} bytes",
                expected_size,
                texture.data.len()
            );
        }

        eprintln!(
            "📊 Final texture: {}x{}, {} bytes",
            texture.width,
            texture.height,
            texture.data.len()
        );

        let width = texture.width;
        let height = texture.height;
        let data = texture.data;
        Ok((width, height, data))
    }

    #[cfg(target_arch = "wasm32")]
    async fn load_environment_rgb_async(path: &str) -> Result<(u32, u32, Vec<u8>)> {
        use rust_raytracer::texture::Texture;
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};

        eprintln!(
            "🔍 [WASM] Attempting to load environment map from: {}",
            path
        );

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(path, &opts)
            .map_err(|e| anyhow!("Failed to create request: {:?}", e))?;

        let window = web_sys::window().ok_or_else(|| anyhow!("No window object"))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow!("Fetch failed: {:?}", e))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| anyhow!("Response is not a Response object"))?;

        if !resp.ok() {
            return Err(anyhow!(
                "HTTP error {}: {}",
                resp.status(),
                resp.status_text()
            ));
        }

        let array_buffer = JsFuture::from(
            resp.array_buffer()
                .map_err(|e| anyhow!("Failed to get array buffer: {:?}", e))?,
        )
        .await
        .map_err(|e| anyhow!("Failed to read array buffer: {:?}", e))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        eprintln!("✅ [WASM] Downloaded {} bytes", bytes.len());

        // Create a temporary file path for the EXR loader
        let mut texture = Texture::from_exr_bytes(&bytes)
            .map_err(|err| anyhow!("Failed to decode EXR: {}", err))?;

        eprintln!(
            "✅ [WASM] Loaded environment map: {}x{}",
            texture.width, texture.height
        );

        // Downsample if needed
        if texture.width > MAX_ENV_DIM || texture.height > MAX_ENV_DIM {
            let scale = (MAX_ENV_DIM as f32 / texture.width.max(texture.height) as f32).min(1.0);
            let new_width = (texture.width as f32 * scale) as u32;
            let new_height = (texture.height as f32 * scale) as u32;

            eprintln!(
                "⚠️ [WASM] Downsampling environment map from {}x{} to {}x{}",
                texture.width, texture.height, new_width, new_height
            );

            let mut new_data = Vec::with_capacity((new_width * new_height * 3) as usize);
            for y in 0..new_height {
                for x in 0..new_width {
                    let src_x = (x as f32 / new_width as f32) * texture.width as f32;
                    let src_y = (y as f32 / new_height as f32) * texture.height as f32;

                    let x0 = src_x.floor() as u32;
                    let y0 = src_y.floor() as u32;
                    let x1 = (x0 + 1).min(texture.width - 1);
                    let y1 = (y0 + 1).min(texture.height - 1);

                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;

                    let get_pixel = |px: u32, py: u32| -> (f32, f32, f32) {
                        let idx = ((py * texture.width + px) * 3) as usize;
                        (
                            texture.data[idx] as f32,
                            texture.data[idx + 1] as f32,
                            texture.data[idx + 2] as f32,
                        )
                    };

                    let p00 = get_pixel(x0, y0);
                    let p10 = get_pixel(x1, y0);
                    let p01 = get_pixel(x0, y1);
                    let p11 = get_pixel(x1, y1);

                    let interpolate = |v00: f32, v10: f32, v01: f32, v11: f32| -> u8 {
                        let top = v00 * (1.0 - fx) + v10 * fx;
                        let bottom = v01 * (1.0 - fx) + v11 * fx;
                        let result = top * (1.0 - fy) + bottom * fy;
                        result.clamp(0.0, 255.0) as u8
                    };

                    new_data.push(interpolate(p00.0, p10.0, p01.0, p11.0));
                    new_data.push(interpolate(p00.1, p10.1, p01.1, p11.1));
                    new_data.push(interpolate(p00.2, p10.2, p01.2, p11.2));
                }
            }

            texture.width = new_width;
            texture.height = new_height;
            texture.data = new_data;
        }

        eprintln!(
            "📊 [WASM] Final texture: {}x{}, {} bytes",
            texture.width,
            texture.height,
            texture.data.len()
        );

        Ok((texture.width, texture.height, texture.data))
    }

    fn create_environment_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgb_data: &[u8],
    ) -> wgpu::Texture {
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        for chunk in rgb_data.chunks(3) {
            rgba_data.push(chunk[0]);
            rgba_data.push(chunk[1]);
            rgba_data.push(chunk[2]);
            rgba_data.push(255);
        }

        eprintln!(
            "📤 Uploading texture to GPU: {}x{}, {} bytes",
            width,
            height,
            rgba_data.len()
        );
        eprintln!("   Sample pixels from RGBA data:");
        for i in 0..5.min(rgba_data.len() / 4) {
            let idx = i * 4;
            eprintln!(
                "     Pixel {}: R={}, G={}, B={}, A={}",
                i,
                rgba_data[idx],
                rgba_data[idx + 1],
                rgba_data[idx + 2],
                rgba_data[idx + 3]
            );
        }

        device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("environment-texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &rgba_data,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_raytracer_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        triangle_buffer: &wgpu::Buffer,
        sphere_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        bvh_buffer: &wgpu::Buffer,
        environment_texture_view: &wgpu::TextureView,
        environment_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raytracer-bind-group"),
            layout,
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(environment_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(environment_sampler),
                },
            ],
        })
    }

    fn rebuild_raytracer_bind_group(&mut self) {
        let environment_texture_view = self
            .environment_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bind_group = Self::create_raytracer_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.triangle_buffer,
            &self.sphere_buffer,
            &self.output_buffer,
            &self.bvh_buffer,
            &environment_texture_view,
            &self.environment_sampler,
        );
    }

    fn request_environment_change(&mut self, path: String) {
        self.pending_environment_change = Some(path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn reload_environment_map(&mut self, path: &str) -> Result<()> {
        match Self::load_environment_rgb(path) {
            Ok((width, height, rgb)) => {
                let texture = Self::create_environment_texture(
                    &self.device,
                    &self.queue,
                    width,
                    height,
                    &rgb,
                );
                self.environment_texture = texture;
                self.environment_path = path.to_string();
                self.rebuild_raytracer_bind_group();
                self.reset_accumulation_history();
                self.camera_moved = true;
                eprintln!("✅ Environment map updated to {}", path);
                Ok(())
            }
            Err(err) => {
                eprintln!("❌ Failed to reload environment map '{}': {}", path, err);
                Err(err)
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn reload_environment_map(&mut self, path: &str) -> Result<()> {
        let _ = path;
        eprintln!("⚠️ Environment map reloading is not supported on WASM");
        Ok(())
    }

    async fn new(window: Arc<winit::window::Window>) -> Result<Self> {
        let size = window.inner_size();
        // Some web platforms report zero-sized canvases until the first layout pass; clamp to 1 so surface configuration succeeds.
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Setup scene
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
            width as f64,
            height as f64,
        );

        let bunny_material = Material::new(Vec3::new(1.0, 1.0, 1.0), 0.5);

        #[cfg(not(target_arch = "wasm32"))]
        let (original_bunny, bunny) = {
            let original = Mesh::from_obj_file("data/bunny.obj", bunny_material)
                .map_err(|err| anyhow!("failed to load bunny OBJ: {err}"))?;
            let mut mesh = original.clone();
            mesh.rotate_y(180.0);
            mesh.transform(10.0, Vec3::new(0.0, -1.0, 4.0));
            eprintln!("Bunny positioned at Z=4.0, centered around Y=-1.0");
            (original, mesh)
        };

        #[cfg(target_arch = "wasm32")]
        let (original_bunny, bunny) = {
            let bunny_obj = include_str!("../../data/bunny.obj");
            log::info!(
                "🔍 WASM BUILD: Embedded OBJ data length: {} bytes",
                bunny_obj.len()
            );

            if bunny_obj.is_empty() {
                log::error!("❌ CRITICAL: Embedded OBJ data is EMPTY!");
                return Err(anyhow!("Embedded bunny.obj is empty"));
            }

            let original = Mesh::from_obj_str(bunny_obj, bunny_material).map_err(|err| {
                log::error!("❌ Failed to parse OBJ: {}", err);
                anyhow!("failed to parse embedded bunny OBJ: {err}")
            })?;

            log::info!(
                "✅ Parsed mesh: {} vertices, {} faces",
                original.vertices.len(),
                original.faces.len() / 3
            );

            if original.vertices.is_empty() {
                log::error!("❌ CRITICAL: Mesh has NO VERTICES after parsing!");
                return Err(anyhow!("Mesh has no vertices"));
            }

            let mut mesh = original.clone();
            mesh.rotate_y(180.0);
            mesh.transform(10.0, Vec3::new(0.0, -1.0, 4.0));
            log::info!("✅ Transformed bunny mesh for web viewer");
            log::info!(
                "📍 Bunny bounds: min=({:.2}, {:.2}, {:.2}), max=({:.2}, {:.2}, {:.2})",
                mesh.bounding_box.min.x,
                mesh.bounding_box.min.y,
                mesh.bounding_box.min.z,
                mesh.bounding_box.max.x,
                mesh.bounding_box.max.y,
                mesh.bounding_box.max.z
            );
            (original, mesh)
        };

        let sphere1 = Sphere::new(
            Vec3::new(0.0, -1001.0, 0.0),
            1000.0,
            Material::new(Vec3::new(1.0, 1.0, 1.0), 0.25),
        );

        // Green metallic sphere - shiny like chrome
        let sphere2 = Sphere::new(
            Vec3::new(2.0, 0.0, 5.0),
            1.0,
            Material::new_metallic(Vec3::new(0.8, 1.0, 0.8), 0.1), // Light green, low roughness = very shiny
        );
        // Blue diffuse sphere for comparison
        let sphere3 = Sphere::new(
            Vec3::new(-1.6, 0.0, 5.0),
            1.0,
            Material::new(Vec3::new(0.0, 0.0, 1.0), 0.5),
        );
        eprintln!("Green sphere at (2.0, 0.0, 5.0), Blue sphere at (-1.6, 0.0, 5.0)");
        let spheres = [sphere1, sphere2, sphere3];

        let (triangles, mut bvh_nodes) = mesh_to_gpu_data(&bunny);
        let triangle_count = triangles.len() as u32;
        log::info!(
            "GPU data: {} triangles, {} BVH nodes",
            triangle_count,
            bvh_nodes.len()
        );
        if bvh_nodes.is_empty() {
            bvh_nodes.push(GpuBvhNode::zeroed());
        }
        let bvh_node_count = if triangle_count == 0 {
            log::warn!("No triangles in mesh!");
            0
        } else {
            bvh_nodes.len() as u32
        };
        let triangles_storage: Cow<[GpuTriangle]> = if triangles.is_empty() {
            log::warn!("Using zeroed triangle storage");
            Cow::Owned(vec![GpuTriangle::zeroed()])
        } else {
            Cow::Owned(triangles)
        };
        let bvh_storage: Cow<[GpuBvhNode]> = Cow::Owned(bvh_nodes);
        log::info!(
            "Scene uniform will have: triangle_count={}, bvh_node_count={}",
            triangle_count,
            bvh_node_count
        );

        let gpu_spheres: Vec<GpuSphere> = spheres.iter().map(sphere_to_gpu).collect();
        let sphere_count = gpu_spheres.len() as u32;
        let spheres_storage: Cow<[GpuSphere]> = if gpu_spheres.is_empty() {
            Cow::Owned(vec![GpuSphere::zeroed()])
        } else {
            Cow::Owned(gpu_spheres)
        };

        let (lower_left_corner, horizontal, vertical) = camera_frame(&camera);

        let samples_per_pixel = 1u32; // 1 sample for real-time interactivity
        let max_bounces = 2u32; // Reduced bounces for speed

        // Calculate padded width for buffer alignment
        let bytes_per_pixel = 16u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let padded_width = padded_bytes_per_row / bytes_per_pixel;

        let scene_uniform = SceneUniform {
            resolution: [width, height, triangle_count, sphere_count],
            camera_position: vec3_to_array(camera.position, 0.0),
            lower_left_corner: vec3_to_array(lower_left_corner, 0.0),
            horizontal: vec3_to_array(horizontal, 0.0),
            vertical: vec3_to_array(vertical, 0.0),
            environment_strength: [1.0, 0.0, 0.0, 0.0], // x: environment strength, yzw: padding
            mesh_color: vec3_to_array(bunny.material.color, 1.0),
            render_config: [samples_per_pixel, max_bounces, padded_width, 0],
            accel_info: [bvh_node_count, 0, 0, 0],
        };

        // Create WGPU instance
        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if cfg!(target_arch = "wasm32") {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find GPU adapter")?;

        #[cfg(target_arch = "wasm32")]
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-raytracer-device"),
                    required_features: wgpu::Features::empty(),
                    // Use the adapter-provided limits to avoid exceeding WebGL capabilities
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .context("failed to create device")?;

        #[cfg(not(target_arch = "wasm32"))]
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-raytracer-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .context("failed to create device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create buffers
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

        // Calculate buffer size with row padding (256-byte alignment for copy operations)
        let bytes_per_pixel = 16u32; // 4 f32s per pixel
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let output_buffer_size = (padded_bytes_per_row * height) as u64;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create accumulation buffer for temporal accumulation
        let accumulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation-buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create texture for display
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create accumulation texture for temporal accumulation
        let accumulation_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("accumulation-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Discover available environment maps and select initial path
        let available_env_maps = Self::discover_environment_maps();
        let selected_environment = available_env_maps
            .iter()
            .position(|p| p == DEFAULT_ENV_MAP)
            .unwrap_or(0);
        let active_environment = available_env_maps
            .get(selected_environment)
            .cloned()
            .unwrap_or_else(|| DEFAULT_ENV_MAP.to_string());

        // Load environment map (HDR texture)
        #[cfg(not(target_arch = "wasm32"))]
        let env_texture_data = match Self::load_environment_rgb(&active_environment) {
            Ok(data) => Some(data),
            Err(error) => {
                eprintln!(
                    "❌ Failed to load environment map '{}': {}, using default",
                    active_environment, error
                );
                None
            }
        };

        #[cfg(target_arch = "wasm32")]
        let env_texture_data: Option<(u32, u32, Vec<u8>)> = {
            match Self::load_environment_rgb_async(&active_environment).await {
                Ok(data) => {
                    eprintln!("✅ [WASM] Successfully loaded environment map");
                    Some(data)
                }
                Err(error) => {
                    eprintln!(
                        "❌ [WASM] Failed to load environment map '{}': {}, using default",
                        active_environment, error
                    );
                    None
                }
            }
        };

        // Create environment texture (or default test pattern if loading failed)
        let (env_width, env_height, env_data) = env_texture_data.unwrap_or_else(|| {
            eprintln!("⚠️ Using default test pattern environment texture");
            // Create a simple 4x4 test pattern: red, green, blue, yellow gradient
            let mut data = Vec::new();
            for y in 0..4 {
                for x in 0..4 {
                    data.push((x * 255 / 3) as u8); // Red gradient
                    data.push((y * 255 / 3) as u8); // Green gradient
                    data.push(128); // Blue constant
                }
            }
            (4u32, 4u32, data)
        });

        let environment_texture =
            Self::create_environment_texture(&device, &queue, env_width, env_height, &env_data);

        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create compute pipeline for raytracing
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let environment_texture_view =
            environment_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = State::create_raytracer_bind_group(
            &device,
            &compute_bind_group_layout,
            &uniform_buffer,
            &triangle_buffer,
            &sphere_buffer,
            &output_buffer,
            &bvh_buffer,
            &environment_texture_view,
            &environment_sampler,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raytracer-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/raytracer.wgsl"
            ))),
        });

        let shader_normals = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raytracer-normals-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/raytracer_normals.wgsl"
            ))),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raytracer-pipeline-layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let compute_pipeline_normals =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("raytracer-normals-pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader_normals,
                entry_point: "main",
            });

        // Create accumulation pipeline
        let accumulation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accumulation-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/accumulate.wgsl"
            ))),
        });

        let accumulation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("accumulation-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let accumulation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation-uniform"),
            size: 16, // vec4<u32>: width, height, accumulated_frames, padded_width
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let accumulation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulation-bind-group"),
            layout: &accumulation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accumulation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: accumulation_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let accumulation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("accumulation-pipeline-layout"),
                bind_group_layouts: &[&accumulation_bind_group_layout],
                push_constant_ranges: &[],
            });

        let accumulation_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accumulation-pipeline"),
                layout: Some(&accumulation_pipeline_layout),
                module: &accumulation_shader,
                entry_point: "main",
            });

        // Create display pipeline
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/display.wgsl"
            ))),
        });

        let display_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("display-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display-bind-group"),
            layout: &display_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("display-pipeline-layout"),
                bind_group_layouts: &[&display_bind_group_layout],
                push_constant_ranges: &[],
            });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display-pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // Initialize UI state
        let ui_state = UIState {
            environment_maps: available_env_maps.clone(),
            selected_environment,
            environment_strength: 1.0,
            samples_per_pixel,
            max_bounces,
            mesh_color: [
                bunny.material.color.x as f32,
                bunny.material.color.y as f32,
                bunny.material.color.z as f32,
            ],
            mesh_position: [0.0, -1.0, 4.0],
            mesh_roughness: 0.5,
            mesh_metallic: 0.0,
            objects: vec![
                SceneObject {
                    name: "Ground Sphere".to_string(),
                    position: [0.0, -1001.0, 0.0],
                    radius: 1000.0,
                    color: [1.0, 1.0, 1.0],
                    roughness: 0.25,
                    metallic: 0.0, // Diffuse ground
                    enabled: true,
                },
                SceneObject {
                    name: "Green Metallic Sphere".to_string(),
                    position: [2.0, 0.0, 5.0],
                    radius: 1.0,
                    color: [0.8, 1.0, 0.8],
                    roughness: 0.1,
                    metallic: 1.0, // Shiny metal
                    enabled: true,
                },
                SceneObject {
                    name: "Blue Diffuse Sphere".to_string(),
                    position: [-1.6, 0.0, 5.0],
                    radius: 1.0,
                    color: [0.0, 0.0, 1.0],
                    roughness: 0.5,
                    metallic: 0.0, // Diffuse
                    enabled: true,
                },
            ],
            show_add_object_dialog: false,
            new_object_position: [0.0, 0.0, 5.0],
            new_object_radius: 1.0,
            new_object_color: [1.0, 1.0, 1.0],
            show_file_upload_dialog: false,
            show_ui: true,
        };

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            uniform_buffer,
            output_buffer,
            bind_group,
            bind_group_layout: compute_bind_group_layout,
            triangle_buffer,
            bvh_buffer,
            sphere_buffer,
            bunny_mesh: bunny,
            original_mesh: original_bunny,
            compute_pipeline,
            compute_pipeline_normals,
            accumulation_pipeline,
            accumulation_bind_group,
            accumulation_uniform_buffer,
            display_pipeline,
            display_bind_group,
            display_bind_group_layout,
            output_texture,
            sampler,
            environment_texture,
            environment_sampler,
            environment_path: active_environment.clone(),
            pending_environment_change: None,
            camera_position: camera.position,
            camera_yaw: 180.0, // Start looking at +Z (toward the scene)
            camera_pitch: 0.0,
            camera,
            scene_uniform,
            last_mouse_pos: None,
            mouse_pressed: false,
            use_raytracing: true, // Start with raytracing mode
            accumulation_buffer,
            accumulation_texture,
            accumulated_frames: 0,
            camera_moved: false,
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            egui_renderer,
            egui_state,
            egui_ctx,
            ui_state,
        };

        // Force initial camera update to sync everything
        state.update_camera(0.0);

        eprintln!(
            "Camera initialized at pos: {:?}, yaw: {}, pitch: {}",
            state.camera_position, state.camera_yaw, state.camera_pitch
        );

        Ok(state)
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate output buffer and texture with new size
            // Note: Buffer size must account for row padding (256-byte alignment)
            let bytes_per_pixel = 16u32; // 4 f32s per pixel
            let unpadded_bytes_per_row = new_size.width * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
            let buffer_size = (padded_bytes_per_row * new_size.height) as u64;

            self.output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("output-buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.accumulation_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("accumulation-buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("output-texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            self.accumulation_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("accumulation-texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // Reset accumulation on resize
            self.accumulated_frames = 0;
            self.camera_moved = true;

            // Recreate bind groups with new buffers/textures
            let output_view = self
                .output_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let environment_texture_view = self
                .environment_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raytracer-bind-group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.triangle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.sphere_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.bvh_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&environment_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.environment_sampler),
                    },
                ],
            });

            self.display_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("display-bind-group"),
                layout: &self.display_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            // Recreate accumulation bind group with new buffers
            self.accumulation_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("accumulation-bind-group"),
                    layout: &self.accumulation_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.accumulation_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.accumulation_uniform_buffer.as_entire_binding(),
                        },
                    ],
                });

            // Update camera aspect ratio with new dimensions
            self.camera.width = new_size.width as f64;
            self.camera.height = new_size.height as f64;

            // Recalculate camera frame with new aspect ratio
            let (lower_left_corner, horizontal, vertical) = camera_frame(&self.camera);

            // Update scene uniform with new dimensions and camera frame
            self.scene_uniform.resolution[0] = new_size.width;
            self.scene_uniform.resolution[1] = new_size.height;
            self.scene_uniform.lower_left_corner = vec3_to_array(lower_left_corner, 0.0);
            self.scene_uniform.horizontal = vec3_to_array(horizontal, 0.0);
            self.scene_uniform.vertical = vec3_to_array(vertical, 0.0);
            // Update padded width for buffer alignment
            let padded_width_pixels = padded_bytes_per_row / bytes_per_pixel;
            self.scene_uniform.render_config[2] = padded_width_pixels;
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::bytes_of(&self.scene_uniform),
            );
        }
    }

    fn update_camera(&mut self, _delta_time: f64) {
        // Calculate camera direction from yaw and pitch
        // Yaw: 0° = +Z, 90° = +X, 180° = -Z, 270° = -X
        let yaw_rad = self.camera_yaw.to_radians();
        let pitch_rad = self.camera_pitch.to_radians();

        let forward = Vec3::new(yaw_rad.sin(), -pitch_rad.sin(), yaw_rad.cos()).normalize();

        // Update camera
        self.camera.position = self.camera_position;
        self.camera.look_direction = forward;

        // Debug: Print camera info on first update (when delta_time is 0.0)
        if _delta_time == 0.0 {
            eprintln!("Camera look direction: {:?}", forward);
        }

        // Recalculate camera frame
        let (lower_left_corner, horizontal, vertical) = camera_frame(&self.camera);

        // Update scene uniform
        self.scene_uniform.camera_position = vec3_to_array(self.camera.position, 0.0);
        self.scene_uniform.lower_left_corner = vec3_to_array(lower_left_corner, 0.0);
        self.scene_uniform.horizontal = vec3_to_array(horizontal, 0.0);
        self.scene_uniform.vertical = vec3_to_array(vertical, 0.0);

        // Upload to GPU
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.scene_uniform),
        );
    }

    fn handle_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.2;
        self.camera_yaw -= delta_x * sensitivity; // Fixed: inverted
        self.camera_pitch += delta_y * sensitivity; // Fixed: inverted

        // Clamp pitch to avoid gimbal lock
        self.camera_pitch = self.camera_pitch.clamp(-89.0, 89.0);

        // Mark camera as moved to reset accumulation
        self.camera_moved = true;

        eprintln!(
            "Mouse delta: ({:.2}, {:.2}) -> Yaw: {:.1}°, Pitch: {:.1}°",
            delta_x, delta_y, self.camera_yaw, self.camera_pitch
        );
    }

    fn update_scene_from_ui(&mut self) {
        // Update environment strength
        self.scene_uniform.environment_strength = [
            self.ui_state.environment_strength,
            0.0, // Padding
            0.0, // Padding
            0.0, // Padding
        ];
        self.scene_uniform.mesh_color = [
            self.ui_state.mesh_color[0],
            self.ui_state.mesh_color[1],
            self.ui_state.mesh_color[2],
            1.0,
        ];

        // Check if mesh properties have changed
        let mesh_changed = self.bunny_mesh.material.color.x != self.ui_state.mesh_color[0] as f64
            || self.bunny_mesh.material.color.y != self.ui_state.mesh_color[1] as f64
            || self.bunny_mesh.material.color.z != self.ui_state.mesh_color[2] as f64
            || self.bunny_mesh.material.roughness != self.ui_state.mesh_roughness as f64
            || self.bunny_mesh.material.metallic != self.ui_state.mesh_metallic as f64;

        // Only rebuild mesh if properties changed
        if mesh_changed {
            // Rebuild mesh with new transformations
            let mut mesh = self.original_mesh.clone();

            // Update mesh material
            mesh.material.color = Vec3::new(
                self.ui_state.mesh_color[0] as f64,
                self.ui_state.mesh_color[1] as f64,
                self.ui_state.mesh_color[2] as f64,
            );
            mesh.material.roughness = self.ui_state.mesh_roughness as f64;
            mesh.material.metallic = self.ui_state.mesh_metallic as f64;

            // Apply transformations (fixed scale and rotation from initialization)
            mesh.rotate_y(180.0);
            mesh.transform(
                10.0,
                Vec3::new(
                    self.ui_state.mesh_position[0] as f64,
                    self.ui_state.mesh_position[1] as f64,
                    self.ui_state.mesh_position[2] as f64,
                ),
            );

            // Rebuild triangle and BVH buffers
            let (triangles, bvh_nodes) = mesh_to_gpu_data(&mesh);

            self.triangle_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("triangle-buffer"),
                        contents: bytemuck::cast_slice(&triangles),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

            self.bvh_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bvh-buffer"),
                    contents: bytemuck::cast_slice(&bvh_nodes),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // Update triangle and BVH counts
            self.scene_uniform.resolution[2] = triangles.len() as u32; // triangle count
            self.scene_uniform.accel_info[0] = bvh_nodes.len() as u32; // BVH node count
            self.bunny_mesh = mesh;
        }

        self.scene_uniform.render_config[0] = self.ui_state.samples_per_pixel;
        self.scene_uniform.render_config[1] = self.ui_state.max_bounces;
        self.scene_uniform.render_config[3] = self.accumulated_frames; // Frame seed for temporal variation

        // Rebuild sphere buffer
        let spheres: Vec<GpuSphere> = self
            .ui_state
            .objects
            .iter()
            .filter(|obj| obj.enabled)
            .map(|obj| {
                let material_type = if obj.metallic > 0.5 { 1.0 } else { 0.0 }; // 0.0 = diffuse, 1.0 = metallic
                GpuSphere {
                    center_radius: [
                        obj.position[0],
                        obj.position[1],
                        obj.position[2],
                        obj.radius,
                    ],
                    color: [obj.color[0], obj.color[1], obj.color[2], 1.0],
                    material: [obj.roughness, obj.metallic, material_type, 0.0],
                }
            })
            .collect();

        self.scene_uniform.resolution[3] = spheres.len() as u32;

        // Recreate sphere buffer
        let spheres_data: Cow<[GpuSphere]> = if spheres.is_empty() {
            Cow::Owned(vec![GpuSphere::zeroed()])
        } else {
            Cow::Owned(spheres)
        };

        self.sphere_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sphere-buffer"),
                contents: bytemuck::cast_slice(spheres_data.as_ref()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Recreate bind group (needed because sphere buffer always changes)
        self.rebuild_raytracer_bind_group();

        // Upload uniform buffer
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.scene_uniform),
        );

        // Reset accumulation when scene changes
        self.accumulated_frames = 0;
        self.camera_moved = true;
    }

    fn padded_bytes_per_row(&self) -> u32 {
        let bytes_per_pixel = 16u32; // 4 f32 values written per pixel
        let unpadded = self.size.width.saturating_mul(bytes_per_pixel);
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        unpadded.div_ceil(align) * align
    }

    fn reset_accumulation_history(&mut self) {
        let padded_bytes_per_row = self.padded_bytes_per_row();
        if padded_bytes_per_row == 0 || self.size.height == 0 {
            self.accumulated_frames = 0;
            return;
        }

        let buffer_size = (padded_bytes_per_row as u64 * self.size.height as u64) as usize;
        let zero_data = vec![0u8; buffer_size];
        self.queue
            .write_buffer(&self.accumulation_buffer, 0, &zero_data);
        self.accumulated_frames = 0;
    }

    fn build_ui(&mut self) -> egui::FullOutput {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let current_fps = self.current_fps;
        let use_raytracing = self.use_raytracing;
        let current_environment_path = self.environment_path.clone();
        let accumulated_frames = self.accumulated_frames;
        let mut needs_update = false;
        let mut requested_environment: Option<String> = None;

        let full_output = {
            let ui_state = &mut self.ui_state;
            self.egui_ctx.run(raw_input, |ctx| {
            if ui_state.show_ui {
                egui::SidePanel::right("scene_editor")
                    .default_width(320.0)
                    .resizable(true)
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.heading("Scene Editor");
                            ui.separator();

                            ui.collapsing("Environment Lighting", |ui| {
                                if ui_state.environment_maps.is_empty() {
                                    ui.label("No HDR maps found in ./data (looking for .exr).");
                                } else {
                                    if ui_state.selected_environment
                                        >= ui_state.environment_maps.len()
                                    {
                                        ui_state.selected_environment = 0;
                                    }
                                    let current_label = Path::new(current_environment_path.as_str())
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or(current_environment_path.as_str());
                                    let mut env_changed = false;
                                    egui::ComboBox::from_label("HDR Map")
                                        .selected_text(current_label)
                                        .show_ui(ui, |ui| {
                                            for (idx, path) in ui_state.environment_maps.iter().enumerate() {
                                                let label = Path::new(path)
                                                    .file_name()
                                                    .and_then(|name| name.to_str())
                                                    .unwrap_or(path);
                                                if ui
                                                    .selectable_value(
                                                        &mut ui_state.selected_environment,
                                                        idx,
                                                        label,
                                                    )
                                                    .changed()
                                                {
                                                    env_changed = true;
                                                }
                                            }
                                        });
                                    if env_changed {
                                        requested_environment =
                                            ui_state.environment_maps.get(ui_state.selected_environment).cloned();
                                    }
                                    if ui.button("Reload Selected").clicked() {
                                        requested_environment =
                                            ui_state.environment_maps.get(ui_state.selected_environment).cloned();
                                    }
                                }

                                ui.add_space(10.0);

                                ui.label("Environment Strength:");
                                if ui
                                    .add(egui::Slider::new(&mut ui_state.environment_strength, 0.0..=5.0))
                                    .changed()
                                {
                                    needs_update = true;
                                }

                                ui.add_space(10.0);
                                ui.separator();

                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    ui.label("📁 Upload Custom HDR Map:");
                                    ui.label("Place .exr files in the ./data folder");
                                    ui.label("and restart the application.");
                                }

                                #[cfg(target_arch = "wasm32")]
                                {
                                    ui.label("📁 Add Custom HDR Maps:");
                                    if ui.button("How to upload...").clicked() {
                                        ui_state.show_file_upload_dialog = true;
                                    }
                                }

                                ui.add_space(5.0);
                                let active_display =
                                    format!("{}", Path::new(current_environment_path.as_str()).display());
                                ui.label("Current map:");
                                ui.monospace(active_display);
                                ui.label(
                                    "HDR data is tone-mapped when loaded. Re-select to apply file edits.",
                                );
                            });

                            ui.add_space(10.0);
                            ui.collapsing("Render Settings", |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Samples/pixel:");
                                    if ui
                                        .add(egui::Slider::new(
                                            &mut ui_state.samples_per_pixel,
                                            1..=16,
                                        ))
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Max bounces:");
                                    if ui.add(egui::Slider::new(&mut ui_state.max_bounces, 1..=5)).changed() {
                                        needs_update = true;
                                    }
                                });
                            });

                            ui.add_space(10.0);
                            ui.collapsing("Mesh", |ui| {
                                ui.label("Bunny Color:");
                                if ui.color_edit_button_rgb(&mut ui_state.mesh_color).changed() {
                                    needs_update = true;
                                }

                                ui.add_space(5.0);
                                ui.horizontal(|ui| {
                                    ui.label("Position:");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut ui_state.mesh_position[0])
                                                .speed(0.1)
                                                .prefix("X:"),
                                        )
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut ui_state.mesh_position[1])
                                                .speed(0.1)
                                                .prefix("Y:"),
                                        )
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut ui_state.mesh_position[2])
                                                .speed(0.1)
                                                .prefix("Z:"),
                                        )
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Roughness:");
                                    if ui
                                        .add(egui::Slider::new(
                                            &mut ui_state.mesh_roughness,
                                            0.0..=1.0,
                                        ))
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Metallic:");
                                    if ui
                                        .add(egui::Slider::new(
                                            &mut ui_state.mesh_metallic,
                                            0.0..=1.0,
                                        ))
                                        .changed()
                                    {
                                        needs_update = true;
                                    }
                                });
                            });

                            ui.add_space(10.0);
                            ui.collapsing("Objects", |ui| {
                                let mut to_remove = None;
                                for (idx, obj) in ui_state.objects.iter_mut().enumerate() {
                                    ui.group(|ui| {
                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut obj.enabled, "").changed() {
                                                needs_update = true;
                                            }
                                            ui.label(&obj.name);
                                            if ui.button("🗑").clicked() {
                                                to_remove = Some(idx);
                                            }
                                        });

                                        if obj.enabled {
                                            ui.horizontal(|ui| {
                                                ui.label("Pos:");
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut obj.position[0])
                                                            .speed(0.1)
                                                            .prefix("X:"),
                                                    )
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut obj.position[1])
                                                            .speed(0.1)
                                                            .prefix("Y:"),
                                                    )
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut obj.position[2])
                                                            .speed(0.1)
                                                            .prefix("Z:"),
                                                    )
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("Radius:");
                                                if ui
                                                    .add(egui::Slider::new(
                                                        &mut obj.radius,
                                                        0.1..=5.0,
                                                    ))
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("Roughness:");
                                                if ui
                                                    .add(egui::Slider::new(
                                                        &mut obj.roughness,
                                                        0.0..=1.0,
                                                    ))
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("Metallic:");
                                                if ui
                                                    .add(egui::Slider::new(
                                                        &mut obj.metallic,
                                                        0.0..=1.0,
                                                    ))
                                                    .changed()
                                                {
                                                    needs_update = true;
                                                }
                                            });
                                            if ui.color_edit_button_rgb(&mut obj.color).changed() {
                                                needs_update = true;
                                            }
                                        }
                                    });
                                }

                                if let Some(idx) = to_remove {
                                    ui_state.objects.remove(idx);
                                    needs_update = true;
                                }

                                ui.add_space(10.0);
                                if ui.button("+ Add Sphere").clicked() {
                                    ui_state.show_add_object_dialog = true;
                                }
                            });
                        });
                    });
            }

            // Add object dialog as a modal window
            if ui_state.show_add_object_dialog {
                egui::Window::new("Add Sphere")
                    .collapsible(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Position:");
                            ui.add(
                                egui::DragValue::new(&mut ui_state.new_object_position[0])
                                    .speed(0.1),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ui_state.new_object_position[1])
                                    .speed(0.1),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ui_state.new_object_position[2])
                                    .speed(0.1),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Radius:");
                            ui.add(egui::Slider::new(
                                &mut ui_state.new_object_radius,
                                0.1..=5.0,
                            ));
                        });
                        ui.label("Color:");
                        ui.color_edit_button_rgb(&mut ui_state.new_object_color);

                        ui.horizontal(|ui| {
                            if ui.button("Add").clicked() {
                                ui_state.objects.push(SceneObject {
                                    name: format!("Sphere {}", ui_state.objects.len() + 1),
                                    position: ui_state.new_object_position,
                                    radius: ui_state.new_object_radius,
                                    color: ui_state.new_object_color,
                                    roughness: 0.5,
                                    metallic: 0.0,
                                    enabled: true,
                                });
                                ui_state.show_add_object_dialog = false;
                                needs_update = true;
                            }
                            if ui.button("Cancel").clicked() {
                                ui_state.show_add_object_dialog = false;
                            }
                        });
                    });
            }

            // File upload dialog (WASM only)
            #[cfg(target_arch = "wasm32")]
            if ui_state.show_file_upload_dialog {
                egui::Window::new("Upload HDR Environment Map")
                    .collapsible(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.heading("How to add custom HDR maps:");
                        ui.add_space(10.0);

                        ui.label("To use your own HDR environment maps:");
                        ui.add_space(5.0);

                        ui.label("1. Place your .exr files in the /data folder");
                        ui.label("2. Rebuild the WASM app with: trunk build --release");
                        ui.label("3. The new maps will appear in the dropdown");

                        ui.add_space(10.0);
                        ui.separator();
                        ui.add_space(10.0);

                        ui.label("💡 Tip: You can find free HDR maps at:");
                        if ui.link("polyhaven.com/hdris").clicked() {
                            if let Some(window) = web_sys::window() {
                                let _ = window.open_with_url("https://polyhaven.com/hdris");
                            }
                        }

                        ui.add_space(10.0);
                        ui.label("Supported format: OpenEXR (.exr)");
                        ui.label("HDR maps are tone-mapped when loaded.");

                        ui.add_space(10.0);
                        if ui.button("Close").clicked() {
                            ui_state.show_file_upload_dialog = false;
                        }
                    });
            }

            // FPS overlay
            egui::Area::new(egui::Id::new("fps"))
                .fixed_pos(egui::pos2(10.0, 10.0))
                .show(ctx, |ui| {
                    ui.label(format!("FPS: {:.1}", current_fps));
                    ui.label(format!(
                        "Mode: {}",
                        if use_raytracing {
                            "Raytracing"
                        } else {
                            "Normals"
                        }
                    ));
                    ui.label(format!("Accumulated Frames: {}", accumulated_frames));
                    ui.label("Press TAB to toggle UI");
                    ui.label("Press SPACE to toggle render mode");
                });
            })
        };

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output.clone());

        if let Some(path) = requested_environment {
            self.request_environment_change(path);
        }

        // Update scene if needed
        if needs_update {
            self.update_scene_from_ui();
        }

        full_output
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if let Some(path) = self.pending_environment_change.take() {
            self.reload_environment_map(&path).unwrap_or_else(|err| {
                eprintln!("❌ Failed to apply environment '{}': {}", path, err)
            });
        }

        // Update FPS
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.fps_timer = Instant::now();

            let mode_name = if self.use_raytracing {
                "Raytracing"
            } else {
                "Normals"
            };
            self.window.set_title(&format!(
                "GPU Raytracer - {} - {:.1} FPS",
                mode_name, self.current_fps
            ));
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-encoder"),
            });

        // Update frame seed for temporal variation in random sampling
        self.scene_uniform.render_config[3] = self.accumulated_frames;
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.scene_uniform),
        );

        // Run compute shader (choose pipeline based on mode)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raytracer-pass"),
                timestamp_writes: None,
            });

            // Select pipeline based on render mode
            let pipeline = if self.use_raytracing {
                &self.compute_pipeline
            } else {
                &self.compute_pipeline_normals
            };

            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            let workgroup_size = [8u32, 8u32];
            let dispatch_x = (self.size.width + workgroup_size[0] - 1).div_ceil(workgroup_size[0]);
            let dispatch_y = (self.size.height + workgroup_size[1] - 1).div_ceil(workgroup_size[1]);
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Temporal accumulation logic
        let bytes_per_pixel = 16u32; // 4 f32s per pixel
        let padded_bytes_per_row = self.padded_bytes_per_row();

        if self.camera_moved {
            // Clear accumulation buffer when camera moves or scene changes
            self.reset_accumulation_history();
            self.camera_moved = false;
        }

        // Run accumulation compute shader to blend current frame with history
        let padded_width_pixels = padded_bytes_per_row / bytes_per_pixel;
        let accumulation_uniform_data: [u32; 4] = [
            self.size.width,
            self.size.height,
            self.accumulated_frames,
            padded_width_pixels,
        ];
        self.queue.write_buffer(
            &self.accumulation_uniform_buffer,
            0,
            bytemuck::cast_slice(&accumulation_uniform_data),
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("accumulation-pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.accumulation_pipeline);
            cpass.set_bind_group(0, &self.accumulation_bind_group, &[]);
            let workgroup_size = [8u32, 8u32];
            let dispatch_x = (self.size.width + workgroup_size[0] - 1).div_ceil(workgroup_size[0]);
            let dispatch_y = (self.size.height + workgroup_size[1] - 1).div_ceil(workgroup_size[1]);
            cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.accumulated_frames += 1;

        // Copy accumulated buffer to texture for display
        // Note: bytes_per_row must be a multiple of 256 (COPY_BYTES_PER_ROW_ALIGNMENT)
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.accumulation_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
        );

        // Build and prepare UI
        let full_output = self.build_ui();
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        // Update egui buffers
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Display on screen
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.display_pipeline);
            rpass.set_bind_group(0, &self.display_bind_group, &[]);
            rpass.draw(0..6, 0..1);

            // Render egui on top
            self.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_descriptor);
        }

        // Free egui textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// Platform-specific entry points
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    pollster::block_on(run())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // On WASM, the real entry point is the start() function
    // This main() is just to satisfy the binary target requirement
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    // Set up panic hook for better error messages in browser console
    console_error_panic_hook::set_once();

    // Initialize console logging
    console_log::init_with_level(log::Level::Info).expect("Could not initialize logger");

    log::info!("WASM module loaded, starting application...");

    // Spawn the async runtime
    wasm_bindgen_futures::spawn_local(async {
        log::info!("Async runtime started");
        match run().await {
            Ok(_) => log::info!("Application exited normally"),
            Err(e) => {
                log::error!("Application failed: {:?}", e);
                // Also try to show in a more visible way
                web_sys::window().and_then(|w| {
                    w.alert_with_message(&format!("Failed to start: {:?}", e))
                        .ok()
                });
            }
        }
    });
}

async fn run() -> Result<()> {
    #[cfg(target_arch = "wasm32")]
    log::info!("Creating event loop...");

    let event_loop = EventLoop::new().context("Failed to create event loop")?;

    #[cfg(target_arch = "wasm32")]
    log::info!("Event loop created successfully");

    #[cfg(target_arch = "wasm32")]
    log::info!("Creating window...");

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("GPU Raytracer - Drag mouse to rotate camera")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .with_visible(true)
            .build(&event_loop)?,
    );

    // Attach canvas to DOM for web
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;

        if let Some(win) = web_sys::window() {
            if let Some(doc) = win.document() {
                if let Some(body) = doc.body() {
                    if let Some(canvas) = window.canvas() {
                        let canvas_elem: web_sys::HtmlCanvasElement = canvas;

                        // Style the canvas to fill the viewport
                        canvas_elem
                            .set_attribute("style", "width: 100%; height: 100%; display: block;")
                            .ok();

                        let device_pixel_ratio = win.device_pixel_ratio();
                        let width = win
                            .inner_width()
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(800.0);
                        let height = win
                            .inner_height()
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(600.0);

                        // Set canvas pixel size to match the viewport
                        canvas_elem.set_width((width * device_pixel_ratio).round() as u32);
                        canvas_elem.set_height((height * device_pixel_ratio).round() as u32);

                        let canvas_node: web_sys::Element = canvas_elem.clone().into();
                        body.append_child(&canvas_node).ok();

                        // Update winit's logical size so the camera aspect matches the viewport
                        let _ =
                            window.request_inner_size(winit::dpi::LogicalSize::new(width, height));
                    }
                }
            }
        }

        log::info!("Canvas attached to document body and resized to viewport");
    }

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("Window created, initializing GPU...");

    #[cfg(target_arch = "wasm32")]
    log::info!("Window created, initializing GPU...");

    #[cfg(target_arch = "wasm32")]
    log::info!("Calling State::new()...");

    let mut state = State::new(window.clone())
        .await
        .context("Failed to create State")?;

    #[cfg(target_arch = "wasm32")]
    log::info!("State created successfully");

    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("GPU initialized, starting event loop...");
        eprintln!("Controls:");
        eprintln!("  - Click and drag to rotate camera");
        eprintln!("  - Press SPACE to toggle between raytracing and normals view");
    }

    #[cfg(target_arch = "wasm32")]
    {
        log::info!("GPU initialized, starting event loop...");
        log::info!("Controls:");
        log::info!("  - Click and drag to rotate camera");
        log::info!("  - Press SPACE to toggle between raytracing and normals view");
    }

    state.window.focus_window();
    state.window.request_redraw();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        if let Some(canvas) = state.window.canvas() {
            let width = canvas.width().max(1);
            let height = canvas.height().max(1);
            state.resize(winit::dpi::PhysicalSize::new(width, height));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    let mut last_update = Instant::now();

    #[cfg(target_arch = "wasm32")]
    let mut last_update = web_time::Instant::now();

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                // Let egui handle the event first
                let event_response = state.egui_state.on_window_event(&state.window, event);

                // Skip input handling if egui consumed the event
                if event_response.consumed {
                    state.window.request_redraw();
                    return;
                }

                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::KeyboardInput {
                        event: key_event, ..
                    } =>
                    {
                        #[allow(clippy::collapsible_if)]
                        if key_event.state == ElementState::Pressed {
                            if let winit::keyboard::PhysicalKey::Code(keycode) =
                                key_event.physical_key
                            {
                                match keycode {
                                    winit::keyboard::KeyCode::Space => {
                                        state.use_raytracing = !state.use_raytracing;
                                        state.accumulated_frames = 0;
                                        state.scene_uniform.render_config[3] = 0;
                                        state.camera_moved = true;
                                        let mode = if state.use_raytracing {
                                            "Raytracing"
                                        } else {
                                            "Normals"
                                        };
                                        #[cfg(not(target_arch = "wasm32"))]
                                        eprintln!("Switched to {} mode", mode);
                                        #[cfg(target_arch = "wasm32")]
                                        log::info!("Switched to {} mode", mode);
                                    }
                                    winit::keyboard::KeyCode::Tab => {
                                        state.ui_state.show_ui = !state.ui_state.show_ui;
                                        let ui_status = if state.ui_state.show_ui {
                                            "shown"
                                        } else {
                                            "hidden"
                                        };
                                        #[cfg(not(target_arch = "wasm32"))]
                                        eprintln!("UI: {}", ui_status);
                                        #[cfg(target_arch = "wasm32")]
                                        log::info!("UI: {}", ui_status);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        state: button_state,
                        button,
                        ..
                    } => {
                        if *button == winit::event::MouseButton::Left {
                            state.mouse_pressed = *button_state == ElementState::Pressed;
                            #[cfg(not(target_arch = "wasm32"))]
                            eprintln!(
                                "Mouse button: {}",
                                if state.mouse_pressed {
                                    "PRESSED"
                                } else {
                                    "RELEASED"
                                }
                            );
                            if !state.mouse_pressed {
                                state.last_mouse_pos = None;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let current_pos = (position.x, position.y);

                        #[allow(clippy::collapsible_if)]
                        if state.mouse_pressed {
                            if let Some(last_pos) = state.last_mouse_pos {
                                let delta_x = current_pos.0 - last_pos.0;
                                let delta_y = current_pos.1 - last_pos.1;
                                state.handle_mouse_motion(delta_x, delta_y);
                            }
                        }

                        state.last_mouse_pos = Some(current_pos);
                    }
                    WindowEvent::RedrawRequested => {
                        // Update camera based on elapsed time
                        #[cfg(not(target_arch = "wasm32"))]
                        let now = Instant::now();
                        #[cfg(target_arch = "wasm32")]
                        let now = web_time::Instant::now();

                        let delta_time = (now - last_update).as_secs_f64();
                        last_update = now;

                        state.update_camera(delta_time);

                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                #[cfg(not(target_arch = "wasm32"))]
                                eprintln!("Surface lost, resizing...");
                                #[cfg(target_arch = "wasm32")]
                                log::warn!("Surface lost, resizing...");
                                state.resize(state.size);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                #[cfg(not(target_arch = "wasm32"))]
                                eprintln!("Out of memory!");
                                #[cfg(target_arch = "wasm32")]
                                log::error!("Out of memory!");
                                target.exit();
                            }
                            Err(e) => {
                                #[cfg(not(target_arch = "wasm32"))]
                                eprintln!("Render error: {:?}", e);
                                #[cfg(target_arch = "wasm32")]
                                log::error!("Render error: {:?}", e);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                state.window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}
