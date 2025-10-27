use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use rust_raytracer::camera::Camera;
use rust_raytracer::material::Material;
use rust_raytracer::mesh::Mesh;
use rust_raytracer::sphere::Sphere;
use rust_raytracer::vec3::Vec3;
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSphere {
    center_radius: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    resolution: [u32; 4],
    camera_position: [f32; 4],
    lower_left_corner: [f32; 4],
    horizontal: [f32; 4],
    vertical: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    mesh_color: [f32; 4],
    render_config: [u32; 4],
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
    compute_pipeline: wgpu::ComputePipeline,
    compute_pipeline_normals: wgpu::ComputePipeline,

    // Display resources
    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,
    output_texture: wgpu::Texture,

    // Scene data
    camera: Camera,
    scene_uniform: SceneUniform,

    // Camera control
    camera_position: Vec3,
    camera_yaw: f64,   // Rotation around Y axis
    camera_pitch: f64, // Rotation up/down

    // Mouse state
    last_mouse_pos: Option<(f64, f64)>,
    mouse_pressed: bool,

    // Render mode
    use_raytracing: bool, // true = raytracing, false = normals view

    // FPS tracking
    frame_count: u32,
    fps_timer: std::time::Instant,
    current_fps: f32,
}

impl State {
    async fn new(window: Arc<winit::window::Window>) -> Result<Self> {
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

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
        let mut bunny = Mesh::from_obj_file("data/bunny.obj", bunny_material)
            .map_err(|err| anyhow!("failed to load bunny OBJ: {err}"))?;
        bunny.rotate_y(180.0);
        bunny.transform(10.0, Vec3::new(0.0, -1.0, 4.0));

        eprintln!("Bunny positioned at Z=4.0, centered around Y=-1.0");

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
        eprintln!("Green sphere at (2.0, 0.0, 5.0), Blue sphere at (-1.6, 0.0, 5.0)");
        let spheres = [sphere2, sphere3];

        let triangles = mesh_to_gpu_triangles(&bunny);
        let triangle_count = triangles.len() as u32;
        let triangles_storage: Cow<[GpuTriangle]> = if triangles.is_empty() {
            Cow::Owned(vec![GpuTriangle::zeroed()])
        } else {
            Cow::Owned(triangles)
        };

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
        let samples_per_pixel = 1u32; // 1 sample for real-time interactivity
        let max_bounces = 2u32; // Reduced bounces for speed

        let scene_uniform = SceneUniform {
            resolution: [width, height, triangle_count, sphere_count],
            camera_position: vec3_to_array(camera.position, 0.0),
            lower_left_corner: vec3_to_array(lower_left_corner, 0.0),
            horizontal: vec3_to_array(horizontal, 0.0),
            vertical: vec3_to_array(vertical, 0.0),
            light_direction: vec3_to_array(directional_dir, directional_strength as f32),
            light_color: vec3_to_array(directional_color, 0.0),
            ambient_color: vec3_to_array(ambient_color, 0.0),
            mesh_color: vec3_to_array(bunny.material.color, 1.0),
            render_config: [samples_per_pixel, max_bounces, 0, 0],
        };

        // Create WGPU instance
        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
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
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raytracer-bind-group"),
            layout: &compute_bind_group_layout,
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
            ],
        });

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
            label: Some("raytracer-pipeline"),
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
            compute_pipeline,
            compute_pipeline_normals,
            display_pipeline,
            display_bind_group,
            output_texture,
            camera_position: camera.position,
            camera_yaw: 180.0, // Start looking at +Z (toward the scene)
            camera_pitch: 0.0,
            camera,
            scene_uniform,
            last_mouse_pos: None,
            mouse_pressed: false,
            use_raytracing: true, // Start with raytracing mode
            frame_count: 0,
            fps_timer: std::time::Instant::now(),
            current_fps: 0.0,
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

        eprintln!(
            "Mouse delta: ({:.2}, {:.2}) -> Yaw: {:.1}°, Pitch: {:.1}°",
            delta_x, delta_y, self.camera_yaw, self.camera_pitch
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Update FPS
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.fps_timer = std::time::Instant::now();

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

        // Copy buffer to texture
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size.width * 16), // 4 f32s (16 bytes) per pixel
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
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("GPU Raytracer - Drag mouse to rotate camera")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .with_visible(true)
            .build(&event_loop)?,
    );

    eprintln!("Window created, initializing GPU...");
    let mut state = pollster::block_on(State::new(window.clone()))?;
    eprintln!("GPU initialized, starting event loop...");
    eprintln!("Controls:");
    eprintln!("  - Click and drag to rotate camera");
    eprintln!("  - Press SPACE to toggle between raytracing and normals view");
    state.window.focus_window();
    state.window.request_redraw();

    let mut last_update = std::time::Instant::now();

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    if key_event.state == ElementState::Pressed {
                        if let winit::keyboard::PhysicalKey::Code(keycode) = key_event.physical_key
                        {
                            if keycode == winit::keyboard::KeyCode::Space {
                                state.use_raytracing = !state.use_raytracing;
                                let mode = if state.use_raytracing {
                                    "Raytracing"
                                } else {
                                    "Normals"
                                };
                                eprintln!("Switched to {} mode", mode);
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

                    if state.mouse_pressed {
                        if let Some(last_pos) = state.last_mouse_pos {
                            let delta_x = current_pos.0 - last_pos.0;
                            let delta_y = current_pos.1 - last_pos.1;
                            state.handle_mouse_motion(delta_x, delta_y);
                        } else {
                            eprintln!("Mouse pressed but no last position");
                        }
                    }

                    state.last_mouse_pos = Some(current_pos);
                }
                WindowEvent::RedrawRequested => {
                    // Update camera based on elapsed time
                    let now = std::time::Instant::now();
                    let delta_time = (now - last_update).as_secs_f64();
                    last_update = now;

                    state.update_camera(delta_time);

                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            eprintln!("Surface lost, resizing...");
                            state.resize(state.size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            target.exit();
                        }
                        Err(e) => {
                            eprintln!("Render error: {:?}", e);
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                state.window.request_redraw();
            }
            _ => {}
        }
    })?;

    Ok(())
}

fn mesh_to_gpu_triangles(mesh: &Mesh) -> Vec<GpuTriangle> {
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

fn sphere_to_gpu(sphere: &Sphere) -> GpuSphere {
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

fn vec3_to_array(vec: Vec3, w: f32) -> [f32; 4] {
    [vec.x as f32, vec.y as f32, vec.z as f32, w]
}

fn camera_frame(camera: &Camera) -> (Vec3, Vec3, Vec3) {
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
