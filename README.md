# rust-rasterizer
A rasterizer implementation in Rust

![Example](doc/bunny_rotated.png)

Try it online: [Live WebGPU Raytracer](https://tchauffi.github.io/rust-rasterizer/)

## Features

This project includes three different raytracing implementations:

1. **CPU Raytracer** - Software-based raytracing running on the CPU
2. **GPU Raytracer** - Hardware-accelerated raytracing using GPU compute shaders (offline rendering)
3. **Live GPU Raytracer** - Real-time interactive GPU raytracer with camera controls

## Usage

### CPU Raytracer (Software Rendering)

The CPU version renders scenes using traditional CPU-based raytracing and outputs to a PPM image file.

```bash
# Build and run (outputs to stdout, redirect to file)
cargo run --release > output.ppm

# Or build first, then run
cargo build --release
./target/release/rust-rasterizer > output.ppm
```

**Features:**
- Full path tracing with multiple bounces
- Direct and indirect lighting
- Mesh support (.obj files)
- Sphere primitives

### GPU Raytracer (Offline Rendering)

The GPU version uses compute shaders to accelerate rendering, outputting to a PPM file.

```bash
# Build and run
cargo run --bin gpu_raytracer --release > output.ppm

# Or build separately
cargo build --bin gpu_raytracer --release
./target/release/gpu_raytracer > output.ppm
```

**Features:**
- GPU-accelerated compute shader rendering
- Same scene quality as CPU version
- Significantly faster rendering times
- Hardware-accelerated ray-triangle intersection

### Live GPU Raytracer (Interactive Real-time)

The live version provides a real-time interactive window where you can navigate the scene.

```bash
# Run the live raytracer
cargo run --bin live_raytracer --release
```

**Controls:**
- **Mouse**: Click and drag to rotate the camera
- **SPACE**: Toggle between raytracing and normals visualization modes
- **Window Title**: Displays current mode and FPS

**Features:**
- Real-time GPU raytracing
- Interactive camera controls
- Two rendering modes:
  - **Raytracing**: Full path tracing with lighting and shadows
  - **Normals**: Fast visualization showing surface normals (useful for debugging)
- Live FPS counter in window title

## Requirements

- Rust (latest stable version)
- For GPU versions: A GPU with compute shader support (Vulkan, Metal, or DirectX 12)

## To do:

- [X] Implement Sphere ray tracing
- [X] Implement Light structures and enhance ray_color function for direct and indirect lighting calculations
- [X] Add more shapes (planes, triangles, meshes)
- [X] Optimize performance using GPU acceleration
- [ ] Add BVH acceleration structure
- [ ] Add texture mapping and material properties
- [X] Implement shadows and reflections
- [X] Create a user interface for scene setup and rendering options
- [X] Write documentation and usage examples
