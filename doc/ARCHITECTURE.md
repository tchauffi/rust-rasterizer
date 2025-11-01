# Rust Raytracer Architecture

## Overview

This project contains both **CPU** and **GPU** implementations of a path tracer. While they share scene data structures, they have separate rendering implementations.

## Project Structure

```
src/
├── lib.rs              # Library exports
├── main.rs             # CPU raytracer entry point
├── bin/
│   ├── gpu_raytracer.rs      # GPU raytracer (offline)
│   └── live_raytracer.rs     # GPU raytracer (interactive)
├── shaders/
│   ├── raytracer.wgsl        # GPU raytracing compute shader
│   ├── raytracer_normals.wgsl # GPU normals visualization
│   └── display.wgsl          # Display shader
├── Core Modules (shared between CPU and GPU)
├── camera.rs           # Camera setup
├── material.rs         # Material properties
├── mesh.rs             # Triangle mesh loading
├── sphere.rs           # Sphere primitive
├── vec3.rs             # 3D vector math
├── ray.rs              # Ray data structure
├── hittable.rs         # Intersection trait
├── aabb.rs             # Bounding boxes
│
├── CPU-Specific Modules
├── scene.rs            # CPU scene management & rendering
├── lights.rs           # CPU lighting computation
│
└── GPU-Specific Modules
    └── gpu_scene.rs    # GPU scene setup utilities
```

## Architecture Patterns

### CPU Raytracer (src/scene.rs + src/lights.rs)

**Design Pattern**: Modular object-oriented with trait-based polymorphism

**Key Features**:
- `Scene` struct manages camera, objects, and lights
- `Light` enum with variants: Point, Directional, Ambient
- Each light type implements `compute()` method for lighting calculation
- `Hittable` trait allows polymorphic objects (spheres, meshes)
- Recursive ray tracing with depth limits

**Lighting Model** (src/lights.rs):
```rust
pub fn compute_contribution(hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3
```
- Separated lighting logic from scene traversal
- Each light type has its own `compute()` implementation
- Shadow rays test occlusion for each light
- Uses Lambertian diffuse shading

**Benefits**:
- ✅ Easy to add new light types
- ✅ Testable lighting code
- ✅ Clear separation of concerns
- ✅ Maintainable and readable

### GPU Raytracer (src/shaders/raytracer.wgsl)

**Design Pattern**: Data-oriented compute shader

**Key Features**:
- All scene data passed as GPU buffers
- Lighting implemented directly in WGSL shader
- Parallel execution across pixels
- Fixed lighting model (directional + ambient)

**Architecture**:
```
Rust Host Code (gpu_raytracer.rs, live_raytracer.rs)
    ↓ (uploads scene data)
GPU Buffers
    ├── SceneUniform (camera, lights, config)
    ├── Triangle Buffer (mesh geometry)
    └── Sphere Buffer (sphere geometry)
    ↓ (compute shader)
WGSL Shader (raytracer.wgsl)
    ├── Ray generation
    ├── Intersection tests
    ├── Lighting calculation
    └── Path tracing
    ↓ (outputs)
Output Buffer → Display
```

**Tradeoffs**:
- ✅ Much faster (GPU parallelism)
- ✅ Real-time capable (interactive mode)
- ❌ Lighting logic duplicated in WGSL
- ❌ Harder to extend (requires shader changes)
- ❌ Can't use Rust's `Light` trait polymorphism

### Shared Module: gpu_scene.rs

Provides utilities for converting Rust scene data to GPU-compatible formats:

```rust
pub struct SceneUniform { /* GPU-compatible scene data */ }
pub struct GpuTriangle { /* GPU-compatible triangle */ }
pub struct GpuSphere { /* GPU-compatible sphere */ }

pub fn mesh_to_gpu_triangles(mesh: &Mesh) -> Vec<GpuTriangle>
pub fn sphere_to_gpu(sphere: &Sphere) -> GpuSphere
pub fn camera_frame(camera: &Camera) -> (Vec3, Vec3, Vec3)
```

**Purpose**:
- Converts Rust structs to GPU buffer data
- Handles float precision (f64 → f32)
- Shared by both GPU implementations

## Lighting Implementation

### CPU (src/lights.rs)

```rust
// Each light type implements compute()
impl PointLight {
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Shadow ray testing
        // Distance-based attenuation
        // Lambertian diffuse
    }
}

impl DirectionalLight {
    pub fn compute(&self, hit: &HitRecord, objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Shadow ray testing
        // Constant intensity
        // Lambertian diffuse
    }
}

impl AmbientLight {
    pub fn compute(&self, _hit: &HitRecord, _objects: &[Box<dyn Hittable>]) -> Vec3 {
        // Simple ambient term (no shadow testing)
    }
}
```

### GPU (src/shaders/raytracer.wgsl)

```wgsl
// Fixed lighting model in shader
fn compute_lighting(hit: HitInfo, normal: vec3<f32>) -> vec3<f32> {
    var light = vec3(0.0);

    // Ambient
    light += scene.ambient_color.rgb;

    // Directional (with shadow test)
    let to_light = -scene.light_direction.xyz;
    if (!in_shadow(hit.point + normal * 0.001, to_light)) {
        let diffuse = max(dot(normal, to_light), 0.0);
        light += scene.light_color.rgb * diffuse * scene.light_direction.w;
    }

    return light;
}
```

## Why Two Implementations?

1. **CPU Raytracer**:
   - Educational and flexible
   - Easy to debug and extend
   - Supports complex lighting models
   - Good for final high-quality renders

2. **GPU Raytracer**:
   - Production-ready performance
   - Interactive real-time rendering
   - Fixed but fast lighting
   - Good for previews and iteration

## Future Improvements

### Unify Lighting
Currently lighting logic is duplicated between CPU (Rust) and GPU (WGSL). Options:

1. **Code generation**: Generate WGSL from Rust lighting code
2. **Shader includes**: Template WGSL with lighting modules
3. **Accept duplication**: Keep optimized versions for each platform

### Add More Light Types
- Area lights (soft shadows)
- Spotlights
- Environment maps (image-based lighting)

### Material System
- Currently basic (color + roughness)
- Could add: metallic, emission, transparency
- PBR (Physically Based Rendering) model

### Acceleration
- CPU: BVH (Bounding Volume Hierarchy) for meshes
- GPU: Already uses some spatial optimization

## Testing

```bash
# CPU raytracer (slow, high quality)
cargo run --release > output.ppm

# GPU raytracer (fast, good quality)
cargo run --bin gpu_raytracer --release > output.ppm

# Interactive GPU raytracer (real-time)
cargo run --bin live_raytracer --release
```

## Key Takeaways

✅ **Modular Design**: Lighting separated from scene traversal
✅ **Platform Optimized**: Different implementations for CPU vs GPU
✅ **Shared Infrastructure**: Common scene data and utilities
✅ **Extensible**: Easy to add new primitives and lights (CPU-side)
✅ **Performance**: GPU version suitable for interactive use
