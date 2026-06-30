# Web Deployment Guide for Rust Raytracer

## Prerequisites

1. Install wasm-pack:
```bash
cargo install wasm-pack
```

2. Install a simple HTTP server:
```bash
cargo install basic-http-server
```

## Building for Web

### Option 1: Quick Test (Modify existing code)

Add these dependencies to Cargo.toml for WASM support:

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
console_error_panic_hook = "0.1"
console_log = "1.0"
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = "0.3"
cfg-if = "1.0"
```

### Option 2: Create a Library Target

1. Add to Cargo.toml:
```toml
[lib]
crate-type = ["cdylib", "rlib"]
name = "rust_raytracer"
path = "src/lib.rs"
```

2. Move your State struct and implementation to lib.rs or make it public

3. Build for web:
```bash
wasm-pack build --target web --out-dir web/pkg
```

### Option 3: Use trunk (Recommended - Easiest)

1. Install trunk:
```bash
cargo install trunk
```

2. Create an index.html in the project root:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Raytracer</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #1a1a1a;
        }
        canvas {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <script type="module">
        import init from './live_raytracer.js';
        init();
    </script>
</body>
</html>
```

3. Add data-trunk attributes:
   - Trunk will automatically compile your WASM binary

4. Build and serve:
```bash
trunk serve --release
```

This will:
- Build your project for WASM target
- Start a development server at http://127.0.0.1:8080
- Auto-reload on changes

## Key Code Changes Needed

### 1. In your main.rs or live_raytracer.rs:

Add at the top:
```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
```

### 2. Initialize panic hook for better error messages:

```rust
#[cfg(target_arch = "wasm32")]
{
    std::panic::set_hook(Box::new(console_error_panic_hook::panic_hook));
    console_log::init().expect("Could not initialize logger");
}
```

### 3. Attach canvas to DOM:

```rust
#[cfg(target_arch = "wasm32")]
{
    use winit::platform::web::WindowExtWebSys;
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let dst = doc.body()?;
            let canvas = web_sys::Element::from(window.canvas()?);
            dst.append_child(&canvas).ok()?;
            Some(())
        })
        .expect("Couldn't append canvas to document body.");
}
```

### 4. Use wasm-bindgen-futures for async:

```rust
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    // your event loop code here
}
```

## WGPU Backend

For web, wgpu will automatically use WebGPU (in Chrome/Edge) or fall back to WebGL2.

Make sure your device limits are compatible:
```rust
let required_limits = wgpu::Limits {
    max_texture_dimension_2d: 4096, // Lower for web compatibility
    ..Default::default()
};
```

## Deploy to Production

### Build optimized WASM:

```bash
trunk build --release
```

Output will be in `dist/` folder. Upload to any static hosting:
- GitHub Pages
- Netlify
- Vercel
- Cloudflare Pages

### Automated GitHub Pages deployment

We ship a GitHub Actions workflow in `.github/workflows/deploy-web.yml` that builds the WebGPU viewer with trunk and publishes the contents of `dist/` to GitHub Pages. To enable it:

1. In the repository settings, open **Pages** and set the source to **GitHub Actions**.
2. Ensure workflow permissions under **Settings → Actions → General** allow GitHub Pages deployments ("Read and write permissions").
3. Push changes to the `main` branch or trigger the workflow manually to deploy.

The workflow first executes `cargo test --workspace --all-targets` to ensure the Rust checks are green, then runs `trunk build --release` with `TRUNK_BUILD_PUBLIC_URL=/rust-raytracer/` so assets resolve correctly when served from `https://<username>.github.io/rust-raytracer/`. If you fork the project or host under a custom domain, adjust that environment variable accordingly.

## Performance Notes

- WASM is typically 20-50% slower than native
- WebGPU is faster than WebGL2 but has limited browser support
- Consider reducing default samples/bounces for web
- Use requestAnimationFrame for smooth rendering

