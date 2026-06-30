# WASM Debugging Guide

## How to Debug the Deployed WASM Application

### 1. **Check Browser Console Logs**

The application is configured with detailed logging. Open your browser's developer tools:

- **Chrome/Edge**: Press `F12` or `Ctrl+Shift+I` (Windows) / `Cmd+Option+I` (Mac)
- **Firefox**: Press `F12` or `Ctrl+Shift+K` (Windows) / `Cmd+Option+K` (Mac)
- **Safari**: Enable Developer Menu in Preferences, then press `Cmd+Option+C`

Look for log messages like:
- `"WASM module loaded, starting application..."`
- `"Embedded OBJ data length: X bytes"`
- `"Parsed mesh: X vertices, Y faces"`
- `"GPU data: X triangles, Y BVH nodes"`
- `"Scene uniform will have: triangle_count=X, bvh_node_count=Y"`

### 2. **Local Testing**

Test the WASM build locally before checking the deployed version:

```bash
# Build and serve locally
trunk serve --open

# Or build without serving
trunk build --release
```

The local server will be available at `http://127.0.0.1:8080` (or the port shown in terminal).

### 3. **Common Issues and Solutions**

#### Issue: Black screen or no bunny mesh visible

**Check 1: Verify mesh data is loaded**
Look in console for:
```
Embedded OBJ data length: XXXXX bytes
Parsed mesh: 2503 vertices, 4968 faces
GPU data: 4968 triangles, XXX BVH nodes
```

If you see `0 triangles`, the mesh didn't load properly.

**Check 2: WebGPU support**
Ensure your browser supports WebGPU:
- Chrome/Edge 113+
- Check `chrome://gpu` or `edge://gpu` to verify WebGPU is enabled

**Check 3: Camera position**
The bunny is positioned at:
- Position: `(0.0, -1.0, 4.0)` 
- Scaled 10x
- Rotated 180° around Y-axis
- Camera at origin looking along negative Z

Try pressing `N` to switch to normals mode to see if geometry is there but lighting is wrong.

#### Issue: "Failed to initialize" error

Check console for specific WebGPU errors. Common causes:
- Browser doesn't support WebGPU
- GPU driver issues
- Insufficient GPU capabilities

#### Issue: Mesh appears locally but not on GitHub Pages

**Step 1: Verify the workflow ran**
- Go to your GitHub repository
- Click "Actions" tab
- Check that "Deploy Web Viewer" workflow completed successfully
- Look at the timestamp to confirm it ran after your latest changes

**Step 2: Hard refresh the page**
The WASM binary is content-hashed, but the browser may cache the HTML:
- Chrome/Edge/Firefox: `Ctrl+Shift+R` (Windows) / `Cmd+Shift+R` (Mac)
- Safari: `Cmd+Option+R`

**Step 3: Check the deployed files**
You can inspect what was actually deployed by checking the workflow artifacts.

### 4. **Current Logging Points**

The application logs at these key points:

1. **Module initialization** (`start()` function)
2. **Mesh loading** (embedded OBJ parsing)
3. **GPU data conversion** (triangles and BVH nodes)
4. **Buffer creation** (when empty/zeroed)
5. **Scene uniform setup** (final counts)

### 5. **Debug Checklist**

When bunny doesn't appear:

- [ ] Open browser console (F12)
- [ ] Look for WASM initialization message
- [ ] Check for mesh loading logs (vertex/face count)
- [ ] Verify triangle count is > 0
- [ ] Check BVH node count is > 0
- [ ] Look for WebGPU errors
- [ ] Try switching render modes (press `N` key)
- [ ] Verify spheres are visible (if spheres show, WebGPU is working)
- [ ] Check camera controls work (WASD + mouse drag)
- [ ] Test locally with `trunk serve`
- [ ] Compare local logs vs deployed logs

### 6. **Viewing Deployment on GitHub Pages**

Your app should be deployed to:
```
https://tchauffi.github.io/rust-raytracer/
```

(Based on the `TRUNK_BUILD_PUBLIC_URL` in the workflow)

### 7. **Manual Deployment Test**

To manually trigger a deployment:
1. Go to GitHub repository
2. Click "Actions" tab
3. Select "Deploy Web Viewer" workflow
4. Click "Run workflow" dropdown
5. Click green "Run workflow" button

This forces a fresh build and deployment.

### 8. **Advanced Debugging**

If basic checks don't reveal the issue:

1. **Add more detailed logging** to `live_raytracer.rs`:
   ```rust
   log::info!("Bunny bounds: min={:?}, max={:?}", 
              bunny.bounding_box.min, bunny.bounding_box.max);
   ```

2. **Check shader compilation** - GPU errors may indicate shader issues

3. **Verify buffer sizes** - Check that buffer creation succeeds:
   ```rust
   log::info!("Triangle buffer size: {} bytes", triangles.len() * size_of::<GpuTriangle>());
   ```

4. **Test with a simpler mesh** - Replace bunny with a single triangle to isolate the issue

### 9. **Expected Output**

When working correctly, you should see in the console:
```
WASM module loaded, starting application...
Async runtime started
Embedded OBJ data length: ~100000 bytes
Computed normals for mesh: 2503 normals
Parsed mesh: 2503 vertices, 4968 faces
GPU data: 4968 triangles, ~8000 BVH nodes
Scene uniform will have: triangle_count=4968, bvh_node_count=~8000
Green sphere at (2.0, 0.0, 5.0), Blue sphere at (-1.6, 0.0, 5.0)
```

And on screen:
- Two spheres (green on right, blue on left)
- White bunny mesh in the center
- Sky gradient background
- Interactive camera with WASD + mouse
