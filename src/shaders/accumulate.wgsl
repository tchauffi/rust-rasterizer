// Accumulation shader for temporal anti-aliasing
// Blends the current frame with the accumulated history

@group(0) @binding(0)
var<storage, read> current_frame: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> accumulation: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> frame_info: vec4<u32>; // x: width, y: height, z: accumulated_frames, w: padded_width

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = frame_info.x;
    let height = frame_info.y;
    let accumulated_frames = frame_info.z;
    let padded_width = frame_info.w;

    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let pixel_index = global_id.y * padded_width + global_id.x;

    // Get current frame sample
    let current = current_frame[pixel_index];

    if (accumulated_frames == 0u) {
        // First frame: just store it
        accumulation[pixel_index] = current;
    } else {
        // Progressive averaging: blend with accumulated history
        let old_accumulated = accumulation[pixel_index];
        let weight = 1.0 / f32(accumulated_frames + 1u);
        let new_accumulated = old_accumulated * (1.0 - weight) + current * weight;
        accumulation[pixel_index] = new_accumulated;
    }
}
