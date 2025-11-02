use crate::vec3::Vec3;
use image::GenericImageView;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

const BYTES_PER_PIXEL: u32 = 3; // RGB

/// Represents a texture with RGB color data
#[allow(dead_code)]
#[derive(Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Interpolation methods for texture sampling
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    Closest,
    Linear,
    Bilinear,
}

#[allow(dead_code)]
impl Texture {
    /// Creates a new texture with the given dimensions and RGB data
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len() as u32, width * height * BYTES_PER_PIXEL);
        Texture {
            width,
            height,
            data,
        }
    }

    /// Samples the texture at UV coordinates with the specified interpolation method
    pub fn get_color_with_interpolation(
        &self,
        u: f64,
        v: f64,
        method: InterpolationMethod,
    ) -> Vec3 {
        match method {
            InterpolationMethod::Closest => self.get_color_closest(u, v),
            InterpolationMethod::Linear => self.get_color_linear(u, v),
            InterpolationMethod::Bilinear => self.get_color_bilinear(u, v),
        }
    }

    /// Samples the texture using nearest-neighbor interpolation
    fn get_color_closest(&self, u: f64, v: f64) -> Vec3 {
        let (x, y) = self.normalize_uv(u, v);
        let x_idx = (x * (self.width - 1) as f64).round() as u32;
        let y_idx = (y * (self.height - 1) as f64).round() as u32;
        self.sample_pixel(x_idx, y_idx)
    }

    /// Samples the texture using linear interpolation along the X axis
    fn get_color_linear(&self, u: f64, v: f64) -> Vec3 {
        let (u, v) = self.normalize_uv(u, v);

        let x = u * (self.width - 1) as f64;
        let y = v * (self.height - 1) as f64;

        let x0 = x.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let fx = x - x0 as f64;

        let y_idx = y.round() as u32;

        let c0 = self.sample_pixel(x0, y_idx);
        let c1 = self.sample_pixel(x1, y_idx);

        c0 * (1.0 - fx) + c1 * fx
    }

    /// Samples the texture using bilinear interpolation
    fn get_color_bilinear(&self, u: f64, v: f64) -> Vec3 {
        let (u, v) = self.normalize_uv(u, v);

        let x = u * (self.width - 1) as f64;
        let y = v * (self.height - 1) as f64;

        let x0 = x.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y0 = y.floor() as u32;
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let c00 = self.sample_pixel(x0, y0);
        let c10 = self.sample_pixel(x1, y0);
        let c01 = self.sample_pixel(x0, y1);
        let c11 = self.sample_pixel(x1, y1);

        let c0 = c00 * (1.0 - fx) + c10 * fx;
        let c1 = c01 * (1.0 - fx) + c11 * fx;

        c0 * (1.0 - fy) + c1 * fy
    }

    /// Normalizes UV coordinates to [0, 1] range with proper V-axis flip
    #[inline]
    fn normalize_uv(&self, u: f64, v: f64) -> (f64, f64) {
        (u.fract(), 1.0 - v.fract())
    }

    /// Samples a single pixel at the given coordinates
    #[inline]
    fn sample_pixel(&self, x: u32, y: u32) -> Vec3 {
        let index = ((y * self.width + x) * BYTES_PER_PIXEL) as usize;
        let r = self.data[index] as f64 / 255.0;
        let g = self.data[index + 1] as f64 / 255.0;
        let b = self.data[index + 2] as f64 / 255.0;
        Vec3::new(r, g, b)
    }

    /// Loads a texture from EXR bytes (for WASM)
    pub fn from_exr_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        use exr::prelude::*;

        eprintln!("🖼️  Loading EXR from {} bytes", bytes.len());

        // Write bytes to a temporary in-memory cursor and use the file-based reader
        let cursor = std::io::Cursor::new(bytes);

        let image_width = Arc::new(AtomicU32::new(0));
        let image_height = Arc::new(AtomicU32::new(0));

        let image = read()
            .no_deep_data()
            .largest_resolution_level()
            .rgba_channels(
                {
                    let image_width = Arc::clone(&image_width);
                    let image_height = Arc::clone(&image_height);
                    move |resolution, _| {
                        let width = resolution.width() as u32;
                        let height = resolution.height() as u32;
                        eprintln!("📊 EXR dimensions: {}x{}", width, height);
                        image_width.store(width, Ordering::Relaxed);
                        image_height.store(height, Ordering::Relaxed);
                        vec![0u8; (width * height * BYTES_PER_PIXEL) as usize]
                    }
                },
                {
                    let image_width = Arc::clone(&image_width);
                    move |data, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                        let width = image_width.load(Ordering::Relaxed).max(1);
                        let index = ((position.y() as u32 * width + position.x() as u32)
                            * BYTES_PER_PIXEL) as usize;
                        if index + 2 < data.len() {
                            let r = r.max(0.0);
                            let g = g.max(0.0);
                            let b = b.max(0.0);

                            const EXPOSURE: f32 = 1.5;
                            let r = r * EXPOSURE;
                            let g = g * EXPOSURE;
                            let b = b * EXPOSURE;

                            let tone_map = |x: f32| x / (1.0 + x);

                            // Apply dithering to reduce banding artifacts
                            let dither_matrix = [
                                [0.0, 0.5, 0.125, 0.625],
                                [0.75, 0.25, 0.875, 0.375],
                                [0.1875, 0.6875, 0.0625, 0.5625],
                                [0.9375, 0.4375, 0.8125, 0.3125],
                            ];
                            let dither_value = dither_matrix[position.y() % 4][position.x() % 4];
                            let dither_amount = (dither_value - 0.5) / 255.0;

                            data[index] =
                                ((tone_map(r) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                            data[index + 1] =
                                ((tone_map(g) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                            data[index + 2] =
                                ((tone_map(b) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                        }
                    }
                },
            )
            .first_valid_layer()
            .all_attributes()
            .from_buffered(cursor)?;

        let layer = &image.layer_data;
        let data = layer.channel_data.pixels.clone();

        Ok(Texture::new(
            image_width.load(Ordering::Relaxed).max(1),
            image_height.load(Ordering::Relaxed).max(1),
            data,
        ))
    }

    /// Loads a texture from an EXR file
    pub fn from_exr(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use exr::prelude::*;

        eprintln!("🖼️  Loading EXR file: {}", path);

        let image_width = Arc::new(AtomicU32::new(0));
        let image_height = Arc::new(AtomicU32::new(0));

        let image = read_first_rgba_layer_from_file(
            path,
            {
                let image_width = Arc::clone(&image_width);
                let image_height = Arc::clone(&image_height);
                move |resolution, _| {
                    let width = resolution.width() as u32;
                    let height = resolution.height() as u32;
                    eprintln!("📊 EXR dimensions: {}x{}", width, height);
                    image_width.store(width, Ordering::Relaxed);
                    image_height.store(height, Ordering::Relaxed);
                    vec![0u8; (width * height * BYTES_PER_PIXEL) as usize]
                }
            },
            {
                let image_width = Arc::clone(&image_width);
                move |data, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                    // Log sample pixels for debugging (every 512 pixels in each dimension)
                    if position.x() % 512 == 0 && position.y() % 512 == 0 {
                        let r_clamped = r.max(0.0);
                        let g_clamped = g.max(0.0);
                        let b_clamped = b.max(0.0);
                        let tone_map = |x: f32| x / (1.0 + x);
                        eprintln!(
                            "  Pixel ({:4}, {:4}): HDR=({:8.3}, {:8.3}, {:8.3}) -> Tone-mapped=({:.3}, {:.3}, {:.3}) -> 8-bit=({}, {}, {})",
                            position.x(),
                            position.y(),
                            r,
                            g,
                            b,
                            tone_map(r_clamped),
                            tone_map(g_clamped),
                            tone_map(b_clamped),
                            (tone_map(r_clamped) * 255.0) as u8,
                            (tone_map(g_clamped) * 255.0) as u8,
                            (tone_map(b_clamped) * 255.0) as u8
                        );
                    }

                    // Calculate width from the position bounds
                    let width = image_width.load(Ordering::Relaxed).max(1);
                    let index = ((position.y() as u32 * width + position.x() as u32)
                        * BYTES_PER_PIXEL) as usize;
                    if index + 2 < data.len() {
                        // Ensure values are positive (some EXR can have negative values)
                        let r = r.max(0.0);
                        let g = g.max(0.0);
                        let b = b.max(0.0);

                        // Apply exposure adjustment - multiply by a factor to brighten the image
                        // This EXR seems to have low values, so we scale them up
                        const EXPOSURE: f32 = 1.5;
                        let r = r * EXPOSURE;
                        let g = g * EXPOSURE;
                        let b = b * EXPOSURE;

                        // Apply Reinhard tone mapping to convert HDR to LDR (compress high values)
                        // Keep in linear space - no gamma correction
                        let tone_map = |x: f32| x / (1.0 + x);

                        // Apply dithering to reduce banding artifacts
                        // Simple ordered dithering using pixel position
                        let dither_matrix = [
                            [0.0, 0.5, 0.125, 0.625],
                            [0.75, 0.25, 0.875, 0.375],
                            [0.1875, 0.6875, 0.0625, 0.5625],
                            [0.9375, 0.4375, 0.8125, 0.3125],
                        ];
                        let dither_value = dither_matrix[position.y() % 4][position.x() % 4];
                        let dither_amount = (dither_value - 0.5) / 255.0; // Small dither in 0-1 range

                        // Convert to 8-bit with dithering (linear space)
                        data[index] =
                            ((tone_map(r) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                        data[index + 1] =
                            ((tone_map(g) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                        data[index + 2] =
                            ((tone_map(b) + dither_amount) * 255.0).clamp(0.0, 255.0) as u8;
                    }
                }
            },
        )?;

        let data = image.layer_data.channel_data.pixels;

        Ok(Texture::new(
            image_width.load(Ordering::Relaxed).max(1),
            image_height.load(Ordering::Relaxed).max(1),
            data,
        ))
    }

    /// Loads a texture from a PNG or JPG file
    pub fn from_image(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let (width, height) = img.dimensions();
        let rgb_img = img.to_rgb8();
        let data = rgb_img.into_raw();

        Ok(Texture::new(width, height, data))
    }
}

impl Default for Texture {
    fn default() -> Self {
        Texture {
            width: 1,
            height: 1,
            data: vec![255, 255, 255],
        }
    }
}

impl std::fmt::Display for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Texture({}x{})", self.width, self.height)
    }
}
