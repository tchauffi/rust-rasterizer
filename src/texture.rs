use crate::vec3::Vec3;
use image::GenericImageView;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    Closest,
    Linear,
    Bilinear,
}

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

    fn get_width(&self) -> u32 {
        self.width
    }

    fn get_height(&self) -> u32 {
        self.height
    }

    /// Loads a texture from an EXR file
    pub fn from_exr(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use exr::prelude::*;

        let image = read_first_rgba_layer_from_file(
            path,
            |resolution, _| {
                let width = resolution.width() as u32;
                let height = resolution.height() as u32;
                vec![0u8; (width * height * BYTES_PER_PIXEL) as usize]
            },
            |data, position, (r, g, b, _a): (f32, f32, f32, f32)| {
                let width = data.len() as u32 / BYTES_PER_PIXEL / position.height() as u32;
                let index = ((position.y() as u32 * width + position.x() as u32) * BYTES_PER_PIXEL)
                    as usize;
                data[index] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                data[index + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                data[index + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            },
        )?;

        let (data, resolution) = (image.layer_data.channel_data.pixels, image.layer_data.size);

        Ok(Texture::new(
            resolution.width() as u32,
            resolution.height() as u32,
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
