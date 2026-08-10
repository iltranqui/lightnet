//! Image loading and Darknet-compatible tensor preprocessing.
//!
//! `PlanarRgbImage` is the boundary between interleaved RGB image memory and
//! Darknet's contiguous channel-major tensor layout.  Conversion happens once
//! at this boundary; the runtime never has to guess an image memory layout.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::TensorShape;
use image::imageops::{self, FilterType};
use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;

#[derive(Clone, Debug, PartialEq)]
pub struct PlanarRgbImage {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl PlanarRgbImage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = ImageReader::open(path)?
            .with_guessed_format()
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to guess image format for {}: {err}",
                    path.display()
                ))
            })?
            .decode()
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "unsupported image format for {}: {err}",
                    path.display()
                ))
            })?
            .to_rgb8();
        Self::from_rgb_image(&image)
    }

    pub fn from_interleaved_rgb_u8(width: usize, height: usize, bytes: &[u8]) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| Error::InvalidConfig("image dimensions overflow".to_owned()))?;

        if bytes.len() != expected {
            return Err(Error::InvalidConfig(format!(
                "expected {expected} bytes for {width}x{height} RGB image, got {}",
                bytes.len()
            )));
        }

        let plane = width * height;
        let mut data = vec![0.0f32; expected];

        for y in 0..height {
            for x in 0..width {
                let src = (y * width + x) * 3;
                let dst = y * width + x;
                data[dst] = bytes[src] as f32 / 255.0;
                data[plane + dst] = bytes[src + 1] as f32 / 255.0;
                data[plane * 2 + dst] = bytes[src + 2] as f32 / 255.0;
            }
        }

        Ok(Self {
            shape: TensorShape::new(width, height, 3, 1),
            data,
        })
    }

    pub fn resize(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "resize dimensions must be non-zero".to_owned(),
            ));
        }
        if self.shape.width == width && self.shape.height == height {
            return Ok(self.clone());
        }
        let rgb = self.to_rgb8()?;
        let resized = imageops::resize(&rgb, width as u32, height as u32, FilterType::Triangle);
        Self::from_rgb_image(&resized)
    }

    pub fn letterbox(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "letterbox dimensions must be non-zero".to_owned(),
            ));
        }

        let (new_width, new_height) =
            fit_inside(self.shape.width, self.shape.height, width, height)?;
        let resized = self.resize(new_width, new_height)?;
        let resized_rgb = resized.to_rgb8()?;
        let mut canvas = RgbImage::new(width as u32, height as u32);
        draw_filled_rect_mut(
            &mut canvas,
            Rect::at(0, 0).of_size(width as u32, height as u32),
            Rgb([128, 128, 128]),
        );
        let x_offset = (width - new_width) / 2;
        let y_offset = (height - new_height) / 2;
        imageops::overlay(&mut canvas, &resized_rgb, x_offset as i64, y_offset as i64);
        Self::from_rgb_image(&canvas)
    }

    pub fn get(&self, channel: usize, y: usize, x: usize) -> f32 {
        let plane = self.shape.width * self.shape.height;
        self.data[channel * plane + y * self.shape.width + x]
    }

    fn from_rgb_image(image: &RgbImage) -> Result<Self> {
        Self::from_interleaved_rgb_u8(
            image.width() as usize,
            image.height() as usize,
            image.as_raw(),
        )
    }

    /// Convert the planar network image back to an interleaved 8-bit RGB image.
    ///
    /// This is public so the Rust example programs can produce the same annotated
    /// image and video artifacts as the C++ examples.
    pub fn to_rgb8(&self) -> Result<RgbImage> {
        if self.shape.channels != 3 {
            return Err(Error::InvalidConfig(format!(
                "expected 3 channels for RGB conversion, got {}",
                self.shape.channels
            )));
        }

        let width = self.shape.width;
        let height = self.shape.height;
        let plane = width * height;
        let mut bytes = vec![0u8; plane * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let dst = idx * 3;
                bytes[dst] = (self.data[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                bytes[dst + 1] = (self.data[plane + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                bytes[dst + 2] = (self.data[plane * 2 + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }

        RgbImage::from_raw(width as u32, height as u32, bytes)
            .ok_or_else(|| Error::InvalidConfig("failed to construct RGB image buffer".to_owned()))
    }
}

fn fit_inside(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Result<(usize, usize)> {
    if src_w == 0 || src_h == 0 {
        return Err(Error::InvalidConfig(
            "source image dimensions must be non-zero".to_owned(),
        ));
    }
    if (dst_w as f32 / src_w as f32) < (dst_h as f32 / src_h as f32) {
        Ok((dst_w.max(1), (src_h.saturating_mul(dst_w) / src_w).max(1)))
    } else {
        Ok(((src_w.saturating_mul(dst_h) / src_h).max(1), dst_h.max(1)))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::PlanarRgbImage;

    #[test]
    fn converts_interleaved_rgb_to_planar() {
        let image = PlanarRgbImage::from_interleaved_rgb_u8(1, 1, &[255, 128, 0])
            .expect("image conversion should succeed");
        assert_eq!(image.data, vec![1.0, 128.0 / 255.0, 0.0]);
    }

    #[test]
    fn resizes_image_bilinearly() {
        let image =
            PlanarRgbImage::from_interleaved_rgb_u8(2, 1, &[0, 0, 0, 255, 0, 0]).expect("image");
        let resized = image.resize(3, 1).expect("resize");
        assert_eq!(resized.shape.width, 3);
        assert!((resized.data[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn letterboxes_image_with_gray_padding() {
        let image =
            PlanarRgbImage::from_interleaved_rgb_u8(2, 1, &[255, 0, 0, 255, 0, 0]).expect("image");
        let boxed = image.letterbox(2, 2).expect("letterbox");
        assert_eq!(boxed.shape.width, 2);
        assert_eq!(boxed.shape.height, 2);
        assert!((boxed.data[0] - 1.0).abs() < 0.01);
        assert!((boxed.data[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn loads_binary_ppm() {
        let path = std::env::temp_dir().join("darknet-test.ppm");
        let mut bytes = b"P6\n1 1\n255\n".to_vec();
        bytes.extend_from_slice(&[255, 128, 0]);
        fs::write(&path, bytes).expect("ppm should write");

        let image = PlanarRgbImage::load(&path).expect("ppm should load");
        assert_eq!(image.shape, crate::model::TensorShape::new(1, 1, 3, 1));
        assert_eq!(image.data, vec![1.0, 128.0 / 255.0, 0.0]);

        let _ = fs::remove_file(path);
    }
}
