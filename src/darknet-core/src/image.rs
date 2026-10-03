//! Image loading and Darknet-compatible tensor preprocessing.
//!
//! `PlanarRgbImage` is the boundary between interleaved RGB image memory and
//! Darknet's contiguous channel-major tensor layout.  Conversion happens once
//! at this boundary; the runtime never has to guess an image memory layout.

use std::io::Cursor;
use std::path::Path;

use crate::error::{Error, Result};
use crate::model::TensorShape;
use image::{ImageReader, RgbImage};

#[derive(Clone, Debug, PartialEq)]
pub struct PlanarRgbImage {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl PlanarRgbImage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let profiling = crate::loader_profile::enabled();
        let read_start = std::time::Instant::now();
        let bytes = std::fs::read(path)?;
        let decode_start = std::time::Instant::now();
        let image = match decode_jpeg_libjpeg(&bytes) {
            Some(image) => image,
            None => ImageReader::new(Cursor::new(&bytes))
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
                .to_rgb8(),
        };
        let planar_start = std::time::Instant::now();
        let planar = Self::from_rgb_image(&image)?;
        if profiling {
            crate::loader_profile::add_load();
            crate::loader_profile::add_read(decode_start - read_start);
            crate::loader_profile::add_decode(planar_start - decode_start);
            crate::loader_profile::add_planar(planar_start.elapsed());
        }
        Ok(planar)
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

    /// Darknet's own float bilinear resize (`Darknet::resize_image`,
    /// src-lib/darknet_image.cpp), used by `darknet detector test`.
    ///
    /// Align-corners sampling (`scale = (src - 1) / (dst - 1)`), horizontal pass
    /// first into an intermediate plane, then vertical; all in `f32` with the
    /// same operation order as C++, so results are bit-identical.
    pub fn resize(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "resize dimensions must be non-zero".to_owned(),
            ));
        }
        let (sw, sh, channels) = (self.shape.width, self.shape.height, self.shape.channels);
        if sw == width && sh == height {
            return Ok(self.clone());
        }

        let w_scale = (sw as f32 - 1.0) / (width as f32 - 1.0);
        let h_scale = (sh as f32 - 1.0) / (height as f32 - 1.0);
        let src_plane = sw * sh;
        let mut part = vec![0.0f32; width * sh * channels];
        let mut out = vec![0.0f32; width * height * channels];

        for ch in 0..channels {
            for y in 0..sh {
                let src_row = &self.data[ch * src_plane + y * sw..][..sw];
                let dst_row = &mut part[(ch * sh + y) * width..][..width];
                for (c, dst) in dst_row.iter_mut().enumerate() {
                    *dst = if c == width - 1 || sw == 1 {
                        src_row[sw - 1]
                    } else {
                        let sx = c as f32 * w_scale;
                        let ix = sx as usize;
                        let dx = sx - ix as f32;
                        (1.0 - dx) * src_row[ix] + dx * src_row[ix + 1]
                    };
                }
            }
            for y in 0..height {
                let sy = if sh == 1 { 0.0 } else { y as f32 * h_scale };
                let iy = sy as usize;
                let dy = sy - iy as f32;
                let row0 = &part[(ch * sh + iy) * width..][..width];
                let dst_row = &mut out[(ch * height + y) * width..][..width];
                for (dst, v) in dst_row.iter_mut().zip(row0) {
                    *dst = (1.0 - dy) * v;
                }
                if y == height - 1 || sh == 1 {
                    continue;
                }
                let row1 = &part[(ch * sh + iy + 1) * width..][..width];
                for (dst, v) in dst_row.iter_mut().zip(row1) {
                    *dst += dy * v;
                }
            }
        }

        Ok(Self {
            shape: TensorShape::new(width, height, channels, 1),
            data: out,
        })
    }

    /// Bit-exact `cv::resize(..., INTER_LINEAR_EXACT)` on 8-bit RGB, as done by
    /// `Darknet::load_image(path, w, h, c)` (src-lib/darknet_image.cpp), which is
    /// the path `darknet detector map` uses.  Half-pixel centres, 8 fractional
    /// bit coefficients, horizontal then vertical fixed-point pass, round at the
    /// end.  The source is recovered losslessly as `u8` (it was `u8 / 255`).
    pub fn resize_linear_exact(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "resize dimensions must be non-zero".to_owned(),
            ));
        }
        if self.shape.width == width && self.shape.height == height {
            return Ok(self.clone());
        }
        let rgb = self.to_rgb8()?;
        let (sw, sh) = (self.shape.width, self.shape.height);
        let src = rgb.as_raw();
        let cx = exact_coeffs(sw, width);
        let cy = exact_coeffs(sh, height);

        // Horizontal pass: units of 1/256.
        let mut tmp = vec![0u32; sh * width * 3];
        for y in 0..sh {
            for (x, &(x0, x1, c0, c1)) in cx.iter().enumerate() {
                for ch in 0..3 {
                    tmp[(y * width + x) * 3 + ch] = src[(y * sw + x0) * 3 + ch] as u32 * c0
                        + src[(y * sw + x1) * 3 + ch] as u32 * c1;
                }
            }
        }
        // Vertical pass: units of 1/65536, rounded.
        let mut bytes = vec![0u8; width * height * 3];
        for (y, &(y0, y1, c0, c1)) in cy.iter().enumerate() {
            for i in 0..width * 3 {
                let v = tmp[y0 * width * 3 + i] * c0 + tmp[y1 * width * 3 + i] * c1;
                bytes[y * width * 3 + i] = ((v + 32768) >> 16) as u8;
            }
        }
        Self::from_interleaved_rgb_u8(width, height, &bytes)
    }

    /// Load an image and resize it with OpenCV `INTER_LINEAR_EXACT` when its
    /// size differs from `width x height` (mirrors `Darknet::load_image` with
    /// desired dimensions).
    pub fn load_resized_exact(path: impl AsRef<Path>, width: usize, height: usize) -> Result<Self> {
        Self::load(path)?.resize_linear_exact(width, height)
    }

    /// `Darknet::letterbox_image`: float resize to fit, pad with 0.5, integer
    /// offsets `(dst - new) / 2`.
    pub fn letterbox(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "letterbox dimensions must be non-zero".to_owned(),
            ));
        }

        let (new_width, new_height) =
            fit_inside(self.shape.width, self.shape.height, width, height)?;
        let resized = self.resize(new_width, new_height)?;
        let channels = self.shape.channels;
        let mut data = vec![0.5f32; width * height * channels];
        let x_offset = (width - new_width) / 2;
        let y_offset = (height - new_height) / 2;
        for ch in 0..channels {
            for y in 0..new_height {
                let src = &resized.data[(ch * new_height + y) * new_width..][..new_width];
                let dst = (ch * height + y + y_offset) * width + x_offset;
                data[dst..dst + new_width].copy_from_slice(src);
            }
        }
        Ok(Self {
            shape: TensorShape::new(width, height, channels, 1),
            data,
        })
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

/// Decode a JPEG with libjpeg(-turbo) defaults (ISLOW DCT, fancy upsampling),
/// which is what OpenCV's `imread` uses in C++ Darknet.  The pure-Rust
/// `image`/zune decoder differs by up to 4 levels on ~10% of pixels.  Returns
/// `None` for non-JPEG data or anything libjpeg rejects, so the caller can fall
/// back to the `image` crate.
fn decode_jpeg_libjpeg(bytes: &[u8]) -> Option<RgbImage> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    // mozjpeg reports fatal libjpeg errors by unwinding.
    std::panic::catch_unwind(|| {
        let mut decoder = mozjpeg::Decompress::new_mem(bytes).ok()?.rgb().ok()?;
        let (width, height) = (decoder.width(), decoder.height());
        let pixels: Vec<[u8; 3]> = decoder.read_scanlines().ok()?;
        decoder.finish().ok()?;
        RgbImage::from_raw(width as u32, height as u32, pixels.into_flattened())
    })
    .ok()
    .flatten()
}

/// OpenCV `INTER_LINEAR_EXACT` taps: `(x0, x1, c0, c1)` with `c0 + c1 == 256`.
fn exact_coeffs(src: usize, dst: usize) -> Vec<(usize, usize, u32, u32)> {
    (0..dst)
        .map(|d| {
            let fsx = (d as f64 + 0.5) * src as f64 / dst as f64 - 0.5;
            let mut x0 = fsx.floor();
            let mut fx = fsx - x0;
            if x0 < 0.0 {
                x0 = 0.0;
                fx = 0.0;
            }
            if x0 >= src as f64 - 1.0 {
                x0 = src as f64 - 1.0;
                fx = 0.0;
            }
            let x0 = x0 as usize;
            let c1 = (fx * 256.0 + 0.5).floor() as u32;
            (x0, (x0 + 1).min(src - 1), 256 - c1, c1)
        })
        .collect()
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

    /// JPEG must decode bit-identically to OpenCV/libjpeg-turbo (fixture RGB
    /// produced by `cv2.imread` + BGR->RGB).
    #[test]
    fn jpeg_decode_matches_opencv_exactly() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let raw = fs::read(format!("{dir}/chroma_420.rgb")).expect("rgb fixture");
        let image = PlanarRgbImage::load(format!("{dir}/chroma_420.jpg")).expect("load jpeg");
        let expected =
            PlanarRgbImage::from_interleaved_rgb_u8(image.shape.width, image.shape.height, &raw)
                .expect("expected image");
        assert_eq!(image.data, expected.data);
    }

    fn fixture(name: &str) -> Vec<u8> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        fs::read(format!("{dir}/{name}")).expect("fixture")
    }

    fn source(i: usize) -> PlanarRgbImage {
        PlanarRgbImage::from_interleaved_rgb_u8(96, 72, &fixture(&format!("resize{i}_src_96x72.rgb")))
            .expect("source")
    }

    /// `resize_linear_exact` must equal `cv2.resize(INTER_LINEAR_EXACT)` byte
    /// for byte (fixtures are crops of LegoGears_v2 frames).
    #[test]
    fn resize_linear_exact_matches_opencv_bytes() {
        for i in 0..3 {
            for (w, h) in [(40, 28), (130, 100)] {
                let got = source(i).resize_linear_exact(w, h).expect("resize");
                let expected = PlanarRgbImage::from_interleaved_rgb_u8(
                    w,
                    h,
                    &fixture(&format!("resize{i}_exact_{w}x{h}.rgb")),
                )
                .unwrap();
                assert_eq!(got.data, expected.data, "fixture {i} {w}x{h}");
            }
        }
    }

    /// `resize` is Darknet's float resize; fixtures come from a float32 numpy
    /// replica of `Darknet::resize_image`.
    #[test]
    fn resize_matches_darknet_float_resize() {
        for i in 0..3 {
            for (w, h) in [(40, 28), (50, 60)] {
                let got = source(i).resize(w, h).expect("resize");
                let raw = fixture(&format!("resize{i}_float_{w}x{h}.f32"));
                let expected: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                assert_eq!(got.data.len(), expected.len());
                let max = got
                    .data
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(max <= 1e-6, "fixture {i} {w}x{h} max diff {max}");
            }
        }
    }

    #[test]
    fn letterbox_uses_half_gray_and_integer_offsets() {
        // 96x72 into 50x60: new 50x37, y offset (60-37)/2 = 11.
        let boxed = source(0).letterbox(50, 60).expect("letterbox");
        let resized = source(0).resize(50, 37).expect("resize");
        assert_eq!(boxed.get(0, 0, 0), 0.5);
        assert_eq!(boxed.get(2, 10, 49), 0.5);
        assert_eq!(boxed.get(2, 48, 0), 0.5);
        assert_eq!(boxed.get(1, 11, 7), resized.get(1, 0, 7));
        assert_eq!(boxed.get(1, 47, 7), resized.get(1, 36, 7));
    }

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
