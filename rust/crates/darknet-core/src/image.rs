use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::model::TensorShape;

#[derive(Clone, Debug, PartialEq)]
pub struct PlanarRgbImage {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl PlanarRgbImage {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase());

        match ext.as_deref() {
            Some("ppm") | Some("pnm") | Some("pgm") => {
                let (width, height, rgb) = decode_netpbm(&bytes)?;
                Self::from_interleaved_rgb_u8(width, height, &rgb)
            }
            Some("bmp") => {
                let (width, height, rgb) = decode_bmp(&bytes)?;
                Self::from_interleaved_rgb_u8(width, height, &rgb)
            }
            _ => Err(Error::InvalidConfig(format!(
                "unsupported image format for {}: expected .ppm/.pgm/.pnm or .bmp",
                path.display()
            ))),
        }
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

        let channels = self.shape.channels;
        let width_scale = if width == 1 {
            0.0
        } else {
            (self.shape.width - 1) as f32 / (width - 1) as f32
        };
        let height_scale = if height == 1 {
            0.0
        } else {
            (self.shape.height - 1) as f32 / (height - 1) as f32
        };
        let mut part = vec![0.0; width * self.shape.height * channels];
        let mut data = vec![0.0; width * height * channels];

        for channel in 0..channels {
            for src_y in 0..self.shape.height {
                for dst_x in 0..width {
                    let value = if dst_x + 1 == width || self.shape.width == 1 {
                        self.get(channel, src_y, self.shape.width - 1)
                    } else {
                        let sample_x = dst_x as f32 * width_scale;
                        let ix = sample_x as usize;
                        let dx = sample_x - ix as f32;
                        (1.0 - dx) * self.get(channel, src_y, ix)
                            + dx * self.get(channel, src_y, ix + 1)
                    };
                    let idx = channel * width * self.shape.height + src_y * width + dst_x;
                    part[idx] = value;
                }
            }
        }

        for channel in 0..channels {
            for dst_y in 0..height {
                let sample_y = dst_y as f32 * height_scale;
                let iy = sample_y as usize;
                let dy = sample_y - iy as f32;
                for dst_x in 0..width {
                    let part_idx = channel * width * self.shape.height + iy * width + dst_x;
                    let mut value = (1.0 - dy) * part[part_idx];
                    if dst_y + 1 != height && self.shape.height != 1 {
                        let next_idx =
                            channel * width * self.shape.height + (iy + 1) * width + dst_x;
                        value += dy * part[next_idx];
                    }
                    let dst_idx = channel * width * height + dst_y * width + dst_x;
                    data[dst_idx] = value;
                }
            }
        }

        Ok(Self {
            shape: TensorShape::new(width, height, channels, 1),
            data,
        })
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
        let mut data = vec![0.5; width * height * self.shape.channels];
        let x_offset = (width - new_width) / 2;
        let y_offset = (height - new_height) / 2;

        for channel in 0..self.shape.channels {
            for y in 0..new_height {
                for x in 0..new_width {
                    let src_idx = channel * new_width * new_height + y * new_width + x;
                    let dst_idx =
                        channel * width * height + (y + y_offset) * width + (x + x_offset);
                    data[dst_idx] = resized.data[src_idx];
                }
            }
        }

        Ok(Self {
            shape: TensorShape::new(width, height, self.shape.channels, 1),
            data,
        })
    }

    pub fn get(&self, channel: usize, y: usize, x: usize) -> f32 {
        let plane = self.shape.width * self.shape.height;
        self.data[channel * plane + y * self.shape.width + x]
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

fn decode_netpbm(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    let mut offset = 0usize;
    let magic = next_pnm_token(bytes, &mut offset)?;
    let width: usize = next_pnm_token(bytes, &mut offset)?
        .parse()
        .map_err(|_| Error::InvalidConfig("invalid NetPBM width".to_owned()))?;
    let height: usize = next_pnm_token(bytes, &mut offset)?
        .parse()
        .map_err(|_| Error::InvalidConfig("invalid NetPBM height".to_owned()))?;
    let max_value: usize = next_pnm_token(bytes, &mut offset)?
        .parse()
        .map_err(|_| Error::InvalidConfig("invalid NetPBM max value".to_owned()))?;
    if max_value != 255 {
        return Err(Error::InvalidConfig(format!(
            "unsupported NetPBM max value {max_value}, expected 255"
        )));
    }
    while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
        offset += 1;
    }

    match magic.as_str() {
        "P6" => {
            let expected = width
                .checked_mul(height)
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(|| Error::InvalidConfig("NetPBM dimensions overflow".to_owned()))?;
            let data = bytes
                .get(offset..offset + expected)
                .ok_or_else(|| Error::InvalidConfig("truncated PPM pixel data".to_owned()))?;
            Ok((width, height, data.to_vec()))
        }
        "P5" => {
            let expected = width
                .checked_mul(height)
                .ok_or_else(|| Error::InvalidConfig("NetPBM dimensions overflow".to_owned()))?;
            let gray = bytes
                .get(offset..offset + expected)
                .ok_or_else(|| Error::InvalidConfig("truncated PGM pixel data".to_owned()))?;
            let mut rgb = Vec::with_capacity(expected * 3);
            for &value in gray {
                rgb.extend_from_slice(&[value, value, value]);
            }
            Ok((width, height, rgb))
        }
        other => Err(Error::InvalidConfig(format!(
            "unsupported NetPBM magic {other}, expected P6 or P5"
        ))),
    }
}

fn next_pnm_token(bytes: &[u8], offset: &mut usize) -> Result<String> {
    loop {
        while *offset < bytes.len() && bytes[*offset].is_ascii_whitespace() {
            *offset += 1;
        }
        if *offset >= bytes.len() {
            return Err(Error::InvalidConfig(
                "unexpected EOF while reading NetPBM header".to_owned(),
            ));
        }
        if bytes[*offset] == b'#' {
            while *offset < bytes.len() && bytes[*offset] != b'\n' {
                *offset += 1;
            }
            continue;
        }
        break;
    }

    let start = *offset;
    while *offset < bytes.len() && !bytes[*offset].is_ascii_whitespace() {
        *offset += 1;
    }
    let token = std::str::from_utf8(&bytes[start..*offset])
        .map_err(|_| Error::InvalidConfig("NetPBM header is not valid UTF-8".to_owned()))?;
    Ok(token.to_owned())
}

fn decode_bmp(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(Error::InvalidConfig("invalid BMP header".to_owned()));
    }

    let pixel_offset = u32::from_le_bytes(bytes[10..14].try_into().expect("slice")) as usize;
    let dib_size = u32::from_le_bytes(bytes[14..18].try_into().expect("slice"));
    if dib_size < 40 {
        return Err(Error::InvalidConfig(format!(
            "unsupported BMP DIB header size {dib_size}"
        )));
    }

    let width = i32::from_le_bytes(bytes[18..22].try_into().expect("slice"));
    let height = i32::from_le_bytes(bytes[22..26].try_into().expect("slice"));
    let planes = u16::from_le_bytes(bytes[26..28].try_into().expect("slice"));
    let bits_per_pixel = u16::from_le_bytes(bytes[28..30].try_into().expect("slice"));
    let compression = u32::from_le_bytes(bytes[30..34].try_into().expect("slice"));

    if planes != 1 || compression != 0 {
        return Err(Error::InvalidConfig(
            "only uncompressed BMP images with one plane are supported".to_owned(),
        ));
    }
    if bits_per_pixel != 24 && bits_per_pixel != 32 {
        return Err(Error::InvalidConfig(format!(
            "unsupported BMP bit depth {bits_per_pixel}, expected 24 or 32"
        )));
    }
    if width <= 0 || height == 0 {
        return Err(Error::InvalidConfig("invalid BMP dimensions".to_owned()));
    }

    let width = width as usize;
    let top_down = height < 0;
    let height = height.unsigned_abs() as usize;
    let bytes_per_pixel = usize::from(bits_per_pixel / 8);
    let row_stride = (width * bytes_per_pixel).div_ceil(4) * 4;
    let image_size = row_stride
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidConfig("BMP dimensions overflow".to_owned()))?;
    let pixels = bytes
        .get(pixel_offset..pixel_offset + image_size)
        .ok_or_else(|| Error::InvalidConfig("truncated BMP pixel data".to_owned()))?;

    let mut rgb = vec![0u8; width * height * 3];
    for row in 0..height {
        let src_row = if top_down { row } else { height - 1 - row };
        let src_base = src_row * row_stride;
        let dst_base = row * width * 3;
        for col in 0..width {
            let src = src_base + col * bytes_per_pixel;
            let dst = dst_base + col * 3;
            rgb[dst] = pixels[src + 2];
            rgb[dst + 1] = pixels[src + 1];
            rgb[dst + 2] = pixels[src];
        }
    }

    Ok((width, height, rgb))
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
