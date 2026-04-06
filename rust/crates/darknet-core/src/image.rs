use crate::error::{Error, Result};
use crate::model::TensorShape;

#[derive(Clone, Debug, PartialEq)]
pub struct PlanarRgbImage {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl PlanarRgbImage {
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
}

#[cfg(test)]
mod tests {
    use super::PlanarRgbImage;

    #[test]
    fn converts_interleaved_rgb_to_planar() {
        let image = PlanarRgbImage::from_interleaved_rgb_u8(1, 1, &[255, 128, 0])
            .expect("image conversion should succeed");
        assert_eq!(image.data, vec![1.0, 128.0 / 255.0, 0.0]);
    }
}
