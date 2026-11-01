//! Image loading in Darknet's input convention: RGB, planar CHW f32 in [0, 1].

use crate::error::{Error, Result};
use std::path::Path;

/// Decoded image as planar CHW f32 / 255.
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn from_rgb8(rgb: &[u8], w: usize, h: usize) -> Image {
        assert_eq!(rgb.len(), w * h * 3);
        let mut data = vec![0f32; w * h * 3];
        for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in px.iter().enumerate() {
                data[c * w * h + i] = v as f32 / 255.0;
            }
        }
        Image { w, h, c: 3, data }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Image> {
        let p = path.as_ref();
        let img = image::open(p)
            .map_err(|source| Error::Image { path: p.display().to_string(), source })?
            .to_rgb8();
        Ok(Image::from_rgb8(img.as_raw(), img.width() as usize, img.height() as usize))
    }

    /// Resize to `w` x `h` like Darknet's `resize_image` (separable bilinear with corner-aligned
    /// sampling; returns a copy when the size already matches).
    pub fn resize(&self, w: usize, h: usize) -> Image {
        if self.w == w && self.h == h {
            return Image { w, h, c: self.c, data: self.data.clone() };
        }
        let at = |c: usize, y: usize, x: usize| self.data[(c * self.h + y) * self.w + x];
        // Horizontal pass into a `w` x `self.h` plane.
        let w_scale = if w > 1 { (self.w as f32 - 1.0) / (w as f32 - 1.0) } else { 0.0 };
        let h_scale = if h > 1 { (self.h as f32 - 1.0) / (h as f32 - 1.0) } else { 0.0 };
        let mut part = vec![0f32; self.c * self.h * w];
        for c in 0..self.c {
            for y in 0..self.h {
                for x in 0..w {
                    part[(c * self.h + y) * w + x] = if x == w - 1 || self.w == 1 {
                        at(c, y, self.w - 1)
                    } else {
                        let sx = x as f32 * w_scale;
                        let ix = sx as usize;
                        let dx = sx - ix as f32;
                        (1.0 - dx) * at(c, y, ix) + dx * at(c, y, ix + 1)
                    };
                }
            }
        }
        let mut out = vec![0f32; self.c * h * w];
        for c in 0..self.c {
            for y in 0..h {
                let sy = y as f32 * h_scale;
                let iy = sy as usize;
                let dy = sy - iy as f32;
                for x in 0..w {
                    let mut v = (1.0 - dy) * part[(c * self.h + iy) * w + x];
                    if !(y == h - 1 || self.h == 1) {
                        v += dy * part[(c * self.h + iy + 1) * w + x];
                    }
                    out[(c * h + y) * w + x] = v;
                }
            }
        }
        Image { w, h, c: self.c, data: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_planar() {
        let img = Image::from_rgb8(&[255, 0, 51, 0, 255, 102], 2, 1);
        assert_eq!(img.data, vec![1.0, 0.0, 0.0, 1.0, 0.2, 0.4]);
    }

    #[test]
    fn resize_corner_aligned() {
        // 2x1 gradient 0 -> 1 upscaled to 3x1: samples at 0, .5, 1.
        let img = Image { w: 2, h: 1, c: 1, data: vec![0.0, 1.0] };
        let r = img.resize(3, 1);
        assert_eq!(r.data, vec![0.0, 0.5, 1.0]);
        assert_eq!(img.resize(2, 1).data, img.data);
    }
}
