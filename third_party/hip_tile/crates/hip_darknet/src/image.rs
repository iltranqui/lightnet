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

    /// Resize RGB input with OpenCV `INTER_LINEAR_EXACT`'s half-pixel sampling and fixed-point
    /// coefficients, matching lightnet's layer-dump preprocessing.
    pub fn resize_linear_exact(&self, w: usize, h: usize) -> Image {
        if self.w == w && self.h == h {
            return Image { w, h, c: self.c, data: self.data.clone() };
        }
        let exact_coeffs = |src: usize, dst: usize| {
            (0..dst)
                .map(|d| {
                    let fs = (d as f64 + 0.5) * src as f64 / dst as f64 - 0.5;
                    let mut x0 = fs.floor();
                    let mut frac = fs - x0;
                    if x0 < 0.0 {
                        x0 = 0.0;
                        frac = 0.0;
                    }
                    if x0 >= src as f64 - 1.0 {
                        x0 = src as f64 - 1.0;
                        frac = 0.0;
                    }
                    let x0 = x0 as usize;
                    let c1 = (frac * 256.0 + 0.5).floor() as u32;
                    (x0, (x0 + 1).min(src - 1), 256 - c1, c1)
                })
                .collect::<Vec<_>>()
        };
        let (cx, cy) = (exact_coeffs(self.w, w), exact_coeffs(self.h, h));
        let src: Vec<u8> = self.data.iter().map(|&v| (v * 255.0).round() as u8).collect();
        let mut tmp = vec![0u32; self.h * w * self.c];
        for y in 0..self.h {
            for (x, &(x0, x1, c0, c1)) in cx.iter().enumerate() {
                for ch in 0..self.c {
                    let a = src[(ch * self.h + y) * self.w + x0] as u32;
                    let b = src[(ch * self.h + y) * self.w + x1] as u32;
                    tmp[(ch * self.h + y) * w + x] = a * c0 + b * c1;
                }
            }
        }
        let mut out = vec![0f32; self.c * h * w];
        for (y, &(y0, y1, c0, c1)) in cy.iter().enumerate() {
            for ch in 0..self.c {
                for x in 0..w {
                    let a = tmp[(ch * self.h + y0) * w + x];
                    let b = tmp[(ch * self.h + y1) * w + x];
                    let v = ((a * c0 + b * c1 + 32768) >> 16) as u8;
                    out[(ch * h + y) * w + x] = v as f32 / 255.0;
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

    #[test]
    fn resize_linear_exact_fixed_point_rgb() {
        // The fixed-point result stays within one byte of an f64 half-pixel bilinear resize
        // (centres at (d + 0.5) * src / dst - 0.5, edges clamped).
        let (sw, sh, dw, dh) = (5usize, 4usize, 3usize, 3usize);
        let rgb: Vec<u8> = (0..sw * sh).flat_map(|v| [(v * 7 % 256) as u8, (v * 13 % 256) as u8, (v * 29 % 256) as u8]).collect();
        let img = Image::from_rgb8(&rgb, sw, sh);
        let resized = img.resize_linear_exact(dw, dh);
        let at = |c: usize, y: usize, x: usize| f64::from(rgb[(y * sw + x) * 3 + c]);
        let axis = |d: usize, src: usize, dst: usize| {
            let f = ((d as f64 + 0.5) * src as f64 / dst as f64 - 0.5).clamp(0.0, src as f64 - 1.0);
            let i = (f.floor() as usize).min(src - 1);
            (i, (i + 1).min(src - 1), f - i as f64)
        };
        for c in 0..3 {
            for y in 0..dh {
                for x in 0..dw {
                    let (y0, y1, fy) = axis(y, sh, dh);
                    let (x0, x1, fx) = axis(x, sw, dw);
                    let top = at(c, y0, x0) * (1.0 - fx) + at(c, y0, x1) * fx;
                    let bot = at(c, y1, x0) * (1.0 - fx) + at(c, y1, x1) * fx;
                    let want = top * (1.0 - fy) + bot * fy;
                    let got = f64::from((resized.data[(c * dh + y) * dw + x] * 255.0).round());
                    assert!((got - want).abs() <= 1.0, "c{c} y{y} x{x}: got {got} want {want}");
                }
            }
        }
    }

    #[test]
    fn resize_linear_exact_same_size_copies_input() {
        let img = Image::from_rgb8(&[0, 1, 2, 3, 4, 5], 2, 1);
        let resized = img.resize_linear_exact(2, 1);
        assert_eq!(resized.data, img.data);
        assert_eq!((resized.w, resized.h, resized.c), (img.w, img.h, img.c));
    }
}
