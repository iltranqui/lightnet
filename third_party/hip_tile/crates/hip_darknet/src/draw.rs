//! Minimal annotation: class-colored rectangles drawn on an image and saved with the `image` crate.

use crate::error::{Error, Result};
use crate::image::Image;
use crate::yolo::Detection;
use std::path::Path;

const PALETTE: [[u8; 3]; 8] = [
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
];

/// Draw each detection's box (2 px, class color) on `img` and write it to `path` (format by extension).
pub fn save_annotated(img: &Image, dets: &[Detection], path: impl AsRef<Path>) -> Result<()> {
    let p = path.as_ref();
    let (w, h) = (img.w, img.h);
    let mut buf = vec![0u8; w * h * 3];
    for i in 0..w * h {
        for c in 0..3 {
            buf[i * 3 + c] = (img.data[c * w * h + i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    for d in dets {
        let color = PALETTE[d.class % PALETTE.len()];
        let (l, t, bw, bh) = d.pixel_rect(w, h);
        let clamp = |v: f32, max: usize| (v.round().max(0.0) as usize).min(max - 1);
        let (x0, x1, y0, y1) = (clamp(l, w), clamp(l + bw, w), clamp(t, h), clamp(t + bh, h));
        for thick in 0..2usize {
            let mut put = |x: usize, y: usize| buf[(y * w + x) * 3..][..3].copy_from_slice(&color);
            for x in x0..=x1 {
                put(x, (y0 + thick).min(h - 1));
                put(x, y1.saturating_sub(thick));
            }
            for y in y0..=y1 {
                put((x0 + thick).min(w - 1), y);
                put(x1.saturating_sub(thick), y);
            }
        }
    }
    let out = image::RgbImage::from_raw(w as u32, h as u32, buf).expect("buffer size");
    out.save(p).map_err(|source| Error::Image { path: p.display().to_string(), source })
}
