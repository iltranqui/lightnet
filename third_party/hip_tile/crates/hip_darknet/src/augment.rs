//! Training augmentation: Darknet's crop/resize jitter, flip, and HSV distortion.
//!
//! Geometry (crop offsets, resize range) and box correction follow `data.cpp:1510-1690` and
//! `correct_boxes`; the colour path follows Darknet's f32 `distort_image`
//! (`darknet_image.cpp:1587-1615`). The OpenCV 8-bit HSV path that `load_data_augment` actually
//! uses is a recorded deviation (`_shared/darknet-semantics.md`, Training data and augmentation).

use crate::data::BoxLabel;
use crate::image::Image;

/// Small deterministic PRNG (xorshift64*), so a run is reproducible from its seed alone.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Darknet `rand_uniform(a, b)`.
    pub fn uniform(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }

    pub fn boolean(&mut self) -> bool {
        self.next_u64() & 1 != 0
    }

    /// Darknet `rand_scale`: `rand_uniform(1, s)` with a coin flip to invert it
    /// (`utils.cpp:1223-1235`). `s == 1` always returns 1.
    pub fn scale(&mut self, s: f32) -> f32 {
        if s == 1.0 {
            return 1.0;
        }
        let v = self.uniform(1.0, s);
        if self.boolean() { 1.0 / v } else { v }
    }

    /// Uniform integer in `[min, max]` inclusive, like Darknet's `rand_int`.
    pub fn int(&mut self, min: i32, max: i32) -> i32 {
        if max <= min {
            return min;
        }
        min + (self.next_u64() % (max - min + 1) as u64) as i32
    }
}

/// Augmentation settings, all lifted from the cfg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Augment {
    pub jitter: f32,
    pub resize: f32,
    pub flip: bool,
    pub hue: f32,
    pub saturation: f32,
    pub exposure: f32,
}

impl Default for Augment {
    fn default() -> Self {
        Augment { jitter: 0.0, resize: 1.0, flip: false, hue: 0.0, saturation: 1.0, exposure: 1.0 }
    }
}

impl Augment {
    /// True when every transformation is the identity, so `prepare` can take the fast path.
    pub fn is_identity(&self) -> bool {
        self.jitter == 0.0
            && self.resize == 1.0
            && !self.flip
            && self.hue == 0.0
            && self.saturation == 1.0
            && self.exposure == 1.0
    }
}

/// Crop rectangle in source pixels, `correct_boxes` transform factors, and the flip flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub flip: bool,
    /// `correct_boxes` factors: `left' = left * scale_x - shift_x`.
    pub scale_x: f32,
    pub scale_y: f32,
    pub shift_x: f32,
    pub shift_y: f32,
}

impl Geometry {
    /// Identity geometry for a source image that is already the network size.
    pub fn identity(w: u32, h: u32) -> Geometry {
        Geometry { left: 0, top: 0, width: w.max(1), height: h.max(1), flip: false, scale_x: 1.0, scale_y: 1.0, shift_x: 0.0, shift_y: 0.0 }
    }

    /// Darknet's `data.cpp:1510-1690` geometry: jitter crop, resize range, optional letterbox-free crop.
    pub fn sample(rng: &mut Rng, ow: u32, oh: u32, aug: &Augment) -> Geometry {
        let (owf, ohf) = (ow as f32, oh as f32);
        let dw = owf * aug.jitter;
        let dh = ohf * aug.jitter;

        let resize_down = if aug.resize > 1.0 { 1.0 / aug.resize } else { aug.resize };
        let resize_up = if aug.resize < 1.0 { 1.0 / aug.resize } else { aug.resize };
        // min_rdw/min_rdh are negative, max_rdw/max_rdh positive.
        let min_rdw = owf * (1.0 - 1.0 / resize_down) / 2.0;
        let min_rdh = ohf * (1.0 - 1.0 / resize_down) / 2.0;
        let max_rdw = owf * (1.0 - 1.0 / resize_up) / 2.0;
        let max_rdh = ohf * (1.0 - 1.0 / resize_up) / 2.0;

        let mut pleft = rng.uniform(-dw, dw) as i32;
        let mut pright = rng.uniform(-dw, dw) as i32;
        let mut ptop = rng.uniform(-dh, dh) as i32;
        let mut pbot = rng.uniform(-dh, dh) as i32;
        if aug.resize < 1.0 {
            pleft += rng.uniform(min_rdw, 0.0) as i32;
            pright += rng.uniform(min_rdw, 0.0) as i32;
            ptop += rng.uniform(min_rdh, 0.0) as i32;
            pbot += rng.uniform(min_rdh, 0.0) as i32;
        } else {
            pleft += rng.uniform(min_rdw, max_rdw) as i32;
            pright += rng.uniform(min_rdw, max_rdw) as i32;
            ptop += rng.uniform(min_rdh, max_rdh) as i32;
            pbot += rng.uniform(min_rdh, max_rdh) as i32;
        }

        let dwidth = (ow as i32 - pleft - pright).max(1);
        let dheight = (oh as i32 - ptop - pbot).max(1);
        let sx = dwidth as f32 / owf;
        let sy = dheight as f32 / ohf;
        Geometry {
            left: pleft,
            top: ptop,
            width: dwidth as u32,
            height: dheight as u32,
            flip: aug.flip && rng.boolean(),
            scale_x: 1.0 / sx,
            scale_y: 1.0 / sy,
            shift_x: (pleft as f32 / owf) / sx,
            shift_y: (ptop as f32 / ohf) / sy,
        }
    }

    /// Darknet `correct_boxes` (`data.cpp:828-908`) for one box, including the
    /// "smaller than one pixel" and "outside the crop" drops handled by `fill_truth_detection`.
    pub fn apply_box(&self, b: &BoxLabel, net_w: u32, net_h: u32) -> Option<BoxLabel> {
        let (mut left, mut right) = (b.x - b.w / 2.0, b.x + b.w / 2.0);
        let (mut top, mut bottom) = (b.y - b.h / 2.0, b.y + b.h / 2.0);
        // Entirely outside the crop: Darknet drops it inside `correct_boxes`.
        if right < 0.0 || bottom < 0.0 || left > 1.0 || top > 1.0 {
            return None;
        }
        left = left * self.scale_x - self.shift_x;
        right = right * self.scale_x - self.shift_x;
        top = top * self.scale_y - self.shift_y;
        bottom = bottom * self.scale_y - self.shift_y;
        if self.flip {
            let swapped = 1.0 - left;
            left = 1.0 - right;
            right = swapped;
        }
        let clamp = |v: f32| v.clamp(0.0, 1.0);
        let (left, right, top, bottom) = (clamp(left), clamp(right), clamp(top), clamp(bottom));
        let mut w = (right - left).clamp(0.0, 1.0);
        let mut h = (bottom - top).clamp(0.0, 1.0);
        let mut x = (left + right) / 2.0;
        let mut y = (top + bottom) / 2.0;
        // `fill_truth_detection`: drop sub-pixel boxes, then clamp centers into (0, 1].
        let lowest_w = 1.0 / net_w as f32;
        let lowest_h = 1.0 / net_h as f32;
        if w < lowest_w || h < lowest_h {
            return None;
        }
        if x <= 0.0 {
            x += lowest_w;
        }
        if y <= 0.0 {
            y += lowest_h;
        }
        if x >= 1.0 {
            x = 1.0 - lowest_w;
        }
        if y >= 1.0 {
            y = 1.0 - lowest_h;
        }
        // Keep widths/heights inside the image so the anchor math cannot produce NaN.
        w = w.min(1.0);
        h = h.min(1.0);
        Some(BoxLabel { class: b.class, x, y, w, h })
    }
}

/// Crop to `geom` with the crop's mean colour outside the image, then resize to the net size
/// (`image_opencv.cpp:308-340`).
pub fn crop_resize(img: &Image, geom: &Geometry, net_w: u32, net_h: u32) -> Image {
    let (w, h) = (geom.width as usize, geom.height as usize);
    let mut cropped = vec![0f32; 3 * w * h];
    let mut sum = [0f64; 3];
    let mut count = 0usize;
    for yy in 0..h {
        let sy = geom.top + yy as i32;
        for xx in 0..w {
            let sx = geom.left + xx as i32;
            if sx >= 0 && sy >= 0 && (sx as u32) < img.w as u32 && (sy as u32) < img.h as u32 {
                let src = (sy as usize * img.w) + sx as usize;
                for c in 0..3 {
                    let v = img.data[c * img.w * img.h + src];
                    cropped[(c * h + yy) * w + xx] = v;
                    sum[c] += v as f64;
                }
                count += 1;
            }
        }
    }
    // Mean-colour fill, like `cropped.setTo(cv::mean(mat))`.
    let mean = if count > 0 { sum.map(|s| (s / count as f64) as f32) } else { [0.0; 3] };
    let mut mean_pixels = 0usize;
    for yy in 0..h {
        let sy = geom.top + yy as i32;
        for xx in 0..w {
            let sx = geom.left + xx as i32;
            let inside = sx >= 0 && sy >= 0 && (sx as u32) < img.w as u32 && (sy as u32) < img.h as u32;
            if !inside {
                for c in 0..3 {
                    cropped[(c * h + yy) * w + xx] = mean[c];
                }
                mean_pixels += 1;
            }
        }
    }
    let _ = mean_pixels;
    let part = Image { w, h, c: 3, data: cropped };
    let mut out = part.resize(net_w as usize, net_h as usize);
    if geom.flip {
        flip_horizontal(&mut out);
    }
    out
}

fn flip_horizontal(img: &mut Image) {
    let (w, h) = (img.w, img.h);
    for c in 0..img.c {
        let plane = &mut img.data[c * w * h..(c + 1) * w * h];
        for y in 0..h {
            let row = &mut plane[y * w..(y + 1) * w];
            row.reverse();
        }
    }
}

/// Darknet `distort_image` (`darknet_image.cpp:1587-1615`): RGB→HSV, scale S and V, shift H, back.
pub fn distort(img: &mut Image, dhue: f32, dsat: f32, dexp: f32) {
    if img.c < 3 {
        for v in img.data.iter_mut() {
            *v = (*v * dexp).clamp(0.0, 1.0);
        }
        return;
    }
    let n = img.w * img.h;
    for i in 0..n {
        let (r, g, b) = (img.data[i], img.data[n + i], img.data[2 * n + i]);
        let (mut hh, mut ss, vv) = rgb_to_hsv(r, g, b);
        ss = (ss * dsat).clamp(0.0, 1.0);
        let vv = (vv * dexp).clamp(0.0, 1.0);
        hh += dhue;
        if hh > 1.0 {
            hh -= 1.0;
        }
        if hh < 0.0 {
            hh += 1.0;
        }
        let (r, g, b) = hsv_to_rgb(hh, ss, vv);
        img.data[i] = r.clamp(0.0, 1.0);
        img.data[n + i] = g.clamp(0.0, 1.0);
        img.data[2 * n + i] = b.clamp(0.0, 1.0);
    }
}

/// Darknet's `rgb_to_hsv` (`darknet_image.cpp:1417-1458`); hue in `[0, 1)`.
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    if max == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let s = delta / max;
    let mut h = if r == max {
        (g - b) / delta
    } else if g == max {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    if h < 0.0 {
        h += 6.0;
    }
    (h / 6.0, s, v)
}

/// Darknet's `hsv_to_rgb` (`darknet_image.cpp:1461-1504`).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = 6.0 * h;
    if s == 0.0 {
        return (v, v, v);
    }
    let index = h.floor();
    let f = h - index;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match index as i32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// One prepared training example: CHW f32 pixels at the network size plus corrected boxes.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub data: Vec<f32>,
    pub labels: Vec<BoxLabel>,
}

/// Apply the full augmentation chain to one sample (`load_data_augment` order: crop/resize → flip →
/// colour), returning network-sized CHW pixels and the boxes transformed with them.
pub fn prepare(rng: &mut Rng, img: &Image, labels: &[BoxLabel], net_w: u32, net_h: u32, aug: &Augment) -> Prepared {
    let geom = if aug.is_identity() {
        Geometry::identity(img.w as u32, img.h as u32)
    } else {
        Geometry::sample(rng, img.w as u32, img.h as u32, aug)
    };
    let mut sized = if aug.is_identity() {
        img.resize(net_w as usize, net_h as usize)
    } else {
        crop_resize(img, &geom, net_w, net_h)
    };
    if !aug.is_identity() {
        let dhue = rng.uniform(-aug.hue, aug.hue);
        let dsat = rng.scale(aug.saturation);
        let dexp = rng.scale(aug.exposure);
        if dhue != 0.0 || dsat != 1.0 || dexp != 1.0 {
            distort(&mut sized, dhue, dsat, dexp);
        }
    }
    let out_labels = labels.iter().filter_map(|b| geom.apply_box(b, net_w, net_h)).collect();
    Prepared { data: sized.data, labels: out_labels }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_bounded() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.unit();
            assert_eq!(x, b.unit());
            assert!((0.0..1.0).contains(&x));
        }
        let mut r = Rng::new(1);
        for _ in 0..100 {
            let s = r.scale(1.5);
            assert!(s > 0.0, "{s}");
            assert!((1.0 / 1.5..=1.0 / (1.0 / 1.5) + 1e-6).contains(&s), "{s}");
        }
        assert_eq!(r.scale(1.0), 1.0);
    }

    #[test]
    fn identity_geometry_keeps_boxes() {
        let g = Geometry::identity(224, 160);
        let b = BoxLabel { class: 2, x: 0.4, y: 0.6, w: 0.2, h: 0.1 };
        let out = g.apply_box(&b, 224, 160).unwrap();
        assert_eq!(out.class, b.class);
        for (a, e) in [(out.x, b.x), (out.y, b.y), (out.w, b.w), (out.h, b.h)] {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }
    }

    #[test]
    fn flip_mirrors_boxes() {
        let g = Geometry { left: 0, top: 0, width: 224, height: 160, flip: true, scale_x: 1.0, scale_y: 1.0, shift_x: 0.0, shift_y: 0.0 };
        let b = BoxLabel { class: 0, x: 0.25, y: 0.5, w: 0.1, h: 0.2 };
        let f = g.apply_box(&b, 224, 160).unwrap();
        assert!((f.x - 0.75).abs() < 1e-6);
        assert!((f.w - 0.1).abs() < 1e-6 && (f.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn crop_zoom_tracks_boxes() {
        // Crop the left half (x in [0, 0.5]) and zoom it back to full frame: a box at x=0.25 stays
        // at 0.25, one at 0.45 moves to 0.9, one at 0.6 disappears.
        let g = Geometry { left: 0, top: 0, width: 112, height: 160, flip: false, scale_x: 2.0, scale_y: 1.0, shift_x: 0.0, shift_y: 0.0 };
        let b = |x: f32| BoxLabel { class: 1, x, y: 0.5, w: 0.1, h: 0.1 };
        assert!((g.apply_box(&b(0.25), 224, 160).unwrap().x - 0.5).abs() < 1e-6);
        assert!((g.apply_box(&b(0.1), 224, 160).unwrap().x - 0.2).abs() < 1e-6);
        assert!(g.apply_box(&b(0.6), 224, 160).is_none());
    }

    #[test]
    fn sub_pixel_boxes_are_dropped() {
        let g = Geometry::identity(224, 160);
        let tiny = BoxLabel { class: 0, x: 0.5, y: 0.5, w: 1.0 / 400.0, h: 0.5 };
        assert!(g.apply_box(&tiny, 224, 160).is_none());
    }

    fn rect_image() -> Image {
        // 8x8 black with a white 2x2 block at (1..3, 1..3).
        let mut data = vec![0f32; 3 * 64];
        for c in 0..3 {
            for y in 1..3 {
                for x in 1..3 {
                    data[c * 64 + y * 8 + x] = 1.0;
                }
            }
        }
        Image { w: 8, h: 8, c: 3, data }
    }

    fn bright_centroid(img: &Image) -> (f32, f32) {
        let n = img.w * img.h;
        let (mut sx, mut sy, mut tot) = (0.0, 0.0, 0.0);
        for y in 0..img.h {
            for x in 0..img.w {
                let v = img.data[y * img.w + x]; // red plane
                sx += v * x as f32;
                sy += v * y as f32;
                tot += v;
            }
        }
        let _ = n;
        (sx / tot, sy / tot)
    }

    #[test]
    fn flip_moves_the_pixels_too() {
        let img = rect_image();
        let geom = Geometry { left: 0, top: 0, width: 8, height: 8, flip: true, scale_x: 1.0, scale_y: 1.0, shift_x: 0.0, shift_y: 0.0 };
        // Network size = source size, so the resize is a no-op and only the flip moves pixels.
        let out = crop_resize(&img, &geom, 8, 8);
        let (cx, _) = bright_centroid(&out);
        // The block spans pixel columns 1..3 (centre 1.5); reversing an 8-pixel row maps index i to
        // 7 - i, so the centre lands at 5.5 (and `correct_boxes` mirrors the box about 0.5).
        assert!((cx - 5.5).abs() < 0.2, "centroid {cx}");
    }

    #[test]
    fn distort_round_trips_hue_and_scales_value() {
        let mut img = Image { w: 1, h: 1, c: 3, data: vec![0.5, 0.25, 0.125] };
        let before = img.data.clone();
        distort(&mut img, 0.0, 1.0, 1.0);
        for (a, b) in before.iter().zip(&img.data) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        let mut img = Image { w: 1, h: 1, c: 3, data: vec![0.4, 0.2, 0.1] };
        distort(&mut img, 0.0, 1.0, 0.5);
        assert!((img.data[0] - 0.2).abs() < 1e-6);
        // Hue shift preserves value and saturation, so the max channel is unchanged.
        let mut img = Image { w: 1, h: 1, c: 3, data: vec![0.4, 0.2, 0.1] };
        distort(&mut img, 0.5, 1.0, 1.0);
        assert!((img.data.iter().cloned().fold(0.0, f32::max) - 0.4).abs() < 1e-5);
    }

    #[test]
    fn prepare_is_deterministic_for_a_seed() {
        let img = rect_image();
        let labels = vec![BoxLabel { class: 0, x: 0.4, y: 0.4, w: 0.2, h: 0.2 }];
        let aug = Augment { jitter: 0.3, resize: 1.5, flip: true, hue: 0.1, saturation: 1.5, exposure: 1.5 };
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        let a = prepare(&mut r1, &img, &labels, 16, 16, &aug);
        let b = prepare(&mut r2, &img, &labels, 16, 16, &aug);
        assert_eq!(a.data, b.data);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.data.len(), 3 * 16 * 16);
    }
}
