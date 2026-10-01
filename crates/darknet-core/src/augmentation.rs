//! Seeded detector augmentation matching the C++ crop/box and u8 HSV path.
use crate::{ConfigFile, Error, PlanarRgbImage, Result, YoloLabel};

#[derive(Clone, Debug)]
pub struct DetectorAugmentation {
    pub jitter: f32,
    pub resize: f32,
    pub flip: bool,
    pub letter_box: bool,
    pub hue: f32,
    pub saturation: f32,
    pub exposure: f32,
    pub max_boxes: usize,
    /// C++ `[net] mixup`: blend two independently augmented samples 50/50
    /// and concatenate their truth lists (mode 1). Cutmix (2/4) is rejected by
    /// the C++ detector loader and mosaic (3) is not implemented here yet.
    pub mixup: bool,
    /// Max `random` across detector heads (`0` disables multi-scale).
    pub random: f32,
    /// C++ `[net] resize_step`; multi-scale dimensions are rounded to it.
    pub resize_step: usize,
}

impl DetectorAugmentation {
    /// Parse once at trainer startup. Reject active, unimplemented augmentation
    /// rather than silently training with a different data distribution.
    pub fn from_config(cfg: &ConfigFile) -> Result<Self> {
        let net = cfg.sections.first().ok_or_else(|| invalid("empty cfg"))?;
        let head = cfg
            .sections
            .iter()
            .rev()
            .find(|s| matches!(s.name.as_str(), "yolo" | "gaussian_yolo" | "region"))
            .ok_or_else(|| invalid("detector augmentation requires a detector head"))?;
        for key in ["mosaic", "cutmix", "blur", "gaussian_noise", "fog", "track", "contrastive"] {
            if net.get_f32(key, 0.0)? != 0.0 {
                return Err(invalid(&format!(
                    "training augmentation {key} is not implemented yet"
                )));
            }
        }
        let mut random = 0.0f32;
        for section in &cfg.sections {
            if matches!(section.name.as_str(), "yolo" | "gaussian_yolo" | "region") {
                random = random.max(section.get_f32("random", 0.0)?);
            }
        }
        let options = Self {
            jitter: head.get_f32("jitter", 0.2)?,
            resize: head.get_f32("resize", 1.0)?,
            flip: net.get_bool_int("flip", true)?,
            letter_box: net.get_bool_int("letter_box", false)?,
            hue: net.get_f32("hue", 0.0)?,
            saturation: net.get_f32("saturation", 1.0)?,
            exposure: net.get_f32("exposure", 1.0)?,
            max_boxes: head.get_usize("max", 200)?,
            mixup: net.get_f32("mixup", 0.0)? != 0.0,
            random,
            resize_step: net.get_usize("resize_step", 32)?,
        };
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<()> {
        if !self.jitter.is_finite()
            || !(0.0..1.0).contains(&self.jitter)
            || !self.resize.is_finite()
            || self.resize <= 0.0
            || !self.hue.is_finite()
            || !(0.0..=1.0).contains(&self.hue)
            || !self.saturation.is_finite()
            || self.saturation <= 0.0
            || !self.exposure.is_finite()
            || self.exposure <= 0.0
            || self.max_boxes == 0
            || !self.random.is_finite()
            || self.random < 0.0
            || self.resize_step == 0
        {
            return Err(invalid(
                "invalid jitter/resize/hue/saturation/exposure/max augmentation settings",
            ));
        }
        Ok(())
    }

    /// C++ `train_detector` multi-scale: every 10 iterations pick a new size
    /// derived from `rand_scale(random)` rounded to `resize_step`.  The first
    /// iteration and the last 100 iterations use the maximum size so rolling
    /// statistics are measured on a consistent shape.
    pub fn multi_scale_dimensions(
        &self,
        base_width: usize,
        base_height: usize,
        seed: u64,
        iteration: usize,
        max_batches: usize,
    ) -> (usize, usize) {
        if self.random <= 0.0 {
            return (base_width, base_height);
        }
        let rand_coef = if self.random != 1.0 { self.random } else { 1.4 };
        let step = self.resize_step.max(1);
        let mut rng = Rng(seed ^ (iteration as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let random_val = rng.scale(rand_coef);
        let dim = |base: usize| -> i64 {
            (random_val * base as f32 / step as f32 + 1.0).round() as i64 * step as i64
        };
        let mut dim_w = dim(base_width);
        let mut dim_h = dim(base_height);
        if random_val < 1.0 && (dim_w > base_width as i64 || dim_h > base_height as i64) {
            dim_w = base_width as i64;
            dim_h = base_height as i64;
        }
        let max_dim = |base: usize| -> i64 {
            (rand_coef * base as f32 / step as f32 + 1.0).round() as i64 * step as i64
        };
        if iteration == 0 || (max_batches != 0 && iteration > max_batches.saturating_sub(100)) {
            dim_w = max_dim(base_width);
            dim_h = max_dim(base_height);
        }
        (
            dim_w.max(step as i64).max(1) as usize,
            dim_h.max(step as i64).max(1) as usize,
        )
    }

    /// Independent random stream keyed by seed and absolute sample ordinal.
    /// Scheduling/prefetch order cannot change a sample's transform.
    pub fn apply(
        &self,
        image: &PlanarRgbImage,
        labels: &[YoloLabel],
        width: usize,
        height: usize,
        seed: u64,
        sample: u64,
    ) -> Result<(PlanarRgbImage, Vec<YoloLabel>)> {
        self.validate()?;
        if width == 0
            || height == 0
            || image.shape.width == 0
            || image.shape.height == 0
            || image.shape.channels != 3
            || image.shape.batch != 1
        {
            return Err(invalid(
                "augmentation requires nonempty single RGB image and output",
            ));
        }
        let mut rng = Rng(seed ^ sample.wrapping_mul(0xd6e8feb86659fd93));
        // Keep the C++ correlations: same resize draws for x/y margins.
        let rr = [rng.unit(), rng.unit()];
        let jr = [rng.unit(), rng.unit(), rng.unit(), rng.unit()];
        let flip = self.flip && rng.unit() >= 0.5;
        let hue = (2.0 * rng.unit() - 1.0) * self.hue;
        let sat = rng.scale(self.saturation);
        let exp = rng.scale(self.exposure);
        let ow = i32::try_from(image.shape.width)
            .map_err(|_| invalid("source width overflows crop geometry"))?;
        let oh = i32::try_from(image.shape.height)
            .map_err(|_| invalid("source height overflows crop geometry"))?;
        let mut margins = [0; 4];
        for (i, margin) in margins.iter_mut().enumerate() {
            let size = if i < 2 { ow } else { oh };
            let jitter = (size as f32 * self.jitter) as i32;
            let down = self.resize.min(1.0 / self.resize);
            let up = self.resize.max(1.0 / self.resize);
            let min = (size as f32 * (1.0 - 1.0 / down) / 2.0) as i32;
            let max = if self.resize < 1.0 {
                0
            } else {
                (size as f32 * (1.0 - 1.0 / up) / 2.0) as i32
            };
            *margin = precalc(-jitter, jitter, jr[i])
                .checked_add(precalc(min, max, rr[i % 2]))
                .ok_or_else(|| invalid("crop margin overflow"))?;
        }
        if self.letter_box {
            let net_ar = width as f32 / height as f32;
            if ow as f32 / oh as f32 > net_ar {
                let delta = (ow as f32 / net_ar - oh as f32) / 2.0;
                margins[2] = (margins[2] as f32 - delta) as i32;
                margins[3] = (margins[3] as f32 - delta) as i32;
            } else {
                let delta = (oh as f32 * net_ar - ow as f32) / 2.0;
                margins[0] = (margins[0] as f32 - delta) as i32;
                margins[1] = (margins[1] as f32 - delta) as i32;
            }
        }
        let crop_width = ow
            .checked_sub(margins[0])
            .and_then(|v| v.checked_sub(margins[1]))
            .ok_or_else(|| invalid("crop width overflow"))?;
        let crop_height = oh
            .checked_sub(margins[2])
            .and_then(|v| v.checked_sub(margins[3]))
            .ok_or_else(|| invalid("crop height overflow"))?;
        let crop = Crop {
            left: margins[0],
            top: margins[2],
            width: crop_width,
            height: crop_height,
            flip,
        };
        if crop.width <= 0 || crop.height <= 0 {
            return Err(invalid(
                "augmentation produced an empty crop; reduce jitter/resize",
            ));
        }
        let mut shuffled = labels.to_vec();
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        shuffled.truncate(self.max_boxes); // C++ caps before filtering.
        let labels = crop.labels(&shuffled, ow, oh, width, height);
        let mut out = crop.image(image, width, height)?;
        for pixel in out.chunks_exact_mut(3) {
            hsv_distort(pixel, hue, sat, exp);
        }
        Ok((
            PlanarRgbImage::from_interleaved_rgb_u8(width, height, &out)?,
            labels,
        ))
    }
}

fn invalid(message: &str) -> Error {
    Error::InvalidConfig(message.into())
}

/// True when this batch should use mixup.  C++ samples `rand_bool()` once per
/// `load_data_detection` call and disables mixing for the whole batch 50% of
/// the time.
pub fn mixup_active(seed: u64, sample_start: u64) -> bool {
    let mut rng = Rng(seed ^ sample_start.wrapping_mul(0x9e3779b97f4a7c15));
    rng.unit() >= 0.5
}

/// C++ `blend_images_cv(a, 0.5, b, 0.5)` + `blend_truth`: 50/50 pixel blend
/// and append the second truth list after the first, capped at `max_boxes`.
pub fn blend_mixup(
    first: &PlanarRgbImage,
    first_labels: &[YoloLabel],
    second: &PlanarRgbImage,
    second_labels: &[YoloLabel],
    max_boxes: usize,
) -> Result<(PlanarRgbImage, Vec<YoloLabel>)> {
    if first.shape.width != second.shape.width
        || first.shape.height != second.shape.height
        || first.shape.channels != second.shape.channels
    {
        return Err(invalid("mixup requires equal-shaped augmented images"));
    }
    let a = first.to_rgb8()?;
    let b = second.to_rgb8()?;
    let mut out = vec![0u8; a.as_raw().len()];
    for ((dst, x), y) in out.iter_mut().zip(a.as_raw()).zip(b.as_raw()) {
        *dst = ((*x as u32 + *y as u32 + 1) / 2) as u8;
    }
    let mut labels = first_labels.to_vec();
    for label in second_labels {
        if labels.len() >= max_boxes {
            break;
        }
        labels.push(label.clone());
    }
    Ok((
        PlanarRgbImage::from_interleaved_rgb_u8(first.shape.width, first.shape.height, &out)?,
        labels,
    ))
}
fn precalc(min: i32, max: i32, unit: f32) -> i32 {
    (unit * (max as f32 - min as f32) + min as f32) as i32
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16777216.0
    }
    fn scale(&mut self, max: f32) -> f32 {
        let value = 1.0 + self.unit() * (max - 1.0);
        if self.unit() < 0.5 {
            value
        } else {
            1.0 / value
        }
    }
}

#[derive(Clone, Copy)]
struct Crop {
    left: i32,
    top: i32,
    width: i32,
    height: i32,
    flip: bool,
}
impl Crop {
    fn labels(
        &self,
        labels: &[YoloLabel],
        ow: i32,
        oh: i32,
        nw: usize,
        nh: usize,
    ) -> Vec<YoloLabel> {
        labels
            .iter()
            .filter_map(|label| {
                let mut left =
                    ((label.x - label.w / 2.0) * ow as f32 - self.left as f32) / self.width as f32;
                let mut right =
                    ((label.x + label.w / 2.0) * ow as f32 - self.left as f32) / self.width as f32;
                let top = (((label.y - label.h / 2.0) * oh as f32 - self.top as f32)
                    / self.height as f32)
                    .clamp(0.0, 1.0);
                let bottom = (((label.y + label.h / 2.0) * oh as f32 - self.top as f32)
                    / self.height as f32)
                    .clamp(0.0, 1.0);
                if self.flip {
                    (left, right) = (1.0 - right, 1.0 - left);
                }
                left = left.clamp(0.0, 1.0);
                right = right.clamp(0.0, 1.0);
                let (w, h) = (right - left, bottom - top);
                if w < 1.0 / nw as f32 || h < 1.0 / nh as f32 {
                    return None;
                }
                Some(YoloLabel {
                    class_id: label.class_id,
                    x: (left + right) / 2.0,
                    y: (top + bottom) / 2.0,
                    w,
                    h,
                })
            })
            .collect()
    }

    fn image(&self, image: &PlanarRgbImage, nw: usize, nh: usize) -> Result<Vec<u8>> {
        let rgb = image.to_rgb8()?;
        let src = rgb.as_raw();
        let (ow, oh) = (image.shape.width, image.shape.height);
        let mut mean = [0u8; 3];
        for c in 0..3 {
            mean[c] = ((src.chunks_exact(3).map(|p| p[c] as u64).sum::<u64>() as f64)
                / (ow * oh) as f64)
                .round_ties_even() as u8;
        }
        // Virtual mean-padded crop avoids allocating potentially huge crops.
        let at = |x: i32, y: i32, c: usize| {
            let (x, y) = (i64::from(x) + i64::from(self.left), i64::from(y) + i64::from(self.top));
            if x < 0 || y < 0 || x >= ow as i64 || y >= oh as i64 {
                mean[c]
            } else {
                src[(y as usize * ow + x as usize) * 3 + c]
            }
        };
        let count = nw
            .checked_mul(nh)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| invalid("augmented image size overflow"))?;
        let mut out = vec![0u8; count];
        for y in 0..nh {
            let (y0, y1, wy) = linear_coeff(y, nh, self.height);
            for x in 0..nw {
                let (x0, x1, wx) = linear_coeff(x, nw, self.width);
                let dx = if self.flip { nw - 1 - x } else { x };
                for c in 0..3 {
                    let row = |sy| at(x0, sy, c) as i64 * (2048 - wx) + at(x1, sy, c) as i64 * wx;
                    // OpenCV's specialized u8 vertical pass drops four horizontal
                    // low bits, then truncates each weighted row before rounding.
                    let value =
                        (((row(y0) >> 4) * (2048 - wy) >> 16) + ((row(y1) >> 4) * wy >> 16) + 2)
                            >> 2;
                    out[(y * nw + dx) * 3 + c] = value.clamp(0, 255) as u8;
                }
            }
        }
        Ok(out)
    }
}
fn linear_coeff(i: usize, dst: usize, src: i32) -> (i32, i32, i64) {
    let pos = ((i as f64 + 0.5) * src as f64 / dst as f64 - 0.5) as f32;
    let base = pos.floor() as i32;
    let weight = ((pos - base as f32) * 2048.0).round_ties_even() as i64;
    (base.clamp(0, src - 1), base.saturating_add(1).clamp(0, src - 1), weight)
}
fn u8_round(x: f32) -> u8 {
    x.round_ties_even().clamp(0.0, 255.0) as u8
}

// OpenCV's 8-bit RGB->HSV integer tables, shift=12, hue range 0..179.
fn hsv_distort(pixel: &mut [u8], hue: f32, sat: f32, exp: f32) {
    if hue == 0.0 && sat == 1.0 && exp == 1.0 {
        return;
    }
    let (r, g, b) = (pixel[0] as i32, pixel[1] as i32, pixel[2] as i32);
    let v = r.max(g).max(b);
    let diff = v - r.min(g).min(b);
    let sdiv = if v == 0 {
        0
    } else {
        (255.0f64 * 4096.0 / v as f64).round_ties_even() as i32
    };
    let hdiv = if diff == 0 {
        0
    } else {
        (180.0f64 * 4096.0 / (6 * diff) as f64).round_ties_even() as i32
    };
    let s = (diff * sdiv + 2048) >> 12;
    let h = if v == r {
        g - b
    } else if v == g {
        b - r + 2 * diff
    } else {
        r - g + 4 * diff
    };
    let mut h = (h * hdiv + 2048) >> 12;
    if h < 0 {
        h += 180;
    }
    // C++ uses saturating u8 addition, not a circular hue shift.
    let h = u8_round(h as f32 + 179.0 * hue) as f32 / 30.0;
    let s = u8_round(s as f32 * sat) as f32 / 255.0;
    let v = u8_round(v as f32 * exp) as f32;
    let sector = h.floor() as usize % 6;
    let f = h - h.floor();
    let table = [
        v,
        v * (1.0 - s),
        v * (1.0 - s * f),
        v * (1.0 - s * (1.0 - f)),
    ];
    let indices = [
        [0, 3, 1],
        [2, 0, 1],
        [1, 0, 3],
        [1, 2, 0],
        [3, 1, 0],
        [0, 1, 2],
    ][sector];
    for c in 0..3 {
        pixel[c] = u8_round(table[indices[c]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn settings(extra: &str) -> DetectorAugmentation {
        DetectorAugmentation::from_config(&ConfigFile::parse_str("aug.cfg", &format!("[net]\nwidth=10\nheight=10\nchannels=3\n{extra}\n[yolo]\njitter=.3\nresize=1.5\n")).unwrap()).unwrap()
    }
    fn source() -> PlanarRgbImage {
        let bytes: Vec<u8> = (0..7 * 5 * 3)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        PlanarRgbImage::from_interleaved_rgb_u8(7, 5, &bytes).unwrap()
    }
    #[test]
    fn opencv_crop_resize_flip_hsv_reference() {
        let src = source();
        let mut worst = 0;
        for row in include_str!("../tests/data/augmentation_opencv.csv")
            .lines()
            .filter(|s| !s.starts_with('#'))
        {
            let cols: Vec<_> = row.split(',').collect();
            let n = |i: usize| cols[i].parse::<i32>().unwrap();
            let f = |i: usize| cols[i].parse::<f32>().unwrap();
            let crop = Crop {
                left: n(0),
                top: n(1),
                width: n(2),
                height: n(3),
                flip: n(4) != 0,
            };
            let mut rgb = crop.image(&src, n(5) as usize, n(6) as usize).unwrap();
            for p in rgb.chunks_exact_mut(3) {
                hsv_distort(p, f(7), f(8), f(9));
            }
            let expected: Vec<u8> = cols[10]
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            assert_eq!(rgb.len(), expected.len());
            let error = rgb
                .iter()
                .zip(expected)
                .map(|(a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            worst = worst.max(error);
            assert!(
                error <= 1,
                "OpenCV rounding tolerance exceeded: {error}, crop {}",
                &row[..row.len().min(50)]
            );
        }
        eprintln!("OpenCV fixture worst absolute u8 error: {worst}");
    }
    #[test]
    fn crop_boxes_clip_mirror_and_reject_subpixel_or_outside() {
        let crop = Crop {
            left: 25,
            top: 0,
            width: 50,
            height: 100,
            flip: true,
        };
        let labels = vec![
            YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 0.5,
                h: 0.5,
            },
            YoloLabel {
                class_id: 1,
                x: 0.25,
                y: 0.5,
                w: 0.2,
                h: 0.2,
            },
            YoloLabel {
                class_id: 2,
                x: 0.1,
                y: 0.5,
                w: 0.1,
                h: 0.1,
            },
            YoloLabel {
                class_id: 3,
                x: 0.5,
                y: 0.5,
                w: 0.001,
                h: 0.001,
            },
        ];
        let out = crop.labels(&labels, 100, 100, 10, 10);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 1.0,
                h: 0.5
            }
        );
        assert!((out[1].x - 0.9).abs() < 1e-6 && (out[1].w - 0.2).abs() < 1e-6);
    }
    #[test]
    fn seeded_samples_are_repeatable_and_order_independent() {
        let opts = settings("hue=.1\nsaturation=1.5\nexposure=1.5\nflip=0");
        let src = source();
        let a = opts.apply(&src, &[], 20, 20, 0, 17).unwrap();
        let b = opts.apply(&src, &[], 20, 20, 0, 18).unwrap();
        assert_eq!(a, opts.apply(&src, &[], 20, 20, 0, 17).unwrap());
        assert_ne!(a, b);
        assert_ne!(a, opts.apply(&src, &[], 20, 20, 1, 17).unwrap());
    }
    #[test]
    fn letter_box_expands_crop_and_transforms_labels() {
        let mut opts = settings("letter_box=1\nflip=0");
        opts.jitter = 0.0;
        opts.resize = 1.0;
        let image =
            PlanarRgbImage::from_interleaved_rgb_u8(100, 50, &vec![127; 100 * 50 * 3]).unwrap();
        let label = YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 0.4,
            h: 0.4,
        };
        let (out, labels) = opts.apply(&image, &[label], 100, 100, 0, 0).unwrap();
        assert!(out.data.iter().all(|x| *x == 127.0 / 255.0));
        assert_eq!(labels.len(), 1);
        assert!((labels[0].h - 0.2).abs() < 1e-6);
        assert!((labels[0].w - 0.4).abs() < 1e-6);
        assert_eq!(labels[0].y, 0.5);
    }

    #[test]
    fn augmented_loader_sample_stream_is_subdivision_independent() {
        use crate::{DataConfig, Network, load_augmented_training_batch_from_paths};
        let dir = std::env::temp_dir().join(format!("darknet-aug-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.ppm");
        let mut bytes = b"P6\n7 5\n255\n".to_vec();
        bytes.extend((0..7 * 5 * 3).map(|i| ((i * 37 + 11) % 256) as u8));
        std::fs::write(&path, bytes).unwrap();
        std::fs::write(dir.join("sample.txt"), "0 0.5 0.5 0.5 0.5\n").unwrap();
        std::fs::write(dir.join("sample.data"), "classes=1\n").unwrap();
        let cfg = ConfigFile::parse_str("aug.cfg", "[net]\nbatch=2\nwidth=10\nheight=10\nchannels=3\nhue=.1\nsaturation=1.5\nexposure=1.5\n[convolutional]\nfilters=6\nsize=1\nactivation=linear\n[yolo]\nmask=0\nanchors=10,13\nnum=1\nclasses=1\njitter=.3\nresize=1.5\n").unwrap();
        let opts = DetectorAugmentation::from_config(&cfg).unwrap();
        let network = Network::from_config(cfg).unwrap();
        let data = DataConfig::from_path(dir.join("sample.data")).unwrap();
        let paths = vec![path];
        let combined =
            load_augmented_training_batch_from_paths(&network, &data, &paths, 0, &opts, 42, 8)
                .unwrap();
        let single = network.with_batch(1);
        for i in 0..2 {
            let batch = load_augmented_training_batch_from_paths(
                &single,
                &data,
                &paths,
                i,
                &opts,
                42,
                8 + i as u64,
            )
            .unwrap();
            let len = batch.input.data.len();
            assert_eq!(
                combined.input.data[i * len..(i + 1) * len],
                batch.input.data
            );
            assert_eq!(combined.labels[i], batch.labels[0]);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn multi_scale_dimensions_follow_cpp_rounding_and_clamps() {
        let opts = settings("");
        assert_eq!(opts.random, 0.0);
        assert_eq!(opts.multi_scale_dimensions(416, 416, 0, 1, 1000), (416, 416));
        let cfg = ConfigFile::parse_str("aug.cfg", "[net]\n[yolo]\nrandom=1\n").unwrap();
        let opts = DetectorAugmentation::from_config(&cfg).unwrap();
        assert_eq!(opts.resize_step, 32);
        // rand_coef 1.4 -> round(1.4*416/32 + 1)*32 = 19*32.
        assert_eq!(opts.multi_scale_dimensions(416, 416, 0, 0, 1000), (608, 608));
        let (w, h) = opts.multi_scale_dimensions(416, 416, 0, 10, 1000);
        assert!(w % 32 == 0 && h % 32 == 0 && w >= 32 && h >= 32);
    }

    #[test]
    fn mixup_blends_pixels_and_appends_labels() {
        let a = PlanarRgbImage::from_interleaved_rgb_u8(2, 1, &[10, 20, 30, 40, 50, 60]).unwrap();
        let b = PlanarRgbImage::from_interleaved_rgb_u8(2, 1, &[30, 40, 50, 60, 70, 80]).unwrap();
        let la = vec![YoloLabel {
            class_id: 0,
            x: 0.25,
            y: 0.5,
            w: 0.5,
            h: 1.0,
        }];
        let lb = vec![YoloLabel {
            class_id: 1,
            x: 0.75,
            y: 0.5,
            w: 0.5,
            h: 1.0,
        }];
        let (blended, labels) = blend_mixup(&a, &la, &b, &lb, 200).unwrap();
        assert_eq!(blended.data.len(), a.data.len());
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].class_id, 0);
        assert_eq!(labels[1].class_id, 1);
        let (_, capped) = blend_mixup(&a, &la, &b, &lb, 1).unwrap();
        assert_eq!(capped.len(), 1);
    }

    #[test]
    fn parser_defaults_and_unsupported_active_options() {
        let opts = settings("");
        assert!(opts.flip);
        assert_eq!(opts.max_boxes, 200);
        for key in [
            "mosaic",
            "cutmix",
            "blur",
            "gaussian_noise",
            "fog",
            "track",
            "contrastive",
        ] {
            let cfg =
                ConfigFile::parse_str("aug.cfg", &format!("[net]\n{key}=1\n[yolo]\n")).unwrap();
            assert!(DetectorAugmentation::from_config(&cfg).is_err());
        }
        let mixup =
            ConfigFile::parse_str("aug.cfg", "[net]\nmixup=1\n[yolo]\n").unwrap();
        assert!(DetectorAugmentation::from_config(&mixup).unwrap().mixup);
        for invalid in ["hue=NaN", "saturation=0", "exposure=inf"] {
            let cfg =
                ConfigFile::parse_str("aug.cfg", &format!("[net]\n{invalid}\n[yolo]\n")).unwrap();
            assert!(DetectorAugmentation::from_config(&cfg).is_err());
        }
        let cfg = ConfigFile::parse_str("aug.cfg", "[net]\n[yolo]\nrandom=1\n").unwrap();
        let opts = DetectorAugmentation::from_config(&cfg).unwrap();
        assert_eq!(opts.random, 1.0);
    }
}
