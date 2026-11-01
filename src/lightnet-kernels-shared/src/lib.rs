//! Per-element kernel bodies shared by the cuda-oxide (PTX) and hip_tile (amdgcn) backends.
//!
//! `no_std`, no dependencies, no nightly features, so both device toolchains compile it.
//! A backend supplies the global thread index, a `&[f32]` view of each input and the output
//! write; everything between those is here, written once.

#![no_std]

/// `x[i] + bias[(i / spatial) % channels]` for a `[batch, channels, spatial]` NCHW buffer.
#[inline(always)]
pub fn bias_add_elem(i: usize, x: &[f32], bias: &[f32], spatial: usize, channels: usize) -> f32 {
    x[i] + bias[(i / spatial) % channels]
}

/// Device math primitives that differ per backend. A backend implements `exp` (and may
/// override `tanh` with a native one); everything built on top is written once here.
pub trait DeviceMath {
    fn exp(x: f32) -> f32;

    /// `tanh(x)` from `exp`, odd-symmetric, saturating to +-1 when `exp` overflows.
    #[inline(always)]
    fn tanh(x: f32) -> f32 {
        let ax = if x < 0.0 { -x } else { x };
        let t = 1.0 - 2.0 / (Self::exp(2.0 * ax) + 1.0);
        if x < 0.0 { -t } else { t }
    }
}

/// Darknet `LOGISTIC`: `1 / (1 + exp(-x))`.
#[inline(always)]
pub fn logistic_elem<M: DeviceMath>(x: f32) -> f32 {
    1.0 / (1.0 + M::exp(-x))
}

/// Darknet `TANH`.
#[inline(always)]
pub fn tanh_elem<M: DeviceMath>(x: f32) -> f32 {
    M::tanh(x)
}

/// Spatial softmax of a `[channels, plane]` NCHW image (one batch item): the value of channel
/// `c` at `pixel`, with the max-stabilised, temperature-scaled softmax over all channels of
/// that pixel (channel stride `plane`). Same order of operations as the Darknet CPU
/// `softmax(input + p, classes, temperature, out + p, spatial)`.
#[inline(always)]
pub fn softmax_spatial_elem<M: DeviceMath>(
    x: &[f32],
    plane: usize,
    channels: usize,
    pixel: usize,
    c: usize,
    temperature: f32,
) -> f32 {
    let mut max = f32::NEG_INFINITY;
    let mut k = 0;
    while k < channels {
        let v = x[k * plane + pixel] / temperature;
        if v > max {
            max = v;
        }
        k += 1;
    }
    let mut sum = 0.0f32;
    let mut k = 0;
    while k < channels {
        sum += M::exp(x[k * plane + pixel] / temperature - max);
        k += 1;
    }
    M::exp(x[c * plane + pixel] / temperature - max) / sum
}

/// Darknet leaky ReLU, slope 0.1.
#[inline(always)]
pub fn leaky_elem(v: f32) -> f32 {
    if v < 0.0 { 0.1 * v } else { v }
}

/// Nearest-neighbour upsample: index into the `[H x W]` input plane for output position
/// `pos` of the `[H*stride x W*stride]` output plane (`ow = w * stride`).
#[inline(always)]
pub fn upsample_src_index(pos: usize, w: usize, ow: usize, stride: usize) -> usize {
    (pos / ow / stride) * w + (pos % ow) / stride
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    struct StdMath;
    impl DeviceMath for StdMath {
        fn exp(x: f32) -> f32 {
            x.exp()
        }
    }

    #[test]
    fn tanh_from_exp_matches_std() {
        let mut worst = 0.0f32;
        for k in -400..=400 {
            let x = k as f32 * 0.05;
            worst = worst.max((tanh_elem::<StdMath>(x) - x.tanh()).abs());
        }
        assert!(worst < 2e-7, "worst {worst}");
        assert_eq!(tanh_elem::<StdMath>(1000.0), 1.0);
        assert_eq!(tanh_elem::<StdMath>(-1000.0), -1.0);
    }

    #[test]
    fn softmax_spatial_sums_to_one_and_matches_naive() {
        let (plane, channels) = (5, 4);
        let x: [f32; 20] = core::array::from_fn(|i| (i as f32 * 0.37).sin() * 3.0);
        for temp in [1.0f32, 0.5] {
            for p in 0..plane {
                let mut total = 0.0f32;
                for c in 0..channels {
                    let got = softmax_spatial_elem::<StdMath>(&x, plane, channels, p, c, temp);
                    let max = (0..channels).map(|k| x[k * plane + p] / temp).fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = (0..channels).map(|k| (x[k * plane + p] / temp - max).exp()).sum();
                    let want = (x[c * plane + p] / temp - max).exp() / sum;
                    assert_eq!(got, want);
                    total += got;
                }
                assert!((total - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn logistic_matches_std() {
        for k in -40..=40 {
            let x = k as f32 * 0.5;
            let want = 1.0 / (1.0 + (-x).exp());
            assert_eq!(logistic_elem::<StdMath>(x), want);
        }
    }

    #[test]
    fn leaky_scales_negatives_only() {
        assert_eq!(leaky_elem(2.0), 2.0);
        assert_eq!(leaky_elem(-2.0), -0.2);
        assert_eq!(leaky_elem(0.0), 0.0);
    }

    #[test]
    fn upsample_index_repeats_each_pixel() {
        // 2x2 input, stride 2 -> 4x4 output.
        let got: [usize; 16] = core::array::from_fn(|p| upsample_src_index(p, 2, 4, 2));
        assert_eq!(got, [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]);
    }

    #[test]
    fn bias_add_picks_channel() {
        let x = [1.0f32; 12];
        let bias = [10.0f32, 20.0, 30.0];
        // spatial 2, channels 3, batch 2: channel pattern 0,0,1,1,2,2,0,0,...
        let got: [f32; 12] = core::array::from_fn(|i| bias_add_elem(i, &x, &bias, 2, 3));
        assert_eq!(got[0], 11.0);
        assert_eq!(got[2], 21.0);
        assert_eq!(got[5], 31.0);
        assert_eq!(got[6], 11.0);
    }
}
