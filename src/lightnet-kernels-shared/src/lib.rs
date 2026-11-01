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
    ///
    /// `1 - 2/(exp(2x)+1)` cancels for small `|x|` (relative error up to a few percent), so
    /// below 0.1 the odd series `x - x^3/3 + 2x^5/15 - 17x^7/315` is used instead (truncation
    /// error under 1e-10 relative at the switch).
    #[inline(always)]
    fn tanh(x: f32) -> f32 {
        let ax = if x < 0.0 { -x } else { x };
        if ax < 0.1 {
            let x2 = x * x;
            return x * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0))));
        }
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

/// Darknet max pool window for output `(oy, ox)` of one `[h x w]` plane.
///
/// The window covers rows `oy * stride_y + ky - off` (columns likewise) for `ky, kx < size`,
/// with `off = padding / 2`. Taps outside the plane are skipped, not read as zero. The scan
/// is row by row and replaces the best value only on strict `>`, so the result is the first
/// maximum; it is `-inf` when no tap is valid. `load(i)` reads element `plane + iy * w + ix`
/// of the input, so each backend supplies its own memory access (slice, raw pointer, f16 decode).
#[inline(always)]
pub fn maxpool_window_max<L: Fn(usize) -> f32>(
    load: L,
    plane: usize,
    hw: (usize, usize),
    out_pos: (usize, usize),
    size: usize,
    strides: (usize, usize),
    off: usize,
) -> f32 {
    maxpool_window_scan(load, plane, hw, out_pos, size, strides, off).0
}

/// Flat index (`plane + iy * w + ix`) of the first maximum of the same window, which is what
/// the backward pass routes the gradient to; `usize::MAX` when no tap beats `-inf`
/// (no valid tap, or only NaN / `-inf`).
#[inline(always)]
pub fn maxpool_window_argmax<L: Fn(usize) -> f32>(
    load: L,
    plane: usize,
    hw: (usize, usize),
    out_pos: (usize, usize),
    size: usize,
    strides: (usize, usize),
    off: usize,
) -> usize {
    maxpool_window_scan(load, plane, hw, out_pos, size, strides, off).1
}

/// The scan behind [`maxpool_window_max`] and [`maxpool_window_argmax`]: `(best, best_index)`.
#[inline(always)]
fn maxpool_window_scan<L: Fn(usize) -> f32>(
    load: L,
    plane: usize,
    (h, w): (usize, usize),
    (oy, ox): (usize, usize),
    size: usize,
    (stride_x, stride_y): (usize, usize),
    off: usize,
) -> (f32, usize) {
    let y0 = (oy * stride_y) as i64 - off as i64;
    let x0 = (ox * stride_x) as i64 - off as i64;
    let mut best = f32::NEG_INFINITY;
    let mut best_i = usize::MAX;
    let mut ky = 0usize;
    while ky < size {
        let iy = y0 + ky as i64;
        if iy >= 0 && iy < h as i64 {
            let mut kx = 0usize;
            while kx < size {
                let ix = x0 + kx as i64;
                if ix >= 0 && ix < w as i64 {
                    let idx = plane + iy as usize * w + ix as usize;
                    let v = load(idx);
                    if v > best {
                        best = v;
                        best_i = idx;
                    }
                }
                kx += 1;
            }
        }
        ky += 1;
    }
    (best, best_i)
}

/// Decode a row `kidx` of the im2col matrix, `kidx = (ci * k + ky) * k + kx`, into
/// `(ci, ky, kx)`. `ci` counts channels inside the group the GEMM works on.
///
/// The convolution index math (this and [`conv_input_offset`]) is `u32` on purpose: it runs
/// per element inside the tile GEMM loops, where 64-bit division is slow on both GPUs. Plane
/// sizes and `C * k * k` are far below `u32::MAX`.
#[inline(always)]
pub fn conv_tap(kidx: u32, k: u32) -> (u32, u32, u32) {
    let kk = k * k;
    let ci = kidx / kk;
    let rem = kidx - ci * kk;
    let ky = rem / k;
    (ci, ky, rem - ky * k)
}

/// Offset `iy * w + ix` inside one `[h x w]` input plane read by tap `(ky, kx)` for output
/// position `pos = oy * ow + ox`, or `None` when the tap falls into the zero padding:
/// `iy = oy * stride_y + ky * dilation - pad` (columns likewise).
#[inline(always)]
pub fn conv_input_offset(
    pos: u32,
    ow: u32,
    (ky, kx): (u32, u32),
    (stride_x, stride_y): (u32, u32),
    dilation: u32,
    pad: u32,
    (h, w): (u32, u32),
) -> Option<u32> {
    let oy = pos / ow;
    let ox = pos - oy * ow;
    let iy = (oy * stride_y + ky * dilation) as i32 - pad as i32;
    let ix = (ox * stride_x + kx * dilation) as i32 - pad as i32;
    // A negative coordinate wraps to a huge `u32`, so one compare covers both bounds.
    if (iy as u32) < h && (ix as u32) < w { Some(iy as u32 * w + ix as u32) } else { None }
}

/// Route (channel concat) copy: for element `i` of a `batch * channels * stride` slice, the
/// flat `(from, to)` indices of `src[(b*src_total + src_offset + c)*stride + s]` and
/// `dst[(b*dst_total + dst_offset + c)*stride + s]`. A grouped route is a nonzero `src_offset`.
#[inline(always)]
pub fn route_copy_index(
    i: usize,
    channels: usize,
    stride: usize,
    (src_total, src_offset): (usize, usize),
    (dst_total, dst_offset): (usize, usize),
) -> (usize, usize) {
    let per_batch = channels * stride;
    let b = i / per_batch;
    let r = i % per_batch;
    let (c, s) = (r / stride, r % stride);
    (
        (b * src_total + src_offset + c) * stride + s,
        (b * dst_total + dst_offset + c) * stride + s,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;
    use std::vec::Vec;

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
    fn tanh_is_accurate_relative_to_its_value_near_zero() {
        // `1 - 2/(exp(2x)+1)` alone loses most digits for small |x|; the series must not.
        let mut worst = 0.0f64;
        for k in 1..=4000 {
            let x = k as f32 * 5e-5; // up to 0.2, across the series/exp switch at 0.1
            for s in [x, -x] {
                let want = (s as f64).tanh();
                let got = tanh_elem::<StdMath>(s) as f64;
                worst = worst.max((got - want).abs() / want.abs());
            }
        }
        assert!(worst < 3e-6, "worst relative error {worst}");
        assert_eq!(tanh_elem::<StdMath>(0.0), 0.0);
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

    fn pool(x: &[f32], hw: (usize, usize), o: (usize, usize), size: usize, st: (usize, usize), off: usize) -> f32 {
        maxpool_window_max(|i| x[i], 0, hw, o, size, st, off)
    }

    #[test]
    fn maxpool_takes_window_max() {
        // 3x3 plane, 2x2 window, stride 1, no padding.
        let x = [1.0, 5.0, 2.0, 4.0, 3.0, 9.0, 7.0, 6.0, 8.0f32];
        assert_eq!(pool(&x, (3, 3), (0, 0), 2, (1, 1), 0), 5.0);
        assert_eq!(pool(&x, (3, 3), (0, 1), 2, (1, 1), 0), 9.0);
        assert_eq!(pool(&x, (3, 3), (1, 1), 2, (1, 1), 0), 9.0);
    }

    #[test]
    fn maxpool_skips_taps_outside_the_plane() {
        // Padding 1 (off 0) with size 2, stride 1: the last window hangs over the edge.
        // Negative values must win over the skipped taps, which are not read as zero.
        let x = [-4.0, -3.0, -2.0, -1.0f32];
        assert_eq!(pool(&x, (2, 2), (1, 1), 2, (1, 1), 0), -1.0);
        // off 1 shifts the origin up/left: window (0,0) covers only tap (0,0).
        assert_eq!(pool(&x, (2, 2), (0, 0), 2, (1, 1), 1), -4.0);
    }

    #[test]
    fn maxpool_empty_window_is_neg_infinity() {
        let x = [1.0f32; 4];
        assert_eq!(pool(&x, (2, 2), (5, 5), 2, (1, 1), 0), f32::NEG_INFINITY);
    }

    #[test]
    fn maxpool_ignores_nan() {
        let x = [f32::NAN, 2.0, 1.0, f32::NAN];
        assert_eq!(pool(&x, (2, 2), (0, 0), 2, (1, 1), 0), 2.0);
    }

    #[test]
    fn maxpool_uses_each_stride() {
        // 2x4 plane, 1x1 window: stride_x 2 picks columns 0 and 2, stride_y 1 picks rows.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0f32];
        assert_eq!(pool(&x, (2, 4), (1, 1), 1, (2, 1), 0), 6.0);
        assert_eq!(pool(&x, (2, 4), (0, 1), 1, (2, 1), 0), 2.0);
    }

    #[test]
    fn maxpool_argmax_is_the_first_maximum() {
        // Two equal maxima in a 2x2 window: the earlier tap (row-major) wins.
        let x = [3.0, 3.0, 1.0, 3.0f32];
        let at = maxpool_window_argmax(|i| x[i], 0, (2, 2), (0, 0), 2, (1, 1), 0);
        assert_eq!(at, 0);
        // Plane offset is included in the index.
        let y = [0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 2.0, 5.0f32];
        let at = maxpool_window_argmax(|i| y[i], 4, (2, 2), (0, 0), 2, (1, 1), 0);
        assert_eq!(at, 5);
    }

    #[test]
    fn maxpool_argmax_without_a_winner_is_max() {
        let x = [f32::NAN; 4];
        assert_eq!(maxpool_window_argmax(|i| x[i], 0, (2, 2), (0, 0), 2, (1, 1), 0), usize::MAX);
        assert_eq!(maxpool_window_argmax(|i| x[i], 0, (2, 2), (9, 9), 2, (1, 1), 0), usize::MAX);
    }

    #[test]
    fn maxpool_reads_the_given_plane() {
        let x = [0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 1.0, 1.0f32];
        assert_eq!(maxpool_window_max(|i| x[i], 4, (2, 2), (0, 0), 2, (1, 1), 0), 7.0);
    }

    /// Direct convolution (no GEMM): the reference the gathered form must equal.
    #[allow(clippy::too_many_arguments)]
    fn conv_direct(x: &[f32], wt: &[f32], c: usize, (h, w): (usize, usize), k: usize, (sx, sy): (usize, usize), dil: usize, pad: usize, (oh, ow): (usize, usize)) -> Vec<f32> {
        let mut y = vec![0.0f32; oh * ow];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = 0.0f32;
                for ci in 0..c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = (oy * sy + ky * dil) as i64 - pad as i64;
                            let ix = (ox * sx + kx * dil) as i64 - pad as i64;
                            if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                                acc += wt[(ci * k + ky) * k + kx] * x[(ci * h + iy as usize) * w + ix as usize];
                            }
                        }
                    }
                }
                y[oy * ow + ox] = acc;
            }
        }
        y
    }

    #[test]
    fn conv_gather_matches_direct_convolution() {
        let (c, h, w) = (3usize, 7usize, 9usize);
        let x: Vec<f32> = (0..c * h * w).map(|i| (i as f32 * 0.37).sin()).collect();
        // (k, stride_x, stride_y, dilation, pad)
        for (k, sx, sy, dil, pad) in [(1, 1, 1, 1, 0), (3, 1, 1, 1, 1), (3, 2, 2, 1, 1), (3, 2, 1, 1, 0), (3, 1, 1, 2, 2), (5, 1, 2, 1, 2)] {
            let eff = dil * (k - 1) + 1;
            let (oh, ow) = ((h + 2 * pad - eff) / sy + 1, (w + 2 * pad - eff) / sx + 1);
            let wt: Vec<f32> = (0..c * k * k).map(|i| (i as f32 * 0.91).cos()).collect();
            let want = conv_direct(&x, &wt, c, (h, w), k, (sx, sy), dil, pad, (oh, ow));
            for (pos, &expected) in want.iter().enumerate() {
                let mut acc = 0.0f32;
                for kidx in 0..(c * k * k) as u32 {
                    let (ci, ky, kx) = conv_tap(kidx, k as u32);
                    assert_eq!((ci * k as u32 + ky) * k as u32 + kx, kidx);
                    let off = conv_input_offset(pos as u32, ow as u32, (ky, kx), (sx as u32, sy as u32), dil as u32, pad as u32, (h as u32, w as u32));
                    if let Some(off) = off {
                        acc += wt[kidx as usize] * x[ci as usize * h * w + off as usize];
                    }
                }
                assert_eq!(acc, expected, "k{k} sx{sx} sy{sy} dil{dil} pad{pad} pos{pos}");
            }
        }
    }

    #[test]
    fn conv_input_offset_reports_padding() {
        // 3x3 kernel, pad 1 on a 2x2 plane: output (0,0) tap (0,0) is outside, (1,1) is (0,0).
        let off = |ky, kx| conv_input_offset(0, 2, (ky, kx), (1, 1), 1, 1, (2, 2));
        assert_eq!(off(0, 0), None);
        assert_eq!(off(1, 1), Some(0));
        assert_eq!(off(2, 2), Some(3));
        assert_eq!(off(2, 0), None);
    }

    #[test]
    fn route_copy_index_slices_channels() {
        // batch 2, 2 channels of 3 pixels taken from a 4-channel source at channel 1,
        // written into a 5-channel destination at channel 2.
        let at = |i| route_copy_index(i, 2, 3, (4, 1), (5, 2));
        assert_eq!(at(0), (3, 6));
        assert_eq!(at(5), (8, 11));
        // batch 1 starts a whole source/destination batch further on.
        assert_eq!(at(6), (15, 21));
        assert_eq!(at(11), (20, 26));
    }

    #[test]
    fn route_copy_index_whole_tensor_is_identity() {
        for i in 0..12 {
            assert_eq!(route_copy_index(i, 3, 2, (3, 0), (3, 0)), (i, i));
        }
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
