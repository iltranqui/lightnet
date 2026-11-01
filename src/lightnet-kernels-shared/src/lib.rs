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
