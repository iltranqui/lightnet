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

#[cfg(test)]
mod tests {
    use super::*;

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
