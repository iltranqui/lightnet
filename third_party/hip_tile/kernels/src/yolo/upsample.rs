//! Nearest-neighbour upsample: `y[c][oy][ox] = x[c][oy / s][ox / s]`.
//!
//! `x` is `[C x H*W]`, `y` is `[C x (H*s)*(W*s)]` (either may be a view with its own `ld`).

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UpsampleParams {
    /// Input width.
    pub w: u32,
    /// Output width (`w * stride`).
    pub ow: u32,
    pub stride: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for UpsampleParams {}

#[kernel]
fn upsample(x: Tensor<f32>, y: TensorMut<f32>, p: UpsampleParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let (xp, ld) = (x.ptr(), x.ld() as usize);
    let n_out = y.cols();
    let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|r, c| {
        let pos = c0 + c as u32;
        let ch = r0 + r as u32;
        if ch < y.rows() && pos < n_out {
            let (oy, ox) = (pos / p.ow, pos % p.ow);
            let i = (oy / p.stride) * p.w + ox / p.stride;
            unsafe { *xp.add(ch as usize * ld + i as usize) }
        } else {
            0.0
        }
    });
    y.store_tile(r0, c0, &t);
}

/// `x: [C x H*W]`, `y: [C x (H*s)*(W*s)]`.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_upsample(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    w: u32,
    stride: u32,
) -> hip_rt::Result<()> {
    assert!(
        x.rows() == y.rows() && x.cols().is_multiple_of(w),
        "upsample: shape mismatch"
    );
    let h = x.cols() / w;
    assert!(
        y.cols() == h * stride * w * stride,
        "upsample: bad output size"
    );
    let p = UpsampleParams {
        w,
        ow: w * stride,
        stride,
    };
    let grid = (
        y.cols().div_ceil(TILE_N as u32),
        y.rows().div_ceil(TILE_M as u32),
    );
    upsample::launch(module, stream, grid, x, y, p)
}
