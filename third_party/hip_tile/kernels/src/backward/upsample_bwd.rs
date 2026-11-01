//! Gradient of nearest-neighbour upsampling.
//!
//! Written as a **gather**: every input element sums the `stride²` upstream gradients that were
//! copied from it in the forward. No scatter, no races, no atomics.
//!
//! `dy` is `[C x (H*s)*(W*s)]`, `dx` is `[C x H*W]` (either may be a view with its own `ld`).

use hip_tile::prelude::*;
use hip_tile::KernelArg;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 64;
pub const TILE_N: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UpsampleBwdParams {
    /// Input width.
    pub w: u32,
    /// Upstream gradient width (`w * stride`).
    pub ow: u32,
    pub stride: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for UpsampleBwdParams {}

#[kernel]
fn upsample_bwd(dy: Tensor<f32>, dx: TensorMut<f32>, p: UpsampleBwdParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let (dyp, ld) = (dy.ptr(), dy.ld() as usize);
    let s = p.stride;
    let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|r, c| {
        let ch = r0 + r as u32;
        let pos = c0 + c as u32;
        if ch >= dx.rows() || pos >= dx.cols() {
            return 0.0;
        }
        let (iy, ix) = (pos / p.w, pos % p.w);
        let mut acc = 0.0f32;
        for dyi in 0..s {
            for dxi in 0..s {
                let oy = iy * s + dyi;
                let ox = ix * s + dxi;
                acc += unsafe { *dyp.add(ch as usize * ld + (oy * p.ow + ox) as usize) };
            }
        }
        acc
    });
    // Accumulate into whatever the caller's buffer already holds.
    dx.store_tile_add(r0, c0, &t);
}

/// `dy: [C x (H*s)*(W*s)]`, `dx: [C x H*W]` (accumulated into).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_upsample_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    dy: hip_tile::Tensor<f32>,
    dx: hip_tile::TensorMut<f32>,
    w: u32,
    stride: u32,
) -> hip_rt::Result<()> {
    assert_eq!(dy.rows(), dx.rows(), "upsample_bwd: channel mismatch");
    assert!(dx.cols().is_multiple_of(w), "upsample_bwd: bad input width");
    let h = dx.cols() / w;
    assert_eq!(dy.cols(), h * stride * w * stride, "upsample_bwd: bad upstream size");
    let p = UpsampleBwdParams { w, ow: w * stride, stride };
    let grid = (dx.cols().div_ceil(TILE_N as u32), dx.rows().div_ceil(TILE_M as u32));
    upsample_bwd::launch(module, stream, grid, dy, dx, p)
}
