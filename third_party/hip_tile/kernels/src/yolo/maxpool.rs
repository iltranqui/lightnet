//! Darknet max pooling (`maxpool_layer`): output = `(in + padding - size) / stride + 1`,
//! window origin `oy * stride - padding / 2`, taps outside the input are ignored (-inf).
//! Darknet's default `padding` is `size - 1`.
//!
//! `x` is `[C x H*W]`, `y` is `[C x OH*OW]` (either may be a view with its own `ld`).

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PoolParams {
    pub h: u32,
    pub w: u32,
    pub size: u32,
    pub stride: u32,
    /// Darknet `padding` (total; the window origin shifts by `padding / 2`).
    pub pad: u32,
    pub oh: u32,
    pub ow: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for PoolParams {}

impl PoolParams {
    /// Darknet shape rule; `pad = None` means the default `size - 1`.
    pub fn new(h: u32, w: u32, size: u32, stride: u32, pad: Option<u32>) -> Self {
        let pad = pad.unwrap_or(size - 1);
        Self {
            h,
            w,
            size,
            stride,
            pad,
            oh: (h + pad - size) / stride + 1,
            ow: (w + pad - size) / stride + 1,
        }
    }
}

#[kernel]
fn maxpool(x: Tensor<f32>, y: TensorMut<f32>, p: PoolParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let (xp, ld) = (x.ptr(), x.ld() as usize);
    let n_out = p.oh * p.ow;
    let off = (p.pad / 2) as i32;
    let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|r, c| {
        let pos = c0 + c as u32;
        let ch = r0 + r as u32;
        if ch < y.rows() && pos < n_out {
            let (oy, ox) = (pos / p.ow, pos % p.ow);
            let (y0, x0) = ((oy * p.stride) as i32 - off, (ox * p.stride) as i32 - off);
            let row = unsafe { xp.add(ch as usize * ld) };
            let mut m = f32::NEG_INFINITY;
            for ky in 0..p.size as i32 {
                for kx in 0..p.size as i32 {
                    let (iy, ix) = (y0 + ky, x0 + kx);
                    if (iy as u32) < p.h && (ix as u32) < p.w {
                        let v = unsafe { *row.add((iy as u32 * p.w + ix as u32) as usize) };
                        if v > m {
                            m = v;
                        }
                    }
                }
            }
            m
        } else {
            0.0
        }
    });
    y.store_tile(r0, c0, &t);
}

/// Pool `x: [C x H*W]` into `y: [C x OH*OW]` (shapes from [`PoolParams::new`]).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_maxpool(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    p: &PoolParams,
) -> hip_rt::Result<()> {
    assert!(x.rows() == y.rows() && x.cols() == p.h * p.w && y.cols() == p.oh * p.ow);
    let grid = (
        y.cols().div_ceil(TILE_N as u32),
        y.rows().div_ceil(TILE_M as u32),
    );
    maxpool::launch(module, stream, grid, x, y, *p)
}
