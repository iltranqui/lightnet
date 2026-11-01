//! Gradient of Darknet's max pooling: each window sends its upstream gradient to the input
//! element it selected (the argmax tap, recomputed here so no index buffer is needed).
//!
//! Written as a **gather** over input positions, not a scatter over windows: for every input
//! element the at-most `size²` windows containing it are examined and a window contributes only
//! if it selected this element. That is race-free even when windows overlap
//! (`stride < size`), and needs no atomics.
//!
//! `dy` is `[C x OH*OW]`, `dx` is `[C x H*W]` (either may be a view with its own `ld`).

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

use crate::yolo::maxpool::PoolParams;

/// Channels per tile and input positions per tile (`BM*BN = 8192`, 32 elements per thread).
pub const TILE_M: usize = 64;
pub const TILE_N: usize = 128;

/// The window that contains `(iy, ix)` for tap `(ky, kx)`, or `None` when the tap does not line up.
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn window_of(iy: i32, ix: i32, ky: i32, kx: i32, off: i32, stride: i32) -> Option<(i32, i32)> {
    // `iy = off + oy*stride + ky`  =>  `oy = (iy - off - ky) / stride`.
    let (uy, ux) = (iy - off - ky, ix - off - kx);
    if uy < 0 || ux < 0 || uy % stride != 0 || ux % stride != 0 {
        return None;
    }
    Some((uy / stride, ux / stride))
}

/// Did the window at `(oy, ox)` select the tap at `(iy, ix)`? Out-of-image taps are ignored, and
/// the first maximum wins (the same tie rule as the forward).
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn selects(shared: &SharedPool, ch: usize, iy: i32, ix: i32, oy: i32, ox: i32) -> bool {
    let p = &shared.p;
    // The input is `[C x H*W]`: this channel starts `ch * H*W` elements in.
    let lane = ch * (p.h as usize * p.w as usize);
    if oy < 0 || ox < 0 || oy >= p.oh as i32 || ox >= p.ow as i32 {
        return false;
    }
    let off = -((p.pad / 2) as i32);
    let mut best = f32::NEG_INFINITY;
    let mut best_at = (i32::MIN, i32::MIN);
    for ky in 0..p.size as i32 {
        for kx in 0..p.size as i32 {
            let sy = off + oy * p.stride as i32 + ky;
            let sx = off + ox * p.stride as i32 + kx;
            if sy < 0 || sx < 0 || sy >= p.h as i32 || sx >= p.w as i32 {
                continue;
            }
            let v = unsafe { *shared.x.add(lane + sy as usize * p.w as usize + sx as usize) };
            if v > best {
                best = v;
                best_at = (sy, sx);
            }
        }
    }
    best_at == (iy, ix)
}

/// Input pointer plus shape, bundled so the per-element closure stays small.
#[cfg(target_arch = "amdgpu")]
struct SharedPool {
    x: *const f32,
    p: PoolParams,
}

#[kernel]
fn maxpool_bwd(x: Tensor<f32>, dy: Tensor<f32>, dx: TensorMut<f32>, p: PoolParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let dyp = dy.ptr();
    let shared = SharedPool { x: x.ptr(), p };
    let off = -((p.pad / 2) as i32);
    let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|r, c| {
        let ch = r0 + r as u32;
        let pos = c0 + c as u32;
        if ch >= dx.rows() || pos >= dx.cols() {
            return 0.0;
        }
        let (iy, ix) = ((pos / p.w) as i32, (pos % p.w) as i32);
        let mut acc = 0.0f32;
        for ky in 0..p.size as i32 {
            for kx in 0..p.size as i32 {
                let Some((oy, ox)) = window_of(iy, ix, ky, kx, off, p.stride as i32) else { continue };
                if selects(&shared, ch as usize, iy, ix, oy, ox) {
                    let i = ch as usize * dy.ld() as usize
                        + oy as usize * p.ow as usize
                        + ox as usize;
                    acc += unsafe { *dyp.add(i) };
                }
            }
        }
        acc
    });
    // Accumulate: the caller's buffer may already hold another branch's gradient.
    dx.store_tile_add(r0, c0, &t);
}

/// `x: [C x H*W]` (the forward input), `dy: [C x OH*OW]`, `dx: [C x H*W]` (accumulated into).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_maxpool_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    dy: hip_tile::Tensor<f32>,
    dx: hip_tile::TensorMut<f32>,
    p: &PoolParams,
) -> hip_rt::Result<()> {
    assert_eq!(x.rows(), dx.rows(), "maxpool_bwd: bad input");
    assert_eq!(dy.rows(), dx.rows(), "maxpool_bwd: bad dy");
    assert_eq!(x.cols(), p.h * p.w, "maxpool_bwd: bad input size");
    assert_eq!(dy.cols(), p.oh * p.ow, "maxpool_bwd: bad dy size");
    assert_eq!(dx.cols(), p.h * p.w, "maxpool_bwd: bad output size");
    // Grid convention: (blocks along columns, blocks along rows) — see `_shared/tile-model.md`.
    let grid = (dx.cols().div_ceil(TILE_N as u32), dx.rows().div_ceil(TILE_M as u32));
    maxpool_bwd::launch(module, stream, grid, x, dy, dx, *p)
}
