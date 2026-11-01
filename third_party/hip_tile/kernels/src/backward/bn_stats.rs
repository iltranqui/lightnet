//! Batch-norm batch statistics for training (`batchnorm_layer.cpp:106-133`): a per-channel mean
//! over `batch*spatial` and the **unbiased** variance (`/(n-1)`, `blas.cpp:417`).
//!
//! One block per channel: the channel is viewed as `[batch x spatial]` with `ld = C*spatial`, so a
//! tile spans up to 8 images at once. Loaded rows beyond the batch are masked to zero, which adds
//! nothing to either sum; the divisor is the true `batch*spatial`.

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BnStatsParams {
    pub batch: u32,
    pub spatial: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for BnStatsParams {}

#[kernel(lds = reduce_lds_bytes::<f32>())]
fn bn_stats(x: Tensor<f32>, mean: TensorMut<f32>, var: TensorMut<f32>, p: BnStatsParams) {
    // One block per channel: the block id along x selects the channel.
    let (_, bj) = block_id();
    let ch = bj;
    if ch >= x.rows() {
        return;
    }
    let (xp, ld) = (x.ptr(), x.ld() as usize);
    // Channel `ch` of image 0 sits `ch * spatial` elements into the row.
    let base = unsafe { xp.add(ch as usize * p.spatial as usize) };
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    let mut r = 0u32;
    while r < p.batch {
        let rows = (p.batch - r).min(TILE_M as u32);
        let mut c = 0u32;
        while c < p.spatial {
            // Load TILE_M rows starting at batch index `r`, columns from `c`.
            let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| {
                let row = r + rr as u32;
                let col = c + cc as u32;
                if rr as u32 >= rows || row >= p.batch || col >= p.spatial {
                    return 0.0;
                }
                unsafe { *base.add(row as usize * ld + col as usize) }
            });
            sum += t.sum();
            sumsq += t.map(|v| v * v).sum();
            c += TILE_N as u32;
        }
        r += TILE_M as u32;
    }
    if thread_id() == 0 {
        let n = (p.batch * p.spatial) as f32;
        let m = sum / n;
        let v = if n > 1.0 { (sumsq - sum * sum / n) / (n - 1.0) } else { 0.0 };
        unsafe {
            *mean.ptr().add(ch as usize) = m;
            *var.ptr().add(ch as usize) = v;
        }
    }
}

/// `x`: view of channel `0` of the batch, i.e. `[C x spatial]` with `ld = C*spatial` and the
/// images at row stride `ld`. `mean`/`var`: `[C]`.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_bn_stats(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    mean: hip_tile::TensorMut<f32>,
    var: hip_tile::TensorMut<f32>,
    p: &BnStatsParams,
) -> hip_rt::Result<()> {
    assert_eq!(mean.cols(), x.rows(), "bn_stats: bad mean");
    assert_eq!(var.cols(), x.rows(), "bn_stats: bad var");
    assert_eq!(x.cols(), p.spatial, "bn_stats: bad spatial");
    // One block per channel: the channel axis is the block id along x.
    let grid = (x.rows(), 1);
    bn_stats::launch(module, stream, grid, x, mean, var, *p)
}
