//! Bias gradient of a plain (non-batch-normalised) conv: `bias_updates[c] += Σ_{b,pos} δ[b][c][pos]`
//! (`convolutional_layer.cpp:backward_bias`). One block per channel, the channel viewed as
//! `[batch x spatial]` with `ld = C*spatial` (the same view as [`super::bn_stats`]).

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BiasBwdParams {
    pub batch: u32,
    pub spatial: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for BiasBwdParams {}

#[kernel(lds = reduce_lds_bytes::<f32>())]
fn bias_bwd(delta: Tensor<f32>, acc: TensorMut<f32>, p: BiasBwdParams) {
    let (_, bj) = block_id();
    let ch = bj;
    if ch >= delta.rows() {
        return;
    }
    let (dp, ld) = (delta.ptr(), delta.ld() as usize);
    let base = unsafe { dp.add(ch as usize * p.spatial as usize) };
    let mut sum = 0.0f32;
    let mut r = 0u32;
    while r < p.batch {
        let mut c = 0u32;
        while c < p.spatial {
            let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| {
                let (row, col) = (r + rr as u32, c + cc as u32);
                if row >= p.batch || col >= p.spatial {
                    return 0.0;
                }
                unsafe { *base.add(row as usize * ld + col as usize) }
            });
            sum += t.sum();
            c += TILE_N as u32;
        }
        r += TILE_M as u32;
    }
    if thread_id() == 0 {
        unsafe { *acc.ptr().add(ch as usize) += sum };
    }
}

/// `delta`: channel 0 of the batch as `[C x spatial]` with `ld = C*spatial`; `acc`: `[C]`
/// (accumulated into).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_bias_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    delta: hip_tile::Tensor<f32>,
    acc: hip_tile::TensorMut<f32>,
    p: &BiasBwdParams,
) -> hip_rt::Result<()> {
    assert_eq!(acc.cols(), delta.rows(), "bias_bwd: bad acc");
    assert_eq!(delta.cols(), p.spatial, "bias_bwd: bad spatial");
    bias_bwd::launch(module, stream, (delta.rows(), 1), delta, acc, *p)
}
