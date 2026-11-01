//! Batch-norm normalise + scale/shift for a training step (`batchnorm_layer.cpp:106-133`).
//!
//! Runs after [`crate::backward::bn_stats`]: `xhat = (x − mean)/sqrt(var + 1e-5)` and
//! `y = act(xhat·scale + bias)`. `xhat` is kept because the backward pass reads it (Darknet's
//! `x_norm`). Elementwise over the whole `[batch x C x spatial]` buffer, one thread per element;
//! the channel of element `i` is `(i / spatial) % C`. The rolling statistics are updated by
//! [`crate::backward::util`]'s `running_update`, not here.
//!
//! The activation is fused so a BN layer is conv (linear, zero bias) → `bn_stats` → this kernel.

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

#[cfg(target_arch = "amdgpu")]
use super::bn_bwd::{BN_EPS, sqrtf};
#[cfg(target_arch = "amdgpu")]
use crate::yolo::conv::ConvParams;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BnFwdParams {
    pub channels: u32,
    pub spatial: u32,
    /// [`ConvParams::LINEAR`] or [`ConvParams::LEAKY`].
    pub act: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for BnFwdParams {}

#[kernel]
#[allow(clippy::too_many_arguments)]
fn bn_fwd(
    x: Tensor<f32>,
    mean: Tensor<f32>,
    var: Tensor<f32>,
    scale: Tensor<f32>,
    bias: Tensor<f32>,
    xhat: TensorMut<f32>,
    y: TensorMut<f32>,
    p: BnFwdParams,
) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < x.cols() as usize {
        let ch = (i / p.spatial as usize) % p.channels as usize;
        let (m, v, g, b) = unsafe { (*mean.ptr().add(ch), *var.ptr().add(ch), *scale.ptr().add(ch), *bias.ptr().add(ch)) };
        let xh = (unsafe { *x.ptr().add(i) } - m) / sqrtf(v + BN_EPS);
        let mut out = xh * g + b;
        if p.act == ConvParams::LEAKY && out < 0.0 {
            out *= 0.1;
        }
        unsafe {
            *xhat.ptr().add(i) = xh;
            *y.ptr().add(i) = out;
        }
    }
}

/// `x`, `xhat`, `y`: flat `[batch*C*spatial]` vectors (`Tensor::vector`); `mean`, `var`, `scale`,
/// `bias`: `[C]`.
#[cfg(not(target_arch = "amdgpu"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_bn_fwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    mean: hip_tile::Tensor<f32>,
    var: hip_tile::Tensor<f32>,
    scale: hip_tile::Tensor<f32>,
    bias: hip_tile::Tensor<f32>,
    xhat: hip_tile::TensorMut<f32>,
    y: hip_tile::TensorMut<f32>,
    p: &BnFwdParams,
) -> hip_rt::Result<()> {
    assert_eq!(x.cols(), xhat.cols(), "bn_fwd: bad xhat");
    assert_eq!(x.cols(), y.cols(), "bn_fwd: bad y");
    assert_eq!(x.cols() % (p.channels * p.spatial), 0, "bn_fwd: length is not batch*C*spatial");
    for (name, t) in [("mean", mean.cols()), ("var", var.cols()), ("scale", scale.cols()), ("bias", bias.cols())] {
        assert!(t >= p.channels, "bn_fwd: bad {name}");
    }
    let grid = (x.cols().div_ceil(BLOCK_THREADS), 1);
    bn_fwd::launch(module, stream, grid, x, mean, var, scale, bias, xhat, y, *p)
}
