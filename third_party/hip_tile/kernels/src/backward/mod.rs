//! Kernels for the training backward pass, in the same f32 NCHW convention as
//! [`crate::yolo`]: every activation is a `[C x H*W]` row-major matrix, so a channel slice of a
//! larger buffer is a view at a row offset.
//!
//! Accumulation, not overwriting: gradients fan in from several layers, so the kernels that
//! write a gradient add to it (`store_tile_add`) and the driver zeros the buffers once per
//! iteration. This is the one place the port deliberately differs from Darknet's `col2im`, which
//! zeroes first (see `_shared/darknet-semantics.md#gradient-routing`).
//!
//! - [`util`]: fill, accumulate, the SGD step, rolling BN statistics.
//! - [`maxpool_bwd`], [`upsample_bwd`]: gather formulations, race-free without atomics.
//! - [`route_bwd`]: channel-block accumulate for route/concat fan-in.
//! - [`bn_stats`], [`bn_fwd`], [`bn_bwd`]: batch-norm statistics, forward and gradients.
//! - [`act_bwd`], [`bias_bwd`]: activation gradient and the plain-conv bias gradient.
//! - [`batched`]: host-side per-image launches of the gather/accumulate kernels for `batch > 1`.
//! - [`conv_bwd_dw`], [`conv_bwd_dx`]: the two convolution gradients.
//! - [`yolo_act`], [`yolo_delta`]: the `[yolo]` forward activation and delta tensor.

use hip_tile::KernelArg;

use crate::yolo::conv::ConvParams;

/// Parameters shared by the two convolution gradients.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ConvBwdParams {
    pub conv: ConvParams,
    /// Images in the batch; the kernels reach them by striding `conv`-shaped views.
    pub batch: u32,
}

// SAFETY: `repr(C)` struct of `u32`s (`ConvParams` is `repr(C)` too).
unsafe impl KernelArg for ConvBwdParams {}

impl ConvBwdParams {
    pub fn new(conv: ConvParams, batch: u32) -> Self {
        Self { conv, batch }
    }
}

pub mod act_bwd;
#[cfg(not(target_arch = "amdgpu"))]
pub mod batched;
pub mod bias_bwd;
pub mod bn_bwd;
pub mod bn_fwd;
pub mod bn_stats;
pub mod conv_bwd_dw;
pub mod conv_bwd_dx;
pub mod maxpool_bwd;
pub mod route_bwd;
pub mod upsample_bwd;
pub mod util;
pub mod yolo_act;
pub mod yolo_delta;

