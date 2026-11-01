//! Elementwise trainer utilities: fill, accumulate, and the SGD step.
//!
//! These are flat f32 loops where every thread owns one element (the pattern of [`crate::saxpy`]),
//! because the operations are inherently elementwise. All of them take `Tensor`/`TensorMut` views
//! so a weight slice, a momentum buffer and its gradient can share the same layout.

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

/// `y[i] = 0`
#[kernel]
fn zero(y: TensorMut<f32>) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < y.cols() as usize {
        unsafe { *y.ptr().add(i) = 0.0 };
    }
}

/// `y[i] += a * x[i]`, the gradient-accumulation primitive (`dX` fan-in, concat slices).
#[kernel]
fn axpy_acc(a: f32, x: Tensor<f32>, y: TensorMut<f32>) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < y.cols() as usize {
        let (xp, yp) = (x.ptr(), y.ptr());
        unsafe { *yp.add(i) += a * *xp.add(i) };
    }
}

/// `y[i] += x[i]` (`axpy_acc` with `a = 1`, so callers do not have to spell the constant out).
#[kernel]
fn acc(x: Tensor<f32>, y: TensorMut<f32>) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < y.cols() as usize {
        let (xp, yp) = (x.ptr(), y.ptr());
        unsafe { *yp.add(i) += *xp.add(i) };
    }
}

/// Darknet's SGD step for one parameter tensor (`convolutional_layer.cpp:2983-3005`):
/// `u += -decay*batch*w; w += (lr/batch)*u; u *= momentum`.
///
/// `decay == 0` gives the bias / BN-scale variant (Darknet never decays those).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SgdParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub decay: f32,
    pub batch: f32,
}

// SAFETY: `repr(C)` struct of `f32`s.
unsafe impl KernelArg for SgdParams {}

#[kernel]
fn sgd_update(w: TensorMut<f32>, u: TensorMut<f32>, p: SgdParams) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < w.cols() as usize {
        let (wp, up) = (w.ptr(), u.ptr());
        let step = p.learning_rate / p.batch;
        let weight = unsafe { *wp.add(i) };
        let mut update = unsafe { *up.add(i) } - p.decay * p.batch * weight;
        unsafe { *wp.add(i) = weight + step * update };
        update *= p.momentum;
        unsafe { *up.add(i) = update };
    }
}

/// `rolling = momentum*rolling + (1 - momentum)*batch_stat` (`batchnorm_layer.cpp:118-123`).
#[kernel]
fn running_update(rolling: TensorMut<f32>, batch_stat: Tensor<f32>, momentum: f32) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < rolling.cols() as usize {
        let (rp, sp) = (rolling.ptr(), batch_stat.ptr());
        let old = unsafe { *rp.add(i) };
        let new = unsafe { *sp.add(i) };
        unsafe { *rp.add(i) = momentum * old + (1.0 - momentum) * new };
    }
}

/// The flat-index kernels use the block's x id (`block_id().1`), i.e. the *first* grid dimension.
fn grid_for(len: u32) -> (u32, u32) {
    (len.div_ceil(BLOCK_THREADS), 1)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_zero(module: &hip_rt::Module, stream: Option<&hip_rt::Stream>, y: hip_tile::TensorMut<f32>) -> hip_rt::Result<()> {
    zero::launch(module, stream, grid_for(y.cols()), y)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_axpy_acc(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    a: f32,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
) -> hip_rt::Result<()> {
    assert_eq!(x.cols(), y.cols(), "axpy_acc: length mismatch");
    axpy_acc::launch(module, stream, grid_for(y.cols()), a, x, y)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_acc(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
) -> hip_rt::Result<()> {
    assert_eq!(x.cols(), y.cols(), "acc: length mismatch");
    acc::launch(module, stream, grid_for(y.cols()), x, y)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_sgd_update(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    w: hip_tile::TensorMut<f32>,
    u: hip_tile::TensorMut<f32>,
    p: &SgdParams,
) -> hip_rt::Result<()> {
    assert_eq!(w.cols(), u.cols(), "sgd_update: length mismatch");
    sgd_update::launch(module, stream, grid_for(w.cols()), w, u, *p)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_running_update(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    rolling: hip_tile::TensorMut<f32>,
    batch_stat: hip_tile::Tensor<f32>,
    momentum: f32,
) -> hip_rt::Result<()> {
    assert_eq!(rolling.cols(), batch_stat.cols(), "running_update: length mismatch");
    running_update::launch(module, stream, grid_for(rolling.cols()), rolling, batch_stat, momentum)
}
