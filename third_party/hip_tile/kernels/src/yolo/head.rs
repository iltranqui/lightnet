//! Elementwise logistic and per-pixel channel softmax heads.

#[cfg(target_arch = "amdgpu")]
use core::slice::from_raw_parts;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;
#[cfg(target_arch = "amdgpu")]
use lightnet_kernels_shared as shared;

#[kernel]
fn logistic(x: Tensor<f32>, y: TensorMut<f32>) {
    let i = block_id().1 * BLOCK_THREADS + thread_id();
    let n = x.rows() * x.cols();
    if i < n {
        #[cfg(target_arch = "amdgpu")]
        {
            // SAFETY: the caller provides contiguous, equally sized input and output tensors.
            let xs = unsafe { from_raw_parts(x.ptr(), n as usize) };
            let value = shared::logistic_elem::<crate::math::AmdMath>(xs[i as usize]);
            unsafe { *y.ptr().add(i as usize) = value };
        }
    }
}

#[kernel]
fn softmax_spatial(x: Tensor<f32>, y: TensorMut<f32>, temperature: f32) {
    let i = block_id().1 * BLOCK_THREADS + thread_id();
    let channels = x.rows();
    let plane = x.cols();
    let n = channels * plane;
    if i < n {
        #[cfg(target_arch = "amdgpu")]
        {
            let pixel = i % plane;
            let c = i / plane;
            // SAFETY: the caller provides a contiguous `[channels x plane]` input.
            let xs = unsafe { from_raw_parts(x.ptr(), n as usize) };
            let value = shared::softmax_spatial_elem::<crate::math::AmdMath>(
                xs, plane as usize, channels as usize, pixel as usize, c as usize, temperature,
            );
            unsafe { *y.ptr().add(i as usize) = value };
        }
    }
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_logistic(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
) -> hip_rt::Result<()> {
    assert!(x.rows() == y.rows() && x.cols() == y.cols() && x.ld() == x.cols() && y.ld() == y.cols(), "logistic: shape mismatch");
    let n = x.rows() * x.cols();
    logistic::launch(module, stream, (n.div_ceil(BLOCK_THREADS), 1), x, y)
}

#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_softmax_spatial(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    temperature: f32,
) -> hip_rt::Result<()> {
    assert!(x.rows() == y.rows() && x.cols() == y.cols() && x.ld() == x.cols() && y.ld() == y.cols(), "softmax_spatial: shape mismatch");
    assert!(temperature.is_finite() && temperature > 0.0, "softmax_spatial: invalid temperature");
    let n = x.rows() * x.cols();
    softmax_spatial::launch(module, stream, (n.div_ceil(BLOCK_THREADS), 1), x, y, temperature)
}
