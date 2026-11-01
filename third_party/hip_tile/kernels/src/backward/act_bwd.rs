//! Activation gradient (`gradient_array`, applied to the delta *before* batch-norm backward):
//! leaky ReLU scales the delta by 0.1 where the activated output is negative
//! (`activations.cpp:gradient`, the sign of the output equals the sign of the pre-activation).

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

/// `delta[i] *= 0.1` where `post[i] < 0`.
#[kernel]
fn leaky_bwd(delta: TensorMut<f32>, post: Tensor<f32>) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < delta.cols() as usize && unsafe { *post.ptr().add(i) } < 0.0 {
        unsafe { *delta.ptr().add(i) *= 0.1 };
    }
}

/// `delta`, `post`: flat `[batch*C*H*W]` vectors.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_leaky_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    delta: hip_tile::TensorMut<f32>,
    post: hip_tile::Tensor<f32>,
) -> hip_rt::Result<()> {
    assert_eq!(delta.cols(), post.cols(), "leaky_bwd: length mismatch");
    leaky_bwd::launch(module, stream, (delta.cols().div_ceil(BLOCK_THREADS), 1), delta, post)
}
