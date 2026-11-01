//! Entry kernels for the element bodies in `lightnet-kernels-shared`.
//!
//! Each entry computes the global thread index, calls the shared body and writes the
//! result. The cuda-oxide backend has the same entries in
//! `src/lightnet-cuda-oxide/src/kernels/shared_ops.rs`.

#[cfg(target_arch = "amdgpu")]
use core::slice::from_raw_parts;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;
#[cfg(target_arch = "amdgpu")]
use lightnet_kernels_shared as shared;

/// `out[i] = x[i] + bias[(i / spatial) % channels]` over `x.cols()` elements.
#[kernel]
fn bias_add(
    x: Tensor<f32>,
    bias: Tensor<f32>,
    out: TensorMut<f32>,
    spatial: u32,
    channels: u32,
) {
    let i = block_id().1 * BLOCK_THREADS + thread_id();
    let n = x.cols();
    if i < n {
        // SAFETY: the host passes `1 x n` buffers, so the pointers are valid for the lengths.
        let (xs, bs) = unsafe {
            (
                from_raw_parts(x.ptr(), n as usize),
                from_raw_parts(bias.ptr(), bias.cols() as usize),
            )
        };
        let v = shared::bias_add_elem(i as usize, xs, bs, spatial as usize, channels as usize);
        // SAFETY: `i < n` and `out` has `n` elements.
        unsafe { *out.ptr().add(i as usize) = v };
    }
}
