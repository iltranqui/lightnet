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

/// Route (channel concat) copy of `n = batch * channels * stride` elements:
/// `dst[to] = src[from]` with `(from, to)` from `route_copy_index`. Grouped routes use a
/// nonzero `src_offset`. The AMD route layer uses zero-copy channel views where it can;
/// this entry serves the copy case and checks the shared index math on the GPU.
#[kernel]
fn route_copy(
    src: Tensor<f32>,
    dst: TensorMut<f32>,
    n: u32,
    channels: u32,
    stride: u32,
    src_total: u32,
    src_offset: u32,
    dst_total: u32,
    dst_offset: u32,
) {
    let i = block_id().1 * BLOCK_THREADS + thread_id();
    if i < n {
        #[cfg(target_arch = "amdgpu")]
        {
            let (from, to) = shared::route_copy_index(
                i as usize,
                channels as usize,
                stride as usize,
                (src_total as usize, src_offset as usize),
                (dst_total as usize, dst_offset as usize),
            );
            // SAFETY: the host sizes `src` and `dst` to cover every `(from, to)` for `i < n`.
            unsafe { *dst.ptr().add(to) = *src.ptr().add(from) };
        }
    }
}

/// Copy `channels` channels of `stride` pixels from `src` (a `src_total`-channel plane stack,
/// starting at channel `src_offset`) into `dst` (`dst_total` channels, at `dst_offset`), batch 1.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_route_copy(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    src: hip_tile::Tensor<f32>,
    dst: hip_tile::TensorMut<f32>,
    (channels, stride): (u32, u32),
    (src_total, src_offset): (u32, u32),
    (dst_total, dst_offset): (u32, u32),
) -> hip_rt::Result<()> {
    let n = channels * stride;
    route_copy::launch(
        module,
        stream,
        (n.div_ceil(BLOCK_THREADS), 1),
        src,
        dst,
        n,
        channels,
        stride,
        src_total,
        src_offset,
        dst_total,
        dst_offset,
    )
}
