//! Route (channel concat) forward and backward.
//!
//! Both directions are the same strided channel-slice copy, so one kernel
//! serves both.  The host launches it once per route input; every thread owns
//! one element of the copied slice, so writes never collide and the result is
//! deterministic.  Grouped routes (`groups` / `group_id`) are just a nonzero
//! `src_offset` into an input with `src_total` channels.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// For `i` over `batch * channels * stride`: with `b`, `c`, `s` the batch,
    /// slice-channel and spatial index,
    /// `dst[(b*dst_total + dst_offset + c)*stride + s] =
    ///  src[(b*src_total + src_offset + c)*stride + s]`.
    #[kernel]
    pub fn route_copy(
        src: &[f32],
        mut dst: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        src_total: usize,
        src_offset: usize,
        dst_total: usize,
        dst_offset: usize,
        stride: usize,
    ) {
        let i = thread::index_1d().get();
        let per_batch = channels * stride;
        if i >= batch * per_batch {
            return;
        }
        let b = i / per_batch;
        let r = i % per_batch;
        let c = r / stride;
        let s = r % stride;
        let from = (b * src_total + src_offset + c) * stride + s;
        let to = (b * dst_total + dst_offset + c) * stride + s;
        // SAFETY: each thread writes a distinct `to`; the host checks that
        // `dst` covers `batch * dst_total * stride` elements.
        unsafe {
            *dst.as_mut_ptr().add(to) = src[from];
        }
    }
}
