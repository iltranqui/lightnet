//! Entry kernels for the element bodies in `lightnet-kernels-shared`.
//!
//! Each entry computes the global thread index, calls the shared body and writes the
//! result. The hip_tile (AMD) backend has the same entries in
//! `third_party/hip_tile/kernels/src/shared_ops.rs`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use lightnet_kernels_shared as shared;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[i] = x[i] + bias[(i / spatial) % channels]`.
    #[kernel]
    pub fn shared_bias_add(
        x: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        spatial: usize,
        channels: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = shared::bias_add_elem(i, x, bias, spatial, channels);
        }
    }
}
