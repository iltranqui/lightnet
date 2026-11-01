//! Entry kernels for the element bodies in `lightnet-kernels-shared`.
//!
//! Each entry computes the global thread index, calls the shared body and writes the
//! result. The hip_tile (AMD) backend has the same entries in
//! `third_party/hip_tile/kernels/src/shared_ops.rs`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use lightnet_kernels_shared as shared;

/// Native `exp` only: `tanh` falls back to the shared exp-based default, which is what the
/// AMD kernels run. Used to measure the native-`tanh` override against it.
pub struct ExpOnlyMath;

impl shared::DeviceMath for ExpOnlyMath {
    #[inline(always)]
    fn exp(x: f32) -> f32 {
        x.exp()
    }
}

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

    /// `native[i] = OxideMath::tanh(x[i])`, `shared[i]` = the exp-based shared default.
    #[kernel]
    pub fn shared_tanh_pair(x: &[f32], mut native: DisjointSlice<f32>, mut exp_based: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = native.get_mut(idx) {
            *o = shared::tanh_elem::<crate::kernels::activations::OxideMath>(x[i]);
        }
        if let Some(o) = exp_based.get_mut(thread::index_1d()) {
            *o = shared::tanh_elem::<ExpOnlyMath>(x[i]);
        }
    }
}
