//! Flat elementwise kernels (one thread per element).

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// out[i] = x[i] + y[i]  (also serves as `shortcut_add`).
    #[kernel]
    pub fn vector_add(x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i] + y[i];
        }
    }

    /// out[i] = alpha * x[i] + y[i]
    #[kernel]
    pub fn saxpy(alpha: f32, x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = alpha * x[i] + y[i];
        }
    }
}
