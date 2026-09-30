//! Optimizer kernels.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Darknet SGD step (updates already point downhill):
    ///   u' = u - decay_times_batch * w
    ///   w_out = w + lr_over_batch * u'
    ///   u_out = momentum * u'
    #[kernel]
    pub fn sgd_update(
        w: &[f32],
        u: &[f32],
        mut w_out: DisjointSlice<f32>,
        mut u_out: DisjointSlice<f32>,
        lr_over_batch: f32,
        momentum: f32,
        decay_times_batch: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < w.len() {
            let wi = w[i];
            let decayed = u[i] - decay_times_batch * wi;
            if let Some(o) = w_out.get_mut(idx) {
                *o = wi + lr_over_batch * decayed;
            }
            if let Some(o) = u_out.get_mut(thread::index_1d()) {
                *o = momentum * decayed;
            }
        }
    }
}
