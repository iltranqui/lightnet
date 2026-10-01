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

    /// Darknet `adam_update_gpu` (updates point downhill, `u` is the
    /// batch-summed gradient).  `coupled_decay` is `decay * batch` folded into
    /// `u` like C++ (L2 inside the moments); `decoupled_decay` is `lr * decay`
    /// applied to `w` directly (AdamW).  Exactly one of them is non-zero.
    ///   u' = u - coupled_decay * w
    ///   m' = b1 * m + (1 - b1) * u'
    ///   v' = b2 * v + (1 - b2) * u'^2
    ///   w' = w + lr * (m' / bc1) / (sqrt(v' / bc2) + eps) - decoupled_decay * w
    /// with bias corrections `bc1 = 1 - b1^t`, `bc2 = 1 - b2^t` (host side).
    #[kernel]
    pub fn adam_update(
        w: &[f32],
        u: &[f32],
        m: &[f32],
        v: &[f32],
        mut w_out: DisjointSlice<f32>,
        mut m_out: DisjointSlice<f32>,
        mut v_out: DisjointSlice<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        coupled_decay: f32,
        decoupled_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < w.len() {
            let wi = w[i];
            let g = u[i] - coupled_decay * wi;
            let mi = beta1 * m[i] + (1.0 - beta1) * g;
            let vi = beta2 * v[i] + (1.0 - beta2) * (g * g);
            let mhat = mi / bias_correction1;
            let vhat = vi / bias_correction2;
            if let Some(o) = w_out.get_mut(idx) {
                *o = wi + lr * mhat / (vhat.sqrt() + eps) - decoupled_decay * wi;
            }
            if let Some(o) = m_out.get_mut(thread::index_1d()) {
                *o = mi;
            }
            if let Some(o) = v_out.get_mut(thread::index_1d()) {
                *o = vi;
            }
        }
    }

    /// Darknet `reset_nan_and_inf` (`fix == 0`: non-finite -> 0) and
    /// `fix_nan_and_inf` (`fix != 0`: non-finite -> `1 / (index + 1)`).
    #[kernel]
    pub fn sanitize_nan_inf(x: &[f32], mut out: DisjointSlice<f32>, fix: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            let v = x[i];
            // `v - v` is 0 for every finite v and NaN for NaN/Inf.
            let bad = v - v != 0.0;
            if let Some(o) = out.get_mut(idx) {
                *o = if !bad {
                    v
                } else if fix != 0 {
                    1.0 / (i as f32 + 1.0)
                } else {
                    0.0
                };
            }
        }
    }

    /// `sanitize_nan_inf(u, reset)` + (`fix_w != 0`) `sanitize_nan_inf(w, fix)` +
    /// `sgd_update` in one pass; same arithmetic, in the same order, as the
    /// three separate kernels.
    #[kernel]
    pub fn sanitized_sgd_update(
        w: &[f32],
        u: &[f32],
        mut w_out: DisjointSlice<f32>,
        mut u_out: DisjointSlice<f32>,
        lr_over_batch: f32,
        momentum: f32,
        decay_times_batch: f32,
        fix_w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < w.len() {
            let w0 = w[i];
            let wi = if fix_w != 0 && w0 - w0 != 0.0 { 1.0 / (i as f32 + 1.0) } else { w0 };
            let u0 = u[i];
            let ui = if u0 - u0 != 0.0 { 0.0 } else { u0 };
            let decayed = ui - decay_times_batch * wi;
            if let Some(o) = w_out.get_mut(idx) {
                *o = wi + lr_over_batch * decayed;
            }
            if let Some(o) = u_out.get_mut(thread::index_1d()) {
                *o = momentum * decayed;
            }
        }
    }

    /// Sanitize + `adam_update` in one pass; also writes the zeroed updates.
    #[kernel]
    pub fn sanitized_adam_update(
        w: &[f32],
        u: &[f32],
        m: &[f32],
        v: &[f32],
        mut w_out: DisjointSlice<f32>,
        mut m_out: DisjointSlice<f32>,
        mut v_out: DisjointSlice<f32>,
        mut u_out: DisjointSlice<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        coupled_decay: f32,
        decoupled_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        fix_w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < w.len() {
            let w0 = w[i];
            let wi = if fix_w != 0 && w0 - w0 != 0.0 { 1.0 / (i as f32 + 1.0) } else { w0 };
            let u0 = u[i];
            let ui = if u0 - u0 != 0.0 { 0.0 } else { u0 };
            let g = ui - coupled_decay * wi;
            let mi = beta1 * m[i] + (1.0 - beta1) * g;
            let vi = beta2 * v[i] + (1.0 - beta2) * (g * g);
            let mhat = mi / bias_correction1;
            let vhat = vi / bias_correction2;
            if let Some(o) = w_out.get_mut(idx) {
                *o = wi + lr * mhat / (vhat.sqrt() + eps) - decoupled_decay * wi;
            }
            if let Some(o) = m_out.get_mut(thread::index_1d()) {
                *o = mi;
            }
            if let Some(o) = v_out.get_mut(thread::index_1d()) {
                *o = vi;
            }
            if let Some(o) = u_out.get_mut(thread::index_1d()) {
                *o = 0.0;
            }
        }
    }

    /// Batch-norm inference fold: `scale = gamma / sqrt(var + eps)`,
    /// `bias_out = bias - mean * scale`.
    #[kernel]
    pub fn bn_fold(
        gamma: &[f32],
        mean: &[f32],
        variance: &[f32],
        bias: &[f32],
        mut scale_out: DisjointSlice<f32>,
        mut bias_out: DisjointSlice<f32>,
        eps: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < gamma.len() {
            let scale = gamma[i] / (variance[i] + eps).sqrt();
            if let Some(o) = scale_out.get_mut(idx) {
                *o = scale;
            }
            if let Some(o) = bias_out.get_mut(thread::index_1d()) {
                *o = bias[i] - mean[i] * scale;
            }
        }
    }
}
