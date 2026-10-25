//! Optimizer kernels.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// One element of the multi-tensor SGD step: `u = (has_grad ? u + g : u)`, then
/// the arithmetic of `sanitized_sgd_update`, in the same order.
#[device]
pub fn sgd_elem(
    w: &[f32],
    u: &[f32],
    g: &[f32],
    w_out: *mut f32,
    u_out: *mut f32,
    i: usize,
    has_grad: u32,
    lr_over_batch: f32,
    momentum: f32,
    decay_times_batch: f32,
    fix_w: u32,
) {
    let w0 = w[i];
    let wi = if fix_w != 0 && w0 - w0 != 0.0 {
        1.0 / (i as f32 + 1.0)
    } else {
        w0
    };
    let u_acc = if has_grad != 0 { u[i] + g[i] } else { u[i] };
    let ui = if u_acc - u_acc != 0.0 { 0.0 } else { u_acc };
    let decayed = ui - decay_times_batch * wi;
    unsafe {
        *w_out.add(i) = wi + lr_over_batch * decayed;
        *u_out.add(i) = momentum * decayed;
    }
}

/// One element of the multi-tensor Adam step (see `sanitized_adam_update`).
#[device]
pub fn adam_elem(
    w: &[f32],
    u: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    w_out: *mut f32,
    m_out: *mut f32,
    v_out: *mut f32,
    u_out: *mut f32,
    i: usize,
    has_grad: u32,
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
    let w0 = w[i];
    let wi = if fix_w != 0 && w0 - w0 != 0.0 {
        1.0 / (i as f32 + 1.0)
    } else {
        w0
    };
    let u_acc = if has_grad != 0 { u[i] + g[i] } else { u[i] };
    let ui = if u_acc - u_acc != 0.0 { 0.0 } else { u_acc };
    let gr = ui - coupled_decay * wi;
    let mi = beta1 * m[i] + (1.0 - beta1) * gr;
    let vi = beta2 * v[i] + (1.0 - beta2) * (gr * gr);
    let mhat = mi / bias_correction1;
    let vhat = vi / bias_correction2;
    unsafe {
        *w_out.add(i) = wi + lr * mhat / (vhat.sqrt() + eps) - decoupled_decay * wi;
        *m_out.add(i) = mi;
        *v_out.add(i) = vi;
        *u_out.add(i) = 0.0;
    }
}

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
            let wi = if fix_w != 0 && w0 - w0 != 0.0 {
                1.0 / (i as f32 + 1.0)
            } else {
                w0
            };
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
            let wi = if fix_w != 0 && w0 - w0 != 0.0 {
                1.0 / (i as f32 + 1.0)
            } else {
                w0
            };
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

    /// `sanitized_sgd_update` over up to three tensors (weights, biases, BN
    /// scales of one layer) in one launch: thread `i` handles element `i` of
    /// the concatenation; unused segments have `n == 0`.  `has_grad[k] != 0`
    /// first adds `g_k` to `u_k` (the pending gradient accumulation).
    #[kernel]
    pub fn sanitized_sgd_update3(
        w0: &[f32],
        u0: &[f32],
        g0: &[f32],
        mut wo0: DisjointSlice<f32>,
        mut uo0: DisjointSlice<f32>,
        w1: &[f32],
        u1: &[f32],
        g1: &[f32],
        mut wo1: DisjointSlice<f32>,
        mut uo1: DisjointSlice<f32>,
        w2: &[f32],
        u2: &[f32],
        g2: &[f32],
        mut wo2: DisjointSlice<f32>,
        mut uo2: DisjointSlice<f32>,
        n0: usize,
        n1: usize,
        n2: usize,
        hg0: u32,
        hg1: u32,
        hg2: u32,
        fix0: u32,
        lr_over_batch: f32,
        momentum: f32,
        decay0: f32,
        decay1: f32,
        decay2: f32,
    ) {
        let i = thread::index_1d().get();
        if i < n0 {
            sgd_elem(
                w0,
                u0,
                g0,
                wo0.as_mut_ptr(),
                uo0.as_mut_ptr(),
                i,
                hg0,
                lr_over_batch,
                momentum,
                decay0,
                fix0,
            );
        } else if i < n0 + n1 {
            sgd_elem(
                w1,
                u1,
                g1,
                wo1.as_mut_ptr(),
                uo1.as_mut_ptr(),
                i - n0,
                hg1,
                lr_over_batch,
                momentum,
                decay1,
                0,
            );
        } else if i < n0 + n1 + n2 {
            sgd_elem(
                w2,
                u2,
                g2,
                wo2.as_mut_ptr(),
                uo2.as_mut_ptr(),
                i - n0 - n1,
                hg2,
                lr_over_batch,
                momentum,
                decay2,
                0,
            );
        }
    }

    /// `sanitized_adam_update` over up to three tensors in one launch (see
    /// `sanitized_sgd_update3`); the decays differ per segment.
    #[kernel]
    pub fn sanitized_adam_update3(
        w0: &[f32],
        u0: &[f32],
        g0: &[f32],
        m0: &[f32],
        v0: &[f32],
        mut wo0: DisjointSlice<f32>,
        mut mo0: DisjointSlice<f32>,
        mut vo0: DisjointSlice<f32>,
        mut uo0: DisjointSlice<f32>,
        w1: &[f32],
        u1: &[f32],
        g1: &[f32],
        m1: &[f32],
        v1: &[f32],
        mut wo1: DisjointSlice<f32>,
        mut mo1: DisjointSlice<f32>,
        mut vo1: DisjointSlice<f32>,
        mut uo1: DisjointSlice<f32>,
        w2: &[f32],
        u2: &[f32],
        g2: &[f32],
        m2: &[f32],
        v2: &[f32],
        mut wo2: DisjointSlice<f32>,
        mut mo2: DisjointSlice<f32>,
        mut vo2: DisjointSlice<f32>,
        mut uo2: DisjointSlice<f32>,
        n0: usize,
        n1: usize,
        n2: usize,
        hg0: u32,
        hg1: u32,
        hg2: u32,
        fix0: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        coupled0: f32,
        decoupled0: f32,
        coupled1: f32,
        decoupled1: f32,
        coupled2: f32,
        decoupled2: f32,
    ) {
        let i = thread::index_1d().get();
        if i < n0 {
            adam_elem(
                w0,
                u0,
                g0,
                m0,
                v0,
                wo0.as_mut_ptr(),
                mo0.as_mut_ptr(),
                vo0.as_mut_ptr(),
                uo0.as_mut_ptr(),
                i,
                hg0,
                lr,
                beta1,
                beta2,
                eps,
                coupled0,
                decoupled0,
                bias_correction1,
                bias_correction2,
                fix0,
            );
        } else if i < n0 + n1 {
            adam_elem(
                w1,
                u1,
                g1,
                m1,
                v1,
                wo1.as_mut_ptr(),
                mo1.as_mut_ptr(),
                vo1.as_mut_ptr(),
                uo1.as_mut_ptr(),
                i - n0,
                hg1,
                lr,
                beta1,
                beta2,
                eps,
                coupled1,
                decoupled1,
                bias_correction1,
                bias_correction2,
                0,
            );
        } else if i < n0 + n1 + n2 {
            adam_elem(
                w2,
                u2,
                g2,
                m2,
                v2,
                wo2.as_mut_ptr(),
                mo2.as_mut_ptr(),
                vo2.as_mut_ptr(),
                uo2.as_mut_ptr(),
                i - n0 - n1,
                hg2,
                lr,
                beta1,
                beta2,
                eps,
                coupled2,
                decoupled2,
                bias_correction1,
                bias_correction2,
                0,
            );
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

    /// A short serial chunk per GPU thread avoids floating-point atomics and
    /// never downloads the inspected parameter/gradient tensor.
    #[kernel]
    pub fn nonfinite_partial(input: &[f32], mut flags: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let chunk = idx.get();
        if chunk >= flags.len() { return; }
        let start = chunk * 256;
        let count = (input.len() - start).min(256);
        let mut found = 0.0;
        for offset in 0..count {
            if !input[start + offset].is_finite() { found = 1.0; break; }
        }
        if let Some(dst) = flags.get_mut(idx) { *dst = found; }
    }

    #[kernel]
    pub fn nonfinite_reduce(flags: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if idx.get() != 0 { return; }
        let mut found = 0.0;
        for i in 0..flags.len() { if flags[i] != 0.0 { found = 1.0; break; } }
        if let Some(dst) = output.get_mut(idx) { *dst = found; }
    }
}
