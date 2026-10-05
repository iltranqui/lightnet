//! Activation forward / backward.  Formulas follow the Rust CPU reference in
//! `darknet-core` (`activate` and the gradient helpers), not the C++ quirks.
//! `kind` mirrors `F32ActivationKind` (see `activation_code` in lib.rs).

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

pub const ACT_LINEAR: u32 = 0;
pub const ACT_LOGISTIC: u32 = 1;
pub const ACT_RELU: u32 = 2;
pub const ACT_LEAKY: u32 = 3;
pub const ACT_MISH: u32 = 4;
pub const ACT_SWISH: u32 = 5;
pub const ACT_TANH: u32 = 6;
pub const ACT_RELU6: u32 = 7;
pub const ACT_RAMP: u32 = 8;
pub const ACT_GELU: u32 = 9;

#[device]
pub fn activate(kind: u32, x: f32) -> f32 {
    if kind == 1 {
        1.0 / (1.0 + (-x).exp())
    } else if kind == 2 {
        x.max(0.0)
    } else if kind == 3 {
        if x > 0.0 { x } else { 0.1 * x }
    } else if kind == 4 {
        x * (1.0 + x.exp()).ln().tanh()
    } else if kind == 5 {
        x / (1.0 + (-x).exp())
    } else if kind == 6 {
        x.tanh()
    } else if kind == 7 {
        x.max(0.0).min(6.0)
    } else if kind == 8 {
        // C++ `x*(x>0) + .1f*x` (same rounding as the CPU runtime).
        (if x > 0.0 { x } else { 0.0 }) + 0.1 * x
    } else if kind == 9 {
        0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh())
    } else {
        x
    }
}

/// d activate / d x, evaluated at the pre-activation input `x`.
#[device]
pub fn gradient(kind: u32, x: f32) -> f32 {
    if kind == 1 {
        let s = 1.0 / (1.0 + (-x).exp());
        s * (1.0 - s)
    } else if kind == 2 {
        if x > 0.0 { 1.0 } else { 0.0 }
    } else if kind == 3 {
        if x > 0.0 { 1.0 } else { 0.1 }
    } else if kind == 4 {
        let softplus = if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            (1.0 + x.exp()).ln()
        };
        let grad_sp = 1.0 - (-softplus).exp();
        let tsp = softplus.tanh();
        let grad_tsp = (1.0 - tsp * tsp) * grad_sp;
        x * grad_tsp + tsp
    } else if kind == 5 {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    } else if kind == 6 {
        let t = x.tanh();
        1.0 - t * t
    } else if kind == 7 {
        if x > 0.0 && x < 6.0 { 1.0 } else { 0.0 }
    } else if kind == 8 {
        if x > 0.0 { 1.1 } else { 0.1 }
    } else if kind == 9 {
        let t = (0.797885 * x + 0.035677 * x * x * x).tanh();
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (0.797885 + 3.0 * 0.035677 * x * x)
    } else {
        1.0
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn activation(x: &[f32], mut out: DisjointSlice<f32>, kind: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = activate(kind, x[i]);
        }
    }

    /// out[i] = delta[i] * gradient(x[i]); `x` is the activation input.
    #[kernel]
    pub fn activation_backward(
        x: &[f32],
        delta: &[f32],
        mut out: DisjointSlice<f32>,
        kind: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = delta[i] * gradient(kind, x[i]);
        }
    }
}

/// Row-wise LayerNorm (`rows x cols`) and its input-gradient, for the
/// transformer family.  One block per row keeps each row's reduction inside a
/// single thread group, so no second pass over the data is needed.
#[cuda_module]
pub mod layernorm {
    use super::*;
    use cuda_device::{kernel, thread};

    /// Longest row this kernel accepts.  The reduction lives in registers (one
    /// thread per row, one block per row), so the only limit is how much a thread
    /// can keep live; the host rejects anything longer.
    pub const LAYERNORM_MAX_COLS: usize = 1024;

    #[kernel]
    pub fn layernorm_forward(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
        has_gamma: u32,
        has_beta: u32,
    ) {
        let r = thread::blockIdx_x() as usize;
        if r >= rows {
            return;
        }
        let base = r * cols;
        let mut mean = 0.0f32;
        let mut i = 0usize;
        while i < cols {
            let v = x[base + i];
            mean += if v.is_finite() { v } else { 0.0 };
            i += 1;
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        i = 0;
        while i < cols {
            let v = x[base + i];
            let d = (if v.is_finite() { v } else { 0.0 }) - mean;
            var += d * d;
            i += 1;
        }
        var /= cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        i = 0;
        while i < cols {
            let v = x[base + i];
            let normalised = ((if v.is_finite() { v } else { 0.0 }) - mean) * inv_std;
            let g = if has_gamma != 0 { gamma[i] } else { 1.0 };
            let b = if has_beta != 0 { beta[i] } else { 0.0 };
            // SAFETY: block `r` only ever writes `r * cols + i`, and each block
            // owns a distinct `r`, so the (block, i) pairs cover the output once.
            unsafe { out.as_mut_ptr().add(base + i).write(g * normalised + b) };
            i += 1;
        }
    }

    #[kernel]
    pub fn layernorm_backward(
        dy: &[f32],
        x: &[f32],
        gamma: &[f32],
        mut dx: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
        has_gamma: u32,
    ) {
        let r = thread::blockIdx_x() as usize;
        if r >= rows {
            return;
        }
        let base = r * cols;
        let mut mean = 0.0f32;
        let mut i = 0usize;
        while i < cols {
            let v = x[base + i];
            mean += if v.is_finite() { v } else { 0.0 };
            i += 1;
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        i = 0;
        while i < cols {
            let v = x[base + i];
            let d = (if v.is_finite() { v } else { 0.0 }) - mean;
            var += d * d;
            i += 1;
        }
        var /= cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let inv_c = 1.0 / cols as f32;
        let mut mean_g = 0.0f32;
        let mut mean_gx = 0.0f32;
        i = 0;
        while i < cols {
            let v = x[base + i];
            let xhat = ((if v.is_finite() { v } else { 0.0 }) - mean) * inv_std;
            let g = dy[base + i] * (if has_gamma != 0 { gamma[i] } else { 1.0 }) * inv_std;
            mean_g += g;
            mean_gx += g * xhat;
            i += 1;
        }
        mean_g *= inv_c;
        mean_gx *= inv_c;
        i = 0;
        while i < cols {
            let v = x[base + i];
            let xhat = ((if v.is_finite() { v } else { 0.0 }) - mean) * inv_std;
            let g = dy[base + i] * (if has_gamma != 0 { gamma[i] } else { 1.0 }) * inv_std;
            // SAFETY: as above, one owner per output element.
            unsafe { dx.as_mut_ptr().add(base + i).write(g - mean_g - xhat * mean_gx) };
            i += 1;
        }
    }
}
