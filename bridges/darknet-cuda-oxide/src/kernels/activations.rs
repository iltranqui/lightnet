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
