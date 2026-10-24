//! EML convolution combine kernels with their softplus/sigmoid helpers.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[device]
    fn eml_softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            x.exp().ln_1p()
        }
    }

    #[device]
    fn eml_sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    /// EML convolution's elementwise combination of its parallel conv outputs.
    #[kernel]
    pub fn eml_convolutional_forward(
        x_branch: &[f32],
        self_branch: &[f32],
        residual: &[f32],
        mut output: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
        has_residual: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = x_branch[i].max(-clamp).min(clamp);
        let y = self_branch[i];
        let mut value = scale * (x.exp() - (eml_softplus(y) + eps).ln());
        if has_residual != 0 {
            value += residual[i];
        }
        if let Some(out) = output.get_mut(idx) {
            *out = value;
        }
    }

    /// Exact EML branch adjoints, with the source's strict clamp gate.
    #[kernel]
    pub fn eml_convolutional_backward(
        x_branch: &[f32],
        self_branch: &[f32],
        output_grad: &[f32],
        mut x_gradient: DisjointSlice<f32>,
        mut self_gradient: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x_gradient.len() {
            return;
        }
        let x_value = x_branch[i];
        let dx = if x_value > -clamp && x_value < clamp {
            x_value.max(-clamp).min(clamp).exp()
        } else {
            0.0
        };
        let y = self_branch[i];
        let dy = -eml_sigmoid(y) / (eml_softplus(y) + eps);
        let scaled_delta = output_grad[i] * scale;
        // SAFETY: every thread writes only element `i` in both output arrays.
        unsafe {
            *x_gradient.as_mut_ptr().add(i) = scaled_delta * dx;
            *self_gradient.as_mut_ptr().add(i) = scaled_delta * dy;
        }
    }
}
