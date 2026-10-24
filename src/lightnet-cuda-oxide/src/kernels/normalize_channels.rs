//! Channel normalization (softmax/norm activations) forward and backward kernels.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// NCHW channel normalization at one `(batch, spatial)` position per thread.
    /// `normalization`: 0 ReLU-normalize, 1 softmax with zero shift, 2 softmax
    /// with channel-max subtraction.
    #[kernel]
    pub fn normalize_channels(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let value = if normalization == 0 {
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                let x = input[base + k * spatial];
                if x > 0.0 {
                    sum += x;
                }
                k += 1;
            }
            let x = input[i];
            if x > 0.0 { x / sum } else { 0.0 }
        } else {
            let maximum = if normalization == 2 {
                let mut maximum = -f32::MAX;
                let mut k = 0usize;
                while k < channels {
                    let x = input[base + k * spatial];
                    if x > maximum || k == 0 {
                        maximum = x;
                    }
                    k += 1;
                }
                maximum
            } else {
                0.0
            };
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                sum += (input[base + k * spatial] - maximum).exp();
                k += 1;
            }
            let value = (input[i] - maximum).exp() / sum;
            if value.is_finite() { value } else { 0.0 }
        };
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    /// Exact channel-normalization adjoint or the C++ CUDA diagonal shortcut.
    #[kernel]
    pub fn normalize_channels_backward(
        activation_input: &[f32],
        output: &[f32],
        delta: &[f32],
        mut gradient: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let y = output[i];
        let value = if cpp_grad != 0 {
            if normalization == 0 {
                if y > 0.0 { delta[i] * y } else { delta[i] }
            } else {
                let value = delta[i] * y * (1.0 - y);
                if value.is_finite() { value } else { 0.0 }
            }
        } else {
            let mut dot = 0.0f32;
            let mut k = 0usize;
            while k < channels {
                let j = base + k * spatial;
                dot += output[j] * delta[j];
                k += 1;
            }
            if normalization == 0 {
                let mut sum = 0.0001f32;
                k = 0;
                while k < channels {
                    let x = activation_input[base + k * spatial];
                    if x > 0.0 {
                        sum += x;
                    }
                    k += 1;
                }
                if activation_input[i] > 0.0 {
                    (delta[i] - dot) / sum
                } else {
                    0.0
                }
            } else if normalization == 2 {
                // epsilon outside exp(x-max) leaves a max-chain term.
                // First maximum is the selected subgradient at exact ties.
                let mut maximum = activation_input[base];
                let mut first_max = 0usize;
                let mut correction = 0.0f32;
                k = 0;
                while k < channels {
                    let j = base + k * spatial;
                    if activation_input[j] > maximum {
                        maximum = activation_input[j];
                        first_max = k;
                    }
                    correction += output[j] * (delta[j] - dot);
                    k += 1;
                }
                let g = y * (delta[i] - dot);
                if (i / spatial) % channels == first_max {
                    g - correction
                } else {
                    g
                }
            } else {
                y * (delta[i] - dot)
            }
        };
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }
}
