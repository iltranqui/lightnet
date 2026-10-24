//! Shortcut kernels: source fit and weight gradients.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[b * to_len + i] = i < min(from_len, to_len) ? x[b * from_len + i] : 0`
    /// (flat per-sample crop / zero-pad, the Darknet mismatched shortcut).
    #[kernel]
    pub fn shortcut_fit(x: &[f32], mut out: DisjointSlice<f32>, from_len: usize, to_len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let b = g / to_len;
            let i = g - b * to_len;
            *o = if i < from_len {
                x[b * from_len + i]
            } else {
                0.0
            };
        }
    }

    /// One thread per shortcut coefficient slot. Each thread traverses its
    /// deterministic batch/feature range and reduces `delta * source`; a short
    /// source contributes only over its available flat prefix.
    #[kernel]
    pub fn shortcut_weight_gradient(
        delta: &[f32],
        source: &[f32],
        mut gradient: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
    ) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if slot >= layer_step {
            return;
        }
        let span = per_sample / layer_step;
        let start = slot * span;
        let end = (start + span).min(per_sample).min(source_per_sample);
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut e = start;
            while e < end {
                sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                e += 1;
            }
            b += 1;
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = sum;
        }
    }

    /// Update one source row of `[input][slot]` `delta * source` products.
    #[kernel]
    pub fn shortcut_weight_gradient_accumulate(
        accumulated: &[f32],
        delta: &[f32],
        source: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let mut value = accumulated[i];
        if i / layer_step == source_index && source_index < num_inputs {
            let slot = i % layer_step;
            let span = per_sample / layer_step;
            let start = slot * span;
            let end = (start + span).min(per_sample).min(source_per_sample);
            let mut sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                let mut e = start;
                while e < end {
                    sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                    e += 1;
                }
                b += 1;
            }
            value += sum;
        }
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    /// Exact per-slot normalization chain for shortcut coefficient gradients.
    /// `normalization`: 0 none, 1 relu, 2 softmax; `cpp_grad` copies raw z.
    #[kernel]
    pub fn shortcut_weight_normalized_gradient(
        raw: &[f32],
        products: &[f32],
        mut gradient: DisjointSlice<f32>,
        num_inputs: usize,
        layer_step: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let slot = i % layer_step;
        let mut value = products[i];
        if normalization == 1 && cpp_grad == 0 {
            let mut positive_sum = 0.0f32;
            let mut dot = 0.0f32;
            let mut input = 0usize;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let positive = raw[j].max(0.0);
                positive_sum += positive;
                dot += products[j] * positive;
                input += 1;
            }
            let sum = 0.0001f32 + positive_sum;
            value = if raw[i] > 0.0 {
                products[i] / sum - dot / (sum * sum)
            } else {
                0.0
            };
        } else if normalization == 2 && cpp_grad == 0 {
            let mut maximum = f32::NEG_INFINITY;
            let mut input = 0usize;
            while input < num_inputs {
                let raw_value = raw[input * layer_step + slot];
                if raw_value > maximum || input == 0 {
                    maximum = raw_value;
                }
                input += 1;
            }
            let mut exp_sum = 0.0f32;
            input = 0;
            while input < num_inputs {
                exp_sum += (raw[input * layer_step + slot] - maximum).exp();
                input += 1;
            }
            let sum = 0.0001f32 + exp_sum;
            let mut dot = 0.0f32;
            input = 0;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let effective = (raw[j] - maximum).exp() / sum;
                dot += products[j] * effective;
                input += 1;
            }
            let effective = (raw[i] - maximum).exp() / sum;
            value = effective * (products[i] - dot);
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }
}
