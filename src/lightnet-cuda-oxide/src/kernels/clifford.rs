//! Clifford rolling products and DropPath residual, with gather adjoints.
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[device]
    fn clifford_silu_slope(x: f32) -> f32 {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    }

    #[kernel]
    pub fn clifford_interaction(
        state: &[f32], context: &[f32], mut output: DisjointSlice<f32>,
        channels: usize, spatial: usize, shift: usize, inner: bool, pooled: bool,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let p = i % spatial;
        let c = (i / spatial) % channels;
        let b = i / (spatial * channels);
        let r = (c + shift) % channels;
        let j = (b * channels + r) * spatial + p;
        let ci = if pooled { b * channels + c } else { i };
        let cj = if pooled { b * channels + r } else { j };
        let product = state[i] * context[cj];
        let value = if inner {
            product / (1.0 + (-product).exp())
        } else {
            product - context[ci] * state[j]
        };
        if let Some(dst) = output.get_mut(idx) { *dst = value; }
    }

    // One thread owns one state gradient; incoming rolled contributions are
    // gathered, including both terms when shift == 0 (no atomic races).
    #[kernel]
    pub fn clifford_interaction_state_backward(
        state: &[f32], context: &[f32], grad: &[f32], mut output: DisjointSlice<f32>,
        channels: usize, spatial: usize, shift: usize, inner: bool, pooled: bool,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let p = i % spatial;
        let c = (i / spatial) % channels;
        let b = i / (spatial * channels);
        let r = (c + shift) % channels;
        let prev = (c + channels - shift) % channels;
        let j = (b * channels + prev) * spatial + p;
        let cr = if pooled { b * channels + r } else { (b * channels + r) * spatial + p };
        let cp = if pooled { b * channels + prev } else { j };
        let value = if inner {
            grad[i] * clifford_silu_slope(state[i] * context[cr]) * context[cr]
        } else {
            grad[i] * context[cr] - grad[j] * context[cp]
        };
        if let Some(dst) = output.get_mut(idx) { *dst = value; }
    }

    #[kernel]
    pub fn clifford_interaction_context_backward(
        state: &[f32], context: &[f32], grad: &[f32], mut output: DisjointSlice<f32>,
        channels: usize, spatial: usize, shift: usize, inner: bool, pooled: bool,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let c = if pooled { i % channels } else { (i / spatial) % channels };
        let b = if pooled { i / channels } else { i / (spatial * channels) };
        let prev = (c + channels - shift) % channels;
        let r = (c + shift) % channels;
        let begin = if pooled { 0 } else { i % spatial };
        let end = if pooled { spatial } else { begin + 1 };
        let mut value = 0.0;
        for p in begin..end {
            let cur = (b * channels + c) * spatial + p;
            let j = (b * channels + prev) * spatial + p;
            if inner {
                value += grad[j] * clifford_silu_slope(state[j] * context[i]) * state[j];
            } else {
                value += grad[j] * state[j] - grad[cur] * state[(b * channels + r) * spatial + p];
            }
        }
        if let Some(dst) = output.get_mut(idx) { *dst = value; }
    }

    #[kernel]
    pub fn clifford_residual(
        input: &[f32], mixed: &[f32], scale: &[f32], mask: &[f32],
        mut output: DisjointSlice<f32>, channels: usize, spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let c = (i / spatial) % channels;
        let b = i / (spatial * channels);
        if let Some(dst) = output.get_mut(idx) {
            *dst = input[i] + mixed[i] * (scale[c] * mask[b]);
        }
    }

    #[kernel]
    pub fn clifford_residual_backward(
        grad: &[f32], scale: &[f32], mask: &[f32],
        mut output: DisjointSlice<f32>, channels: usize, spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        if let Some(dst) = output.get_mut(idx) {
            *dst = grad[i] * mask[i / (spatial * channels)] * scale[(i / spatial) % channels];
        }
    }

    #[kernel]
    pub fn clifford_scale_backward(
        grad: &[f32], mixed: &[f32], mask: &[f32], mut output: DisjointSlice<f32>,
        channels: usize, spatial: usize,
    ) {
        let idx = thread::index_1d();
        let c = idx.get();
        if c >= channels { return; }
        let mut value = 0.0;
        for b in 0..mask.len() {
            for p in 0..spatial {
                let i = (b * channels + c) * spatial + p;
                value += grad[i] * mask[b] * mixed[i];
            }
        }
        if let Some(dst) = output.get_mut(idx) { *dst = value; }
    }
}
