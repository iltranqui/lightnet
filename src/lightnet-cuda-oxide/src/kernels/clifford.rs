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
#[kernel]
pub fn clifford_fourier_coeff(input: &[f32], mut re: DisjointSlice<f32>, mut im: DisjointSlice<f32>,
    width: usize, height: usize, modes_h: usize, modes_w: usize) {
    let idx = thread::index_1d(); let i = idx.get();
    if i >= re.len() { return; }
    let modes = modes_h * modes_w; let mode = i % modes;
    let kh = mode / modes_w; let kw = mode % modes_w;
    let spatial = width * height; let start = (i / modes) * spatial;
    let mut a = 0.0; let mut b = 0.0;
    for y in 0..height { for x in 0..width {
        let theta = 6.283185307179586f32 * ((kh * y) as f32 / height as f32 + (kw * x) as f32 / width as f32);
        a += input[start + y * width + x] * theta.cos();
        b -= input[start + y * width + x] * theta.sin();
    }}
    if let Some(v) = re.get_mut(idx) { *v = a; }
    if let Some(v) = im.get_mut(thread::index_1d()) { *v = b; }
}

#[kernel]
pub fn clifford_fourier_synthesis(re: &[f32], im: &[f32], wr: &[f32], wi: &[f32],
    mut out: DisjointSlice<f32>, channels: usize, width: usize, height: usize,
    modes_h: usize, modes_w: usize, adjoint: bool) {
    let idx = thread::index_1d(); let i = idx.get(); if i >= out.len() { return; }
    let spatial = width * height; let modes = modes_h * modes_w;
    let p = i % spatial; let channel = (i / spatial) % channels;
    let start = (i / spatial) * modes; let mut value = 0.0;
    for mode in 0..modes {
        let kh = mode / modes_w; let kw = mode % modes_w;
        let theta = 6.283185307179586f32 * ((kh * (p / width)) as f32 / height as f32 + (kw * (p % width)) as f32 / width as f32);
        let a = re[start + mode]; let b = im[start + mode];
        let u = wr[channel * modes + mode]; let v = wi[channel * modes + mode];
        let (yr, yi) = if adjoint { (a * u + b * v, -a * v + b * u) } else { (a * u - b * v, a * v + b * u) };
        let mult = if kw == 0 { 1.0 } else { 2.0 };
        value += mult * (yr * theta.cos() - yi * theta.sin()) / spatial as f32;
    }
    if let Some(v) = out.get_mut(idx) { *v = value; }
}

#[kernel]
pub fn clifford_fourier_weights(re: &[f32], im: &[f32], dre: &[f32], dim: &[f32],
    mut dwre: DisjointSlice<f32>, mut dwim: DisjointSlice<f32>, batch: usize,
    channels: usize, spatial: usize, modes_h: usize, modes_w: usize) {
    let idx = thread::index_1d(); let i = idx.get(); if i >= dwre.len() { return; }
    let modes = modes_h * modes_w; let kw = (i % modes) % modes_w;
    let scale = (if kw == 0 { 1.0 } else { 2.0 }) / spatial as f32;
    let mut a = 0.0; let mut b = 0.0;
    for n in 0..batch { let j = n * channels * modes + i;
        let dr = dre[j] * scale; let di = dim[j] * scale;
        a += dr * re[j] + di * im[j]; b += -dr * im[j] + di * re[j];
    }
    if let Some(v) = dwre.get_mut(idx) { *v = a; }
    if let Some(v) = dwim.get_mut(thread::index_1d()) { *v = b; }
}

}
