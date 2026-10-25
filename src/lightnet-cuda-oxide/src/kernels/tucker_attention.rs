//! Tucker window attention and its exact adjoint. Padding tokens participate
//! in attention; finite scores are neither clamped nor given a padding mask.
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[device]
    fn tucker_sanitize(value: f32) -> f32 {
        if value.is_nan() { 0.0 }
        else if value == f32::INFINITY { 80.0 }
        else if value == f32::NEG_INFINITY { -80.0 }
        else { value }
    }

    #[kernel]
    pub fn tucker_scores(q: &[f32], k: &[f32], mut scores: DisjointSlice<f32>,
        tokens: usize, channels: usize, heads: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= scores.len() { return; }
        let tk = i % tokens;
        let tq = (i / tokens) % tokens;
        let h = (i / (tokens * tokens)) % heads;
        let w = i / (tokens * tokens * heads);
        let dim = channels / heads;
        let qi = (w * tokens + tq) * channels + h * dim;
        let ki = (w * tokens + tk) * channels + h * dim;
        let mut dot = 0.0;
        for d in 0..dim { dot += q[qi + d] * k[ki + d]; }
        if let Some(dst) = scores.get_mut(idx) { *dst = dot * (1.0 / (dim as f32).sqrt()); }
    }

    #[kernel]
    pub fn tucker_block_maxima(scores: &[f32], mut maxima: DisjointSlice<f32>,
        tokens: usize, block: usize, blocks: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= maxima.len() { return; }
        let b = i % blocks;
        let row = i / blocks;
        let start = b * block;
        let end = (start + block).min(tokens);
        let mut maximum = f32::NEG_INFINITY;
        for t in start..end { maximum = maximum.max(tucker_sanitize(scores[row * tokens + t])); }
        if let Some(dst) = maxima.get_mut(idx) { *dst = maximum; }
    }

    #[kernel]
    pub fn tucker_select_blocks(maxima: &[f32], mut selected: DisjointSlice<f32>,
        tokens: usize, block: usize, blocks: usize, topk: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= selected.len() { return; }
        let b = i % blocks;
        let row = i / blocks;
        let local = (row % tokens) / block;
        let value = maxima[i];
        let mut rank = 0usize;
        for other in 0..blocks {
            let m = maxima[row * blocks + other];
            if other != local && (m > value || (m == value && other < b)) { rank += 1; }
        }
        let keep = b == local || rank < topk - 1;
        if let Some(dst) = selected.get_mut(idx) { *dst = if keep { 1.0 } else { 0.0 }; }
    }

    #[kernel]
    pub fn tucker_probabilities(scores: &[f32], selected: &[f32],
        mut probabilities: DisjointSlice<f32>, tokens: usize, sparse: bool,
        block: usize, blocks: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= probabilities.len() { return; }
        let row = i / tokens;
        let tk = i % tokens;
        let start = row * tokens;
        let mut maximum = if sparse { f32::NEG_INFINITY } else { scores[start] };
        for t in 0..tokens {
            if !sparse || selected[row * blocks + t / block] != 0.0 {
                let value = if sparse { tucker_sanitize(scores[start + t]) } else { scores[start + t] };
                if maximum < value { maximum = value; }
            }
        }
        let mut sum = 0.0;
        for t in 0..tokens {
            if !sparse || selected[row * blocks + t / block] != 0.0 {
                let value = if sparse { tucker_sanitize(scores[start + t]) } else { scores[start + t] };
                sum += (value - maximum).exp();
            }
        }
        let mut probability = 0.0;
        if sparse && (!sum.is_finite() || sum <= 0.0) {
            let local = (row % tokens) / block;
            let end = ((local + 1) * block).min(tokens);
            if tk / block == local { probability = 1.0 / (end - local * block) as f32; }
        } else if !sparse || selected[row * blocks + tk / block] != 0.0 {
            let value = if sparse { tucker_sanitize(scores[i]) } else { scores[i] };
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            probability = if sparse { (value - maximum).exp() / sum }
                else { (value - maximum).exp() * inv };
        }
        if let Some(dst) = probabilities.get_mut(idx) { *dst = probability; }
    }

    #[kernel]
    pub fn tucker_score_activity(scores: &[f32], mut active: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= active.len() { return; }
        if let Some(dst) = active.get_mut(idx) { *dst = if scores[i].is_finite() { 1.0 } else { 0.0 }; }
    }

    #[kernel]
    pub fn tucker_context(v: &[f32], probabilities: &[f32], mut output: DisjointSlice<f32>,
        tokens: usize, channels: usize, heads: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let dim = channels / heads;
        let c = i % channels;
        let tq = (i / channels) % tokens;
        let w = i / (channels * tokens);
        let row = ((w * heads + c / dim) * tokens + tq) * tokens;
        let mut sum = 0.0;
        for tk in 0..tokens { sum += probabilities[row + tk] * v[(w * tokens + tk) * channels + c]; }
        if let Some(dst) = output.get_mut(idx) { *dst = sum; }
    }

    #[kernel]
    pub fn tucker_probability_grad(v: &[f32], grad: &[f32], mut dp: DisjointSlice<f32>,
        tokens: usize, channels: usize, heads: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dp.len() { return; }
        let tk = i % tokens;
        let tq = (i / tokens) % tokens;
        let h = (i / (tokens * tokens)) % heads;
        let w = i / (tokens * tokens * heads);
        let dim = channels / heads;
        let qi = (w * tokens + tq) * channels + h * dim;
        let ki = (w * tokens + tk) * channels + h * dim;
        let mut sum = 0.0;
        for d in 0..dim { sum += grad[qi + d] * v[ki + d]; }
        if let Some(dst) = dp.get_mut(idx) { *dst = sum; }
    }

    #[kernel]
    pub fn tucker_score_grad(dp: &[f32], probabilities: &[f32], active: &[f32],
        mut ds: DisjointSlice<f32>, tokens: usize, sparse: bool)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= ds.len() { return; }
        let row = (i / tokens) * tokens;
        let mut dot = 0.0;
        for tk in 0..tokens { dot += dp[row + tk] * probabilities[row + tk]; }
        let value = if active[i] == 0.0 || (sparse && probabilities[i] == 0.0) { 0.0 }
            else { probabilities[i] * (dp[i] - dot) };
        if let Some(dst) = ds.get_mut(idx) { *dst = value; }
    }

    /// Gather adjoints: one writer per Q/K/V element, no floating atomics.
    #[kernel]
    pub fn tucker_qkv_grad(q: &[f32], k: &[f32], grad: &[f32], probabilities: &[f32],
        ds: &[f32], mut output: DisjointSlice<f32>, tokens: usize, channels: usize,
        heads: usize, kind: usize)
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let dim = channels / heads;
        let c = i % channels;
        let t = (i / channels) % tokens;
        let w = i / (channels * tokens);
        let base = (w * heads + c / dim) * tokens * tokens;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut sum = 0.0;
        for other in 0..tokens {
            let feature = (w * tokens + other) * channels + c;
            if kind == 0 {
                let value = ds[base + t * tokens + other];
                if value != 0.0 { sum += value * k[feature] * scale; }
            } else if kind == 1 {
                let value = ds[base + other * tokens + t];
                if value != 0.0 { sum += value * q[feature] * scale; }
            } else {
                sum += probabilities[base + other * tokens + t] * grad[feature];
            }
        }
        if let Some(dst) = output.get_mut(idx) { *dst = sum; }
    }
}
