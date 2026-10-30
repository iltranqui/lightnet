//! Host reference for ViT's block-sparse attention and fixed-mask adjoint.

pub fn forward(qkv: &[f32], batch: usize, tokens: usize, channels: usize,
    heads: usize, block: usize, topk: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(batch > 0 && tokens > 0 && channels > 0 && heads > 0 && channels % heads == 0 && block > 0 && topk > 0);
    assert_eq!(qkv.len(), batch * tokens * 3 * channels);
    let dim = channels / heads;
    let mut probabilities = vec![0.0; batch * heads * tokens * tokens];
    let mut active = vec![0.0; probabilities.len()];
    let mut context = vec![0.0; batch * tokens * channels];
    let blocks = tokens.div_ceil(block);
    for b in 0..batch { for h in 0..heads { for query in 0..tokens {
        let row = ((b * heads + h) * tokens + query) * tokens;
        let mut scores = vec![0.0f32; tokens];
        for key in 0..tokens {
            let mut dot = 0.0;
            for d in 0..dim {
                let ch = h * dim + d;
                let q = qkv[(b * tokens + query) * 3 * channels + ch];
                let k = qkv[(b * tokens + key) * 3 * channels + channels + ch];
                let q = if q.is_finite() { q.clamp(-128.0, 128.0) } else { 0.0 };
                let k = if k.is_finite() { k.clamp(-128.0, 128.0) } else { 0.0 };
                dot += q * k;
            }
            let raw = dot / (dim as f32).sqrt();
            active[row + key] = if raw.is_finite() && raw.abs() < 10.0 { 1.0 } else { 0.0 };
            scores[key] = if raw.is_nan() { 0.0 } else { raw.clamp(-10.0, 10.0) };
        }
        let local = query / block;
        let mut selected = vec![false; blocks];
        selected[local] = true;
        for _ in 1..topk.min(blocks) {
            let mut best = None;
            let mut best_value = f32::NEG_INFINITY;
            for candidate in 0..blocks {
                if selected[candidate] { continue; }
                let value = scores[candidate * block..((candidate + 1) * block).min(tokens)]
                    .iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if best.is_none() || value > best_value { best = Some(candidate); best_value = value; }
            }
            if let Some(candidate) = best { selected[candidate] = true; }
        }
        let max = scores.iter().enumerate().filter(|(i, _)| selected[i / block])
            .map(|(_, &value)| value).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for key in 0..tokens { if selected[key / block] {
            probabilities[row + key] = (scores[key] - max).exp();
            sum += probabilities[row + key];
        }}
        for key in 0..tokens { probabilities[row + key] /= sum; }
        for d in 0..dim { let ch = h * dim + d;
            for key in 0..tokens {
                context[(b * tokens + query) * channels + ch] += probabilities[row + key]
                    * qkv[(b * tokens + key) * 3 * channels + 2 * channels + ch];
            }
        }
    }}}
    (context, probabilities, active)
}

pub fn backward(qkv: &[f32], p: &[f32], active: &[f32], context: &[f32], grad: &[f32],
    batch: usize, tokens: usize, channels: usize, heads: usize) -> Vec<f32> {
    let dim = channels / heads;
    let mut out = vec![0.0; qkv.len()];
    for b in 0..batch { for h in 0..heads { for query in 0..tokens {
        let row = ((b * heads + h) * tokens + query) * tokens;
        let mut dot = 0.0f32;
        for d in 0..dim { let ch = h * dim + d;
            dot += grad[(b * tokens + query) * channels + ch] * context[(b * tokens + query) * channels + ch];
        }
        for key in 0..tokens {
            let mut dp = 0.0f32;
            for d in 0..dim { let ch = h * dim + d;
                dp += grad[(b * tokens + query) * channels + ch] * qkv[(b * tokens + key) * 3 * channels + 2 * channels + ch];
                out[(b * tokens + key) * 3 * channels + 2 * channels + ch] += p[row + key] * grad[(b * tokens + query) * channels + ch];
            }
            let ds = p[row + key] * (dp - dot);
            if active[row + key] == 0.0 || ds == 0.0 { continue; }
            for d in 0..dim { let ch = h * dim + d;
                let qi = (b * tokens + query) * 3 * channels + ch;
                let ki = (b * tokens + key) * 3 * channels + channels + ch;
                let q = qkv[qi]; let k = qkv[ki];
                if q.is_finite() && q.abs() < 128.0 {
                    out[qi] += ds * (if k.is_finite() { k.clamp(-128.0, 128.0) } else { 0.0 }) / (dim as f32).sqrt();
                }
                if k.is_finite() && k.abs() < 128.0 {
                    out[ki] += ds * (if q.is_finite() { q.clamp(-128.0, 128.0) } else { 0.0 }) / (dim as f32).sqrt();
                }
            }
        }
    }}}
    out
}
