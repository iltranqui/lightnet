//! Window-local Tucker attention, without Swin's score clamp or padding mask.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub windows: usize,
    pub tokens: usize,
    pub channels: usize,
    pub heads: usize,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
}

impl AttentionGeometry {
    pub fn lengths(self) -> Result<(usize, usize), String> {
        if self.windows == 0 || self.tokens == 0 || self.channels == 0 || self.heads == 0
            || self.channels % self.heads != 0
        {
            return Err("Tucker attention requires positive dimensions and channels divisible by heads".into());
        }
        let features = self.windows.checked_mul(self.tokens)
            .and_then(|n| n.checked_mul(self.channels)).ok_or("Tucker feature size overflow")?;
        let scores = self.windows.checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.tokens)).and_then(|n| n.checked_mul(self.tokens))
            .ok_or("Tucker score size overflow")?;
        Ok((features, scores))
    }

    pub fn sparse(self) -> (bool, usize, usize) {
        let block = self.msa_block_size.max(1).min(self.tokens);
        let blocks = self.tokens.div_ceil(block);
        let topk = self.msa_topk.max(1).min(blocks);
        (self.msa_enabled && self.tokens > 1 && topk < blocks, block, topk)
    }
}

fn sanitized(value: f32) -> f32 {
    if value.is_nan() { 0.0 }
    else if value == f32::INFINITY { 80.0 }
    else if value == f32::NEG_INFINITY { -80.0 }
    else { value }
}

pub fn forward(q: &[f32], k: &[f32], v: &[f32], g: AttentionGeometry)
    -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
    let (features, scores) = g.lengths().expect("Tucker attention geometry");
    assert_eq!((q.len(), k.len(), v.len()), (features, features, features));
    let (sparse, block, topk) = g.sparse();
    let blocks = g.tokens.div_ceil(block);
    let dim = g.channels / g.heads;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut p = vec![0.0; scores];
    let mut active = vec![0.0; scores];
    let mut context = vec![0.0; features];
    for w in 0..g.windows {
        for h in 0..g.heads {
            for tq in 0..g.tokens {
                let row = ((w * g.heads + h) * g.tokens + tq) * g.tokens;
                let mut values = vec![0.0; g.tokens];
                for tk in 0..g.tokens {
                    let mut dot = 0.0;
                    for d in 0..dim {
                        dot += q[(w * g.tokens + tq) * g.channels + h * dim + d]
                            * k[(w * g.tokens + tk) * g.channels + h * dim + d];
                    }
                    let score = dot * scale;
                    active[row + tk] = if score.is_finite() { 1.0 } else { 0.0 };
                    values[tk] = if sparse { sanitized(score) } else { score };
                }
                let local = tq / block;
                let mut selected = vec![!sparse; blocks];
                if sparse {
                    let mut maxima = vec![f32::NEG_INFINITY; blocks];
                    for (i, &value) in values.iter().enumerate() {
                        maxima[i / block] = maxima[i / block].max(value);
                    }
                    selected[local] = true;
                    for _ in 1..topk {
                        let mut best = None;
                        for b in 0..blocks {
                            if !selected[b] && best.is_none_or(|old| maxima[b] > maxima[old]) {
                                best = Some(b);
                            }
                        }
                        if let Some(b) = best { selected[b] = true; }
                    }
                }
                let mut maximum = if sparse { f32::NEG_INFINITY } else { values[0] };
                for (i, &value) in values.iter().enumerate() {
                    if selected[i / block] && maximum < value { maximum = value; }
                }
                let mut sum = 0.0;
                for i in 0..g.tokens {
                    if selected[i / block] {
                        p[row + i] = (values[i] - maximum).exp();
                        sum += p[row + i];
                    }
                }
                if sparse && (!sum.is_finite() || sum <= 0.0) {
                    let start = local * block;
                    let end = (start + block).min(g.tokens);
                    for i in 0..g.tokens {
                        p[row + i] = if i >= start && i < end { 1.0 / (end - start) as f32 } else { 0.0 };
                    }
                } else {
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for i in 0..g.tokens {
                        if sparse { p[row + i] /= sum; } else { p[row + i] *= inv; }
                    }
                }
                for tk in 0..g.tokens {
                    for d in 0..dim {
                        context[(w * g.tokens + tq) * g.channels + h * dim + d] +=
                            p[row + tk] * v[(w * g.tokens + tk) * g.channels + h * dim + d];
                    }
                }
            }
        }
    }
    (context, p, active)
}

pub fn backward(q: &[f32], k: &[f32], v: &[f32], p: &[f32], active: &[f32],
    grad: &[f32], g: AttentionGeometry) -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
    let (features, scores) = g.lengths().expect("Tucker attention geometry");
    assert_eq!((q.len(), k.len(), v.len(), grad.len()), (features, features, features, features));
    assert_eq!((p.len(), active.len()), (scores, scores));
    let dim = g.channels / g.heads;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut dq = vec![0.0; features];
    let mut dk = vec![0.0; features];
    let mut dv = vec![0.0; features];
    for w in 0..g.windows {
        for h in 0..g.heads {
            for tq in 0..g.tokens {
                let row = ((w * g.heads + h) * g.tokens + tq) * g.tokens;
                let qi = (w * g.tokens + tq) * g.channels + h * dim;
                let mut dp = vec![0.0; g.tokens];
                let mut dot = 0.0;
                for tk in 0..g.tokens {
                    let ki = (w * g.tokens + tk) * g.channels + h * dim;
                    for d in 0..dim {
                        dp[tk] += grad[qi + d] * v[ki + d];
                        dv[ki + d] += p[row + tk] * grad[qi + d];
                    }
                    dot += dp[tk] * p[row + tk];
                }
                for tk in 0..g.tokens {
                    let ds = p[row + tk] * (dp[tk] - dot);
                    if active[row + tk] == 0.0 || ds == 0.0
                        || (g.sparse().0 && p[row + tk] == 0.0) { continue; }
                    let ki = (w * g.tokens + tk) * g.channels + h * dim;
                    for d in 0..dim {
                        dq[qi + d] += ds * k[ki + d] * scale;
                        dk[ki + d] += ds * q[qi + d] * scale;
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}
