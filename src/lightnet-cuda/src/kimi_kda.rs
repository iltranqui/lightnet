//! Low-rank causal delta-rule contracts; all projections use token-major rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaGeometry { pub batch: usize, pub tokens: usize, pub channels: usize, pub rank: usize }

impl KdaGeometry {
    pub fn validate(self) -> Result<(), String> {
        if [self.batch, self.tokens, self.channels, self.rank].contains(&0)
            || self.rank > 64 || self.rank > self.channels {
            return Err("Kimi KDA invalid scan dimensions".into());
        }
        [self.batch, self.tokens, self.rank, self.channels].iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or("Kimi KDA scan workspace overflow")?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct KdaForward<T> { pub output: T, pub keys: T, pub inverse: T, pub norm_squared: T, pub states: T }
#[derive(Clone, Debug)]
pub struct KdaGrad<T> { pub q: T, pub k: T, pub v: T, pub beta: T, pub decay: T }

pub fn scan(q: &[f32], raw_k: &[f32], v: &[f32], beta: &[f32], decay: &[f32], g: KdaGeometry)
    -> KdaForward<Vec<f32>>
{
    g.validate().expect("Kimi KDA geometry");
    let (b, t, c, r) = (g.batch, g.tokens, g.channels, g.rank);
    let rows = b * t;
    assert_eq!(q.len(), rows * r);
    assert_eq!(raw_k.len(), rows * r);
    assert_eq!(v.len(), rows * c);
    assert_eq!(beta.len(), rows * r);
    assert_eq!(decay.len(), rows * r);
    let mut out = KdaForward { output: vec![0.0; rows * c], keys: raw_k.to_vec(),
        inverse: vec![0.0; rows], norm_squared: vec![0.0; rows], states: vec![0.0; rows * r * c] };
    for batch in 0..b { for token in 0..t {
        let row = batch * t + token;
        let norm = raw_k[row * r..(row + 1) * r].iter().map(|x| x * x).sum::<f32>();
        out.norm_squared[row] = norm;
        let inv = if norm.is_finite() { norm.max(1e-12).sqrt().recip() } else { 0.0 };
        out.inverse[row] = inv;
        for j in 0..r { out.keys[row * r + j] = if inv == 0.0 { 0.0 } else { raw_k[row * r + j] * inv }; }
        for ch in 0..c {
            let mut prediction = 0.0;
            if token > 0 { for j in 0..r { prediction += out.keys[row * r + j] * out.states[((row - 1) * r + j) * c + ch]; } }
            let error = v[row * c + ch] - prediction;
            let mut value = 0.0;
            for j in 0..r {
                let previous = if token == 0 { 0.0 } else { out.states[((row - 1) * r + j) * c + ch] };
                let state = crate::mambavision::sigmoid(decay[row * r + j]) * previous
                    + crate::mambavision::sigmoid(beta[row * r + j]) * out.keys[row * r + j] * error;
                out.states[(row * r + j) * c + ch] = state;
                value += q[row * r + j] * state;
            }
            out.output[row * c + ch] = value / (r as f32).sqrt();
        }
    }}
    out
}

pub fn backward(q: &[f32], v: &[f32], beta: &[f32], decay: &[f32],
    forward: &KdaForward<Vec<f32>>, dy: &[f32], g: KdaGeometry) -> KdaGrad<Vec<f32>>
{
    g.validate().expect("Kimi KDA geometry");
    let (b, t, c, r) = (g.batch, g.tokens, g.channels, g.rank);
    let rows = b * t;
    assert_eq!(q.len(), rows * r);
    assert_eq!(v.len(), rows * c);
    assert_eq!(beta.len(), rows * r);
    assert_eq!(decay.len(), rows * r);
    assert_eq!(forward.keys.len(), rows * r);
    assert_eq!(forward.inverse.len(), rows);
    assert_eq!(forward.norm_squared.len(), rows);
    assert_eq!(forward.states.len(), rows * r * c);
    assert_eq!(dy.len(), rows * c);
    let mut out = KdaGrad { q: vec![0.0; rows * r], k: vec![0.0; rows * r], v: vec![0.0; rows * c],
        beta: vec![0.0; rows * r], decay: vec![0.0; rows * r] };
    for batch in 0..b {
        let mut carry = vec![0.0; r * c];
        for token in (0..t).rev() {
            let row = batch * t + token;
            for ch in 0..c {
                let mut prediction = 0.0;
                if token > 0 { for j in 0..r { prediction += forward.keys[row * r + j] * forward.states[((row - 1) * r + j) * c + ch]; } }
                let error = v[row * c + ch] - prediction;
                let mut ds = [0.0; 64];
                let mut de = 0.0;
                for j in 0..r {
                    let k = forward.keys[row * r + j];
                    let beta = crate::mambavision::sigmoid(beta[row * r + j]);
                    let previous = if token == 0 { 0.0 } else { forward.states[((row - 1) * r + j) * c + ch] };
                    out.q[row * r + j] += dy[row * c + ch] * forward.states[(row * r + j) * c + ch] / (r as f32).sqrt();
                    ds[j] = dy[row * c + ch] * q[row * r + j] / (r as f32).sqrt() + carry[j * c + ch];
                    out.decay[row * r + j] += ds[j] * previous;
                    out.beta[row * r + j] += ds[j] * k * error;
                    out.k[row * r + j] += ds[j] * beta * error;
                    de += ds[j] * beta * k;
                }
                for j in 0..r {
                    let previous = if token == 0 { 0.0 } else { forward.states[((row - 1) * r + j) * c + ch] };
                    carry[j * c + ch] = if token == 0 { 0.0 } else {
                        ds[j] * crate::mambavision::sigmoid(decay[row * r + j]) - de * forward.keys[row * r + j] };
                    out.k[row * r + j] -= de * previous;
                }
                out.v[row * c + ch] = de;
            }
            let norm = forward.norm_squared[row];
            assert_ne!(norm, 1e-12, "Kimi KDA undefined key-normalization derivative");
            let dot = (0..r).map(|j| forward.keys[row * r + j] * out.k[row * r + j]).sum::<f32>();
            for j in 0..r {
                let i = row * r + j;
                out.k[i] = if !norm.is_finite() { 0.0 }
                    else if norm < 1e-12 { out.k[i] * forward.inverse[row] }
                    else { forward.inverse[row] * (out.k[i] - forward.keys[i] * dot) };
                let beta = crate::mambavision::sigmoid(beta[i]);
                let decay = crate::mambavision::sigmoid(decay[i]);
                out.beta[i] *= beta * (1.0 - beta);
                out.decay[i] *= decay * (1.0 - decay);
            }
        }
    }
    out
}

/// Stable index tie-breaking matches the CPU Kimi router, unlike threshold
/// masking (which may select more than top_k when logits tie).
pub fn gates(logits: &[f32], rows: usize, experts: usize, top_k: usize, temperature: f32) -> Vec<f32> {
    assert!(rows > 0 && experts > 0 && top_k <= experts && temperature > 0.0 && temperature.is_finite());
    assert_eq!(logits.len(), rows * experts);
    let mut out = vec![0.0; logits.len()];
    for row in 0..rows {
        let base = row * experts;
        let maximum = logits[base..base + experts].iter().map(|x| x / temperature).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for e in 0..experts {
            let logit = logits[base + e] / temperature;
            let rank = (0..experts).filter(|&j| logits[base + j] / temperature > logit
                || (logits[base + j] / temperature == logit && j < e)).count();
            if top_k == 0 || rank < top_k { out[base + e] = (logit - maximum).exp(); sum += out[base + e]; }
        }
        if !sum.is_finite() || sum <= 0.0 { sum = 1.0; }
        for e in 0..experts { out[base + e] /= sum; }
    }
    out
}
