//! Causal low-rank delta-rule. One thread owns each independent batch scan;
//! GEMM projections and expert MLPs remain parallel backend operations.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
        else { let e = x.exp(); e / (1.0 + e) }
    }

    #[kernel]
    pub fn scan(
        q: &[f32], raw_k: &[f32], v: &[f32], beta: &[f32], decay: &[f32],
        mut keys: DisjointSlice<f32>, mut inverse: DisjointSlice<f32>,
        mut norm_squared: DisjointSlice<f32>, mut states: DisjointSlice<f32>,
        mut output: DisjointSlice<f32>, batch: usize, tokens: usize, channels: usize, rank: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batch { return; }
        // SAFETY: each thread owns all rows and states in its batch slab.
        unsafe {
            let kp = keys.as_mut_ptr();
            let hp = states.as_mut_ptr();
            for token in 0..tokens {
                let row = b * tokens + token;
                let mut norm = 0.0;
                for j in 0..rank { let k = raw_k[row * rank + j]; norm += k * k; }
                let inv = if norm.is_finite() { 1.0 / norm.max(1e-12).sqrt() } else { 0.0 };
                *inverse.as_mut_ptr().add(row) = inv;
                *norm_squared.as_mut_ptr().add(row) = norm;
                for j in 0..rank { *kp.add(row * rank + j) = if inv == 0.0 { 0.0 } else { raw_k[row * rank + j] * inv }; }
                for c in 0..channels {
                    let mut prediction = 0.0;
                    if token > 0 { for j in 0..rank { prediction += *kp.add(row * rank + j) * *hp.add(((row - 1) * rank + j) * channels + c); } }
                    let error = v[row * channels + c] - prediction;
                    let mut value = 0.0;
                    for j in 0..rank {
                        let previous = if token == 0 { 0.0 } else { *hp.add(((row - 1) * rank + j) * channels + c) };
                        let h = sigmoid(decay[row * rank + j]) * previous
                            + sigmoid(beta[row * rank + j]) * *kp.add(row * rank + j) * error;
                        *hp.add((row * rank + j) * channels + c) = h;
                        value += q[row * rank + j] * h;
                    }
                    *output.as_mut_ptr().add(row * channels + c) = value / (rank as f32).sqrt();
                }
            }
        }
    }

    #[kernel]
    pub fn backward(
        q: &[f32], keys: &[f32], v: &[f32], beta: &[f32], decay: &[f32],
        inverse: &[f32], norm_squared: &[f32], states: &[f32], dy: &[f32],
        mut dq: DisjointSlice<f32>, mut dk: DisjointSlice<f32>, mut dv: DisjointSlice<f32>,
        mut db: DisjointSlice<f32>, mut dd: DisjointSlice<f32>, mut carry: DisjointSlice<f32>,
        batch: usize, tokens: usize, channels: usize, rank: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batch { return; }
        // SAFETY: each batch owns its carry and all output-gradient rows.
        unsafe {
            let qp = dq.as_mut_ptr(); let kp = dk.as_mut_ptr();
            let bp = db.as_mut_ptr(); let dp = dd.as_mut_ptr(); let cp = carry.as_mut_ptr();
            let carry_base = b * rank * channels;
            for i in 0..rank * channels { *cp.add(carry_base + i) = 0.0; }
            for reverse in 0..tokens {
                let token = tokens - 1 - reverse;
                let row = b * tokens + token;
                for j in 0..rank {
                    let i = row * rank + j;
                    *qp.add(i) = 0.0; *kp.add(i) = 0.0; *bp.add(i) = 0.0; *dp.add(i) = 0.0;
                }
                for c in 0..channels {
                    let mut prediction = 0.0;
                    if token > 0 { for j in 0..rank { prediction += keys[row * rank + j] * states[((row - 1) * rank + j) * channels + c]; } }
                    let error = v[row * channels + c] - prediction;
                    let mut ds = [0.0_f32; 64];
                    let mut de = 0.0;
                    for j in 0..rank {
                        let i = row * rank + j;
                        let previous = if token == 0 { 0.0 } else { states[((row - 1) * rank + j) * channels + c] };
                        *qp.add(i) += dy[row * channels + c] * states[(row * rank + j) * channels + c] / (rank as f32).sqrt();
                        ds[j] = dy[row * channels + c] * q[i] / (rank as f32).sqrt() + *cp.add(carry_base + j * channels + c);
                        *dp.add(i) += ds[j] * previous;
                        *bp.add(i) += ds[j] * keys[i] * error;
                        *kp.add(i) += ds[j] * sigmoid(beta[i]) * error;
                        de += ds[j] * sigmoid(beta[i]) * keys[i];
                    }
                    for j in 0..rank {
                        let previous = if token == 0 { 0.0 } else { states[((row - 1) * rank + j) * channels + c] };
                        *cp.add(carry_base + j * channels + c) = if token == 0 { 0.0 }
                            else { ds[j] * sigmoid(decay[row * rank + j]) - de * keys[row * rank + j] };
                        *kp.add(row * rank + j) -= de * previous;
                    }
                    *dv.as_mut_ptr().add(row * channels + c) = de;
                }
                let norm = norm_squared[row];
                let mut dot = 0.0;
                for j in 0..rank { dot += keys[row * rank + j] * *kp.add(row * rank + j); }
                for j in 0..rank {
                    let i = row * rank + j;
                    *kp.add(i) = if !norm.is_finite() { 0.0 }
                        else if norm < 1e-12 { *kp.add(i) * inverse[row] }
                        else { inverse[row] * (*kp.add(i) - keys[i] * dot) };
                    let beta = sigmoid(beta[i]); let decay = sigmoid(decay[i]);
                    *bp.add(i) *= beta * (1.0 - beta);
                    *dp.add(i) *= decay * (1.0 - decay);
                }
            }
        }
    }

    #[kernel]
    pub fn router(
        logits: &[f32], mut output: DisjointSlice<f32>, rows: usize,
        experts: usize, top_k: usize, temperature: f32,
    ) {
        let row = thread::index_1d().get();
        if row >= rows { return; }
        let base = row * experts;
        let mut maximum = f32::NEG_INFINITY;
        for e in 0..experts { maximum = maximum.max(logits[base + e] / temperature); }
        let mut gates = [0.0_f32; 64];
        let mut sum = 0.0;
        for e in 0..experts {
            let logit = logits[base + e] / temperature;
            let mut rank = 0;
            for j in 0..experts {
                let other = logits[base + j] / temperature;
                if other > logit || (other == logit && j < e) { rank += 1; }
            }
            if top_k == 0 || rank < top_k { gates[e] = (logit - maximum).exp(); sum += gates[e]; }
        }
        if !sum.is_finite() || sum <= 0.0 { sum = 1.0; }
        // SAFETY: row-owned, fully written expert slab.
        unsafe { for e in 0..experts { *output.as_mut_ptr().add(base + e) = gates[e] / sum; } }
    }
}
