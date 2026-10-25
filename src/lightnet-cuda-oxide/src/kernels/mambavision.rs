//! MambaVision depthwise/SiLU, selective scan and block-top-k MSA adjoints.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    fn mamba_sigmoid(x: f32) -> f32 {
        if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
        else { let e = x.exp(); e / (1.0 + e) }
    }
    fn mamba_softplus(x: f32) -> f32 {
        if x > 20.0 { x } else if x < -20.0 { x.exp() } else { x.exp().ln_1p() }
    }
    fn mamba_silu_grad(x: f32) -> f32 {
        let s = mamba_sigmoid(x);
        s * (1.0 + x * (1.0 - s))
    }

    #[kernel]
    pub fn mamba_depthwise(
        input: &[f32], weights: &[f32], bias: &[f32], mut pre: DisjointSlice<f32>,
        mut output: DisjointSlice<f32>, tokens: usize, channels: usize,
        branch: usize, kernel: usize, pad: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let t = i % tokens;
        let c = (i / tokens) % channels;
        let b = i / (tokens * channels);
        let mut sum = bias[c];
        for k in 0..kernel {
            if t + k >= pad && t + k - pad < tokens {
                sum += weights[c * kernel + k]
                    * input[(b * tokens + t + k - pad) * 2 * channels + branch * channels + c];
            }
        }
        if let Some(dst) = pre.get_mut(thread::index_1d()) { *dst = sum; }
        if let Some(dst) = output.get_mut(idx) { *dst = sum * mamba_sigmoid(sum); }
    }

    #[kernel]
    pub fn mamba_depthwise_dx(
        weights: &[f32], pre: &[f32], grad: &[f32], mut output: DisjointSlice<f32>,
        tokens: usize, channels: usize, branch: usize, kernel: usize, pad: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let col = i % (2 * channels);
        let row = i / (2 * channels);
        let b = row / tokens;
        let src = row % tokens;
        let mut sum = 0.0;
        if col / channels == branch {
            let c = col % channels;
            for k in 0..kernel {
                if src + pad >= k && src + pad - k < tokens {
                    let j = (b * channels + c) * tokens + src + pad - k;
                    sum += grad[j] * mamba_silu_grad(pre[j]) * weights[c * kernel + k];
                }
            }
        }
        if let Some(dst) = output.get_mut(idx) { *dst = sum; }
    }

    #[kernel]
    pub fn mamba_depthwise_params(
        input: &[f32], pre: &[f32], grad: &[f32], mut dw: DisjointSlice<f32>,
        mut db: DisjointSlice<f32>, batch: usize, tokens: usize, channels: usize,
        branch: usize, kernel: usize, pad: usize,
    ) {
        let c = thread::index_1d().get();
        if c >= channels { return; }
        let wp = dw.as_mut_ptr();
        let mut bias = 0.0;
        // SAFETY: one channel thread owns its kernel row and bias element.
        unsafe {
            for k in 0..kernel { *wp.add(c * kernel + k) = 0.0; }
            for b in 0..batch { for t in 0..tokens {
                let i = (b * channels + c) * tokens + t;
                let delta = grad[i] * mamba_silu_grad(pre[i]);
                bias += delta;
                for k in 0..kernel {
                    if t + k >= pad && t + k - pad < tokens {
                        *wp.add(c * kernel + k) += delta
                            * input[(b * tokens + t + k - pad) * 2 * channels + branch * channels + c];
                    }
                }
            }}
            *db.as_mut_ptr().add(c) = bias;
        }
    }

    #[kernel]
    pub fn mamba_scan(
        x: &[f32], projection: &[f32], dt_weights: &[f32], dt_bias: &[f32],
        a_log: &[f32], d_skip: &[f32], initial: &[f32], mut output: DisjointSlice<f32>,
        mut dt_pre: DisjointSlice<f32>, mut states: DisjointSlice<f32>,
        mut state_out: DisjointSlice<f32>, mut boundary: DisjointSlice<f32>,
        batch: usize, tokens: usize, channels: usize, state: usize, rank: usize,
    ) {
        let lane = thread::index_1d().get();
        if lane >= batch * channels { return; }
        let c = lane % channels;
        let b = lane / channels;
        let width = rank + 2 * state;
        let hp = states.as_mut_ptr();
        let mut seam = 0.0;
        // SAFETY: this lane owns every token/state in one (batch, channel).
        unsafe {
            for t in 0..tokens {
                let i = lane * tokens + t;
                let row = (b * tokens + t) * width;
                let mut pre = dt_bias[c];
                for r in 0..rank { pre += dt_weights[c * rank + r] * projection[row + r]; }
                *dt_pre.as_mut_ptr().add(i) = pre;
                let dt = mamba_softplus(pre);
                let mut y = d_skip[c] * x[i];
                for s in 0..state {
                    let raw = dt * -a_log[c * state + s].exp();
                    if raw == -60.0 || raw == 20.0 { seam = 1.0; }
                    let abar = raw.clamp(-60.0, 20.0).exp();
                    let prev = if t == 0 { initial[lane * state + s] }
                        else { *hp.add((i - 1) * state + s) };
                    let h = abar * prev + dt * projection[row + rank + s] * x[i];
                    *hp.add(i * state + s) = h;
                    y += h * projection[row + rank + state + s];
                    if t + 1 == tokens { *state_out.as_mut_ptr().add(lane * state + s) = h; }
                }
                *output.as_mut_ptr().add(i) = y;
            }
            *boundary.as_mut_ptr().add(lane) = seam;
        }
    }

    /// Each channel thread sums its parameter gradients over batch/time. The
    /// shared B/C projection gets a private channel slab, reduced next launch.
    #[kernel]
    pub fn mamba_scan_backward(
        x: &[f32], projection: &[f32], dt_weights: &[f32], a_log: &[f32],
        d_skip: &[f32], initial: &[f32], dt_pre: &[f32], states: &[f32],
        grad: &[f32], state_grad: &[f32], mut dx: DisjointSlice<f32>,
        mut projection_parts: DisjointSlice<f32>, mut dw: DisjointSlice<f32>,
        mut db: DisjointSlice<f32>, mut da: DisjointSlice<f32>, mut dd: DisjointSlice<f32>,
        mut dh: DisjointSlice<f32>, batch: usize, tokens: usize, channels: usize,
        state: usize, rank: usize,
    ) {
        let c = thread::index_1d().get();
        if c >= channels { return; }
        let width = rank + 2 * state;
        let pp = projection_parts.as_mut_ptr();
        let wp = dw.as_mut_ptr();
        let ap = da.as_mut_ptr();
        let hp = dh.as_mut_ptr();
        let mut bias = 0.0;
        let mut skip = 0.0;
        // SAFETY: channel-owned gradient rows/slabs are disjoint across threads.
        unsafe {
            for r in 0..rank { *wp.add(c * rank + r) = 0.0; }
            for s in 0..state { *ap.add(c * state + s) = 0.0; }
            for b in 0..batch {
                let lane = b * channels + c;
                for s in 0..state { *hp.add(lane * state + s) = state_grad[lane * state + s]; }
                for rev in 0..tokens {
                    let t = tokens - 1 - rev;
                    let i = lane * tokens + t;
                    let row = (b * tokens + t) * width;
                    let part = i * width;
                    for k in 0..width { *pp.add(part + k) = 0.0; }
                    let dt = mamba_softplus(dt_pre[i]);
                    let mut du = grad[i] * d_skip[c];
                    skip += grad[i] * x[i];
                    let mut ddt = 0.0;
                    for s in 0..state {
                        let si = lane * state + s;
                        let prev = if t == 0 { initial[si] } else { states[(i - 1) * state + s] };
                        let a = -a_log[c * state + s].exp();
                        let raw = dt * a;
                        let abar = raw.clamp(-60.0, 20.0).exp();
                        let delta = grad[i] * projection[row + rank + state + s] + *hp.add(si);
                        *hp.add(si) = delta * abar;
                        *pp.add(part + rank + state + s) = grad[i] * states[i * state + s];
                        *pp.add(part + rank + s) = delta * dt * x[i];
                        du += delta * dt * projection[row + rank + s];
                        ddt += delta * projection[row + rank + s] * x[i];
                        if raw > -60.0 && raw < 20.0 {
                            let dz = delta * prev * abar;
                            ddt += dz * a;
                            *ap.add(c * state + s) += dz * dt * a;
                        }
                    }
                    *dx.as_mut_ptr().add(i) = du;
                    let delta = ddt * mamba_sigmoid(dt_pre[i]);
                    bias += delta;
                    for r in 0..rank {
                        *wp.add(c * rank + r) += delta * projection[row + r];
                        *pp.add(part + r) = delta * dt_weights[c * rank + r];
                    }
                }
            }
            *db.as_mut_ptr().add(c) = bias;
            *dd.as_mut_ptr().add(c) = skip;
        }
    }

    #[kernel]
    pub fn mamba_scan_projection_reduce(
        parts: &[f32], mut output: DisjointSlice<f32>, tokens: usize, channels: usize, width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let k = i % width;
        let row = i / width;
        let b = row / tokens;
        let t = row % tokens;
        let mut sum = 0.0;
        for c in 0..channels { sum += parts[((b * channels + c) * tokens + t) * width + k]; }
        if let Some(dst) = output.get_mut(idx) { *dst = sum; }
    }

    #[kernel]
    pub fn mamba_msa(
        input: &[f32], mut output: DisjointSlice<f32>, mut probabilities: DisjointSlice<f32>,
        mut active: DisjointSlice<f32>, mut undefined: DisjointSlice<f32>,
        mut selected: DisjointSlice<f32>, batch: usize, tokens: usize, channels: usize,
        heads: usize, block: usize, topk: usize,
    ) {
        let row = thread::index_1d().get();
        if row >= batch * heads * tokens { return; }
        let q = row % tokens;
        let head = (row / tokens) % heads;
        let b = row / (tokens * heads);
        let dim = channels / heads;
        let blocks = (tokens + block - 1) / block;
        let base = row * tokens;
        let qb = (b * tokens + q) * channels + head * dim;
        let prob = probabilities.as_mut_ptr();
        let sel = selected.as_mut_ptr();
        let mut bad = 0.0;
        // SAFETY: each thread owns its attention row, block mask and context head.
        unsafe {
            for k in 0..tokens {
                let kb = (b * tokens + k) * channels + head * dim;
                let mut dot = 0.0;
                for j in 0..dim { dot += input[qb + j] * input[kb + j]; }
                dot /= (dim as f32).sqrt();
                if !dot.is_finite() || dot == -10.0 || dot == 10.0 { bad = 1.0; }
                *prob.add(base + k) = if dot.is_nan() { 0.0 } else { dot.clamp(-10.0, 10.0) };
                *active.as_mut_ptr().add(base + k) = if dot > -10.0 && dot < 10.0 { 1.0 } else { 0.0 };
            }
            for blk in 0..blocks { *sel.add(row * blocks + blk) = 0.0; }
            *sel.add(row * blocks + q / block) = 1.0;
            for _ in 1..topk {
                let mut best = blocks;
                let mut best_score = f32::NEG_INFINITY;
                for blk in 0..blocks {
                    if *sel.add(row * blocks + blk) != 0.0 { continue; }
                    let mut score = f32::NEG_INFINITY;
                    for k in blk * block..((blk + 1) * block).min(tokens) {
                        score = score.max(*prob.add(base + k));
                    }
                    if best == blocks || score > best_score { best = blk; best_score = score; }
                }
                if best < blocks { *sel.add(row * blocks + best) = 1.0; }
            }
            let mut max = f32::NEG_INFINITY;
            for k in 0..tokens {
                if *sel.add(row * blocks + k / block) != 0.0 { max = max.max(*prob.add(base + k)); }
            }
            let mut sum = 0.0;
            for k in 0..tokens {
                let p = if *sel.add(row * blocks + k / block) != 0.0 {
                    (*prob.add(base + k) - max).exp()
                } else { 0.0 };
                *prob.add(base + k) = p;
                sum += p;
            }
            for k in 0..tokens { *prob.add(base + k) /= sum; }
            for j in 0..dim {
                let mut value = 0.0;
                for k in 0..tokens {
                    value += *prob.add(base + k) * input[(b * tokens + k) * channels + head * dim + j];
                }
                *output.as_mut_ptr().add(qb + j) = value;
            }
            *undefined.as_mut_ptr().add(row) = bad;
        }
    }

    #[kernel]
    pub fn mamba_msa_ds(
        input: &[f32], probabilities: &[f32], active: &[f32], grad: &[f32],
        mut score_grad: DisjointSlice<f32>, batch: usize, tokens: usize, channels: usize, heads: usize,
    ) {
        let row = thread::index_1d().get();
        if row >= batch * heads * tokens { return; }
        let q = row % tokens;
        let head = (row / tokens) % heads;
        let b = row / (tokens * heads);
        let dim = channels / heads;
        let qb = (b * tokens + q) * channels + head * dim;
        let base = row * tokens;
        let ds = score_grad.as_mut_ptr();
        let mut dot = 0.0;
        // SAFETY: this thread owns one score-gradient row in both passes.
        unsafe {
            for k in 0..tokens {
                let kb = (b * tokens + k) * channels + head * dim;
                let mut dp = 0.0;
                for j in 0..dim { dp += grad[qb + j] * input[kb + j]; }
                *ds.add(base + k) = dp;
                dot += dp * probabilities[base + k];
            }
            for k in 0..tokens {
                *ds.add(base + k) = if active[base + k] != 0.0 {
                    probabilities[base + k] * (*ds.add(base + k) - dot) / (dim as f32).sqrt()
                } else { 0.0 };
            }
        }
    }

    #[kernel]
    pub fn mamba_msa_dx(
        input: &[f32], probabilities: &[f32], ds: &[f32], grad: &[f32],
        mut output: DisjointSlice<f32>, tokens: usize, channels: usize, heads: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() { return; }
        let c = i % channels;
        let token = (i / channels) % tokens;
        let b = i / (tokens * channels);
        let head = c / (channels / heads);
        let row = ((b * heads + head) * tokens + token) * tokens;
        let mut sum = 0.0;
        for k in 0..tokens {
            let j = (b * tokens + k) * channels + c;
            let other = ((b * heads + head) * tokens + k) * tokens + token;
            sum += ds[row + k] * input[j];
            sum += ds[other] * input[j];
            sum += probabilities[other] * grad[j];
        }
        if let Some(dst) = output.get_mut(idx) { *dst = sum; }
    }
}
