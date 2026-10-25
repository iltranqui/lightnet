//! MambaVision primitive contracts and host references. Scan inputs preserve
//! the CPU layer's BDT storage; the selective projection is row-major.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MambaScanGeometry {
    pub batch: usize,
    pub tokens: usize,
    pub channels: usize,
    pub state: usize,
    pub rank: usize,
}

impl MambaScanGeometry {
    pub fn width(self) -> usize { self.rank + 2 * self.state }
    pub fn validate(self) -> Result<(), String> {
        if [self.batch, self.tokens, self.channels, self.state, self.rank].contains(&0) {
            return Err("MambaVision scan dimensions must be positive".into());
        }
        let width = self.state.checked_mul(2).and_then(|s| s.checked_add(self.rank))
            .ok_or("MambaVision scan width overflow")?;
        for dims in [
            vec![self.batch, self.channels, self.tokens, width],
            vec![self.batch, self.channels, self.tokens, self.state],
            vec![self.channels, self.rank],
        ] {
            dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or("MambaVision scan workspace overflow")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MambaScanForward<T> {
    pub output: T,
    pub dt_pre: T,
    pub states: T,
    pub state_out: T,
    /// One flag per scan lane; nonzero means a clamp-boundary derivative.
    pub boundary: T,
}

#[derive(Clone, Debug)]
pub struct MambaScanGrad<T> {
    pub input: T,
    pub projection: T,
    pub dt_weights: T,
    pub dt_bias: T,
    pub a_log: T,
    pub d_skip: T,
    pub state_in: T,
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
    else { let e = x.exp(); e / (1.0 + e) }
}

pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { x.exp() } else { x.exp().ln_1p() }
}

pub fn depthwise(
    input: &[f32], weights: &[f32], bias: &[f32], batch: usize,
    tokens: usize, channels: usize, branch: usize, kernel: usize, causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert!(batch > 0 && tokens > 0 && channels > 0 && branch < 2 && kernel > 0 && kernel % 2 == 1);
    assert_eq!(input.len(), batch * tokens * 2 * channels);
    assert_eq!(weights.len(), channels * kernel);
    assert_eq!(bias.len(), channels);
    let pad = if causal { kernel - 1 } else { kernel / 2 };
    let mut pre = vec![0.0; batch * channels * tokens];
    let mut output = pre.clone();
    for b in 0..batch { for c in 0..channels { for t in 0..tokens {
        let i = (b * channels + c) * tokens + t;
        let mut sum = bias[c];
        for k in 0..kernel {
            if t + k >= pad && t + k - pad < tokens {
                sum += weights[c * kernel + k]
                    * input[(b * tokens + t + k - pad) * 2 * channels + branch * channels + c];
            }
        }
        pre[i] = sum;
        output[i] = sum * sigmoid(sum);
    }}}
    (pre, output)
}

pub fn depthwise_backward(
    input: &[f32], weights: &[f32], pre: &[f32], grad: &[f32], batch: usize,
    tokens: usize, channels: usize, branch: usize, kernel: usize, causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(batch > 0 && tokens > 0 && channels > 0 && branch < 2 && kernel > 0 && kernel % 2 == 1);
    assert_eq!(input.len(), batch * tokens * 2 * channels);
    assert_eq!(weights.len(), channels * kernel);
    assert_eq!(pre.len(), batch * channels * tokens);
    assert_eq!(grad.len(), pre.len());
    let pad = if causal { kernel - 1 } else { kernel / 2 };
    let mut dx = vec![0.0; input.len()];
    let mut dw = vec![0.0; weights.len()];
    let mut db = vec![0.0; channels];
    for b in 0..batch { for c in 0..channels { for t in 0..tokens {
        let i = (b * channels + c) * tokens + t;
        let sig = sigmoid(pre[i]);
        let delta = grad[i] * sig * (1.0 + pre[i] * (1.0 - sig));
        db[c] += delta;
        for k in 0..kernel {
            if t + k >= pad && t + k - pad < tokens {
                let j = (b * tokens + t + k - pad) * 2 * channels + branch * channels + c;
                dw[c * kernel + k] += delta * input[j];
                dx[j] += delta * weights[c * kernel + k];
            }
        }
    }}}
    (dx, dw, db)
}

pub fn scan(
    x: &[f32], projection: &[f32], dt_weights: &[f32], dt_bias: &[f32],
    a_log: &[f32], d_skip: &[f32], state_in: &[f32], g: MambaScanGeometry,
) -> MambaScanForward<Vec<f32>> {
    g.validate().expect("MambaVision scan geometry");
    let (b, t, d, s, r, w) = (g.batch, g.tokens, g.channels, g.state, g.rank, g.width());
    assert_eq!(x.len(), b * d * t);
    assert_eq!(projection.len(), b * t * w);
    assert_eq!(dt_weights.len(), d * r);
    assert_eq!(dt_bias.len(), d);
    assert_eq!(a_log.len(), d * s);
    assert_eq!(d_skip.len(), d);
    assert_eq!(state_in.len(), b * d * s);
    let mut out = MambaScanForward {
        output: vec![0.0; x.len()], dt_pre: vec![0.0; x.len()],
        states: vec![0.0; b * d * t * s], state_out: state_in.to_vec(),
        boundary: vec![0.0; b * d],
    };
    for batch in 0..b { for c in 0..d { for token in 0..t {
        let i = (batch * d + c) * t + token;
        let row = (batch * t + token) * w;
        let mut pre = dt_bias[c];
        for rank in 0..r { pre += dt_weights[c * r + rank] * projection[row + rank]; }
        out.dt_pre[i] = pre;
        let dt = softplus(pre);
        let mut y = d_skip[c] * x[i];
        for state in 0..s {
            let raw = dt * -a_log[c * s + state].exp();
            if raw == -60.0 || raw == 20.0 { out.boundary[batch * d + c] = 1.0; }
            let a = raw.clamp(-60.0, 20.0).exp();
            let si = (batch * d + c) * s + state;
            let h = a * out.state_out[si] + dt * projection[row + r + state] * x[i];
            out.states[i * s + state] = h;
            out.state_out[si] = h;
            y += h * projection[row + r + s + state];
        }
        out.output[i] = y;
    }}}
    out
}

pub fn scan_backward(
    x: &[f32], projection: &[f32], dt_weights: &[f32], a_log: &[f32],
    d_skip: &[f32], state_in: &[f32], dt_pre: &[f32], states: &[f32],
    grad: &[f32], state_grad: &[f32], g: MambaScanGeometry,
) -> MambaScanGrad<Vec<f32>> {
    g.validate().expect("MambaVision scan geometry");
    let (b, t, d, s, r, w) = (g.batch, g.tokens, g.channels, g.state, g.rank, g.width());
    assert_eq!(x.len(), b * d * t);
    assert_eq!(projection.len(), b * t * w);
    assert_eq!(dt_weights.len(), d * r);
    assert_eq!(a_log.len(), d * s);
    assert_eq!(d_skip.len(), d);
    assert_eq!(state_in.len(), b * d * s);
    assert_eq!(dt_pre.len(), x.len());
    assert_eq!(states.len(), x.len() * s);
    assert_eq!(grad.len(), x.len());
    assert_eq!(state_grad.len(), state_in.len());
    let mut out = MambaScanGrad {
        input: vec![0.0; x.len()], projection: vec![0.0; projection.len()],
        dt_weights: vec![0.0; d * r], dt_bias: vec![0.0; d],
        a_log: vec![0.0; d * s], d_skip: vec![0.0; d], state_in: state_grad.to_vec(),
    };
    for batch in 0..b { for c in 0..d { for token in (0..t).rev() {
        let i = (batch * d + c) * t + token;
        let row = (batch * t + token) * w;
        let dt = softplus(dt_pre[i]);
        let mut ddt = 0.0;
        out.input[i] += grad[i] * d_skip[c];
        out.d_skip[c] += grad[i] * x[i];
        for state in 0..s {
            let si = (batch * d + c) * s + state;
            let prev = if token == 0 { state_in[si] } else { states[(i - 1) * s + state] };
            let a = -a_log[c * s + state].exp();
            let raw = dt * a;
            let abar = raw.clamp(-60.0, 20.0).exp();
            let dh = grad[i] * projection[row + r + s + state] + out.state_in[si];
            out.state_in[si] = dh * abar;
            out.projection[row + r + s + state] += grad[i] * states[i * s + state];
            out.projection[row + r + state] += dh * dt * x[i];
            out.input[i] += dh * dt * projection[row + r + state];
            ddt += dh * projection[row + r + state] * x[i];
            if raw > -60.0 && raw < 20.0 {
                let dz = dh * prev * abar;
                ddt += dz * a;
                out.a_log[c * s + state] += dz * dt * a;
            }
        }
        let delta = ddt * sigmoid(dt_pre[i]);
        out.dt_bias[c] += delta;
        for rank in 0..r {
            out.dt_weights[c * r + rank] += delta * projection[row + rank];
            out.projection[row + rank] += delta * dt_weights[c * r + rank];
        }
    }}}
    out
}
pub fn msa_forward(
    input: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    block_size: usize,
    topk: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<bool>, bool), String> {
    if heads == 0 || channels % heads != 0 || block_size == 0 || topk == 0 {
        return Err("MambaVision MSA requires positive block/head geometry".into());
    }
    let dim = channels / heads;
    let score_count = checked_product(&[batch, heads, tokens, tokens], "MSA scores")?;
    let mut output = vec![0.0; batch * tokens * channels];
    let mut probabilities = vec![0.0; score_count];
    let mut score_active = vec![false; score_count];
    let mut undefined = false;
    let blocks = tokens.div_ceil(block_size.min(tokens));
    let block_size = block_size.min(tokens);
    let topk = topk.min(blocks);
    for b in 0..batch {
        for head in 0..heads {
            for query in 0..tokens {
                let score_base = ((b * heads + head) * tokens + query) * tokens;
                let mut scores = vec![0.0; tokens];
                for key in 0..tokens {
                    let qbase = (b * tokens + query) * channels + head * dim;
                    let kbase = (b * tokens + key) * channels + head * dim;
                    let dot = (0..dim)
                        .map(|i| input[qbase + i] * input[kbase + i])
                        .sum::<f32>()
                        / (dim as f32).sqrt();
                    if dot.is_nan()
                        || dot.is_infinite()
                        || dot == -MSA_SCORE_CLIP
                        || dot == MSA_SCORE_CLIP
                    {
                        undefined = true;
                    }
                    scores[key] = if dot.is_nan() {
                        0.0
                    } else if dot.is_infinite() {
                        if dot.is_sign_positive() {
                            MSA_SCORE_CLIP
                        } else {
                            -MSA_SCORE_CLIP
                        }
                    } else {
                        dot.clamp(-MSA_SCORE_CLIP, MSA_SCORE_CLIP)
                    };
                    score_active[score_base + key] = dot > -MSA_SCORE_CLIP && dot < MSA_SCORE_CLIP;
                }
                let mut selected = vec![false; blocks];
                let local = (query / block_size).min(blocks - 1);
                selected[local] = true;
                let mut selected_count = 1;
                while selected_count < topk {
                    let mut best: Option<usize> = None;
                    for block in 0..blocks {
                        if selected[block] {
                            continue;
                        }
                        let start = block * block_size;
                        let end = (start + block_size).min(tokens);
                        let block_max = scores[start..end]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        if best.is_none_or(|current| {
                            let best_start = current * block_size;
                            let best_end = (best_start + block_size).min(tokens);
                            block_max
                                > scores[best_start..best_end]
                                    .iter()
                                    .copied()
                                    .fold(f32::NEG_INFINITY, f32::max)
                        }) {
                            best = Some(block);
                        }
                    }
                    if let Some(block) = best {
                        selected[block] = true;
                        selected_count += 1;
                    } else {
                        break;
                    }
                }
                let selected_max = scores
                    .iter()
                    .enumerate()
                    .filter(|(key, _)| selected[*key / block_size])
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for key in 0..tokens {
                    if selected[key / block_size] {
                        let value = (scores[key] - selected_max).exp();
                        probabilities[score_base + key] = value;
                        sum += value;
                    }
                }
                if !sum.is_finite() || sum <= 0.0 {
                    let start = local * block_size;
                    let end = (start + block_size).min(tokens);
                    probabilities[score_base + start..score_base + end]
                        .fill(1.0 / (end - start) as f32);
                } else {
                    for key in 0..tokens {
                        probabilities[score_base + key] /= sum;
                    }
                }
                let outbase = (b * tokens + query) * channels + head * dim;
                for key in 0..tokens {
                    let probability = probabilities[score_base + key];
                    let vbase = (b * tokens + key) * channels + head * dim;
                    for i in 0..dim {
                        output[outbase + i] += probability * input[vbase + i];
                    }
                }
            }
        }
    }
    Ok((output, probabilities, score_active, undefined))
}

pub fn msa_backward(
    input: &[f32],
    probabilities: &[f32],
    score_active: &[bool],
    doutput: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
) -> Result<Vec<f32>, String> {
    let dim = channels / heads;
    let mut dinput = vec![0.0; batch * tokens * channels];
    let mut d_attn = vec![0.0; tokens];
    for b in 0..batch {
        for head in 0..heads {
            for query in 0..tokens {
                let score_base = ((b * heads + head) * tokens + query) * tokens;
                let qbase = (b * tokens + query) * channels + head * dim;
                let dybase = qbase;
                let mut softmax_dot = 0.0;
                for key in 0..tokens {
                    let probability = probabilities[score_base + key];
                    let vbase = (b * tokens + key) * channels + head * dim;
                    let mut da = 0.0;
                    for i in 0..dim {
                        da += doutput[dybase + i] * input[vbase + i];
                        dinput[vbase + i] += probability * doutput[dybase + i];
                    }
                    d_attn[key] = da;
                    softmax_dot += da * probability;
                }
                for key in 0..tokens {
                    let ds = probabilities[score_base + key] * (d_attn[key] - softmax_dot);
                    if ds == 0.0 || !score_active[score_base + key] {
                        continue;
                    }
                    let kbase = (b * tokens + key) * channels + head * dim;
                    let scale = ds / (dim as f32).sqrt();
                    for i in 0..dim {
                        dinput[qbase + i] += scale * input[kbase + i];
                        dinput[kbase + i] += scale * input[qbase + i];
                    }
                }
            }
        }
    }
    if dinput.iter().any(|value| !value.is_finite()) {
        return Err("MambaVision MSA produced a nonfinite gradient".into());
    }
    Ok(dinput)
}

const MSA_SCORE_CLIP: f32 = 10.0;
fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or_else(|| format!("MambaVision {what} overflow"))
}
