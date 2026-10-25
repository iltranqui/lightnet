//! CPU reference math for Tucker-factorized, windowed self-attention.
//!
//! The source implementation is audited in `tasks/rust/tucker_attention_plan.md`.
//! Rust backward differentiates this module's forward; it does not copy the
//! source's handwritten gradient path.

use crate::model::Activation;
use crate::vit::{sparse_softmax_row, tensor_activation, tensor_activation_grad};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TuckerGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub size: usize,
    pub heads: usize,
    pub rank_q: usize,
    pub rank_k: usize,
    pub rank_v: usize,
    pub rank_o: usize,
    pub activation: Activation,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuckerAttentionParams {
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
    channels: usize,
    heads: usize,
    rank_q: usize,
    rank_k: usize,
    rank_v: usize,
    rank_o: usize,
}

impl TuckerAttentionParams {
    pub fn zeroed(
        channels: usize,
        heads: usize,
        rank_q: usize,
        rank_k: usize,
        rank_v: usize,
        rank_o: usize,
    ) -> Result<Self, String> {
        let weight_count = tucker_weight_count(channels, heads, rank_q, rank_k, rank_v, rank_o)?;
        channels
            .checked_add(weight_count)
            .ok_or("Tucker parameter count overflow")?;
        Ok(Self {
            biases: vec![0.0; channels],
            weights: vec![0.0; weight_count],
            channels,
            heads,
            rank_q,
            rank_k,
            rank_v,
            rank_o,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.biases.len() + self.weights.len()
    }

    pub fn to_blob(&self) -> Result<Vec<f32>, String> {
        let expected = tucker_weight_count(
            self.channels,
            self.heads,
            self.rank_q,
            self.rank_k,
            self.rank_v,
            self.rank_o,
        )?;
        self.channels
            .checked_add(expected)
            .ok_or("Tucker parameter count overflow")?;
        if self.biases.len() != self.channels || self.weights.len() != expected {
            return Err(format!(
                "Tucker biases/weights have lengths {}/{}, expected {}/{}",
                self.biases.len(),
                self.weights.len(),
                self.channels,
                expected
            ));
        }
        let mut blob = Vec::with_capacity(self.parameter_count());
        blob.extend_from_slice(&self.biases);
        blob.extend_from_slice(&self.weights);
        Ok(blob)
    }

    pub fn from_blob(
        blob: &[f32],
        channels: usize,
        heads: usize,
        rank_q: usize,
        rank_k: usize,
        rank_v: usize,
        rank_o: usize,
    ) -> Result<Self, String> {
        let weight_count = tucker_weight_count(channels, heads, rank_q, rank_k, rank_v, rank_o)?;
        let expected = channels
            .checked_add(weight_count)
            .ok_or("Tucker blob size overflow")?;
        if blob.len() != expected {
            return Err(format!(
                "Tucker blob has {} values, expected {expected}",
                blob.len()
            ));
        }
        Ok(Self {
            biases: blob[..channels].to_vec(),
            weights: blob[channels..].to_vec(),
            channels,
            heads,
            rank_q,
            rank_k,
            rank_v,
            rank_o,
        })
    }

    pub(super) fn matches_geometry(&self, g: &TuckerGeometry) -> bool {
        self.channels == g.channels
            && self.heads == g.heads
            && self.rank_q == g.rank_q
            && self.rank_k == g.rank_k
            && self.rank_v == g.rank_v
            && self.rank_o == g.rank_o
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuckerAttentionForwardCache {
    geometry: TuckerGeometry,
    windows: Vec<TuckerWindowCache>,
}

#[derive(Clone, Debug, PartialEq)]
struct TuckerWindowCache {
    input: Vec<f32>,
    q_latent: Vec<f32>,
    k_latent: Vec<f32>,
    v_latent: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probabilities: Vec<f32>,
    score_grad_active: Vec<bool>,
    context: Vec<f32>,
    o_mix: Vec<f32>,
    pre_activation: Vec<f32>,
    activated: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TuckerAttentionGradients {
    pub input: Vec<f32>,
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct TuckerOffsets {
    q_basis: usize,
    k_basis: usize,
    v_basis: usize,
    q_core: usize,
    k_core: usize,
    v_core: usize,
    o_core: usize,
    o_basis: usize,
    total: usize,
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("Tucker {what} size overflow"))
}

pub(crate) fn tucker_weight_count(
    channels: usize,
    heads: usize,
    rank_q: usize,
    rank_k: usize,
    rank_v: usize,
    rank_o: usize,
) -> Result<usize, String> {
    if channels == 0
        || heads == 0
        || channels % heads != 0
        || rank_q == 0
        || rank_k == 0
        || rank_v == 0
        || rank_o == 0
    {
        return Err(
            "Tucker weight geometry requires positive ranks and channels divisible by heads".into(),
        );
    }
    let dim = channels / heads;
    let sizes = [
        checked_product(&[channels, rank_q], "Q basis")?,
        checked_product(&[channels, rank_k], "K basis")?,
        checked_product(&[channels, rank_v], "V basis")?,
        checked_product(&[heads, rank_q, dim], "Q core")?,
        checked_product(&[heads, rank_k, dim], "K core")?,
        checked_product(&[heads, rank_v, dim], "V core")?,
        checked_product(&[heads, dim, rank_o], "O core")?,
        checked_product(&[rank_o, channels], "O basis")?,
    ];
    sizes
        .into_iter()
        .try_fold(0usize, |sum, size| sum.checked_add(size))
        .ok_or_else(|| String::from("Tucker total weight count overflow"))
}

fn offsets(g: &TuckerGeometry) -> Result<TuckerOffsets, String> {
    let dim = g.channels / g.heads;
    let q_basis = 0usize;
    let k_basis = q_basis
        .checked_add(checked_product(&[g.channels, g.rank_q], "Q basis")?)
        .ok_or("Tucker weight offset overflow")?;
    let v_basis = k_basis
        .checked_add(checked_product(&[g.channels, g.rank_k], "K basis")?)
        .ok_or("Tucker weight offset overflow")?;
    let q_core = v_basis
        .checked_add(checked_product(&[g.channels, g.rank_v], "V basis")?)
        .ok_or("Tucker weight offset overflow")?;
    let k_core = q_core
        .checked_add(checked_product(&[g.heads, g.rank_q, dim], "Q core")?)
        .ok_or("Tucker weight offset overflow")?;
    let v_core = k_core
        .checked_add(checked_product(&[g.heads, g.rank_k, dim], "K core")?)
        .ok_or("Tucker weight offset overflow")?;
    let o_core = v_core
        .checked_add(checked_product(&[g.heads, g.rank_v, dim], "V core")?)
        .ok_or("Tucker weight offset overflow")?;
    let o_basis = o_core
        .checked_add(checked_product(&[g.heads, dim, g.rank_o], "O core")?)
        .ok_or("Tucker weight offset overflow")?;
    let total = o_basis
        .checked_add(checked_product(&[g.rank_o, g.channels], "O basis")?)
        .ok_or("Tucker weight offset overflow")?;
    Ok(TuckerOffsets {
        q_basis,
        k_basis,
        v_basis,
        q_core,
        k_core,
        v_core,
        o_core,
        o_basis,
        total,
    })
}

pub(crate) fn validate_geometry(g: &TuckerGeometry) -> Result<(), String> {
    if g.batch == 0
        || g.channels == 0
        || g.height == 0
        || g.width == 0
        || g.size == 0
        || g.heads == 0
        || g.channels % g.heads != 0
        || g.rank_q == 0
        || g.rank_k == 0
        || g.rank_v == 0
        || g.rank_o == 0
    {
        return Err(
            "Tucker geometry requires positive dimensions/ranks and channels divisible by heads"
                .into(),
        );
    }
    let tokens = checked_product(&[g.size, g.size], "window token count")?;
    let padded_h = g
        .height
        .div_ceil(g.size)
        .checked_mul(g.size)
        .ok_or("Tucker padded height overflow")?;
    let padded_w = g
        .width
        .div_ceil(g.size)
        .checked_mul(g.size)
        .ok_or("Tucker padded width overflow")?;
    let windows = checked_product(
        &[g.batch, padded_h / g.size, padded_w / g.size],
        "window count",
    )?;
    checked_product(&[windows, tokens, g.channels], "window input")?;
    checked_product(&[windows, g.heads, tokens, tokens], "attention scores")?;
    checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    for rank in [g.rank_q, g.rank_k, g.rank_v, g.rank_o] {
        checked_product(&[tokens, rank], "window rank workspace")?;
    }
    checked_product(&[tokens, g.channels], "window channel workspace")?;
    Ok(())
}

fn dense_softmax_source(scores: &[f32]) -> Vec<f32> {
    let mut max_score = scores[0];
    for &score in &scores[1..] {
        if max_score < score {
            max_score = score;
        }
    }
    let mut probabilities: Vec<f32> = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect();
    let sum: f32 = probabilities.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for probability in &mut probabilities {
        *probability *= inv;
    }
    probabilities
}

fn should_use_msa(g: &TuckerGeometry, tokens: usize) -> (bool, usize, usize) {
    let block = g.msa_block_size.max(1).min(tokens);
    let blocks = tokens.div_ceil(block);
    let topk = g.msa_topk.max(1).min(blocks);
    (g.msa_enabled && tokens > 1 && topk < blocks, block, topk)
}

fn project_latent(
    tokens: &[f32],
    basis: &[f32],
    rows: usize,
    channels: usize,
    rank: usize,
) -> Vec<f32> {
    let mut latent = vec![0.0; rows * rank];
    for t in 0..rows {
        for r in 0..rank {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += tokens[t * channels + c] * basis[c * rank + r];
            }
            latent[t * rank + r] = sum;
        }
    }
    latent
}

fn expand_heads(
    latent: &[f32],
    core: &[f32],
    tokens: usize,
    heads: usize,
    rank: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; tokens * heads * dim];
    for t in 0..tokens {
        for h in 0..heads {
            for r in 0..rank {
                let latent_value = latent[t * rank + r];
                for d in 0..dim {
                    out[(t * heads + h) * dim + d] += latent_value * core[(h * rank + r) * dim + d];
                }
            }
        }
    }
    out
}

fn forward_window(
    input: Vec<f32>,
    params: &TuckerAttentionParams,
    g: &TuckerGeometry,
    offsets: TuckerOffsets,
    use_msa: bool,
    msa_block: usize,
    msa_topk: usize,
) -> Result<(Vec<f32>, TuckerWindowCache), String> {
    let tokens = g.size * g.size;
    let dim = g.channels / g.heads;
    let q_basis = &params.weights[offsets.q_basis..offsets.k_basis];
    let k_basis = &params.weights[offsets.k_basis..offsets.v_basis];
    let v_basis = &params.weights[offsets.v_basis..offsets.q_core];
    let q_core = &params.weights[offsets.q_core..offsets.k_core];
    let k_core = &params.weights[offsets.k_core..offsets.v_core];
    let v_core = &params.weights[offsets.v_core..offsets.o_core];
    let o_core = &params.weights[offsets.o_core..offsets.o_basis];
    let o_basis = &params.weights[offsets.o_basis..offsets.total];
    let q_latent = project_latent(&input, q_basis, tokens, g.channels, g.rank_q);
    let k_latent = project_latent(&input, k_basis, tokens, g.channels, g.rank_k);
    let v_latent = project_latent(&input, v_basis, tokens, g.channels, g.rank_v);
    let q = expand_heads(&q_latent, q_core, tokens, g.heads, g.rank_q, dim);
    let k = expand_heads(&k_latent, k_core, tokens, g.heads, g.rank_k, dim);
    let v = expand_heads(&v_latent, v_core, tokens, g.heads, g.rank_v, dim);
    let scale = 1.0 / (dim as f32).sqrt();
    let score_count = checked_product(&[g.heads, tokens, tokens], "attention scores")?;
    let mut probabilities = vec![0.0; score_count];
    let mut score_grad_active = vec![true; score_count];
    for h in 0..g.heads {
        for tq in 0..tokens {
            let row = (h * tokens + tq) * tokens;
            let mut scores = vec![0.0; tokens];
            for tk in 0..tokens {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += q[(tq * g.heads + h) * dim + d] * k[(tk * g.heads + h) * dim + d];
                }
                scores[tk] = dot * scale;
                score_grad_active[row + tk] = scores[tk].is_finite();
            }
            let row_probabilities = if use_msa {
                sparse_softmax_row(&scores, tq, msa_block, msa_topk, 0.0)?
            } else {
                dense_softmax_source(&scores)
            };
            probabilities[row..row + tokens].copy_from_slice(&row_probabilities);
        }
    }
    let mut context = vec![0.0; tokens * g.channels];
    for h in 0..g.heads {
        for tq in 0..tokens {
            for tk in 0..tokens {
                let probability = probabilities[(h * tokens + tq) * tokens + tk];
                for d in 0..dim {
                    context[tq * g.channels + h * dim + d] +=
                        probability * v[(tk * g.heads + h) * dim + d];
                }
            }
        }
    }
    let mut o_mix = vec![0.0; tokens * g.rank_o];
    for t in 0..tokens {
        for r in 0..g.rank_o {
            for c in 0..g.channels {
                o_mix[t * g.rank_o + r] += context[t * g.channels + c] * o_core[c * g.rank_o + r];
            }
        }
    }
    let mut pre_activation = vec![0.0; tokens * g.channels];
    let mut activated = vec![0.0; tokens * g.channels];
    for t in 0..tokens {
        for n in 0..g.channels {
            let mut value = params.biases[n];
            for r in 0..g.rank_o {
                value += o_mix[t * g.rank_o + r] * o_basis[r * g.channels + n];
            }
            value += input[t * g.channels + n];
            pre_activation[t * g.channels + n] = value;
            activated[t * g.channels + n] = tensor_activation(&g.activation, value)?;
        }
    }
    Ok((
        activated.clone(),
        TuckerWindowCache {
            input,
            q_latent,
            k_latent,
            v_latent,
            q,
            k,
            v,
            probabilities,
            score_grad_active,
            context,
            o_mix,
            pre_activation,
            activated,
        },
    ))
}

pub(crate) fn tucker_attention_forward(
    input: &[f32],
    params: &TuckerAttentionParams,
    g: &TuckerGeometry,
) -> Result<(Vec<f32>, TuckerAttentionForwardCache), String> {
    validate_geometry(g)?;
    if !params.matches_geometry(g) {
        return Err("Tucker parameter geometry does not match the layer".into());
    }
    let offsets = offsets(g)?;
    if params.biases.len() != g.channels || params.weights.len() != offsets.total {
        return Err(format!(
            "Tucker biases/weights have lengths {}/{}, expected {}/{}",
            params.biases.len(),
            params.weights.len(),
            g.channels,
            offsets.total
        ));
    }
    let expected_input = checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    if input.len() != expected_input {
        return Err(format!(
            "Tucker input has {} values, expected {expected_input}",
            input.len()
        ));
    }
    let padded_h = g.height.div_ceil(g.size) * g.size;
    let padded_w = g.width.div_ceil(g.size) * g.size;
    let windows_h = padded_h / g.size;
    let windows_w = padded_w / g.size;
    let tokens = g.size * g.size;
    let (use_msa, msa_block, msa_topk) = should_use_msa(g, tokens);
    let mut output = vec![0.0; expected_input];
    let mut windows = Vec::with_capacity(checked_product(
        &[g.batch, windows_h, windows_w],
        "window count",
    )?);
    for b in 0..g.batch {
        for wy in 0..windows_h {
            for wx in 0..windows_w {
                let mut window_input = vec![0.0; tokens * g.channels];
                for yy in 0..g.size {
                    let y = wy * g.size + yy;
                    if y >= g.height {
                        continue;
                    }
                    for xx in 0..g.size {
                        let x = wx * g.size + xx;
                        if x >= g.width {
                            continue;
                        }
                        let t = yy * g.size + xx;
                        for c in 0..g.channels {
                            window_input[t * g.channels + c] =
                                input[((b * g.channels + c) * g.height + y) * g.width + x];
                        }
                    }
                }
                let (window_output, cache) = forward_window(
                    window_input,
                    params,
                    g,
                    offsets,
                    use_msa,
                    msa_block,
                    msa_topk,
                )?;
                for yy in 0..g.size {
                    let y = wy * g.size + yy;
                    if y >= g.height {
                        continue;
                    }
                    for xx in 0..g.size {
                        let x = wx * g.size + xx;
                        if x >= g.width {
                            continue;
                        }
                        let t = yy * g.size + xx;
                        for c in 0..g.channels {
                            output[((b * g.channels + c) * g.height + y) * g.width + x] =
                                window_output[t * g.channels + c];
                        }
                    }
                }
                windows.push(cache);
            }
        }
    }
    Ok((
        output,
        TuckerAttentionForwardCache {
            geometry: g.clone(),
            windows,
        },
    ))
}

pub(crate) fn tucker_attention_backward(
    output_gradient: &[f32],
    params: &TuckerAttentionParams,
    cache: &TuckerAttentionForwardCache,
) -> Result<TuckerAttentionGradients, String> {
    let g = &cache.geometry;
    validate_geometry(g)?;
    if !params.matches_geometry(g) {
        return Err("Tucker parameter geometry does not match the forward cache".into());
    }
    let expected = checked_product(&[g.batch, g.channels, g.height, g.width], "output gradient")?;
    if output_gradient.len() != expected {
        return Err(format!(
            "Tucker output gradient has {} values, expected {expected}",
            output_gradient.len()
        ));
    }
    let offsets = offsets(g)?;
    if params.weights.len() != offsets.total || params.biases.len() != g.channels {
        return Err("Tucker parameter lengths do not match the forward cache".into());
    }
    let padded_h = g.height.div_ceil(g.size) * g.size;
    let padded_w = g.width.div_ceil(g.size) * g.size;
    let windows_h = padded_h / g.size;
    let windows_w = padded_w / g.size;
    let expected_windows = checked_product(&[g.batch, windows_h, windows_w], "window count")?;
    if cache.windows.len() != expected_windows {
        return Err(format!(
            "Tucker forward cache has {} windows, expected {expected_windows}",
            cache.windows.len()
        ));
    }
    let dim = g.channels / g.heads;
    let tokens = g.size * g.size;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut gradients = TuckerAttentionGradients {
        input: vec![0.0; expected],
        biases: vec![0.0; g.channels],
        weights: vec![0.0; offsets.total],
    };
    let q_basis = &params.weights[offsets.q_basis..offsets.k_basis];
    let k_basis = &params.weights[offsets.k_basis..offsets.v_basis];
    let v_basis = &params.weights[offsets.v_basis..offsets.q_core];
    let q_core = &params.weights[offsets.q_core..offsets.k_core];
    let k_core = &params.weights[offsets.k_core..offsets.v_core];
    let v_core = &params.weights[offsets.v_core..offsets.o_core];
    let o_core = &params.weights[offsets.o_core..offsets.o_basis];
    let o_basis = &params.weights[offsets.o_basis..offsets.total];
    let (use_msa, _, _) = should_use_msa(g, tokens);
    let mut window_index = 0;
    for b in 0..g.batch {
        for wy in 0..windows_h {
            for wx in 0..windows_w {
                let window = &cache.windows[window_index];
                window_index += 1;
                if window.input.len() != tokens * g.channels
                    || window.probabilities.len() != g.heads * tokens * tokens
                    || window.score_grad_active.len() != g.heads * tokens * tokens
                    || window.context.len() != tokens * g.channels
                    || window.o_mix.len() != tokens * g.rank_o
                {
                    return Err("Tucker window cache shape mismatch".into());
                }
                let mut d_pre = vec![0.0; tokens * g.channels];
                for yy in 0..g.size {
                    let y = wy * g.size + yy;
                    if y >= g.height {
                        continue;
                    }
                    for xx in 0..g.size {
                        let x = wx * g.size + xx;
                        if x >= g.width {
                            continue;
                        }
                        let t = yy * g.size + xx;
                        for c in 0..g.channels {
                            let upstream = output_gradient
                                [((b * g.channels + c) * g.height + y) * g.width + x];
                            let index = t * g.channels + c;
                            d_pre[index] = upstream
                                * tensor_activation_grad(
                                    &g.activation,
                                    window.pre_activation[index],
                                    window.activated[index],
                                )?;
                        }
                    }
                }
                let mut d_tokens = vec![0.0; tokens * g.channels];
                let mut d_context = vec![0.0; tokens * g.channels];
                let mut d_o_mix = vec![0.0; tokens * g.rank_o];
                for t in 0..tokens {
                    for n in 0..g.channels {
                        let dy = d_pre[t * g.channels + n];
                        gradients.biases[n] += dy;
                        d_tokens[t * g.channels + n] += dy;
                        for r in 0..g.rank_o {
                            gradients.weights[offsets.o_basis + r * g.channels + n] +=
                                window.o_mix[t * g.rank_o + r] * dy;
                            d_o_mix[t * g.rank_o + r] += o_basis[r * g.channels + n] * dy;
                        }
                    }
                }
                for t in 0..tokens {
                    for c in 0..g.channels {
                        for r in 0..g.rank_o {
                            gradients.weights[offsets.o_core + c * g.rank_o + r] +=
                                window.context[t * g.channels + c] * d_o_mix[t * g.rank_o + r];
                            d_context[t * g.channels + c] +=
                                o_core[c * g.rank_o + r] * d_o_mix[t * g.rank_o + r];
                        }
                    }
                }
                let mut d_q = vec![0.0; tokens * g.heads * dim];
                let mut d_k = vec![0.0; tokens * g.heads * dim];
                let mut d_v = vec![0.0; tokens * g.heads * dim];
                for h in 0..g.heads {
                    for tq in 0..tokens {
                        let row = (h * tokens + tq) * tokens;
                        let mut d_prob = vec![0.0; tokens];
                        let mut dot = 0.0;
                        for tk in 0..tokens {
                            let probability = window.probabilities[row + tk];
                            for d in 0..dim {
                                let context_grad = d_context[tq * g.channels + h * dim + d];
                                let value_index = (tk * g.heads + h) * dim + d;
                                d_prob[tk] += context_grad * window.v[value_index];
                                d_v[value_index] += probability * context_grad;
                            }
                            dot += d_prob[tk] * probability;
                        }
                        for tk in 0..tokens {
                            let score_index = row + tk;
                            let d_score = window.probabilities[score_index] * (d_prob[tk] - dot);
                            if !window.score_grad_active[score_index] || d_score == 0.0 {
                                continue;
                            }
                            if use_msa && window.probabilities[score_index] == 0.0 {
                                continue;
                            }
                            for d in 0..dim {
                                let q_index = (tq * g.heads + h) * dim + d;
                                let k_index = (tk * g.heads + h) * dim + d;
                                d_q[q_index] += d_score * window.k[k_index] * scale;
                                d_k[k_index] += d_score * window.q[q_index] * scale;
                            }
                        }
                    }
                }
                let mut d_q_latent = vec![0.0; tokens * g.rank_q];
                let mut d_k_latent = vec![0.0; tokens * g.rank_k];
                let mut d_v_latent = vec![0.0; tokens * g.rank_v];
                for t in 0..tokens {
                    for h in 0..g.heads {
                        for d in 0..dim {
                            let q_grad = d_q[(t * g.heads + h) * dim + d];
                            for r in 0..g.rank_q {
                                let core_index = (h * g.rank_q + r) * dim + d;
                                gradients.weights[offsets.q_core + core_index] +=
                                    window.q_latent[t * g.rank_q + r] * q_grad;
                                d_q_latent[t * g.rank_q + r] += q_core[core_index] * q_grad;
                            }
                            let k_grad = d_k[(t * g.heads + h) * dim + d];
                            for r in 0..g.rank_k {
                                let core_index = (h * g.rank_k + r) * dim + d;
                                gradients.weights[offsets.k_core + core_index] +=
                                    window.k_latent[t * g.rank_k + r] * k_grad;
                                d_k_latent[t * g.rank_k + r] += k_core[core_index] * k_grad;
                            }
                            let v_grad = d_v[(t * g.heads + h) * dim + d];
                            for r in 0..g.rank_v {
                                let core_index = (h * g.rank_v + r) * dim + d;
                                gradients.weights[offsets.v_core + core_index] +=
                                    window.v_latent[t * g.rank_v + r] * v_grad;
                                d_v_latent[t * g.rank_v + r] += v_core[core_index] * v_grad;
                            }
                        }
                    }
                    for c in 0..g.channels {
                        for r in 0..g.rank_q {
                            gradients.weights[offsets.q_basis + c * g.rank_q + r] +=
                                window.input[t * g.channels + c] * d_q_latent[t * g.rank_q + r];
                            d_tokens[t * g.channels + c] +=
                                q_basis[c * g.rank_q + r] * d_q_latent[t * g.rank_q + r];
                        }
                        for r in 0..g.rank_k {
                            gradients.weights[offsets.k_basis + c * g.rank_k + r] +=
                                window.input[t * g.channels + c] * d_k_latent[t * g.rank_k + r];
                            d_tokens[t * g.channels + c] +=
                                k_basis[c * g.rank_k + r] * d_k_latent[t * g.rank_k + r];
                        }
                        for r in 0..g.rank_v {
                            gradients.weights[offsets.v_basis + c * g.rank_v + r] +=
                                window.input[t * g.channels + c] * d_v_latent[t * g.rank_v + r];
                            d_tokens[t * g.channels + c] +=
                                v_basis[c * g.rank_v + r] * d_v_latent[t * g.rank_v + r];
                        }
                    }
                }
                for yy in 0..g.size {
                    let y = wy * g.size + yy;
                    if y >= g.height {
                        continue;
                    }
                    for xx in 0..g.size {
                        let x = wx * g.size + xx;
                        if x >= g.width {
                            continue;
                        }
                        let t = yy * g.size + xx;
                        for c in 0..g.channels {
                            gradients.input[((b * g.channels + c) * g.height + y) * g.width + x] +=
                                d_tokens[t * g.channels + c];
                        }
                    }
                }
            }
        }
    }
    Ok(gradients)
}

#[cfg(test)]
mod tests {
    use super::{
        TuckerAttentionParams, TuckerGeometry, offsets, tucker_attention_backward,
        tucker_attention_forward,
    };
    use crate::model::Activation;

    fn geometry(height: usize, width: usize, size: usize, msa: bool) -> TuckerGeometry {
        TuckerGeometry {
            batch: 1,
            channels: 2,
            height,
            width,
            size,
            heads: 1,
            rank_q: 1,
            rank_k: 2,
            rank_v: 1,
            rank_o: 2,
            activation: Activation::Linear,
            msa_enabled: msa,
            msa_block_size: 1,
            msa_topk: 1,
        }
    }

    fn fill_weights(g: &TuckerGeometry, weights: &mut TuckerAttentionParams) {
        for (i, value) in weights.weights.iter_mut().enumerate() {
            *value = (i as f32 * 0.071).sin() * 0.4 + 0.2;
        }
        for (i, value) in weights.biases.iter_mut().enumerate() {
            *value = 0.03 * (i as f32 + 1.0);
        }
        assert_eq!(weights.weights.len(), offsets(g).unwrap().total);
    }

    fn loss(
        input: &[f32],
        params: &TuckerAttentionParams,
        g: &TuckerGeometry,
        seed: &[f32],
    ) -> f32 {
        let (output, _) = tucker_attention_forward(input, params, g).unwrap();
        output.iter().zip(seed).map(|(y, dy)| y * dy).sum()
    }

    #[test]
    fn factorized_local_attention_matches_hand_computation_for_one_token() {
        let mut g = geometry(1, 1, 1, false);
        g.channels = 1;
        g.heads = 1;
        g.rank_q = 1;
        g.rank_k = 1;
        g.rank_v = 1;
        g.rank_o = 1;
        let mut params = TuckerAttentionParams::zeroed(1, 1, 1, 1, 1, 1).unwrap();
        params.weights.fill(1.0);
        params.biases[0] = 0.5;
        let (output, cache) = tucker_attention_forward(&[2.0], &params, &g).unwrap();
        assert_eq!(output, [4.5]); // identity residual + bias + (2*1*1*1*1)
        let grad = tucker_attention_backward(&[1.0], &params, &cache).unwrap();
        assert_eq!(grad.input.len(), 1);
        assert!(grad.input[0].is_finite());
    }

    #[test]
    fn two_by_two_window_matches_hand_computed_attention_rows() {
        let mut g = geometry(2, 2, 2, false);
        g.channels = 1;
        g.rank_k = 1;
        g.rank_o = 1;
        let mut params = TuckerAttentionParams::zeroed(1, 1, 1, 1, 1, 1).unwrap();
        params.weights.fill(1.0);
        let input = [0.0, 1.0, 2.0, 3.0];
        let (output, _) = tucker_attention_forward(&input, &params, &g).unwrap();
        let mut expected = [0.0; 4];
        for query in 0..4 {
            let scores = input.map(|key| input[query] * key);
            let max_score = scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = scores.map(|score| (score - max_score).exp());
            let denominator: f32 = exponentials.iter().sum();
            let context: f32 = exponentials
                .iter()
                .zip(input)
                .map(|(value, key)| value / denominator * key)
                .sum();
            expected[query] = context + input[query];
        }
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn axis_layout_keeps_batch_window_token_and_channel_axes_independent() {
        let mut g = geometry(1, 2, 2, false);
        g.batch = 2;
        g.heads = 2;
        g.rank_q = 1;
        g.rank_k = 1;
        g.rank_v = 1;
        g.rank_o = 1;
        let mut params = TuckerAttentionParams::zeroed(2, 2, 1, 1, 1, 1).unwrap();
        let o = offsets(&g).unwrap();
        params.weights[o.q_basis..o.k_basis].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.k_basis..o.v_basis].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.v_basis..o.q_core].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.q_core..o.k_core].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.k_core..o.v_core].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.v_core..o.o_core].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.o_core..o.o_basis].copy_from_slice(&[1.0, 0.0]);
        params.weights[o.o_basis..o.total].copy_from_slice(&[1.0, 0.0]);

        // NCHW: channel 0 drives attention; channel 1 is an independent sentinel.
        // Each 1x2 image is padded to one 2x2 window with two zero tokens.
        let input = [1.0, 2.0, 101.0, 102.0, 10.0, 20.0, 201.0, 202.0];
        let (output, _) = tucker_attention_forward(&input, &params, &g).unwrap();
        let window_context = |values: [f32; 2], query: f32| {
            let scores = [query * values[0], query * values[1], 0.0, 0.0];
            let maximum = scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
            let probabilities = scores.map(|score| (score - maximum).exp());
            (probabilities[0] * values[0] + probabilities[1] * values[1])
                / probabilities.into_iter().sum::<f32>()
        };
        let expected = [
            1.0 + window_context([1.0, 2.0], 1.0),
            2.0 + window_context([1.0, 2.0], 2.0),
            101.0,
            102.0,
            10.0 + window_context([10.0, 20.0], 10.0),
            20.0 + window_context([10.0, 20.0], 20.0),
            201.0,
            202.0,
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn partial_windows_zero_pad_but_preserve_valid_identity_residuals() {
        let mut g = geometry(3, 3, 2, true);
        g.msa_block_size = 1;
        g.msa_topk = 1;
        let mut params = TuckerAttentionParams::zeroed(2, 1, 1, 2, 1, 2).unwrap();
        params.biases = vec![0.0; 2];
        let input: Vec<f32> = (0..18).map(|i| i as f32 * 0.1 - 0.7).collect();
        let (output, cache) = tucker_attention_forward(&input, &params, &g).unwrap();
        assert_eq!(output, input);
        assert_eq!(cache.windows.len(), 4);
        let grad = tucker_attention_backward(&vec![1.0; input.len()], &params, &cache).unwrap();
        assert_eq!(grad.input, vec![1.0; input.len()]);
    }

    #[test]
    fn all_factor_blocks_and_input_gradient_match_finite_differences() {
        let mut g = geometry(2, 2, 2, false);
        g.activation = Activation::Tanh;
        let mut params = TuckerAttentionParams::zeroed(2, 1, 1, 2, 1, 2).unwrap();
        fill_weights(&g, &mut params);
        let input = [-0.31, 0.22, 0.48, -0.17, 0.15, 0.33, -0.42, 0.27];
        let seed = [0.3, -0.5, 0.7, -0.2, -0.6, 0.1, 0.4, 0.8];
        let (_, cache) = tucker_attention_forward(&input, &params, &g).unwrap();
        let gradients = tucker_attention_backward(&seed, &params, &cache).unwrap();
        let eps = 1.0e-3;
        for index in [0, 1, 4, 7, 10, 13, 16, 20] {
            let analytic = gradients.weights[index];
            let old = params.weights[index];
            params.weights[index] = old + eps;
            let plus = loss(&input, &params, &g, &seed);
            params.weights[index] = old - eps;
            let minus = loss(&input, &params, &g, &seed);
            params.weights[index] = old;
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (analytic - numeric).abs() < 2.0e-3,
                "weight[{index}] {analytic} vs {numeric}"
            );
        }
        for index in 0..params.biases.len() {
            let analytic = gradients.biases[index];
            let old = params.biases[index];
            params.biases[index] = old + eps;
            let plus = loss(&input, &params, &g, &seed);
            params.biases[index] = old - eps;
            let minus = loss(&input, &params, &g, &seed);
            params.biases[index] = old;
            assert!((analytic - (plus - minus) / (2.0 * eps)).abs() < 2.0e-3);
        }
        for index in 0..input.len() {
            let mut x = input;
            x[index] += eps;
            let plus = loss(&x, &params, &g, &seed);
            x[index] -= 2.0 * eps;
            let minus = loss(&x, &params, &g, &seed);
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (gradients.input[index] - numeric).abs() < 3.0e-3,
                "input[{index}] {} vs {numeric}",
                gradients.input[index]
            );
        }
    }

    #[test]
    fn sparse_attention_backward_matches_finite_differences_with_stable_routes() {
        let mut g = geometry(2, 2, 2, true);
        g.msa_block_size = 1;
        g.msa_topk = 2;
        let mut params = TuckerAttentionParams::zeroed(2, 1, 1, 2, 1, 2).unwrap();
        fill_weights(&g, &mut params);
        let input = [-0.4, 0.2, 0.6, -0.1, 0.3, 0.8, -0.5, 0.4];
        let seed = [0.3, -0.5, 0.7, -0.2, -0.6, 0.1, 0.4, 0.8];
        let (_, cache) = tucker_attention_forward(&input, &params, &g).unwrap();
        let gradients = tucker_attention_backward(&seed, &params, &cache).unwrap();
        let eps = 1.0e-4;
        for index in 0..input.len() {
            let mut x = input;
            x[index] += eps;
            let plus = loss(&x, &params, &g, &seed);
            x[index] -= 2.0 * eps;
            let minus = loss(&x, &params, &g, &seed);
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (gradients.input[index] - numeric).abs() < 4.0e-3,
                "input[{index}] {} vs {numeric}",
                gradients.input[index]
            );
        }
    }

    #[test]
    fn sparse_attention_disables_when_topk_covers_every_block() {
        let mut sparse = geometry(2, 2, 2, true);
        sparse.msa_block_size = 1;
        sparse.msa_topk = 4;
        let mut dense = sparse.clone();
        dense.msa_enabled = false;
        let mut params = TuckerAttentionParams::zeroed(2, 1, 1, 2, 1, 2).unwrap();
        fill_weights(&sparse, &mut params);
        let input = [-0.31, 0.22, 0.48, -0.17, 0.15, 0.33, -0.42, 0.27];
        let (sparse_output, _) = tucker_attention_forward(&input, &params, &sparse).unwrap();
        let (dense_output, _) = tucker_attention_forward(&input, &params, &dense).unwrap();
        assert_eq!(sparse_output, dense_output);
    }

    #[test]
    fn multi_head_attention_input_gradient_matches_finite_difference() {
        let mut g = geometry(2, 2, 2, false);
        g.heads = 2;
        g.rank_k = 1;
        let mut params = TuckerAttentionParams::zeroed(2, 2, 1, 1, 1, 2).unwrap();
        fill_weights(&g, &mut params);
        let input = [-0.31, 0.22, 0.48, -0.17, 0.15, 0.33, -0.42, 0.27];
        let seed = [0.3, -0.5, 0.7, -0.2, -0.6, 0.1, 0.4, 0.8];
        let (_, cache) = tucker_attention_forward(&input, &params, &g).unwrap();
        let gradients = tucker_attention_backward(&seed, &params, &cache).unwrap();
        let eps = 1.0e-3;
        let mut plus = input;
        plus[3] += eps;
        let mut minus = input;
        minus[3] -= eps;
        let numeric =
            (loss(&plus, &params, &g, &seed) - loss(&minus, &params, &g, &seed)) / (2.0 * eps);
        assert!((gradients.input[3] - numeric).abs() < 3.0e-3);
    }

    #[test]
    fn tucker_parameter_blob_is_biases_then_flat_weights() {
        let mut params = TuckerAttentionParams::zeroed(2, 1, 1, 2, 1, 2).unwrap();
        params.biases.copy_from_slice(&[0.4, -0.2]);
        for (i, value) in params.weights.iter_mut().enumerate() {
            *value = i as f32 + 1.0;
        }
        let blob = params.to_blob().unwrap();
        assert_eq!(&blob[..2], &[0.4, -0.2]);
        assert_eq!(&blob[2..], params.weights);
        assert_eq!(
            TuckerAttentionParams::from_blob(&blob, 2, 1, 1, 2, 1, 2).unwrap(),
            params
        );
        assert!(
            TuckerAttentionParams::from_blob(&blob[..blob.len() - 1], 2, 1, 1, 2, 1, 2).is_err()
        );
    }
}
