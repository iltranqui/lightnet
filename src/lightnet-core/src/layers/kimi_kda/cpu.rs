//! CPU reference math for the source-derived Kimi delta-rule/MoE image mixer.
//!
//! The audited C++ reference is on `Jev_laya`. This implementation retains its
//! `[B,C,H,W]` input/output layout, low-rank causal scan, shared/routed experts,
//! and flat `.weights` order. Backward differentiates this Rust forward exactly.

use std::ops::Range;

use crate::model::{Activation, TensorShape};
use crate::vit::{
    layer_norm_backward, layer_norm_forward, tensor_activation, tensor_activation_grad,
};

const LAYERNORM_EPS: f32 = 1.0e-5;
const KEY_NORM_FLOOR: f32 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct KimiKdaParams {
    /// Source stream stores the complete biases array before the weights array.
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
    channels: usize,
    rank: usize,
    experts: usize,
    hidden: usize,
}

impl KimiKdaParams {
    pub fn zeroed(
        channels: usize,
        rank: usize,
        experts: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        let offsets = KimiOffsets::new(channels, rank, experts, hidden)?;
        Ok(Self {
            biases: vec![0.0; offsets.total_biases],
            weights: vec![0.0; offsets.total_weights],
            channels,
            rank,
            experts,
            hidden,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.biases.len() + self.weights.len()
    }

    pub fn to_blob(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let mut blob = Vec::with_capacity(self.parameter_count());
        blob.extend_from_slice(&self.biases);
        blob.extend_from_slice(&self.weights);
        Ok(blob)
    }

    pub fn from_blob(
        blob: &[f32],
        channels: usize,
        rank: usize,
        experts: usize,
        hidden: usize,
    ) -> Result<Self, String> {
        let offsets = KimiOffsets::new(channels, rank, experts, hidden)?;
        let total = offsets
            .total_biases
            .checked_add(offsets.total_weights)
            .ok_or("Kimi KDA blob size overflow")?;
        if blob.len() != total {
            return Err(format!(
                "Kimi KDA parameter blob has {} values, expected {total}",
                blob.len()
            ));
        }
        Ok(Self {
            biases: blob[..offsets.total_biases].to_vec(),
            weights: blob[offsets.total_biases..].to_vec(),
            channels,
            rank,
            experts,
            hidden,
        })
    }

    fn validate(&self) -> Result<(), String> {
        let offsets = KimiOffsets::new(self.channels, self.rank, self.experts, self.hidden)?;
        if self.biases.len() != offsets.total_biases || self.weights.len() != offsets.total_weights
        {
            return Err(format!(
                "Kimi KDA biases/weights have lengths {}/{}, expected {}/{}",
                self.biases.len(),
                self.weights.len(),
                offsets.total_biases,
                offsets.total_weights
            ));
        }
        Ok(())
    }

    fn matches(&self, g: &KimiKdaGeometry) -> bool {
        self.channels == g.channels
            && self.rank == g.rank
            && self.experts == g.experts
            && self.hidden == g.hidden
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiKdaForwardCache {
    /// C++ `l.cost`: load-balance auxiliary objective for this layer.
    pub load_balance_cost: f32,
    geometry: KimiKdaGeometry,
    tokens: Vec<f32>,
    ln1: Vec<f32>,
    ln1_cache: crate::vit::LayerNormCache,
    q: Vec<f32>,
    k: Vec<f32>,
    key_inverse_norm: Vec<f32>,
    key_norm_squared: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
    state: Vec<f32>,
    attention: Vec<f32>,
    pre_expert: Vec<f32>,
    ln2: Vec<f32>,
    ln2_cache: crate::vit::LayerNormCache,
    router_logits: Vec<f32>,
    gates: Vec<f32>,
    expert_pre: Vec<f32>,
    expert_act: Vec<f32>,
    expert_out: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KimiKdaGradients {
    pub input: Vec<f32>,
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KimiKdaGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub rank: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub temperature: f32,
    pub load_balance: f32,
    pub activation: Activation,
}

impl KimiKdaGeometry {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.batch == 0 || self.channels == 0 || self.height == 0 || self.width == 0 {
            return Err("Kimi KDA requires a nonempty BxCxHxW input".into());
        }
        if self.rank == 0 || self.rank > self.channels || self.rank > 64 {
            return Err(format!(
                "Kimi KDA rank must be in 1..=min(channels, 64), got {} for C={}",
                self.rank, self.channels
            ));
        }
        if self.experts == 0 || self.experts > 64 || self.top_k > self.experts {
            return Err(format!(
                "Kimi KDA requires 1..=64 experts and top_k in 0..=experts, got E={} top_k={}",
                self.experts, self.top_k
            ));
        }
        if self.hidden == 0 {
            return Err("Kimi KDA hidden width must be positive".into());
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err("Kimi KDA temperature must be finite and positive".into());
        }
        if !self.load_balance.is_finite() || self.load_balance < 0.0 {
            return Err("Kimi KDA load_balance must be finite and nonnegative".into());
        }
        if matches!(
            self.activation,
            Activation::NormalizeChannels
                | Activation::NormalizeChannelsSoftmax
                | Activation::NormalizeChannelsSoftmaxMaxval
                | Activation::Other(_)
        ) {
            return Err("Kimi KDA requires a supported scalar activation".into());
        }
        let tokens = checked_product(&[self.height, self.width], "token count")?;
        let rows = checked_product(&[self.batch, tokens], "token batch")?;
        let p = self
            .experts
            .checked_add(1)
            .ok_or("Kimi KDA expert count overflow")?;
        checked_product(&[self.batch, self.channels, tokens], "input")?;
        checked_product(&[rows, self.channels], "token workspace")?;
        checked_product(&[rows, self.rank, self.channels], "scan state")?;
        checked_product(&[rows, self.experts], "router workspace")?;
        checked_product(&[p, rows, self.hidden], "expert hidden workspace")?;
        checked_product(&[p, rows, self.channels], "expert output workspace")?;
        KimiOffsets::new(self.channels, self.rank, self.experts, self.hidden)?;
        Ok(())
    }

    pub(crate) fn from_layer(
        input: TensorShape,
        spec: &crate::model::KimiKdaSpec,
    ) -> Result<Self, String> {
        if input.channels != spec.filters {
            return Err(format!(
                "Kimi KDA filters={} must equal input channels={}",
                spec.filters, input.channels
            ));
        }
        let g = Self {
            batch: input.batch,
            channels: input.channels,
            height: input.height,
            width: input.width,
            rank: spec.rank,
            experts: spec.experts,
            top_k: spec.top_k,
            hidden: spec.hidden,
            temperature: spec.temperature,
            load_balance: spec.load_balance,
            activation: spec.activation.clone(),
        };
        g.validate()?;
        Ok(g)
    }

    pub(crate) fn tokens(&self) -> usize {
        self.height * self.width
    }

    pub(crate) fn rows(&self) -> usize {
        self.batch * self.tokens()
    }
}

pub(crate) fn validate_geometry(
    input: TensorShape,
    spec: &crate::model::KimiKdaSpec,
) -> Result<(), String> {
    KimiKdaGeometry::from_layer(input, spec).map(|_| ())
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KimiKdaWeightBlock {
    pub name: &'static str,
    pub range: Range<usize>,
}

pub(crate) fn kimi_kda_parameter_counts(
    channels: usize,
    rank: usize,
    experts: usize,
    hidden: usize,
) -> Result<(usize, usize), String> {
    let offsets = KimiOffsets::new(channels, rank, experts, hidden)?;
    Ok((offsets.total_biases, offsets.total_weights))
}

pub(crate) fn kimi_kda_weight_blocks(
    channels: usize,
    rank: usize,
    experts: usize,
    hidden: usize,
) -> Result<Vec<KimiKdaWeightBlock>, String> {
    let o = KimiOffsets::new(channels, rank, experts, hidden)?;
    Ok(vec![
        KimiKdaWeightBlock {
            name: "q",
            range: o.wq..o.wk,
        },
        KimiKdaWeightBlock {
            name: "k",
            range: o.wk..o.wv,
        },
        KimiKdaWeightBlock {
            name: "v",
            range: o.wv..o.wo,
        },
        KimiKdaWeightBlock {
            name: "o",
            range: o.wo..o.wbeta,
        },
        KimiKdaWeightBlock {
            name: "beta",
            range: o.wbeta..o.wdecay,
        },
        KimiKdaWeightBlock {
            name: "decay",
            range: o.wdecay..o.wrouter,
        },
        KimiKdaWeightBlock {
            name: "router",
            range: o.wrouter..o.w1,
        },
        KimiKdaWeightBlock {
            name: "expert_w1",
            range: o.w1..o.w2,
        },
        KimiKdaWeightBlock {
            name: "expert_w2",
            range: o.w2..o.total_weights,
        },
    ])
}

pub(crate) fn initialize_kimi_kda_params(
    params: &mut KimiKdaParams,
    mut uniform_weights: impl FnMut(&mut [f32], f32),
) -> Result<(), String> {
    params.validate()?;
    let offsets = KimiOffsets::new(params.channels, params.rank, params.experts, params.hidden)?;
    params.biases.fill(0.0);
    params.biases[offsets.ln1_gamma..offsets.ln1_gamma + params.channels].fill(1.0);
    params.biases[offsets.ln2_gamma..offsets.ln2_gamma + params.channels].fill(1.0);
    params.biases[offsets.bbeta..offsets.bbeta + params.rank].fill(-2.0);
    params.biases[offsets.bdecay..offsets.bdecay + params.rank].fill(2.0);
    params.weights.fill(0.0);
    let bound_qk = (6.0 / (params.channels + params.rank) as f32).sqrt();
    let bound_v = (6.0 / (2 * params.channels) as f32).sqrt();
    let bound_expert = (6.0 / (params.channels + params.hidden) as f32).sqrt();
    for block in
        kimi_kda_weight_blocks(params.channels, params.rank, params.experts, params.hidden)?
    {
        let bound = match block.name {
            "q" | "k" | "beta" | "decay" => bound_qk,
            "v" => bound_v,
            "router" => bound_qk * 0.05,
            "expert_w1" => bound_expert,
            "o" | "expert_w2" => 0.0,
            _ => unreachable!("known Kimi KDA weight block"),
        };
        if bound > 0.0 {
            uniform_weights(&mut params.weights[block.range], bound);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct KimiOffsets {
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    wbeta: usize,
    wdecay: usize,
    wrouter: usize,
    w1: usize,
    w2: usize,
    total_weights: usize,
    ln1_gamma: usize,
    ln1_beta: usize,
    ln2_gamma: usize,
    ln2_beta: usize,
    bq: usize,
    bk: usize,
    bv: usize,
    bbeta: usize,
    bdecay: usize,
    bo: usize,
    brouter: usize,
    b1: usize,
    b2: usize,
    total_biases: usize,
}

impl KimiOffsets {
    fn new(channels: usize, rank: usize, experts: usize, hidden: usize) -> Result<Self, String> {
        if channels == 0 || rank == 0 || experts == 0 || hidden == 0 {
            return Err("Kimi KDA parameter geometry must be positive".into());
        }
        let p = experts
            .checked_add(1)
            .ok_or("Kimi KDA expert count overflow")?;
        let rc = checked_product(&[rank, channels], "Q/K projection")?;
        let cc = checked_product(&[channels, channels], "channel projection")?;
        let ec = checked_product(&[experts, channels], "router projection")?;
        let phc = checked_product(&[p, hidden, channels], "expert W1")?;
        let pch = checked_product(&[p, channels, hidden], "expert W2")?;

        let wq = 0;
        let wk = checked_add(wq, rc, "weight offset")?;
        let wv = checked_add(wk, rc, "weight offset")?;
        let wo = checked_add(wv, cc, "weight offset")?;
        let wbeta = checked_add(wo, cc, "weight offset")?;
        let wdecay = checked_add(wbeta, rc, "weight offset")?;
        let wrouter = checked_add(wdecay, rc, "weight offset")?;
        let w1 = checked_add(wrouter, ec, "weight offset")?;
        let w2 = checked_add(w1, phc, "weight offset")?;
        let total_weights = checked_add(w2, pch, "weight count")?;

        let ln1_gamma = 0;
        let ln1_beta = channels;
        let ln2_gamma = checked_product(&[2, channels], "bias offset")?;
        let ln2_beta = checked_product(&[3, channels], "bias offset")?;
        let bq = checked_product(&[4, channels], "bias offset")?;
        let bk = checked_add(bq, rank, "bias offset")?;
        let bv = checked_add(bk, rank, "bias offset")?;
        let bbeta = checked_add(bv, channels, "bias offset")?;
        let bdecay = checked_add(bbeta, rank, "bias offset")?;
        let bo = checked_add(bdecay, rank, "bias offset")?;
        let brouter = checked_add(bo, channels, "bias offset")?;
        let b1 = checked_add(brouter, experts, "bias offset")?;
        let b2 = checked_add(
            b1,
            checked_product(&[p, hidden], "expert b1")?,
            "bias offset",
        )?;
        let total_biases = checked_add(
            b2,
            checked_product(&[p, channels], "expert b2")?,
            "bias count",
        )?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            wbeta,
            wdecay,
            wrouter,
            w1,
            w2,
            total_weights,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            bq,
            bk,
            bv,
            bbeta,
            bdecay,
            bo,
            brouter,
            b1,
            b2,
            total_biases,
        })
    }
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("Kimi KDA {what} size overflow"))
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("Kimi KDA {what} overflow"))
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn flatten_nchw(input: &[f32], g: &KimiKdaGeometry) -> Vec<f32> {
    let tokens = g.tokens();
    let mut output = vec![0.0; g.batch * tokens * g.channels];
    for b in 0..g.batch {
        for t in 0..tokens {
            for c in 0..g.channels {
                output[(b * tokens + t) * g.channels + c] =
                    input[(b * g.channels + c) * tokens + t];
            }
        }
    }
    output
}

fn deflatten_tokens(tokens: &[f32], g: &KimiKdaGeometry) -> Vec<f32> {
    let spatial = g.tokens();
    let mut output = vec![0.0; g.batch * g.channels * spatial];
    for b in 0..g.batch {
        for t in 0..spatial {
            for c in 0..g.channels {
                output[(b * g.channels + c) * spatial + t] =
                    tokens[(b * spatial + t) * g.channels + c];
            }
        }
    }
    output
}

fn softmax_gates(logits: &[f32], experts: usize, top_k: usize) -> Vec<f32> {
    let mut selected = vec![false; experts];
    for e in 0..experts {
        let rank = (0..experts)
            .filter(|&j| logits[j] > logits[e] || (logits[j] == logits[e] && j < e))
            .count();
        selected[e] = top_k == 0 || rank < top_k;
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut gates = vec![0.0; experts];
    let mut sum = 0.0;
    for e in 0..experts {
        if selected[e] {
            gates[e] = (logits[e] - maximum).exp();
            sum += gates[e];
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        sum = 1.0;
    }
    for gate in &mut gates {
        *gate /= sum;
    }
    gates
}

pub(crate) fn kimi_kda_forward(
    input: &[f32],
    input_shape: TensorShape,
    spec: &crate::model::KimiKdaSpec,
    params: &KimiKdaParams,
) -> Result<(Vec<f32>, KimiKdaForwardCache), String> {
    let g = KimiKdaGeometry::from_layer(input_shape, spec)?;
    params.validate()?;
    if !params.matches(&g) {
        return Err("Kimi KDA parameters do not match layer geometry".into());
    }
    let expected_input = checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    if input.len() != expected_input {
        return Err(format!(
            "Kimi KDA input has {} values, expected {expected_input}",
            input.len()
        ));
    }

    let offsets = KimiOffsets::new(g.channels, g.rank, g.experts, g.hidden)?;
    let tokens = flatten_nchw(input, &g);
    let rows = g.rows();
    let channels = g.channels;
    let rank = g.rank;
    let experts = g.experts;
    let hidden = g.hidden;
    let shared_and_routed = experts + 1;
    let sqrt_rank = (rank as f32).sqrt();

    let (ln1, ln1_cache) = layer_norm_forward(
        &tokens,
        &params.biases[offsets.ln1_gamma..offsets.ln1_gamma + channels],
        &params.biases[offsets.ln1_beta..offsets.ln1_beta + channels],
        rows,
        channels,
        LAYERNORM_EPS,
    )?;

    let mut q = vec![0.0; rows * rank];
    let mut k = vec![0.0; rows * rank];
    let mut key_inverse_norm = vec![0.0; rows];
    let mut key_norm_squared = vec![0.0; rows];
    let mut v = vec![0.0; rows * channels];
    let mut beta = vec![0.0; rows * rank];
    let mut decay = vec![0.0; rows * rank];
    for row in 0..rows {
        let x = &ln1[row * channels..(row + 1) * channels];
        for r in 0..rank {
            let mut q_value = params.biases[offsets.bq + r];
            let mut k_value = params.biases[offsets.bk + r];
            let mut beta_value = params.biases[offsets.bbeta + r];
            let mut decay_value = params.biases[offsets.bdecay + r];
            for c in 0..channels {
                q_value += params.weights[offsets.wq + r * channels + c] * x[c];
                k_value += params.weights[offsets.wk + r * channels + c] * x[c];
                beta_value += params.weights[offsets.wbeta + r * channels + c] * x[c];
                decay_value += params.weights[offsets.wdecay + r * channels + c] * x[c];
            }
            q[row * rank + r] = q_value;
            k[row * rank + r] = k_value;
            beta[row * rank + r] = sigmoid(beta_value);
            decay[row * rank + r] = sigmoid(decay_value);
        }
        let norm_squared = k[row * rank..(row + 1) * rank]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        key_norm_squared[row] = norm_squared;
        if !norm_squared.is_finite() {
            k[row * rank..(row + 1) * rank].fill(0.0);
            key_inverse_norm[row] = 0.0;
        } else {
            let inverse_norm = 1.0 / norm_squared.max(KEY_NORM_FLOOR).sqrt();
            for value in &mut k[row * rank..(row + 1) * rank] {
                *value *= inverse_norm;
            }
            key_inverse_norm[row] = inverse_norm;
        }
        for c in 0..channels {
            let mut value = params.biases[offsets.bv + c];
            for i in 0..channels {
                value += params.weights[offsets.wv + c * channels + i] * x[i];
            }
            v[row * channels + c] = value;
        }
    }

    let mut state = vec![0.0; rows * rank * channels];
    let mut attention = vec![0.0; rows * channels];
    for b in 0..g.batch {
        for t in 0..g.tokens() {
            let row = b * g.tokens() + t;
            for c in 0..channels {
                let mut prediction = 0.0;
                for r in 0..rank {
                    if t > 0 {
                        prediction += k[row * rank + r]
                            * state[(row - 1) * rank * channels + r * channels + c];
                    }
                }
                let error = v[row * channels + c] - prediction;
                for r in 0..rank {
                    let previous = if t > 0 {
                        state[(row - 1) * rank * channels + r * channels + c]
                    } else {
                        0.0
                    };
                    state[row * rank * channels + r * channels + c] = decay[row * rank + r]
                        * previous
                        + beta[row * rank + r] * k[row * rank + r] * error;
                }
            }
            for c in 0..channels {
                let mut value = 0.0;
                for r in 0..rank {
                    value += q[row * rank + r] * state[row * rank * channels + r * channels + c];
                }
                attention[row * channels + c] = value / sqrt_rank;
            }
        }
    }

    let mut pre_expert = vec![0.0; rows * channels];
    for row in 0..rows {
        for j in 0..channels {
            let mut value = params.biases[offsets.bo + j];
            for i in 0..channels {
                value +=
                    params.weights[offsets.wo + j * channels + i] * attention[row * channels + i];
            }
            pre_expert[row * channels + j] = tokens[row * channels + j] + value;
        }
    }
    let (ln2, ln2_cache) = layer_norm_forward(
        &pre_expert,
        &params.biases[offsets.ln2_gamma..offsets.ln2_gamma + channels],
        &params.biases[offsets.ln2_beta..offsets.ln2_beta + channels],
        rows,
        channels,
        LAYERNORM_EPS,
    )?;

    let mut router_logits = vec![0.0; rows * experts];
    let mut gates = vec![0.0; rows * experts];
    let mut expert_pre = vec![0.0; shared_and_routed * rows * hidden];
    let mut expert_act = vec![0.0; shared_and_routed * rows * hidden];
    let mut expert_out = vec![0.0; shared_and_routed * rows * channels];
    for row in 0..rows {
        let x = &ln2[row * channels..(row + 1) * channels];
        for e in 0..experts {
            let mut logit = params.biases[offsets.brouter + e];
            for c in 0..channels {
                logit += params.weights[offsets.wrouter + e * channels + c] * x[c];
            }
            router_logits[row * experts + e] = logit / g.temperature;
        }
        gates[row * experts..(row + 1) * experts].copy_from_slice(&softmax_gates(
            &router_logits[row * experts..(row + 1) * experts],
            experts,
            g.top_k,
        ));
        for expert in 0..shared_and_routed {
            let hidden_base = (expert * rows + row) * hidden;
            let output_base = (expert * rows + row) * channels;
            for j in 0..hidden {
                let mut value = params.biases[offsets.b1 + expert * hidden + j];
                for c in 0..channels {
                    value +=
                        params.weights[offsets.w1 + (expert * hidden + j) * channels + c] * x[c];
                }
                expert_pre[hidden_base + j] = value;
                expert_act[hidden_base + j] = tensor_activation(&g.activation, value)?;
            }
            for c in 0..channels {
                let mut value = params.biases[offsets.b2 + expert * channels + c];
                for j in 0..hidden {
                    value += params.weights[offsets.w2 + (expert * channels + c) * hidden + j]
                        * expert_act[hidden_base + j];
                }
                expert_out[output_base + c] = value;
            }
        }
    }

    let mut output_tokens = vec![0.0; rows * channels];
    for row in 0..rows {
        let shared = &expert_out[row * channels..(row + 1) * channels];
        for c in 0..channels {
            let mut value = pre_expert[row * channels + c] + shared[c];
            for e in 0..experts {
                value +=
                    gates[row * experts + e] * expert_out[((e + 1) * rows + row) * channels + c];
            }
            output_tokens[row * channels + c] = value;
        }
    }

    let load_balance_cost = if g.load_balance > 0.0 {
        let mut means = vec![0.0; experts];
        for row in 0..rows {
            for e in 0..experts {
                means[e] += gates[row * experts + e];
            }
        }
        let mut squared_sum = 0.0;
        for mean in &mut means {
            *mean /= rows as f32;
            squared_sum += *mean * *mean;
        }
        g.load_balance * (experts as f32 * squared_sum - 1.0).max(0.0)
    } else {
        0.0
    };

    Ok((
        deflatten_tokens(&output_tokens, &g),
        KimiKdaForwardCache {
            load_balance_cost,
            geometry: g,
            tokens,
            ln1,
            ln1_cache,
            q,
            k,
            key_inverse_norm,
            key_norm_squared,
            v,
            beta,
            decay,
            state,
            attention,
            pre_expert,
            ln2,
            ln2_cache,
            router_logits,
            gates,
            expert_pre,
            expert_act,
            expert_out,
        },
    ))
}

pub(crate) fn kimi_kda_backward(
    output_gradient: &[f32],
    params: &KimiKdaParams,
    cache: &KimiKdaForwardCache,
) -> Result<KimiKdaGradients, String> {
    let g = &cache.geometry;
    g.validate()?;
    params.validate()?;
    if !params.matches(g) {
        return Err("Kimi KDA backward parameters do not match cached geometry".into());
    }
    let input_len = checked_product(&[g.batch, g.channels, g.height, g.width], "input gradient")?;
    if output_gradient.len() != input_len {
        return Err(format!(
            "Kimi KDA output gradient has {} values, expected {input_len}",
            output_gradient.len()
        ));
    }
    let offsets = KimiOffsets::new(g.channels, g.rank, g.experts, g.hidden)?;
    let rows = g.rows();
    let channels = g.channels;
    let rank = g.rank;
    let experts = g.experts;
    let hidden = g.hidden;
    let p = experts + 1;
    let sqrt_rank = (rank as f32).sqrt();
    let dy = flatten_nchw(output_gradient, g);
    let mut gradients = KimiKdaGradients {
        input: vec![0.0; input_len],
        biases: vec![0.0; offsets.total_biases],
        weights: vec![0.0; offsets.total_weights],
    };

    // Shared/routed MLPs and router. The output's first residual sends dy
    // directly into pre_expert; expert and router paths form the LN2 adjoint.
    let mut d_ln2 = vec![0.0; rows * channels];
    let mut gate_means = vec![0.0; experts];
    for row in 0..rows {
        for e in 0..experts {
            gate_means[e] += cache.gates[row * experts + e];
        }
    }
    for mean in &mut gate_means {
        *mean /= rows as f32;
    }
    for row in 0..rows {
        let x = &cache.ln2[row * channels..(row + 1) * channels];
        let dy_row = &dy[row * channels..(row + 1) * channels];
        let mut raw_gate = vec![0.0; experts];
        for e in 0..experts {
            let routed = &cache.expert_out
                [((e + 1) * rows + row) * channels..((e + 1) * rows + row + 1) * channels];
            raw_gate[e] = dy_row.iter().zip(routed).map(|(a, b)| a * b).sum();
        }

        for expert in 0..p {
            let gate = if expert == 0 {
                1.0
            } else {
                cache.gates[row * experts + expert - 1]
            };
            let hidden_base = (expert * rows + row) * hidden;
            let mut d_activation = vec![0.0; hidden];
            for c in 0..channels {
                let d_out = dy_row[c] * gate;
                gradients.biases[offsets.b2 + expert * channels + c] += d_out;
                for j in 0..hidden {
                    gradients.weights[offsets.w2 + (expert * channels + c) * hidden + j] +=
                        d_out * cache.expert_act[hidden_base + j];
                    d_activation[j] +=
                        d_out * params.weights[offsets.w2 + (expert * channels + c) * hidden + j];
                }
            }
            for j in 0..hidden {
                let d_pre = d_activation[j]
                    * tensor_activation_grad(
                        &g.activation,
                        cache.expert_pre[hidden_base + j],
                        cache.expert_act[hidden_base + j],
                    )?;
                gradients.biases[offsets.b1 + expert * hidden + j] += d_pre;
                for c in 0..channels {
                    gradients.weights[offsets.w1 + (expert * hidden + j) * channels + c] +=
                        d_pre * x[c];
                    d_ln2[row * channels + c] +=
                        d_pre * params.weights[offsets.w1 + (expert * hidden + j) * channels + c];
                }
            }
        }

        let mut weighted = 0.0;
        for e in 0..experts {
            let balance_gradient =
                2.0 * g.load_balance * experts as f32 * gate_means[e] / rows as f32;
            weighted += cache.gates[row * experts + e] * (raw_gate[e] + balance_gradient);
        }
        for e in 0..experts {
            let gate = cache.gates[row * experts + e];
            let logit = cache.router_logits[row * experts + e];
            let rank_e = (0..experts)
                .filter(|&j| {
                    let other = cache.router_logits[row * experts + j];
                    other > logit || (other == logit && j < e)
                })
                .count();
            let d_logit = if gate > 0.0 && (g.top_k == 0 || rank_e < g.top_k) {
                let balance_gradient =
                    2.0 * g.load_balance * experts as f32 * gate_means[e] / rows as f32;
                gate * (raw_gate[e] + balance_gradient - weighted) / g.temperature
            } else {
                0.0
            };
            gradients.biases[offsets.brouter + e] += d_logit;
            for c in 0..channels {
                gradients.weights[offsets.wrouter + e * channels + c] += d_logit * x[c];
                d_ln2[row * channels + c] +=
                    d_logit * params.weights[offsets.wrouter + e * channels + c];
            }
        }
    }

    let (d_ln2_input, d_ln2_gamma, d_ln2_beta) = layer_norm_backward(
        &d_ln2,
        &params.biases[offsets.ln2_gamma..offsets.ln2_gamma + channels],
        &cache.ln2_cache,
    )?;
    gradients.biases[offsets.ln2_gamma..offsets.ln2_gamma + channels].copy_from_slice(&d_ln2_gamma);
    gradients.biases[offsets.ln2_beta..offsets.ln2_beta + channels].copy_from_slice(&d_ln2_beta);

    // The second residual adds the LN2 adjoint to dy before the attention
    // projection; the first residual also sends dy directly to the tokens.
    let mut d_pre_expert = dy.clone();
    for (dst, src) in d_pre_expert.iter_mut().zip(d_ln2_input) {
        *dst += src;
    }
    let mut d_attention = vec![0.0; rows * channels];
    for row in 0..rows {
        for j in 0..channels {
            let d_out = d_pre_expert[row * channels + j];
            gradients.biases[offsets.bo + j] += d_out;
            for i in 0..channels {
                gradients.weights[offsets.wo + j * channels + i] +=
                    d_out * cache.attention[row * channels + i];
                d_attention[row * channels + i] +=
                    d_out * params.weights[offsets.wo + j * channels + i];
            }
        }
    }

    // Reverse the causal delta-rule state sequence. `d_carry` is dL/dh_t
    // arriving from the next token; each transition replaces it with dL/dh_{t-1}.
    let mut d_ln1 = vec![0.0; rows * channels];
    let mut d_v = vec![0.0; rows * channels];
    let mut d_q = vec![0.0; rows * rank];
    let mut d_beta_pre = vec![0.0; rows * rank];
    let mut d_decay_pre = vec![0.0; rows * rank];
    for b in 0..g.batch {
        let mut d_carry = vec![0.0; rank * channels];
        for t in (0..g.tokens()).rev() {
            let row = b * g.tokens() + t;
            let x = &cache.ln1[row * channels..(row + 1) * channels];
            let q = &cache.q[row * rank..(row + 1) * rank];
            let k = &cache.k[row * rank..(row + 1) * rank];
            let beta = &cache.beta[row * rank..(row + 1) * rank];
            let decay = &cache.decay[row * rank..(row + 1) * rank];
            let state_base = row * rank * channels;
            let previous_base = if t > 0 {
                (row - 1) * rank * channels
            } else {
                usize::MAX
            };
            for r in 0..rank {
                let mut gradient = 0.0;
                for c in 0..channels {
                    gradient += d_attention[row * channels + c]
                        * cache.state[state_base + r * channels + c]
                        / sqrt_rank;
                }
                d_q[row * rank + r] = gradient;
            }
            let mut d_beta_logit = vec![0.0; rank];
            let mut d_decay_logit = vec![0.0; rank];
            let mut d_k = vec![0.0; rank];
            for c in 0..channels {
                let mut prediction = 0.0;
                if t > 0 {
                    for r in 0..rank {
                        prediction += k[r] * cache.state[previous_base + r * channels + c];
                    }
                }
                let error = cache.v[row * channels + c] - prediction;
                let mut ds = vec![0.0; rank];
                let mut d_error = 0.0;
                for r in 0..rank {
                    ds[r] = d_attention[row * channels + c] * q[r] / sqrt_rank
                        + d_carry[r * channels + c];
                    let previous = if t > 0 {
                        cache.state[previous_base + r * channels + c]
                    } else {
                        0.0
                    };
                    d_decay_logit[r] += ds[r] * previous;
                    d_beta_logit[r] += ds[r] * k[r] * error;
                    d_k[r] += ds[r] * beta[r] * error;
                    d_error += ds[r] * beta[r] * k[r];
                }
                for r in 0..rank {
                    let previous = if t > 0 {
                        cache.state[previous_base + r * channels + c]
                    } else {
                        0.0
                    };
                    if t > 0 {
                        d_carry[r * channels + c] = ds[r] * decay[r] - d_error * k[r];
                        d_k[r] -= d_error * previous;
                    } else {
                        d_carry[r * channels + c] = 0.0;
                    }
                }
                d_v[row * channels + c] = d_error;
            }

            // Pull dL/d(k_normalized) through the exact piecewise normalization
            // in forward, including the denominator floor and nonfinite branch.
            let norm_squared = cache.key_norm_squared[row];
            if !norm_squared.is_finite() {
                d_k.fill(0.0);
            } else if norm_squared < KEY_NORM_FLOOR {
                for r in 0..rank {
                    d_k[r] *= cache.key_inverse_norm[row];
                }
            } else if norm_squared == KEY_NORM_FLOOR {
                return Err(
                    "Kimi KDA key-normalization derivative is undefined at the floor boundary"
                        .into(),
                );
            } else {
                let dot = k.iter().zip(&d_k).map(|(a, b)| a * b).sum::<f32>();
                for r in 0..rank {
                    d_k[r] = cache.key_inverse_norm[row] * (d_k[r] - k[r] * dot);
                }
            }
            for r in 0..rank {
                d_beta_pre[row * rank + r] = d_beta_logit[r] * beta[r] * (1.0 - beta[r]);
                d_decay_pre[row * rank + r] = d_decay_logit[r] * decay[r] * (1.0 - decay[r]);
                gradients.biases[offsets.bq + r] += d_q[row * rank + r];
                gradients.biases[offsets.bk + r] += d_k[r];
                gradients.biases[offsets.bbeta + r] += d_beta_pre[row * rank + r];
                gradients.biases[offsets.bdecay + r] += d_decay_pre[row * rank + r];
                for c in 0..channels {
                    gradients.weights[offsets.wq + r * channels + c] += d_q[row * rank + r] * x[c];
                    gradients.weights[offsets.wk + r * channels + c] += d_k[r] * x[c];
                    gradients.weights[offsets.wbeta + r * channels + c] +=
                        d_beta_pre[row * rank + r] * x[c];
                    gradients.weights[offsets.wdecay + r * channels + c] +=
                        d_decay_pre[row * rank + r] * x[c];
                    d_ln1[row * channels + c] += d_q[row * rank + r]
                        * params.weights[offsets.wq + r * channels + c]
                        + d_k[r] * params.weights[offsets.wk + r * channels + c]
                        + d_beta_pre[row * rank + r]
                            * params.weights[offsets.wbeta + r * channels + c]
                        + d_decay_pre[row * rank + r]
                            * params.weights[offsets.wdecay + r * channels + c];
                }
            }
            for c in 0..channels {
                gradients.biases[offsets.bv + c] += d_v[row * channels + c];
                for i in 0..channels {
                    gradients.weights[offsets.wv + c * channels + i] +=
                        d_v[row * channels + c] * x[i];
                    d_ln1[row * channels + i] +=
                        d_v[row * channels + c] * params.weights[offsets.wv + c * channels + i];
                }
            }
        }
    }

    let (d_ln1_input, d_ln1_gamma, d_ln1_beta) = layer_norm_backward(
        &d_ln1,
        &params.biases[offsets.ln1_gamma..offsets.ln1_gamma + channels],
        &cache.ln1_cache,
    )?;
    gradients.biases[offsets.ln1_gamma..offsets.ln1_gamma + channels].copy_from_slice(&d_ln1_gamma);
    gradients.biases[offsets.ln1_beta..offsets.ln1_beta + channels].copy_from_slice(&d_ln1_beta);
    let mut d_tokens = dy;
    for (dst, src) in d_tokens.iter_mut().zip(d_ln1_input) {
        *dst += src;
    }
    gradients.input = deflatten_tokens(&d_tokens, g);

    if gradients
        .input
        .iter()
        .chain(&gradients.biases)
        .chain(&gradients.weights)
        .any(|value| !value.is_finite())
    {
        return Err("Kimi KDA produced a nonfinite gradient".into());
    }
    Ok(gradients)
}

#[cfg(test)]
mod tests {
    use super::{
        KEY_NORM_FLOOR, KimiKdaGeometry, KimiKdaParams, kimi_kda_backward, kimi_kda_forward,
    };
    use crate::model::{Activation, TensorShape};

    fn geometry(batch: usize, channels: usize, height: usize, width: usize) -> KimiKdaGeometry {
        KimiKdaGeometry {
            batch,
            channels,
            height,
            width,
            rank: 1,
            experts: 2,
            top_k: 0,
            hidden: 2,
            temperature: 0.8,
            load_balance: 0.03,
            activation: Activation::Gelu,
        }
    }

    fn spec(g: &KimiKdaGeometry) -> crate::model::KimiKdaSpec {
        crate::model::KimiKdaSpec {
            filters: g.channels,
            rank: g.rank,
            experts: g.experts,
            top_k: g.top_k,
            hidden: g.hidden,
            temperature: g.temperature,
            load_balance: g.load_balance,
            activation: g.activation.clone(),
        }
    }

    fn fill_parameters(g: &KimiKdaGeometry, params: &mut KimiKdaParams) {
        for (i, value) in params.weights.iter_mut().enumerate() {
            *value = (i as f32 * 0.173 + 0.2).sin() * 0.18;
        }
        for (i, value) in params.biases.iter_mut().enumerate() {
            *value = (i as f32 * 0.113 - 0.4).cos() * 0.12;
        }
        // Keep the top-k set and key norm away from nondifferentiable boundaries.
        let offsets = super::KimiOffsets::new(g.channels, g.rank, g.experts, g.hidden).unwrap();
        for r in 0..g.rank {
            params.biases[offsets.bk + r] += 0.9;
        }
        assert_eq!(params.weights.len(), offsets.total_weights);
    }

    fn objective(input: &[f32], params: &KimiKdaParams, g: &KimiKdaGeometry, seed: &[f32]) -> f32 {
        let (output, cache) = kimi_kda_forward(input, shape(g), &spec(g), params).unwrap();
        output.iter().zip(seed).map(|(y, dy)| y * dy).sum::<f32>() + cache.load_balance_cost
    }

    fn shape(g: &KimiKdaGeometry) -> TensorShape {
        TensorShape::new(g.width, g.height, g.channels, g.batch)
    }

    #[test]
    fn low_rank_delta_rule_and_two_residuals_match_a_hand_computation() {
        let mut g = geometry(1, 2, 1, 1);
        g.rank = 1;
        g.experts = 1;
        g.top_k = 1;
        g.hidden = 1;
        g.temperature = 1.0;
        g.load_balance = 0.0;
        g.activation = Activation::Linear;
        let mut params = KimiKdaParams::zeroed(2, 1, 1, 1).unwrap();
        let o = super::KimiOffsets::new(2, 1, 1, 1).unwrap();
        params.biases[o.ln1_gamma..o.ln1_gamma + 2].fill(1.0);
        params.biases[o.ln2_gamma..o.ln2_gamma + 2].fill(1.0);
        params.biases[o.bq] = 1.0;
        params.biases[o.bk] = 1.0;
        params.biases[o.bv..o.bv + 2].copy_from_slice(&[2.0, -1.0]);
        params.biases[o.bbeta] = 0.0;
        params.weights[o.wo] = 1.0;
        params.weights[o.wo + 3] = 1.0;
        params.biases[o.bo..o.bo + 2].copy_from_slice(&[0.25, -0.25]);
        params.biases[o.b2..o.b2 + 2].copy_from_slice(&[0.1, 0.2]);
        params.biases[o.b2 + 2..o.b2 + 4].copy_from_slice(&[0.3, -0.1]);
        let input = [1.0, -1.0];
        let (output, cache) = kimi_kda_forward(&input, shape(&g), &spec(&g), &params).unwrap();
        let h = [1.0, -0.5]; // beta=.5, normalized k=1, v=[2,-1], zero initial state
        let pre = [1.0 + h[0] + 0.25, -1.0 + h[1] - 0.25];
        let expected = [pre[0] + 0.1 + 0.3, pre[1] + 0.2 - 0.1];
        assert!((output[0] - expected[0]).abs() < 2.0e-6);
        assert!((output[1] - expected[1]).abs() < 2.0e-6);
        assert_eq!(cache.load_balance_cost, 0.0);
    }

    #[test]
    fn axis_layout_keeps_nchw_raster_scans_independent_per_batch() {
        let mut g = geometry(2, 2, 1, 2);
        g.activation = Activation::Linear;
        let mut params = KimiKdaParams::zeroed(g.channels, g.rank, g.experts, g.hidden).unwrap();
        let o = super::KimiOffsets::new(g.channels, g.rank, g.experts, g.hidden).unwrap();
        params.biases[o.ln1_gamma..o.ln1_gamma + g.channels].fill(1.0);
        params.biases[o.bq] = 1.0;
        params.biases[o.bk] = 1.0;
        params.biases[o.bdecay] = 3.0_f32.ln();
        params.weights[o.wv] = 1.0;
        params.weights[o.wo] = 1.0;

        // NCHW rows become tokens [b0t0,b0t1,b1t0,b1t1], each with two channels.
        let input = [1.0, 1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0];
        let (output, _) = kimi_kda_forward(&input, shape(&g), &spec(&g), &params).unwrap();
        let norm_one = 1.0 / (1.0 + 1.0e-5_f32).sqrt();
        let norm_two = 2.0 / (4.0 + 1.0e-5_f32).sqrt();
        // q=k=1, beta=1/2, decay=3/4: h0=v0/2 and h1=h0/4+v1/2.
        let expected = [
            1.0 + 0.5 * norm_one,
            1.0 + 0.125 * norm_one,
            -1.0,
            1.0,
            2.0 + 0.5 * norm_two,
            -2.0 - 0.375 * norm_two,
            -2.0,
            2.0,
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn backward_matches_finite_differences_for_input_and_every_parameter() {
        let g = geometry(1, 2, 1, 3);
        let mut params = KimiKdaParams::zeroed(g.channels, g.rank, g.experts, g.hidden).unwrap();
        fill_parameters(&g, &mut params);
        let input: Vec<f32> = (0..g.batch * g.channels * g.height * g.width)
            .map(|i| (i as f32 * 0.31 - 0.7).sin() * 0.6)
            .collect();
        let seed: Vec<f32> = (0..input.len())
            .map(|i| (i as f32 * 0.19 + 0.3).cos() * 0.4)
            .collect();
        let (_, cache) = kimi_kda_forward(&input, shape(&g), &spec(&g), &params).unwrap();
        let analytic = kimi_kda_backward(&seed, &params, &cache).unwrap();
        let epsilon = 1.0e-3;
        for i in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            let numeric = (objective(&plus, &params, &g, &seed)
                - objective(&minus, &params, &g, &seed))
                / (2.0 * epsilon);
            assert!(
                (analytic.input[i] - numeric).abs() < 3.0e-3,
                "input[{i}] analytic={} numeric={numeric}",
                analytic.input[i]
            );
        }
        for i in 0..params.biases.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.biases[i] += epsilon;
            minus.biases[i] -= epsilon;
            let numeric = (objective(&input, &plus, &g, &seed)
                - objective(&input, &minus, &g, &seed))
                / (2.0 * epsilon);
            assert!(
                (analytic.biases[i] - numeric).abs() < 4.0e-3,
                "biases[{i}] analytic={} numeric={numeric}",
                analytic.biases[i]
            );
        }
        for i in 0..params.weights.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.weights[i] += epsilon;
            minus.weights[i] -= epsilon;
            let numeric = (objective(&input, &plus, &g, &seed)
                - objective(&input, &minus, &g, &seed))
                / (2.0 * epsilon);
            assert!(
                (analytic.weights[i] - numeric).abs() < 4.0e-3,
                "weights[{i}] analytic={} numeric={numeric}",
                analytic.weights[i]
            );
        }
    }

    #[test]
    fn key_norm_floor_uses_the_linear_floor_derivative() {
        let mut g = geometry(1, 2, 1, 1);
        g.top_k = 0;
        g.load_balance = 0.0;
        let mut params = KimiKdaParams::zeroed(g.channels, g.rank, g.experts, g.hidden).unwrap();
        fill_parameters(&g, &mut params);
        let o = super::KimiOffsets::new(g.channels, g.rank, g.experts, g.hidden).unwrap();
        params.biases[o.bk] = KEY_NORM_FLOOR.sqrt() * 0.25;
        params.weights[o.wk..o.wv].fill(0.0);
        let input = [0.2, -0.3];
        let seed = [0.7, -0.4];
        let (output, cache) = kimi_kda_forward(&input, shape(&g), &spec(&g), &params).unwrap();
        assert_eq!(
            cache.key_norm_squared[0],
            (KEY_NORM_FLOOR.sqrt() * 0.25).powi(2)
        );
        assert!(output.iter().all(|value| value.is_finite()));
        let grad = kimi_kda_backward(&seed, &params, &cache).unwrap();
        assert!(
            grad.weights[o.wk..o.wv]
                .iter()
                .all(|value| value.is_finite())
        );
        let epsilon = 1.0e-8;
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus.biases[o.bk] += epsilon;
        minus.biases[o.bk] -= epsilon;
        let numeric = (objective(&input, &plus, &g, &seed) - objective(&input, &minus, &g, &seed))
            / (2.0 * epsilon);
        assert!(
            (grad.biases[o.bk] - numeric).abs() < 2.0,
            "floor-branch key bias analytic={} numeric={numeric}",
            grad.biases[o.bk]
        );
    }

    #[test]
    fn zero_top_k_is_dense_and_positive_top_k_uses_stable_lower_index_ties() {
        let logits = [1.0, 1.0, 0.0];
        let dense = super::softmax_gates(&logits, 3, 0);
        assert!(dense.iter().all(|gate| *gate > 0.0));
        let top_one = super::softmax_gates(&logits, 3, 1);
        assert_eq!(top_one, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn parameter_blob_is_biases_then_weights_and_checks_exact_length() {
        let params = KimiKdaParams::zeroed(4, 2, 3, 5).unwrap();
        let blob = params.to_blob().unwrap();
        assert_eq!(blob.len(), params.parameter_count());
        let restored = KimiKdaParams::from_blob(&blob, 4, 2, 3, 5).unwrap();
        assert_eq!(restored, params);
        assert!(KimiKdaParams::from_blob(&blob[..blob.len() - 1], 4, 2, 3, 5).is_err());
    }
}
