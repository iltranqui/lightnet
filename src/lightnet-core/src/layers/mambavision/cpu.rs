//! CPU reference math for the `Jev_laya` MambaVision block.
//!
//! This implementation follows the source forward contract but computes the
//! exact derivative of the Rust forward, including scan-state and piecewise
//! clamp derivatives. It is also the host reference used by GPU fallback.

use std::ops::Range;

use crate::model::{Activation, MambaVisionSpec, TensorShape};
use crate::vit::{tensor_activation, tensor_activation_grad};

const SCAN_MIN: f32 = -60.0;
const SCAN_MAX: f32 = 20.0;
const MSA_SCORE_CLIP: f32 = 10.0;
const LAYERNORM_EPS: f32 = 1.0e-5;

#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionParams {
    /// Flat Rust parameter payload in the audited C++ `.weights` order.
    pub values: Vec<f32>,
    channels: usize,
    filters: usize,
    state_size: usize,
    conv_size: usize,
    dt_rank: usize,
    ffn_ratio: usize,
}

impl MambaVisionParams {
    pub fn zeroed(
        channels: usize,
        filters: usize,
        state_size: usize,
        conv_size: usize,
        dt_rank: usize,
        ffn_ratio: usize,
    ) -> Result<Self, String> {
        let count =
            MambaOffsets::new(channels, filters, state_size, conv_size, dt_rank, ffn_ratio)?.total;
        Ok(Self {
            values: vec![0.0; count],
            channels,
            filters,
            state_size,
            conv_size,
            dt_rank,
            ffn_ratio,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.values.len()
    }

    pub fn to_blob(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        Ok(self.values.clone())
    }

    pub fn from_blob(
        blob: &[f32],
        channels: usize,
        filters: usize,
        state_size: usize,
        conv_size: usize,
        dt_rank: usize,
        ffn_ratio: usize,
    ) -> Result<Self, String> {
        let expected =
            MambaOffsets::new(channels, filters, state_size, conv_size, dt_rank, ffn_ratio)?.total;
        if blob.len() != expected {
            return Err(format!(
                "MambaVision parameter blob has {} values, expected {expected}",
                blob.len()
            ));
        }
        Ok(Self {
            values: blob.to_vec(),
            channels,
            filters,
            state_size,
            conv_size,
            dt_rank,
            ffn_ratio,
        })
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        let expected = MambaOffsets::new(
            self.channels,
            self.filters,
            self.state_size,
            self.conv_size,
            self.dt_rank,
            self.ffn_ratio,
        )?
        .total;
        if self.values.len() != expected {
            return Err(format!(
                "MambaVision parameter vector has {} values, expected {expected}",
                self.values.len()
            ));
        }
        Ok(())
    }

    pub(super) fn matches(&self, g: &MambaVisionGeometry) -> bool {
        self.channels == g.channels
            && self.filters == g.filters
            && self.state_size == g.state_size
            && self.conv_size == g.conv_size
            && self.dt_rank == g.dt_rank
            && self.ffn_ratio == g.ffn_ratio
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionState {
    /// State layout `[batch, filters/2, state]`, row-major.
    pub values: Vec<f32>,
}

impl MambaVisionState {
    pub fn zeroed(batch: usize, filters: usize, state_size: usize) -> Result<Self, String> {
        if filters < 2 || filters % 2 != 0 || state_size == 0 {
            return Err("MambaVision state requires even filters >= 2 and state >= 1".into());
        }
        let len = checked_product(&[batch, filters / 2, state_size], "state")?;
        Ok(Self {
            values: vec![0.0; len],
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionForward {
    /// Output in NCHW layout.
    pub output: Vec<f32>,
    pub state_out: MambaVisionState,
    cache: MambaVisionCache,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionGradients {
    /// Input gradient in NCHW layout.
    pub input: Vec<f32>,
    /// Mathematical derivative in the same flat order as `MambaVisionParams::values`.
    pub params: Vec<f32>,
    /// Derivative with respect to the incoming `[batch, filters/2, state]` state.
    pub state_in: MambaVisionState,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MambaVisionGeometry {
    pub(super) batch: usize,
    pub(super) channels: usize,
    pub(super) height: usize,
    pub(super) width: usize,
    pub(super) filters: usize,
    pub(super) state_size: usize,
    pub(super) conv_size: usize,
    pub(super) dt_rank: usize,
    pub(super) ffn_ratio: usize,
    pub(super) activation: Activation,
    pub(super) msa_enabled: bool,
    pub(super) msa_block_size: usize,
    pub(super) msa_topk: usize,
    pub(super) msa_heads: usize,
    pub(super) causal: bool,
}

impl MambaVisionGeometry {
    pub(crate) fn from_layer(input: TensorShape, spec: &MambaVisionSpec) -> Result<Self, String> {
        let g = Self {
            batch: input.batch,
            channels: input.channels,
            height: input.height,
            width: input.width,
            filters: spec.filters,
            state_size: spec.state,
            conv_size: spec.conv_size,
            dt_rank: spec.dt_rank,
            ffn_ratio: spec.ffn_ratio,
            activation: spec.activation.clone(),
            msa_enabled: spec.msa_enabled,
            msa_block_size: spec.msa_block_size,
            msa_topk: spec.msa_topk,
            msa_heads: spec.msa_heads,
            causal: spec.causal,
        };
        g.validate()?;
        Ok(g)
    }

    fn validate(&self) -> Result<(), String> {
        if self.batch == 0 || self.channels == 0 || self.height == 0 || self.width == 0 {
            return Err("MambaVision requires a nonempty BxCxHxW input".into());
        }
        if self.filters < 2 || self.filters % 2 != 0 {
            return Err("MambaVision filters must be even and >= 2".into());
        }
        if self.state_size == 0 || self.conv_size == 0 || self.conv_size % 2 == 0 {
            return Err(
                "MambaVision state must be positive and conv_size must be positive odd".into(),
            );
        }
        if self.dt_rank == 0 || self.ffn_ratio == 0 || self.msa_heads == 0 {
            return Err("MambaVision dt_rank, ffn_ratio, and msa_heads must be positive".into());
        }
        if self.filters % self.msa_heads != 0 {
            return Err("MambaVision filters must be divisible by msa_heads".into());
        }
        let tokens = checked_product(&[self.height, self.width], "token count")?;
        let rows = checked_product(&[self.batch, tokens], "token batch")?;
        checked_product(&[self.batch, self.channels, tokens], "input")?;
        checked_product(&[self.batch, self.filters, tokens], "output")?;
        checked_product(&[self.batch, self.half(), tokens], "scan")?;
        checked_product(
            &[self.batch, self.half(), tokens, self.state_size],
            "scan state",
        )?;
        checked_product(&[rows, self.filters, self.ffn_ratio], "FFN workspace")?;
        if self.msa_enabled {
            checked_product(&[self.batch, self.msa_heads, tokens, tokens], "MSA scores")?;
        }
        MambaOffsets::new(
            self.channels,
            self.filters,
            self.state_size,
            self.conv_size,
            self.dt_rank,
            self.ffn_ratio,
        )?;
        Ok(())
    }

    pub(super) fn tokens(&self) -> usize {
        self.height * self.width
    }

    pub(super) fn rows(&self) -> usize {
        self.batch * self.tokens()
    }

    pub(super) fn half(&self) -> usize {
        self.filters / 2
    }

    pub(super) fn hidden(&self) -> usize {
        self.filters * self.ffn_ratio
    }

    pub(super) fn projection_width(&self) -> usize {
        self.dt_rank + 2 * self.state_size
    }
}

pub(crate) fn validate_geometry(input: TensorShape, spec: &MambaVisionSpec) -> Result<(), String> {
    MambaVisionGeometry::from_layer(input, spec).map(|_| ())
}

#[derive(Clone, Debug, PartialEq)]
struct MambaVisionCache {
    geometry: MambaVisionGeometry,
    offsets: MambaOffsets,
    state_in: Vec<f32>,
    tokens: Vec<f32>,
    ln1_xhat: Vec<f32>,
    ln1_out: Vec<f32>,
    ln1_inv_std: Vec<f32>,
    in_proj: Vec<f32>,
    x_conv_pre: Vec<f32>,
    x_conv: Vec<f32>,
    z_conv_pre: Vec<f32>,
    z_conv: Vec<f32>,
    x_proj: Vec<f32>,
    dt_pre: Vec<f32>,
    dt: Vec<f32>,
    scan_state: Vec<f32>,
    scan_out: Vec<f32>,
    mixer_cat: Vec<f32>,
    msa_input: Vec<f32>,
    msa_probabilities: Vec<f32>,
    msa_score_active: Vec<bool>,
    msa_has_undefined_score: bool,
    mixer_out: Vec<f32>,
    pre_res2: Vec<f32>,
    ln2_xhat: Vec<f32>,
    ln2_out: Vec<f32>,
    ln2_inv_std: Vec<f32>,
    ffn_pre: Vec<f32>,
    ffn_act: Vec<f32>,
    scan_clamp_boundary: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MambaParamBlock {
    pub name: &'static str,
    pub range: Range<usize>,
    pub decay: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MambaOffsets {
    pub(super) in_w: Range<usize>,
    pub(super) in_b: Range<usize>,
    pub(super) conv_x_w: Range<usize>,
    pub(super) conv_x_b: Range<usize>,
    pub(super) conv_z_w: Range<usize>,
    pub(super) conv_z_b: Range<usize>,
    pub(super) x_proj_w: Range<usize>,
    pub(super) dt_proj_w: Range<usize>,
    pub(super) dt_proj_b: Range<usize>,
    pub(super) a_log: Range<usize>,
    pub(super) d_skip: Range<usize>,
    pub(super) out_w: Range<usize>,
    pub(super) out_b: Range<usize>,
    pub(super) res_w: Option<Range<usize>>,
    pub(super) ln1_gamma: Range<usize>,
    pub(super) ln1_beta: Range<usize>,
    pub(super) ln2_gamma: Range<usize>,
    pub(super) ln2_beta: Range<usize>,
    pub(super) ffn1_w: Range<usize>,
    pub(super) ffn1_b: Range<usize>,
    pub(super) ffn2_w: Range<usize>,
    pub(super) ffn2_b: Range<usize>,
    pub(super) total: usize,
}

impl MambaOffsets {
    pub(super) fn new(
        channels: usize,
        filters: usize,
        state_size: usize,
        conv_size: usize,
        dt_rank: usize,
        ffn_ratio: usize,
    ) -> Result<Self, String> {
        if channels == 0 || filters < 2 || filters % 2 != 0 {
            return Err("MambaVision parameters require channels > 0 and even filters >= 2".into());
        }
        if state_size == 0 || conv_size == 0 || conv_size % 2 == 0 || dt_rank == 0 || ffn_ratio == 0
        {
            return Err("MambaVision parameter geometry contains a zero/invalid dimension".into());
        }
        let d = filters / 2;
        let p = dt_rank
            .checked_add(
                state_size
                    .checked_mul(2)
                    .ok_or("MambaVision projection overflow")?,
            )
            .ok_or("MambaVision projection overflow")?;
        let hidden = filters
            .checked_mul(ffn_ratio)
            .ok_or("MambaVision FFN size overflow")?;
        let mut cursor = 0usize;
        let mut take = |len: usize| -> Result<Range<usize>, String> {
            let end = cursor
                .checked_add(len)
                .ok_or("MambaVision parameter count overflow")?;
            let range = cursor..end;
            cursor = end;
            Ok(range)
        };
        let in_w = take(checked_product(&[filters, channels], "input projection")?)?;
        let in_b = take(filters)?;
        let conv_x_w = take(checked_product(&[d, conv_size], "x convolution")?)?;
        let conv_x_b = take(d)?;
        let conv_z_w = take(checked_product(&[d, conv_size], "z convolution")?)?;
        let conv_z_b = take(d)?;
        let x_proj_w = take(checked_product(&[p, d], "x projection")?)?;
        let dt_proj_w = take(checked_product(&[d, dt_rank], "dt projection")?)?;
        let dt_proj_b = take(d)?;
        let a_log = take(checked_product(&[d, state_size], "A_log")?)?;
        let d_skip = take(d)?;
        let out_w = take(checked_product(&[filters, filters], "output projection")?)?;
        let out_b = take(filters)?;
        let res_w = if channels != filters {
            Some(take(checked_product(
                &[filters, channels],
                "residual projection",
            )?)?)
        } else {
            None
        };
        let ln1_gamma = take(channels)?;
        let ln1_beta = take(channels)?;
        let ln2_gamma = take(filters)?;
        let ln2_beta = take(filters)?;
        let ffn1_w = take(checked_product(&[hidden, filters], "FFN1")?)?;
        let ffn1_b = take(hidden)?;
        let ffn2_w = take(checked_product(&[filters, hidden], "FFN2")?)?;
        let ffn2_b = take(filters)?;
        Ok(Self {
            in_w,
            in_b,
            conv_x_w,
            conv_x_b,
            conv_z_w,
            conv_z_b,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            a_log,
            d_skip,
            out_w,
            out_b,
            res_w,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            ffn1_w,
            ffn1_b,
            ffn2_w,
            ffn2_b,
            total: cursor,
        })
    }

    fn blocks(&self) -> Vec<MambaParamBlock> {
        let mut blocks = vec![
            MambaParamBlock {
                name: "mambavision_in_proj_weights",
                range: self.in_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_in_proj_bias",
                range: self.in_b.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_conv_x",
                range: self.conv_x_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_conv_x_bias",
                range: self.conv_x_b.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_conv_z",
                range: self.conv_z_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_conv_z_bias",
                range: self.conv_z_b.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_x_proj",
                range: self.x_proj_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_dt_proj",
                range: self.dt_proj_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_dt_bias",
                range: self.dt_proj_b.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_A_log",
                range: self.a_log.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_D",
                range: self.d_skip.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_out_proj",
                range: self.out_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_out_bias",
                range: self.out_b.clone(),
                decay: false,
            },
        ];
        if let Some(range) = &self.res_w {
            blocks.push(MambaParamBlock {
                name: "mambavision_res_proj",
                range: range.clone(),
                decay: true,
            });
        }
        blocks.extend([
            MambaParamBlock {
                name: "mambavision_ln1_gamma",
                range: self.ln1_gamma.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_ln1_beta",
                range: self.ln1_beta.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_ln2_gamma",
                range: self.ln2_gamma.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_ln2_beta",
                range: self.ln2_beta.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_ffn_w1",
                range: self.ffn1_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_ffn_b1",
                range: self.ffn1_b.clone(),
                decay: false,
            },
            MambaParamBlock {
                name: "mambavision_ffn_w2",
                range: self.ffn2_w.clone(),
                decay: true,
            },
            MambaParamBlock {
                name: "mambavision_ffn_b2",
                range: self.ffn2_b.clone(),
                decay: false,
            },
        ]);
        blocks
    }
}

pub(crate) fn mambavision_weight_blocks(
    channels: usize,
    spec: &MambaVisionSpec,
) -> Result<Vec<MambaParamBlock>, String> {
    Ok(MambaOffsets::new(
        channels,
        spec.filters,
        spec.state,
        spec.conv_size,
        spec.dt_rank,
        spec.ffn_ratio,
    )?
    .blocks())
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("MambaVision {what} size overflow"))
}

/// Run one source-shaped MambaVision layer. `shape` and `spec` come from the
/// parsed cfg. `state_in=None` means the zero initial scan state.
pub fn mambavision_forward(
    input: &[f32],
    shape: TensorShape,
    spec: &MambaVisionSpec,
    params: &MambaVisionParams,
    state_in: Option<&MambaVisionState>,
) -> Result<MambaVisionForward, String> {
    let g = MambaVisionGeometry::from_layer(shape, spec)?;
    params.validate()?;
    if !params.matches(&g) {
        return Err("MambaVision parameter geometry does not match layer".into());
    }
    let input_len = checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    if input.len() != input_len {
        return Err(format!(
            "MambaVision input has {} values, expected {input_len}",
            input.len()
        ));
    }
    let state_len = checked_product(&[g.batch, g.half(), g.state_size], "state")?;
    let initial_state = match state_in {
        Some(state) if state.values.len() == state_len => state.values.clone(),
        Some(state) => {
            return Err(format!(
                "MambaVision incoming state has {} values, expected {state_len}",
                state.values.len()
            ));
        }
        None => vec![0.0; state_len],
    };
    let o = MambaOffsets::new(
        g.channels,
        g.filters,
        g.state_size,
        g.conv_size,
        g.dt_rank,
        g.ffn_ratio,
    )?;
    let p = &params.values;
    let b = g.batch;
    let t = g.tokens();
    let m = g.rows();
    let c = g.channels;
    let n = g.filters;
    let d = g.half();
    let s = g.state_size;
    let r = g.dt_rank;
    let width = g.projection_width();
    let hidden = g.hidden();

    let mut cache = MambaVisionCache {
        geometry: g.clone(),
        offsets: o.clone(),
        state_in: initial_state.clone(),
        tokens: vec![0.0; m * c],
        ln1_xhat: vec![0.0; m * c],
        ln1_out: vec![0.0; m * c],
        ln1_inv_std: vec![0.0; m],
        in_proj: vec![0.0; m * n],
        x_conv_pre: vec![0.0; b * d * t],
        x_conv: vec![0.0; b * d * t],
        z_conv_pre: vec![0.0; b * d * t],
        z_conv: vec![0.0; b * d * t],
        x_proj: vec![0.0; m * width],
        dt_pre: vec![0.0; b * d * t],
        dt: vec![0.0; b * d * t],
        scan_state: vec![0.0; b * d * t * s],
        scan_out: vec![0.0; b * d * t],
        mixer_cat: vec![0.0; m * n],
        msa_input: Vec::new(),
        msa_probabilities: Vec::new(),
        msa_score_active: Vec::new(),
        msa_has_undefined_score: false,
        mixer_out: vec![0.0; m * n],
        pre_res2: vec![0.0; m * n],
        ln2_xhat: vec![0.0; m * n],
        ln2_out: vec![0.0; m * n],
        ln2_inv_std: vec![0.0; m],
        ffn_pre: vec![0.0; m * hidden],
        ffn_act: vec![0.0; m * hidden],
        scan_clamp_boundary: false,
    };

    // NCHW -> token-major rows.
    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..c {
                cache.tokens[row * c + channel] = input[(batch * c + channel) * t + token];
            }
        }
    }
    layernorm_forward(
        &cache.tokens,
        c,
        &p[o.ln1_gamma.clone()],
        &p[o.ln1_beta.clone()],
        &mut cache.ln1_xhat,
        &mut cache.ln1_inv_std,
        &mut cache.ln1_out,
    );
    linear_forward(
        &cache.ln1_out,
        m,
        c,
        n,
        &p[o.in_w.clone()],
        &p[o.in_b.clone()],
        &mut cache.in_proj,
    );
    depthwise_conv_forward(
        &cache.in_proj,
        n,
        0,
        b,
        t,
        d,
        g.conv_size,
        g.causal,
        &p[o.conv_x_w.clone()],
        &p[o.conv_x_b.clone()],
        &mut cache.x_conv_pre,
        &mut cache.x_conv,
    );
    depthwise_conv_forward(
        &cache.in_proj,
        n,
        d,
        b,
        t,
        d,
        g.conv_size,
        g.causal,
        &p[o.conv_z_w.clone()],
        &p[o.conv_z_b.clone()],
        &mut cache.z_conv_pre,
        &mut cache.z_conv,
    );
    linear_forward(
        &cache.x_conv,
        m,
        d,
        width,
        &p[o.x_proj_w.clone()],
        &[],
        &mut cache.x_proj,
    );

    // Positive dt, state-selective B/C, and the recurrent diagonal scan.
    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..d {
                let mut pre = p[o.dt_proj_b.start + channel];
                for rank in 0..r {
                    pre += p[o.dt_proj_w.start + channel * r + rank]
                        * cache.x_proj[row * width + rank];
                }
                let index = bdt(batch, channel, token, d, t);
                cache.dt_pre[index] = pre;
                cache.dt[index] = softplus(pre);
            }
        }
    }
    for batch in 0..b {
        for channel in 0..d {
            let mut previous = vec![0.0; s];
            let state_offset = (batch * d + channel) * s;
            previous.copy_from_slice(&initial_state[state_offset..state_offset + s]);
            for token in 0..t {
                let index = bdt(batch, channel, token, d, t);
                let u = cache.x_conv[index];
                let dt = cache.dt[index];
                let row = (batch * t + token) * width;
                let b_start = row + r;
                let c_start = row + r + s;
                let mut y = p[o.d_skip.start + channel] * u;
                for state in 0..s {
                    let raw_log = p[o.a_log.start + channel * s + state];
                    let a = -raw_log.exp();
                    let z = dt * a;
                    if z == SCAN_MIN || z == SCAN_MAX {
                        cache.scan_clamp_boundary = true;
                    }
                    let abar = z.clamp(SCAN_MIN, SCAN_MAX).exp();
                    let h = abar * previous[state] + dt * cache.x_proj[b_start + state] * u;
                    cache.scan_state[state_index(batch, channel, token, state, d, t, s)] = h;
                    previous[state] = h;
                    y += h * cache.x_proj[c_start + state];
                }
                cache.scan_out[index] = y;
            }
            // Initial state is retained in the cache for the scan derivative.
            cache.state_in[state_offset..state_offset + s]
                .copy_from_slice(&initial_state[state_offset..state_offset + s]);
        }
    }

    // Store `[B,D,S]` final scan state by reading the last saved scan row.
    let mut final_state = vec![0.0; state_len];
    for batch in 0..b {
        for channel in 0..d {
            for state in 0..s {
                final_state[(batch * d + channel) * s + state] =
                    cache.scan_state[state_index(batch, channel, t - 1, state, d, t, s)];
            }
        }
    }

    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..d {
                cache.mixer_cat[row * n + channel] =
                    cache.scan_out[bdt(batch, channel, token, d, t)];
                cache.mixer_cat[row * n + d + channel] =
                    cache.z_conv[bdt(batch, channel, token, d, t)];
            }
        }
    }
    if g.msa_enabled {
        cache.msa_input = cache.mixer_cat.clone();
        let (output, probabilities, score_active, undefined) = msa_forward(
            &cache.mixer_cat,
            b,
            t,
            n,
            g.msa_heads,
            g.msa_block_size,
            g.msa_topk,
        )?;
        cache.mixer_cat = output;
        cache.msa_probabilities = probabilities;
        cache.msa_score_active = score_active;
        cache.msa_has_undefined_score = undefined;
    }
    linear_forward(
        &cache.mixer_cat,
        m,
        n,
        n,
        &p[o.out_w.clone()],
        &p[o.out_b.clone()],
        &mut cache.mixer_out,
    );
    if let Some(res_w) = &o.res_w {
        linear_forward(
            &cache.tokens,
            m,
            c,
            n,
            &p[res_w.clone()],
            &[],
            &mut cache.pre_res2,
        );
        for i in 0..m * n {
            cache.pre_res2[i] += cache.mixer_out[i];
        }
    } else {
        for i in 0..m * n {
            cache.pre_res2[i] = cache.tokens[i] + cache.mixer_out[i];
        }
    }
    layernorm_forward(
        &cache.pre_res2,
        n,
        &p[o.ln2_gamma.clone()],
        &p[o.ln2_beta.clone()],
        &mut cache.ln2_xhat,
        &mut cache.ln2_inv_std,
        &mut cache.ln2_out,
    );
    linear_forward(
        &cache.ln2_out,
        m,
        n,
        hidden,
        &p[o.ffn1_w.clone()],
        &p[o.ffn1_b.clone()],
        &mut cache.ffn_pre,
    );
    for i in 0..m * hidden {
        cache.ffn_act[i] = tensor_activation(&g.activation, cache.ffn_pre[i])?;
    }
    let mut ffn_out = vec![0.0; m * n];
    linear_forward(
        &cache.ffn_act,
        m,
        hidden,
        n,
        &p[o.ffn2_w.clone()],
        &p[o.ffn2_b.clone()],
        &mut ffn_out,
    );
    let mut output = vec![0.0; b * n * t];
    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..n {
                output[(batch * n + channel) * t + token] =
                    cache.pre_res2[row * n + channel] + ffn_out[row * n + channel];
            }
        }
    }
    Ok(MambaVisionForward {
        output,
        state_out: MambaVisionState {
            values: final_state,
        },
        cache,
    })
}

/// Backpropagate through `mambavision_forward`. `state_out_gradient` is an
/// optional Rust extension that permits exact gradients across chunked calls.
pub fn mambavision_backward(
    output_gradient: &[f32],
    params: &MambaVisionParams,
    forward: &MambaVisionForward,
    state_out_gradient: Option<&MambaVisionState>,
) -> Result<MambaVisionGradients, String> {
    params.validate()?;
    let cache = &forward.cache;
    let g = &cache.geometry;
    let o = &cache.offsets;
    if !params.matches(g) || output_gradient.len() != forward.output.len() {
        return Err("MambaVision backward input/parameter geometry mismatch".into());
    }
    if cache.scan_clamp_boundary {
        return Err("MambaVision scan derivative is undefined at a clamp boundary".into());
    }
    if cache.msa_has_undefined_score {
        return Err(
            "MambaVision MSA score derivative is undefined for nonfinite or boundary scores".into(),
        );
    }
    let b = g.batch;
    let t = g.tokens();
    let m = g.rows();
    let c = g.channels;
    let n = g.filters;
    let d = g.half();
    let s = g.state_size;
    let r = g.dt_rank;
    let width = g.projection_width();
    let hidden = g.hidden();
    let p = &params.values;
    let state_len = b * d * s;
    let mut dparams = vec![0.0; params.values.len()];

    // Convert NCHW output seed to token-major rows. The residual branch gets
    // the direct output gradient, in addition to the FFN gradient.
    let mut dout = vec![0.0; m * n];
    for batch in 0..b {
        for token in 0..t {
            for channel in 0..n {
                dout[(batch * t + token) * n + channel] =
                    output_gradient[(batch * n + channel) * t + token];
            }
        }
    }
    let mut dpre_res2 = dout.clone();
    let mut dln2_out = vec![0.0; m * n];
    let mut dffn_act = vec![0.0; m * hidden];
    let (ffn2_weight_grad, ffn2_bias_grad) =
        split_two_mut(&mut dparams, o.ffn2_w.clone(), o.ffn2_b.clone());
    linear_backward(
        &cache.ffn_act,
        &dout,
        m,
        hidden,
        n,
        &p[o.ffn2_w.clone()],
        ffn2_weight_grad,
        ffn2_bias_grad,
        &mut dffn_act,
    );
    let mut dffn_pre = vec![0.0; m * hidden];
    for i in 0..m * hidden {
        dffn_pre[i] = dffn_act[i]
            * tensor_activation_grad(&g.activation, cache.ffn_pre[i], cache.ffn_act[i])?;
    }
    let (ffn1_weight_grad, ffn1_bias_grad) =
        split_two_mut(&mut dparams, o.ffn1_w.clone(), o.ffn1_b.clone());
    linear_backward(
        &cache.ln2_out,
        &dffn_pre,
        m,
        n,
        hidden,
        &p[o.ffn1_w.clone()],
        ffn1_weight_grad,
        ffn1_bias_grad,
        &mut dln2_out,
    );
    let mut dln2_xhat = vec![0.0; m * n];
    for row in 0..m {
        for channel in 0..n {
            let index = row * n + channel;
            dparams[o.ln2_gamma.start + channel] += dln2_out[index] * cache.ln2_xhat[index];
            dparams[o.ln2_beta.start + channel] += dln2_out[index];
            dln2_xhat[index] = dln2_out[index] * p[o.ln2_gamma.start + channel];
        }
    }
    let dln2_in = layernorm_backward(&dln2_xhat, &cache.ln2_xhat, &cache.ln2_inv_std, n);
    for i in 0..m * n {
        dpre_res2[i] += dln2_in[i];
    }

    let mut dtokens = vec![0.0; m * c];
    let dmixer_out = dpre_res2.clone();
    if let Some(res_w) = &o.res_w {
        let mut ignored_bias = Vec::new();
        linear_backward(
            &cache.tokens,
            &dpre_res2,
            m,
            c,
            n,
            &p[res_w.clone()],
            &mut dparams[res_w.clone()],
            &mut ignored_bias,
            &mut dtokens,
        );
    } else {
        for i in 0..m * c {
            dtokens[i] += dpre_res2[i];
        }
    }

    let mut dmixer_cat = vec![0.0; m * n];
    let (out_weight_grad, out_bias_grad) =
        split_two_mut(&mut dparams, o.out_w.clone(), o.out_b.clone());
    linear_backward(
        &cache.mixer_cat,
        &dmixer_out,
        m,
        n,
        n,
        &p[o.out_w.clone()],
        out_weight_grad,
        out_bias_grad,
        &mut dmixer_cat,
    );
    if g.msa_enabled {
        dmixer_cat = msa_backward(
            &cache.msa_input,
            &cache.msa_probabilities,
            &cache.msa_score_active,
            &dmixer_cat,
            b,
            t,
            n,
            g.msa_heads,
        )?;
    }

    let mut dscan_out = vec![0.0; b * d * t];
    let mut dz_conv = vec![0.0; b * d * t];
    for batch in 0..b {
        for token in 0..t {
            let row = (batch * t + token) * n;
            for channel in 0..d {
                dscan_out[bdt(batch, channel, token, d, t)] += dmixer_cat[row + channel];
                dz_conv[bdt(batch, channel, token, d, t)] += dmixer_cat[row + d + channel];
            }
        }
    }

    let mut dxconv = vec![0.0; b * d * t];
    let mut dxproj = vec![0.0; m * width];
    let mut ddt = vec![0.0; b * d * t];
    let mut state_in_grad = vec![0.0; state_len];
    let state_out_gradient = match state_out_gradient {
        Some(state) if state.values.len() == state_len => state.values.as_slice(),
        Some(state) => {
            return Err(format!(
                "MambaVision outgoing-state gradient has {} values, expected {state_len}",
                state.values.len()
            ));
        }
        None => &[],
    };
    for batch in 0..b {
        for channel in 0..d {
            let mut dh_next = vec![0.0; s];
            if !state_out_gradient.is_empty() {
                let start = (batch * d + channel) * s;
                dh_next.copy_from_slice(&state_out_gradient[start..start + s]);
            }
            for token in (0..t).rev() {
                let index = bdt(batch, channel, token, d, t);
                let u = cache.x_conv[index];
                let dt = cache.dt[index];
                let row = (batch * t + token) * width;
                let b_start = row + r;
                let c_start = row + r + s;
                let dy = dscan_out[index];
                dparams[o.d_skip.start + channel] += dy * u;
                dxconv[index] += dy * p[o.d_skip.start + channel];
                let mut dh_prev = vec![0.0; s];
                for state in 0..s {
                    let h = cache.scan_state[state_index(batch, channel, token, state, d, t, s)];
                    let hprev = if token == 0 {
                        cache.state_in[(batch * d + channel) * s + state]
                    } else {
                        cache.scan_state[state_index(batch, channel, token - 1, state, d, t, s)]
                    };
                    let a_log = p[o.a_log.start + channel * s + state];
                    let a = -a_log.exp();
                    let raw = dt * a;
                    let abar = raw.clamp(SCAN_MIN, SCAN_MAX).exp();
                    let dh = dy * cache.x_proj[c_start + state] + dh_next[state];
                    let b_value = cache.x_proj[b_start + state];
                    dxproj[c_start + state] += dy * h;
                    let d_abar = dh * hprev;
                    dh_prev[state] += dh * abar;
                    ddt[index] += dh * b_value * u;
                    dxproj[b_start + state] += dh * dt * u;
                    dxconv[index] += dh * dt * b_value;
                    if raw > SCAN_MIN && raw < SCAN_MAX {
                        let d_exp_arg = d_abar * abar;
                        ddt[index] += d_exp_arg * a;
                        dparams[o.a_log.start + channel * s + state] += d_exp_arg * dt * a;
                    }
                }
                dh_next = dh_prev;
            }
            let start = (batch * d + channel) * s;
            state_in_grad[start..start + s].copy_from_slice(&dh_next);
        }
    }

    // dt = softplus(dt_pre), then the low-rank dt projection.
    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..d {
                let index = bdt(batch, channel, token, d, t);
                let dpre = ddt[index] * sigmoid(cache.dt_pre[index]);
                dparams[o.dt_proj_b.start + channel] += dpre;
                for rank in 0..r {
                    dparams[o.dt_proj_w.start + channel * r + rank] +=
                        dpre * cache.x_proj[row * width + rank];
                    dxproj[row * width + rank] += dpre * p[o.dt_proj_w.start + channel * r + rank];
                }
            }
        }
    }

    let mut dxconv_from_proj = vec![0.0; b * d * t];
    linear_backward(
        &cache.x_conv,
        &dxproj,
        m,
        d,
        width,
        &p[o.x_proj_w.clone()],
        &mut dparams[o.x_proj_w.clone()],
        &mut [],
        &mut dxconv_from_proj,
    );
    for i in 0..dxconv.len() {
        dxconv[i] += dxconv_from_proj[i];
    }

    let mut din_proj = vec![0.0; m * n];
    let (conv_x_weight_grad, conv_x_bias_grad) =
        split_two_mut(&mut dparams, o.conv_x_w.clone(), o.conv_x_b.clone());
    depthwise_conv_backward(
        &cache.in_proj,
        n,
        0,
        b,
        t,
        d,
        g.conv_size,
        g.causal,
        &p[o.conv_x_w.clone()],
        &cache.x_conv_pre,
        &dxconv,
        conv_x_weight_grad,
        conv_x_bias_grad,
        &mut din_proj,
    );
    let (conv_z_weight_grad, conv_z_bias_grad) =
        split_two_mut(&mut dparams, o.conv_z_w.clone(), o.conv_z_b.clone());
    depthwise_conv_backward(
        &cache.in_proj,
        n,
        d,
        b,
        t,
        d,
        g.conv_size,
        g.causal,
        &p[o.conv_z_w.clone()],
        &cache.z_conv_pre,
        &dz_conv,
        conv_z_weight_grad,
        conv_z_bias_grad,
        &mut din_proj,
    );
    let mut dln1_xhat = vec![0.0; m * c];
    let (in_weight_grad, in_bias_grad) =
        split_two_mut(&mut dparams, o.in_w.clone(), o.in_b.clone());
    linear_backward(
        &cache.ln1_out,
        &din_proj,
        m,
        c,
        n,
        &p[o.in_w.clone()],
        in_weight_grad,
        in_bias_grad,
        &mut dln1_xhat,
    );
    for row in 0..m {
        for channel in 0..c {
            let index = row * c + channel;
            dparams[o.ln1_gamma.start + channel] += dln1_xhat[index] * cache.ln1_xhat[index];
            dparams[o.ln1_beta.start + channel] += dln1_xhat[index];
            dln1_xhat[index] *= p[o.ln1_gamma.start + channel];
        }
    }
    let dln1_in = layernorm_backward(&dln1_xhat, &cache.ln1_xhat, &cache.ln1_inv_std, c);
    for i in 0..m * c {
        dtokens[i] += dln1_in[i];
    }
    let mut input_grad = vec![0.0; b * c * t];
    for batch in 0..b {
        for token in 0..t {
            let row = batch * t + token;
            for channel in 0..c {
                input_grad[(batch * c + channel) * t + token] = dtokens[row * c + channel];
            }
        }
    }
    Ok(MambaVisionGradients {
        input: input_grad,
        params: dparams,
        state_in: MambaVisionState {
            values: state_in_grad,
        },
    })
}

fn bdt(batch: usize, channel: usize, token: usize, channels: usize, tokens: usize) -> usize {
    (batch * channels + channel) * tokens + token
}

fn state_index(
    batch: usize,
    channel: usize,
    token: usize,
    state: usize,
    channels: usize,
    tokens: usize,
    state_size: usize,
) -> usize {
    (((batch * channels + channel) * tokens + token) * state_size) + state
}

fn split_two_mut<'a>(
    values: &'a mut [f32],
    first: Range<usize>,
    second: Range<usize>,
) -> (&'a mut [f32], &'a mut [f32]) {
    if first.end <= second.start {
        let (before_second, after_second) = values.split_at_mut(second.start);
        (&mut before_second[first], &mut after_second[..second.len()])
    } else {
        debug_assert!(second.end <= first.start);
        let (before_first, after_first) = values.split_at_mut(first.start);
        (&mut after_first[..first.len()], &mut before_first[second])
    }
}

fn linear_forward(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weights: &[f32],
    biases: &[f32],
    output: &mut [f32],
) {
    for row in 0..rows {
        for out in 0..output_dim {
            let mut sum = if biases.is_empty() { 0.0 } else { biases[out] };
            for i in 0..input_dim {
                sum += input[row * input_dim + i] * weights[out * input_dim + i];
            }
            output[row * output_dim + out] = sum;
        }
    }
}

fn linear_backward(
    input: &[f32],
    dout: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weights: &[f32],
    dweights: &mut [f32],
    dbiases: &mut [f32],
    dinput: &mut [f32],
) {
    for row in 0..rows {
        for out in 0..output_dim {
            let grad = dout[row * output_dim + out];
            if !dbiases.is_empty() {
                dbiases[out] += grad;
            }
            for i in 0..input_dim {
                dweights[out * input_dim + i] += grad * input[row * input_dim + i];
                dinput[row * input_dim + i] += grad * weights[out * input_dim + i];
            }
        }
    }
}

fn layernorm_forward(
    input: &[f32],
    channels: usize,
    gamma: &[f32],
    beta: &[f32],
    xhat: &mut [f32],
    inv_std: &mut [f32],
    output: &mut [f32],
) {
    for (row, values) in input.chunks_exact(channels).enumerate() {
        let mean = values.iter().sum::<f32>() / channels as f32;
        let variance = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / channels as f32;
        let inverse = 1.0 / (variance + LAYERNORM_EPS).sqrt();
        inv_std[row] = inverse;
        for channel in 0..channels {
            let normalized = (values[channel] - mean) * inverse;
            xhat[row * channels + channel] = normalized;
            output[row * channels + channel] = normalized * gamma[channel] + beta[channel];
        }
    }
}

fn layernorm_backward(dout: &[f32], xhat: &[f32], inv_std: &[f32], channels: usize) -> Vec<f32> {
    let rows = inv_std.len();
    let mut dx = vec![0.0; rows * channels];
    for row in 0..rows {
        let start = row * channels;
        let mut sum = 0.0;
        let mut dot = 0.0;
        for channel in 0..channels {
            sum += dout[start + channel];
            dot += dout[start + channel] * xhat[start + channel];
        }
        for channel in 0..channels {
            dx[start + channel] = inv_std[row]
                * (dout[start + channel] - (sum + xhat[start + channel] * dot) / channels as f32);
        }
    }
    dx
}

fn depthwise_conv_forward(
    projected: &[f32],
    projected_channels: usize,
    branch_offset: usize,
    batch: usize,
    tokens: usize,
    channels: usize,
    kernel: usize,
    causal: bool,
    weights: &[f32],
    biases: &[f32],
    pre: &mut [f32],
    output: &mut [f32],
) {
    let pad = if causal { kernel - 1 } else { kernel / 2 };
    for b in 0..batch {
        for channel in 0..channels {
            for token in 0..tokens {
                let mut sum = biases[channel];
                for k in 0..kernel {
                    let src = token as isize + k as isize - pad as isize;
                    if src >= 0 && (src as usize) < tokens {
                        sum += weights[channel * kernel + k]
                            * projected[(b * tokens + src as usize) * projected_channels
                                + branch_offset
                                + channel];
                    }
                }
                let index = bdt(b, channel, token, channels, tokens);
                pre[index] = sum;
                output[index] = silu(sum);
            }
        }
    }
}

fn depthwise_conv_backward(
    projected: &[f32],
    projected_channels: usize,
    branch_offset: usize,
    batch: usize,
    tokens: usize,
    channels: usize,
    kernel: usize,
    causal: bool,
    weights: &[f32],
    pre: &[f32],
    dout: &[f32],
    dweights: &mut [f32],
    dbiases: &mut [f32],
    dprojected: &mut [f32],
) {
    let pad = if causal { kernel - 1 } else { kernel / 2 };
    for b in 0..batch {
        for channel in 0..channels {
            for token in 0..tokens {
                let index = bdt(b, channel, token, channels, tokens);
                let dpre = dout[index] * silu_grad(pre[index]);
                dbiases[channel] += dpre;
                for k in 0..kernel {
                    let src = token as isize + k as isize - pad as isize;
                    if src >= 0 && (src as usize) < tokens {
                        let projected_index = (b * tokens + src as usize) * projected_channels
                            + branch_offset
                            + channel;
                        dweights[channel * kernel + k] += dpre * projected[projected_index];
                        dprojected[projected_index] += dpre * weights[channel * kernel + k];
                    }
                }
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

fn msa_forward(
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

fn msa_backward(
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

/// Return the matrix parameter blocks and their source decay grouping.
pub(crate) fn mambavision_param_blocks(
    params: &MambaVisionParams,
) -> Result<Vec<MambaParamBlock>, String> {
    params.validate()?;
    Ok(MambaOffsets::new(
        params.channels,
        params.filters,
        params.state_size,
        params.conv_size,
        params.dt_rank,
        params.ffn_ratio,
    )?
    .blocks())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(_channels: usize, filters: usize, state: usize, conv: usize) -> MambaVisionSpec {
        MambaVisionSpec {
            filters,
            state,
            conv_size: conv,
            dt_rank: 1,
            ffn_ratio: 1,
            activation: Activation::Linear,
            msa_enabled: false,
            msa_block_size: 2,
            msa_topk: 1,
            msa_heads: 1,
            causal: true,
        }
    }

    fn tiny_params(channels: usize, filters: usize, conv: usize) -> MambaVisionParams {
        let mut p = MambaVisionParams::zeroed(channels, filters, 1, conv, 1, 1).unwrap();
        for (i, value) in p.values.iter_mut().enumerate() {
            *value = ((i % 11) as f32 - 5.0) * 0.015;
        }
        let offsets = MambaOffsets::new(channels, filters, 1, conv, 1, 1).unwrap();
        p.values[offsets.ln1_gamma].fill(1.0);
        p.values[offsets.ln2_gamma].fill(1.0);
        p.values[offsets.a_log].fill(0.0);
        p.values[offsets.d_skip].fill(0.5);
        p
    }

    fn loss(output: &[f32], seed: &[f32]) -> f32 {
        output.iter().zip(seed).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn zero_mixer_with_equal_channels_is_residual_identity() {
        let shape = TensorShape::new(1, 1, 2, 1);
        let spec = spec(2, 2, 1, 1);
        let mut params = MambaVisionParams::zeroed(2, 2, 1, 1, 1, 1).unwrap();
        let offsets = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        params.values[offsets.ln1_gamma].fill(1.0);
        params.values[offsets.ln2_gamma].fill(1.0);
        let input = [0.25, -0.5];
        let forward = mambavision_forward(&input, shape, &spec, &params, None).unwrap();
        assert_eq!(forward.output, input);
        assert_eq!(forward.state_out.values, vec![0.0]);
    }

    #[test]
    fn selective_scan_hand_computation_uses_the_audited_recurrence() {
        let shape = TensorShape::new(1, 1, 2, 1);
        let spec = spec(2, 2, 1, 1);
        let mut params = MambaVisionParams::zeroed(2, 2, 1, 1, 1, 1).unwrap();
        let offsets = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        params.values[offsets.in_b.start] = 1.0;
        params.values[offsets.conv_x_w.start] = 1.0;
        params.values[offsets.x_proj_w.clone()].fill(1.0);
        params.values[offsets.out_w.start] = 1.0;
        params.values[offsets.out_w.start + 3] = 1.0;
        params.values[offsets.ln1_gamma.clone()].fill(1.0);
        params.values[offsets.ln2_gamma.clone()].fill(1.0);
        let input = [0.0, 0.0];
        let state = MambaVisionState { values: vec![0.4] };
        let forward = mambavision_forward(&input, shape, &spec, &params, Some(&state)).unwrap();
        let convolved_x = silu(1.0);
        let dt = std::f32::consts::LN_2;
        let expected_state = 0.5 * state.values[0] + dt * convolved_x * convolved_x;
        let scan_output = expected_state * convolved_x;
        let expected_output = [scan_output, 0.0];
        assert!((forward.state_out.values[0] - expected_state).abs() < 1.0e-6);
        for (actual, expected) in forward.output.iter().zip(expected_output) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn selective_scan_adjoint_uses_token_projection_coefficients() {
        let shape = TensorShape::new(2, 1, 2, 1);
        let spec = spec(2, 2, 1, 1);
        let mut params = MambaVisionParams::zeroed(2, 2, 1, 1, 1, 1).unwrap();
        let o = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        params.values[o.in_b.start] = 1.0;
        params.values[o.conv_x_w.start] = 1.0;
        params.values[o.x_proj_w.clone()].fill(1.0);
        params.values[o.out_w.start] = 1.0;
        params.values[o.ln1_gamma.clone()].fill(1.0);
        params.values[o.ln2_gamma.clone()].fill(1.0);
        let input = [0.0; 4];
        let state = MambaVisionState { values: vec![0.4] };
        let seed = [0.7, -0.2, 0.3, 0.1];
        let forward = mambavision_forward(&input, shape, &spec, &params, Some(&state)).unwrap();
        let analytic = mambavision_backward(&seed, &params, &forward, None).unwrap();
        let eps = 1.0e-3;
        for index in o.x_proj_w.clone().chain(o.dt_proj_b.clone()).chain(o.a_log.clone()) {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.values[index] += eps;
            minus.values[index] -= eps;
            let fp = mambavision_forward(&input, shape, &spec, &plus, Some(&state)).unwrap();
            let fm = mambavision_forward(&input, shape, &spec, &minus, Some(&state)).unwrap();
            let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
            assert!((fd - analytic.params[index]).abs() < 1.0e-4,
                "scan parameter[{index}] fd={fd} analytic={}", analytic.params[index]);
        }
        let plus = MambaVisionState { values: vec![state.values[0] + eps] };
        let minus = MambaVisionState { values: vec![state.values[0] - eps] };
        let fp = mambavision_forward(&input, shape, &spec, &params, Some(&plus)).unwrap();
        let fm = mambavision_forward(&input, shape, &spec, &params, Some(&minus)).unwrap();
        let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
        assert!((fd - analytic.state_in.values[0]).abs() < 1.0e-4);
    }

    #[test]
    fn axis_layout_maps_nchw_rows_and_scan_state_per_batch() {
        let shape = TensorShape::new(2, 1, 2, 2);
        let spec = spec(2, 2, 1, 1);
        let mut params = MambaVisionParams::zeroed(2, 2, 1, 1, 1, 1).unwrap();
        let o = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        params.values[o.ln1_gamma.clone()].fill(1.0);
        params.values[o.ln2_gamma.clone()].fill(1.0);
        // Route input channel 0 through the scan x branch, then x_conv into C.
        params.values[o.in_w.start] = 1.0;
        params.values[o.conv_x_w.start] = 1.0;
        params.values[o.x_proj_w.start + 2] = 1.0;
        params.values[o.out_w.start] = 1.0;

        // NCHW: each token has opposite channel sentinels, with distinct values
        // across both spatial positions and batches.
        let input = [1.0, 2.0, -1.0, -2.0, -3.0, 4.0, 3.0, -4.0];
        let state = MambaVisionState {
            values: vec![0.4, -0.2],
        };
        let forward = mambavision_forward(&input, shape, &spec, &params, Some(&state)).unwrap();
        let scan_inputs =
            [1.0_f32, 2.0, -3.0, 4.0].map(|x| silu(x / (x * x + LAYERNORM_EPS).sqrt()));
        let expected_output = [
            1.0 + 0.2 * scan_inputs[0],
            2.0 + 0.1 * scan_inputs[1],
            -1.0,
            -2.0,
            -3.0 - 0.1 * scan_inputs[2],
            4.0 - 0.05 * scan_inputs[3],
            3.0,
            -4.0,
        ];
        let expected_state = [0.1, -0.05];
        for (actual, expected) in forward.output.iter().zip(expected_output) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
        assert_eq!(forward.state_out.values, expected_state);
    }

    #[test]
    fn incoming_state_gradient_matches_finite_difference() {
        let shape = TensorShape::new(2, 1, 2, 1);
        let spec = spec(2, 2, 1, 1);
        let params = tiny_params(2, 2, 1);
        let input = [0.2, -0.3, 0.5, 0.1];
        let state = MambaVisionState { values: vec![0.17] };
        let seed = [0.3, -0.2, 0.5, 0.4];
        let forward = mambavision_forward(&input, shape, &spec, &params, Some(&state)).unwrap();
        let state_seed = MambaVisionState { values: vec![0.23] };
        let analytic = mambavision_backward(&seed, &params, &forward, Some(&state_seed)).unwrap();
        let objective = |result: &MambaVisionForward| {
            loss(&result.output, &seed) + result.state_out.values[0] * state_seed.values[0]
        };
        let eps = 1.0e-3;
        let plus = MambaVisionState {
            values: vec![state.values[0] + eps],
        };
        let minus = MambaVisionState {
            values: vec![state.values[0] - eps],
        };
        let fp = mambavision_forward(&input, shape, &spec, &params, Some(&plus)).unwrap();
        let fm = mambavision_forward(&input, shape, &spec, &params, Some(&minus)).unwrap();
        let fd = (objective(&fp) - objective(&fm)) / (2.0 * eps);
        assert!(
            (fd - analytic.state_in.values[0]).abs() < 2.0e-3,
            "fd={fd} analytic={}",
            analytic.state_in.values[0]
        );
    }

    #[test]
    fn input_and_every_parameter_block_match_finite_difference() {
        let shape = TensorShape::new(2, 1, 2, 1);
        let mut spec = spec(2, 2, 1, 1);
        spec.msa_enabled = true;
        spec.msa_block_size = 1;
        spec.msa_topk = 2;
        spec.msa_heads = 2;
        spec.activation = Activation::Gelu;
        let mut params = tiny_params(2, 2, 1);
        // Keep all MSA scores inside the clipping interval and away from ties.
        let offsets = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        for value in &mut params.values[offsets.in_w.clone()] {
            *value *= 0.3;
        }
        let input = [0.21, -0.31, 0.47, 0.16];
        let seed = [0.3, -0.2, 0.5, 0.4];
        let forward = mambavision_forward(&input, shape, &spec, &params, None).unwrap();
        let analytic = mambavision_backward(&seed, &params, &forward, None).unwrap();
        let eps = 2.0e-3;
        for i in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[i] += eps;
            minus[i] -= eps;
            let fp = mambavision_forward(&plus, shape, &spec, &params, None).unwrap();
            let fm = mambavision_forward(&minus, shape, &spec, &params, None).unwrap();
            let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
            assert!(
                (fd - analytic.input[i]).abs() < 1.5e-2,
                "input[{i}] fd={fd} analytic={}",
                analytic.input[i]
            );
        }
        for block in offsets.blocks() {
            if block.range.is_empty() {
                continue;
            }
            let i = block.range.start;
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.values[i] += eps;
            minus.values[i] -= eps;
            let fp = mambavision_forward(&input, shape, &spec, &plus, None).unwrap();
            let fm = mambavision_forward(&input, shape, &spec, &minus, None).unwrap();
            let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
            assert!(
                (fd - analytic.params[i]).abs() < 2.0e-2,
                "{}[{i}] fd={fd} analytic={}",
                block.name,
                analytic.params[i]
            );
        }
    }

    #[test]
    fn conv_size_one_streaming_matches_full_sequence() {
        let spec = spec(2, 2, 1, 1);
        let params = tiny_params(2, 2, 1);
        let full_shape = TensorShape::new(4, 1, 2, 1);
        let input = [0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, -0.2];
        let full = mambavision_forward(&input, full_shape, &spec, &params, None).unwrap();
        let mut state = MambaVisionState::zeroed(1, 2, 1).unwrap();
        let mut streamed = vec![0.0; 8];
        for token in 0..4 {
            let slice = [input[token], input[4 + token]];
            let shape = TensorShape::new(1, 1, 2, 1);
            let step = mambavision_forward(&slice, shape, &spec, &params, Some(&state)).unwrap();
            streamed[token] = step.output[0];
            streamed[4 + token] = step.output[1];
            state = step.state_out;
        }
        for (a, b) in streamed.iter().zip(full.output) {
            assert!((a - b).abs() < 2.0e-5, "streamed={a} full={b}");
        }
    }

    #[test]
    fn centered_and_causal_depthwise_convolution_match_finite_differences() {
        let shape = TensorShape::new(3, 1, 2, 1);
        let params = tiny_params(2, 2, 3);
        let input = [0.21, -0.31, 0.47, 0.16, -0.22, 0.38];
        let seed = [0.3, -0.2, 0.5, 0.4, -0.1, 0.25];
        let eps = 1.0e-3;
        for causal in [false, true] {
            let mut spec = spec(2, 2, 1, 3);
            spec.causal = causal;
            let forward = mambavision_forward(&input, shape, &spec, &params, None).unwrap();
            let analytic = mambavision_backward(&seed, &params, &forward, None).unwrap();
            for i in [0usize, 3, 5] {
                let mut plus = input;
                let mut minus = input;
                plus[i] += eps;
                minus[i] -= eps;
                let fp = mambavision_forward(&plus, shape, &spec, &params, None).unwrap();
                let fm = mambavision_forward(&minus, shape, &spec, &params, None).unwrap();
                let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
                assert!(
                    (fd - analytic.input[i]).abs() < 2.0e-2,
                    "causal={causal} input[{i}] fd={fd} analytic={}",
                    analytic.input[i]
                );
            }
            let offset = MambaOffsets::new(2, 2, 1, 3, 1, 1).unwrap();
            let i = offset.conv_x_w.start + 1;
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.values[i] += eps;
            minus.values[i] -= eps;
            let fp = mambavision_forward(&input, shape, &spec, &plus, None).unwrap();
            let fm = mambavision_forward(&input, shape, &spec, &minus, None).unwrap();
            let fd = (loss(&fp.output, &seed) - loss(&fm.output, &seed)) / (2.0 * eps);
            assert!(
                (fd - analytic.params[i]).abs() < 2.0e-2,
                "causal={causal} conv_x[{i}] fd={fd} analytic={}",
                analytic.params[i]
            );
        }
    }

    #[test]
    fn scan_clamp_masks_a_log_derivative_outside_the_active_range() {
        let shape = TensorShape::new(2, 1, 2, 1);
        let spec = spec(2, 2, 1, 1);
        let mut params = tiny_params(2, 2, 1);
        let offsets = MambaOffsets::new(2, 2, 1, 1, 1, 1).unwrap();
        params.values[offsets.a_log.clone()].fill(7.0);
        let input = [0.21, -0.31, 0.47, 0.16];
        let state = MambaVisionState { values: vec![0.19] };
        let seed = [0.3, -0.2, 0.5, 0.4];
        let forward = mambavision_forward(&input, shape, &spec, &params, Some(&state)).unwrap();
        let analytic = mambavision_backward(&seed, &params, &forward, None).unwrap();
        assert!(
            offsets
                .a_log
                .clone()
                .all(|index| analytic.params[index] == 0.0)
        );
        let i = offsets.a_log.start;
        let eps = 1.0e-3;
        let mut plus = params.clone();
        let mut minus = params;
        plus.values[i] += eps;
        minus.values[i] -= eps;
        let fp = mambavision_forward(&input, shape, &spec, &plus, Some(&state)).unwrap();
        let fm = mambavision_forward(&input, shape, &spec, &minus, Some(&state)).unwrap();
        assert_eq!(fp.output, fm.output);
    }

    #[test]
    fn msa_score_clip_has_zero_gradient_outside_the_active_range() {
        let batch = 1;
        let tokens = 2;
        let channels = 2;
        let query = [100.0, 0.0, 100.0, 0.0];
        let (output, probabilities, score_active, undefined) =
            msa_forward(&query, batch, tokens, channels, 1, 1, 2).unwrap();
        assert!(!undefined);
        let seed = [0.5, -0.25, 0.75, 0.125];
        let gradient = msa_backward(
            &query,
            &probabilities,
            &score_active,
            &seed,
            batch,
            tokens,
            channels,
            1,
        )
        .unwrap();
        assert_eq!(gradient, [0.625, -0.0625, 0.625, -0.0625]);
        assert!(output.iter().all(|value| value.is_finite()));

        let boundary_query = [2.0, 5.0];
        let (_, _, _, boundary_undefined) =
            msa_forward(&boundary_query, batch, tokens, 1, 1, 1, 2).unwrap();
        assert!(boundary_undefined);
    }

    #[test]
    fn source_weight_order_and_parameter_count_are_stable() {
        let spec = spec(2, 4, 2, 3);
        let blocks = mambavision_weight_blocks(2, &spec).unwrap();
        assert_eq!(blocks.first().unwrap().name, "mambavision_in_proj_weights");
        assert_eq!(blocks[1].name, "mambavision_in_proj_bias");
        assert_eq!(blocks.last().unwrap().name, "mambavision_ffn_b2");
        let total: usize = blocks.iter().map(|block| block.range.len()).sum();
        assert_eq!(
            total,
            MambaVisionParams::zeroed(2, 4, 2, 3, 1, 1)
                .unwrap()
                .parameter_count()
        );
    }
}
