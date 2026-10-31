use std::ops::Range;
use crate::model::{Activation, MambaVisionSpec, TensorShape};

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

    pub(super) fn blocks(&self) -> Vec<MambaParamBlock> {
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

pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("MambaVision {what} size overflow"))
}
