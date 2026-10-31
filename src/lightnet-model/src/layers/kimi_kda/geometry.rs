use std::ops::Range;

use crate::model::{Activation, TensorShape};

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct KimiKdaGeometry {
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
    #[doc(hidden)]
    pub fn validate(&self) -> Result<(), String> {
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

    #[doc(hidden)]
    pub fn from_layer(
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

    #[doc(hidden)]
    pub fn tokens(&self) -> usize {
        self.height * self.width
    }

    #[doc(hidden)]
    pub fn rows(&self) -> usize {
        self.batch * self.tokens()
    }
}

#[doc(hidden)]
pub fn validate_geometry(
    input: TensorShape,
    spec: &crate::model::KimiKdaSpec,
) -> Result<(), String> {
    KimiKdaGeometry::from_layer(input, spec).map(|_| ())
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct KimiKdaWeightBlock {
    pub name: &'static str,
    pub range: Range<usize>,
}

#[doc(hidden)]
pub fn kimi_kda_parameter_counts(
    channels: usize,
    rank: usize,
    experts: usize,
    hidden: usize,
) -> Result<(usize, usize), String> {
    let offsets = KimiOffsets::new(channels, rank, experts, hidden)?;
    Ok((offsets.total_biases, offsets.total_weights))
}
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct KimiOffsets {
    #[doc(hidden)]
    pub wq: usize,
    #[doc(hidden)]
    pub wk: usize,
    #[doc(hidden)]
    pub wv: usize,
    #[doc(hidden)]
    pub wo: usize,
    #[doc(hidden)]
    pub wbeta: usize,
    #[doc(hidden)]
    pub wdecay: usize,
    #[doc(hidden)]
    pub wrouter: usize,
    #[doc(hidden)]
    pub w1: usize,
    #[doc(hidden)]
    pub w2: usize,
    #[doc(hidden)]
    pub total_weights: usize,
    #[doc(hidden)]
    pub ln1_gamma: usize,
    #[doc(hidden)]
    pub ln1_beta: usize,
    #[doc(hidden)]
    pub ln2_gamma: usize,
    #[doc(hidden)]
    pub ln2_beta: usize,
    #[doc(hidden)]
    pub bq: usize,
    #[doc(hidden)]
    pub bk: usize,
    #[doc(hidden)]
    pub bv: usize,
    #[doc(hidden)]
    pub bbeta: usize,
    #[doc(hidden)]
    pub bdecay: usize,
    #[doc(hidden)]
    pub bo: usize,
    #[doc(hidden)]
    pub brouter: usize,
    #[doc(hidden)]
    pub b1: usize,
    #[doc(hidden)]
    pub b2: usize,
    #[doc(hidden)]
    pub total_biases: usize,
}

impl KimiOffsets {
    #[doc(hidden)]
    pub fn new(channels: usize, rank: usize, experts: usize, hidden: usize) -> Result<Self, String> {
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

#[doc(hidden)]
pub fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("Kimi KDA {what} size overflow"))
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("Kimi KDA {what} overflow"))
}

