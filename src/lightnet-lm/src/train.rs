//! CPU training engine: micro-batch loss accumulation, AdamW, and state.

use lightnet_cuda::host_ops::{AdamParams, adam_update};

use crate::optim::{UpdateOptions, UpdateReport, clip_global_norm};
use crate::{
    CpuDecoder, DecoderConfig, DecoderGrads, DecoderParams, Error, LossSummary, Result, RngState,
    TokenBatch,
};

pub const CHECKPOINT_PRECISION: &str = "f32";

/// Full optimizer state.  Tied storage is represented once, by the
/// canonical parameter registry.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineState {
    pub config: DecoderConfig,
    pub parameters: DecoderParams,
    pub first_moment: Vec<Vec<f32>>,
    pub second_moment: Vec<Vec<f32>>,
    pub pending: DecoderGrads,
    pub pending_tokens: u64,
    pub model_rng: RngState,
    pub precision: String,
    /// Number of successful updates so far (Adam bias-correction step).
    pub step: u32,
    /// Tokens consumed by successful updates only.
    pub updated_tokens: u64,
}

pub trait LmEngine {
    fn loss_sum_backward(&mut self, batch: &TokenBatch) -> Result<LossSummary>;
    fn apply_update(&mut self, options: &UpdateOptions) -> Result<UpdateReport>;
    fn export_state(&self) -> Result<EngineState>;
    fn import_state(&mut self, state: &EngineState) -> Result<()>;
}

/// Deterministic f32 CPU engine.
#[derive(Clone, Debug)]
pub struct CpuEngine {
    cfg: DecoderConfig,
    decoder: CpuDecoder,
    first_moment: Vec<Vec<f32>>,
    second_moment: Vec<Vec<f32>>,
    pending: DecoderGrads,
    pending_tokens: u64,
    model_rng: RngState,
    step: u32,
    updated_tokens: u64,
}

impl CpuEngine {
    pub fn new(cfg: DecoderConfig, seed: u64) -> Result<Self> {
        cfg.validate()?;
        let decoder = CpuDecoder::new(cfg.clone(), seed)?;
        let first_moment = zero_like(decoder.parameters());
        let second_moment = zero_like(decoder.parameters());
        let pending = DecoderGrads {
            values: zero_like(decoder.parameters()),
        };
        Ok(Self {
            cfg,
            decoder,
            first_moment,
            second_moment,
            pending,
            pending_tokens: 0,
            model_rng: RngState::new(seed),
            step: 0,
            updated_tokens: 0,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    pub fn parameters(&self) -> &DecoderParams {
        self.decoder.parameters()
    }

    pub fn decoder(&self) -> &CpuDecoder {
        &self.decoder
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn updated_tokens(&self) -> u64 {
        self.updated_tokens
    }

    pub fn pending_tokens(&self) -> u64 {
        self.pending_tokens
    }

    pub fn model_rng(&self) -> RngState {
        self.model_rng
    }

    /// Deterministic inference loss (no dropout, no accumulation).
    pub fn eval_loss_sum(&self, batch: &TokenBatch) -> Result<LossSummary> {
        batch.validate(&self.cfg)?;
        let forward = self.decoder.forward(batch, None)?;
        let (summary, _) = self.decoder.loss(&forward, batch)?;
        Ok(summary)
    }
}

impl LmEngine for CpuEngine {
    fn loss_sum_backward(&mut self, batch: &TokenBatch) -> Result<LossSummary> {
        batch.validate(&self.cfg)?;
        let forward = self.decoder.forward(batch, Some(&mut self.model_rng))?;
        let (summary, seed) = self.decoder.loss(&forward, batch)?;
        if summary.active_tokens == 0 {
            return Ok(summary);
        }
        let grads = self.decoder.backward(&forward, &seed)?;
        if grads.values.len() != self.pending.values.len() {
            return Err(Error::InvalidConfig(
                "backward produced the wrong number of gradients".to_owned(),
            ));
        }
        for (pending, grad) in self.pending.values.iter_mut().zip(grads.values) {
            if pending.len() != grad.len() {
                return Err(Error::InvalidConfig(
                    "backward produced a gradient with the wrong shape".to_owned(),
                ));
            }
            for (slot, value) in pending.iter_mut().zip(grad) {
                *slot += value;
            }
        }
        self.pending_tokens = self
            .pending_tokens
            .checked_add(summary.active_tokens)
            .ok_or_else(|| Error::InvalidConfig("active-token counter overflow".to_owned()))?;
        Ok(summary)
    }

    fn apply_update(&mut self, options: &UpdateOptions) -> Result<UpdateReport> {
        if self.pending_tokens == 0 {
            return Ok(UpdateReport {
                applied: false,
                global_norm: 0.0,
                clip_factor: 1.0,
            });
        }
        validate_update_options(options)?;
        if options.active_tokens != 0 && options.active_tokens != self.pending_tokens {
            return Err(Error::InvalidConfig(format!(
                "update reports {} active tokens but {} were accumulated",
                options.active_tokens, self.pending_tokens
            )));
        }
        let next_step = self.step.checked_add(1).ok_or_else(|| {
            Error::InvalidConfig("optimizer step counter would overflow u32".to_owned())
        })?;
        if options.step != 0 && options.step != next_step {
            return Err(Error::InvalidConfig(format!(
                "update requests step {}, expected {next_step}",
                options.step
            )));
        }
        for values in &self.first_moment {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidConfig(
                    "first moment contains a non-finite value".to_owned(),
                ));
            }
        }
        for values in &self.second_moment {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidConfig(
                    "second moment contains a non-finite value".to_owned(),
                ));
            }
        }
        let tokens = self.pending_tokens;
        let mut updates = DecoderGrads {
            values: self
                .pending
                .values
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .map(|&value| {
                            if value.is_finite() {
                                (f64::from(value) / tokens as f64) as f32
                            } else {
                                f32::NAN
                            }
                        })
                        .collect()
                })
                .collect(),
        };
        let (global_norm, clip_factor) = clip_global_norm(&mut updates, options.clip_norm)?;

        let current = self.decoder.parameters();
        if current.tensors.len() != updates.values.len() {
            return Err(Error::InvalidConfig(
                "update tensor count does not match the parameter registry".to_owned(),
            ));
        }
        let mut parameters = current.clone();
        let mut first_moment = Vec::with_capacity(self.first_moment.len());
        let mut second_moment = Vec::with_capacity(self.second_moment.len());
        for (index, parameter) in parameters.tensors.iter_mut().enumerate() {
            let update = &updates.values[index];
            if update.len() != parameter.values.len() {
                return Err(Error::InvalidConfig(format!(
                    "update for {} has {} values, expected {}",
                    parameter.name,
                    update.len(),
                    parameter.values.len()
                )));
            }
            let params = AdamParams {
                lr: options.lr,
                beta1: options.beta1,
                beta2: options.beta2,
                eps: options.eps,
                decay: if parameter.decay {
                    options.weight_decay
                } else {
                    0.0
                },
                batch: 1,
                step: next_step,
                decoupled_decay: true,
            };
            let (weights, m, v) = adam_update(
                &parameter.values,
                update,
                &self.first_moment[index],
                &self.second_moment[index],
                &params,
            );
            if weights.iter().any(|value| !value.is_finite())
                || m.iter().any(|value| !value.is_finite())
                || v.iter().any(|value| !value.is_finite())
            {
                return Err(Error::InvalidConfig(
                    "optimizer produced a non-finite parameter or moment".to_owned(),
                ));
            }
            parameter.values = weights;
            first_moment.push(m);
            second_moment.push(v);
        }

        self.decoder.set_parameters(parameters)?;
        self.first_moment = first_moment;
        self.second_moment = second_moment;
        self.pending = DecoderGrads {
            values: zero_like(self.decoder.parameters()),
        };
        self.pending_tokens = 0;
        self.step = next_step;
        self.updated_tokens = self
            .updated_tokens
            .checked_add(tokens)
            .ok_or_else(|| Error::InvalidConfig("updated-token counter overflow".to_owned()))?;
        Ok(UpdateReport {
            applied: true,
            global_norm,
            clip_factor,
        })
    }

    fn export_state(&self) -> Result<EngineState> {
        Ok(EngineState {
            config: self.cfg.clone(),
            parameters: self.decoder.parameters().clone(),
            first_moment: self.first_moment.clone(),
            second_moment: self.second_moment.clone(),
            pending: self.pending.clone(),
            pending_tokens: self.pending_tokens,
            model_rng: self.model_rng,
            precision: CHECKPOINT_PRECISION.to_owned(),
            step: self.step,
            updated_tokens: self.updated_tokens,
        })
    }

    fn import_state(&mut self, state: &EngineState) -> Result<()> {
        if state.config != self.cfg {
            return Err(Error::InvalidConfig(
                "engine state belongs to a different decoder config".to_owned(),
            ));
        }
        if state.precision != CHECKPOINT_PRECISION {
            return Err(Error::InvalidConfig(format!(
                "engine precision {:?} is not supported by the CPU engine",
                state.precision
            )));
        }
        validate_engine_arrays(state)?;

        let mut decoder = self.decoder.clone();
        decoder.set_parameters(state.parameters.clone())?;
        self.decoder = decoder;
        self.first_moment = state.first_moment.clone();
        self.second_moment = state.second_moment.clone();
        self.pending = state.pending.clone();
        self.pending_tokens = state.pending_tokens;
        self.model_rng = state.model_rng;
        self.step = state.step;
        self.updated_tokens = state.updated_tokens;
        Ok(())
    }
}

fn validate_engine_arrays(state: &EngineState) -> Result<usize> {
    let count = state.parameters.tensors.len();
    if count == 0 {
        return Err(Error::InvalidConfig(
            "engine state has no parameters".to_owned(),
        ));
    }
    if state.first_moment.len() != count
        || state.second_moment.len() != count
        || state.pending.values.len() != count
    {
        return Err(Error::InvalidConfig(
            "engine state arrays do not match the parameter registry".to_owned(),
        ));
    }
    for (index, parameter) in state.parameters.tensors.iter().enumerate() {
        let len = parameter.values.len();
        if parameter.values.len() != parameter.shape.iter().product::<usize>() {
            return Err(Error::InvalidConfig(format!(
                "parameter {} has {} values for shape {:?}",
                parameter.name, len, parameter.shape
            )));
        }
        if parameter.values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidConfig(format!(
                "parameter {} contains a non-finite value",
                parameter.name
            )));
        }
        if state.first_moment[index].len() != len
            || state.second_moment[index].len() != len
            || state.pending.values[index].len() != len
        {
            return Err(Error::InvalidConfig(format!(
                "engine state tensor {index} has the wrong length"
            )));
        }
        if state.first_moment[index]
            .iter()
            .chain(state.second_moment[index].iter())
            .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidConfig(format!(
                "engine state tensor {index} has a non-finite moment"
            )));
        }
    }
    Ok(count)
}

pub struct Trainer<E: LmEngine> {
    engine: E,
    pending_microbatches: u32,
    pending_tokens: u64,
    last_loss_sum: f64,
}

impl<E: LmEngine> Trainer<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            pending_microbatches: 0,
            pending_tokens: 0,
            last_loss_sum: 0.0,
        }
    }

    /// Restore the trainer counters that accompany an engine's pending grads.
    pub fn with_pending(engine: E, pending_microbatches: u32, pending_tokens: u64) -> Self {
        Self {
            engine,
            pending_microbatches,
            pending_tokens,
            last_loss_sum: 0.0,
        }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    pub fn pending_microbatches(&self) -> u32 {
        self.pending_microbatches
    }

    pub fn pending_tokens(&self) -> u64 {
        self.pending_tokens
    }

    pub fn last_loss_sum(&self) -> f64 {
        self.last_loss_sum
    }

    pub fn microbatch(&mut self, batch: &TokenBatch) -> Result<LossSummary> {
        let summary = self.engine.loss_sum_backward(batch)?;
        self.pending_microbatches = self
            .pending_microbatches
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConfig("microbatch counter overflow".to_owned()))?;
        self.pending_tokens = self
            .pending_tokens
            .checked_add(summary.active_tokens)
            .ok_or_else(|| Error::InvalidConfig("active-token counter overflow".to_owned()))?;
        self.last_loss_sum += summary.sum;
        Ok(summary)
    }

    pub fn finish_update(&mut self, mut options: UpdateOptions) -> Result<UpdateReport> {
        if self.pending_microbatches == 0 {
            return Ok(UpdateReport {
                applied: false,
                global_norm: 0.0,
                clip_factor: 1.0,
            });
        }
        options.active_tokens = self.pending_tokens;
        let report = self.engine.apply_update(&options)?;
        self.pending_microbatches = 0;
        self.pending_tokens = 0;
        self.last_loss_sum = 0.0;
        Ok(report)
    }
}

fn validate_update_options(options: &UpdateOptions) -> Result<()> {
    if !options.lr.is_finite() || options.lr < 0.0 {
        return Err(Error::InvalidConfig(
            "learning rate must be finite and non-negative".to_owned(),
        ));
    }
    if !options.beta1.is_finite() || !(0.0..1.0).contains(&options.beta1) {
        return Err(Error::InvalidConfig("beta1 must be in [0, 1)".to_owned()));
    }
    if !options.beta2.is_finite() || !(0.0..1.0).contains(&options.beta2) {
        return Err(Error::InvalidConfig("beta2 must be in [0, 1)".to_owned()));
    }
    if !options.eps.is_finite() || options.eps <= 0.0 {
        return Err(Error::InvalidConfig(
            "eps must be finite and positive".to_owned(),
        ));
    }
    if !options.weight_decay.is_finite() || options.weight_decay < 0.0 {
        return Err(Error::InvalidConfig(
            "weight decay must be finite and non-negative".to_owned(),
        ));
    }
    if !options.clip_norm.is_finite() || options.clip_norm <= 0.0 {
        return Err(Error::InvalidConfig(
            "clip norm must be finite and positive".to_owned(),
        ));
    }
    Ok(())
}

fn zero_like(parameters: &DecoderParams) -> Vec<Vec<f32>> {
    parameters
        .tensors
        .iter()
        .map(|parameter| vec![0.0f32; parameter.values.len()])
        .collect()
}
