//! Token-count schedule, gradient clipping, and update contracts.

use crate::{DecoderGrads, Error, Result};

/// Warmup plus one cosine decay over token counts, floored at `min_lr`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenSchedule {
    pub warmup_tokens: u64,
    pub total_tokens: u64,
    pub peak_lr: f64,
    pub min_lr: f64,
}

impl TokenSchedule {
    /// Learning rate at `updated_tokens`.
    ///
    /// Warmup is linear and starts at zero (unless warmup is zero, in which
    /// case the first token is already at the peak).  After warmup a single
    /// half-cosine reaches `min_lr` exactly at `total_tokens` and stays there.
    pub fn lr(&self, updated_tokens: u64) -> Result<f32> {
        if self.total_tokens <= self.warmup_tokens {
            return Err(Error::InvalidConfig(
                "schedule total_tokens must exceed warmup_tokens".to_owned(),
            ));
        }
        if !self.peak_lr.is_finite()
            || !self.min_lr.is_finite()
            || self.peak_lr < 0.0
            || self.min_lr < 0.0
            || self.min_lr > self.peak_lr
        {
            return Err(Error::InvalidConfig(
                "schedule learning rates must be finite and ordered 0 <= min_lr <= peak_lr"
                    .to_owned(),
            ));
        }
        let value = if updated_tokens >= self.total_tokens {
            self.min_lr
        } else if updated_tokens < self.warmup_tokens {
            self.peak_lr * updated_tokens as f64 / self.warmup_tokens as f64
        } else {
            let progress = (updated_tokens - self.warmup_tokens) as f64
                / (self.total_tokens - self.warmup_tokens) as f64;
            let progress = progress.clamp(0.0, 1.0);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cosine
        };
        Ok(value as f32)
    }
}

/// One optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdateOptions {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub clip_norm: f32,
    pub active_tokens: u64,
    /// Zero means "next step"; a nonzero value must match the next step.
    pub step: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpdateReport {
    pub applied: bool,
    pub global_norm: f64,
    pub clip_factor: f32,
}

/// Clip every unique parameter gradient by one global L2 norm.
///
/// Returns the pre-clip norm and the applied scale (`1` when no clipping was
/// needed).  Non-finite gradients and a non-positive cap are errors.
pub fn clip_global_norm(grads: &mut DecoderGrads, max_norm: f32) -> Result<(f64, f32)> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(Error::InvalidConfig(
            "clip norm must be finite and positive".to_owned(),
        ));
    }
    let mut sum = 0.0f64;
    for values in &grads.values {
        for &value in values {
            if !value.is_finite() {
                return Err(Error::InvalidConfig(
                    "gradient contains a non-finite value".to_owned(),
                ));
            }
            sum += f64::from(value) * f64::from(value);
        }
    }
    let norm = sum.sqrt();
    let factor = if norm > f64::from(max_norm) {
        (f64::from(max_norm) / norm) as f32
    } else {
        1.0
    };
    if factor != 1.0 {
        for values in &mut grads.values {
            for value in values {
                *value *= factor;
            }
        }
    }
    Ok((norm, factor))
}
