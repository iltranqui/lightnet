//! Integer token batches, loss accounting, and the deterministic RNG.
//!
//! Row indexing is `b * time + t`; activation indexing is
//! `(b * time + t) * hidden + d`.

use crate::{DecoderConfig, Error, Result};

/// A batch of integer token rows.
///
/// `targets[r] == -100` marks an ignored row (padding).  Every other target is
/// a vocabulary index.  `positions` drive RoPE; `segments` isolate packed
/// documents.  `valid[r]` is false for padding rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenBatch {
    pub batch: usize,
    pub time: usize,
    pub input_ids: Vec<u32>,
    pub targets: Vec<i32>,
    pub positions: Vec<u32>,
    pub segments: Vec<u32>,
    pub valid: Vec<bool>,
}

impl TokenBatch {
    /// Number of rows, `batch * time`.
    pub fn rows(&self) -> usize {
        self.batch * self.time
    }

    /// Validate lengths, vocabulary bounds, positions, and context length.
    pub fn validate(&self, cfg: &DecoderConfig) -> Result<()> {
        if self.batch == 0 {
            return Err(Error::InvalidConfig("token batch has zero rows".to_owned()));
        }
        if self.time == 0 {
            return Err(Error::InvalidConfig(
                "token batch has zero sequence length".to_owned(),
            ));
        }
        if self.time > cfg.context {
            return Err(Error::InvalidConfig(format!(
                "token batch time ({}) exceeds context ({})",
                self.time, cfg.context
            )));
        }
        let rows = self.batch.checked_mul(self.time).ok_or_else(|| {
            Error::InvalidConfig("token batch row count overflows usize".to_owned())
        })?;
        for (name, len) in [
            ("input_ids", self.input_ids.len()),
            ("targets", self.targets.len()),
            ("positions", self.positions.len()),
            ("segments", self.segments.len()),
            ("valid", self.valid.len()),
        ] {
            if len != rows {
                return Err(Error::InvalidConfig(format!(
                    "{name} has length {len}, expected {rows}"
                )));
            }
        }
        for (row, &id) in self.input_ids.iter().enumerate() {
            if id as usize >= cfg.vocab_size {
                return Err(Error::InvalidConfig(format!(
                    "input id {id} at row {row} is outside vocabulary ({})",
                    cfg.vocab_size
                )));
            }
        }
        for (row, &target) in self.targets.iter().enumerate() {
            if target == -100 {
                continue;
            }
            if target < 0 || target as usize >= cfg.vocab_size {
                return Err(Error::InvalidConfig(format!(
                    "target {target} at row {row} is outside vocabulary ({})",
                    cfg.vocab_size
                )));
            }
        }
        for (row, &position) in self.positions.iter().enumerate() {
            if position as usize >= cfg.context {
                return Err(Error::InvalidConfig(format!(
                    "position {position} at row {row} is outside context ({})",
                    cfg.context
                )));
            }
        }
        Ok(())
    }
}

/// Summed loss and the number of targets that contributed to it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LossSummary {
    pub sum: f64,
    pub active_tokens: u64,
}

/// SplitMix64 state.  The seed is the initial state; each draw first adds the
/// golden-gamma increment `0x9E3779B97F4A7C15`, then applies the standard
/// SplitMix64 finalizer and returns the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngState {
    pub state: u64,
}

impl RngState {
    pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
