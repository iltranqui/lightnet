//! Decoder configuration and a minimal `.cfg` reader.
//!
//! `lightnet-lm` must not depend on `lightnet-core`, so this module carries the
//! small subset of the Darknet `.cfg` syntax the decoder needs: named sections
//! followed by `key=value` pairs, `#`/`;` comments, case-insensitive names, and
//! quoted values are *not* interpreted.  Unknown sections and keys are errors;
//! nothing is silently substituted.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// One parsed `[name]` section with its `key=value` pairs.
///
/// Keys are lowercased; values keep their original text (trimmed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawSection {
    pub name: String,
    pub line: usize,
    pub values: BTreeMap<String, String>,
}

impl RawSection {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn required(&self, key: &str) -> Result<&str> {
        self.get(key).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing key \"{key}\" in [{}] at line {}",
                self.name, self.line
            ))
        })
    }

    fn required_usize(&self, key: &str) -> Result<usize> {
        let value = self.required(key)?;
        value.trim().parse::<usize>().map_err(|_| {
            Error::Parse(format!(
                "expected an unsigned integer for key \"{key}\" in [{}] at line {}",
                self.name, self.line
            ))
        })
    }

    fn required_f32(&self, key: &str) -> Result<f32> {
        let value = self.required(key)?;
        value.trim().parse::<f32>().map_err(|_| {
            Error::Parse(format!(
                "expected a float for key \"{key}\" in [{}] at line {}",
                self.name, self.line
            ))
        })
    }

    fn required_flag(&self, key: &str) -> Result<bool> {
        let value = self.required(key)?;
        match value.trim() {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(Error::Parse(format!(
                "expected 0 or 1 for key \"{key}\" in [{}] at line {}",
                self.name, self.line
            ))),
        }
    }
}

/// A parsed configuration file: an ordered list of named sections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigFile {
    pub path: PathBuf,
    pub sections: Vec<RawSection>,
}

impl ConfigFile {
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        Self::parse_str(path, &text)
    }

    pub fn parse_str(path: impl AsRef<Path>, text: &str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut sections = Vec::new();
        let mut current: Option<RawSection> = None;

        for (idx, raw_line) in text.lines().enumerate() {
            let line_number = idx + 1;
            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with('[') {
                let name = line
                    .strip_prefix('[')
                    .and_then(|rest| rest.strip_suffix(']'))
                    .ok_or_else(|| {
                        Error::Parse(format!("unterminated section header at line {line_number}"))
                    })?
                    .trim();
                if name.is_empty() {
                    return Err(Error::Parse(format!(
                        "empty section name at line {line_number}"
                    )));
                }
                if let Some(section) = current.take() {
                    sections.push(section);
                }
                current = Some(RawSection {
                    name: name.to_ascii_lowercase(),
                    line: line_number,
                    values: BTreeMap::new(),
                });
                continue;
            }

            let (key, value) = line.split_once('=').ok_or_else(|| {
                Error::Parse(format!(
                    "expected \"key=value\" outside a section at line {line_number}"
                ))
            })?;
            let key = key.trim().to_ascii_lowercase();
            if key.is_empty() {
                return Err(Error::Parse(format!("empty key at line {line_number}")));
            }
            let section = current.as_mut().ok_or_else(|| {
                Error::Parse(format!(
                    "key \"{key}\" appears before any section at line {line_number}"
                ))
            })?;
            if section
                .values
                .insert(key.clone(), value.trim().to_owned())
                .is_some()
            {
                return Err(Error::Parse(format!(
                    "duplicate key \"{key}\" in [{}] at line {line_number}",
                    section.name
                )));
            }
        }

        if let Some(section) = current {
            sections.push(section);
        }
        Ok(Self { path, sections })
    }

    pub fn section(&self, name: &str) -> Option<&RawSection> {
        let name = name.to_ascii_lowercase();
        self.sections.iter().find(|section| section.name == name)
    }
}

fn strip_comment(line: &str) -> &str {
    let line = line.split_once('#').map_or(line, |(value, _)| value);
    line.split_once(';').map_or(line, |(value, _)| value)
}

/// Accepted keys of a `[decoder]` section.  Any other key is an error.
const DECODER_KEYS: [&str; 11] = [
    "vocab_size",
    "context",
    "hidden",
    "intermediate",
    "layers",
    "q_heads",
    "kv_heads",
    "rope_theta",
    "rms_eps",
    "dropout",
    "tie_embeddings",
];

/// Geometry of the causal GQA decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub context: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub dropout: f32,
    pub tie_embeddings: bool,
}

impl DecoderConfig {
    /// Parse and validate the single `[decoder]` section.
    ///
    /// `[net]` / `[network]` metadata sections are tolerated and ignored; any
    /// other section is rejected, so a detector config cannot be mistaken for a
    /// decoder one.
    pub fn from_cfg(cfg: &ConfigFile) -> Result<Self> {
        let mut decoders = cfg.sections.iter().filter(|s| s.name == "decoder");
        let section = decoders.next().ok_or_else(|| {
            Error::InvalidConfig(format!(
                "{}: no [decoder] section found",
                cfg.path.display()
            ))
        })?;
        if decoders.next().is_some() {
            return Err(Error::InvalidConfig(format!(
                "{}: more than one [decoder] section",
                cfg.path.display()
            )));
        }
        for other in &cfg.sections {
            if !matches!(other.name.as_str(), "decoder" | "net" | "network") {
                return Err(Error::InvalidConfig(format!(
                    "{}: unsupported section [{}] at line {}",
                    cfg.path.display(),
                    other.name,
                    other.line
                )));
            }
        }
        for key in section.values.keys() {
            if !DECODER_KEYS.contains(&key.as_str()) {
                return Err(Error::InvalidConfig(format!(
                    "{}: unknown key \"{key}\" in [decoder] at line {}",
                    cfg.path.display(),
                    section.line
                )));
            }
        }

        let config = Self {
            vocab_size: section.required_usize("vocab_size")?,
            context: section.required_usize("context")?,
            hidden: section.required_usize("hidden")?,
            intermediate: section.required_usize("intermediate")?,
            layers: section.required_usize("layers")?,
            q_heads: section.required_usize("q_heads")?,
            kv_heads: section.required_usize("kv_heads")?,
            rope_theta: section.required_f32("rope_theta")?,
            rms_eps: section.required_f32("rms_eps")?,
            dropout: section.required_f32("dropout")?,
            tie_embeddings: section.required_flag("tie_embeddings")?,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate geometry and numeric ranges.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("context", self.context),
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("layers", self.layers),
            ("q_heads", self.q_heads),
            ("kv_heads", self.kv_heads),
        ] {
            if value == 0 {
                return Err(Error::InvalidConfig(format!("{name} must be positive")));
            }
        }
        if !self.hidden.is_multiple_of(self.q_heads) {
            return Err(Error::InvalidConfig(format!(
                "hidden ({}) must be divisible by q_heads ({})",
                self.hidden, self.q_heads
            )));
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(Error::InvalidConfig(format!(
                "q_heads ({}) must be divisible by kv_heads ({})",
                self.q_heads, self.kv_heads
            )));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(Error::InvalidConfig(format!(
                "head dimension ({}) must be even for split-half RoPE",
                self.head_dim()
            )));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::InvalidConfig(
                "rope_theta must be finite and positive".to_owned(),
            ));
        }
        if !self.rms_eps.is_finite() || self.rms_eps <= 0.0 {
            return Err(Error::InvalidConfig(
                "rms_eps must be finite and positive".to_owned(),
            ));
        }
        if !self.dropout.is_finite() || self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(Error::InvalidConfig(
                "dropout must be finite and in [0, 1)".to_owned(),
            ));
        }
        if !self.tie_embeddings {
            return Err(Error::InvalidConfig(
                "tie_embeddings=0 is not supported; this decoder requires a tied head".to_owned(),
            ));
        }
        Ok(())
    }

    /// Per-query head width, `hidden / q_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.q_heads
    }

    /// Number of unique f32 parameters in the canonical Task 4 layout.
    ///
    /// The tied output head shares the embedding tensor and is counted once.
    /// All arithmetic is checked so absurd geometries fail instead of wrapping.
    pub fn parameter_count(&self) -> Result<u64> {
        self.validate()?;
        let hidden = to_u64(self.hidden)?;
        let intermediate = to_u64(self.intermediate)?;
        let layers = to_u64(self.layers)?;
        let kv_dim = to_u64(self.kv_heads)?
            .checked_mul(to_u64(self.head_dim())?)
            .ok_or_else(overflow)?;

        let embedding = to_u64(self.vocab_size)?
            .checked_mul(hidden)
            .ok_or_else(overflow)?;
        let mut per_layer = hidden; // attention RMSNorm scale
        per_layer = per_layer
            .checked_add(hidden.checked_mul(hidden).ok_or_else(overflow)?) // Q
            .ok_or_else(overflow)?;
        per_layer = per_layer
            .checked_add(hidden.checked_mul(kv_dim).ok_or_else(overflow)?) // K
            .ok_or_else(overflow)?;
        per_layer = per_layer
            .checked_add(hidden.checked_mul(kv_dim).ok_or_else(overflow)?) // V
            .ok_or_else(overflow)?;
        per_layer = per_layer
            .checked_add(hidden.checked_mul(hidden).ok_or_else(overflow)?) // O
            .ok_or_else(overflow)?;
        per_layer = per_layer.checked_add(hidden).ok_or_else(overflow)?; // MLP RMSNorm scale
        for _ in 0..3 {
            per_layer = per_layer
                .checked_add(hidden.checked_mul(intermediate).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }

        embedding
            .checked_add(per_layer.checked_mul(layers).ok_or_else(overflow)?)
            .and_then(|total| total.checked_add(hidden)) // final RMSNorm scale
            .ok_or_else(overflow)
    }
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow())
}

fn overflow() -> Error {
    Error::InvalidConfig("parameter count overflows u64".to_owned())
}
