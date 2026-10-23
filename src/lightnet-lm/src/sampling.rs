//! Deterministic sampling and single-sequence generation.

use crate::{
    BOS_ID, ByteBpe, CpuDecoder, EOS_ID, Error, KvCache, PAD_ID, Result, RngState, UNK_ID,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl SamplingOptions {
    pub fn rng(&self) -> RngState {
        RngState::new(self.seed)
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Draw one token.  PAD, BOS, and UNK can never be selected; EOS can.
pub fn sample_token(logits: &[f32], options: &SamplingOptions, rng: &mut RngState) -> Result<u32> {
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(Error::InvalidConfig(
            "temperature must be finite and non-negative".to_owned(),
        ));
    }
    if !options.top_p.is_finite() || options.top_p <= 0.0 || options.top_p > 1.0 {
        return Err(Error::InvalidConfig(
            "top_p must be finite and in (0, 1]".to_owned(),
        ));
    }
    let mut allowed = Vec::with_capacity(logits.len().saturating_sub(3));
    for (id, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidConfig(
                "logits contain a non-finite value".to_owned(),
            ));
        }
        let id = id as u32;
        if id == PAD_ID || id == BOS_ID || id == UNK_ID {
            continue;
        }
        allowed.push((id, value));
    }
    if allowed.is_empty() {
        return Err(Error::InvalidConfig(
            "no allowed token candidates".to_owned(),
        ));
    }

    if options.temperature == 0.0 {
        let mut best = allowed[0];
        for &(id, value) in &allowed[1..] {
            if value > best.1 || (value == best.1 && id < best.0) {
                best = (id, value);
            }
        }
        return Ok(best.0);
    }

    let scale = 1.0 / f64::from(options.temperature);
    let max = allowed
        .iter()
        .map(|&(_, value)| f64::from(value) * scale)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0f64;
    let mut candidates: Vec<(u32, f64)> = allowed
        .into_iter()
        .map(|(id, value)| {
            let weight = ((f64::from(value) * scale) - max).exp();
            total += weight;
            (id, weight)
        })
        .collect();
    if total <= 0.0 || !total.is_finite() {
        return Err(Error::InvalidConfig(
            "logits do not form a usable distribution".to_owned(),
        ));
    }
    for (_, probability) in &mut candidates {
        *probability /= total;
    }
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    if options.top_k != 0 && candidates.len() > options.top_k {
        candidates.truncate(options.top_k);
    }
    let mut cumulative = 0.0f64;
    let mut keep = candidates.len();
    for (index, &(_, probability)) in candidates.iter().enumerate() {
        cumulative += probability;
        if cumulative >= f64::from(options.top_p) {
            keep = index + 1;
            break;
        }
    }
    candidates.truncate(keep);
    let mass: f64 = candidates.iter().map(|(_, probability)| *probability).sum();
    if mass <= 0.0 || !mass.is_finite() {
        return Err(Error::InvalidConfig(
            "filtered distribution has no mass".to_owned(),
        ));
    }
    if candidates.len() == 1 {
        return Ok(candidates[0].0);
    }
    let draw = uniform(rng) * mass;
    let mut cumulative = 0.0f64;
    for &(id, probability) in &candidates {
        cumulative += probability;
        if draw < cumulative {
            return Ok(id);
        }
    }
    Ok(candidates[candidates.len() - 1].0)
}

fn uniform(rng: &mut RngState) -> f64 {
    ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Generate text tokens for one prompt, stopping at EOS.
///
/// Empty text is prefilled with BOS.  EOS is never included in the result.
pub fn generate(
    decoder: &CpuDecoder,
    tokenizer: &ByteBpe,
    prompt: &str,
    options: &SamplingOptions,
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    let cfg = decoder.config();
    let ids = if prompt.is_empty() {
        vec![BOS_ID]
    } else {
        tokenizer.encode(prompt)?
    };
    if ids.is_empty() {
        return Err(Error::InvalidConfig(
            "generation prompt is empty".to_owned(),
        ));
    }
    if ids.len() > cfg.context {
        return Err(Error::InvalidConfig(format!(
            "prompt length {} exceeds context {}",
            ids.len(),
            cfg.context
        )));
    }
    let mut cache = KvCache::new(cfg, 1)?;
    let mut rng = options.rng();
    let mut logits = decoder.prefill(&ids, &mut cache)?;
    let mut output = Vec::new();
    for _ in 0..max_new_tokens {
        let token = sample_token(&logits, options, &mut rng)?;
        if token == EOS_ID {
            break;
        }
        output.push(token);
        if cache.len() >= cfg.context {
            break;
        }
        logits = decoder.decode(&[token], &mut cache)?;
    }
    Ok(output)
}
