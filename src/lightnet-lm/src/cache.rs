//! Single-sequence CPU KV cache.
//!
//! Per layer the layout is `[kv_heads, context, head_dim]`: rotated keys and
//! unrotated values.  Generation supports batch one; other sizes are rejected.

use crate::{DecoderConfig, Error, Result};

#[derive(Clone, Debug)]
pub(crate) struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// One sequence's cache, with the model revision that produced it.
#[derive(Clone, Debug)]
pub struct KvCache {
    cfg: DecoderConfig,
    batch: usize,
    len: usize,
    revision: u64,
    layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(cfg: &DecoderConfig, batch: usize) -> Result<Self> {
        cfg.validate()?;
        if batch != 1 {
            return Err(Error::InvalidConfig(format!(
                "CPU KV cache supports batch 1, got {batch}"
            )));
        }
        let head_dim = cfg.head_dim();
        let width = cfg
            .kv_heads
            .checked_mul(cfg.context)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::InvalidConfig("KV cache size overflows usize".to_owned()))?;
        Ok(Self {
            cfg: cfg.clone(),
            batch,
            len: 0,
            revision: 0,
            layers: (0..cfg.layers)
                .map(|_| LayerCache {
                    keys: vec![0.0; width],
                    values: vec![0.0; width],
                })
                .collect(),
        })
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn set_revision(&mut self, revision: u64) {
        self.revision = revision;
    }

    pub(crate) fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub(crate) fn keys(&self, layer: usize) -> &[f32] {
        &self.layers[layer].keys
    }

    pub(crate) fn values(&self, layer: usize) -> &[f32] {
        &self.layers[layer].values
    }

    pub(crate) fn write(
        &mut self,
        layer: usize,
        offset: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        let head_dim = self.cfg.head_dim();
        let width = self.cfg.kv_heads * self.cfg.context * head_dim;
        if keys.len() != values.len()
            || !keys.len().is_multiple_of(head_dim)
            || offset.checked_mul(head_dim).is_none()
        {
            return Err(Error::InvalidConfig(
                "KV cache write has an invalid shape".to_owned(),
            ));
        }
        let heads = keys.len() / head_dim;
        if heads > self.cfg.kv_heads {
            return Err(Error::InvalidConfig(
                "KV cache write has too many heads".to_owned(),
            ));
        }
        for head in 0..heads {
            let dst = head
                .checked_mul(self.cfg.context)
                .and_then(|value| value.checked_mul(head_dim))
                .and_then(|value| {
                    offset
                        .checked_mul(head_dim)
                        .and_then(|offset| value.checked_add(offset))
                })
                .ok_or_else(|| {
                    Error::InvalidConfig("KV cache write is out of bounds".to_owned())
                })?;
            let end = dst
                .checked_add(head_dim)
                .filter(|end| *end <= width)
                .ok_or_else(|| {
                    Error::InvalidConfig("KV cache write is out of bounds".to_owned())
                })?;
            let src = head * head_dim;
            self.layers[layer].keys[dst..end].copy_from_slice(&keys[src..src + head_dim]);
            self.layers[layer].values[dst..end].copy_from_slice(&values[src..src + head_dim]);
        }
        Ok(())
    }
}
