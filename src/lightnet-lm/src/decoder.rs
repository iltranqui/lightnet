//! Trainable CPU decoder: pre-norm causal GQA blocks with a tied head.
//!
//! Projection weights are `[in, out]` row-major: `y[r, o] = sum_i x[r, i] *
//! w[i, out + o]`.  The tied head reuses the `[vocab, hidden]` embedding as a
//! `[hidden, vocab]` matrix, so the embedding is the only tensor that receives
//! two gradient streams (gather and head).

use crate::attention::{causal_gqa, causal_gqa_backward};
use crate::cache::KvCache;
use crate::ops::{
    embedding, embedding_backward, rms_norm, rms_norm_backward, rope_in_place, sparse_ce, swiglu,
    swiglu_backward,
};
use crate::{DecoderConfig, Error, LossSummary, Result, RngState, TokenBatch};

/// One named f32 tensor in the canonical registry.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub id: u32,
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub decay: bool,
}

/// The unique parameters of a decoder, in canonical order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecoderParams {
    pub tensors: Vec<Parameter>,
}

/// Gradients indexed by parameter ID; every vector matches its parameter shape.
#[derive(Clone, Debug, PartialEq)]
pub struct DecoderGrads {
    pub values: Vec<Vec<f32>>,
}

/// Logits plus the opaque tape needed for the exact adjoint.
#[derive(Clone, Debug)]
pub struct CpuForward {
    pub logits: Vec<f32>,
    pub tape: DecoderTape,
}

/// Backward intermediates.  Fields are private so callers cannot depend on
/// their layout; one tape is valid only for the decoder revision that made it.
#[derive(Clone, Debug)]
pub struct DecoderTape {
    batch: TokenBatch,
    rows: usize,
    revision: u64,
    x0: Vec<f32>,
    layers: Vec<LayerTape>,
    final_norm: Vec<f32>,
}

impl DecoderTape {
    /// Rotated keys and unrotated values for one layer, `[rows, kv_width]`.
    pub(crate) fn layer_kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        self.layers
            .get(layer)
            .map(|tape| (tape.k.as_slice(), tape.v.as_slice()))
    }

    /// Final RMSNorm input (`[rows, hidden]`), exposed for tied-head checks.
    pub fn final_hidden(&self) -> &[f32] {
        &self.x0
    }

    /// Final RMSNorm output (`[rows, hidden]`), exposed for tied-head checks.
    pub fn final_norm(&self) -> &[f32] {
        &self.final_norm
    }
}

#[derive(Clone, Debug)]
struct LayerTape {
    norm1_input: Vec<f32>,
    norm1_output: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_dropped: Vec<f32>,
    attention_mask: Option<Vec<f32>>,
    norm2_input: Vec<f32>,
    norm2_output: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mlp_dropped: Vec<f32>,
    mlp_mask: Option<Vec<f32>>,
}

/// CPU reference decoder.
#[derive(Clone, Debug)]
pub struct CpuDecoder {
    cfg: DecoderConfig,
    params: DecoderParams,
    revision: u64,
}

impl CpuDecoder {
    /// Build the canonical parameter registry with deterministic initialization.
    pub fn new(cfg: DecoderConfig, seed: u64) -> Result<Self> {
        cfg.validate()?;
        cfg.parameter_count()?;
        let dims = Dims::new(&cfg)?;
        let tensor_count = cfg
            .layers
            .checked_mul(9)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| Error::InvalidConfig("parameter tensor count overflowed".to_owned()))?;
        let mut rng = RngState::new(seed);
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut id = 0usize;
        let output_std = 0.02 / ((2 * cfg.layers) as f32).sqrt();

        tensors.push(matrix(
            &mut id,
            "embedding",
            vec![cfg.vocab_size, cfg.hidden],
            0.02,
            true,
            &mut rng,
        ));
        for layer in 0..cfg.layers {
            let prefix = format!("blocks.{layer}");
            tensors.push(scale(&mut id, &format!("{prefix}.attn_norm"), cfg.hidden));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.attn_q"),
                vec![cfg.hidden, cfg.hidden],
                0.02,
                true,
                &mut rng,
            ));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.attn_k"),
                vec![cfg.hidden, dims.kv_dim],
                0.02,
                true,
                &mut rng,
            ));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.attn_v"),
                vec![cfg.hidden, dims.kv_dim],
                0.02,
                true,
                &mut rng,
            ));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.attn_o"),
                vec![cfg.hidden, cfg.hidden],
                output_std,
                true,
                &mut rng,
            ));
            tensors.push(scale(&mut id, &format!("{prefix}.mlp_norm"), cfg.hidden));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.mlp_gate"),
                vec![cfg.hidden, cfg.intermediate],
                0.02,
                true,
                &mut rng,
            ));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.mlp_up"),
                vec![cfg.hidden, cfg.intermediate],
                0.02,
                true,
                &mut rng,
            ));
            tensors.push(matrix(
                &mut id,
                &format!("{prefix}.mlp_down"),
                vec![cfg.intermediate, cfg.hidden],
                output_std,
                true,
                &mut rng,
            ));
        }
        tensors.push(scale(&mut id, "final_norm", cfg.hidden));

        if tensors.len() != tensor_count {
            return Err(Error::InvalidConfig(format!(
                "decoder built {} parameters, expected {}",
                tensors.len(),
                tensor_count
            )));
        }
        let params = DecoderParams { tensors };
        if params
            .tensors
            .iter()
            .any(|tensor| tensor.values.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::InvalidConfig(
                "parameter initialization produced a non-finite value".to_owned(),
            ));
        }
        Ok(Self {
            cfg,
            params,
            revision: 0,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    /// Monotonic revision; replacement through [`Self::set_parameters`] bumps it.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn parameters(&self) -> &DecoderParams {
        &self.params
    }

    /// Replace every parameter after checking IDs, names, shapes, and finiteness.
    pub fn set_parameters(&mut self, params: DecoderParams) -> Result<()> {
        if params.tensors.len() != self.params.tensors.len() {
            return Err(Error::InvalidConfig(format!(
                "parameter registry has {} tensors, expected {}",
                params.tensors.len(),
                self.params.tensors.len()
            )));
        }
        for (index, (incoming, current)) in
            params.tensors.iter().zip(&self.params.tensors).enumerate()
        {
            if incoming.id != index as u32 {
                return Err(Error::InvalidConfig(format!(
                    "parameter {index} has id {}, expected {index}",
                    incoming.id
                )));
            }
            if incoming.name != current.name
                || incoming.shape != current.shape
                || incoming.decay != current.decay
            {
                return Err(Error::InvalidConfig(format!(
                    "parameter {} ({}) does not match the canonical registry",
                    incoming.id, incoming.name
                )));
            }
            if incoming.values.len() != current.values.len()
                || incoming.values.len() != incoming.shape.iter().product::<usize>()
            {
                return Err(Error::InvalidConfig(format!(
                    "parameter {} has {} values for shape {:?}",
                    incoming.name,
                    incoming.values.len(),
                    incoming.shape
                )));
            }
            if incoming.values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidConfig(format!(
                    "parameter {} contains a non-finite value",
                    incoming.name
                )));
            }
        }
        self.params = params;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Forward pass.  `training_rng` enables inverted dropout; `None` is the
    /// deterministic inference path and consumes no randomness.
    pub fn forward(
        &self,
        batch: &TokenBatch,
        training_rng: Option<&mut RngState>,
    ) -> Result<CpuForward> {
        self.cfg.validate()?;
        batch.validate(&self.cfg)?;
        let dims = Dims::new(&self.cfg)?;
        let rows = batch.rows();
        let mut rng = training_rng;

        let embedding_table = self.tensor(0)?.values.as_slice();
        let mut hidden_state = embedding(embedding_table, &batch.input_ids, self.cfg.hidden)?;
        let mut layers = Vec::with_capacity(self.cfg.layers);

        for layer in 0..self.cfg.layers {
            let base = block_base(layer);
            let norm1_input = hidden_state.clone();
            let norm1 = rms_norm(
                &norm1_input,
                &self.tensor(base)?.values,
                rows,
                self.cfg.rms_eps,
            )?;
            let mut q = linear(
                &norm1,
                &self.tensor(base + 1)?.values,
                rows,
                self.cfg.hidden,
                self.cfg.hidden,
            );
            let mut k = linear(
                &norm1,
                &self.tensor(base + 2)?.values,
                rows,
                self.cfg.hidden,
                dims.kv_dim,
            );
            let v = linear(
                &norm1,
                &self.tensor(base + 3)?.values,
                rows,
                self.cfg.hidden,
                dims.kv_dim,
            );
            rope_in_place(
                &mut q,
                &batch.positions,
                self.cfg.q_heads,
                dims.head_dim,
                self.cfg.rope_theta,
                false,
            )?;
            rope_in_place(
                &mut k,
                &batch.positions,
                self.cfg.kv_heads,
                dims.head_dim,
                self.cfg.rope_theta,
                false,
            )?;
            let attention = causal_gqa(&q, &k, &v, batch, &self.cfg)?;
            let (attention_dropped, attention_mask) =
                dropout(&attention, self.cfg.dropout, rng.as_deref_mut());
            let projection = linear(
                &attention_dropped,
                &self.tensor(base + 4)?.values,
                rows,
                self.cfg.hidden,
                self.cfg.hidden,
            );
            for index in 0..hidden_state.len() {
                hidden_state[index] += projection[index];
            }

            let norm2_input = hidden_state.clone();
            let norm2 = rms_norm(
                &norm2_input,
                &self.tensor(base + 5)?.values,
                rows,
                self.cfg.rms_eps,
            )?;
            let gate = linear(
                &norm2,
                &self.tensor(base + 6)?.values,
                rows,
                self.cfg.hidden,
                self.cfg.intermediate,
            );
            let up = linear(
                &norm2,
                &self.tensor(base + 7)?.values,
                rows,
                self.cfg.hidden,
                self.cfg.intermediate,
            );
            let activated = swiglu(&gate, &up)?;
            let (mlp_dropped, mlp_mask) = dropout(&activated, self.cfg.dropout, rng.as_deref_mut());
            let down = linear(
                &mlp_dropped,
                &self.tensor(base + 8)?.values,
                rows,
                self.cfg.intermediate,
                self.cfg.hidden,
            );
            for index in 0..hidden_state.len() {
                hidden_state[index] += down[index];
            }

            layers.push(LayerTape {
                norm1_input,
                norm1_output: norm1,
                q,
                k,
                v,
                attention_dropped,
                attention_mask,
                norm2_input,
                norm2_output: norm2,
                gate,
                up,
                mlp_dropped,
                mlp_mask,
            });
        }

        let final_index = final_norm_index(self.cfg.layers);
        let final_norm = rms_norm(
            &hidden_state,
            &self.tensor(final_index)?.values,
            rows,
            self.cfg.rms_eps,
        )?;
        let logits = tied_head(
            &final_norm,
            embedding_table,
            rows,
            self.cfg.vocab_size,
            self.cfg.hidden,
        );

        Ok(CpuForward {
            logits,
            tape: DecoderTape {
                batch: batch.clone(),
                rows,
                revision: self.revision,
                x0: hidden_state,
                layers,
                final_norm,
            },
        })
    }

    /// Fill an empty cache from a prompt and return the last position's logits.
    pub fn prefill(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        self.cfg.validate()?;
        if ids.is_empty() {
            return Err(Error::InvalidConfig(
                "prefill requires at least one token".to_owned(),
            ));
        }
        if ids.len() > self.cfg.context {
            return Err(Error::InvalidConfig(format!(
                "prefill length {} exceeds context {}",
                ids.len(),
                self.cfg.context
            )));
        }
        if !cache.is_empty() {
            return Err(Error::InvalidConfig(
                "prefill requires an empty cache".to_owned(),
            ));
        }
        if cache.batch() != 1 {
            return Err(Error::InvalidConfig(
                "KV cache batch does not match the prefill".to_owned(),
            ));
        }
        for (row, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                return Err(Error::InvalidConfig(format!(
                    "prefill token {id} at row {row} is outside vocabulary ({})",
                    self.cfg.vocab_size
                )));
            }
        }
        let time = ids.len();
        let batch = TokenBatch {
            batch: 1,
            time,
            input_ids: ids.to_vec(),
            targets: vec![-100; time],
            positions: (0..time as u32).collect(),
            segments: vec![0; time],
            valid: vec![true; time],
        };
        let forward = self.forward(&batch, None)?;
        for layer in 0..self.cfg.layers {
            let (keys, values) = forward.tape.layer_kv(layer).ok_or_else(|| {
                Error::InvalidConfig("forward tape is missing a layer".to_owned())
            })?;
            let width = self.cfg.kv_heads * self.cfg.head_dim();
            for position in 0..time {
                let start = position * width;
                cache.write(
                    layer,
                    position,
                    &keys[start..start + width],
                    &values[start..start + width],
                )?;
            }
        }
        cache.set_len(time);
        cache.set_revision(self.revision);
        Ok(forward.logits[(time - 1) * self.cfg.vocab_size..].to_vec())
    }

    /// Append one token to a prefilled cache and return its logits.
    pub fn decode(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        self.cfg.validate()?;
        if ids.len() != 1 {
            return Err(Error::InvalidConfig(
                "decode appends exactly one token".to_owned(),
            ));
        }
        let token = ids[0];
        if token as usize >= self.cfg.vocab_size {
            return Err(Error::InvalidConfig(format!(
                "decode token {token} is outside vocabulary ({})",
                self.cfg.vocab_size
            )));
        }
        if cache.batch() != 1 || cache.revision() != self.revision {
            return Err(Error::InvalidConfig(
                "KV cache belongs to a different decoder revision".to_owned(),
            ));
        }
        if cache.len() >= self.cfg.context {
            return Err(Error::InvalidConfig(format!(
                "KV cache is full ({} of {})",
                cache.len(),
                self.cfg.context
            )));
        }
        let dims = Dims::new(&self.cfg)?;
        let hidden = self.cfg.hidden;
        let position = cache.len();
        let embedding_table = self.tensor(0)?.values.as_slice();
        let mut hidden_state = embedding(embedding_table, ids, hidden)?;
        let group = self.cfg.q_heads / self.cfg.kv_heads;
        let scale = 1.0 / (dims.head_dim as f32).sqrt();
        let fan = self.cfg.context;

        for layer in 0..self.cfg.layers {
            let base = block_base(layer);
            let norm1 = rms_norm(
                &hidden_state,
                &self.tensor(base)?.values,
                1,
                self.cfg.rms_eps,
            )?;
            let mut q = linear(&norm1, &self.tensor(base + 1)?.values, 1, hidden, hidden);
            let mut k = linear(
                &norm1,
                &self.tensor(base + 2)?.values,
                1,
                hidden,
                dims.kv_dim,
            );
            let v = linear(
                &norm1,
                &self.tensor(base + 3)?.values,
                1,
                hidden,
                dims.kv_dim,
            );
            let positions = [position as u32];
            rope_in_place(
                &mut q,
                &positions,
                self.cfg.q_heads,
                dims.head_dim,
                self.cfg.rope_theta,
                false,
            )?;
            rope_in_place(
                &mut k,
                &positions,
                self.cfg.kv_heads,
                dims.head_dim,
                self.cfg.rope_theta,
                false,
            )?;
            cache.write(layer, position, &k, &v)?;
            let total = position + 1;
            let keys = cache.keys(layer);
            let values = cache.values(layer);
            let mut attention = vec![0.0f32; hidden];
            for q_head in 0..self.cfg.q_heads {
                let kv_head = q_head / group;
                let query = &q[q_head * dims.head_dim..(q_head + 1) * dims.head_dim];
                let mut scores = vec![0.0f32; total];
                for (time, score) in scores.iter_mut().enumerate() {
                    let offset = kv_head * fan * dims.head_dim + time * dims.head_dim;
                    *score = query
                        .iter()
                        .zip(&keys[offset..offset + dims.head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                }
                softmax_in_place(&mut scores);
                let target = &mut attention[q_head * dims.head_dim..(q_head + 1) * dims.head_dim];
                for (time, &weight) in scores.iter().enumerate() {
                    let offset = kv_head * fan * dims.head_dim + time * dims.head_dim;
                    for (value, &cached) in target
                        .iter_mut()
                        .zip(&values[offset..offset + dims.head_dim])
                    {
                        *value += weight * cached;
                    }
                }
            }
            let projection = linear(
                &attention,
                &self.tensor(base + 4)?.values,
                1,
                hidden,
                hidden,
            );
            for (value, delta) in hidden_state.iter_mut().zip(projection) {
                *value += delta;
            }

            let norm2 = rms_norm(
                &hidden_state,
                &self.tensor(base + 5)?.values,
                1,
                self.cfg.rms_eps,
            )?;
            let gate = linear(
                &norm2,
                &self.tensor(base + 6)?.values,
                1,
                hidden,
                self.cfg.intermediate,
            );
            let up = linear(
                &norm2,
                &self.tensor(base + 7)?.values,
                1,
                hidden,
                self.cfg.intermediate,
            );
            let activated = swiglu(&gate, &up)?;
            let down = linear(
                &activated,
                &self.tensor(base + 8)?.values,
                1,
                self.cfg.intermediate,
                hidden,
            );
            for (value, delta) in hidden_state.iter_mut().zip(down) {
                *value += delta;
            }
        }

        let final_norm = rms_norm(
            &hidden_state,
            &self.tensor(final_norm_index(self.cfg.layers))?.values,
            1,
            self.cfg.rms_eps,
        )?;
        cache.set_len(position + 1);
        Ok(tied_head(
            &final_norm,
            embedding_table,
            1,
            self.cfg.vocab_size,
            hidden,
        ))
    }

    /// Exact parameter gradient for `logits_seed`.
    ///
    /// The seed is the repo's descent direction `one_hot - softmax`, i.e.
    /// `-dLoss/dlogits` (as returned by [`sparse_ce`]).  Backward propagates
    /// that sign unchanged, so the result is also a descent direction and the
    /// optimizer adds `lr * update` (matching `adam_update`).
    pub fn backward(&self, forward: &CpuForward, logits_seed: &[f32]) -> Result<DecoderGrads> {
        Ok(self.backward_impl(forward, logits_seed)?.0)
    }

    /// Split the tied embedding gradient into its head and gather terms.
    /// Useful for diagnostics and for the tied-head invariant test.
    pub fn embedding_gradient_terms(
        &self,
        forward: &CpuForward,
        logits_seed: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (_, head, gather) = self.backward_impl(forward, logits_seed)?;
        Ok((head, gather))
    }

    fn backward_impl(
        &self,
        forward: &CpuForward,
        logits_seed: &[f32],
    ) -> Result<(DecoderGrads, Vec<f32>, Vec<f32>)> {
        if forward.tape.revision != self.revision {
            return Err(Error::InvalidConfig(
                "forward tape belongs to a different decoder revision".to_owned(),
            ));
        }
        if logits_seed.len() != forward.logits.len() {
            return Err(Error::InvalidConfig(format!(
                "logits seed has length {}, expected {}",
                logits_seed.len(),
                forward.logits.len()
            )));
        }
        let cfg = &self.cfg;
        let dims = Dims::new(cfg)?;
        let rows = forward.tape.rows;
        let hidden = cfg.hidden;
        let vocab = cfg.vocab_size;
        let embedding_values = self.tensor(0)?.values.as_slice();
        let mut grads = DecoderGrads {
            values: vec![Vec::new(); self.params.tensors.len()],
        };

        let mut d_norm = vec![0.0; rows * hidden];
        let mut d_embedding = vec![0.0; vocab * hidden];
        for row in 0..rows {
            let seed_row = &logits_seed[row * vocab..(row + 1) * vocab];
            let norm_row = &forward.tape.final_norm[row * hidden..(row + 1) * hidden];
            for class in 0..vocab {
                let seed = seed_row[class];
                if seed == 0.0 {
                    continue;
                }
                let embedding_row = &embedding_values[class * hidden..(class + 1) * hidden];
                for channel in 0..hidden {
                    d_norm[row * hidden + channel] += seed * embedding_row[channel];
                    d_embedding[class * hidden + channel] += seed * norm_row[channel];
                }
            }
        }

        let final_index = final_norm_index(cfg.layers);
        let (mut d_stream, d_final_norm) = rms_norm_backward(
            &forward.tape.x0,
            &self.tensor(final_index)?.values,
            &d_norm,
            rows,
            cfg.rms_eps,
        )?;
        grads.values[final_index] = d_final_norm;

        for layer in (0..cfg.layers).rev() {
            let base = block_base(layer);
            let tape = &forward.tape.layers[layer];

            let (d_down, mut d_mlp_dropped) = linear_backward(
                &tape.mlp_dropped,
                &self.tensor(base + 8)?.values,
                &d_stream,
                rows,
                cfg.intermediate,
                hidden,
            );
            if let Some(mask) = &tape.mlp_mask {
                for (value, &mask) in d_mlp_dropped.iter_mut().zip(mask) {
                    *value *= mask;
                }
            }
            let (d_gate, d_up) = swiglu_backward(&tape.gate, &tape.up, &d_mlp_dropped)?;
            let (d_gate_w, d_norm2_gate) = linear_backward(
                &tape.norm2_output,
                &self.tensor(base + 6)?.values,
                &d_gate,
                rows,
                hidden,
                cfg.intermediate,
            );
            let (d_up_w, d_norm2_up) = linear_backward(
                &tape.norm2_output,
                &self.tensor(base + 7)?.values,
                &d_up,
                rows,
                hidden,
                cfg.intermediate,
            );
            let mut d_norm2 = d_norm2_gate;
            for (value, addition) in d_norm2.iter_mut().zip(d_norm2_up) {
                *value += addition;
            }
            let (d_mlp_branch, d_mlp_norm) = rms_norm_backward(
                &tape.norm2_input,
                &self.tensor(base + 5)?.values,
                &d_norm2,
                rows,
                cfg.rms_eps,
            )?;

            let (d_o, mut d_attention_dropped) = linear_backward(
                &tape.attention_dropped,
                &self.tensor(base + 4)?.values,
                &d_stream,
                rows,
                hidden,
                hidden,
            );
            if let Some(mask) = &tape.attention_mask {
                for (value, &mask) in d_attention_dropped.iter_mut().zip(mask) {
                    *value *= mask;
                }
            }
            let (mut d_q, mut d_k, d_v) = causal_gqa_backward(
                &tape.q,
                &tape.k,
                &tape.v,
                &d_attention_dropped,
                &forward.tape.batch,
                cfg,
            )?;
            rope_in_place(
                &mut d_q,
                &forward.tape.batch.positions,
                cfg.q_heads,
                dims.head_dim,
                cfg.rope_theta,
                true,
            )?;
            rope_in_place(
                &mut d_k,
                &forward.tape.batch.positions,
                cfg.kv_heads,
                dims.head_dim,
                cfg.rope_theta,
                true,
            )?;
            let (d_q_w, d_norm1_q) = linear_backward(
                &tape.norm1_output,
                &self.tensor(base + 1)?.values,
                &d_q,
                rows,
                hidden,
                hidden,
            );
            let (d_k_w, d_norm1_k) = linear_backward(
                &tape.norm1_output,
                &self.tensor(base + 2)?.values,
                &d_k,
                rows,
                hidden,
                dims.kv_dim,
            );
            let (d_v_w, d_norm1_v) = linear_backward(
                &tape.norm1_output,
                &self.tensor(base + 3)?.values,
                &d_v,
                rows,
                hidden,
                dims.kv_dim,
            );
            let mut d_norm1 = d_norm1_q;
            for (value, addition) in d_norm1.iter_mut().zip(d_norm1_k) {
                *value += addition;
            }
            for (value, addition) in d_norm1.iter_mut().zip(d_norm1_v) {
                *value += addition;
            }
            let (d_attention_branch, d_attn_norm) = rms_norm_backward(
                &tape.norm1_input,
                &self.tensor(base)?.values,
                &d_norm1,
                rows,
                cfg.rms_eps,
            )?;

            for index in 0..d_stream.len() {
                d_stream[index] += d_mlp_branch[index] + d_attention_branch[index];
            }

            grads.values[base] = d_attn_norm;
            grads.values[base + 1] = d_q_w;
            grads.values[base + 2] = d_k_w;
            grads.values[base + 3] = d_v_w;
            grads.values[base + 4] = d_o;
            grads.values[base + 5] = d_mlp_norm;
            grads.values[base + 6] = d_gate_w;
            grads.values[base + 7] = d_up_w;
            grads.values[base + 8] = d_down;
        }

        let d_gather = embedding_backward(&forward.tape.batch.input_ids, &d_stream, vocab, hidden)?;
        let d_head = d_embedding.clone();
        for (value, &addition) in d_embedding.iter_mut().zip(&d_gather) {
            *value += addition;
        }
        grads.values[0] = d_embedding;

        Ok((grads, d_head, d_gather))
    }

    /// Sparse cross-entropy over the forward logits, returning the summed loss.
    pub fn loss(
        &self,
        forward: &CpuForward,
        batch: &TokenBatch,
    ) -> Result<(LossSummary, Vec<f32>)> {
        sparse_ce(&forward.logits, &batch.targets, self.cfg.vocab_size)
    }

    fn tensor(&self, index: usize) -> Result<&Parameter> {
        self.params.tensors.get(index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing parameter {index} in the registry"))
        })
    }
}

struct Dims {
    kv_dim: usize,
    head_dim: usize,
}

impl Dims {
    fn new(cfg: &DecoderConfig) -> Result<Self> {
        let kv_dim = checked_mul(cfg.kv_heads, cfg.head_dim(), "K/V width")?;
        Ok(Self {
            kv_dim,
            head_dim: cfg.head_dim(),
        })
    }
}

fn block_base(layer: usize) -> usize {
    1 + 9 * layer
}

fn final_norm_index(layers: usize) -> usize {
    1 + 9 * layers
}

fn matrix(
    id: &mut usize,
    name: &str,
    shape: Vec<usize>,
    std: f32,
    decay: bool,
    rng: &mut RngState,
) -> Parameter {
    let length = shape.iter().product::<usize>();
    let values = (0..length).map(|_| normal(&mut *rng) * std).collect();
    let parameter = Parameter {
        id: *id as u32,
        name: name.to_owned(),
        shape,
        values,
        decay,
    };
    *id += 1;
    parameter
}

fn scale(id: &mut usize, name: &str, hidden: usize) -> Parameter {
    let parameter = Parameter {
        id: *id as u32,
        name: name.to_owned(),
        shape: vec![hidden],
        values: vec![1.0; hidden],
        decay: false,
    };
    *id += 1;
    parameter
}

fn normal(rng: &mut RngState) -> f32 {
    let unit = 1.0 / (1u64 << 53) as f64;
    let u1 = ((rng.next_u64() >> 11) as f64 + 0.5) * unit;
    let u2 = ((rng.next_u64() >> 11) as f64 + 0.5) * unit;
    let value = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    value as f32
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn linear(x: &[f32], weight: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut result = vec![0.0; rows * output];
    for row in 0..rows {
        let source = &x[row * input..(row + 1) * input];
        let target = &mut result[row * output..(row + 1) * output];
        for (channel, &value) in source.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let weight_row = &weight[channel * output..(channel + 1) * output];
            for (result, &weight) in target.iter_mut().zip(weight_row) {
                *result += value * weight;
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn linear_backward(
    x: &[f32],
    weight: &[f32],
    dy: &[f32],
    rows: usize,
    input: usize,
    output: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_weight = vec![0.0; input * output];
    let mut d_x = vec![0.0; rows * input];
    for row in 0..rows {
        let source = &x[row * input..(row + 1) * input];
        let d_row = &dy[row * output..(row + 1) * output];
        let d_target = &mut d_x[row * input..(row + 1) * input];
        for (channel, &value) in source.iter().enumerate() {
            let weight_row = &weight[channel * output..(channel + 1) * output];
            let d_weight_row = &mut d_weight[channel * output..(channel + 1) * output];
            for ((d_weight, &d_output), &weight) in
                d_weight_row.iter_mut().zip(d_row).zip(weight_row)
            {
                *d_weight += value * d_output;
                d_target[channel] += weight * d_output;
            }
        }
    }
    (d_weight, d_x)
}

fn tied_head(
    hidden_state: &[f32],
    embedding: &[f32],
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0; rows * vocab];
    for row in 0..rows {
        let source = &hidden_state[row * hidden..(row + 1) * hidden];
        let target = &mut logits[row * vocab..(row + 1) * vocab];
        for (class, value) in target.iter_mut().enumerate() {
            let embedding_row = &embedding[class * hidden..(class + 1) * hidden];
            *value = source
                .iter()
                .zip(embedding_row)
                .map(|(&left, &right)| left * right)
                .sum();
        }
    }
    logits
}

fn dropout(
    values: &[f32],
    probability: f32,
    rng: Option<&mut RngState>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let Some(rng) = rng.filter(|_| probability > 0.0) else {
        return (values.to_vec(), None);
    };
    let scale = 1.0 / (1.0 - probability);
    let mask: Vec<f32> = values
        .iter()
        .map(|_| {
            if unit(&mut *rng) < probability {
                0.0
            } else {
                scale
            }
        })
        .collect();
    let out = values
        .iter()
        .zip(&mask)
        .map(|(&value, &mask)| value * mask)
        .collect();
    (out, Some(mask))
}

fn unit(rng: &mut RngState) -> f32 {
    let numerator = (rng.next_u64() >> 40) as f32;
    numerator / (1u64 << 24) as f32
}

fn checked_mul(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::InvalidConfig(format!("{what} overflows usize")))
}
