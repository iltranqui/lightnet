//! LAYA kernels: act head, marker gather, head, encoder and scorer packing and attention.

use super::activations::exact_gelu_value;
use super::convert::{f16_bits_to_f32, f32_to_f16_bits};
use super::layers_extra::{marker_position};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Source ah_features: one thread per sample, serial option reductions.
    /// Masked slots retain finite -1e4 logits, even if all slots are masked.
    #[kernel]
    pub fn laya_act_features(
        head: &[f32],
        logits: &[f32],
        markers: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
        marker_channels: usize,
        head_width: usize,
        marker_width: usize,
    ) {
        let b = thread::index_1d().get();
        if b < output.len() / (hidden + 4) {
            let first_active = markers[(b * marker_channels + 1) * marker_width] > 0.5;
            let mut max = if first_active {
                logits[b * options]
            } else {
                -1e4
            };
            let mut valid = 0usize;
            let mut j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                if active {
                    valid += 1;
                }
                max = max.max(z);
                j += 1;
            }
            let mut sum = 0.0f32;
            j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                sum += (z - max).exp();
                j += 1;
            }
            let mut top1 = -1.0f32;
            let mut top2 = -1.0f32;
            let mut entropy = 0.0f32;
            j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                let p = (z - max).exp() / sum;
                entropy -= p * p.max(1e-9).ln();
                if p > top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
                j += 1;
            }
            let count = if valid < 2 { 2.0 } else { valid as f32 };
            let base = b * (hidden + 4);
            // SAFETY: layout/storage are checked by the host. Threads own
            // disjoint sample rows and write every CLS/summary element once.
            unsafe {
                let mut c = 0;
                while c < hidden {
                    *output.as_mut_ptr().add(base + c) = head[(b * hidden + c) * head_width];
                    c += 1;
                }
                *output.as_mut_ptr().add(base + hidden) = top1;
                *output.as_mut_ptr().add(base + hidden + 1) = top1 - top2;
                *output.as_mut_ptr().add(base + hidden + 2) = entropy / count.ln();
                *output.as_mut_ptr().add(base + hidden + 3) = count / 255.0;
            }
        }
    }

    /// Detached LAYA act-head features from half-storage sources.
    #[kernel]
    pub fn laya_act_features_f16(
        head: &[u16],
        logits: &[u16],
        markers: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: usize,
        options: usize,
        marker_channels: usize,
        head_width: usize,
        marker_width: usize,
    ) {
        let b = thread::index_1d().get();
        if b < output.len() / (hidden + 4) {
            let mut max = f32::NEG_INFINITY;
            let mut valid = 0usize;
            let mut j = 0usize;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                if active {
                    valid += 1;
                }
                max = max.max(z);
                j += 1;
            }
            let mut sum = 0.0f32;
            j = 0;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                sum += (z - max).exp();
                j += 1;
            }
            let mut top1 = -1.0f32;
            let mut top2 = -1.0f32;
            let mut entropy = 0.0f32;
            j = 0;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                let p = (z - max).exp() / sum;
                entropy -= p * p.max(1.0e-9).ln();
                if p > top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
                j += 1;
            }
            let count = if valid < 2 { 2.0 } else { valid as f32 };
            let base = b * (hidden + 4);
            // SAFETY: layout/storage are validated and each thread owns one row.
            unsafe {
                let mut c = 0usize;
                while c < hidden {
                    *output.as_mut_ptr().add(base + c) = head[(b * hidden + c) * head_width];
                    c += 1;
                }
                *output.as_mut_ptr().add(base + hidden) = f32_to_f16_bits(top1, 0);
                *output.as_mut_ptr().add(base + hidden + 1) = f32_to_f16_bits(top1 - top2, 0);
                *output.as_mut_ptr().add(base + hidden + 2) =
                    f32_to_f16_bits(entropy / count.ln(), 0);
                *output.as_mut_ptr().add(base + hidden + 3) = f32_to_f16_bits(count / 255.0, 0);
            }
        }
    }

    /// Scatter the pooled CLS adjoint into token zero of each NCHW channel.
    /// One thread owns each output element; summary channels are detached.
    #[kernel]
    pub fn laya_act_cls_scatter(
        dcls: &[f32],
        head_grad: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: usize,
    ) {
        let i = thread::index_1d();
        let idx = i.get();
        if idx < output.len() {
            let value = head_grad[idx]
                + if idx % tokens == 0 {
                    dcls[idx / tokens]
                } else {
                    0.0
                };
            // SAFETY: host validation proves the source extents and each thread
            // writes exactly one output element.
            unsafe {
                *output.as_mut_ptr().add(idx) = value;
            }
        }
    }

    /// One thread owns each gathered output; both inputs stay device-resident.
    #[kernel]
    pub fn marker_gather(
        features: &[f32],
        markers: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < output.len() {
            let j = i % options;
            let channel = i / options;
            let b = channel / hidden;
            let pos = marker_position(markers[(b * marker_channels + 2 * j) * sequence], sequence);
            // SAFETY: validated NCHW extents bound channel/pos and each thread owns i.
            unsafe {
                *output.as_mut_ptr().add(i) = features[channel * sequence + pos];
            }
        }
    }

    /// Half-storage inference: decode only marker metadata; feature bits are
    /// copied without floating arithmetic (including NaN payloads/subnormals).
    #[kernel]
    pub fn marker_gather_f16(
        features: &[u16],
        markers: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < output.len() {
            let j = i % options;
            let channel = i / options;
            let b = channel / hidden;
            let marker = f16_bits_to_f32(markers[(b * marker_channels + 2 * j) * sequence]);
            let pos = marker_position(marker, sequence);
            // SAFETY: checked extents bound the read; one thread owns each output.
            unsafe {
                *output.as_mut_ptr().add(i) = features[channel * sequence + pos];
            }
        }
    }

    /// One thread per input feature. Ascending option reduction is deterministic
    /// even for duplicate markers, without atomics or cross-thread races.
    #[kernel]
    pub fn marker_gather_backward(
        markers: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < dx.len() {
            let channel = i / sequence;
            let b = channel / hidden;
            let pos = i % sequence;
            let mut sum = 0.0f32;
            let mut j = 0;
            while j < options {
                let selected =
                    marker_position(markers[(b * marker_channels + 2 * j) * sequence], sequence);
                if selected == pos {
                    sum += dy[channel * options + j];
                }
                j += 1;
            }
            // SAFETY: the host validates storage; each thread writes one input gradient.
            unsafe {
                *dx.as_mut_ptr().add(i) = sum;
            }
        }
    }

    /// Reorder and type-condition channel-major LayaHead source features.
    #[kernel]
    pub fn laya_head_prepare(
        input: &[f32],
        type_embedding: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        hidden: usize,
        channels: usize,
        sequence: usize,
        qtype: i32,
    ) {
        let index = thread::index_1d().get();
        if index >= tokens.len() {
            return;
        }
        let b = index / (sequence * hidden);
        let rem = index - b * sequence * hidden;
        let t = rem / hidden;
        let h = rem - t * hidden;
        let selected = if qtype >= 0 {
            qtype as usize
        } else {
            let raw = input[(b * channels + hidden + 1) * sequence];
            (raw.round() as i32).max(0).min(2) as usize
        };
        let value =
            input[(b * channels + h) * sequence + t] + type_embedding[selected * hidden + h];
        // SAFETY: input shape and metadata are checked by the host. Each
        // thread owns one token feature; h=0 owns the per-token mask write.
        unsafe {
            *tokens.as_mut_ptr().add(index) = value;
            if h == 0 {
                *mask.as_mut_ptr().add(b * sequence + t) =
                    input[(b * channels + hidden) * sequence + t];
            }
        }
    }

    /// Pack token-major `[B,T,H]` states into NCHW channel-major `[B,H,1,T]`.
    #[kernel]
    pub fn laya_head_pack(
        tokens: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let b = index / (hidden * sequence);
            let rem = index - b * hidden * sequence;
            let h = rem / sequence;
            let t = rem - h * sequence;
            let source = (b * sequence + t) * hidden + h;
            // SAFETY: equal element counts and the inverse NCHW permutation
            // guarantee source and destination indices are in range.
            unsafe {
                *output.as_mut_ptr().add(index) = tokens[source];
            }
        }
    }

    /// Gather token embeddings and copy the channel-major attention mask.
    #[kernel]
    pub fn laya_encoder_embed(
        input: &[f32],
        embeddings: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        vocab: usize,
    ) {
        let index = thread::index_1d().get();
        if index < tokens.len() {
            let h = index % hidden;
            let row = index / hidden;
            let t = row % sequence;
            let b = row / sequence;
            let raw_id = input[(b * 2) * sequence + t];
            let id = (raw_id.round() as i32).max(0).min(vocab as i32 - 1) as usize;
            // SAFETY: host shape checks validate the input and embedding extents.
            unsafe {
                *tokens.as_mut_ptr().add(index) = embeddings[id * hidden + h];
                if h == 0 {
                    *mask.as_mut_ptr().add(b * sequence + t) = input[(b * 2 + 1) * sequence + t];
                }
            }
        }
    }

    /// Rotary, key-masked online attention with optional local-window restriction.
    #[kernel]
    pub fn laya_encoder_attention(
        qkv: &[f32],
        mask: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
        heads: usize,
        window_radius: usize,
        sliding: bool,
    ) {
        let query_head = thread::index_1d().get();
        let query_heads = output.len() / (hidden / heads);
        if query_head >= query_heads {
            return;
        }
        let head_dim = hidden / heads;
        let half = head_dim / 2;
        let head = query_head % heads;
        let row = query_head / heads;
        let query = row % sequence;
        let batch = row / sequence;
        let head_start = head * head_dim;
        let qrow = row * 3 * hidden + head_start;
        let out_base = row * hidden + head_start;
        // SAFETY: each query/head thread owns this non-overlapping output span.
        unsafe {
            let out_ptr = output.as_mut_ptr().add(out_base);
            let mut d = 0;
            while d < head_dim {
                *out_ptr.add(d) = 0.0;
                d += 1;
            }
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut key = 0;
            while key < sequence {
                if mask[batch * sequence + key] > 0.5
                    && (!sliding
                        || ((query as isize - key as isize).abs() as usize) <= window_radius)
                {
                    let krow = (batch * sequence + key) * 3 * hidden + hidden + head_start;
                    let mut dot = 0.0f32;
                    let mut j = 0;
                    while j < half {
                        let c_q = cos[query * half + j];
                        let s_q = sin[query * half + j];
                        let qa = qkv[qrow + j];
                        let qb = qkv[qrow + j + half];
                        let q0 = qa * c_q - qb * s_q;
                        let q1 = qb * c_q + qa * s_q;
                        let c_k = cos[key * half + j];
                        let s_k = sin[key * half + j];
                        let ka = qkv[krow + j];
                        let kb = qkv[krow + j + half];
                        let k0 = ka * c_k - kb * s_k;
                        let k1 = kb * c_k + ka * s_k;
                        dot += q0 * k0 + q1 * k1;
                        j += 1;
                    }
                    let score = dot * scale;
                    let next_max = running_max.max(score);
                    let alpha = if running_sum == 0.0 {
                        0.0
                    } else {
                        (running_max - next_max).exp()
                    };
                    let probability = (score - next_max).exp();
                    let vrow = (batch * sequence + key) * 3 * hidden + 2 * hidden + head_start;
                    d = 0;
                    while d < head_dim {
                        let old = *out_ptr.add(d);
                        *out_ptr.add(d) = old * alpha + probability * qkv[vrow + d];
                        d += 1;
                    }
                    running_sum = running_sum * alpha + probability;
                    running_max = next_max;
                }
                key += 1;
            }
            if running_sum > 0.0 {
                d = 0;
                while d < head_dim {
                    *out_ptr.add(d) /= running_sum;
                    d += 1;
                }
            }
        }
    }

    /// First-chunk exact-erf GELU multiplied by the raw second GeGLU chunk.
    #[kernel]
    pub fn laya_encoder_geglu(input: &[f32], mut output: DisjointSlice<f32>, intermediate: usize) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let column = index % intermediate;
            let row = index / intermediate;
            let base = row * 2 * intermediate;
            let x = input[base + column];
            let gate = input[base + intermediate + column];
            let value = exact_gelu_value(x);
            // SAFETY: all rows have the validated doubled intermediate extent.
            unsafe {
                *output.as_mut_ptr().add(index) = value * gate;
            }
        }
    }

    /// Pack token-major `[B*T,H]` into channel-major `[B,H,1,T]`.
    #[kernel]
    pub fn laya_encoder_pack(
        tokens: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let batch = index / (hidden * sequence);
            let rem = index - batch * hidden * sequence;
            let h = rem / sequence;
            let t = rem % sequence;
            let source = (batch * sequence + t) * hidden + h;
            // SAFETY: input/output element counts are equal.
            unsafe {
                *output.as_mut_ptr().add(index) = tokens[source];
            }
        }
    }

    /// Unpack channel-major scorer vectors `[B,H,1,K]` into `[B*K,H]` rows.
    #[kernel]
    pub fn laya_scorer_unpack(
        input: &[f32],
        mut rows: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
    ) {
        let index = thread::index_1d().get();
        if index < rows.len() {
            let row = index / hidden;
            let channel = index % hidden;
            let batch = row / options;
            let option = row % options;
            let source = (batch * hidden + channel) * options + option;
            // SAFETY: validated shape extents make both layouts equal-sized.
            unsafe {
                *rows.as_mut_ptr().add(index) = input[source];
            }
        }
    }

    /// Scatter option rows `[B*K,H]` to channel-major `[B,H,1,K]`.
    #[kernel]
    pub fn laya_scorer_pack_input_grad(
        rows: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let batch = index / (hidden * options);
            let rem = index - batch * hidden * options;
            let channel = rem / options;
            let option = rem % options;
            let source = (batch * options + option) * hidden + channel;
            // SAFETY: validated equal element counts bound each transpose index.
            unsafe {
                *output.as_mut_ptr().add(index) = rows[source];
            }
        }
    }

    /// Pack `[B*K]` option logits into `[B,1,1,K]` (identity in storage).
    #[kernel]
    pub fn laya_scorer_pack(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        if index < output.len() {
            // SAFETY: host validation guarantees equal input/output lengths.
            unsafe {
                *output.as_mut_ptr().add(index) = input[index];
            }
        }
    }

    /// LayaHead's masked attention. Skips invalid keys and writes a zero
    /// context for queries whose batch has no valid key. One thread owns one
    /// (batch, head, query) output slice and uses online softmax state.
    #[kernel]
    pub fn laya_head_attention_forward(
        qkv: &[f32],
        mask: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let head_start = h * head_dim;
        let query_row = b * tokens + tq;
        let q_base = query_row * 3 * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut acc = [0.0f32; 128];
        let mut d = 0usize;
        while d < head_dim {
            q[d] = qkv[q_base + d];
            d += 1;
        }
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut any_valid = false;
        let mut key = 0usize;
        while key < tokens {
            if mask[b * tokens + key] > 0.5 {
                any_valid = true;
                let key_row = b * tokens + key;
                let k_base = key_row * 3 * channels + channels + head_start;
                let mut dot = 0.0f32;
                d = 0;
                while d < head_dim {
                    dot += q[d] * qkv[k_base + d];
                    d += 1;
                }
                let score = dot * scale;
                let new_max = running_max.max(score);
                let correction = (running_max - new_max).exp();
                let probability = (score - new_max).exp();
                running_sum = running_sum * correction + probability;
                let v_base = key_row * 3 * channels + 2 * channels + head_start;
                d = 0;
                while d < head_dim {
                    acc[d] = acc[d] * correction + probability * qkv[v_base + d];
                    d += 1;
                }
                running_max = new_max;
            }
            key += 1;
        }
        let output_base = query_row * channels + head_start;
        d = 0;
        while d < head_dim {
            let value = if any_valid { acc[d] / running_sum } else { 0.0 };
            // SAFETY: every (batch, head, query) thread owns disjoint channels.
            unsafe {
                *output.as_mut_ptr().add(output_base + d) = value;
            }
            d += 1;
        }
    }
}
