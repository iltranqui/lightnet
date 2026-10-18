//! Softmax, reorg, dropout and scale_channels (the connected layer goes
//! through the GEMM and bias kernels instead).
//!
//! Every kernel owns its output elements exclusively (no atomics), so results
//! are deterministic.  Semantics follow `src-lib/{softmax,reorg,dropout,
//! scale_channels}_layer.*` and the `darknet-core` CPU runtime; the host
//! references live in `darknet_cuda::host_ops`.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Device twin of `darknet_cuda::dropout_unit`: must stay bit-identical.
#[device]
pub fn dropout_unit(seed: u64, index: u64) -> f32 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0
}

/// Match CPU half-away rounding without f32's (x+0.5) halfway rounding error.
/// Clamp before converting so NaN/infinities cannot poison integer indexing.
#[device]
pub fn marker_position(value: f32, sequence: usize) -> usize {
    let value = value as f64;
    if !(value >= 0.0) {
        0
    } else if value >= (sequence - 1) as f64 {
        sequence - 1
    } else {
        (value + 0.5) as usize
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

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

    /// Copy one contiguous sequence chunk to an independent output tensor.
    #[kernel]
    pub fn sequence_slice_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < output.len() {
            // SAFETY: the host validates the timestep range against input.
            unsafe {
                *output.as_mut_ptr().add(i) = input[offset + i];
            }
        }
    }

    /// Copy one contiguous recurrent timestep into its disjoint output range.
    #[kernel]
    pub fn sequence_concat_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            // SAFETY: host-provided offsets partition the output by timestep.
            unsafe {
                *output.as_mut_ptr().add(offset + i) = input[i];
            }
        }
    }

    /// Exact LSTM cell/gate derivative for one BPTT step. Each thread owns one
    /// hidden element and writes all gate deltas plus the cell gradient for the
    /// preceding step.
    #[kernel]
    pub fn lstm_gate_backward(
        hidden_gradient: &[f32],
        cell: &[f32],
        previous_cell: &[f32],
        forget: &[f32],
        input_gate: &[f32],
        candidate: &[f32],
        output_gate: &[f32],
        next_cell_gradient: &[f32],
        mut forget_gradient: DisjointSlice<f32>,
        mut input_gate_gradient: DisjointSlice<f32>,
        mut candidate_gradient: DisjointSlice<f32>,
        mut output_gate_gradient: DisjointSlice<f32>,
        mut cell_gradient_out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= hidden_gradient.len() {
            return;
        }
        let tanh_cell = cell[i].tanh();
        let output_delta = hidden_gradient[i] * tanh_cell;
        let cell_delta = hidden_gradient[i] * output_gate[i] * (1.0 - tanh_cell * tanh_cell)
            + next_cell_gradient[i];
        let mut forget_delta = cell_delta * previous_cell[i];
        let mut input_delta = cell_delta * candidate[i];
        let mut candidate_delta = cell_delta * input_gate[i];
        forget_delta *= forget[i] * (1.0 - forget[i]);
        input_delta *= input_gate[i] * (1.0 - input_gate[i]);
        candidate_delta *= 1.0 - candidate[i] * candidate[i];
        let mut output_delta = output_delta;
        output_delta *= output_gate[i] * (1.0 - output_gate[i]);
        // SAFETY: each thread writes the matching element in each output.
        unsafe {
            *forget_gradient.as_mut_ptr().add(i) = forget_delta;
            *input_gate_gradient.as_mut_ptr().add(i) = input_delta;
            *candidate_gradient.as_mut_ptr().add(i) = candidate_delta;
            *output_gate_gradient.as_mut_ptr().add(i) = output_delta;
            *cell_gradient_out.as_mut_ptr().add(i) = cell_delta * forget[i];
        }
    }

    /// Global-average pool followed by the MoE router's affine projection. One
    /// output thread computes one (sample, expert) logit; spatial reductions
    /// follow the same sequential f32 order as the CPU reference.
    #[kernel]
    pub fn moe_router_logits(
        input: &[f32],
        router_weights: &[f32],
        router_biases: &[f32],
        mut logits: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        experts: usize,
        batch: usize,
        temperature: f32,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= batch * experts {
            return;
        }
        let b = linear / experts;
        let e = linear % experts;
        let inv_spatial = 1.0 / spatial as f32;
        let temperature = if temperature > 1e-6 {
            temperature
        } else {
            1e-6
        };
        let mut value = router_biases[e];
        let mut c = 0usize;
        while c < channels {
            let base = (b * channels + c) * spatial;
            let mut sum = 0.0f32;
            let mut s = 0usize;
            while s < spatial {
                sum += input[base + s];
                s += 1;
            }
            let pooled = sum * inv_spatial;
            value += router_weights[e * channels + c] * pooled;
            c += 1;
        }
        if let Some(out) = logits.get_mut(idx) {
            *out = value / temperature;
        }
    }

    /// Top-k masked softmax. A single thread owns each row; counting strictly
    /// greater logits reproduces C++'s `v < kth` mask, including ties.
    #[kernel]
    pub fn moe_router_softmax_topk(
        logits: &[f32],
        mut gates: DisjointSlice<f32>,
        batch: usize,
        experts: usize,
        top_k: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batch {
            return;
        }
        let base = b * experts;
        let mut maximum = -f32::MAX;
        let mut e = 0usize;
        while e < experts {
            let value = logits[base + e];
            let mut keep = true;
            if top_k > 0 && top_k < experts {
                let mut greater = 0usize;
                let mut j = 0usize;
                while j < experts {
                    if logits[base + j] > value {
                        greater += 1;
                    }
                    j += 1;
                }
                keep = greater < top_k;
            }
            let masked = if keep { value } else { -f32::MAX };
            if masked > maximum {
                maximum = masked;
            }
            e += 1;
        }

        let mut sum = 0.0f32;
        e = 0;
        while e < experts {
            let value = logits[base + e];
            let mut keep = true;
            if top_k > 0 && top_k < experts {
                let mut greater = 0usize;
                let mut j = 0usize;
                while j < experts {
                    if logits[base + j] > value {
                        greater += 1;
                    }
                    j += 1;
                }
                keep = greater < top_k;
            }
            let masked = if keep { value } else { -f32::MAX };
            let exp = if masked <= -f32::MAX * 0.5 {
                0.0
            } else {
                (masked - maximum).exp()
            };
            // SAFETY: this thread uniquely owns the `experts`-element gate row.
            unsafe {
                *gates.as_mut_ptr().add(base + e) = exp;
            }
            sum += exp;
            e += 1;
        }

        if sum <= 0.0 || !sum.is_finite() {
            let uniform = 1.0 / experts as f32;
            e = 0;
            while e < experts {
                // SAFETY: this thread uniquely owns the `experts`-element gate row.
                unsafe {
                    *gates.as_mut_ptr().add(base + e) = uniform;
                }
                e += 1;
            }
        } else {
            e = 0;
            while e < experts {
                // SAFETY: this thread uniquely owns the `experts`-element gate row.
                unsafe {
                    *gates.as_mut_ptr().add(base + e) /= sum;
                }
                e += 1;
            }
        }
    }

    /// Global-average pool in NCHW storage to `[batch, channels]`; one thread
    /// sums the spatial plane for one sample/channel in CPU reference order.
    #[kernel]
    pub fn moe_router_pool(
        input: &[f32],
        mut pooled: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        batch: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= batch * channels {
            return;
        }
        let base = linear * spatial;
        let mut sum = 0.0f32;
        let mut s = 0usize;
        while s < spatial {
            sum += input[base + s];
            s += 1;
        }
        if let Some(out) = pooled.get_mut(idx) {
            *out = sum * (1.0 / spatial as f32);
        }
    }

    /// Broadcast the derivative of a global-average pooled descriptor.
    #[kernel]
    pub fn moe_router_pool_backward(
        pooled_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = input_gradient.get_mut(idx) {
            *out = pooled_gradient[i / spatial] * (1.0 / spatial as f32);
        }
    }

    /// Extract one expert's gate from the row-major `[batch, experts]` tensor.
    #[kernel]
    pub fn moe_gate_column(
        gates: &[f32],
        mut column: DisjointSlice<f32>,
        experts: usize,
        expert: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        if let Some(out) = column.get_mut(idx) {
            *out = gates[b * experts + expert];
        }
    }

    /// Add the exact auxiliary load-balance derivative and top-k softmax Jacobian.
    /// One thread owns a batch row, preserving the CPU's ordered expert reduction.
    #[kernel]
    pub fn moe_router_gate_backward(
        gates: &[f32],
        gate_dots: &[f32],
        mut dlogit: DisjointSlice<f32>,
        batch: usize,
        experts: usize,
        load_balance: f32,
        temperature: f32,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        if b >= batch {
            return;
        }
        let base = b * experts;
        let balance_scale = if load_balance <= 0.0 {
            0.0
        } else {
            load_balance * 2.0 * experts as f32 / batch as f32
        };
        let temperature = if temperature > 1e-6 {
            temperature
        } else {
            1e-6
        };
        let mut weighted = 0.0f32;
        let mut e = 0usize;
        while e < experts {
            let mut mean_gate = 0.0f32;
            let mut sample = 0usize;
            while sample < batch {
                mean_gate += gates[sample * experts + e];
                sample += 1;
            }
            mean_gate /= batch as f32;
            let raw = gate_dots[base + e] + balance_scale * mean_gate;
            weighted += gates[base + e] * raw;
            e += 1;
        }
        e = 0;
        while e < experts {
            let mut mean_gate = 0.0f32;
            let mut sample = 0usize;
            while sample < batch {
                mean_gate += gates[sample * experts + e];
                sample += 1;
            }
            mean_gate /= batch as f32;
            let raw = gate_dots[base + e] + balance_scale * mean_gate;
            // SAFETY: this thread uniquely owns the expert row for batch item `b`.
            unsafe {
                *dlogit.as_mut_ptr().add(base + e) =
                    gates[base + e] * (raw - weighted) / temperature;
            }
            e += 1;
        }
    }

    /// Stage 1 of per-sample gate dot: 256 strided partials per sample.
    #[kernel]
    pub fn moe_gate_dot_partial(
        output_gradient: &[f32],
        expert_output: &[f32],
        mut partial: DisjointSlice<f32>,
        per_image: usize,
        batch: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let sample = linear / 256;
        let lane = linear % 256;
        if sample >= batch {
            return;
        }
        let base = sample * per_image;
        let mut sum = 0.0f32;
        let mut i = lane;
        while i < per_image {
            sum += output_gradient[base + i] * expert_output[base + i];
            i += 256;
        }
        if let Some(out) = partial.get_mut(idx) {
            *out = sum;
        }
    }

    /// Stage 2: fold 256 partials into one gate derivative per sample.
    #[kernel]
    pub fn moe_gate_dot_reduce(partial: &[f32], mut dots: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let sample = idx.get();
        if let Some(out) = dots.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut lane = 0usize;
            while lane < 256 {
                sum += partial[sample * 256 + lane];
                lane += 1;
            }
            *out = sum;
        }
    }

    /// Fold 256 gate-dot partials into one expert column of `[batch, experts]`.
    #[kernel]
    pub fn moe_gate_dot_reduce_column(
        partial: &[f32],
        mut dots: DisjointSlice<f32>,
        experts: usize,
        expert: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        let output_index = b * experts + expert;
        if output_index >= dots.len() {
            return;
        }
        let mut sum = 0.0f32;
        let mut lane = 0usize;
        while lane < 256 {
            sum += partial[b * 256 + lane];
            lane += 1;
        }
        // SAFETY: this launch owns one column element per batch row; callers
        // launch experts serially and each launch writes a disjoint matrix column.
        unsafe {
            *dots.as_mut_ptr().add(output_index) = sum;
        }
    }

    /// `out[b * to_len + i] = i < min(from_len, to_len) ? x[b * from_len + i] : 0`
    /// (flat per-sample crop / zero-pad, the Darknet mismatched shortcut).
    #[kernel]
    pub fn shortcut_fit(x: &[f32], mut out: DisjointSlice<f32>, from_len: usize, to_len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let b = g / to_len;
            let i = g - b * to_len;
            *o = if i < from_len {
                x[b * from_len + i]
            } else {
                0.0
            };
        }
    }

    /// One query row per thread. Online max/sum and the value accumulator keep
    /// the attention workspace linear in sequence length; the host rejects
    /// head dimensions above the fixed local accumulator size.
    #[kernel]
    pub fn flash_attention_forward(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
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
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut acc = [0.0f32; 128];
        let q_base = query_row * 3 * channels + head_start;
        let mut d = 0usize;
        while d < head_dim {
            let raw = qkv[q_base + d];
            q[d] = if raw.is_finite() {
                raw.max(-128.0).min(128.0)
            } else {
                0.0
            };
            d += 1;
        }

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let mut dot = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw = qkv[k_base + d];
                let k = if raw.is_finite() {
                    raw.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                d += 1;
            }
            let raw_score = dot * scale;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let new_max = running_max.max(score);
            let correction = (running_max - new_max).exp();
            let p = (score - new_max).exp();
            running_sum = running_sum * correction + p;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            d = 0;
            while d < head_dim {
                acc[d] = acc[d] * correction + p * qkv[v_base + d];
                d += 1;
            }
            running_max = new_max;
            tk += 1;
        }

        let output_base = query_row * channels + head_start;
        d = 0;
        while d < head_dim {
            // SAFETY: each (batch, head, query) thread owns this head's disjoint
            // output channel interval for one query row.
            unsafe {
                *out.as_mut_ptr().add(output_base + d) = acc[d] / running_sum;
            }
            d += 1;
        }
        // LSE is flattened as `[batch, heads, tokens]`, matching `query`.
        unsafe {
            *lse.as_mut_ptr().add(query) = running_max + running_sum.ln();
        }
    }

    /// One query row per thread computes dQ and D_i = dot(context_i, dO_i).
    /// The following key-row kernel consumes D_i to form dK and dV without atomics.
    #[kernel]
    pub fn flash_attention_backward_dq(
        qkv: &[f32],
        context: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        mut dqkv: DisjointSlice<f32>,
        mut row_dot: DisjointSlice<f32>,
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
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let q_base = query_row * 3 * channels + head_start;
        let row_base = query_row * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut dq = [0.0f32; 128];
        let mut row_product = 0.0f32;
        let mut d = 0usize;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            q[d] = if raw_q.is_finite() {
                raw_q.max(-128.0).min(128.0)
            } else {
                0.0
            };
            row_product += output_grad[row_base + d] * context[row_base + d];
            dq[d] = 0.0;
            d += 1;
        }

        let lse_i = lse[query];
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_k = qkv[k_base + d];
                let k = if raw_k.is_finite() {
                    raw_k.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                dprob += output_grad[row_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse_i).exp();
            if active {
                let dscore = p * (dprob - row_product);
                d = 0;
                while d < head_dim {
                    let raw_k = qkv[k_base + d];
                    let k = if raw_k.is_finite() {
                        raw_k.max(-128.0).min(128.0)
                    } else {
                        0.0
                    };
                    dq[d] += dscore * k * scale;
                    d += 1;
                }
            }
            tk += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            let grad = if raw_q.is_finite() && raw_q.abs() < 128.0 {
                dq[d]
            } else {
                0.0
            };
            // SAFETY: each query/head thread owns this Q channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(query_row * 3 * channels + head_start + d) = grad;
            }
            d += 1;
        }
        // SAFETY: one thread owns each flattened (batch, head, query) scalar.
        unsafe {
            *row_dot.as_mut_ptr().add(query) = row_product;
        }
    }

    /// One key row per thread accumulates dK/dV across all query rows; every output
    /// row is owned by one thread, so the recompute path requires no atomics.
    #[kernel]
    pub fn flash_attention_backward_dkdv(
        qkv: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        row_dot: &[f32],
        mut dqkv: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let key = thread::index_1d().get();
        let key_count = batch * heads * tokens;
        if key >= key_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = key / tokens;
        let tk = key - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let key_row = b * tokens + tk;
        let head_start = h * head_dim;
        let k_base = key_row * 3 * channels + channels + head_start;
        let v_base = key_row * 3 * channels + 2 * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut k = [0.0f32; 128];
        let mut dk = [0.0f32; 128];
        let mut dv = [0.0f32; 128];
        let mut d = 0usize;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            k[d] = if raw_k.is_finite() {
                raw_k.max(-128.0).min(128.0)
            } else {
                0.0
            };
            dk[d] = 0.0;
            dv[d] = 0.0;
            d += 1;
        }

        let mut tq = 0usize;
        while tq < tokens {
            let query_row = b * tokens + tq;
            let query = (b * heads + h) * tokens + tq;
            let q_base = query_row * 3 * channels + head_start;
            let dy_base = query_row * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q * k[d];
                dprob += output_grad[dy_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse[query]).exp();
            let dscore = p * (dprob - row_dot[query]);
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dv[d] += p * output_grad[dy_base + d];
                if active {
                    dk[d] += dscore * q * scale;
                }
                d += 1;
            }
            tq += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            let dkey = if raw_k.is_finite() && raw_k.abs() < 128.0 {
                dk[d]
            } else {
                0.0
            };
            // SAFETY: each key/head thread owns one K and one V channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + channels + head_start + d) = dkey;
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + 2 * channels + head_start + d) = dv[d];
            }
            d += 1;
        }
    }

    /// One thread per softmax group of `group_size` contiguous elements
    /// (`n_groups = batch * groups`): max, exp + sum, normalise.
    #[kernel]
    pub fn softmax_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut max = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < group_size {
            let v = x[base + i];
            if v > max {
                max = v;
            }
            i += 1;
        }
        let mut sum = 0.0f32;
        i = 0;
        while i < group_size {
            let e = (x[base + i] - max).exp();
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`;
            // the host checks `out` covers `n_groups * group_size`.
            unsafe {
                *out.as_mut_ptr().add(base + i) = e;
            }
            sum += e;
            i += 1;
        }
        if sum > 0.0 {
            i = 0;
            while i < group_size {
                // SAFETY: as above; this thread already wrote the element.
                unsafe {
                    *out.as_mut_ptr().add(base + i) /= sum;
                }
                i += 1;
            }
        }
    }

    /// `dx = y * (dy - sum(y * dy))`, one thread per group.
    #[kernel]
    pub fn softmax_backward(
        y: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut dot = 0.0f32;
        let mut i = 0usize;
        while i < group_size {
            dot += y[base + i] * dy[base + i];
            i += 1;
        }
        i = 0;
        while i < group_size {
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`.
            unsafe {
                *dx.as_mut_ptr().add(base + i) = y[base + i] * (dy[base + i] - dot);
            }
            i += 1;
        }
    }

    /// Big `[b, cb, hb, wb]` -> small `[b, cb*s*s, hs, ws]`: one thread per
    /// small element gathers its big source.
    #[kernel]
    pub fn reorg_to_small(
        big: &[f32],
        mut small: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= small.len() {
            return;
        }
        let cs = cb * stride * stride;
        let plane = hs * ws;
        let bk = i / plane;
        let r = i % plane;
        let b = bk / cs;
        let k = bk % cs;
        let c2 = k % cb;
        let off = k / cb;
        let h2 = (r / ws) * stride + off / stride;
        let w2 = (r % ws) * stride + off % stride;
        if let Some(o) = small.get_mut(idx) {
            *o = big[((b * cb + c2) * hb + h2) * wb + w2];
        }
    }

    /// Small -> big: one thread per big element reads its small source, or
    /// writes 0 on the border not covered when `hb % stride != 0`.
    #[kernel]
    pub fn reorg_to_big(
        small: &[f32],
        mut big: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= big.len() {
            return;
        }
        let plane = hb * wb;
        let bc = i / plane;
        let r = i % plane;
        let b = bc / cb;
        let c2 = bc % cb;
        let h2 = r / wb;
        let w2 = r % wb;
        let j = h2 / stride;
        let x = w2 / stride;
        let value = if j >= hs || x >= ws {
            0.0
        } else {
            let off = (h2 % stride) * stride + w2 % stride;
            let k = off * cb + c2;
            small[((b * cb * stride * stride + k) * hs + j) * ws + x]
        };
        if let Some(o) = big.get_mut(idx) {
            *o = value;
        }
    }

    /// `out[b, j*rows + i, s] = x[b, i*cols + j, s]`: one thread per output element.
    #[kernel]
    pub fn channel_permute(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        len: usize,
    ) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let per = rows * cols * len;
            let b = g / per;
            let r = g % per;
            let p = r / len; // output channel j*rows + i
            let s = r % len;
            let j = p / rows;
            let i = p % rows;
            *o = x[b * per + (i * cols + j) * len + s];
        }
    }

    /// `mask[i] = 0` when `dropout_unit(seed, i) < probability` else `scale`;
    /// `out[i] = x[i] * mask[i]`.
    #[kernel]
    pub fn dropout_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        probability: f32,
        scale: f32,
        seed: u64,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = if dropout_unit(seed, i as u64) < probability {
            0.0
        } else {
            scale
        };
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }

    /// `dx[i] = dy[i] * mask[i]`.
    #[kernel]
    pub fn dropout_backward(dy: &[f32], mask: &[f32], mut dx: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = dx.get_mut(idx) {
            *o = dy[i] * mask[i];
        }
    }

    /// DropBlock stage 1, one thread per `(sample, channel)` plane: picks the
    /// cell with the lowest quantised random value below `prob`
    /// (`darknet_cuda::host_ops::dropblock_corner`), writes the plane of
    /// `keep` (`1`, `0` inside the block) and `dropped[plane]` (`bs^2` or 0).
    #[kernel]
    pub fn dropblock_plane(
        mut keep: DisjointSlice<f32>,
        mut dropped: DisjointSlice<f32>,
        seed: u64,
        w: usize,
        h: usize,
        prob: f32,
        bs: usize,
    ) {
        let idx = thread::index_1d();
        let plane = idx.get();
        let spatial = w * h;
        if plane >= dropped.len() {
            return;
        }
        let mut best = 1_000_000i32;
        let mut cell = usize::MAX;
        let mut i = 0usize;
        while i < spatial {
            let u = dropout_unit(seed, (plane * spatial + i) as u64);
            if u < prob {
                let q = (u * 1_000_000.0) as i32;
                if q < best {
                    best = q;
                    cell = i;
                }
            }
            i += 1;
        }
        let mut bx = 0usize;
        let mut by = 0usize;
        if cell != usize::MAX {
            bx = cell % w;
            by = cell / w;
            if bx > w - bs {
                bx -= w - bs;
            }
            if by > h - bs {
                by -= h - bs;
            }
            if bx > w - bs {
                bx = w - bs;
            }
            if by > h - bs {
                by = h - bs;
            }
        }
        // SAFETY: this thread owns plane `plane` of `keep` (`spatial` cells)
        // and slot `plane` of `dropped`; the host allocates both exactly.
        unsafe {
            let base = keep.as_mut_ptr().add(plane * spatial);
            let mut j = 0usize;
            while j < spatial {
                let y = j / w;
                let x = j % w;
                let inside = cell != usize::MAX && x >= bx && x < bx + bs && y >= by && y < by + bs;
                *base.add(j) = if inside { 0.0 } else { 1.0 };
                j += 1;
            }
            *dropped.as_mut_ptr().add(plane) = if cell != usize::MAX {
                (bs * bs) as f32
            } else {
                0.0
            };
        }
    }

    /// DropBlock stage 2, one thread per sample: sums `dropped` over the
    /// sample's `channels` planes in order and writes
    /// `host_ops::dropblock_scale(sum, outputs)`.
    #[kernel]
    pub fn dropblock_scales(
        dropped: &[f32],
        mut scales: DisjointSlice<f32>,
        channels: usize,
        outputs: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        let mut sum = 0.0f32;
        let mut c = 0usize;
        while c < channels && b * channels + c < dropped.len() {
            sum += dropped[b * channels + c];
            c += 1;
        }
        let keep = 1.0 - sum / outputs as f32;
        if let Some(o) = scales.get_mut(idx) {
            *o = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        }
    }

    /// DropBlock stage 3, one thread per element: `mask = keep * scale[b]`,
    /// `out = x * mask`; `per_batch = channels * h * w`.
    #[kernel]
    pub fn dropblock_apply(
        x: &[f32],
        keep: &[f32],
        scales: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        per_batch: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = keep[i] * scales[i / per_batch];
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }

    /// Darknet `constrain_ongpu`: `out = min(limit, max(-limit, x))`, NaN -> `-limit`.
    #[kernel]
    pub fn constrain(x: &[f32], mut out: DisjointSlice<f32>, limit: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            let mut v = x[i];
            if !(v > -limit) {
                v = -limit;
            }
            if v > limit {
                v = limit;
            }
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Darknet `gradient_centralization_gpu`, one thread per filter of
    /// `f_size` contiguous values (sequential sum, = the host reference); a
    /// plain copy when `enabled == 0`.
    #[kernel]
    pub fn gradient_centralize(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        f_size: usize,
        filters: usize,
        enabled: u32,
    ) {
        let f = thread::index_1d().get();
        if f >= filters {
            return;
        }
        let base = f * f_size;
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < f_size {
            sum += x[base + i];
            i += 1;
        }
        let mean = if enabled != 0 {
            sum / f_size as f32
        } else {
            0.0
        };
        // SAFETY: this thread owns the `f_size` values of filter `f`.
        unsafe {
            let dst = out.as_mut_ptr().add(base);
            let mut i = 0usize;
            while i < f_size {
                *dst.add(i) = x[base + i] - mean;
                i += 1;
            }
        }
    }

    /// `out[i] = scale[si(i)] * src[i]`, `si` = `[batch, hw]` (`scale_wh != 0`)
    /// or `[batch, channels]` index of element `i`.
    #[kernel]
    pub fn scale_channels_forward(
        scale: &[f32],
        src: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = out.get_mut(idx) {
            *o = scale[si] * src[i];
        }
    }

    /// `d_source[i] = delta[i] * scale[si(i)]`.
    #[kernel]
    pub fn scale_channels_dsource(
        delta: &[f32],
        scale: &[f32],
        mut d_src: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = d_src.get_mut(idx) {
            *o = delta[i] * scale[si];
        }
    }

    /// `d_scale[j] = sum delta * src` over the elements sharing scale `j`;
    /// one thread per scale element, fixed summation order.
    #[kernel]
    pub fn scale_channels_dscale(
        delta: &[f32],
        src: &[f32],
        mut d_scale: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= d_scale.len() {
            return;
        }
        let mut sum = 0.0f32;
        if scale_wh != 0 {
            let b = j / hw;
            let s = j % hw;
            let mut c = 0usize;
            while c < channels {
                let i = (b * channels + c) * hw + s;
                sum += delta[i] * src[i];
                c += 1;
            }
        } else {
            let base = j * hw;
            let mut s = 0usize;
            while s < hw {
                sum += delta[base + s] * src[base + s];
                s += 1;
            }
        }
        if let Some(o) = d_scale.get_mut(idx) {
            *o = sum;
        }
    }

    /// One thread per shortcut coefficient slot. Each thread traverses its
    /// deterministic batch/feature range and reduces `delta * source`; a short
    /// source contributes only over its available flat prefix.
    #[kernel]
    pub fn shortcut_weight_gradient(
        delta: &[f32],
        source: &[f32],
        mut gradient: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
    ) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if slot >= layer_step {
            return;
        }
        let span = per_sample / layer_step;
        let start = slot * span;
        let end = (start + span).min(per_sample).min(source_per_sample);
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut e = start;
            while e < end {
                sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                e += 1;
            }
            b += 1;
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = sum;
        }
    }

    /// Update one source row of `[input][slot]` `delta * source` products.
    #[kernel]
    pub fn shortcut_weight_gradient_accumulate(
        accumulated: &[f32],
        delta: &[f32],
        source: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let mut value = accumulated[i];
        if i / layer_step == source_index && source_index < num_inputs {
            let slot = i % layer_step;
            let span = per_sample / layer_step;
            let start = slot * span;
            let end = (start + span).min(per_sample).min(source_per_sample);
            let mut sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                let mut e = start;
                while e < end {
                    sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                    e += 1;
                }
                b += 1;
            }
            value += sum;
        }
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    /// Exact per-slot normalization chain for shortcut coefficient gradients.
    /// `normalization`: 0 none, 1 relu, 2 softmax; `cpp_grad` copies raw z.
    #[kernel]
    pub fn shortcut_weight_normalized_gradient(
        raw: &[f32],
        products: &[f32],
        mut gradient: DisjointSlice<f32>,
        num_inputs: usize,
        layer_step: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let slot = i % layer_step;
        let mut value = products[i];
        if normalization == 1 && cpp_grad == 0 {
            let mut positive_sum = 0.0f32;
            let mut dot = 0.0f32;
            let mut input = 0usize;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let positive = raw[j].max(0.0);
                positive_sum += positive;
                dot += products[j] * positive;
                input += 1;
            }
            let sum = 0.0001f32 + positive_sum;
            value = if raw[i] > 0.0 {
                products[i] / sum - dot / (sum * sum)
            } else {
                0.0
            };
        } else if normalization == 2 && cpp_grad == 0 {
            let mut maximum = f32::NEG_INFINITY;
            let mut input = 0usize;
            while input < num_inputs {
                let raw_value = raw[input * layer_step + slot];
                if raw_value > maximum || input == 0 {
                    maximum = raw_value;
                }
                input += 1;
            }
            let mut exp_sum = 0.0f32;
            input = 0;
            while input < num_inputs {
                exp_sum += (raw[input * layer_step + slot] - maximum).exp();
                input += 1;
            }
            let sum = 0.0001f32 + exp_sum;
            let mut dot = 0.0f32;
            input = 0;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let effective = (raw[j] - maximum).exp() / sum;
                dot += products[j] * effective;
                input += 1;
            }
            let effective = (raw[i] - maximum).exp() / sum;
            value = effective * (products[i] - dot);
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }

    /// NCHW channel normalization at one `(batch, spatial)` position per thread.
    /// `normalization`: 0 ReLU-normalize, 1 softmax with zero shift, 2 softmax
    /// with channel-max subtraction.
    #[kernel]
    pub fn normalize_channels(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let value = if normalization == 0 {
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                let x = input[base + k * spatial];
                if x > 0.0 {
                    sum += x;
                }
                k += 1;
            }
            let x = input[i];
            if x > 0.0 { x / sum } else { 0.0 }
        } else {
            let maximum = if normalization == 2 {
                let mut maximum = -f32::MAX;
                let mut k = 0usize;
                while k < channels {
                    let x = input[base + k * spatial];
                    if x > maximum || k == 0 {
                        maximum = x;
                    }
                    k += 1;
                }
                maximum
            } else {
                0.0
            };
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                sum += (input[base + k * spatial] - maximum).exp();
                k += 1;
            }
            let value = (input[i] - maximum).exp() / sum;
            if value.is_finite() { value } else { 0.0 }
        };
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    #[device]
    fn eml_softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            x.exp().ln_1p()
        }
    }

    #[device]
    fn eml_sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    /// EML convolution's elementwise combination of its parallel conv outputs.
    #[kernel]
    pub fn eml_convolutional_forward(
        x_branch: &[f32],
        self_branch: &[f32],
        residual: &[f32],
        mut output: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
        has_residual: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = x_branch[i].max(-clamp).min(clamp);
        let y = self_branch[i];
        let mut value = scale * (x.exp() - (eml_softplus(y) + eps).ln());
        if has_residual != 0 {
            value += residual[i];
        }
        if let Some(out) = output.get_mut(idx) {
            *out = value;
        }
    }

    /// Exact EML branch adjoints, with the source's strict clamp gate.
    #[kernel]
    pub fn eml_convolutional_backward(
        x_branch: &[f32],
        self_branch: &[f32],
        output_grad: &[f32],
        mut x_gradient: DisjointSlice<f32>,
        mut self_gradient: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x_gradient.len() {
            return;
        }
        let x_value = x_branch[i];
        let dx = if x_value > -clamp && x_value < clamp {
            x_value.max(-clamp).min(clamp).exp()
        } else {
            0.0
        };
        let y = self_branch[i];
        let dy = -eml_sigmoid(y) / (eml_softplus(y) + eps);
        let scaled_delta = output_grad[i] * scale;
        // SAFETY: every thread writes only element `i` in both output arrays.
        unsafe {
            *x_gradient.as_mut_ptr().add(i) = scaled_delta * dx;
            *self_gradient.as_mut_ptr().add(i) = scaled_delta * dy;
        }
    }

    /// Exact channel-normalization adjoint or the C++ CUDA diagonal shortcut.
    #[kernel]
    pub fn normalize_channels_backward(
        activation_input: &[f32],
        output: &[f32],
        delta: &[f32],
        mut gradient: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let y = output[i];
        let value = if cpp_grad != 0 {
            if normalization == 0 {
                if y > 0.0 { delta[i] * y } else { delta[i] }
            } else {
                let value = delta[i] * y * (1.0 - y);
                if value.is_finite() { value } else { 0.0 }
            }
        } else {
            let mut dot = 0.0f32;
            let mut k = 0usize;
            while k < channels {
                let j = base + k * spatial;
                dot += output[j] * delta[j];
                k += 1;
            }
            if normalization == 0 {
                let mut sum = 0.0001f32;
                k = 0;
                while k < channels {
                    let x = activation_input[base + k * spatial];
                    if x > 0.0 {
                        sum += x;
                    }
                    k += 1;
                }
                if activation_input[i] > 0.0 {
                    (delta[i] - dot) / sum
                } else {
                    0.0
                }
            } else {
                y * (delta[i] - dot)
            }
        };
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }
}

/// Fused residual + bias epilogue for a transformer block:
/// `out[i] = residual[i] + y[i] + bias[i % cols]`.
///
/// The bias is per **column**: the row index is the token, so it must not shift
/// the offset.  `has_bias = 0` makes it a plain residual add.
#[cuda_module]
pub mod residual {
    use super::*;
    use cuda_device::{kernel, thread};

    #[kernel]
    pub fn residual_bias(
        residual: &[f32],
        y: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        cols: usize,
        has_bias: u32,
    ) {
        let i = thread::index_1d();
        let idx = i.get();
        if idx >= out.len() {
            return;
        }
        let b = if has_bias != 0 { bias[idx % cols] } else { 0.0 };
        if let Some(slot) = out.get_mut(i) {
            *slot = residual[idx] + y[idx] + b;
        }
    }
}
