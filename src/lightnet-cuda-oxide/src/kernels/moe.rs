//! MoE kernels: router logits, softmax top-k, pooling, gate columns and gate gradients.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

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
}
