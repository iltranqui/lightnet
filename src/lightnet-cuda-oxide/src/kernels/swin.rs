//! Swin kernels: padded/shifted window partition and its inverse, the plain
//! two-stream mHC mix, windowed attention with relative-position bias and the
//! additive shift/padding mask, and the adjoints the native backward needs.
//! Every kernel mirrors `lightnet_core::swin` / `swin::block` index for index.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Source `constrain`: NaN maps to `-limit`, everything else clamps.
    #[device]
    fn swin_constrain(value: f32, limit: f32) -> f32 {
        if value.is_nan() {
            -limit
        } else if value > limit {
            limit
        } else if value < -limit {
            -limit
        } else {
            value
        }
    }

    /// NCHW `[batch, channels, height, width]` to raster `[batch*nW*T, channels]`
    /// window rows, fusing the bottom/right zero padding to `padded_*` and the
    /// cyclic shift `output[y, x] = padded[(y + shift) % Hp, (x + shift) % Wp]`.
    /// The same gather is the adjoint of `swin_window_unpartition`.
    #[kernel]
    pub fn swin_window_partition(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let tokens = window * window;
        let windows_w = padded_width / window;
        let windows = (padded_height / window) * windows_w;
        let channel = i % channels;
        let row = i / channels;
        let token = row % tokens;
        let window_index = row / tokens;
        let batch = window_index / windows;
        let spatial = window_index % windows;
        let y = (spatial / windows_w) * window + token / window;
        let x = (spatial % windows_w) * window + token % window;
        let source_y = (y + shift) % padded_height;
        let source_x = (x + shift) % padded_width;
        let value = if source_y < height && source_x < width {
            input[((batch * channels + channel) * height + source_y) * width + source_x]
        } else {
            0.0
        };
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Window rows back to cropped NCHW: `output[y, x]` reads the window
    /// element at padded position `((y - shift) % Hp, (x - shift) % Wp)`.
    /// `clip > 0` applies the source output constrain; `clip == 0` is the
    /// plain gather used as the adjoint of `swin_window_partition`.
    #[kernel]
    pub fn swin_window_unpartition(
        windows: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        clip: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = i % width;
        let y = (i / width) % height;
        let channel = (i / (width * height)) % channels;
        let batch = i / (width * height * channels);
        let tokens = window * window;
        let windows_w = padded_width / window;
        let spatial_windows = (padded_height / window) * windows_w;
        let padded_y = (y + padded_height - shift) % padded_height;
        let padded_x = (x + padded_width - shift) % padded_width;
        let window_index =
            batch * spatial_windows + (padded_y / window) * windows_w + padded_x / window;
        let token = (padded_y % window) * window + padded_x % window;
        let mut value = windows[(window_index * tokens + token) * channels + channel];
        if clip > 0.0 {
            value = swin_constrain(value, clip);
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Unclamped two-stream mHC mix `skip_scale * skip + branch_scale * branch`.
    #[kernel]
    pub fn swin_mhc_residual(
        skip: &[f32],
        branch: &[f32],
        mut output: DisjointSlice<f32>,
        skip_scale: f32,
        branch_scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let value = skip_scale * skip[i] + branch_scale * branch[i];
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Input adjoints of `swin_mhc_residual`.
    #[kernel]
    pub fn swin_mhc_residual_backward(
        grad: &[f32],
        mut d_skip: DisjointSlice<f32>,
        mut d_branch: DisjointSlice<f32>,
        skip_scale: f32,
        branch_scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= grad.len() {
            return;
        }
        let g = grad[i];
        if let Some(dst) = d_skip.get_mut(thread::index_1d()) {
            *dst = g * skip_scale;
        }
        if let Some(dst) = d_branch.get_mut(thread::index_1d()) {
            *dst = g * branch_scale;
        }
    }

    /// Per-element terms of the three mHC scalar reductions of
    /// `swin::mhc_residual_backward`, in planar `[3, len]` order:
    /// `g*(p*s+(1-p)*b)`, `g*((1-p)*s+p*b)` and `g*(post_skip-post_branch)*(s-b)`.
    #[kernel]
    pub fn swin_mhc_param_terms(
        skip: &[f32],
        branch: &[f32],
        grad: &[f32],
        mut terms: DisjointSlice<f32>,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let len = grad.len();
        if j >= terms.len() {
            return;
        }
        let plane = j / len;
        let i = j % len;
        let s = skip[i];
        let b = branch[i];
        let g = grad[i];
        let value = if plane == 0 {
            g * (mix * s + (1.0 - mix) * b)
        } else if plane == 1 {
            g * ((1.0 - mix) * s + mix * b)
        } else {
            g * (post_skip - post_branch) * (s - b)
        };
        if let Some(dst) = terms.get_mut(idx) {
            *dst = value;
        }
    }

    /// Windowed multi-head attention, one `(window, head, query)` row per
    /// thread, as `swin::window_attention_forward_with_indices`: Q/K clamp to
    /// +-256, `score = dot*scale + bias[rel] + mask`, clamp to +-20, stabilised
    /// softmax with the `1e-9` denominator epsilon. Writes the head's context
    /// channels, the probability row and the score-clamp activity row (1/0).
    #[kernel]
    pub fn swin_window_attention_forward(
        qkv: &[f32],
        relative_position_bias: &[f32],
        mask: &[f32],
        mut context: DisjointSlice<f32>,
        mut probabilities: DisjointSlice<f32>,
        mut active: DisjointSlice<f32>,
        total_windows: usize,
        spatial_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
    ) {
        let thread_index = thread::index_1d().get();
        let tokens = window * window;
        if thread_index >= total_windows * heads * tokens {
            return;
        }
        let head_dim = channels / heads;
        let query = thread_index % tokens;
        let window_head = thread_index / tokens;
        let head = window_head % heads;
        let window_index = window_head / heads;
        let mask_base = ((window_index % spatial_windows) * tokens + query) * tokens;
        let table_width = 2 * window - 1;
        let table_len = table_width * table_width;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let row_base = thread_index * tokens;
        let query_base = (window_index * tokens + query) * 3 * channels + head * head_dim;
        let query_y = query / window;
        let query_x = query % window;
        let probability = probabilities.as_mut_ptr();
        let activity = active.as_mut_ptr();

        let mut max_score = f32::NEG_INFINITY;
        let mut key = 0usize;
        while key < tokens {
            let key_base =
                (window_index * tokens + key) * 3 * channels + channels + head * head_dim;
            let mut dot = 0.0f32;
            let mut d = 0usize;
            while d < head_dim {
                dot += swin_constrain(qkv[query_base + d], 256.0)
                    * swin_constrain(qkv[key_base + d], 256.0);
                d += 1;
            }
            let dy = query_y + window - 1 - key / window;
            let dx = query_x + window - 1 - key % window;
            let bias = relative_position_bias[head * table_len + dy * table_width + dx];
            let raw = (dot * scale + bias) + mask[mask_base + key];
            let score = swin_constrain(raw, 20.0);
            let live = raw.is_finite() && raw > -20.0 && raw < 20.0;
            // SAFETY: this thread owns row `row_base..row_base + tokens` of
            // both `[windows, heads, tokens, tokens]` outputs.
            unsafe {
                *probability.add(row_base + key) = score;
                *activity.add(row_base + key) = if live { 1.0 } else { 0.0 };
            }
            if score > max_score {
                max_score = score;
            }
            key += 1;
        }
        let mut sum = 0.0f32;
        key = 0;
        while key < tokens {
            // SAFETY: own row, written above by this thread.
            unsafe {
                let e = (*probability.add(row_base + key) - max_score).exp();
                *probability.add(row_base + key) = e;
                sum += e;
            }
            key += 1;
        }
        let uniform = !sum.is_finite() || sum <= 0.0;
        let inverse_sum = 1.0 / (sum + 1.0e-9);
        key = 0;
        while key < tokens {
            // SAFETY: own row.
            unsafe {
                let p = if uniform {
                    1.0 / tokens as f32
                } else {
                    *probability.add(row_base + key) * inverse_sum
                };
                *probability.add(row_base + key) = p;
            }
            key += 1;
        }

        let context_base = (window_index * tokens + query) * channels + head * head_dim;
        let out = context.as_mut_ptr();
        let mut d = 0usize;
        while d < head_dim {
            let mut acc = 0.0f32;
            key = 0;
            while key < tokens {
                let value_index =
                    (window_index * tokens + key) * 3 * channels + 2 * channels + head * head_dim + d;
                // SAFETY: own probability row.
                let p = unsafe { *probability.add(row_base + key) };
                acc += p * qkv[value_index];
                key += 1;
            }
            // SAFETY: each (window, head, query) thread owns this head's
            // disjoint channel interval of one context row.
            unsafe {
                *out.add(context_base + d) = acc;
            }
            d += 1;
        }
    }

    /// Row half of the attention adjoint, one `(window, head, query)` row per
    /// thread: `dP = dctx . v`, `dS = active * P * (dP - sum(P * dP))` into
    /// `score_grad`, then the clamped-Q gradient `scale * dS . k` into
    /// `query_grad` `[rows, channels]` (zero where Q left `(-256, 256)`).
    #[kernel]
    pub fn swin_window_attention_backward_rows(
        qkv: &[f32],
        probabilities: &[f32],
        active: &[f32],
        context_grad: &[f32],
        mut score_grad: DisjointSlice<f32>,
        mut query_grad: DisjointSlice<f32>,
        total_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
    ) {
        let thread_index = thread::index_1d().get();
        let tokens = window * window;
        if thread_index >= total_windows * heads * tokens {
            return;
        }
        let head_dim = channels / heads;
        let query = thread_index % tokens;
        let window_head = thread_index / tokens;
        let head = window_head % heads;
        let window_index = window_head / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let row_base = thread_index * tokens;
        let query_row = window_index * tokens + query;
        let grad_base = query_row * channels + head * head_dim;
        let ds = score_grad.as_mut_ptr();

        let mut key = 0usize;
        while key < tokens {
            let value_base =
                (window_index * tokens + key) * 3 * channels + 2 * channels + head * head_dim;
            let mut dp = 0.0f32;
            let mut d = 0usize;
            while d < head_dim {
                dp += context_grad[grad_base + d] * qkv[value_base + d];
                d += 1;
            }
            // SAFETY: this thread owns one `[tokens]` row of `score_grad`.
            unsafe {
                *ds.add(row_base + key) = dp;
            }
            key += 1;
        }
        let mut dot = 0.0f32;
        key = 0;
        while key < tokens {
            // SAFETY: own row.
            unsafe {
                dot += probabilities[row_base + key] * *ds.add(row_base + key);
            }
            key += 1;
        }
        key = 0;
        while key < tokens {
            let index = row_base + key;
            // SAFETY: own row.
            unsafe {
                *ds.add(index) = if active[index] != 0.0 {
                    probabilities[index] * (*ds.add(index) - dot)
                } else {
                    0.0
                };
            }
            key += 1;
        }

        let query_base = query_row * 3 * channels + head * head_dim;
        let dq = query_grad.as_mut_ptr();
        let mut d = 0usize;
        while d < head_dim {
            let raw_q = qkv[query_base + d];
            let mut gradient = 0.0f32;
            key = 0;
            while key < tokens {
                let key_index =
                    (window_index * tokens + key) * 3 * channels + channels + head * head_dim + d;
                // SAFETY: own row.
                let score = unsafe { *ds.add(row_base + key) };
                gradient += score * swin_constrain(qkv[key_index], 256.0);
                key += 1;
            }
            let live = raw_q.is_finite() && raw_q > -256.0 && raw_q < 256.0;
            // SAFETY: each (window, head, query) thread owns this head's
            // disjoint channel interval of one query-gradient row.
            unsafe {
                *dq.add(grad_base + d) = if live { gradient * scale } else { 0.0 };
            }
            d += 1;
        }
    }

    /// Column half of the attention adjoint, one `(window, head, key)` per
    /// thread, writing that token's fused `[3*channels]` QKV gradient for the
    /// head: Q from `query_grad`, clamped-K `scale * dS^T . q` (zero outside
    /// `(-256, 256)`) and V `P^T . dctx`.
    #[kernel]
    pub fn swin_window_attention_backward_cols(
        qkv: &[f32],
        probabilities: &[f32],
        score_grad: &[f32],
        context_grad: &[f32],
        query_grad: &[f32],
        mut qkv_grad: DisjointSlice<f32>,
        total_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
    ) {
        let thread_index = thread::index_1d().get();
        let tokens = window * window;
        if thread_index >= total_windows * heads * tokens {
            return;
        }
        let head_dim = channels / heads;
        let key = thread_index % tokens;
        let window_head = thread_index / tokens;
        let head = window_head % heads;
        let window_index = window_head / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let matrix_base = window_head * tokens * tokens;
        let key_row = window_index * tokens + key;
        let out_base = key_row * 3 * channels + head * head_dim;
        let out = qkv_grad.as_mut_ptr();
        let mut d = 0usize;
        while d < head_dim {
            let raw_k = qkv[out_base + channels + d];
            let mut dk = 0.0f32;
            let mut dv = 0.0f32;
            let mut query = 0usize;
            while query < tokens {
                let index = matrix_base + query * tokens + key;
                let query_row = window_index * tokens + query;
                dk += score_grad[index]
                    * swin_constrain(qkv[query_row * 3 * channels + head * head_dim + d], 256.0);
                dv += probabilities[index] * context_grad[query_row * channels + head * head_dim + d];
                query += 1;
            }
            let live = raw_k.is_finite() && raw_k > -256.0 && raw_k < 256.0;
            // SAFETY: each (window, head, key) thread owns this head's three
            // disjoint channel intervals of one fused QKV-gradient row.
            unsafe {
                *out.add(out_base + d) = query_grad[key_row * channels + head * head_dim + d];
                *out.add(out_base + channels + d) = if live { dk * scale } else { 0.0 };
                *out.add(out_base + 2 * channels + d) = dv;
            }
            d += 1;
        }
    }

    /// Relative-position bias gradient `[heads, (2*window-1)^2]`, one table
    /// entry per thread: the sum of `score_grad` over every window and every
    /// (query, key) pair whose offset selects that entry.
    #[kernel]
    pub fn swin_relative_bias_backward(
        score_grad: &[f32],
        mut output: DisjointSlice<f32>,
        total_windows: usize,
        window: usize,
        heads: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let tokens = window * window;
        let table_width = 2 * window - 1;
        let table_len = table_width * table_width;
        let head = i / table_len;
        let entry = i % table_len;
        // `entry = (qy - ky + window - 1) * table_width + (qx - kx + window - 1)`.
        let offset_y = entry / table_width;
        let offset_x = entry % table_width;
        let mut sum = 0.0f32;
        let mut window_index = 0usize;
        while window_index < total_windows {
            let matrix_base = (window_index * heads + head) * tokens * tokens;
            let mut query_y = 0usize;
            while query_y < window {
                let shifted_y = query_y + window - 1;
                if shifted_y >= offset_y && shifted_y - offset_y < window {
                    let key_y = shifted_y - offset_y;
                    let mut query_x = 0usize;
                    while query_x < window {
                        let shifted_x = query_x + window - 1;
                        if shifted_x >= offset_x && shifted_x - offset_x < window {
                            let key_x = shifted_x - offset_x;
                            let query = query_y * window + query_x;
                            let key = key_y * window + key_x;
                            sum += score_grad[matrix_base + query * tokens + key];
                        }
                        query_x += 1;
                    }
                }
                query_y += 1;
            }
            window_index += 1;
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = sum;
        }
    }
}
