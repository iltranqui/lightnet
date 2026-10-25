//! ViT kernels: patchify, positional embedding, feature clamp, mHC residual and
//! token-to-NCHW, plus their adjoints for the native backward.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// ViT patch gather: NCHW input to `[batch, tokens, patch*patch*channels]`
    /// with the patch interior ordered `(dy, dx, channel)` and zero padding,
    /// matching `lightnet_core::vit::patchify_nchw`.
    #[kernel]
    pub fn vit_patchify(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let patch_dim = patch_size * patch_size * channels;
        let tokens = out_h * out_w;
        let column = i % patch_dim;
        let token = (i / patch_dim) % tokens;
        let batch = i / (patch_dim * tokens);
        let channel = column % channels;
        let tap = column / channels;
        let dx = tap % patch_size;
        let dy = tap / patch_size;
        let y = (token / out_w) * patch_stride + dy;
        let x = (token % out_w) * patch_stride + dx;
        let value = if y < pad || y - pad >= height || x < pad || x - pad >= width {
            0.0
        } else {
            input[(batch * channels + channel) * height * width + (y - pad) * width + (x - pad)]
        };
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Add a `[tokens, hidden]` positional table to every `[tokens, hidden]`
    /// sample slab of a `[batch, tokens, hidden]` tensor.
    #[kernel]
    pub fn vit_add_pos_embed(
        tokens: &[f32],
        pos: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let value = tokens[i] + pos[i % pos.len()];
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Source `clamp_feature`: finite values clamp to `[-clip, clip]`, the rest
    /// become zero.
    #[kernel]
    pub fn vit_clamp_feature(input: &[f32], mut output: DisjointSlice<f32>, clip: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = input[i];
        let value = if x.is_finite() {
            if x > clip {
                clip
            } else if x < -clip {
                -clip
            } else {
                x
            }
        } else {
            0.0
        };
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// ViT mHC residual mix `clamp(a*clamp(skip) + b*clamp(branch))`.
    #[kernel]
    pub fn vit_mhc_residual(
        skip: &[f32],
        branch: &[f32],
        mut output: DisjointSlice<f32>,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let mut s = skip[i];
        s = if s.is_finite() {
            s.max(-clip).min(clip)
        } else {
            0.0
        };
        let mut b = branch[i];
        b = if b.is_finite() {
            b.max(-clip).min(clip)
        } else {
            0.0
        };
        let mut v = skip_scale * s + branch_scale * b;
        v = if v.is_finite() {
            v.max(-clip).min(clip)
        } else {
            0.0
        };
        if let Some(dst) = output.get_mut(idx) {
            *dst = v;
        }
    }

    /// `[batch, tokens, hidden]` to NCHW `[batch, hidden, tokens]`.
    #[kernel]
    pub fn vit_tokens_to_nchw(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: usize,
        hidden: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let token = i % tokens;
        let channel = (i / tokens) % hidden;
        let batch = i / (tokens * hidden);
        let value = input[(batch * tokens + token) * hidden + channel];
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// NCHW `[batch, hidden, tokens]` to `[batch, tokens, hidden]`; the adjoint
    /// (and inverse) of `vit_tokens_to_nchw`.
    #[kernel]
    pub fn vit_nchw_to_tokens(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: usize,
        hidden: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let channel = i % hidden;
        let token = (i / hidden) % tokens;
        let batch = i / (tokens * hidden);
        let value = input[(batch * hidden + channel) * tokens + token];
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Adjoint of `vit_patchify`, written as a per-pixel gather: each NCHW
    /// input element sums every patch tap that sampled it, in the token order
    /// of `scatter_patch_gradients_nchw`, so overlapping patches need no atomics.
    #[kernel]
    pub fn vit_patch_scatter(
        patch_grad: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
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
        let patch_dim = patch_size * patch_size * channels;
        let tokens = out_h * out_w;
        // Padded coordinates; token `t` covers `[t * stride, t * stride + patch)`.
        let yp = y + pad;
        let xp = x + pad;
        let ty_first = if yp + 1 > patch_size {
            (yp + 1 - patch_size + patch_stride - 1) / patch_stride
        } else {
            0
        };
        let tx_first = if xp + 1 > patch_size {
            (xp + 1 - patch_size + patch_stride - 1) / patch_stride
        } else {
            0
        };
        let ty_end = (yp / patch_stride + 1).min(out_h);
        let tx_end = (xp / patch_stride + 1).min(out_w);
        let mut sum = 0.0f32;
        let mut ty = ty_first;
        while ty < ty_end {
            let dy = yp - ty * patch_stride;
            let mut tx = tx_first;
            while tx < tx_end {
                let dx = xp - tx * patch_stride;
                let row = batch * tokens + ty * out_w + tx;
                sum += patch_grad[row * patch_dim + (dy * patch_size + dx) * channels + channel];
                tx += 1;
            }
            ty += 1;
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = sum;
        }
    }

    /// Adjoint of `vit_clamp_feature` at raw input `raw`: the gradient passes
    /// only where `raw` was finite and strictly inside `(-clip, clip)`.
    #[kernel]
    pub fn vit_clamp_backward(raw: &[f32], grad: &[f32], mut output: DisjointSlice<f32>, clip: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = raw[i];
        let value = if x.is_finite() && x.abs() < clip { grad[i] } else { 0.0 };
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Input adjoints of `vit_mhc_residual`, matching
    /// `lightnet_core::vit::mhc_residual_backward`: zero where the mixed value
    /// saturated or the seed is non-finite, and through each inner clamp mask.
    #[kernel]
    pub fn vit_mhc_residual_backward(
        skip: &[f32],
        branch: &[f32],
        grad: &[f32],
        mut d_skip: DisjointSlice<f32>,
        mut d_branch: DisjointSlice<f32>,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= grad.len() {
            return;
        }
        let raw_s = skip[i];
        let raw_b = branch[i];
        let s = if raw_s.is_finite() { raw_s.max(-clip).min(clip) } else { 0.0 };
        let b = if raw_b.is_finite() { raw_b.max(-clip).min(clip) } else { 0.0 };
        let mixed = skip_scale * s + branch_scale * b;
        let g = grad[i];
        let live = mixed.is_finite() && mixed.abs() < clip && g.is_finite();
        let ds = if live && raw_s.is_finite() && raw_s.abs() < clip {
            g * skip_scale
        } else {
            0.0
        };
        let db = if live && raw_b.is_finite() && raw_b.abs() < clip {
            g * branch_scale
        } else {
            0.0
        };
        if let Some(dst) = d_skip.get_mut(thread::index_1d()) {
            *dst = ds;
        }
        if let Some(dst) = d_branch.get_mut(thread::index_1d()) {
            *dst = db;
        }
    }

    /// Per-element terms of the mHC scalar gradients in planar `[3, len]`
    /// order: `g*(p*s+(1-p)*b)`, `g*((1-p)*s+p*b)` and
    /// `g*(post_skip-post_branch)*(s-b)`, zero where the element is skipped.
    /// A per-plane sum gives the `grad_post_skip`, `grad_post_branch` and
    /// `grad_p` accumulators of `mhc_residual_backward`.
    #[kernel]
    pub fn vit_mhc_param_terms(
        skip: &[f32],
        branch: &[f32],
        grad: &[f32],
        mut terms: DisjointSlice<f32>,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        clip: f32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let len = grad.len();
        if j >= terms.len() {
            return;
        }
        let plane = j / len;
        let i = j % len;
        let raw_s = skip[i];
        let raw_b = branch[i];
        let s = if raw_s.is_finite() { raw_s.max(-clip).min(clip) } else { 0.0 };
        let b = if raw_b.is_finite() { raw_b.max(-clip).min(clip) } else { 0.0 };
        let mixed = skip_scale * s + branch_scale * b;
        let g = grad[i];
        let value = if mixed.is_finite() && mixed.abs() < clip && g.is_finite() {
            if plane == 0 {
                g * (mix * s + (1.0 - mix) * b)
            } else if plane == 1 {
                g * ((1.0 - mix) * s + mix * b)
            } else {
                g * (post_skip - post_branch) * (s - b)
            }
        } else {
            0.0
        };
        if let Some(dst) = terms.get_mut(idx) {
            *dst = value;
        }
    }

    /// Learned positional-table gradient: sum the `[batch, table]` token
    /// gradient over the batch, one thread per table entry.
    #[kernel]
    pub fn vit_pos_embed_backward(grad: &[f32], mut output: DisjointSlice<f32>, batch: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        let table = output.len();
        if i >= table {
            return;
        }
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            sum += grad[b * table + i];
            b += 1;
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = sum;
        }
    }
}
