//! ViT kernels: patchify, positional embedding, feature clamp, mHC residual and token-to-NCHW.

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
}
