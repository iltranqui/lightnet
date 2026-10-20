//! Fused per-sample FiLM channel affine and adjoint.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `y[b,c,s] = x[b,c,s] * (1 + gamma[b,c]) + beta[b,c]`.
    #[kernel]
    pub fn film_affine(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
    ) {
        let i = thread::index_1d().get();
        if i < x.len() {
            let channel = i / spatial % channels;
            let batch_channel = i / (channels * spatial) * channels + channel;
            if let Some(value) = output.get_mut(thread::index_1d()) {
                *value = x[i] * (1.0 + gamma[batch_channel]) + beta[batch_channel];
            }
        }
    }

    /// `d_input = d_output * (1 + gamma)`, one owned output per element.
    #[kernel]
    pub fn film_affine_dx(
        dy: &[f32],
        gamma: &[f32],
        mut dx: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
    ) {
        let i = thread::index_1d().get();
        if i < dy.len() {
            let channel = i / spatial % channels;
            let batch_channel = i / (channels * spatial) * channels + channel;
            if let Some(value) = dx.get_mut(thread::index_1d()) {
                *value = dy[i] * (1.0 + gamma[batch_channel]);
            }
        }
    }

    /// Propagate per-sample channel gradients through the gamma/beta
    /// projections and their ReLU embedding.
    #[kernel]
    pub fn film_embedding_backward(
        embedding: &[f32],
        dgamma: &[f32],
        dbeta: &[f32],
        w_gamma: &[f32],
        w_beta: &[f32],
        mut d_embedding: DisjointSlice<f32>,
        channels: usize,
        embed_dim: usize,
    ) {
        let index = thread::index_1d();
        let idx = index.get();
        if idx < d_embedding.len() {
            let batch = idx / embed_dim;
            let feature = idx % embed_dim;
            let mut gradient = 0.0f32;
            let mut channel = 0usize;
            while channel < channels {
                let channel_index = batch * channels + channel;
                let weight = channel * embed_dim + feature;
                gradient +=
                    dgamma[channel_index] * w_gamma[weight] + dbeta[channel_index] * w_beta[weight];
                channel += 1;
            }
            if embedding[idx] <= 0.0 {
                gradient = 0.0;
            }
            if let Some(value) = d_embedding.get_mut(index) {
                *value = gradient;
            }
        }
    }

    /// Produce every shared FiLM projection parameter gradient and the optional
    /// per-sample HF-source gradient. Each target element has one owner thread.
    #[kernel]
    pub fn film_projection_backward(
        hf: &[f32],
        embedding: &[f32],
        dgamma: &[f32],
        dbeta: &[f32],
        d_embedding: &[f32],
        w_embed: &[f32],
        mut dw_embed: DisjointSlice<f32>,
        mut db_embed: DisjointSlice<f32>,
        mut dw_gamma: DisjointSlice<f32>,
        mut db_gamma: DisjointSlice<f32>,
        mut dw_beta: DisjointSlice<f32>,
        mut db_beta: DisjointSlice<f32>,
        mut dhf: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        param_count: usize,
        embed_dim: usize,
    ) {
        let idx = thread::index_1d().get();
        if idx < channels * embed_dim {
            let channel = idx / embed_dim;
            let feature = idx % embed_dim;
            let mut gamma_sum = 0.0f32;
            let mut beta_sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                let channel_index = b * channels + channel;
                let embedded = embedding[b * embed_dim + feature];
                gamma_sum += dgamma[channel_index] * embedded;
                beta_sum += dbeta[channel_index] * embedded;
                b += 1;
            }
            if let Some(value) = dw_gamma.get_mut(thread::index_1d()) {
                *value = gamma_sum;
            }
            if let Some(value) = dw_beta.get_mut(thread::index_1d()) {
                *value = beta_sum;
            }
        }
        if idx < channels {
            let mut gamma_sum = 0.0f32;
            let mut beta_sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                gamma_sum += dgamma[b * channels + idx];
                beta_sum += dbeta[b * channels + idx];
                b += 1;
            }
            if let Some(value) = db_gamma.get_mut(thread::index_1d()) {
                *value = gamma_sum;
            }
            if let Some(value) = db_beta.get_mut(thread::index_1d()) {
                *value = beta_sum;
            }
        }
        if idx < embed_dim {
            let mut sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                sum += d_embedding[b * embed_dim + idx];
                b += 1;
            }
            if let Some(value) = db_embed.get_mut(thread::index_1d()) {
                *value = sum;
            }
        }
        if idx < embed_dim * param_count {
            let feature = idx / param_count;
            let parameter = idx % param_count;
            let mut sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                sum += d_embedding[b * embed_dim + feature] * hf[b * param_count + parameter];
                b += 1;
            }
            if let Some(value) = dw_embed.get_mut(thread::index_1d()) {
                *value = sum;
            }
        }
        if idx < batch * param_count {
            let b = idx / param_count;
            let parameter = idx % param_count;
            let mut sum = 0.0f32;
            let mut feature = 0usize;
            while feature < embed_dim {
                sum += d_embedding[b * embed_dim + feature]
                    * w_embed[feature * param_count + parameter];
                feature += 1;
            }
            if let Some(value) = dhf.get_mut(thread::index_1d()) {
                *value = sum;
            }
        }
    }

    /// Spatially reduce `d_gamma = sum(dy*x)` and `d_beta = sum(dy)`.
    /// One thread owns each `(batch, channel)` reduction result.
    #[kernel]
    pub fn film_affine_param_grads(
        x: &[f32],
        dy: &[f32],
        mut dgamma: DisjointSlice<f32>,
        mut dbeta: DisjointSlice<f32>,
        spatial: usize,
    ) {
        let bc = thread::index_1d();
        let idx = bc.get();
        if idx < dgamma.len() {
            let offset = idx * spatial;
            let mut scale_sum = 0.0f32;
            let mut bias_sum = 0.0f32;
            let mut s = 0usize;
            while s < spatial {
                scale_sum += dy[offset + s] * x[offset + s];
                bias_sum += dy[offset + s];
                s += 1;
            }
            if let Some(value) = dgamma.get_mut(bc) {
                *value = scale_sum;
            }
            if let Some(value) = dbeta.get_mut(thread::index_1d()) {
                *value = bias_sum;
            }
        }
    }
}
