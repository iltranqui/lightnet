//! Native f32 Tucker attention: padded window gather, low-rank projections,
//! dense or minimax sparse attention and the exact CPU-reference adjoint.
//! Native SGD/Adam/Muon keep serialized factors and gradients resident.
use crate::error::{Error, Result};
use crate::layers::vit::gpu::device_activation;
use crate::model::{Layer, TuckerAttentionSpec};
use crate::runtime::{GpuExecutableNetwork, LayerTrainBuffers, ResidentMomentSlot,
    ResidentParameterRange, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::host_ops::tucker_attention::AttentionGeometry;
use super::{TuckerAttentionParams, TuckerGeometry};

pub(crate) fn geometry(layer: &Layer, spec: &TuckerAttentionSpec) -> TuckerGeometry {
    TuckerGeometry {
        batch: layer.input_shape.batch,
        channels: layer.input_shape.channels,
        height: layer.input_shape.height,
        width: layer.input_shape.width,
        size: spec.size,
        heads: spec.heads,
        rank_q: spec.rank_q,
        rank_k: spec.rank_k,
        rank_v: spec.rank_v,
        rank_o: spec.rank_o,
        activation: spec.activation.clone(),
        msa_enabled: spec.msa_enabled,
        msa_block_size: spec.msa_block_size,
        msa_topk: spec.msa_topk,
    }
}

pub(crate) fn supported(g: &TuckerGeometry) -> bool {
    super::validate_geometry(g).is_ok() && device_activation(&g.activation).is_some()
}

/// The eight flat CPU blocks, expressed as connected `[outputs, inputs]`
/// matrices. Q/K/V cores additionally permute `[head, rank, dim]`.
fn blocks(g: &TuckerGeometry) -> [(usize, usize, bool); 8] {
    let c = g.channels;
    [(c, g.rank_q, false), (c, g.rank_k, false), (c, g.rank_v, false),
        (g.rank_q, c, true), (g.rank_k, c, true), (g.rank_v, c, true),
        (c, g.rank_o, false), (g.rank_o, c, false)]
}

fn source_index(input: usize, output: usize, inputs: usize, outputs: usize,
    core: bool, heads: usize) -> usize
{
    if core {
        let dim = outputs / heads;
        (output / dim * inputs + input) * dim + output % dim
    } else {
        input * outputs + output
    }
}

fn attention_geometry(g: &TuckerGeometry) -> Result<AttentionGeometry> {
    super::validate_geometry(g).map_err(Error::InvalidConfig)?;
    let windows = g.batch * g.height.div_ceil(g.size) * g.width.div_ceil(g.size);
    let a = AttentionGeometry {
        windows, tokens: g.size * g.size, channels: g.channels, heads: g.heads,
        msa_enabled: g.msa_enabled, msa_block_size: g.msa_block_size, msa_topk: g.msa_topk,
    };
    a.lengths().map_err(Error::InvalidConfig)?;
    let rows = windows * a.tokens;
    for rank in [g.rank_q, g.rank_k, g.rank_v, g.rank_o] {
        rows.checked_mul(rank).ok_or_else(|| Error::InvalidConfig("Tucker latent size overflow".into()))?;
    }
    Ok(a)
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: TuckerGeometry,
    input: T,
    latent: [T; 3],
    qkv: [T; 3],
    probabilities: T,
    active: T,
    context: T,
    o_mix: T,
    pre_activation: T,
    weights: Vec<T>,
    resident_generation: Option<u64>,
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_tucker_attention_gpu(
        &self, layer: &Layer, spec: &TuckerAttentionSpec, params: &TuckerAttentionParams,
        current_input: &B::Tensor, config: ElementwiseLaunchConfig, training: bool,
    ) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        let g = geometry(layer, spec);
        let a = attention_geometry(&g)?;
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "tucker_attention GPU activation".into(),
        })?;
        if layer.output_shape != layer.input_shape || !params.matches_geometry(&g) {
            return Err(Error::InvalidConfig("Tucker parameter/output geometry does not match the layer".into()));
        }
        let host = params.to_blob().map_err(Error::InvalidConfig)?;
        let rows = a.windows * a.tokens;
        let c = g.channels;
        let up = |shape: &[usize], data: &[f32]| self.backend.upload_f32(shape, data)
            .map_err(|e| backend_error("upload Tucker parameters", e));
        let mut weights = Vec::with_capacity(8);
        let mut offset = 0;
        let (bias, resident_generation) = if self.resident_parameter_path_enabled() {
            let layout = vec![
                ResidentParameterRange { range:0..c, decay:false, coupled_decay:true,
                    moment_slot:Some(ResidentMomentSlot::Biases), active:true },
                ResidentParameterRange { range:c..host.len(), decay:true, coupled_decay:true,
                    moment_slot:Some(ResidentMomentSlot::Weights), active:true },
            ];
            let mut slices = vec![0..c];
            let mut start = c;
            for (inputs, outputs, _) in blocks(&g) {
                slices.push(start..start + inputs * outputs);
                start += inputs * outputs;
            }
            let (tensors, generation) = self.resident_parameter_slices(layer.index, &host,
                layout, &slices, config)?;
            let mut tensors = tensors.into_iter();
            let bias = tensors.next().expect("Tucker bias slice");
            for ((inputs, outputs, core), source) in blocks(&g).into_iter().zip(tensors) {
                // [inputs, outputs] -> [outputs, inputs]; for Q/K/V
                // cores the head is an independent batch dimension.
                let heads = if core { g.heads } else { 1 };
                weights.push(self.backend.vit_nchw_to_tokens_f32(&source, heads, 1,
                    outputs / heads, inputs, config)
                    .map_err(|e| backend_error("pack resident Tucker factor", e))?);
            }
            (bias, Some(generation))
        } else {
            for (inputs, outputs, core) in blocks(&g) {
                let count = inputs * outputs;
                let mut packed = vec![0.0; count];
                for out in 0..outputs {
                    for input in 0..inputs {
                        packed[out * inputs + input] = params.weights[offset
                            + source_index(input, out, inputs, outputs, core, g.heads)];
                    }
                }
                weights.push(up(&[outputs, inputs], &packed)?);
                offset += count;
            }
            (up(&[c], &params.biases)?, None)
        };
        let project = |x: &B::Tensor, index: usize, bias: Option<&B::Tensor>| -> Result<B::Tensor> {
            let (inputs, outputs, _) = blocks(&g)[index];
            let zero;
            let bias = match bias {
                Some(bias) => bias,
                None => { zero = up(&[outputs], &vec![0.0; outputs])?; &zero }
            };
            self.backend.connected_f32(x, &weights[index], bias, rows, inputs, outputs, config)
                .map_err(|e| backend_error("GPU Tucker projection", e))
        };
        let ph = g.height.div_ceil(g.size) * g.size;
        let pw = g.width.div_ceil(g.size) * g.size;
        let input = self.backend.swin_window_partition_f32(current_input, g.batch, c,
            g.height, g.width, ph, pw, g.size, 0, config)
            .map_err(|e| backend_error("GPU Tucker window partition", e))?;
        let latent = [project(&input, 0, None)?, project(&input, 1, None)?, project(&input, 2, None)?];
        let qkv = [project(&latent[0], 3, None)?, project(&latent[1], 4, None)?, project(&latent[2], 5, None)?];
        let (context, probabilities, active) = self.backend
            .tucker_attention_f32(&qkv[0], &qkv[1], &qkv[2], a, config)
            .map_err(|e| backend_error("GPU Tucker attention", e))?;
        let o_mix = project(&context, 6, None)?;
        // The identity residual is added before the activation.
        let projection = project(&o_mix, 7, Some(&bias))?;
        let pre_activation = self.backend.residual_bias_f32(&projection, &input, None,
            &[rows, c], rows, c, config)
            .map_err(|e| backend_error("GPU Tucker residual", e))?;
        let activated = self.backend.activation_f32(&pre_activation, activation, config)
            .map_err(|e| backend_error("GPU Tucker activation", e))?;
        let output = self.backend.swin_window_unpartition_f32(&activated, g.batch, c,
            g.height, g.width, ph, pw, g.size, 0, 0.0, config)
            .map_err(|e| backend_error("GPU Tucker window unpartition", e))?;
        let cache = training.then(|| DeviceCache {
            geometry: g, input, latent, qkv, probabilities, active, context, o_mix,
            pre_activation, weights, resident_generation,
        });
        Ok((output, cache))
    }

    /// Host reference for unmapped activations and training networks whose
    /// other layers require the whole-network CPU backward fallback.
    pub(crate) fn forward_tucker_attention_host(
        &self, layer: &Layer, spec: &TuckerAttentionSpec, weights: &TuckerAttentionParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::TuckerAttentionForwardCache)> {
        let host_input = self.download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download Tucker attention input", err))?;
        let (data, cache) = super::tucker_attention_forward(&host_input.data, weights, &geometry(layer, spec))
            .map_err(Error::InvalidConfig)?;
        let output = self.backend.upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload Tucker attention output", err))?;
        Ok((output, cache))
    }

    pub(crate) fn backward_tucker_attention_gpu(
        &self, layer: &Layer, spec: &TuckerAttentionSpec,
        forward: &crate::runtime::GpuForwardState<B::Tensor>, grad: B::Tensor,
        config: ElementwiseLaunchConfig, layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let g = geometry(layer, spec);
        let a = attention_geometry(&g)?;
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "tucker_attention GPU backward activation".into(),
        })?;
        let cache = forward.tucker_attention_device_caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig(format!("missing Tucker attention device training cache for layer #{}", layer.index)))?;
        if cache.geometry != g {
            return Err(Error::InvalidConfig("Tucker device cache geometry mismatch".into()));
        }
        let be = &self.backend;
        let err = |e| backend_error("GPU Tucker backward", e);
        let rows = a.windows * a.tokens;
        let c = g.channels;
        let ph = g.height.div_ceil(g.size) * g.size;
        let pw = g.width.div_ceil(g.size) * g.size;
        // Cropped output adjoint: padded queries have zero upstream, while
        // padded keys/values still receive gradients through attention.
        let upstream = be.swin_window_partition_f32(&grad, g.batch, c, g.height, g.width,
            ph, pw, g.size, 0, config).map_err(err)?;
        let d_pre = be.activation_backward_f32(&cache.pre_activation,
            &upstream, activation, config).map_err(err)?;
        let backward = |x: &B::Tensor, dy: &B::Tensor, index: usize| {
            let (inputs, outputs, _) = blocks(&g)[index];
            be.connected_backward_f32(x, &cache.weights[index], dy, rows, inputs, outputs, config)
                .map_err(err)
        };
        let (d_o_mix, d_o_basis, d_bias) = backward(&cache.o_mix, &d_pre, 7)?;
        let (d_context, d_o_core, _) = backward(&cache.context, &d_o_mix, 6)?;
        let (dq, dk, dv) = be.tucker_attention_backward_f32(&cache.qkv[0], &cache.qkv[1],
            &cache.qkv[2], &cache.probabilities, &cache.active, &d_context, a, config).map_err(err)?;
        let mut d_input = d_pre;
        let mut parameter_grads = vec![(7, d_o_basis), (6, d_o_core)];
        for (branch, dy) in [dq, dk, dv].into_iter().enumerate() {
            let (d_latent, d_core, _) = backward(&cache.latent[branch], &dy, branch + 3)?;
            let (dx, d_basis, _) = backward(&cache.input, &d_latent, branch)?;
            d_input = be.residual_bias_f32(&d_input, &dx, None, &[rows, c], rows, c, config).map_err(err)?;
            parameter_grads.push((branch, d_basis));
            parameter_grads.push((branch + 3, d_core));
        }
        let d_input = be.swin_window_unpartition_f32(&d_input, g.batch, c, g.height, g.width,
            ph, pw, g.size, 0, 0.0, config).map_err(err)?;
        // Restore the serialized CPU factor order before accumulation.
        let descriptions = blocks(&g);
        let mut offsets = [0usize; 8];
        let mut total = 0;
        for (i, &(inputs, outputs, _)) in descriptions.iter().enumerate() {
            offsets[i] = total;
            total += inputs * outputs;
        }
        if let Some(generation) = cache.resident_generation {
            let mut gradients = vec![(0..c, d_bias)];
            for (i, tensor) in parameter_grads {
                let (inputs, outputs, core) = descriptions[i];
                let heads = if core { g.heads } else { 1 };
                let source = be.vit_tokens_to_nchw_f32(&tensor, heads, 1,
                    outputs / heads, inputs, config).map_err(err)?;
                let start = c + offsets[i];
                gradients.push((start..start + inputs * outputs, source));
            }
            self.accumulate_resident_parameter_gradients(layer.index, generation, gradients, config)?;
        } else {
            let mut d_weights = vec![0.0; total];
            for (i, tensor) in parameter_grads {
                let values = be.download_f32(&tensor).map_err(err)?;
                let (inputs, outputs, core) = descriptions[i];
                if values.len() != inputs * outputs {
                    return Err(Error::InvalidConfig("Tucker factor gradient length mismatch".into()));
                }
                for out in 0..outputs {
                    for input in 0..inputs {
                        d_weights[offsets[i] + source_index(input, out, inputs, outputs, core, g.heads)] =
                            values[out * inputs + input];
                    }
                }
            }
            let d_bias = be.download_f32(&d_bias).map_err(err)?;
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::TuckerAttention(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig("missing Tucker attention train buffers".into()));
            };
            if train.weight_updates.len() != total || train.bias_updates.len() != d_bias.len() {
                return Err(Error::InvalidConfig("Tucker attention train buffer length mismatch".into()));
            }
            for (dst, src) in train.weight_updates.iter_mut().zip(d_weights) { *dst -= src; }
            for (dst, src) in train.bias_updates.iter_mut().zip(d_bias) { *dst -= src; }
        }
        self.accumulate_previous_gpu_gradient(layer.index, d_input, layer_output_gradients)
    }
}
