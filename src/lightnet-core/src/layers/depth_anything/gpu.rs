//! Frozen DINO encoder on the device, with exact RGB input adjoints.
//! Bicubic interpolation touches only constant positional weights on upload;
//! activations and all transformer backward math stay on the device.
use std::ops::Range;
use super::{DepthAnythingGeometry, DepthAnythingWeights};
use crate::error::{Error, Result};
use crate::model::{DepthAnythingSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

#[derive(Clone, Debug)]
struct Block<T> { weights: Vec<T>, input: T, after_attention: T, qkv: T, context: T, lse: T, fc1_pre: T, output: T }

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: DepthAnythingGeometry,
    patch_weights: T,
    norm_gamma: T,
    blocks: Vec<Block<T>>,
    normalized_taps: Vec<T>,
}

pub(crate) fn supported(layer: &Layer, spec: &DepthAnythingSpec) -> bool {
    let g = spec.geometry(layer.input_shape);
    g.validate().is_ok() && g.dim / g.heads <= 128
}

fn fields(g: &DepthAnythingGeometry) -> Result<Vec<Range<usize>>> {
    let mut offset = 0;
    super::weight_array_sizes(g).map_err(Error::InvalidConfig)?.into_iter().map(|(_, len)| {
        let end = offset + len; let r = offset..end; offset = end; Ok(r)
    }).collect()
}

impl<B: F32TensorOpsBackend> GpuExecutableNetwork<B> where B::Error: std::fmt::Display {
    pub(crate) fn forward_depth_anything_gpu(&self, layer: &Layer, spec: &DepthAnythingSpec,
        params: &DepthAnythingWeights, input: &B::Tensor, config: ElementwiseLaunchConfig,
        training: bool) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        let g = spec.geometry(layer.input_shape);
        g.validate().map_err(Error::InvalidConfig)?;
        if g.dim / g.heads > 128 || !params.matches(&g) {
            return Err(Error::InvalidConfig("DepthAnything native geometry/weights mismatch".into()));
        }
        let ranges = fields(&g)?;
        let values = params.as_slice();
        let be = &self.backend;
        let err = |e| backend_error("GPU DepthAnything forward", e);
        let up = |data: &[f32]| be.upload_f32(&[data.len()], data).map_err(err);
        let shared = |i: usize| up(&values[ranges[i].clone()]);
        let (b, e, gh, gw, tokens) = (g.batch, g.dim, g.grid_h(), g.grid_w(), g.token_count().map_err(Error::InvalidConfig)?);
        let patches = tokens - 1;
        let rows = b * tokens;
        let patch_weights = shared(0)?;
        let patch_bias = shared(1)?;
        let norm_gamma = shared(4)?;
        let norm_beta = shared(5)?;
        let normalized = if g.apply_imagenet_norm {
            let shift = up(&g.imagenet_mean.map(|x| -x))?;
            let scale = up(&g.imagenet_std.map(|x| 1.0/x))?;
            let shifted = be.bias_add_f32(input, &shift, 3, g.height*g.width, config).map_err(err)?;
            be.channel_scale_f32(&shifted, &scale, 3, g.height*g.width, config).map_err(err)?
        } else { input.clone() };
        let patch_features = be.conv2d_f32(&normalized, &patch_weights, &[b,e,gh,gw],
            3, g.height, g.width, e, g.patch_size, g.patch_size, g.patch_size, 1, 0, 1, config).map_err(err)?;
        let patch_features = be.bias_add_f32(&patch_features, &patch_bias, e, patches, config).map_err(err)?;
        let patch_features = be.vit_nchw_to_tokens_f32(&patch_features, b, gh, gw, e, config).map_err(err)?;
        let mut cls = Vec::with_capacity(b*e);
        for _ in 0..b { cls.extend_from_slice(&values[ranges[2].clone()]); }
        let cls = be.upload_f32(&[b,1,1,e], &cls).map_err(err)?;
        let hidden = be.route_concat_f32(&[&cls, &patch_features], &[0,0], &[1,patches],
            &[1,patches], &[b,tokens,1,e], e, config).map_err(err)?;
        let original_pos = &values[ranges[3].clone()];
        let resized_pos = super::bicubic_pos_embed(&original_pos[e..], g.pos_grid_h, g.pos_grid_w,
            e, gh, gw).map_err(Error::InvalidConfig)?;
        let mut pos = original_pos[..e].to_vec(); pos.extend(resized_pos);
        let pos = be.upload_f32(&[tokens,e], &pos).map_err(err)?;
        let mut hidden = be.vit_add_pos_embed_f32(&hidden, &pos, config).map_err(err)?;
        let add = |x: &B::Tensor, y: &B::Tensor| be.residual_bias_f32(x, y, None,
            &[rows,e], rows, e, config).map_err(err);
        let norm = |x: &B::Tensor, gamma: &B::Tensor, beta: &B::Tensor| be.layernorm_forward_f32(x,
            Some(gamma), Some(beta), &[rows,e], rows, e, g.ln_eps, config).map_err(err);
        let mut blocks = Vec::with_capacity(g.depth);
        let mut normalized_taps = vec![None; g.taps.len()];
        let mut taps = vec![None; g.taps.len()];
        for block in 0..g.depth {
            // Source blob is tensor-type-major, not block-major.
            let p = (6..20).map(|i| {
                let stride = ranges[i].len() / g.depth;
                let start = ranges[i].start + block*stride;
                up(&values[start..start+stride])
            }).collect::<Result<Vec<_>>>()?;
            let ln1 = norm(&hidden, &p[0], &p[1])?;
            let qkv = be.connected_f32(&ln1, &p[2], &p[3], rows, e, 3*e, config).map_err(err)?;
            let (context, lse) = be.flash_attention_forward_f32(&qkv, &[b,tokens,e],
                &[b,g.heads,tokens], b, tokens, e, g.heads).map_err(err)?;
            let projected = be.connected_f32(&context, &p[4], &p[5], rows, e, e, config).map_err(err)?;
            let branch = be.channel_scale_f32(&projected, &p[6], e, 1, config).map_err(err)?;
            let after_attention = add(&hidden, &branch)?;
            let ln2 = norm(&after_attention, &p[7], &p[8])?;
            let fc1_pre = be.connected_f32(&ln2, &p[9], &p[10], rows, e, g.mlp_dim, config).map_err(err)?;
            let act = be.gelu_erf_f32(&fc1_pre, &[rows,g.mlp_dim], config).map_err(err)?;
            let ffn = be.connected_f32(&act, &p[11], &p[12], rows, g.mlp_dim, e, config).map_err(err)?;
            let branch = be.channel_scale_f32(&ffn, &p[13], e, 1, config).map_err(err)?;
            let output = add(&after_attention, &branch)?;
            for (tap_index, &tap_block) in g.taps.iter().enumerate() {
                if tap_block != block { continue; }
                let normalized = norm(&output, &norm_gamma, &norm_beta)?;
                let patch_tokens = be.route_concat_f32(&[&normalized], &[1], &[patches], &[tokens],
                    &[b,patches,1,e], e, config).map_err(err)?;
                let tap = be.vit_tokens_to_nchw_f32(&patch_tokens, b, gh, gw, e, config).map_err(err)?;
                let tap = if g.output_clamp > 0.0 {
                    be.depth_tap_clamp_f32(&tap, None, &[b,e,gh,gw], g.output_clamp, config).map_err(err)?
                } else { tap };
                taps[tap_index] = Some(tap);
                normalized_taps[tap_index] = Some(normalized);
            }
            let input = hidden; hidden = output.clone();
            blocks.push(Block { weights: p, input, after_attention, qkv, context, lse, fc1_pre, output });
        }
        let taps = taps.into_iter().collect::<Option<Vec<_>>>().ok_or_else(|| Error::InvalidConfig("DepthAnything missing native tap".into()))?;
        let refs = taps.iter().collect::<Vec<_>>();
        let count = taps.len();
        let output = be.route_concat_f32(&refs, &vec![0;count], &vec![e;count], &vec![e;count],
            &tensor_backend_shape(layer.output_shape), patches, config).map_err(err)?;
        let normalized_taps = normalized_taps.into_iter().collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::InvalidConfig("DepthAnything missing tap training cache".into()))?;
        Ok((output, training.then(|| DeviceCache { geometry: g, patch_weights, norm_gamma, blocks, normalized_taps })))
    }

    pub(crate) fn backward_depth_anything_gpu(&self, layer: &Layer, spec: &DepthAnythingSpec,
        forward: &crate::runtime::GpuForwardState<B::Tensor>,
        grad: B::Tensor, config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>]) -> Result<()> {
        if crate::runtime::GradMode::from_env() == crate::runtime::GradMode::Cpp { return Ok(()); }
        let g = spec.geometry(layer.input_shape);
        let caches = &forward.depth_anything_device_caches;
        let cache = caches.get(layer.index).and_then(Option::as_ref).ok_or_else(|| Error::InvalidConfig("missing DepthAnything device cache".into()))?;
        if cache.geometry != g { return Err(Error::InvalidConfig("DepthAnything device cache geometry mismatch".into())); }
        let be = &self.backend;
        let err = |e| backend_error("GPU DepthAnything backward", e);
        let (b,e,gh,gw,tokens) = (g.batch,g.dim,g.grid_h(),g.grid_w(),g.token_count().map_err(Error::InvalidConfig)?);
        let (rows, patches) = (b*tokens,tokens-1);
        let add = |x: &B::Tensor,y: &B::Tensor| be.residual_bias_f32(x,y,None,&[rows,e],rows,e,config).map_err(err);
        let norm = |x: &B::Tensor,dy: &B::Tensor,gamma: &B::Tensor| be.layernorm_backward_f32(dy,x,
            Some(gamma), &[rows,e], rows,e,g.ln_eps,config).map_err(err);
        let linear = |dy: &B::Tensor, weight: &B::Tensor, ins: usize, outs: usize| {
            // Connected backward's dx is independent of x. Frozen weights do
            // not need dw/db; use zero input rather than retain every activation.
            let zero = be.zeros_f32(&[rows,ins]).map_err(err)?;
            Ok::<_, Error>(be.connected_backward_f32(&zero,weight,dy,rows,ins,outs,config).map_err(err)?.0)
        };
        let count = g.taps.len();
        let split = be.route_backward_f32(&grad, &vec![vec![b,e,gh,gw];count], &vec![0;count],
            &vec![e;count],&vec![e;count],patches,config).map_err(err)?;
        if split.len()!=count { return Err(Error::InvalidConfig("DepthAnything tap gradient arity".into())); }
        let mut dh = be.zeros_f32(&[rows,e]).map_err(err)?;
        for block_index in (0..g.depth).rev() {
            let block = &cache.blocks[block_index];
            let p = &block.weights;
            for (tap_index,&tap_block) in g.taps.iter().enumerate() {
                if tap_block!=block_index { continue; }
                let normalized = &cache.normalized_taps[tap_index];
                let patch_tokens = be.route_concat_f32(&[normalized], &[1], &[patches], &[tokens],
                    &[b,patches,1,e],e,config).map_err(err)?;
                let raw_tap = be.vit_tokens_to_nchw_f32(&patch_tokens,b,gh,gw,e,config).map_err(err)?;
                let dtap = if g.output_clamp>0.0 {
                    be.depth_tap_clamp_f32(&raw_tap,Some(&split[tap_index]),&[b,e,gh,gw],g.output_clamp,config).map_err(err)?
                } else { split[tap_index].clone() };
                let dtap = be.vit_nchw_to_tokens_f32(&dtap,b,gh,gw,e,config).map_err(err)?;
                let mut full = be.route_backward_f32(&dtap,&[vec![b,tokens,1,e]],&[1],&[patches],&[tokens],e,config).map_err(err)?;
                let full = full.pop().ok_or_else(|| Error::InvalidConfig("DepthAnything patch adjoint arity".into()))?;
                dh = add(&dh,&norm(&block.output,&full,&cache.norm_gamma)?)?;
            }
            let dmlp = be.channel_scale_f32(&dh,&p[13],e,1,config).map_err(err)?;
            let dact = linear(&dmlp,&p[11],g.mlp_dim,e)?;
            let dpre = be.gelu_erf_backward_f32(&block.fc1_pre,&dact,&[rows,g.mlp_dim],config).map_err(err)?;
            let dln2 = linear(&dpre,&p[9],e,g.mlp_dim)?;
            let dafter = add(&dh,&norm(&block.after_attention,&dln2,&p[7])?)?;
            let dprojected = be.channel_scale_f32(&dafter,&p[6],e,1,config).map_err(err)?;
            let dcontext = linear(&dprojected,&p[4],e,e)?;
            let dqkv = be.flash_attention_backward_f32(&block.qkv,&block.context,&dcontext,&block.lse,
                &[b,tokens,3*e],b,tokens,e,g.heads).map_err(err)?;
            let dln1 = linear(&dqkv,&p[2],e,3*e)?;
            dh = add(&dafter,&norm(&block.input,&dln1,&p[0])?)?;
        }
        let dpatch = be.route_concat_f32(&[&dh],&[1],&[patches],&[tokens],&[b,patches,1,e],e,config).map_err(err)?;
        let dpatch = be.vit_tokens_to_nchw_f32(&dpatch,b,gh,gw,e,config).map_err(err)?;
        let mut dx = be.conv2d_backward_input_f32(&dpatch,&cache.patch_weights,&[b,3,g.height,g.width],
            3,g.height,g.width,e,g.patch_size,g.patch_size,g.patch_size,1,0,1,config).map_err(err)?;
        if g.apply_imagenet_norm {
            let scale = be.upload_f32(&[3],&g.imagenet_std.map(|x|1.0/x)).map_err(err)?;
            dx = be.channel_scale_f32(&dx,&scale,3,g.height*g.width,config).map_err(err)?;
        }
        self.accumulate_previous_gpu_gradient(layer.index,dx,layer_output_gradients)
    }
}
