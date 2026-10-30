//! GPU-runtime glue for `[swin]` (and its `[transformer]` alias): a native f32
//! forward, a training forward that keeps its intermediates on the device, and
//! the native backward that consumes them, all the exact composition of
//! `swin::block::swin_forward`. Parameter gradients are staged into the host
//! train buffers, where the existing SGD/Adam/Muon arms update them.
//!
//! Variants outside [`supported`] keep the host round trip in [`GpuExecutableNetwork::forward_swin_host`]
//! and the CPU backward fallback (see `requires_cpu_backward_fallback`).
use super::block::SwinGeometry;
use crate::error::{Error, Result};
use crate::model::{Activation, Layer, SwinSpec, TensorShape};
use crate::runtime::{
    GpuExecutableNetwork, LayerTrainBuffers, ResidentMomentSlot, ResidentParameterRange, backend_error, tensor_backend_shape,
};
use crate::swin::SWIN_MHC_DEFAULTS;
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

/// LayerNorm epsilon of both Swin LayerNorms (`swin_block::swin_forward`).
const LN_EPS: f32 = 1.0e-5;
/// Source output constrain applied after the crop.
const OUTPUT_CLIP: f32 = 100.0;

fn device_activation(activation: &Activation) -> Option<F32ActivationKind> {
    match activation {
        Activation::Revleaky => Some(F32ActivationKind::Leaky),
        _ => crate::layers::vit::gpu::device_activation(activation),
    }
}

/// Block geometry of one `[swin]` layer.
pub(crate) fn geometry(layer: &Layer, spec: &SwinSpec) -> SwinGeometry {
    SwinGeometry {
        batch: layer.input_shape.batch,
        channels: layer.input_shape.channels,
        height: layer.input_shape.height,
        width: layer.input_shape.width,
        filters: spec.filters,
        heads: spec.heads,
        window: spec.size,
        shift: spec.shift,
        ffn_ratio: spec.ffn_ratio,
        activation: spec.activation.clone(),
    }
}

/// Whether the native path covers this geometry. The same gate selects the
/// native training forward and backward, so the two always agree.
pub(crate) fn supported(g: &SwinGeometry) -> bool {
    g.heads > 0
        && g.window > 0
        && g.channels % g.heads == 0
        && device_activation(&g.activation).is_some()
}

/// Derived window extents of one geometry.
#[derive(Clone, Copy, Debug)]
struct Extents {
    padded_height: usize,
    padded_width: usize,
    spatial_windows: usize,
    total_windows: usize,
    tokens: usize,
    rows: usize,
    hidden: usize,
    /// Cyclic shift of the partition (`window / 2` when shifted, else 0).
    shift: usize,
}

fn extents(g: &SwinGeometry) -> Result<Extents> {
    let overflow = |what: &str| Error::InvalidConfig(format!("[swin] {what} size overflow"));
    let (padded_height, padded_width) =
        crate::swin::padded_shape(g.height, g.width, g.window).map_err(Error::InvalidConfig)?;
    let spatial_windows = (padded_height / g.window) * (padded_width / g.window);
    let tokens = g.window * g.window;
    let total_windows = g
        .batch
        .checked_mul(spatial_windows)
        .ok_or_else(|| overflow("window count"))?;
    let rows = total_windows
        .checked_mul(tokens)
        .ok_or_else(|| overflow("window rows"))?;
    let hidden = g
        .filters
        .checked_mul(g.ffn_ratio)
        .ok_or_else(|| overflow("FFN hidden"))?;
    Ok(Extents {
        padded_height,
        padded_width,
        spatial_windows,
        total_windows,
        tokens,
        rows,
        hidden,
        shift: if g.shift { g.window / 2 } else { 0 },
    })
}

/// Device intermediates of one native training forward: the operands of the
/// native backward, plus the weight uploads it reuses. Attention keeps its
/// `[windows, heads, tokens, tokens]` probabilities and score-clamp activity;
/// `output` is the clamped layer output, whose `|y| < 100` mask is exactly
/// the open-interval derivative of the output constrain.
#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    resident_generation: Option<u64>,
    resident_ranges: Vec<std::ops::Range<usize>>,
    resident_mhc: Option<T>,
    geometry: SwinGeometry,
    windowed: T,
    ln1: T,
    qkv: T,
    context: T,
    probabilities: T,
    active: T,
    first_skip: T,
    projection: T,
    res1: T,
    ln2: T,
    ffn1_raw: T,
    ffn1_act: T,
    ffn2: T,
    output: T,
    qkv_weight: T,
    wo: T,
    ffn_w1: T,
    ffn_w2: T,
    res_proj: Option<T>,
    ln1_gamma: T,
    ln2_gamma: T,
}

/// Native inference forward; no intermediates are kept.
pub(crate) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    g: &SwinGeometry,
    weights: &super::SwinWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    Ok(run(backend, g, weights, input, output_shape, config, false, None)?.0)
}

/// Native training forward: the same composition as [`forward`], returning the
/// device cache that [`GpuExecutableNetwork::backward_swin_gpu`] consumes.
pub(crate) fn forward_training<B: F32TensorOpsBackend>(
    backend: &B,
    g: &SwinGeometry,
    weights: &super::SwinWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, DeviceCache<B::Tensor>)>
where
    B::Error: std::fmt::Display,
{
    let (output, cache) = run(backend, g, weights, input, output_shape, config, true, None)?;
    let cache = cache.ok_or_else(|| {
        Error::InvalidConfig("[swin] native training forward produced no cache".into())
    })?;
    Ok((output, cache))
}

fn run<B: F32TensorOpsBackend>(
    backend: &B,
    g: &SwinGeometry,
    weights: &super::SwinWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
    keep: bool,
    resident: Option<&[B::Tensor]>,
) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)>
where
    B::Error: std::fmt::Display,
{
    let activation = device_activation(&g.activation)
        .ok_or_else(|| Error::InvalidConfig("[swin] activation has no device path".into()))?;
    if !supported(g) || g.batch == 0 || g.height == 0 || g.width == 0 || g.ffn_ratio == 0 {
        return Err(Error::InvalidConfig(
            "[swin] native path requires positive dimensions and channels divisible by heads"
                .into(),
        ));
    }
    if !weights.matches_geometry(g.channels, g.filters, g.heads, g.window, g.ffn_ratio) {
        return Err(Error::InvalidConfig(
            "[swin] parameter geometry does not match the layer".into(),
        ));
    }
    // Validates every parameter block length against the geometry.
    weights.to_blob().map_err(Error::InvalidConfig)?;
    if weights.mhc.len() != SWIN_MHC_DEFAULTS.len() {
        return Err(Error::InvalidConfig(format!(
            "[swin] mHC has {} parameters, expected {}",
            weights.mhc.len(),
            SWIN_MHC_DEFAULTS.len()
        )));
    }
    if output_shape.batch != g.batch
        || output_shape.channels != g.filters
        || output_shape.height != g.height
        || output_shape.width != g.width
    {
        return Err(Error::InvalidConfig(format!(
            "[swin] output shape {output_shape} does not match {}x{}x{}",
            g.filters, g.height, g.width
        )));
    }
    let e = extents(g)?;
    let (c, f, rows) = (g.channels, g.filters, e.rows);
    let up = |what: &str, shape: &[usize], data: &[f32]| {
        backend
            .upload_f32(shape, data)
            .map_err(|err| backend_error(&format!("upload swin {what}"), err))
    };
    let parameter = |slot: usize, what: &str, shape: &[usize], data: &[f32]| {
        if let Some(p) = resident { Ok(p[slot].clone()) } else { up(what,shape,data) }
    };
    let table_width = 2 * g.window - 1;
    let qkv_weight = parameter(0,"qkv weights", &[3 * c, c], &weights.qkv_weight)?;
    let qkv_bias = parameter(1,"qkv bias", &[3 * c], &weights.qkv_bias)?;
    let wo = parameter(2,"wo weights", &[f, c], &weights.wo)?;
    let wo_bias = parameter(3,"wo bias", &[f], &weights.wo_bias)?;
    let ln1_gamma = parameter(4,"ln1 gamma", &[c], &weights.ln1_gamma)?;
    let ln1_beta = parameter(5,"ln1 beta", &[c], &weights.ln1_beta)?;
    let ln2_gamma = parameter(6,"ln2 gamma", &[f], &weights.ln2_gamma)?;
    let ln2_beta = parameter(7,"ln2 beta", &[f], &weights.ln2_beta)?;
    let ffn_w1 = parameter(8,"ffn1 weights", &[e.hidden, f], &weights.ffn_w1)?;
    let ffn_b1 = parameter(9,"ffn1 bias", &[e.hidden], &weights.ffn_b1)?;
    let ffn_w2 = parameter(10,"ffn2 weights", &[f, e.hidden], &weights.ffn_w2)?;
    let ffn_b2 = parameter(11,"ffn2 bias", &[f], &weights.ffn_b2)?;
    let res_proj = weights
        .res_proj
        .as_ref()
        .map(|w| parameter(12,"residual projection", &[f, c], w))
        .transpose()?;
    let bias_slot=12+usize::from(weights.res_proj.is_some());
    let relative_bias = parameter(bias_slot,
        "relative-position bias",
        &[g.heads, table_width * table_width],
        &weights.relative_position_bias,
    )?;
    // The shift-region / padding mask depends only on the spatial geometry.
    let mask_host = crate::swin::build_attention_mask(
        g.height,
        g.width,
        e.padded_height,
        e.padded_width,
        g.window,
        g.shift,
    )
    .map_err(Error::InvalidConfig)?;
    let mask = up("attention mask", &[e.spatial_windows, e.tokens, e.tokens], &mask_host)?;
    let (skip0, branch0, ..) =
        crate::swin::mhc_parameters(&weights.mhc, 0).map_err(Error::InvalidConfig)?;
    let (skip1, branch1, ..) =
        crate::swin::mhc_parameters(&weights.mhc, 1).map_err(Error::InvalidConfig)?;

    // Torch `Linear` (`[out, in]` weights) is the connected-layer GEMM with a
    // device bias epilogue.
    let lin = |what: &str,
               x: &B::Tensor,
               w: &B::Tensor,
               b: &B::Tensor,
               inf: usize,
               outf: usize|
     -> Result<B::Tensor> {
        backend
            .connected_f32(x, w, b, rows, inf, outf, config)
            .map_err(|err| backend_error(&format!("GPU swin {what}"), err))
    };
    let norm = |what: &str, x: &B::Tensor, gamma: &B::Tensor, beta: &B::Tensor, cols: usize| {
        backend
            .layernorm_forward_f32(x, Some(gamma), Some(beta), &[rows, cols], rows, cols, LN_EPS, config)
            .map_err(|err| backend_error(&format!("GPU swin {what}"), err))
    };
    let mix = |what: &str, skip: &B::Tensor, branch: &B::Tensor, a: f32, b: f32| {
        backend
            .swin_mhc_residual_f32(skip, branch, a, b, config)
            .map_err(|err| backend_error(&format!("GPU swin {what}"), err))
    };

    // Pad, cyclic shift and partition in one gather.
    let windowed = backend
        .swin_window_partition_f32(
            input,
            g.batch,
            c,
            g.height,
            g.width,
            e.padded_height,
            e.padded_width,
            g.window,
            e.shift,
            config,
        )
        .map_err(|err| backend_error("GPU swin partition", err))?;
    let ln1 = norm("ln1", &windowed, &ln1_gamma, &ln1_beta, c)?;
    let qkv = lin("qkv", &ln1, &qkv_weight, &qkv_bias, c, 3 * c)?;
    let (context, probabilities, active) = backend
        .swin_window_attention_f32(
            &qkv,
            &relative_bias,
            &mask,
            e.total_windows,
            e.spatial_windows,
            g.window,
            c,
            g.heads,
            config,
        )
        .map_err(|err| backend_error("GPU swin attention", err))?;
    let projection = lin("wo", &context, &wo, &wo_bias, c, f)?;
    // The C->N skip projection has no bias in the Rust block.
    let first_skip = match &res_proj {
        Some(res_proj) => {
            let zero_bias = up("residual projection bias", &[f], &vec![0.0; f])?;
            lin("residual projection", &windowed, res_proj, &zero_bias, c, f)?
        }
        None => windowed.clone(),
    };
    let res1 = if let Some(p)=resident { backend.swin_resident_mhc_forward_f32(&first_skip,&projection,&p[bias_slot+1],0,config).map_err(|e|backend_error("GPU swin resident mHC",e))? } else {mix("attention residual", &first_skip, &projection, skip0, branch0)?};
    let ln2 = norm("ln2", &res1, &ln2_gamma, &ln2_beta, f)?;
    let ffn1_raw = lin("ffn1", &ln2, &ffn_w1, &ffn_b1, f, e.hidden)?;
    let ffn1_act = backend
        .activation_f32(&ffn1_raw, activation, config)
        .map_err(|err| backend_error("GPU swin activation", err))?;
    let ffn2 = lin("ffn2", &ffn1_act, &ffn_w2, &ffn_b2, e.hidden, f)?;
    let mixed = if let Some(p)=resident { backend.swin_resident_mhc_forward_f32(&res1,&ffn2,&p[bias_slot+1],1,config).map_err(|e|backend_error("GPU swin resident mHC",e))? } else {mix("ffn residual", &res1, &ffn2, skip1, branch1)?};
    // Unpartition, reverse shift, crop and the source output constrain.
    let output = backend
        .swin_window_unpartition_f32(
            &mixed,
            g.batch,
            f,
            g.height,
            g.width,
            e.padded_height,
            e.padded_width,
            g.window,
            e.shift,
            OUTPUT_CLIP,
            config,
        )
        .map_err(|err| backend_error("GPU swin output", err))?;
    let cache = keep.then(|| DeviceCache {
        resident_generation: None,
        resident_ranges: Vec::new(),
        resident_mhc: resident.map(|p|p[bias_slot+1].clone()),
        geometry: g.clone(),
        windowed,
        ln1,
        qkv,
        context,
        probabilities,
        active,
        first_skip,
        projection,
        res1,
        ln2,
        ffn1_raw,
        ffn1_act,
        ffn2,
        output: output.clone(),
        qkv_weight,
        wo,
        ffn_w1,
        ffn_w2,
        res_proj,
        ln1_gamma,
        ln2_gamma,
    });
    Ok((output, cache))
}

/// Host part of the mHC parameter adjoint for one site: the six logits'
/// gradients from the device reductions `[grad_post_skip, grad_post_branch,
/// grad_p]`, exactly as `swin::mhc_residual_backward` folds them (non-finite
/// logits keep a zero sanitizer derivative).
fn mhc_site_gradient(params: &[f32], site: usize, sums: &[f32], out: &mut [f32]) -> Result<()> {
    let (_, _, p, post_skip, post_branch) =
        crate::swin::mhc_parameters(params, site).map_err(Error::InvalidConfig)?;
    let &[grad_post_skip, grad_post_branch, grad_p] = sums else {
        return Err(Error::InvalidConfig(format!(
            "[swin] mHC site {site} reduction has {} values, expected 3",
            sums.len()
        )));
    };
    let o = site * 6;
    let z = 0.5 * (params[o] + params[o + 3] - params[o + 1] - params[o + 2]);
    if z.is_finite() {
        let dz = 0.5 * grad_p * p * (1.0 - p);
        out[o] = dz;
        out[o + 1] = -dz;
        out[o + 2] = -dz;
        out[o + 3] = dz;
    }
    if params[o + 4].is_finite() {
        out[o + 4] = grad_post_skip * post_skip * (1.0 - 0.5 * post_skip);
    }
    if params[o + 5].is_finite() {
        out[o + 5] = grad_post_branch * post_branch * (1.0 - 0.5 * post_branch);
    }
    Ok(())
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_swin_native_gpu(&self, layer: &Layer, g: &SwinGeometry,
        weights: &super::SwinWeights, input: &B::Tensor, config: ElementwiseLaunchConfig,
        training: bool) -> Result<(B::Tensor,Option<DeviceCache<B::Tensor>>)> {
        let mut cursor=0; let mut ranges=Vec::new(); let mut layout=Vec::new();
        let projection=weights.res_proj.is_some();
        for (slot,part) in weights.serialized_parts().iter().enumerate() {
            let range=cursor..cursor+part.len(); cursor=range.end; ranges.push(range.clone());
            let decay=matches!(slot,0|2|8|10) || (projection && slot==12);
            layout.push(ResidentParameterRange { range, decay, coupled_decay: true,
                moment_slot:Some(ResidentMomentSlot::ChildWeights(slot)),active:true });
        }
        let blob=weights.to_blob().map_err(Error::InvalidConfig)?;
        let (parameters,generation)=if self.resident_parameter_path_enabled() {
            let (p,g)=self.resident_parameter_slices(layer.index,&blob,layout,&ranges,config)?;
            (Some(p),Some(g))
        } else {(None,None)};
        let (output,mut cache)=run(&self.backend,g,weights,input,layer.output_shape,config,training,parameters.as_deref())?;
        if let Some(cache)=cache.as_mut() { cache.resident_generation=generation; cache.resident_ranges=ranges; }
        Ok((output,cache))
    }
    /// Download the input, run the host Swin forward, upload the output. Used
    /// for variants outside [`supported`] and for training networks that take
    /// the CPU backward fallback for another layer.
    pub(crate) fn forward_swin_host(
        &self,
        layer: &Layer,
        spec: &SwinSpec,
        weights: &super::SwinWeights,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::block::SwinForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download swin input", err))?;
        let geometry = geometry(layer, spec);
        let (data, cache) = super::block::swin_forward(&host_input.data, weights, &geometry)
            .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload swin output", err))?;
        Ok((output, cache))
    }

    /// Native `[swin]` backward from the device cache of the training forward:
    /// output constrain, window layout, both mHC residuals, MLP, LayerNorms,
    /// windowed attention (relative bias included), the QKV and optional skip
    /// projections all run on the device, and the input gradient stays there.
    /// Parameter gradients are then downloaded once and staged into the host
    /// train buffers (`updates -= grad`, as `accumulate_swin_updates`).
    pub(crate) fn backward_swin_gpu(
        &self,
        layer: &Layer,
        geometry: &SwinGeometry,
        weights: &super::SwinWeights,
        forward: &crate::runtime::GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let activation = device_activation(&geometry.activation).ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[swin] unsupported GPU activation {:?}", geometry.activation),
            }
        })?;
        let caches = &forward.swin_device_caches;
        let cache = caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing [swin] device forward cache for layer #{}; run a native training forward first",
                    layer.index
                ))
            })?;
        if cache.geometry != *geometry {
            return Err(Error::InvalidConfig(format!(
                "[swin] device cache of layer #{} was built for another geometry",
                layer.index
            )));
        }
        if weights.res_proj.is_some() != cache.res_proj.is_some() {
            return Err(Error::InvalidConfig(format!(
                "[swin] residual projection of layer #{} does not match its device cache",
                layer.index
            )));
        }
        let g = geometry;
        let e = extents(g)?;
        let (c, f, rows) = (g.channels, g.filters, e.rows);
        let be = &self.backend;
        let err = |what: &str| {
            let what = what.to_owned();
            move |e: B::Error| backend_error(&format!("GPU swin backward {what}"), e)
        };
        let (skip0, branch0, p0, post_skip0, post_branch0) =
            crate::swin::mhc_parameters(&weights.mhc, 0).map_err(Error::InvalidConfig)?;
        let (skip1, branch1, p1, post_skip1, post_branch1) =
            crate::swin::mhc_parameters(&weights.mhc, 1).map_err(Error::InvalidConfig)?;

        // LayerNorm adjoint: input gradient from the existing kernel; gamma
        // and beta reduce `dy * xhat` and `dy` over rows (xhat is the
        // unscaled normalization of the cached input).
        let layer_norm = |what: &str,
                          dy: &B::Tensor,
                          x: &B::Tensor,
                          gamma: &B::Tensor,
                          cols: usize|
         -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
            let dx = be
                .layernorm_backward_f32(dy, x, Some(gamma), &[rows, cols], rows, cols, LN_EPS, config)
                .map_err(err(what))?;
            let xhat = be
                .layernorm_forward_f32(x, None, None, &[rows, cols], rows, cols, LN_EPS, config)
                .map_err(err(what))?;
            let dy_xhat = be
                .vector_mul_f32(dy, &xhat, &[rows, cols], config)
                .map_err(err(what))?;
            let dgamma = be.bias_backward_f32(&dy_xhat, cols, 1, config).map_err(err(what))?;
            let dbeta = be.bias_backward_f32(dy, cols, 1, config).map_err(err(what))?;
            Ok((dx, dgamma, dbeta))
        };
        // Explicit-shape add (no bias): the two addends carry different
        // nominal shapes but the same `[rows, cols]` layout.
        let add = |what: &str, x: &B::Tensor, y: &B::Tensor, cols: usize| {
            be.residual_bias_f32(x, y, None, &[rows, cols], rows, cols, config)
                .map_err(err(what))
        };

        // Output constrain (open interval, read off the clamped output), then
        // the partition gather, which is the adjoint of unpartition + crop.
        let d_output = be
            .vit_clamp_backward_f32(&cache.output, &grad, OUTPUT_CLIP, config)
            .map_err(err("output clamp"))?;
        let d_mixed = be
            .swin_window_partition_f32(
                &d_output,
                g.batch,
                f,
                g.height,
                g.width,
                e.padded_height,
                e.padded_width,
                g.window,
                e.shift,
                config,
            )
            .map_err(err("output layout"))?;
        let (d_res1_skip, d_ffn2, mhc1_sums) = if let Some(params)=&cache.resident_mhc {
            be.swin_resident_mhc_backward_f32(&cache.res1,&cache.ffn2,&d_mixed,params,1,config).map_err(err("ffn resident residual"))?
        } else { be
            .swin_mhc_residual_backward_f32(
                &cache.res1,
                &cache.ffn2,
                &d_mixed,
                skip1,
                branch1,
                p1,
                post_skip1,
                post_branch1,
                config,
            )
            .map_err(err("ffn residual"))? };
        let (d_ffn1_act, d_ffn_w2, d_ffn_b2) = be
            .connected_backward_f32(&cache.ffn1_act, &cache.ffn_w2, &d_ffn2, rows, e.hidden, f, config)
            .map_err(err("ffn2"))?;
        let d_ffn1_raw = be
            .activation_backward_f32(&cache.ffn1_raw, &d_ffn1_act, activation, config)
            .map_err(err("activation"))?;
        let (d_ln2, d_ffn_w1, d_ffn_b1) = be
            .connected_backward_f32(&cache.ln2, &cache.ffn_w1, &d_ffn1_raw, rows, f, e.hidden, config)
            .map_err(err("ffn1"))?;
        let (d_res1_norm, d_ln2_gamma, d_ln2_beta) =
            layer_norm("ln2", &d_ln2, &cache.res1, &cache.ln2_gamma, f)?;
        let d_res1 = add("ln2 residual", &d_res1_norm, &d_res1_skip, f)?;
        let (d_first_skip, d_projection, mhc0_sums) = if let Some(params)=&cache.resident_mhc {
            be.swin_resident_mhc_backward_f32(&cache.first_skip,&cache.projection,&d_res1,params,0,config).map_err(err("attention resident residual"))?
        } else { be
            .swin_mhc_residual_backward_f32(
                &cache.first_skip,
                &cache.projection,
                &d_res1,
                skip0,
                branch0,
                p0,
                post_skip0,
                post_branch0,
                config,
            )
            .map_err(err("attention residual"))? };
        let (d_windowed_skip, d_res_proj) = match &cache.res_proj {
            Some(res_proj) => {
                let (dx, dw, _) = be
                    .connected_backward_f32(&cache.windowed, res_proj, &d_first_skip, rows, c, f, config)
                    .map_err(err("residual projection"))?;
                (dx, Some(dw))
            }
            None => (d_first_skip, None),
        };
        let (d_context, d_wo, d_wo_bias) = be
            .connected_backward_f32(&cache.context, &cache.wo, &d_projection, rows, c, f, config)
            .map_err(err("wo"))?;
        let (d_qkv, d_relative_bias) = be
            .swin_window_attention_backward_f32(
                &cache.qkv,
                &cache.probabilities,
                &cache.active,
                &d_context,
                e.total_windows,
                g.window,
                c,
                g.heads,
                config,
            )
            .map_err(err("attention"))?;
        let (d_ln1, d_qkv_weight, d_qkv_bias) = be
            .connected_backward_f32(&cache.ln1, &cache.qkv_weight, &d_qkv, rows, c, 3 * c, config)
            .map_err(err("qkv"))?;
        let (d_windowed_norm, d_ln1_gamma, d_ln1_beta) =
            layer_norm("ln1", &d_ln1, &cache.windowed, &cache.ln1_gamma, c)?;
        let d_windowed = add("ln1 residual", &d_windowed_norm, &d_windowed_skip, c)?;
        // Unpartition, reverse shift and crop: the adjoint of the input gather.
        let d_input = be
            .swin_window_unpartition_f32(
                &d_windowed,
                g.batch,
                c,
                g.height,
                g.width,
                e.padded_height,
                e.padded_width,
                g.window,
                e.shift,
                0.0,
                config,
            )
            .map_err(err("input layout"))?;

        if let Some(generation)=cache.resident_generation {
            let mhc=be.route_concat_f32(&[&mhc0_sums,&mhc1_sums],&[0,0],&[6,6],&[6,6],&[12],1,config).map_err(err("mHC gradient packing"))?;
            let mut gradients=vec![d_qkv_weight,d_qkv_bias,d_wo,d_wo_bias,d_ln1_gamma,d_ln1_beta,
                d_ln2_gamma,d_ln2_beta,d_ffn_w1,d_ffn_b1,d_ffn_w2,d_ffn_b2];
            if let Some(projection)=d_res_proj { gradients.push(projection); }
            gradients.push(d_relative_bias); gradients.push(mhc);
            self.accumulate_resident_parameter_gradients(layer.index,generation,
                cache.resident_ranges.iter().cloned().zip(gradients).collect(),config)?;
            return self.accumulate_previous_gpu_gradient(layer.index,d_input,layer_output_gradients);
        }
        // Parameter staging: one download per tensor at the optimizer boundary.
        let host = |what: &str, t: &B::Tensor| {
            be.download_f32(t)
                .map_err(|e| backend_error(&format!("download swin {what} gradient"), e))
        };
        let mut d_mhc = vec![0.0; SWIN_MHC_DEFAULTS.len()];
        mhc_site_gradient(&weights.mhc, 0, &host("mHC site 0", &mhc0_sums)?, &mut d_mhc)?;
        mhc_site_gradient(&weights.mhc, 1, &host("mHC site 1", &mhc1_sums)?, &mut d_mhc)?;
        let d_res_proj = d_res_proj
            .map(|t| host("residual projection", &t))
            .transpose()?;
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::Swin(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing Swin train buffers for layer #{}",
                    layer.index
                )));
            };
            let u = &mut train.updates;
            let mut staged = vec![
                ("qkv_weight", &mut u.qkv_weight, host("qkv_weight", &d_qkv_weight)?),
                ("qkv_bias", &mut u.qkv_bias, host("qkv_bias", &d_qkv_bias)?),
                ("wo", &mut u.wo, host("wo", &d_wo)?),
                ("wo_bias", &mut u.wo_bias, host("wo_bias", &d_wo_bias)?),
                ("ln1_gamma", &mut u.ln1_gamma, host("ln1_gamma", &d_ln1_gamma)?),
                ("ln1_beta", &mut u.ln1_beta, host("ln1_beta", &d_ln1_beta)?),
                ("ln2_gamma", &mut u.ln2_gamma, host("ln2_gamma", &d_ln2_gamma)?),
                ("ln2_beta", &mut u.ln2_beta, host("ln2_beta", &d_ln2_beta)?),
                ("ffn_w1", &mut u.ffn_w1, host("ffn_w1", &d_ffn_w1)?),
                ("ffn_b1", &mut u.ffn_b1, host("ffn_b1", &d_ffn_b1)?),
                ("ffn_w2", &mut u.ffn_w2, host("ffn_w2", &d_ffn_w2)?),
                ("ffn_b2", &mut u.ffn_b2, host("ffn_b2", &d_ffn_b2)?),
                (
                    "relative_position_bias",
                    &mut u.relative_position_bias,
                    host("relative_position_bias", &d_relative_bias)?,
                ),
                ("mhc", &mut u.mhc, d_mhc),
            ];
            match (&mut u.res_proj, d_res_proj) {
                (Some(dst), Some(gradient)) => staged.push(("res_proj", dst, gradient)),
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "[swin] residual projection gradient mismatch at layer #{}",
                        layer.index
                    )));
                }
            }
            for (name, dst, gradient) in staged {
                if dst.len() != gradient.len() {
                    return Err(Error::InvalidConfig(format!(
                        "[swin] {name} gradient has {} values, expected {}",
                        gradient.len(),
                        dst.len()
                    )));
                }
                for (dst, src) in dst.iter_mut().zip(gradient) {
                    *dst -= src;
                }
            }
        }
        self.accumulate_previous_gpu_gradient(layer.index, d_input, layer_output_gradients)
    }
}
