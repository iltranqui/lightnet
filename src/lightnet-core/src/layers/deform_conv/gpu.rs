//! GPU-runtime glue for `[deform_conv]`.
use crate::error::{Error, Result};
use crate::model::{DeformConvSpec, Layer, TensorShape};
use crate::runtime::{
    LayerTrainBuffers,
    GpuForwardState,
    DeformConvParams, DeformConvRuntimeCache, DeformDeviceCache, GpuExecutableNetwork,
    backend_error, forward_deform_conv_with_cache, gpu_activation_kind, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

/// Output of one `[deform_conv]` forward: the activated tensor, the pre-activation tensor
/// backward needs, and whichever cache the chosen path produced.
pub(crate) struct DeformConvGpuForward<T> {
    pub(crate) output: T,
    pub(crate) activation_input: T,
    pub(crate) device_cache: Option<DeformDeviceCache<T>>,
    pub(crate) host_cache: Option<DeformConvRuntimeCache>,
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Fully native path: sanitize, offset predictor, offset envelope, optional mask,
    /// bilinear accumulate, bias and activation, all on the device.
    pub(crate) fn forward_deform_conv_device_gpu(
        &self,
        layer: &Layer,
        spec: &DeformConvSpec,
        params: &DeformConvParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<DeformConvGpuForward<B::Tensor>> {
        let geometry = lightnet_cuda::host_ops::deform_conv::DeformGeometry {
            batch: layer.input_shape.batch,
            channels: layer.input_shape.channels,
            filters: spec.filters,
            channels_per_group: layer.input_shape.channels / spec.groups.max(1),
            filters_per_group: spec.filters / spec.groups.max(1),
            input_height: layer.input_shape.height,
            input_width: layer.input_shape.width,
            output_height: layer.output_shape.height,
            output_width: layer.output_shape.width,
            size: spec.size,
            stride_x: spec.stride_x,
            stride_y: spec.stride_y,
            padding: spec.padding,
            dilation: spec.dilation,
        };
        let kernels = spec.size * spec.size;
        let plane = geometry.output_width * geometry.output_height;
        let in_shape = tensor_backend_shape(layer.input_shape);
        let out_shape = tensor_backend_shape(layer.output_shape);
        let sanitized = self
            .backend
            .deform_sanitize_f32(current_input, &in_shape, launch_config)
            .map_err(|err| backend_error("deform_conv sanitize", err))?;
        let offset_w = self
            .backend
            .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
            .map_err(|err| backend_error("upload deform offset weights", err))?;
        let offset_b = self
            .backend
            .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
            .map_err(|err| backend_error("upload deform offset biases", err))?;
        let predictor_shape = [
            geometry.batch,
            2 * kernels,
            geometry.output_height,
            geometry.output_width,
        ];
        let raw_offsets = self
            .backend
            .deform_predictor_f32(
                &sanitized,
                &offset_w,
                &offset_b,
                &predictor_shape,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
                2 * kernels,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                geometry.output_height,
                geometry.output_width,
                launch_config,
            )
            .map_err(|err| backend_error("deform offset predictor", err))?;
        let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
        let (mean, inv_std, stddev, normalize) = {
            let (m, i, n) = self
                .backend
                .deform_offset_stats(
                    &raw_offsets,
                    2 * kernels * plane,
                    max_offset,
                    launch_config,
                )
                .map_err(|err| backend_error("deform offset stats", err))?;
            // `stddev` is not exposed by the trait signature;
            // recover it on the host from the small [batch] tensor.
            let inv = self
                .backend
                .download_f32(&i)
                .map_err(|err| backend_error("download deform inv_std", err))?;
            let std_host: Vec<f32> = inv
                .iter()
                .map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 })
                .collect();
            let stddev = self
                .backend
                .upload_f32(&[std_host.len()], &std_host)
                .map_err(|err| backend_error("upload deform stddev", err))?;
            (m, i, stddev, n)
        };
        let offsets = self
            .backend
            .deform_offset_envelope(
                &raw_offsets,
                &mean,
                &inv_std,
                &normalize,
                2 * kernels,
                plane,
                max_offset,
                &predictor_shape,
                launch_config,
            )
            .map_err(|err| backend_error("deform offset envelope", err))?;
        let (masks, mask_logits) = if spec.use_mask {
            let weights = params.mask_weights.as_ref().ok_or_else(|| {
                Error::InvalidConfig(
                    "[deform_conv]: missing mask weights".to_owned(),
                )
            })?;
            let biases = params.mask_biases.as_ref().ok_or_else(|| {
                Error::InvalidConfig(
                    "[deform_conv]: missing mask biases".to_owned(),
                )
            })?;
            let w = self
                .backend
                .upload_f32(&[weights.len()], weights)
                .map_err(|err| backend_error("upload deform mask weights", err))?;
            let b = self
                .backend
                .upload_f32(&[biases.len()], biases)
                .map_err(|err| backend_error("upload deform mask biases", err))?;
            let logits_shape = [
                geometry.batch,
                kernels,
                geometry.output_height,
                geometry.output_width,
            ];
            let logits = self
                .backend
                .deform_predictor_f32(
                    &sanitized,
                    &w,
                    &b,
                    &logits_shape,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                    kernels,
                    geometry.size,
                    geometry.stride_x,
                    geometry.stride_y,
                    geometry.padding,
                    geometry.dilation,
                    geometry.output_height,
                    geometry.output_width,
                    launch_config,
                )
                .map_err(|err| backend_error("deform mask predictor", err))?;
            let masks = self
                .backend
                .deform_mask_sigmoid(&logits, &logits_shape, launch_config)
                .map_err(|err| backend_error("deform mask sigmoid", err))?;
            (Some(masks), Some(logits))
        } else {
            (None, None)
        };
        let main_w = self
            .backend
            .upload_f32(&[params.main.weights.len()], &params.main.weights)
            .map_err(|err| backend_error("upload deform main weights", err))?;
        let empty_masks = self
            .backend
            .zeros_f32(&[1])
            .map_err(|err| backend_error("deform empty masks", err))?;
        let raw = self
            .backend
            .deform_conv_accumulate_f32(
                &sanitized,
                &offsets,
                masks.as_ref().unwrap_or(&empty_masks),
                &main_w,
                geometry,
                masks.is_some(),
                &out_shape,
                launch_config,
            )
            .map_err(|err| backend_error("deform accumulate", err))?;
        let main_bias = self
            .backend
            .upload_f32(&[params.main.biases.len()], &params.main.biases)
            .map_err(|err| backend_error("upload deform main biases", err))?;
        let biased = self
            .backend
            .bias_add_f32(&raw, &main_bias, spec.filters, plane, launch_config)
            .map_err(|err| backend_error("deform bias add", err))?;
        let activation_input = biased.clone();
        let device_cache = DeformDeviceCache {
            input_sanitized: sanitized,
            raw_offsets,
            mean,
            inv_std,
            stddev,
            normalize,
            offsets,
            masks,
            mask_logits,
        };
        let output = self
            .backend
            .activation_f32(
                &biased,
                gpu_activation_kind(layer.index, &spec.activation)?,
                launch_config,
            )
            .map_err(|err| backend_error("deform activation", err))?;
        Ok(DeformConvGpuForward {
            output,
            activation_input,
            device_cache: Some(device_cache),
            host_cache: None,
        })
    }

    /// Host-reference path for networks that need the CPU backward fallback: download, run
    /// the CPU forward, upload the activated output and the pre-activation cache.
    pub(crate) fn forward_deform_conv_host_gpu(
        &self,
        layer: &Layer,
        spec: &DeformConvSpec,
        params: &DeformConvParams,
        current_input: &B::Tensor,
    ) -> Result<DeformConvGpuForward<B::Tensor>> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download deform_conv input", err))?;
        let (host_output, host_activation, cache) = forward_deform_conv_with_cache(
            spec,
            params,
            &host_input,
            layer.output_shape,
        )?;
        let host_cache = Some(cache);
        let activation_input = self
            .backend
            .upload_f32(
                &tensor_backend_shape(layer.output_shape),
                &host_activation.data,
            )
            .map_err(|err| backend_error("upload deform_conv activation cache", err))?;
        let output = self
            .backend
            .upload_f32(
                &tensor_backend_shape(layer.output_shape),
                &host_output.data,
            )
            .map_err(|err| backend_error("upload deform_conv output", err))?;
        Ok(DeformConvGpuForward {
            output,
            activation_input,
            device_cache: None,
            host_cache,
        })
    }

    /// Native DeformConv backward.
    pub(crate) fn backward_deform_conv_gpu(
        &self,
        layer: &Layer,
        spec: &DeformConvSpec,
        params: &DeformConvParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        // Native path. Device forward/backward; the only host hop is
        // the batch-coupled offset-normalization Jacobian.
        let (main_weights, offset_weights) = (
            params.main.weights.clone(),
            params.offset_weights.clone(),
        );
        let (mask_weights, mask_biases) =
            (params.mask_weights.clone(), params.mask_biases.clone());
        // Clone the device intermediates out of the RefCell so the
        // following `&mut self` calls can borrow freely.
        let (input_sanitized, raw_offsets, mean, inv_std, stddev, normalize, offsets, masks, mask_logits) = {
            let caches = self.deform_device_caches.borrow();
            let c = caches[layer.index].as_ref().ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing device [deform_conv] cache for layer #{}",
                    layer.index
                ))
            })?;
            (
                c.input_sanitized.clone(),
                c.raw_offsets.clone(),
                c.mean.clone(),
                c.inv_std.clone(),
                c.stddev.clone(),
                c.normalize.clone(),
                c.offsets.clone(),
                c.masks.clone(),
                c.mask_logits.clone(),
            )
        };
        let plane = layer.output_shape.width * layer.output_shape.height;
        let kernels = spec.size * spec.size;
        let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
        // Offset/mask tensors are `[B, 2K, OH, OW]` and `[B, K, OH, OW]`;
        // they are NOT the layer output shape.
        let offsets_shape = tensor_backend_shape(TensorShape::new(
            layer.output_shape.width,
            layer.output_shape.height,
            2 * kernels,
            layer.input_shape.batch,
        ));
        let masks_shape = tensor_backend_shape(TensorShape::new(
            layer.output_shape.width,
            layer.output_shape.height,
            kernels,
            layer.input_shape.batch,
        ));
        let geometry = lightnet_cuda::host_ops::deform_conv::DeformGeometry {
            batch: layer.input_shape.batch,
            channels: layer.input_shape.channels,
            filters: spec.filters,
            channels_per_group: layer.input_shape.channels / spec.groups.max(1),
            filters_per_group: spec.filters / spec.groups.max(1),
            input_height: layer.input_shape.height,
            input_width: layer.input_shape.width,
            output_height: layer.output_shape.height,
            output_width: layer.output_shape.width,
            size: spec.size,
            stride_x: spec.stride_x,
            stride_y: spec.stride_y,
            padding: spec.padding,
            dilation: spec.dilation,
        };
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing activation cache for [deform_conv] layer #{}",
                    layer.index
                ))
            })?;
        let kind = gpu_activation_kind(layer.index, &spec.activation)?;
        let delta = self
            .backend
            .activation_backward_f32(activation_input, &grad, kind, config)
            .map_err(|err| backend_error("deform activation backward", err))?;
        let d_main_bias = self
            .backend
            .bias_backward_f32(&delta, spec.filters, plane, config)
            .map_err(|err| backend_error("deform bias backward", err))?;
        let empty = self
            .backend
            .zeros_f32(&[1])
            .map_err(|err| backend_error("deform empty masks", err))?;
        let main_w_device = self
            .backend
            .upload_f32(&[main_weights.len()], &main_weights)
            .map_err(|err| backend_error("upload deform main weights", err))?;
        let (d_input_a, d_offsets, d_masks, d_main_w) = self
            .backend
            .deform_conv_accumulate_backward_f32(
                &input_sanitized,
                &offsets,
                masks.as_ref().unwrap_or(&empty),
                &main_w_device,
                &delta,
                geometry,
                masks.is_some(),
                config,
            )
            .map_err(|err| backend_error("deform accumulate backward", err))?;
        let d_pre = self
            .backend
            .deform_envelope_pre_backward_f32(
                &raw_offsets,
                &mean,
                &inv_std,
                &normalize,
                &d_offsets,
                2 * kernels,
                plane,
                max_offset,
                &offsets_shape,
                config,
            )
            .map_err(|err| backend_error("deform envelope backward", err))?;
        let d_stage1 = {
            let d_pre_host = self
                .backend
                .download_f32(&d_pre)
                .map_err(|err| backend_error("download deform d_pre", err))?;
            let mean_host = self
                .backend
                .download_f32(&mean)
                .map_err(|err| backend_error("download deform mean", err))?;
            let std_host = self
                .backend
                .download_f32(&stddev)
                .map_err(|err| backend_error("download deform stddev", err))?;
            let normalize_host = self
                .backend
                .download_f32(&normalize)
                .map_err(|err| backend_error("download deform normalize", err))?;
            let raw_host = self
                .backend
                .download_f32(&raw_offsets)
                .map_err(|err| backend_error("download deform raw offsets", err))?;
            let stage1: Vec<f32> = raw_host
                .iter()
                .copied()
                .map(lightnet_cuda::host_ops::deform_conv::sanitize)
                .collect();
            let grads =
                lightnet_cuda::host_ops::deform_conv::offset_normalization_backward(
                    &stage1,
                    &mean_host,
                    &std_host,
                    &normalize_host,
                    &d_pre_host,
                    2 * kernels * plane,
                )
                .map_err(Error::InvalidConfig)?;
            let scaled: Vec<f32> = grads
                .iter()
                .zip(&raw_host)
                .map(|(&d, &raw)| {
                    let slope = if raw.is_finite() && raw.abs() < 10_000.0 {
                        1.0
                    } else {
                        0.0
                    };
                    d * slope
                })
                .collect();
            self.backend
                .upload_f32(&offsets_shape, &scaled)
                .map_err(|err| backend_error("upload deform d_stage1", err))?
        };
        let off_w_device = self
            .backend
            .upload_f32(&[offset_weights.len()], &offset_weights)
            .map_err(|err| backend_error("upload deform offset weights", err))?;
        let (d_input_o, d_off_w, d_off_b) =
            self.backend
                .deform_predictor_backward_f32(
                    &input_sanitized,
                    &off_w_device,
                    &d_stage1,
                    layer.input_shape.batch,
                    layer.input_shape.channels,
                    2 * kernels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    layer.output_shape.height,
                    layer.output_shape.width,
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    spec.dilation,
                    config,
                )
                .map_err(|err| {
                    backend_error("deform offset predictor backward", err)
                })?;
        let (d_input_m, d_mask_w, d_mask_b) = match (
            mask_logits.as_ref(),
            mask_weights.as_ref(),
            mask_biases.as_ref(),
        ) {
            (Some(logits), Some(weights), Some(_)) => {
                let d_logits = self
                    .backend
                    .deform_mask_sigmoid_backward_f32(
                        logits,
                        &d_masks,
                        &masks_shape,
                        config,
                    )
                    .map_err(|err| {
                        backend_error("deform mask sigmoid backward", err)
                    })?;
                let w_device = self
                    .backend
                    .upload_f32(&[weights.len()], weights)
                    .map_err(|err| {
                        backend_error("upload deform mask weights", err)
                    })?;
                let (d_in, d_w, d_b) = self
                    .backend
                    .deform_predictor_backward_f32(
                        &input_sanitized,
                        &w_device,
                        &d_logits,
                        layer.input_shape.batch,
                        layer.input_shape.channels,
                        kernels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        layer.output_shape.height,
                        layer.output_shape.width,
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        spec.dilation,
                        config,
                    )
                    .map_err(|err| {
                        backend_error("deform mask predictor backward", err)
                    })?;
                (Some(d_in), Some(d_w), Some(d_b))
            }
            _ => (None, None, None),
        };
        let mut sum = self
            .backend
            .vector_add_f32(&d_input_a, &d_input_o, config)
            .map_err(|err| backend_error("deform input grad sum", err))?;
        if let Some(d_in) = d_input_m.as_ref() {
            sum = self
                .backend
                .vector_add_f32(&sum, d_in, config)
                .map_err(|err| backend_error("deform input grad sum", err))?;
        }
        // The raw (unsanitized) layer input drives the final slope.
        let raw_input = if layer.index == 0 {
            forward.input_gpu.clone()
        } else {
            forward.gpu_layer_outputs[layer.index - 1].clone()
        };
        let d_input = self
            .backend
            .deform_scale_input_gradient_f32(
                &raw_input,
                &sum,
                &tensor_backend_shape(layer.input_shape),
                config,
            )
            .map_err(|err| backend_error("deform input slope", err))?;
        {
            let d_main = self
                .backend
                .download_f32(&d_main_w)
                .map_err(|err| backend_error("download deform main w grad", err))?;
            let d_bias = self
                .backend
                .download_f32(&d_main_bias)
                .map_err(|err| backend_error("download deform main b grad", err))?;
            let d_ow = self
                .backend
                .download_f32(&d_off_w)
                .map_err(|err| backend_error("download deform off w grad", err))?;
            let d_ob = self
                .backend
                .download_f32(&d_off_b)
                .map_err(|err| backend_error("download deform off b grad", err))?;
            let d_mw = match d_mask_w.as_ref() {
                Some(t) => Some(
                    self.backend
                        .download_f32(t)
                        .map_err(|err| backend_error("download deform mask w", err))?,
                ),
                None => None,
            };
            let d_mb = match d_mask_b.as_ref() {
                Some(t) => Some(
                    self.backend
                        .download_f32(t)
                        .map_err(|err| backend_error("download deform mask b", err))?,
                ),
                None => None,
            };
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let LayerTrainBuffers::DeformConv(train) = &mut buffers[layer.index] else {
                return Err(Error::InvalidConfig(format!(
                    "[deform_conv] layer #{} has no train buffers",
                    layer.index
                )));
            };
            for (dst, src) in train.main.weight_updates.iter_mut().zip(d_main) {
                *dst += src;
            }
            for (dst, src) in train.main.bias_updates.iter_mut().zip(d_bias) {
                *dst += src;
            }
            for (dst, src) in train.offset_weight_updates.iter_mut().zip(d_ow) {
                *dst += src;
            }
            for (dst, src) in train.offset_bias_updates.iter_mut().zip(d_ob) {
                *dst += src;
            }
            if let (Some(dw), Some(db)) = (d_mw, d_mb) {
                if let (Some(dst_w), Some(dst_b)) = (
                    train.mask_weight_updates.as_mut(),
                    train.mask_bias_updates.as_mut(),
                ) {
                    for (dst, src) in dst_w.iter_mut().zip(dw) {
                        *dst += src;
                    }
                    for (dst, src) in dst_b.iter_mut().zip(db) {
                        *dst += src;
                    }
                }
            }
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            d_input,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
