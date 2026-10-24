//! GPU-runtime glue for `[deform_conv]`.
use crate::error::{Error, Result};
use crate::model::{DeformConvSpec, Layer};
use crate::runtime::{
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
}
