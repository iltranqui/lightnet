//! GPU-runtime glue for `[convolutional]`.
use crate::error::{Error, Result};
use crate::model::{ConvolutionalSpec, Layer};
use crate::runtime::{
    GradMode,
    GpuForwardState,
    GpuBatchNormCache, GpuExecutableNetwork, backend_error, gpu_activation_kind,
    resolve_convolutional_owner, resolve_convolutional_params, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, NanInfMode};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native convolution, then batch-statistics batch norm (training) or bias, then
    /// activation or channel normalization. Returns the output, the pre-activation tensor kept
    /// for backward, and the batch-norm cache when batch statistics were used.
    pub(crate) fn forward_convolutional_gpu(
        &self,
        layer: &Layer,
        spec: &ConvolutionalSpec,
        current_input: &B::Tensor,
        batch_norm_training: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        B::Tensor,
        Option<GpuBatchNormCache<B::Tensor>>,
    )> {
        let mut batch_norm_cache = None;
        let conv = resolve_convolutional_params(&self.network, &self.params, layer.index)?;
        let owner = resolve_convolutional_owner(&self.network, layer.index)?;
        let cached = if batch_norm_training && conv.batch_norm.is_some() {
            self.cached_convolution_params_raw(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        } else {
            self.cached_convolution_params(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        };
        // Batch-norm training convolves with the raw weights; the
        // folded (rolling-stat) weights serve inference.
        let weights = if batch_norm_training && conv.batch_norm.is_some() {
            &cached.raw_weights
        } else {
            &cached.weights
        };
        let conv_output = self
            .backend
            .conv2d_f32(
                current_input,
                weights,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                launch_config,
            )
            .map_err(|err| backend_error("convolution", err))?;
        let biased = match conv.batch_norm.as_ref() {
            Some(batch_norm) if batch_norm_training => {
                let (normalized, cache) = self.forward_batchnorm_training_gpu(
                    &conv_output,
                    &cached,
                    batch_norm,
                    layer.output_shape,
                    spec.share_index.is_none(),
                    launch_config,
                )?;
                batch_norm_cache = Some(cache);
                normalized
            }
            _ => self
                .backend
                .bias_add_f32(
                    &conv_output,
                    &cached.biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("convolution bias add", err))?,
        };
        let activation_input = biased.clone();
        let output = if let Some(normalization) = spec.activation.to_channel_normalization() {
            self.backend
                .normalize_channels_f32(
                    &biased,
                    &tensor_backend_shape(layer.output_shape),
                    layer.output_shape.batch,
                    layer.output_shape.channels,
                    layer.output_shape.width * layer.output_shape.height,
                    normalization,
                    launch_config,
                )
                .map_err(|err| {
                    backend_error("convolution channel normalization", err)
                })?
        } else {
            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
            self.backend
                .activation_f32(&biased, activation, launch_config)
                .map_err(|err| backend_error("convolution activation", err))?
        };
        Ok((output, activation_input, batch_norm_cache))
    }

    /// Compact inference forward: native convolution with fused bias and activation.
    pub(crate) fn forward_convolutional_compact_gpu(
        &self,
        layer: &Layer,
        spec: &ConvolutionalSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let conv =
            resolve_convolutional_params(&self.network, &self.params, layer.index)?;
        let owner = resolve_convolutional_owner(&self.network, layer.index)?;
        let cached = self.cached_convolution_params(
            owner,
            layer.input_shape.channels,
            spec,
            conv,
        )?;
        let conv_output = self
            .backend
            .conv2d_f32(
                current_input,
                &cached.weights,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                launch_config,
            )
            .map_err(|err| backend_error("convolution", err))?;
        Ok(if let Some(normalization) = spec.activation.to_channel_normalization() {
            let biased = self
                .backend
                .bias_add_f32(
                    &conv_output,
                    &cached.biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("convolution bias add", err))?;
            self.backend
                .normalize_channels_f32(
                    &biased,
                    &tensor_backend_shape(layer.output_shape),
                    layer.output_shape.batch,
                    layer.output_shape.channels,
                    layer.output_shape.width * layer.output_shape.height,
                    normalization,
                    launch_config,
                )
                .map_err(|err| {
                    backend_error("convolution channel normalization", err)
                })?
        } else {
            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
            self.backend
                .bias_activation_f32(
                    &conv_output,
                    &cached.biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    activation,
                    launch_config,
                )
                .map_err(|err| backend_error("convolution bias + activation", err))?
        })
    }

    /// Native convolution backward.
    pub(crate) fn backward_convolutional_gpu(
        &self,
        layer: &Layer,
        spec: &ConvolutionalSpec,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let conv =
            resolve_convolutional_params(&self.network, &self.params, layer.index)?;
        let owner = resolve_convolutional_owner(&self.network, layer.index)?;
        // The batch-statistics BN backward only needs the raw tensors.
        let cached = if forward.gpu_batch_norm_caches[layer.index].is_some() {
            self.cached_convolution_params_raw(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        } else {
            self.cached_convolution_params(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        };
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU activation cache for convolutional layer #{}",
                    layer.index
                ))
            })?;
        let try_fix_nan = self.network.net.try_fix_nan;
        // C++ `try_fix_nan`: clamp the output delta to [-1, 1]
        // before the activation gradient.
        let grad = if try_fix_nan {
            self.backend
                .constrain_f32(&grad, &tensor_backend_shape(layer.output_shape), 1.0, config)
                .map_err(|err| backend_error("GPU try_fix_nan delta clamp", err))?
        } else {
            grad
        };
        let delta = if let Some(normalization) =
            spec.activation.to_channel_normalization()
        {
            let output = &forward.gpu_layer_outputs[layer.index];
            self.backend
                .normalize_channels_backward_f32(
                    activation_input,
                    output,
                    &grad,
                    &tensor_backend_shape(layer.output_shape),
                    layer.output_shape.batch,
                    layer.output_shape.channels,
                    layer.output_shape.width * layer.output_shape.height,
                    normalization,
                    GradMode::from_env() == GradMode::Cpp,
                    config,
                )
                .map_err(|err| {
                    backend_error("GPU convolution channel-normalization backward", err)
                })?
        } else {
            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
            self.backend
                .activation_backward_f32(activation_input, &grad, activation, config)
                .map_err(|err| backend_error("GPU convolution activation backward", err))?
        };
        let current_input =
            if layer.index == 0 {
                &forward.input_gpu
            } else {
                forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU layer output for convolutional layer #{}",
                    layer.index
                ))
            })?
            };
        let (bias_grad, scale_grad, grad_for_conv) = if let Some(cache) =
            forward.gpu_batch_norm_caches[layer.index].as_ref()
        {
            // Batch-statistics batch norm: exact dx, dscale, dbias.
            let (dx, dscale, dbias) = self
                .backend
                .batchnorm_train_backward_f32(
                    &delta,
                    &cache.x_hat,
                    &cache.variance,
                    &cache.scales,
                    &tensor_backend_shape(layer.output_shape),
                    config,
                )
                .map_err(|err| backend_error("GPU batch-norm backward", err))?;
            (dbias, Some(dscale), dx)
        } else {
            let bias_grad = self
                .backend
                .bias_backward_f32(
                    &delta,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    config,
                )
                .map_err(|err| backend_error("GPU convolution bias backward", err))?;
            let scale_grad = if let Some(batch_norm) = conv.batch_norm.as_ref() {
                let raw_output = self
                    .backend
                    .conv2d_f32(
                        current_input,
                        &cached.raw_weights,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.filters,
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.dilation,
                        spec.padding,
                        spec.groups.max(1),
                        config,
                    )
                    .map_err(|err| backend_error("GPU batch-norm raw convolution", err))?;
                let scale_grad = self.batch_norm_scale_gradient_gpu(
                    &raw_output,
                    &delta,
                    batch_norm,
                    layer.output_shape,
                )?;
                Some(scale_grad)
            } else {
                None
            };
            let grad_for_conv = if let Some(scale) = cached.batch_norm_scales.as_ref() {
                self.backend
                    .channel_scale_f32(
                        &delta,
                        scale,
                        spec.filters,
                        layer.output_shape.width * layer.output_shape.height,
                        config,
                    )
                    .map_err(|err| backend_error("GPU batch-norm delta scale", err))?
            } else {
                delta.clone()
            };
            (bias_grad, scale_grad, grad_for_conv)
        };
        let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
        let weight_grad = self
            .backend
            .conv2d_backward_weights_f32(
                current_input,
                &grad_for_conv,
                &[spec.filters, grouped_channels, spec.size, spec.size],
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                config,
            )
            .map_err(|err| backend_error("GPU convolution weight backward", err))?;
        let input_grad = self
            .backend
            .conv2d_backward_input_f32(
                &grad_for_conv,
                &cached.raw_weights,
                &tensor_backend_shape(layer.input_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                config,
            )
            .map_err(|err| backend_error("GPU convolution input backward", err))?;
        let input_grad = if try_fix_nan {
            self.backend
                .sanitize_nan_inf_f32(
                    &input_grad,
                    &tensor_backend_shape(layer.input_shape),
                    NanInfMode::Reset,
                    config,
                )
                .map_err(|err| backend_error("GPU try_fix_nan input delta reset", err))?
        } else {
            input_grad
        };
        self.accumulate_device_train_buffer(owner, weight_grad, bias_grad, scale_grad)?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
