//! GPU-runtime glue for `[eml_convolutional]`.
use crate::error::{Error, Result};
use crate::model::{ConvolutionalSpec, EmlConvolutionalSpec, Layer, TensorShape};
use crate::runtime::{
    GpuForwardState,
    ConvolutionalParams, EmlConvolutionalParams, EmlForwardCache, GpuBatchNormCache,
    GpuEmlBackwardCache, GpuExecutableNetwork, backend_error, forward_convolutional_with_cache,
    tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_eml_branch_gpu(
        &self,
        owner: usize,
        slot: usize,
        input: &B::Tensor,
        input_shape: TensorShape,
        output_shape: TensorShape,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
        batch_norm_training: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<GpuBatchNormCache<B::Tensor>>)> {
        if spec.batch_normalize != params.batch_norm.is_some() {
            return Err(Error::InvalidConfig(format!(
                "EML branch BN parameter mismatch at layer #{owner}, slot {slot}"
            )));
        }
        let use_batch_stats = batch_norm_training && params.batch_norm.is_some();
        let cached = if use_batch_stats {
            self.cached_convolution_params_raw_at(owner, slot, input_shape.channels, spec, params)?
        } else {
            self.cached_convolution_params_at(owner, slot, input_shape.channels, spec, params)?
        };
        let weights = if use_batch_stats {
            &cached.raw_weights
        } else {
            &cached.weights
        };
        let raw_output = self
            .backend
            .conv2d_f32(
                input,
                weights,
                &tensor_backend_shape(output_shape),
                input_shape.channels,
                input_shape.height,
                input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                config,
            )
            .map_err(|err| backend_error("EML branch convolution", err))?;
        if let Some(batch_norm) = params.batch_norm.as_ref()
            && batch_norm_training
        {
            let (output, cache) = self.forward_batchnorm_training_gpu(
                &raw_output,
                &cached,
                batch_norm,
                output_shape,
                true,
                config,
            )?;
            return Ok((output, Some(cache)));
        }
        let output = self
            .backend
            .bias_add_f32(
                &raw_output,
                &cached.biases,
                spec.filters,
                output_shape.width * output_shape.height,
                config,
            )
            .map_err(|err| backend_error("EML branch bias", err))?;
        Ok((output, None))
    }

    /// Both EML branches on the device plus the native combine. Returns the output, the
    /// device cache for no-BN backward, and, when `host_copies` is set, the host branch
    /// outputs the CPU backward fallback consumes.
    pub(crate) fn forward_eml_convolutional_gpu(
        &self,
        layer: &Layer,
        spec: &EmlConvolutionalSpec,
        eml: &EmlConvolutionalParams,
        current_input: &B::Tensor,
        batch_norm_training: bool,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        GpuEmlBackwardCache<B::Tensor>,
        Option<EmlForwardCache>,
    )> {
        let (input_output, input_batch_norm) = self.forward_eml_branch_gpu(
            layer.index,
            0,
            current_input,
            layer.input_shape,
            layer.output_shape,
            &spec.branch,
            &eml.input,
            batch_norm_training,
            launch_config,
        )?;
        let (self_output, self_batch_norm) = self.forward_eml_branch_gpu(
            layer.index,
            1,
            current_input,
            layer.input_shape,
            layer.output_shape,
            &spec.branch,
            &eml.self_params,
            batch_norm_training,
            launch_config,
        )?;

        let use_residual = spec.residual
            && layer.input_shape.element_count() == layer.output_shape.element_count();
        let residual = use_residual.then_some(current_input);
        let gpu_eml_cache = GpuEmlBackwardCache {
            input_branch_output: input_output.clone(),
            self_branch_output: self_output.clone(),
            input_batch_norm,
            self_batch_norm,
        };
        let output = self
            .backend
            .eml_convolutional_f32(
                &input_output,
                &self_output,
                residual,
                &tensor_backend_shape(layer.output_shape),
                spec.eml_clamp,
                spec.eml_eps,
                spec.eml_scale,
                launch_config,
            )
            .map_err(|err| backend_error("EML branch combination", err))?;

        let mut eml_cache = None;
        if host_copies {
            // The CPU backward fallback consumes the branch outputs,
            // but the public forward still runs both branches and
            // their combine on the selected tensor backend.
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download EML input for fallback", err))?;
            let (input_branch_output, _) = forward_convolutional_with_cache(
                &spec.branch,
                &eml.input,
                &host_input,
                layer.output_shape,
            );
            let (self_branch_output, _) = forward_convolutional_with_cache(
                &spec.branch,
                &eml.self_params,
                &host_input,
                layer.output_shape,
            );
            eml_cache = Some(EmlForwardCache {
                input_branch_output,
                self_branch_output,
            });
        }
        Ok((output, gpu_eml_cache, eml_cache))
    }

    /// Native EML backward over both branches.
    pub(crate) fn backward_eml_convolutional_gpu(
        &self,
        layer: &Layer,
        spec: &EmlConvolutionalSpec,
        params: &EmlConvolutionalParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let cache = forward
            .gpu_eml_caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing EML device cache for GPU layer #{}",
                    layer.index
                ))
            })?;
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for EML layer #{}",
                    layer.index
                ))
            })?
        };
        let (input_delta, self_delta) = self
            .backend
            .eml_convolutional_backward_f32(
                &cache.input_branch_output,
                &cache.self_branch_output,
                &grad,
                &tensor_backend_shape(layer.output_shape),
                spec.eml_clamp,
                spec.eml_eps,
                spec.eml_scale,
                config,
            )
            .map_err(|err| backend_error("EML branch backward", err))?;
        let mut input_gradient = None;
        for (slot, (branch_params, branch_delta, bn_cache)) in [
            (
                &params.input,
                input_delta,
                cache.input_batch_norm.as_ref(),
            ),
            (
                &params.self_params,
                self_delta,
                cache.self_batch_norm.as_ref(),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if branch_params.batch_norm.is_some() != spec.branch.batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "EML BN parameter mismatch at layer #{}, slot {slot}",
                    layer.index
                )));
            }
            let cached = if bn_cache.is_some() {
                self.cached_convolution_params_raw_at(
                    layer.index,
                    slot,
                    layer.input_shape.channels,
                    &spec.branch,
                    branch_params,
                )?
            } else {
                self.cached_convolution_params_at(
                    layer.index,
                    slot,
                    layer.input_shape.channels,
                    &spec.branch,
                    branch_params,
                )?
            };
            let (grad_for_conv, bias_gradient, scale_gradient) =
                if let Some(bn_cache) = bn_cache {
                    let (grad_for_conv, scale_gradient, bias_gradient) = self
                        .backend
                        .batchnorm_train_backward_f32(
                            &branch_delta,
                            &bn_cache.x_hat,
                            &bn_cache.variance,
                            &bn_cache.scales,
                            &tensor_backend_shape(layer.output_shape),
                            config,
                        )
                        .map_err(|err| {
                            backend_error("EML batch-norm backward", err)
                        })?;
                    (grad_for_conv, bias_gradient, Some(scale_gradient))
                } else if let Some(batch_norm) = branch_params.batch_norm.as_ref() {
                    let raw_output = self
                        .backend
                        .conv2d_f32(
                            current_input,
                            &cached.raw_weights,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.branch.filters,
                            spec.branch.size,
                            spec.branch.stride_x,
                            spec.branch.stride_y,
                            spec.branch.dilation,
                            spec.branch.padding,
                            spec.branch.groups.max(1),
                            config,
                        )
                        .map_err(|err| {
                            backend_error("EML rolling-BN raw convolution", err)
                        })?;
                    let scale_gradient = self.batch_norm_scale_gradient_gpu(
                        &raw_output,
                        &branch_delta,
                        batch_norm,
                        layer.output_shape,
                    )?;
                    let bias_gradient = self
                        .backend
                        .bias_backward_f32(
                            &branch_delta,
                            spec.branch.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("EML rolling-BN bias backward", err)
                        })?;
                    let grad_for_conv = if let Some(scale) =
                        cached.batch_norm_scales.as_ref()
                    {
                        self.backend
                            .channel_scale_f32(
                                &branch_delta,
                                scale,
                                spec.branch.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                config,
                            )
                            .map_err(|err| {
                                backend_error("EML rolling-BN gradient scale", err)
                            })?
                    } else {
                        branch_delta.clone()
                    };
                    (grad_for_conv, bias_gradient, Some(scale_gradient))
                } else {
                    let bias_gradient = self
                        .backend
                        .bias_backward_f32(
                            &branch_delta,
                            spec.branch.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("EML branch bias backward", err)
                        })?;
                    (branch_delta.clone(), bias_gradient, None)
                };
            let weight_gradient = self
                .backend
                .conv2d_backward_weights_f32(
                    current_input,
                    &grad_for_conv,
                    &[
                        spec.branch.filters,
                        layer.input_shape.channels / spec.branch.groups.max(1),
                        spec.branch.size,
                        spec.branch.size,
                    ],
                    layer.input_shape.channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.branch.filters,
                    spec.branch.size,
                    spec.branch.stride_x,
                    spec.branch.stride_y,
                    spec.branch.dilation,
                    spec.branch.padding,
                    spec.branch.groups.max(1),
                    config,
                )
                .map_err(|err| backend_error("EML branch weight backward", err))?;
            let branch_input_gradient = self
                .backend
                .conv2d_backward_input_f32(
                    &grad_for_conv,
                    &cached.raw_weights,
                    &tensor_backend_shape(layer.input_shape),
                    layer.input_shape.channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.branch.filters,
                    spec.branch.size,
                    spec.branch.stride_x,
                    spec.branch.stride_y,
                    spec.branch.dilation,
                    spec.branch.padding,
                    spec.branch.groups.max(1),
                    config,
                )
                .map_err(|err| backend_error("EML branch input backward", err))?;
            self.accumulate_eml_device_gradients(
                layer.index,
                slot,
                weight_gradient,
                bias_gradient,
                scale_gradient,
            )?;
            input_gradient = Some(match input_gradient {
                Some(accumulated) => self
                    .backend
                    .saxpy_f32(1.0, &branch_input_gradient, &accumulated, config)
                    .map_err(|err| backend_error("EML input-gradient sum", err))?,
                None => branch_input_gradient,
            });
        }
        let mut input_gradient = input_gradient.expect("two EML branches");
        if spec.residual
            && layer.input_shape.element_count() == layer.output_shape.element_count()
        {
            input_gradient = self
                .backend
                .saxpy_f32(1.0, &grad, &input_gradient, config)
                .map_err(|err| backend_error("EML residual input gradient", err))?;
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
