//! GPU-runtime glue for `[moe_convolutional]`.
use crate::error::{Error, Result};
use crate::model::{Layer, MoeSpec};
use crate::runtime::{
    BATCH_NORM_EPSILON,
    validate_batchnorm_params,
    GpuForwardState,
    GpuExecutableNetwork, GpuMoeBackwardCache, MoeForwardCache, MoeParams, backend_error,
    forward_convolutional_with_cache, gpu_activation_kind, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Dense-masked MoE forward: router gates, every expert convolution and the gate-weighted
    /// blend all run on the device. Returns the output, the device cache for no-BN backward,
    /// and, when `host_copies` is set, the host cache the CPU backward fallback consumes.
    pub(crate) fn forward_moe_gpu(
        &self,
        layer: &Layer,
        spec: &MoeSpec,
        moe: &MoeParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        GpuMoeBackwardCache<B::Tensor>,
        Option<MoeForwardCache>,
    )> {
        let host_input = if host_copies {
            Some(
                self.download_tensor(current_input, layer.input_shape)
                    .map_err(|err| backend_error("download MoE router input", err))?,
            )
        } else {
            None
        };
        let batch = layer.input_shape.batch;
        let router_weights = self
            .backend
            .upload_f32(
                &[spec.experts, layer.input_shape.channels],
                &moe.router_weights,
            )
            .map_err(|err| backend_error("upload MoE router weights", err))?;
        let router_biases = self
            .backend
            .upload_f32(&[spec.experts], &moe.router_biases)
            .map_err(|err| backend_error("upload MoE router biases", err))?;
        let router_gates = self
            .backend
            .moe_router_gates_f32(
                current_input,
                &router_weights,
                &router_biases,
                batch,
                layer.input_shape.channels,
                layer.input_shape.width * layer.input_shape.height,
                spec.experts,
                spec.top_k,
                crate::moe_conv::router_temperature(spec),
                launch_config,
            )
            .map_err(|err| backend_error("compute MoE router gates", err))?;
        let host_experts = if host_copies {
            Some(
                self.backend
                    .download_f32(&router_gates)
                    .map_err(|err| backend_error("download MoE router gates", err))?,
            )
        } else {
            None
        };
        let pooled_gpu = self
            .backend
            .moe_router_pool_f32(
                current_input,
                batch,
                layer.input_shape.channels,
                layer.input_shape.width * layer.input_shape.height,
                launch_config,
            )
            .map_err(|err| backend_error("pool MoE router input", err))?;
        let per_image = layer.output_shape.element_count();
        let mut blended_gpu = None;
        let mut host_cache = host_copies.then(|| MoeForwardCache {
            gates: host_experts
                .as_ref()
                .expect("host cache requires host gates")
                .clone(),
            expert_outputs: Vec::with_capacity(spec.experts),
            expert_activation_inputs: Vec::with_capacity(spec.experts),
            pooled: crate::moe_conv::pool_for_router(
                &host_input
                    .as_ref()
                    .expect("host cache requires host input")
                    .data,
                batch,
                layer.input_shape.channels,
                layer.input_shape.width * layer.input_shape.height,
            ),
        });
        let mut expert_outputs_gpu = Vec::with_capacity(spec.experts);
        let mut expert_activation_inputs_gpu = Vec::with_capacity(spec.experts);
        for (slot, expert) in moe.experts.iter().enumerate() {
            let cached = self.cached_convolution_params_at(
                layer.index,
                slot,
                layer.input_shape.channels,
                &spec.expert,
                expert,
            )?;
            let raw_output = self
                .backend
                .conv2d_f32(
                    current_input,
                    &cached.weights,
                    &tensor_backend_shape(layer.output_shape),
                    layer.input_shape.channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.expert.filters,
                    spec.expert.size,
                    spec.expert.stride_x,
                    spec.expert.stride_y,
                    spec.expert.dilation,
                    spec.expert.padding,
                    spec.expert.groups.max(1),
                    launch_config,
                )
                .map_err(|err| backend_error("MoE expert convolution", err))?;
            let biased = self
                .backend
                .bias_add_f32(
                    &raw_output,
                    &cached.biases,
                    spec.expert.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("MoE expert bias", err))?;
            let activation = gpu_activation_kind(layer.index, &spec.expert.activation)?;
            let output_gpu = self
                .backend
                .activation_f32(&biased, activation, launch_config)
                .map_err(|err| backend_error("MoE expert activation", err))?;
            let sample_gates = self
                .backend
                .moe_gate_column_f32(
                    &router_gates,
                    batch,
                    spec.experts,
                    slot,
                    launch_config,
                )
                .map_err(|err| backend_error("select MoE expert gates", err))?;
            // Treat each sample as one channel with `per_image`
            // elements, applying its scalar gate to the full expert
            // output without moving that output back to the host.
            let weighted = self
                .backend
                .channel_scale_f32(
                    &output_gpu,
                    &sample_gates,
                    batch,
                    per_image,
                    launch_config,
                )
                .map_err(|err| backend_error("scale MoE expert output", err))?;
            blended_gpu = Some(match blended_gpu {
                Some(accumulated) => self
                    .backend
                    .saxpy_f32(1.0, &weighted, &accumulated, launch_config)
                    .map_err(|err| backend_error("blend MoE expert outputs", err))?,
                None => weighted,
            });
            expert_outputs_gpu.push(output_gpu.clone());
            expert_activation_inputs_gpu.push(biased.clone());
            if let Some(cache) = &mut host_cache {
                // The CPU fallback needs host expert outputs and
                // activation caches. Device backward uses the
                // tensors retained above instead.
                let (cpu_output, activation_input) = forward_convolutional_with_cache(
                    &spec.expert,
                    expert,
                    host_input.as_ref().expect("host cache requires host input"),
                    layer.output_shape,
                );
                cache.expert_outputs.push(cpu_output);
                cache.expert_activation_inputs.push(activation_input);
            }
        }
        let moe_cache = host_cache;
        let gpu_moe_cache = GpuMoeBackwardCache {
            gates: router_gates,
            pooled: pooled_gpu,
            expert_outputs: expert_outputs_gpu,
            expert_activation_inputs: expert_activation_inputs_gpu,
        };
        let output = blended_gpu.ok_or_else(|| {
            Error::InvalidConfig(format!(
                "layer #{} ([moe_convolutional]) has no expert outputs",
                layer.index
            ))
        })?;
        Ok((output, gpu_moe_cache, moe_cache))
    }

    /// Native MoE backward over the router and every expert.
    pub(crate) fn backward_moe_gpu(
        &self,
        layer: &Layer,
        spec: &MoeSpec,
        params: &MoeParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let cache = forward
            .gpu_moe_caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing MoE device cache for GPU layer #{}",
                    layer.index
                ))
            })?;
        let batch = layer.input_shape.batch;
        let channels = layer.input_shape.channels;
        let experts = spec.experts;
        let output_spatial = layer.output_shape.width * layer.output_shape.height;
        let per_image = layer.output_shape.element_count();
        if experts == 0
            || cache.expert_activation_inputs.len() != experts
            || params.experts.len() != experts
            || params
                .experts
                .iter()
                .any(|expert| expert.batch_norm.is_some() != spec.expert.batch_normalize)
            || params.router_weights.len() != experts * channels
            || params.router_biases.len() != experts
        {
            return Err(Error::InvalidConfig(format!(
                "MoE GPU backward cache or parameter shape mismatch at layer #{}",
                layer.index
            )));
        }
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for MoE layer #{}",
                    layer.index
                ))
            })?
        };
        let dlogit = self
            .backend
            .moe_router_backward_f32(
                &cache.gates,
                &grad,
                &cache.expert_outputs,
                batch,
                experts,
                per_image,
                spec.load_balance,
                crate::moe_conv::router_temperature(spec),
                config,
            )
            .map_err(|err| backend_error("MoE router backward", err))?;
        let pooled = cache.pooled.clone();
        let router_weights = self
            .backend
            .upload_f32(&[experts, channels], &params.router_weights)
            .map_err(|err| backend_error("upload MoE router weights", err))?;
        let (pooled_gradient, router_weight_gradient, router_bias_gradient) = self
            .backend
            .connected_backward_f32(
                &pooled,
                &router_weights,
                &dlogit,
                batch,
                channels,
                experts,
                config,
            )
            .map_err(|err| backend_error("MoE router connected backward", err))?;
        let router_input_gradient = self
            .backend
            .moe_router_pool_backward_f32(
                &pooled_gradient,
                &tensor_backend_shape(layer.input_shape),
                config,
            )
            .map_err(|err| backend_error("MoE router pool backward", err))?;

        let activation = gpu_activation_kind(layer.index, &spec.expert.activation)?;
        let mut expert_input_gradient = None;
        let mut expert_weight_gradients = Vec::with_capacity(experts);
        let mut expert_bias_gradients = Vec::with_capacity(experts);
        let mut expert_scale_gradients = Vec::with_capacity(experts);
        for expert_index in 0..experts {
            let expert = &params.experts[expert_index];
            let cached = self.cached_convolution_params_at(
                layer.index,
                expert_index,
                channels,
                &spec.expert,
                expert,
            )?;
            let activation_input =
                cache.expert_activation_inputs[expert_index].clone();
            let sample_gates = self
                .backend
                .moe_gate_column_f32(
                    &cache.gates,
                    batch,
                    experts,
                    expert_index,
                    config,
                )
                .map_err(|err| backend_error("select MoE backward expert gates", err))?;
            let gated_gradient = self
                .backend
                .channel_scale_f32(&grad, &sample_gates, batch, per_image, config)
                .map_err(|err| backend_error("scale MoE expert output gradient", err))?;
            let activation_delta = self
                .backend
                .activation_backward_f32(
                    &activation_input,
                    &gated_gradient,
                    activation,
                    config,
                )
                .map_err(|err| backend_error("MoE expert activation backward", err))?;
            let (delta, bias_gradient, scale_gradient) =
                if let Some(batch_norm) = expert.batch_norm.as_ref() {
                    validate_batchnorm_params(batch_norm, spec.expert.filters)?;
                    let raw_output = self
                        .backend
                        .conv2d_f32(
                            current_input,
                            &cached.raw_weights,
                            &tensor_backend_shape(layer.output_shape),
                            channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.expert.filters,
                            spec.expert.size,
                            spec.expert.stride_x,
                            spec.expert.stride_y,
                            spec.expert.dilation,
                            spec.expert.padding,
                            spec.expert.groups.max(1),
                            config,
                        )
                        .map_err(|err| backend_error("MoE raw expert convolution", err))?;
                    let mean = batch_norm
                        .rolling_mean
                        .iter()
                        .map(|value| -*value)
                        .collect::<Vec<_>>();
                    let inv_std = batch_norm
                        .rolling_variance
                        .iter()
                        .map(|variance| {
                            1.0 / (variance + BATCH_NORM_EPSILON).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let centered = self
                        .backend
                        .upload_f32(&[spec.expert.filters], &mean)
                        .and_then(|mean| {
                            self.backend.bias_add_f32(
                                &raw_output,
                                &mean,
                                spec.expert.filters,
                                output_spatial,
                                config,
                            )
                        })
                        .map_err(|err| {
                            backend_error("center MoE raw expert convolution", err)
                        })?;
                    let inv_std_gpu = self
                        .backend
                        .upload_f32(&[spec.expert.filters], &inv_std)
                        .map_err(|err| {
                            backend_error("upload MoE expert BN inverse std", err)
                        })?;
                    let x_hat = self
                        .backend
                        .channel_scale_f32(
                            &centered,
                            &inv_std_gpu,
                            spec.expert.filters,
                            output_spatial,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("normalize MoE expert BN input", err)
                        })?;
                    let scaled_delta = self
                        .backend
                        .vector_mul_f32(
                            &activation_delta,
                            &x_hat,
                            &tensor_backend_shape(layer.output_shape),
                            config,
                        )
                        .map_err(|err| {
                            backend_error("multiply MoE expert BN scale gradient", err)
                        })?;
                    let scale_gradient = self
                        .backend
                        .bias_backward_f32(
                            &scaled_delta,
                            spec.expert.filters,
                            output_spatial,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("MoE expert BN scale backward", err)
                        })?;
                    let bias_gradient = self
                        .backend
                        .bias_backward_f32(
                            &activation_delta,
                            spec.expert.filters,
                            output_spatial,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("MoE expert BN bias backward", err)
                        })?;
                    let affine = batch_norm
                        .scales
                        .iter()
                        .zip(&inv_std)
                        .map(|(scale, inv_std)| scale * inv_std)
                        .collect::<Vec<_>>();
                    let affine_gpu = self
                        .backend
                        .upload_f32(&[spec.expert.filters], &affine)
                        .map_err(|err| {
                            backend_error("upload MoE expert BN affine gradient", err)
                        })?;
                    let delta = self
                        .backend
                        .channel_scale_f32(
                            &activation_delta,
                            &affine_gpu,
                            spec.expert.filters,
                            output_spatial,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("scale MoE expert BN gradient", err)
                        })?;
                    (delta, bias_gradient, Some(scale_gradient))
                } else {
                    let bias_gradient = self
                        .backend
                        .bias_backward_f32(
                            &activation_delta,
                            spec.expert.filters,
                            output_spatial,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("MoE expert bias backward", err)
                        })?;
                    (activation_delta, bias_gradient, None)
                };
            let weight_gradient = self
                .backend
                .conv2d_backward_weights_f32(
                    current_input,
                    &delta,
                    &[
                        spec.expert.filters,
                        channels / spec.expert.groups.max(1),
                        spec.expert.size,
                        spec.expert.size,
                    ],
                    channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.expert.filters,
                    spec.expert.size,
                    spec.expert.stride_x,
                    spec.expert.stride_y,
                    spec.expert.dilation,
                    spec.expert.padding,
                    spec.expert.groups.max(1),
                    config,
                )
                .map_err(|err| backend_error("MoE expert weight backward", err))?;
            let input_gradient = self
                .backend
                .conv2d_backward_input_f32(
                    &delta,
                    &cached.raw_weights,
                    &tensor_backend_shape(layer.input_shape),
                    channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.expert.filters,
                    spec.expert.size,
                    spec.expert.stride_x,
                    spec.expert.stride_y,
                    spec.expert.dilation,
                    spec.expert.padding,
                    spec.expert.groups.max(1),
                    config,
                )
                .map_err(|err| backend_error("MoE expert input backward", err))?;
            self.accumulate_gpu_gradient(
                &mut expert_input_gradient,
                input_gradient,
            )?;
            expert_weight_gradients.push(weight_gradient);
            expert_bias_gradients.push(bias_gradient);
            expert_scale_gradients.push(scale_gradient);
        }
        self.accumulate_device_moe_train_buffer(
            layer.index,
            &router_weight_gradient,
            &router_bias_gradient,
            &expert_weight_gradients,
            &expert_bias_gradients,
            &expert_scale_gradients,
        )?;
        let expert_input_gradient = expert_input_gradient.ok_or_else(|| {
            Error::InvalidConfig(format!(
                "MoE layer #{} has no expert input gradients",
                layer.index
            ))
        })?;
        let input_gradient = self
            .backend
            .vector_add_f32(
                &expert_input_gradient,
                &router_input_gradient,
                config,
            )
            .map_err(|err| backend_error("combine MoE input gradients", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
