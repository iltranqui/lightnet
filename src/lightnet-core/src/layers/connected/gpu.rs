//! GPU-runtime glue for `[connected]`.
use crate::error::{Error, Result};
use crate::model::{ConnectedSpec, Layer};
use crate::runtime::{
    LayerTrainBuffers,
    validate_batchnorm_params,
    GpuForwardState,
    BATCH_NORM_EPSILON, ConnectedParams, GpuExecutableNetwork, backend_error,
    forward_connected_with_cache, gpu_activation_kind, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native fully-connected forward with inference batch norm folded into the weights;
    /// activations without a device kind round-trip through the host reference. Returns
    /// the output and the pre-activation tensor kept for backward.
    pub(crate) fn forward_connected_gpu(
        &self,
        layer: &Layer,
        spec: &ConnectedSpec,
        connected: &ConnectedParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<B::Tensor>)> {
        let mut activation_input_gpu = None;
        let inputs = layer.input_shape.element_count();
        if connected.weights.len() != spec.output * inputs
            || connected.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "connected layer #{} expects {} weights and {} biases, got {} and {}",
                layer.index,
                spec.output * inputs,
                spec.output,
                connected.weights.len(),
                connected.biases.len()
            )));
        }
        let output = if let Some(activation) = spec.activation.to_cuda_f32_kind() {
            // Inference batch norm is a per-output affine map, so
            // fold it into the uploaded weights and biases.
            let (weights, biases) = match connected.batch_norm.as_ref() {
                Some(bn) => {
                    let mut weights = connected.weights.clone();
                    let mut biases = connected.biases.clone();
                    for (o, row) in weights.chunks_mut(inputs).enumerate() {
                        let k = bn.scales[o]
                            / (bn.rolling_variance[o] + BATCH_NORM_EPSILON).sqrt();
                        row.iter_mut().for_each(|w| *w *= k);
                        biases[o] -= bn.rolling_mean[o] * k;
                    }
                    (weights, biases)
                }
                None => (connected.weights.clone(), connected.biases.clone()),
            };
            let weights_gpu = self
                .backend
                .upload_f32(&[spec.output, inputs], &weights)
                .map_err(|err| backend_error("upload connected weights", err))?;
            let biases_gpu = self
                .backend
                .upload_f32(&[spec.output], &biases)
                .map_err(|err| backend_error("upload connected biases", err))?;
            let pre_activation = self
                .backend
                .connected_f32(
                    current_input,
                    &weights_gpu,
                    &biases_gpu,
                    layer.input_shape.batch,
                    inputs,
                    spec.output,
                    launch_config,
                )
                .map_err(|err| backend_error("connected", err))?;
            activation_input_gpu = Some(pre_activation.clone());
            self.backend
                .activation_f32(&pre_activation, activation, launch_config)
                .map_err(|err| backend_error("connected activation", err))?
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download connected input", err))?;
            let (host_output, host_activation_input) = forward_connected_with_cache(
                spec,
                connected,
                &host_input,
                layer.output_shape,
            );
            if let Some(activation_input) = host_activation_input {
                activation_input_gpu = Some(
                    self.backend
                        .upload_f32(
                            &tensor_backend_shape(layer.output_shape),
                            &activation_input.data,
                        )
                        .map_err(|err| {
                            backend_error("upload connected activation cache", err)
                        })?,
                );
            }
            self.backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload connected output", err))?
        };
        Ok((output, activation_input_gpu))
    }

    /// Compact inference forward: weights are folded with batch norm, activation has a device kind.
    pub(crate) fn forward_connected_compact_gpu(
        &self,
        layer: &Layer,
        spec: &ConnectedSpec,
        connected: &ConnectedParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let inputs = layer.input_shape.element_count();
        if connected.weights.len() != spec.output * inputs
            || connected.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "connected layer #{} expects {} weights and {} biases, got {} and {}",
                layer.index,
                spec.output * inputs,
                spec.output,
                connected.weights.len(),
                connected.biases.len()
            )));
        }
        let (weights, biases) = match connected.batch_norm.as_ref() {
            Some(bn) => {
                let mut weights = connected.weights.clone();
                let mut biases = connected.biases.clone();
                for (o, row) in weights.chunks_mut(inputs).enumerate() {
                    let k = bn.scales[o]
                        / (bn.rolling_variance[o] + BATCH_NORM_EPSILON).sqrt();
                    row.iter_mut().for_each(|w| *w *= k);
                    biases[o] -= bn.rolling_mean[o] * k;
                }
                (weights, biases)
            }
            None => (connected.weights.clone(), connected.biases.clone()),
        };
        let weights_gpu = self
            .backend
            .upload_f32(&[spec.output, inputs], &weights)
            .map_err(|err| backend_error("compact connected weights", err))?;
        let biases_gpu = self
            .backend
            .upload_f32(&[spec.output], &biases)
            .map_err(|err| backend_error("compact connected biases", err))?;
        let pre_activation = self
            .backend
            .connected_f32(
                current_input,
                &weights_gpu,
                &biases_gpu,
                layer.input_shape.batch,
                inputs,
                spec.output,
                launch_config,
            )
            .map_err(|err| backend_error("compact connected", err))?;
        Ok(self.backend
            .activation_f32(
                &pre_activation,
                gpu_activation_kind(layer.index, &spec.activation)?,
                launch_config,
            )
            .map_err(|err| backend_error("compact connected activation", err))?)
    }

    /// Native fully-connected backward.
    pub(crate) fn backward_connected_gpu(
        &self,
        layer: &Layer,
        spec: &ConnectedSpec,
        connected: &ConnectedParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        // The gradient chain stays on the device; only the connected
        // GEMM itself runs on the host (`connected_backward_f32`).
        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU activation cache for connected layer #{}",
                    layer.index
                ))
            })?;
        let activation_delta = self
            .backend
            .activation_backward_f32(activation_input, &grad, activation, config)
            .map_err(|err| backend_error("GPU connected activation backward", err))?;
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU layer output for connected layer #{}",
                    layer.index
                ))
            })?
        };
        let inputs = layer.input_shape.element_count();
        let weights_gpu = self
            .backend
            .upload_f32(&[spec.output, inputs], &connected.weights)
            .map_err(|err| backend_error("upload connected weights", err))?;
        let output_shape = tensor_backend_shape(layer.output_shape);
        let (delta, bias_grad, batch_norm_scale_grad) =
            if let Some(batch_norm) = connected.batch_norm.as_ref() {
                validate_batchnorm_params(batch_norm, spec.output)?;
                let batch = layer.input_shape.batch;
                let reciprocal_std = batch_norm
                    .rolling_variance
                    .iter()
                    .map(|variance| {
                        1.0 / (variance + BATCH_NORM_EPSILON).sqrt()
                    })
                    .collect::<Vec<_>>();
                let per_batch_reciprocal_std = (0..batch)
                    .flat_map(|_| reciprocal_std.iter().copied())
                    .collect::<Vec<_>>();
                let centered_scale = batch_norm
                    .rolling_mean
                    .iter()
                    .zip(&reciprocal_std)
                    .map(|(mean, reciprocal)| -mean * reciprocal)
                    .collect::<Vec<_>>();
                let zero_bias = self
                    .backend
                    .zeros_f32(&[spec.output])
                    .map_err(|err| backend_error("allocate connected raw bias", err))?;
                let raw = self
                    .backend
                    .connected_f32(
                        current_input,
                        &weights_gpu,
                        &zero_bias,
                        batch,
                        inputs,
                        spec.output,
                        config,
                    )
                    .map_err(|err| backend_error("connected raw output for BN backward", err))?;
                let reciprocal_std_gpu = self
                    .backend
                    .upload_f32(&[batch, spec.output], &per_batch_reciprocal_std)
                    .map_err(|err| backend_error("upload connected BN reciprocal std", err))?;
                let scaled_raw = self
                    .backend
                    .scale_channels_f32(
                        &reciprocal_std_gpu,
                        &raw,
                        &output_shape,
                        false,
                        config,
                    )
                    .map_err(|err| backend_error("connected BN scale raw", err))?;
                let offset = self
                    .backend
                    .upload_f32(&[spec.output], &centered_scale)
                    .map_err(|err| backend_error("upload connected BN center", err))?;
                let normalized = self
                    .backend
                    .bias_add_f32(
                        &scaled_raw,
                        &offset,
                        spec.output,
                        1,
                        config,
                    )
                    .map_err(|err| backend_error("connected BN normalization", err))?;
                let scale_products = self
                    .backend
                    .vector_mul_f32(
                        &activation_delta,
                        &normalized,
                        &output_shape,
                        config,
                    )
                    .map_err(|err| backend_error("connected BN scale products", err))?;
                let scale_gradient = self
                    .backend
                    .bias_backward_f32(&scale_products, spec.output, 1, config)
                    .map_err(|err| backend_error("connected BN scale gradient", err))?;
                let delta = self
                    .backend
                    .scale_channels_f32(
                        &self
                            .backend
                            .upload_f32(
                                &[batch, spec.output],
                                &(0..batch)
                                    .flat_map(|_| {
                                        batch_norm
                                            .scales
                                            .iter()
                                            .zip(&reciprocal_std)
                                            .map(|(scale, reciprocal)| scale * reciprocal)
                                    })
                                    .collect::<Vec<_>>(),
                            )
                            .map_err(|err| backend_error("upload connected BN backward scales", err))?,
                        &activation_delta,
                        &output_shape,
                        false,
                        config,
                    )
                    .map_err(|err| backend_error("connected BN raw delta", err))?;
                let bias_gradient = self
                    .backend
                    .bias_backward_f32(&activation_delta, spec.output, 1, config)
                    .map_err(|err| backend_error("connected BN bias gradient", err))?;
                (delta, bias_gradient, Some(scale_gradient))
            } else {
                (activation_delta, self.backend.zeros_f32(&[spec.output]).map_err(|err| backend_error("allocate connected bias gradient", err))?, None)
            };
        let (input_grad, weight_grad, computed_bias_grad) = self
            .backend
            .connected_backward_f32(
                current_input,
                &weights_gpu,
                &delta,
                layer.input_shape.batch,
                inputs,
                spec.output,
                config,
            )
            .map_err(|err| backend_error("GPU connected backward", err))?;
        let bias_grad = if connected.batch_norm.is_some() {
            bias_grad
        } else {
            computed_bias_grad
        };
        self.accumulate_connected_host_gradients(
            layer.index,
            &weight_grad,
            &bias_grad,
            inputs,
            spec.output,
        )?;
        if let Some(scale_gradient) = batch_norm_scale_grad {
            let values = self
                .backend
                .download_f32(&scale_gradient)
                .map_err(|err| backend_error("download connected BN scale gradient", err))?;
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::Connected(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected train buffers for BN layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = train.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected BN gradient buffer for layer #{}",
                    layer.index
                )));
            };
            for (slot, value) in batch_norm.scale_updates.iter_mut().zip(values) {
                *slot += value;
            }
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
