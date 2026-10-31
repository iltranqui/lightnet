//! GPU-runtime glue for `[shortcut]` (plain and weighted).
use crate::error::{Error, Result};
use crate::model::{Layer, ShortcutSpec};
use crate::runtime::{
    LayerTrainBuffers,
    GradMode,
    GpuForwardState,
    GpuExecutableNetwork, ShortcutParams, backend_error, gpu_activation_kind,
    resolve_relative_layer, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Weighted shortcut; returns the output and its pre-activation sum.
    pub(crate) fn forward_weighted_shortcut_gpu(
        &self,
        layer: &Layer,
        spec: &ShortcutSpec,
        params: &ShortcutParams,
        current_input: &B::Tensor,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let mut inputs = Vec::with_capacity(spec.from.len() + 1);
        inputs.push(current_input.clone());
        for raw_index in &spec.from {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source = gpu_layer_outputs
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                })?;
            let source_layer = self.network.layers.get(source_index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing source layer #{source_index} for shortcut layer #{}",
                    layer.index
                ))
            })?;
            if source_layer.output_shape.batch != layer.output_shape.batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                    layer.index, layer.output_shape, source_layer.output_shape
                )));
            }
            if source_layer.output_shape == layer.output_shape {
                inputs.push(source.clone());
            } else {
                let fitted = self
                    .backend
                    .shortcut_fit_f32(
                        source,
                        layer.output_shape.batch,
                        &tensor_backend_shape(layer.output_shape),
                        launch_config,
                    )
                    .map_err(|err| backend_error("weighted shortcut resize source", err))?;
                inputs.push(fitted);
            }
        }
        let layer_step = spec.weight_slots(layer.output_shape.channels);
        let per_sample = layer.output_shape.element_count();
        let effective = crate::shortcut_weights::effective_weights(
            &params.weights,
            spec.weights_normalization,
            layer_step,
        );
        let refs: Vec<&B::Tensor> = inputs.iter().collect();
        let accumulated = self
            .backend
            .shortcut_weighted_sum_f32(
                &refs,
                &effective,
                layer_step,
                per_sample / layer_step,
                launch_config,
            )
            .map_err(|err| backend_error("weighted shortcut sum", err))?;
        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let output = self
            .backend
            .activation_f32(&accumulated, activation, launch_config)
            .map_err(|err| backend_error("weighted shortcut activation", err))?;
        Ok((output, accumulated))
    }

    /// Plain shortcut; returns the output and its pre-activation sum.
    pub(crate) fn forward_shortcut_gpu(
        &self,
        layer: &Layer,
        spec: &ShortcutSpec,
        current_input: &B::Tensor,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let mut accumulated = current_input.clone();
        for raw_index in &spec.from {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source = gpu_layer_outputs
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                })?;
            let source_layer = self.network.layers.get(source_index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing source layer #{source_index} for shortcut layer #{}",
                    layer.index
                ))
            })?;
            if source_layer.output_shape.batch != layer.output_shape.batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                    layer.index, layer.output_shape, source_layer.output_shape
                )));
            }
            // Darknet resizes a mismatched source flat per sample
            // (crop / zero-pad), then adds it.
            let fitted;
            let source = if source_layer.output_shape != layer.output_shape {
                fitted = self
                    .backend
                    .shortcut_fit_f32(
                        source,
                        layer.output_shape.batch,
                        &tensor_backend_shape(layer.output_shape),
                        launch_config,
                    )
                    .map_err(|err| backend_error("shortcut resize source", err))?;
                &fitted
            } else {
                source
            };
            accumulated = self
                .backend
                .shortcut_add_f32(&accumulated, source, launch_config)
                .map_err(|err| backend_error("shortcut add", err))?;
        }

        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let output = self
            .backend
            .activation_f32(&accumulated, activation, launch_config)
            .map_err(|err| backend_error("shortcut activation", err))?;
        Ok((output, accumulated))
    }

    /// Compact inference forward of `[shortcut]` without per-feature weights.
    pub(crate) fn forward_shortcut_compact_gpu(
        &self,
        layer: &Layer,
        spec: &ShortcutSpec,
        gpu_layer_outputs: &[Option<B::Tensor>],
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let mut accumulated = current_input.clone();
        for raw_index in &spec.from {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source = gpu_layer_outputs
                .get(source_index)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                })?;
            let source_layer =
                self.network.layers.get(source_index).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing source layer #{source_index} for shortcut layer #{}",
                        layer.index
                    ))
                })?;
            if source_layer.output_shape.batch != layer.output_shape.batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                    layer.index, layer.output_shape, source_layer.output_shape
                )));
            }
            // Darknet resizes a mismatched source flat per sample
            // (crop / zero-pad), then adds it.
            let fitted;
            let source = if source_layer.output_shape != layer.output_shape {
                fitted = self
                    .backend
                    .shortcut_fit_f32(
                        source,
                        layer.output_shape.batch,
                        &tensor_backend_shape(layer.output_shape),
                        launch_config,
                    )
                    .map_err(|err| backend_error("shortcut resize source", err))?;
                &fitted
            } else {
                source
            };
            accumulated = self
                .backend
                .shortcut_add_f32(&accumulated, source, launch_config)
                .map_err(|err| backend_error("shortcut add", err))?;
        }

        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let output = self
            .backend
            .activation_f32(&accumulated, activation, launch_config)
            .map_err(|err| backend_error("shortcut activation", err))?;
        Ok(output)
    }

    /// Native weighted-shortcut backward.
    pub(crate) fn backward_weighted_shortcut_gpu(
        &self,
        layer: &Layer,
        spec: &ShortcutSpec,
        params: &ShortcutParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU activation cache for weighted shortcut layer #{}",
                    layer.index
                ))
            })?;
        let delta = self
            .backend
            .activation_backward_f32(activation_input, &grad, activation, config)
            .map_err(|err| {
                backend_error("GPU weighted shortcut activation backward", err)
            })?;
        if layer.input_shape != layer.output_shape {
            return Err(Error::InvalidConfig(format!(
                "weighted shortcut layer #{} input/output shapes differ: {} vs {}",
                layer.index, layer.input_shape, layer.output_shape
            )));
        }
        let per_sample = layer.output_shape.element_count();
        let layer_step = spec.weight_slots(layer.output_shape.channels);
        if layer_step == 0
            || per_sample % layer_step != 0
            || params.weights.len() != spec.weight_count(layer.output_shape.channels)
        {
            return Err(Error::InvalidConfig(format!(
                "weighted shortcut layer #{} has inconsistent weight geometry",
                layer.index
            )));
        }
        let slots_per_feature = per_sample / layer_step;
        let effective = crate::shortcut_weights::effective_weights(
            &params.weights,
            spec.weights_normalization,
            layer_step,
        );
        let scale_pattern = |source_slot: usize| {
            (0..layer.output_shape.batch * per_sample)
                .map(|index| {
                    let feature = index % per_sample;
                    effective[feature / slots_per_feature + source_slot * layer_step]
                })
                .collect::<Vec<_>>()
        };
        let backend_shape = tensor_backend_shape(layer.output_shape);
        let input_scales = self
            .backend
            .upload_f32(&backend_shape, &scale_pattern(0))
            .map_err(|err| {
                backend_error("upload weighted shortcut input scales", err)
            })?;
        let input_gradient = self
            .backend
            .vector_mul_f32(&delta, &input_scales, &backend_shape, config)
            .map_err(|err| backend_error("weighted shortcut input gradient", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;

        let num_inputs = spec.from.len() + 1;
        let mut raw_weight_products = self
            .backend
            .zeros_f32(&[num_inputs, layer_step])
            .map_err(|err| backend_error("allocate weighted shortcut products", err))?;
        let input_gpu = if layer.index == 0 {
            &forward.input_gpu
        } else {
            &forward.gpu_layer_outputs[layer.index - 1]
        };
        raw_weight_products = self
            .backend
            .shortcut_weight_gradient_accumulate_f32(
                &raw_weight_products,
                &delta,
                input_gpu,
                layer.output_shape.batch,
                per_sample,
                layer.input_shape.element_count(),
                num_inputs,
                layer_step,
                0,
                config,
            )
            .map_err(|err| {
                backend_error("weighted shortcut input weight reduction", err)
            })?;
        let mut source_shapes = Vec::with_capacity(spec.from.len());
        for (source_slot, raw_index) in spec.from.iter().enumerate() {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source_layer = self.network.layers.get(source_index).ok_or(
                Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                },
            )?;
            let source_shape = source_layer.output_shape;
            if source_shape.batch != layer.output_shape.batch {
                return Err(Error::InvalidConfig(format!(
                    "weighted shortcut layer #{} mixes incompatible batch shapes",
                    layer.index
                )));
            }
            let source_gpu = forward.gpu_layer_outputs.get(source_index).ok_or(
                Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                },
            )?;
            raw_weight_products = self
                .backend
                .shortcut_weight_gradient_accumulate_f32(
                    &raw_weight_products,
                    &delta,
                    source_gpu,
                    layer.output_shape.batch,
                    per_sample,
                    source_shape.element_count(),
                    num_inputs,
                    layer_step,
                    source_slot + 1,
                    config,
                )
                .map_err(|err| {
                    backend_error("weighted shortcut source weight reduction", err)
                })?;
            source_shapes.push(source_shape);
        }
        let raw_weights = self
            .backend
            .upload_f32(&[num_inputs, layer_step], &params.weights)
            .map_err(|err| backend_error("upload weighted shortcut raw weights", err))?;
        let normalization = match spec.weights_normalization {
            crate::model::ShortcutWeightsNorm::None => {
                lightnet_cuda::F32ShortcutWeightNormalization::None
            }
            crate::model::ShortcutWeightsNorm::Relu => {
                lightnet_cuda::F32ShortcutWeightNormalization::Relu
            }
            crate::model::ShortcutWeightsNorm::Softmax => {
                lightnet_cuda::F32ShortcutWeightNormalization::Softmax
            }
        };
        let weight_gradient = self
            .backend
            .shortcut_weight_normalized_gradient_f32(
                &raw_weights,
                &raw_weight_products,
                num_inputs,
                layer_step,
                normalization,
                GradMode::from_env() == GradMode::Cpp,
                config,
            )
            .map_err(|err| {
                backend_error("weighted shortcut normalization derivative", err)
            })?;
        let weight_gradient = self
            .backend
            .download_f32(&weight_gradient)
            .map_err(|err| {
                backend_error("download weighted shortcut weight gradient", err)
            })?;
        let mut train_buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Shortcut(train)) =
            train_buffers.get_mut(layer.index)
        else {
            return Err(Error::InvalidConfig(format!(
                "missing weighted shortcut train buffers for layer #{}",
                layer.index
            )));
        };
        if train.weight_updates.len() != weight_gradient.len() {
            return Err(Error::InvalidConfig(format!(
                "weighted shortcut train buffer mismatch for layer #{}",
                layer.index
            )));
        }
        for (update, gradient) in
            train.weight_updates.iter_mut().zip(weight_gradient)
        {
            *update += gradient;
        }
        drop(train_buffers);

        for (source_slot, (raw_index, source_shape)) in
            spec.from.iter().zip(source_shapes).enumerate()
        {
            let source_scales = self
                .backend
                .upload_f32(&backend_shape, &scale_pattern(source_slot + 1))
                .map_err(|err| {
                    backend_error("upload weighted shortcut source scales", err)
                })?;
            let output_gradient = self
                .backend
                .vector_mul_f32(&delta, &source_scales, &backend_shape, config)
                .map_err(|err| {
                    backend_error("weighted shortcut source gradient", err)
                })?;
            let source_gradient = if source_shape != layer.output_shape {
                self.backend
                    .shortcut_fit_f32(
                        &output_gradient,
                        source_shape.batch,
                        &tensor_backend_shape(source_shape),
                        config,
                    )
                    .map_err(|err| {
                        backend_error("weighted shortcut source gradient fit", err)
                    })?
            } else {
                output_gradient
            };
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            self.accumulate_gpu_gradient(
                &mut layer_output_gradients[source_index],
                source_gradient,
            )?;
        }
        Ok(())
    }

    /// Native shortcut backward.
    pub(crate) fn backward_shortcut_gpu(
        &self,
        layer: &Layer,
        spec: &ShortcutSpec,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU activation cache for shortcut layer #{}",
                    layer.index
                ))
            })?;
        let accum_grad = self
            .backend
            .activation_backward_f32(activation_input, &grad, activation, config)
            .map_err(|err| backend_error("GPU shortcut activation backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            accum_grad.clone(),
            layer_output_gradients,
        )?;
        for raw_index in &spec.from {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source_shape = self
                .network
                .layers
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                })?
                .output_shape;
            // Mismatched source: crop / zero-pad the gradient flat
            // per sample, like the forward add.
            let source_grad = if source_shape != layer.output_shape {
                self.backend
                    .shortcut_fit_f32(
                        &accum_grad,
                        source_shape.batch,
                        &tensor_backend_shape(source_shape),
                        config,
                    )
                    .map_err(|err| backend_error("GPU shortcut source gradient", err))?
            } else {
                accum_grad.clone()
            };
            self.accumulate_gpu_gradient(
                &mut layer_output_gradients[source_index],
                source_grad,
            )?;
        }
        Ok(())
    }
}
