//! GPU-runtime glue for `[shortcut]` (plain and weighted).
use crate::error::{Error, Result};
use crate::model::{Layer, ShortcutSpec};
use crate::runtime::{
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
}
