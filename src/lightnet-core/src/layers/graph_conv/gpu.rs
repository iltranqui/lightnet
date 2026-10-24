//! GPU-runtime glue for `[graph_conv]`.
use crate::error::{Error, Result};
use crate::model::{GraphConvSpec, Layer};
use crate::runtime::{
    GpuBatchNormCache, GpuExecutableNetwork, GraphConvParams, GraphConvRuntimeCache, Tensor,
    backend_error, forward_graph_conv_with_cache, graph_conv_aggregation_geometry,
    gpu_activation_kind, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

/// Output of one `[graph_conv]` forward: the activated tensor, the pre-activation tensor
/// backward needs, and whichever cache the chosen path produced.
pub(crate) struct GraphConvGpuForward<T: Clone> {
    pub(crate) output: T,
    pub(crate) activation_input: T,
    pub(crate) host_cache: Option<GraphConvRuntimeCache>,
    pub(crate) batch_norm_cache: Option<GpuBatchNormCache<T>>,
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Host-reference path for variants the kernels cannot cover or when the CPU backward
    /// fallback is needed: runs the CPU forward on the previous layer's host output.
    pub(crate) fn forward_graph_conv_host_gpu(
        &self,
        layer: &Layer,
        spec: &GraphConvSpec,
        params: &GraphConvParams,
        input: &Tensor,
        layer_outputs: &[Tensor],
        training: bool,
    ) -> Result<GraphConvGpuForward<B::Tensor>> {
        let current_host = if layer.index == 0 {
            input
        } else {
            layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing host input for GraphConv fallback layer #{}",
                    layer.index
                ))
            })?
        };
        let (host_output, host_activation, cache) = forward_graph_conv_with_cache(
            spec,
            params,
            current_host,
            layer.output_shape,
            training,
        )?;
        let host_cache = Some(cache);
        let activation_input = self
            .backend
            .upload_f32(
                &tensor_backend_shape(layer.output_shape),
                &host_activation.data,
            )
            .map_err(|err| backend_error("upload GraphConv fallback activation", err))?;
        let output = self
            .backend
            .upload_f32(
                &tensor_backend_shape(layer.output_shape),
                &host_output.data,
            )
            .map_err(|err| backend_error("upload GraphConv fallback output", err))?;
        Ok(GraphConvGpuForward {
            output,
            activation_input,
            host_cache,
            batch_norm_cache: None,
        })
    }

    /// Native path: uniform or learned aggregation, pointwise projection with optional
    /// self term, batch norm or bias, then activation or channel normalization.
    pub(crate) fn forward_graph_conv_device_gpu(
        &self,
        layer: &Layer,
        spec: &GraphConvSpec,
        params: &GraphConvParams,
        current_input: &B::Tensor,
        training: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<GraphConvGpuForward<B::Tensor>> {
        let mut batch_norm_cache = None;
        let geometry = graph_conv_aggregation_geometry(
            spec,
            layer.input_shape,
            layer.output_shape,
        );
        let groups = spec.groups.max(1);
        let aggregate = if spec.edge_mode == 0 {
            self.backend
                .graph_conv_aggregate_f32(current_input, geometry, launch_config)
                .map_err(|err| {
                    backend_error("GraphConv uniform aggregation", err)
                })?
        } else {
            let edge_kernel_values =
                params.edge_kernel.as_deref().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv learned edge kernel is missing".to_owned(),
                    )
                })?;
            let edge_bias_values =
                params.edge_biases.as_deref().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv learned edge biases are missing".to_owned(),
                    )
                })?;
            let edge_kernel = self
                .backend
                .upload_f32(&[edge_kernel_values.len()], edge_kernel_values)
                .map_err(|err| {
                    backend_error("upload GraphConv edge kernel", err)
                })?;
            let edge_biases = self
                .backend
                .upload_f32(&[edge_bias_values.len()], edge_bias_values)
                .map_err(|err| {
                    backend_error("upload GraphConv edge biases", err)
                })?;
            self.backend
                .graph_conv_learned_aggregate_f32(
                    current_input,
                    &edge_kernel,
                    &edge_biases,
                    geometry,
                    groups,
                    launch_config,
                )
                .map_err(|err| {
                    backend_error("GraphConv learned aggregation", err)
                })?
        };
        let weight_shape =
            [spec.filters, layer.input_shape.channels / groups, 1, 1];
        let weights = self
            .backend
            .upload_f32(&weight_shape, &params.projection.weights)
            .map_err(|err| {
                backend_error("upload GraphConv projection weights", err)
            })?;
        let raw = self
            .backend
            .conv2d_f32(
                &aggregate,
                &weights,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.output_shape.height,
                layer.output_shape.width,
                spec.filters,
                1,
                1,
                1,
                1,
                0,
                groups,
                launch_config,
            )
            .map_err(|err| backend_error("GraphConv pointwise projection", err))?;
        let raw = if spec.use_self {
            let reference = self
                .backend
                .graph_conv_reference_f32(current_input, geometry, launch_config)
                .map_err(|err| {
                    backend_error("GraphConv center-reference gather", err)
                })?;
            let self_values = params.self_weights.as_deref().ok_or_else(|| {
                Error::InvalidConfig(
                    "GraphConv self weights are missing".to_owned(),
                )
            })?;
            let self_weights = self
                .backend
                .upload_f32(&weight_shape, self_values)
                .map_err(|err| {
                    backend_error("upload GraphConv self weights", err)
                })?;
            let self_raw = self
                .backend
                .conv2d_f32(
                    &reference,
                    &self_weights,
                    &tensor_backend_shape(layer.output_shape),
                    layer.input_shape.channels,
                    layer.output_shape.height,
                    layer.output_shape.width,
                    spec.filters,
                    1,
                    1,
                    1,
                    1,
                    0,
                    groups,
                    launch_config,
                )
                .map_err(|err| backend_error("GraphConv self projection", err))?;
            self.backend
                .shortcut_add_f32(&raw, &self_raw, launch_config)
                .map_err(|err| backend_error("GraphConv projection sum", err))?
        } else {
            raw
        };
        let biased = if spec.batch_normalize && training {
            let batch_norm =
                params.projection.batch_norm.as_ref().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv batch-norm parameters are missing".to_owned(),
                    )
                })?;
            let (normalized, cache) = self
                .forward_batchnorm_training_gpu_from_host_params(
                    &raw,
                    &params.projection.biases,
                    batch_norm,
                    layer.output_shape,
                    launch_config,
                )?;
            batch_norm_cache = Some(cache);
            normalized
        } else if spec.batch_normalize {
            let mut batch_norm =
                params.projection.batch_norm.clone().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv batch-norm parameters are missing".to_owned(),
                    )
                })?;
            batch_norm.biases.clone_from(&params.projection.biases);
            self.forward_batchnorm_inference_gpu(
                &raw,
                layer.output_shape,
                &batch_norm,
                launch_config,
            )?
        } else {
            let biases = self
                .backend
                .upload_f32(&[spec.filters], &params.projection.biases)
                .map_err(|err| {
                    backend_error("upload GraphConv projection biases", err)
                })?;
            self.backend
                .bias_add_f32(
                    &raw,
                    &biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("GraphConv projection bias", err))?
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
                    backend_error("GraphConv channel normalization", err)
                })?
        } else {
            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
            self.backend
                .activation_f32(&biased, activation, launch_config)
                .map_err(|err| backend_error("GraphConv activation", err))?
        };
        Ok(GraphConvGpuForward {
            output,
            activation_input,
            host_cache: None,
            batch_norm_cache,
        })
    }

    /// Compact inference forward: native aggregation, projection, batch norm fold and activation.
    pub(crate) fn forward_graph_conv_compact_gpu(
        &self,
        layer: &Layer,
        spec: &GraphConvSpec,
        params: &GraphConvParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let input = current_input;
        let groups = spec.groups.max(1);
        let geometry = graph_conv_aggregation_geometry(
            spec,
            layer.input_shape,
            layer.output_shape,
        );
        let aggregate = if spec.edge_mode == 0 {
            self.backend
                .graph_conv_aggregate_f32(input, geometry, launch_config)
                .map_err(|err| backend_error("compact GraphConv aggregation", err))?
        } else {
            let edge_kernel_values =
                params.edge_kernel.as_deref().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv learned edge kernel is missing".to_owned(),
                    )
                })?;
            let edge_bias_values = params.edge_biases.as_deref().ok_or_else(|| {
                Error::InvalidConfig(
                    "GraphConv learned edge biases are missing".to_owned(),
                )
            })?;
            let edge_kernel = self
                .backend
                .upload_f32(&[edge_kernel_values.len()], edge_kernel_values)
                .map_err(|err| backend_error("compact GraphConv edge kernel", err))?;
            let edge_biases = self
                .backend
                .upload_f32(&[edge_bias_values.len()], edge_bias_values)
                .map_err(|err| backend_error("compact GraphConv edge biases", err))?;
            self.backend
                .graph_conv_learned_aggregate_f32(
                    input,
                    &edge_kernel,
                    &edge_biases,
                    geometry,
                    groups,
                    launch_config,
                )
                .map_err(|err| {
                    backend_error("compact GraphConv learned aggregation", err)
                })?
        };
        let weight_shape = [spec.filters, layer.input_shape.channels / groups, 1, 1];
        let weights = self
            .backend
            .upload_f32(&weight_shape, &params.projection.weights)
            .map_err(|err| {
                backend_error("compact GraphConv projection weights", err)
            })?;
        let mut raw = self
            .backend
            .conv2d_f32(
                &aggregate,
                &weights,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.output_shape.height,
                layer.output_shape.width,
                spec.filters,
                1,
                1,
                1,
                1,
                0,
                groups,
                launch_config,
            )
            .map_err(|err| backend_error("compact GraphConv projection", err))?;
        if spec.use_self {
            let reference = self
                .backend
                .graph_conv_reference_f32(input, geometry, launch_config)
                .map_err(|err| {
                    backend_error("compact GraphConv reference gather", err)
                })?;
            let self_values = params.self_weights.as_deref().ok_or_else(|| {
                Error::InvalidConfig("GraphConv self weights are missing".to_owned())
            })?;
            let self_weights = self
                .backend
                .upload_f32(&weight_shape, self_values)
                .map_err(|err| backend_error("compact GraphConv self weights", err))?;
            let self_raw = self
                .backend
                .conv2d_f32(
                    &reference,
                    &self_weights,
                    &tensor_backend_shape(layer.output_shape),
                    layer.input_shape.channels,
                    layer.output_shape.height,
                    layer.output_shape.width,
                    spec.filters,
                    1,
                    1,
                    1,
                    1,
                    0,
                    groups,
                    launch_config,
                )
                .map_err(|err| {
                    backend_error("compact GraphConv self projection", err)
                })?;
            raw = self
                .backend
                .shortcut_add_f32(&raw, &self_raw, launch_config)
                .map_err(|err| {
                    backend_error("compact GraphConv projection sum", err)
                })?;
        }
        let biased = if spec.batch_normalize {
            let mut batch_norm =
                params.projection.batch_norm.clone().ok_or_else(|| {
                    Error::InvalidConfig(
                        "GraphConv batch-norm parameters are missing".to_owned(),
                    )
                })?;
            batch_norm.biases.clone_from(&params.projection.biases);
            self.forward_batchnorm_inference_gpu(
                &raw,
                layer.output_shape,
                &batch_norm,
                launch_config,
            )?
        } else {
            let biases = self
                .backend
                .upload_f32(&[spec.filters], &params.projection.biases)
                .map_err(|err| backend_error("compact GraphConv biases", err))?;
            self.backend
                .bias_add_f32(
                    &raw,
                    &biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("compact GraphConv bias", err))?
        };
        Ok(if let Some(normalization) = spec.activation.to_channel_normalization() {
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
                    backend_error("compact GraphConv channel normalization", err)
                })?
        } else {
            self.backend
                .activation_f32(
                    &biased,
                    gpu_activation_kind(layer.index, &spec.activation)?,
                    launch_config,
                )
                .map_err(|err| backend_error("compact GraphConv activation", err))?
        })
    }
}
