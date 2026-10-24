//! GPU-runtime glue for `[moe_convolutional]`.
use crate::error::{Error, Result};
use crate::model::{Layer, MoeSpec};
use crate::runtime::{
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
}
