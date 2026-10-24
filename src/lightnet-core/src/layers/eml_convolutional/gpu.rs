//! GPU-runtime glue for `[eml_convolutional]`.
use crate::error::{Error, Result};
use crate::model::{ConvolutionalSpec, EmlConvolutionalSpec, Layer, TensorShape};
use crate::runtime::{
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
}
