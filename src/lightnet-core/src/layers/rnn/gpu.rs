//! GPU-runtime glue for `[rnn]`.
use crate::error::Result;
use crate::model::{Activation, Layer, RnnSpec};
use crate::runtime::{
    GpuExecutableNetwork, GpuRnnBackwardCache, RecurrentForwardCache, RnnParams, backend_error,
    forward_rnn_layer, forward_rnn_layer_device, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native sequence forward when the activations have device kinds; otherwise the host
    /// reference. Returns the output, the host cache for CPU backward, and the device cache.
    pub(crate) fn forward_rnn_gpu(
        &self,
        layer: &Layer,
        spec: &RnnSpec,
        recurrent: &RnnParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        Option<RecurrentForwardCache>,
        Option<GpuRnnBackwardCache<B::Tensor>>,
    )> {
        if spec.activation.to_cuda_f32_kind().is_some()
            && (spec.logistic != 2 || Activation::Loggy.to_cuda_f32_kind().is_some())
        {
            let (output, cache, device_cache) = forward_rnn_layer_device(
                &self.backend,
                current_input,
                layer.input_shape,
                spec,
                recurrent,
                layer.output_shape,
                layer.index,
                host_copies,
                launch_config,
            )?;
            Ok((
                output,
                cache.map(RecurrentForwardCache::Rnn),
                Some(device_cache),
            ))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download RNN input", err))?;
            let (host_output, cache) =
                forward_rnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
            let output = self
                .backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload RNN output", err))?;
            Ok((output, Some(RecurrentForwardCache::Rnn(cache)), None))
        }
    }
}
