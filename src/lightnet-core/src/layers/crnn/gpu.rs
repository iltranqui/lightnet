//! GPU-runtime glue for `[crnn]`.
use crate::error::Result;
use crate::model::{CrnnSpec, Layer};
use crate::runtime::{
    CrnnParams, GpuCrnnBackwardCache, GpuExecutableNetwork, RecurrentForwardCache, backend_error,
    forward_crnn_layer, forward_crnn_layer_device, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native sequence forward when the activation has a device kind; otherwise the host
    /// reference. Returns the output, the host cache for CPU backward, and the device cache.
    pub(crate) fn forward_crnn_gpu(
        &self,
        layer: &Layer,
        spec: &CrnnSpec,
        recurrent: &CrnnParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        Option<RecurrentForwardCache>,
        Option<GpuCrnnBackwardCache<B::Tensor>>,
    )> {
        if spec.activation.to_cuda_f32_kind().is_some() {
            let (output, cache, device_cache) = forward_crnn_layer_device(
                &self.backend,
                spec,
                recurrent,
                current_input,
                layer.input_shape,
                layer.output_shape,
                layer.index,
                host_copies,
                launch_config,
            )?;
            Ok((
                output,
                cache.map(RecurrentForwardCache::Crnn),
                Some(device_cache),
            ))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download CRNN input", err))?;
            let (host_output, cache) =
                forward_crnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
            let output = self
                .backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload CRNN output", err))?;
            Ok((output, Some(RecurrentForwardCache::Crnn(cache)), None))
        }
    }
}
