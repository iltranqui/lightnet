//! GPU-runtime glue for `[mambavision]`: host forward behind the device tensor boundary.
use crate::error::{Error, Result};
use crate::model::{Layer, MambaVisionSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::F32TensorOpsBackend;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Download the input, run the host MambaVision forward, upload the output.
    pub(crate) fn forward_mambavision_host(
        &self,
        layer: &Layer,
        spec: &MambaVisionSpec,
        weights: &super::MambaVisionParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::MambaVisionForward)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download MambaVision input", err))?;
        let forward = super::mambavision_forward(
            &host_input.data,
            host_input.shape,
            spec,
            weights,
            None,
        )
        .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &forward.output)
            .map_err(|err| backend_error("upload MambaVision output", err))?;
        Ok((output, forward))
    }
}
