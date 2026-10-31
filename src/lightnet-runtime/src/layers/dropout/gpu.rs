//! GPU-runtime glue for `[dropout]`.
use crate::error::Result;
use crate::model::Layer;
use crate::runtime::{
    GpuExecutableNetwork, GpuForwardState, backend_error, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn backward_dropout_gpu(
        &self,
        layer: &Layer,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = match forward.cpu_forward.dropout_masks[layer.index].as_ref() {
            Some(mask) => {
                let mask_gpu = self
                    .backend
                    .upload_f32(&tensor_backend_shape(mask.shape), &mask.data)
                    .map_err(|err| backend_error("upload dropout mask", err))?;
                self.backend
                    .dropout_backward_f32(
                        &grad,
                        &mask_gpu,
                        &tensor_backend_shape(mask.shape),
                        config,
                    )
                    .map_err(|err| backend_error("GPU dropout backward", err))?
            }
            None => grad,
        };
        self.accumulate_previous_gpu_gradient(layer.index, input_grad, layer_output_gradients)
    }
}
