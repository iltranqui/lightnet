//! GPU-runtime glue for `[region]`.
use crate::error::{Error, Result};
use crate::model::{Layer, RegionSpec};
use crate::runtime::{
    GpuExecutableNetwork, GpuForwardState, backend_error, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn backward_region_gpu(
        &self,
        layer: &Layer,
        spec: &RegionSpec,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing GPU Region output for layer #{}",
                layer.index
            ))
        })?;
        let input_grad = self
            .backend
            .region_backward_f32(
                output,
                &grad,
                &tensor_backend_shape(layer.input_shape),
                spec.num,
                spec.coords + spec.classes + 1,
                spec.classes,
                spec.softmax,
                config,
            )
            .map_err(|err| backend_error("GPU Region backward", err))?;
        self.accumulate_previous_gpu_gradient(layer.index, input_grad, layer_output_gradients)
    }
}
