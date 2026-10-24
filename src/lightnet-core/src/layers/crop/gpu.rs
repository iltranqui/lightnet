//! GPU-runtime glue for `[crop]`.
use crate::error::{Error, Result};
use crate::model::{CropSpec, Layer};
use crate::runtime::{
    GpuForwardState,
    CropForwardCache, GpuExecutableNetwork, backend_error, plan_crop_forward, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Plan the crop on the host, upload the sampling tables, run the bilinear kernel.
    pub(crate) fn forward_crop_gpu(
        &self,
        layer: &Layer,
        spec: &CropSpec,
        current_input: &B::Tensor,
        training: bool,
        rng_state: &mut u64,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, CropForwardCache)> {
        let (cache, adjustments) = plan_crop_forward(
            spec,
            layer.input_shape,
            layer.output_shape,
            training,
            rng_state,
        )?;
        let source_x = self
            .backend
            .upload_f32(&[cache.source_x.len()], &cache.source_x)
            .map_err(|err| backend_error("upload crop source x", err))?;
        let source_y = self
            .backend
            .upload_f32(&[cache.source_y.len()], &cache.source_y)
            .map_err(|err| backend_error("upload crop source y", err))?;
        let input_scales = self
            .backend
            .upload_f32(&[cache.input_scales.len()], &cache.input_scales)
            .map_err(|err| backend_error("upload crop input scales", err))?;
        let input_offsets = self
            .backend
            .upload_f32(&[cache.input_offsets.len()], &cache.input_offsets)
            .map_err(|err| backend_error("upload crop input offsets", err))?;
        let input_saturations: Vec<f32> = adjustments
            .iter()
            .map(|adjustment| adjustment.saturation)
            .collect();
        let input_exposures: Vec<f32> = adjustments
            .iter()
            .map(|adjustment| adjustment.exposure)
            .collect();
        let input_saturations = self
            .backend
            .upload_f32(&[input_saturations.len()], &input_saturations)
            .map_err(|err| backend_error("upload crop saturations", err))?;
        let input_exposures = self
            .backend
            .upload_f32(&[input_exposures.len()], &input_exposures)
            .map_err(|err| backend_error("upload crop exposures", err))?;
        let output = self
            .backend
            .crop_bilinear_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                &source_x,
                &source_y,
                &input_scales,
                &input_offsets,
                &input_saturations,
                &input_exposures,
                training,
                launch_config,
            )
            .map_err(|err| backend_error("GPU crop", err))?;
        Ok((output, cache))
    }

    /// Crop backward.
    pub(crate) fn backward_crop_gpu(
        &self,
        layer: &Layer,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let cache = forward.cpu_forward.crop_caches[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing crop forward cache for GPU layer #{}",
                    layer.index
                ))
            })?;
        let source_x = self
            .backend
            .upload_f32(&[cache.source_x.len()], &cache.source_x)
            .map_err(|err| backend_error("upload crop backward source x", err))?;
        let source_y = self
            .backend
            .upload_f32(&[cache.source_y.len()], &cache.source_y)
            .map_err(|err| backend_error("upload crop backward source y", err))?;
        let input_scales = self
            .backend
            .upload_f32(&[cache.input_scales.len()], &cache.input_scales)
            .map_err(|err| backend_error("upload crop backward input scales", err))?;
        let input_grad = self
            .backend
            .crop_bilinear_backward_f32(
                &grad,
                &tensor_backend_shape(layer.input_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                &source_x,
                &source_y,
                &input_scales,
                config,
            )
            .map_err(|err| backend_error("GPU crop backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
