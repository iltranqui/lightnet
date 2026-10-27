//! Depth Anything and depth-pooling backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn depth_backend_tap_clamp_f32(
        &self,
        raw: &OxideTensorF32,
        gradient: Option<&OxideTensorF32>,
        shape: &[usize],
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let grad = self.f32v(gradient.unwrap_or(raw))?;
        let raw = self.f32v(raw)?;
        let len = shape
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or_else(|| OxideError::Shape("depth clamp shape overflow".into()))?;
        if raw.len() != len || grad.len() != len || !clip.is_finite() || clip <= 0.0 {
            return Err(OxideError::Shape("depth clamp geometry mismatch".into()));
        }
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: one writer per element; both input buffers have checked length.
        unsafe {
            self.vit.depth_tap_clamp(
                &self.stream,
                launch::elementwise(len, config),
                &raw.buf,
                &grad.buf,
                &mut out,
                clip,
                gradient.is_some(),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    pub(crate) fn depth_backend_maxpool_depth_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        input_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        if x.shape().len() != 4
            || output_shape.len() != 4
            || input_channels == 0
            || output_shape[1] == 0
            || output_shape[2] == 0
            || output_shape[3] == 0
            || output_shape[1] > input_channels
            || x.shape()[1] != input_channels
            || x.shape()[0] != output_shape[0]
            || x.shape()[2] != output_shape[2]
            || x.shape()[3] != output_shape[3]
        {
            return Err(shape_err(
                "maxpool_depth_f32",
                "incompatible NCHW shapes".to_owned(),
            ));
        }
        let spatial = output_shape[2] * output_shape[3];
        let mut output = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: shape checks establish matching batch/spatial dimensions and
        // valid strided channel groups; each output is written exactly once.
        unsafe {
            self.pool.maxpool_depth_forward(
                &self.stream,
                launch::elementwise(output.len(), config),
                &x.buf,
                &mut output,
                input_channels,
                output_shape[1],
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn depth_backend_maxpool_depth_backward_f32(
        &self,
        x: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        output_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        if input_shape.len() != 4
            || x.shape() != input_shape
            || output_grad.shape().len() != 4
            || output_channels == 0
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] == 0
            || input_shape[3] == 0
            || output_channels > input_shape[1]
            || output_grad.shape()[0] != input_shape[0]
            || output_grad.shape()[1] != output_channels
            || output_grad.shape()[2] != input_shape[2]
            || output_grad.shape()[3] != input_shape[3]
        {
            return Err(shape_err(
                "maxpool_depth_backward_f32",
                "incompatible NCHW shapes".to_owned(),
            ));
        }
        let spatial = input_shape[2] * input_shape[3];
        let mut dx = self.alloc_uninit(x.len())?;
        // SAFETY: shape checks establish matching extents; each input gradient
        // has one owner and all entries are written.
        unsafe {
            self.pool.maxpool_depth_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &x.buf,
                &output_grad.buf,
                &mut dx,
                input_shape[1],
                output_channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }
}
