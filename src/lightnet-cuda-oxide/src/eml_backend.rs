//! Native EML convolution backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn eml_convolutional_backend_f32(
        &self,
        x_branch: &OxideTensorF32,
        self_branch: &OxideTensorF32,
        residual: Option<&OxideTensorF32>,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "eml_convolutional_f32";
        let clamp = if clamp > 0.0 { clamp } else { 4.0 };
        let eps = if eps > 0.0 { eps } else { 1.0e-6 };
        let total: usize = output_shape.iter().product();
        if output_shape.is_empty() || total == 0 {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let x = self.f32v(x_branch)?;
        let y = self.f32v(self_branch)?;
        let residual_tensor = residual.map(|tensor| self.f32v(tensor)).transpose()?;
        let has_residual = residual_tensor.is_some();
        let x = &*x;
        let y = &*y;
        let residual = residual_tensor.as_ref().map(|tensor| &**tensor);
        if x.len() != total
            || y.len() != total
            || x.shape() != output_shape
            || y.shape() != output_shape
            || residual
                .is_some_and(|tensor| tensor.len() != total || tensor.shape() != output_shape)
        {
            return Err(shape_err(
                op,
                format!(
                    "branch/residual shapes must match {output_shape:?}/{total}; got {:?}/{}, {:?}/{}, {:?}",
                    x.shape(),
                    x.len(),
                    y.shape(),
                    y.len(),
                    residual.map(|tensor| (tensor.shape(), tensor.len()))
                ),
            ));
        }
        let mut output = self.alloc_uninit(total)?;
        let residual = residual.unwrap_or(x);
        // SAFETY: one thread owns each output index; input extents and shapes
        // match the validated nonempty output extent above.
        unsafe {
            self.eml.eml_convolutional_forward(
                &self.stream,
                launch::elementwise(total, config),
                &x.buf,
                &y.buf,
                &residual.buf,
                &mut output,
                clamp,
                eps,
                scale,
                has_residual as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn eml_convolutional_backend_backward_f32(
        &self,
        x_branch: &OxideTensorF32,
        self_branch: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "eml_convolutional_backward_f32";
        let clamp = if clamp > 0.0 { clamp } else { 4.0 };
        let eps = if eps > 0.0 { eps } else { 1.0e-6 };
        let total: usize = output_shape.iter().product();
        if output_shape.is_empty() || total == 0 {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let x = self.f32v(x_branch)?;
        let y = self.f32v(self_branch)?;
        let delta = self.f32v(output_grad)?;
        let x = &*x;
        let y = &*y;
        let delta = &*delta;
        if x.len() != total
            || y.len() != total
            || delta.len() != total
            || x.shape() != output_shape
            || y.shape() != output_shape
            || delta.shape() != output_shape
        {
            return Err(shape_err(
                op,
                format!(
                    "branch/delta shapes must match {output_shape:?}/{total}; got {:?}/{}, {:?}/{}, {:?}/{}",
                    x.shape(),
                    x.len(),
                    y.shape(),
                    y.len(),
                    delta.shape(),
                    delta.len()
                ),
            ));
        }
        let mut x_gradient = self.alloc_uninit(total)?;
        let mut self_gradient = self.alloc_uninit(total)?;
        // SAFETY: each thread writes matching positions in both gradients;
        // the three input extents and shapes have been checked above.
        unsafe {
            self.eml.eml_convolutional_backward(
                &self.stream,
                launch::elementwise(total, config),
                &x.buf,
                &y.buf,
                &delta.buf,
                &mut x_gradient,
                &mut self_gradient,
                clamp,
                eps,
                scale,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(x_gradient, output_shape),
            OxideTensorF32::new(self_gradient, output_shape),
        ))
    }
}
