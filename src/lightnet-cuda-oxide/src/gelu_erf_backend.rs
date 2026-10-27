//! Native GELU erf backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err, shape_err};

impl OxideBackend {
    pub(crate) fn gelu_erf_backend_f32(
        &self,
        input: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "gelu_erf_f32";
        let len = lightnet_cuda::host_ops::gelu_erf::checked_len(shape)
            .map_err(|err| shape_err(op, err))?;
        if input.len() != len || input.shape() != shape {
            return Err(shape_err(op, "input shape/storage mismatch".to_owned()));
        }
        let input_f = self.f32v(input)?;
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: checked nonempty storage; each thread owns one output.
        unsafe {
            self.activations.gelu_erf(
                &self.stream,
                launch::elementwise(len, config),
                &input_f.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    pub(crate) fn gelu_erf_backward_backend_f32(
        &self,
        input: &OxideTensorF32,
        seeds: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "gelu_erf_backward_f32";
        let len = lightnet_cuda::host_ops::gelu_erf::checked_len(shape)
            .map_err(|err| shape_err(op, err))?;
        if input.len() != len
            || seeds.len() != len
            || input.shape() != shape
            || seeds.shape() != shape
        {
            return Err(shape_err(
                op,
                "input/seed shape or storage mismatch".to_owned(),
            ));
        }
        let input_f = self.f32v(input)?;
        let seeds_f = self.f32v(seeds)?;
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: all three arrays have the same checked length.
        unsafe {
            self.activations.gelu_erf_backward(
                &self.stream,
                launch::elementwise(len, config),
                &input_f.buf,
                &seeds_f.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }
}
