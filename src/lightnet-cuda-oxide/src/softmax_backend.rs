//! Native softmax backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{same_len, OxideBackend, OxideError, OxideTensorF32, launch, launch_err, shape_err};

impl OxideBackend {
    pub(crate) fn softmax_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "softmax_f32";
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", x.len()),
            ));
        }
        if group_size == 0 || x.is_empty() || x.len() % group_size != 0 {
            return Err(shape_err(
                op,
                format!(
                    "length {} is not a multiple of group size {group_size}",
                    x.len()
                ),
            ));
        }
        let n_groups = x.len() / group_size;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.softmax.softmax_forward(
                &self.stream,
                launch::elementwise(n_groups, config),
                &x.buf,
                &mut out,
                group_size,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    pub(crate) fn softmax_backend_backward_f32(
        &self,
        output: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "softmax_backward_f32";
        same_len(op, output, output_grad)?;
        if shape.iter().product::<usize>() != output.len() {
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", output.len()),
            ));
        }
        if group_size == 0 || output.is_empty() || output.len() % group_size != 0 {
            return Err(shape_err(
                op,
                format!(
                    "length {} is not a multiple of group size {group_size}",
                    output.len()
                ),
            ));
        }
        let n_groups = output.len() / group_size;
        let mut dx = self.alloc_uninit(output.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.softmax.softmax_backward(
                &self.stream,
                launch::elementwise(n_groups, config),
                &output.buf,
                &output_grad.buf,
                &mut dx,
                group_size,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }
}
