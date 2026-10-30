//! Native softmax backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{same_len, OxideBackend, OxideError, OxideTensorF16, OxideTensorF32, launch, launch_err, shape_err};

impl OxideBackend {
    /// f16-storage spatial softmax over channels (f32 math, `temperature` divides the logits).
    pub(crate) fn spatial_softmax_f16(
        &self, input: &OxideTensorF16, shape: &[usize], temperature: f32, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let op = "spatial_softmax_f16";
        if shape.len() != 4 || shape.contains(&0) || shape.iter().product::<usize>() != input.len() || temperature <= 0.0 {
            return Err(shape_err(op, format!("incompatible shape {shape:?} or temperature {temperature}")));
        }
        let plane = shape[2] * shape[3];
        let mut out = self.alloc16_uninit(input.len())?;
        // SAFETY: one pixel thread owns disjoint channel-strided output slots.
        unsafe {
            self.amp_k.amp_spatial_softmax_forward(&self.stream, launch::elementwise(shape[0] * plane, config), &input.buf, &mut out, shape[1], plane, temperature, self.half_fmt.code())
        }.map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, shape))
    }

    /// Backward of `spatial_softmax_f16` from the f16 probabilities.
    pub(crate) fn spatial_softmax_backward_f16(
        &self, output: &OxideTensorF16, output_grad: &OxideTensorF16, shape: &[usize], temperature: f32, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let op = "spatial_softmax_backward_f16";
        if shape.len() != 4 || shape.contains(&0) || shape.iter().product::<usize>() != output.len() || output_grad.len() != output.len() || temperature <= 0.0 {
            return Err(shape_err(op, format!("incompatible shape {shape:?} or temperature {temperature}")));
        }
        let plane = shape[2] * shape[3];
        let mut dx = self.alloc16_uninit(output.len())?;
        // SAFETY: one pixel thread writes disjoint channel-strided locations.
        unsafe {
            self.amp_k.amp_spatial_softmax_backward(&self.stream, launch::elementwise(shape[0] * plane, config), &output.buf, &output_grad.buf, &mut dx, shape[1], plane, temperature, self.half_fmt.code())
        }.map_err(launch_err)?;
        Ok(OxideTensorF16::new(dx, shape))
    }

    pub(crate) fn spatial_softmax_backend_f32(
        &self, input: &OxideTensorF32, shape: &[usize], config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if input.is_half() {
            let out = self.spatial_softmax_f16(&self.as_half(input)?, shape, 1.0, config)?;
            return Ok(self.wrap_half(out));
        }
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let op = "spatial_softmax_f32";
        if shape.len() != 4 || shape.contains(&0) || shape.iter().product::<usize>() != input.len() || input.shape() != shape {
            return Err(shape_err(op, format!("incompatible shape {shape:?}")));
        }
        let plane = shape[2] * shape[3];
        let mut out = self.alloc_uninit(input.len())?;
        // SAFETY: one pixel thread owns disjoint channel-strided output slots.
        unsafe {
            self.softmax.spatial_softmax_forward(&self.stream, launch::elementwise(shape[0] * plane, config), &input.buf, &mut out, shape[1], plane)
        }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }
    pub(crate) fn spatial_softmax_backend_backward_f32(
        &self, output: &OxideTensorF32, output_grad: &OxideTensorF32,
        shape: &[usize], config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if output.is_half() && output_grad.is_half() {
            let dx = self.spatial_softmax_backward_f16(&self.as_half(output)?, &self.as_half(output_grad)?, shape, 1.0, config)?;
            return Ok(self.wrap_half(dx));
        }
        let output_f = self.f32v(output)?;
        let grad_f = self.f32v(output_grad)?;
        let output = &*output_f;
        let grad = &*grad_f;
        let op = "spatial_softmax_backward_f32";
        same_len(op, output, grad)?;
        if shape.len() != 4 || shape.contains(&0) || shape.iter().product::<usize>() != output.len() || output.shape() != shape || grad.shape() != shape {
            return Err(shape_err(op, format!("incompatible shape {shape:?}")));
        }
        let plane = shape[2] * shape[3];
        let mut dx = self.alloc_uninit(output.len())?;
        // SAFETY: one pixel thread writes disjoint channel-strided locations.
        unsafe {
            self.softmax.spatial_softmax_backward(&self.stream, launch::elementwise(shape[0] * plane, config), &output.buf, &grad.buf, &mut dx, shape[1], plane)
        }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }
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
