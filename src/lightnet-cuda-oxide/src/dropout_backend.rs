//! dropout backend dispatch.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, same_len, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn dropout_f32_backend(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        probability: f32,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "dropout_f32";
        if !(0.0..1.0).contains(&probability) || x.is_empty() {
            return Err(shape_err(
                op,
                format!("probability {probability} not in [0, 1) or empty input"),
            ));
        }
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", x.len()),
            ));
        }
        let mut out = self.alloc_uninit(x.len())?;
        let mut mask = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `out` and `mask` slot.
        unsafe {
            self.dropout.dropout_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                &mut mask,
                probability,
                1.0 / (1.0 - probability),
                seed,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(out, shape),
            OxideTensorF32::new(mask, shape),
        ))
    }

    pub(crate) fn dropout_backward_f32_backend(
        &self,
        output_grad: &OxideTensorF32,
        mask: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let mask_f = self.f32v(mask)?;
        let mask = &*mask_f;
        same_len("dropout_backward_f32", output_grad, mask)?;
        if shape.iter().product::<usize>() != mask.len() {
            return Err(shape_err(
                "dropout_backward_f32",
                format!("shape {shape:?} does not match {} elements", mask.len()),
            ));
        }
        let mut dx = self.alloc_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.dropout.dropout_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mask.buf,
                &mut dx,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }

    pub(crate) fn dropblock_f32_backend(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        block_prob: f32,
        block_size: usize,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "dropblock_f32";
        if shape.len() != 4 || shape.iter().product::<usize>() != x.len() || x.is_empty() {
            return Err(shape_err(
                op,
                format!("shape {shape:?} is not NCHW of {} elements", x.len()),
            ));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if block_size == 0 || block_size > w.min(h) || !(0.0..=1.0).contains(&block_prob) {
            return Err(shape_err(
                op,
                format!(
                    "block_size {block_size} must be in 1..={} and prob {block_prob} in [0, 1]",
                    w.min(h)
                ),
            ));
        }
        let planes = n * c;
        let mut keep = self.alloc_uninit(x.len())?;
        let mut dropped = self.alloc_uninit(planes)?;
        let mut scales = self.alloc_uninit(n)?;
        let mut out = self.alloc_uninit(x.len())?;
        let mut mask = self.alloc_uninit(x.len())?;
        // SAFETY: stage 1 writes only its own plane of `keep` and slot of
        // `dropped`; stage 2 one `scales` slot per sample; stage 3 one element.
        unsafe {
            self.dropout.dropblock_plane(
                &self.stream,
                launch::elementwise(planes, config),
                &mut keep,
                &mut dropped,
                seed,
                w,
                h,
                block_prob,
                block_size,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: see above.
        unsafe {
            self.dropout.dropblock_scales(
                &self.stream,
                launch::elementwise(n, config),
                &dropped,
                &mut scales,
                c,
                c * h * w,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: see above.
        unsafe {
            self.dropout.dropblock_apply(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &keep,
                &scales,
                &mut out,
                &mut mask,
                c * h * w,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(out, shape),
            OxideTensorF32::new(mask, shape),
        ))
    }
}
