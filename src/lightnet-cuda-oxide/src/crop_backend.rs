//! Native crop backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{nchw_batch, shape_err, OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

/// Validated crop geometry: `(batch, out_h, out_w)`.
fn crop_layout(
    op: &str,
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    coords: (usize, usize),
    adjustments: &[usize],
) -> Result<(), OxideError> {
    if channels == 0 || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(shape_err(
            op,
            "channels and spatial dims must be > 0".to_owned(),
        ));
    }
    let expected = batch * out_h * out_w;
    if coords.0 != expected || coords.1 != expected {
        return Err(shape_err(
            op,
            format!(
                "expected {expected} source coordinates, got x={} y={}",
                coords.0, coords.1
            ),
        ));
    }
    let valid_adjustments = match adjustments {
        [scales] => *scales == batch,
        [scales, offsets] => *scales == batch && *offsets == batch * channels,
        _ => false,
    };
    if !valid_adjustments {
        return Err(shape_err(
            op,
            format!(
                "expected per-batch scales ({batch}) and optional per-channel offsets ({}), got {adjustments:?}",
                batch * channels
            ),
        ));
    }
    Ok(())
}

impl OxideBackend {
    pub(crate) fn crop_backend_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &OxideTensorF32,
        source_y: &OxideTensorF32,
        input_scales: &OxideTensorF32,
        input_offsets: &OxideTensorF32,
        input_saturations: &OxideTensorF32,
        input_exposures: &OxideTensorF32,
        apply_hsv: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let source_x_f = self.f32v(source_x)?;
        let source_x = &*source_x_f;
        let source_y_f = self.f32v(source_y)?;
        let source_y = &*source_y_f;
        let input_scales_f = self.f32v(input_scales)?;
        let input_scales = &*input_scales_f;
        let input_offsets_f = self.f32v(input_offsets)?;
        let input_offsets = &*input_offsets_f;
        let input_saturations_f = self.f32v(input_saturations)?;
        let input_saturations = &*input_saturations_f;
        let input_exposures_f = self.f32v(input_exposures)?;
        let input_exposures = &*input_exposures_f;
        let op = "crop_bilinear_f32";
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, h, w]"),
            ));
        }
        let batch = output_shape[0];
        crop_layout(
            op,
            batch,
            channels,
            input_height,
            input_width,
            output_shape[2],
            output_shape[3],
            (source_x.len(), source_y.len()),
            &[input_scales.len(), input_offsets.len()],
        )?;
        if input_saturations.len() != batch || input_exposures.len() != batch {
            return Err(shape_err(
                op,
                format!(
                    "expected one saturation/exposure adjustment per batch ({batch}), got {} and {}",
                    input_saturations.len(),
                    input_exposures.len()
                ),
            ));
        }
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}",
                    x.len()
                ),
            ));
        }
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; taps are bounds-checked and
        // coordinate / adjustment lengths were validated above.
        unsafe {
            self.resample.crop_bilinear(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &source_x.buf,
                &source_y.buf,
                &input_scales.buf,
                &input_offsets.buf,
                &input_saturations.buf,
                &input_exposures.buf,
                apply_hsv,
                &mut out,
                channels,
                input_height,
                input_width,
                output_shape[2],
                output_shape[3],
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    pub(crate) fn crop_backend_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &OxideTensorF32,
        source_y: &OxideTensorF32,
        input_scales: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let source_x_f = self.f32v(source_x)?;
        let source_x = &*source_x_f;
        let source_y_f = self.f32v(source_y)?;
        let source_y = &*source_y_f;
        let input_scales_f = self.f32v(input_scales)?;
        let input_scales = &*input_scales_f;
        let op = "crop_bilinear_backward_f32";
        let g = output_grad.shape();
        if g.len() != 4 || g[1] != channels {
            return Err(shape_err(
                op,
                format!("gradient shape {g:?} is not [batch, {channels}, h, w]"),
            ));
        }
        let batch = g[0];
        crop_layout(
            op,
            batch,
            channels,
            input_height,
            input_width,
            g[2],
            g[3],
            (source_x.len(), source_y.len()),
            &[input_scales.len()],
        )?;
        nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )
        .and_then(|b| {
            if b == batch {
                Ok(())
            } else {
                Err(shape_err(
                    op,
                    format!("input batch {b} != gradient batch {batch}"),
                ))
            }
        })?;
        let dx = self.alloc(input_shape.iter().product())?;
        // SAFETY: `dx` is a live zeroed buffer of the validated extent; the
        // kernel only touches it through in-image atomic adds.
        unsafe {
            self.resample.crop_bilinear_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &output_grad.buf,
                &source_x.buf,
                &source_y.buf,
                &input_scales.buf,
                dx.cu_deviceptr() as *mut f32,
                channels,
                input_height,
                input_width,
                g[2],
                g[3],
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

}
