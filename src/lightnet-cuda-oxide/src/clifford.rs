//! Native Clifford products and residual. Tensor lengths are checked before launch.
use lightnet_cuda::ElementwiseLaunchConfig;
use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

fn geometry(batch: usize, channels: usize, spatial: usize) -> Result<usize, OxideError> {
    if batch == 0 || channels == 0 || spatial == 0 {
        return Err(OxideError::Shape("clifford: dimensions must be positive".to_owned()));
    }
    batch.checked_mul(channels).and_then(|n| n.checked_mul(spatial))
        .ok_or_else(|| OxideError::Shape("clifford: element count overflows".to_owned()))
}

impl OxideBackend {
pub(crate) fn clifford_fourier_native(
    &self, input: &OxideTensorF32, grad: Option<&OxideTensorF32>, wr: &OxideTensorF32, wi: &OxideTensorF32,
    batch: usize, channels: usize, height: usize, width: usize, mh: usize, mw: usize,
    config: ElementwiseLaunchConfig,
) -> Result<(OxideTensorF32, Option<(OxideTensorF32, OxideTensorF32)>), OxideError> {
    let spatial = height.checked_mul(width).ok_or_else(|| OxideError::Shape("Fourier spatial overflow".into()))?;
    let modes = mh.checked_mul(mw).filter(|&n| n > 0).ok_or_else(|| OxideError::Shape("Fourier modes invalid".into()))?;
    let len = geometry(batch, channels, spatial)?;
    let weights = channels.checked_mul(modes).ok_or_else(|| OxideError::Shape("Fourier weights overflow".into()))?;
    let coeffs = batch.checked_mul(weights).ok_or_else(|| OxideError::Shape("Fourier coefficients overflow".into()))?;
    let input = self.f32v(input)?; let wr = self.f32v(wr)?; let wi = self.f32v(wi)?;
    let grad = grad.map(|g| self.f32v(g)).transpose()?;
    if input.len() != len || wr.len() != weights || wi.len() != weights || grad.as_ref().is_some_and(|g| g.len() != len) {
        return Err(OxideError::Shape("Fourier length mismatch".into()));
    }
    let mut re = self.alloc_uninit(coeffs)?; let mut im = self.alloc_uninit(coeffs)?;
    // SAFETY: validated sizes and each thread owns one coefficient or output.
    unsafe { self.clifford.clifford_fourier_coeff(&self.stream, launch::elementwise(coeffs, config),
        &input.buf, &mut re, &mut im, width, height, mh, mw) }.map_err(launch_err)?;
    let mut out = self.alloc_uninit(len)?;
    if let Some(grad) = grad {
        let mut dre = self.alloc_uninit(coeffs)?; let mut dim = self.alloc_uninit(coeffs)?;
        unsafe { self.clifford.clifford_fourier_coeff(&self.stream, launch::elementwise(coeffs, config),
            &grad.buf, &mut dre, &mut dim, width, height, mh, mw) }.map_err(launch_err)?;
        unsafe { self.clifford.clifford_fourier_synthesis(&self.stream, launch::elementwise(len, config),
            &dre, &dim, &wr.buf, &wi.buf, &mut out, channels, width, height, mh, mw, true) }.map_err(launch_err)?;
        let mut dw = self.alloc_uninit(weights)?; let mut dv = self.alloc_uninit(weights)?;
        unsafe { self.clifford.clifford_fourier_weights(&self.stream, launch::elementwise(weights, config),
            &re, &im, &dre, &dim, &mut dw, &mut dv, batch, channels, spatial, mh, mw) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(out, &[batch, channels, height, width]), Some((OxideTensorF32::new(dw, &[channels, modes]), OxideTensorF32::new(dv, &[channels, modes])))))
    } else {
        unsafe { self.clifford.clifford_fourier_synthesis(&self.stream, launch::elementwise(len, config),
            &re, &im, &wr.buf, &wi.buf, &mut out, channels, width, height, mh, mw, false) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(out, &[batch, channels, height, width]), None))
    }
}

    pub(crate) fn clifford_interaction_native(
        &self, state: &OxideTensorF32, context: &OxideTensorF32,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let len = geometry(batch, channels, spatial)?;
        let state = self.f32v(state)?;
        let context = self.f32v(context)?;
        if state.len() != len || context.len() != if pooled { len / spatial } else { len } {
            return Err(OxideError::Shape("clifford interaction: length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread per output; all rolled reads are within validated extents.
        unsafe {
            self.clifford.clifford_interaction(&self.stream, launch::elementwise(len, config),
                &state.buf, &context.buf, &mut output, channels, spatial, shift % channels, inner, pooled)
        }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, channels, 1, spatial]))
    }

    pub(crate) fn clifford_interaction_backward_native(
        &self, state: &OxideTensorF32, context: &OxideTensorF32, grad: &OxideTensorF32,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let len = geometry(batch, channels, spatial)?;
        let ctx_len = if pooled { len / spatial } else { len };
        let state = self.f32v(state)?;
        let context = self.f32v(context)?;
        let grad = self.f32v(grad)?;
        if state.len() != len || grad.len() != len || context.len() != ctx_len {
            return Err(OxideError::Shape("clifford interaction backward: length mismatch".to_owned()));
        }
        let mut ds = self.alloc_uninit(len)?;
        let mut dc = self.alloc_uninit(ctx_len)?;
        // SAFETY: gather adjoints give each output element one writer, including
        // pooled reductions and aliased channels (shift zero or repeated shifts).
        unsafe {
            self.clifford.clifford_interaction_state_backward(
                &self.stream, launch::elementwise(len, config), &state.buf, &context.buf,
                &grad.buf, &mut ds, channels, spatial, shift % channels, inner, pooled)
        }.map_err(launch_err)?;
        unsafe {
            self.clifford.clifford_interaction_context_backward(
                &self.stream, launch::elementwise(ctx_len, config), &state.buf, &context.buf,
                &grad.buf, &mut dc, channels, spatial, shift % channels, inner, pooled)
        }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(ds, &[batch, channels, 1, spatial]),
            OxideTensorF32::new(dc, &[batch, channels, 1, if pooled { 1 } else { spatial }])))
    }

    pub(crate) fn clifford_residual_native(
        &self, input: &OxideTensorF32, mixed: &OxideTensorF32, scale: &OxideTensorF32,
        mask: &OxideTensorF32, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let len = geometry(batch, channels, spatial)?;
        let input = self.f32v(input)?;
        let mixed = self.f32v(mixed)?;
        let scale = self.f32v(scale)?;
        let mask = self.f32v(mask)?;
        if input.len() != len || mixed.len() != len || scale.len() != channels || mask.len() != batch {
            return Err(OxideError::Shape("clifford residual: length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: each output element reads one validated input, scale and sample mask.
        unsafe {
            self.clifford.clifford_residual(&self.stream, launch::elementwise(len, config),
                &input.buf, &mixed.buf, &scale.buf, &mask.buf, &mut output, channels, spatial)
        }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, input.shape()))
    }

    pub(crate) fn clifford_residual_backward_native(
        &self, mixed: &OxideTensorF32, grad: &OxideTensorF32, scale: &OxideTensorF32,
        mask: &OxideTensorF32, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let len = geometry(batch, channels, spatial)?;
        let mixed = self.f32v(mixed)?;
        let grad = self.f32v(grad)?;
        let scale = self.f32v(scale)?;
        let mask = self.f32v(mask)?;
        if mixed.len() != len || grad.len() != len || scale.len() != channels || mask.len() != batch {
            return Err(OxideError::Shape("clifford residual backward: length mismatch".to_owned()));
        }
        let mut dm = self.alloc_uninit(len)?;
        let mut ds = self.alloc_uninit(channels)?;
        // SAFETY: each element/channel has one writer; no intermediate atomics.
        unsafe {
            self.clifford.clifford_residual_backward(&self.stream, launch::elementwise(len, config),
                &grad.buf, &scale.buf, &mask.buf, &mut dm, channels, spatial)
        }.map_err(launch_err)?;
        unsafe {
            self.clifford.clifford_scale_backward(&self.stream, launch::elementwise(channels, config),
                &grad.buf, &mixed.buf, &mask.buf, &mut ds, channels, spatial)
        }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(dm, grad.shape()), OxideTensorF32::new(ds, &[channels])))
    }
}
