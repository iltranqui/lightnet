//! scale_channels backend dispatch.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, same_len, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn scale_channels_f32_backend(
        &self,
        scale: &OxideTensorF32,
        source: &OxideTensorF32,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        let source_f = self.f32v(source)?;
        let source = &*source_f;
        let (channels, hw) =
            self.scale_channels_layout("scale_channels_f32", scale, source, shape, scale_wh)?;
        let mut out = self.alloc_uninit(source.len())?;
        // SAFETY: one thread per output element; scale index is in range by
        // the layout check.
        unsafe {
            self.scale_channels.scale_channels_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &scale.buf,
                &source.buf,
                &mut out,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    pub(crate) fn scale_channels_backward_f32_backend(
        &self,
        delta: &OxideTensorF32,
        scale: &OxideTensorF32,
        source: &OxideTensorF32,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        let source_f = self.f32v(source)?;
        let source = &*source_f;
        let op = "scale_channels_backward_f32";
        let (channels, hw) = self.scale_channels_layout(op, scale, source, shape, scale_wh)?;
        same_len(op, delta, source)?;
        let mut d_src = self.alloc_uninit(source.len())?;
        let mut d_scale = self.alloc_uninit(scale.len())?;
        // SAFETY: one thread per `d_src` element / per scale element.
        unsafe {
            self.scale_channels.scale_channels_dsource(
                &self.stream,
                launch::elementwise(d_src.len(), config),
                &delta.buf,
                &scale.buf,
                &mut d_src,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: as above; each scale element is owned by one thread.
        unsafe {
            self.scale_channels.scale_channels_dscale(
                &self.stream,
                launch::elementwise(d_scale.len(), config),
                &delta.buf,
                &source.buf,
                &mut d_scale,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(d_scale, scale.shape()),
            OxideTensorF32::new(d_src, shape),
        ))
    }
}
