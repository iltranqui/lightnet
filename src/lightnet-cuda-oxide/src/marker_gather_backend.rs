//! marker_gather backend dispatch.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn marker_gather_f32_backend(
        &self,
        features: &OxideTensorF32,
        markers: &OxideTensorF32,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "marker_gather_f32";
        let d = lightnet_cuda::host_ops::marker_gather::Layout::new(
            feature_shape,
            marker_shape,
            options,
        )
        .map_err(|err| shape_err(op, err))?;
        if features.len() != d.input_len || markers.len() != d.markers_len {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let features_f = self.f32v(features)?;
        let markers_f = self.f32v(markers)?;
        let mut output = self.alloc_uninit(d.output_len)?;
        // SAFETY: checked dimensions/storage bound every feature and marker index;
        // one thread owns each output element.
        unsafe {
            self.laya.marker_gather(
                &self.stream,
                launch::elementwise(d.output_len, config),
                &features_f.buf,
                &markers_f.buf,
                &mut output,
                d.hidden,
                d.sequence,
                d.marker_channels,
                d.options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &d.output_shape()))
    }

    pub(crate) fn marker_gather_backward_f32_backend(
        &self,
        markers: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "marker_gather_backward_f32";
        let d = lightnet_cuda::host_ops::marker_gather::Layout::new(
            feature_shape,
            marker_shape,
            options,
        )
        .map_err(|err| shape_err(op, err))?;
        if markers.len() != d.markers_len || output_grad.len() != d.output_len {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let markers_f = self.f32v(markers)?;
        let grad_f = self.f32v(output_grad)?;
        let mut dx = self.alloc_uninit(d.input_len)?;
        // SAFETY: validated layout bounds reads and one thread owns each dx.
        unsafe {
            self.laya.marker_gather_backward(
                &self.stream,
                launch::elementwise(d.input_len, config),
                &markers_f.buf,
                &grad_f.buf,
                &mut dx,
                d.hidden,
                d.sequence,
                d.marker_channels,
                d.options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, feature_shape))
    }
}
