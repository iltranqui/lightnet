//! Dense detection-head forward backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn centernet_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "centernet_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] < classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible CenterNet shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.centernet_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn modern_yolo_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "modern_yolo_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] <= classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible modern-YOLO shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.modern_yolo_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }
}
