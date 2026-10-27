//! Native channel shuffle backend operation.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err, shape_err};

impl OxideBackend {
    pub(crate) fn channel_shuffle_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if shape.len() != 4
            || groups == 0
            || shape[1] % groups != 0
            || x.len() != shape.iter().product::<usize>()
        {
            return Err(shape_err(
                "channel_shuffle_f32",
                format!(
                    "bad shape {shape:?} / groups {groups} for {} elements",
                    x.len()
                ),
            ));
        }
        let (rows, cols) = if inverse {
            (shape[1] / groups, groups)
        } else {
            (groups, shape[1] / groups)
        };
        let len = shape[2] * shape[3];
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; every source index is < x.len() by the shape check.
        unsafe {
            self.channel_shuffle.channel_permute(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                rows,
                cols,
                len,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }
}
