//! Native f32 DCNv4 math, before runtime bias/BN/activation. No host tensor hops.

use darknet_cuda::ElementwiseLaunchConfig;
use darknet_cuda::host_ops::dcnv4::Dcnv4Composition;

use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err, shape_err};

/// Positive batch-summed gradients, in CPU/source parameter order.
pub struct Dcnv4NativeGradients {
    pub input: OxideTensorF32,
    pub offset_weights: OxideTensorF32,
    pub offset_biases: OxideTensorF32,
}

impl OxideBackend {
    /// Raw output plus dense NHWC offset cache. Projection weights use [P,C,ky,kx]
    /// and P is ceil(groups * 3 * (size²-remove_center) / 8) * 8.
    /// No runtime selection, BN, activation or optimizer eligibility change.
    pub fn dcnv4_forward_native(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        desc: Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dcnv4_forward_native";
        self.check_dcnv4_composition(input, weights, desc, op)?;
        let g = desc.geometry;
        let p = g.offset_dim();
        if biases.len() != p {
            return Err(shape_err(op, "offset bias length mismatch".to_owned()));
        }
        let ch = desc.coarse_height();
        let cw = desc.coarse_width();
        let coarse_len = g.batch * ch * cw * p;
        let mut coarse = self.alloc(coarse_len)?;
        // SAFETY: descriptor/storage checked before launch; one thread per
        // coarse slot, with bounds-checked predictor taps and explicit extents.
        unsafe {
            self.dcnv4_projection.dcnv4_project(
                &self.stream,
                launch::elementwise(coarse_len, config),
                &input.buf,
                &weights.buf,
                &biases.buf,
                &mut coarse,
                g.channels,
                g.input_height,
                g.input_width,
                ch,
                cw,
                p,
                g.size,
                g.stride_y * desc.d_stride,
                g.stride_x * desc.d_stride,
                g.padding,
                g.dilation,
            )
        }
        .map_err(launch_err)?;
        let dense_len = g.batch * g.spatial() * p;
        let mut dense = self.alloc(dense_len)?;
        // SAFETY: dense/coarse extents checked; each thread owns one dense slot.
        unsafe {
            self.dcnv4_projection.dcnv4_offset_upsample(
                &self.stream,
                launch::elementwise(dense_len, config),
                &coarse,
                &mut dense,
                ch,
                cw,
                g.output_height,
                g.output_width,
                p,
                desc.d_stride,
            )
        }
        .map_err(launch_err)?;
        let offsets = OxideTensorF32::new(dense, &[g.batch, g.spatial(), p]);
        let output = self.dcnv4_aggregate_native(input, &offsets, g, desc.offset_scale, config)?;
        Ok((output, offsets))
    }

    /// Exact Rust-mode adjoint using forward's dense offset cache. No averaging,
    /// 0.01 learning-rate multiplier or parameter updates belong in this API.
    pub fn dcnv4_backward_native(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        offsets: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        desc: Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<Dcnv4NativeGradients, OxideError> {
        let op = "dcnv4_backward_native";
        self.check_dcnv4_composition(input, weights, desc, op)?;
        let g = desc.geometry;
        let p = g.offset_dim();
        // Aggregate validates offset and output seed storage before allocation.
        let (input_grad, dense_grad) = self.dcnv4_aggregate_backward_native(
            input,
            offsets,
            output_grad,
            g,
            desc.offset_scale,
            config,
        )?;
        let ch = desc.coarse_height();
        let cw = desc.coarse_width();
        let coarse_len = g.batch * ch * cw * p;
        let coarse_grad = self.alloc(coarse_len)?;
        // SAFETY: checked extents; zeroed output with only atomic scatter writers.
        unsafe {
            self.dcnv4_projection.dcnv4_offset_upsample_backward(
                &self.stream,
                launch::elementwise(dense_grad.len(), config),
                &dense_grad.buf,
                coarse_grad.cu_deviceptr() as *mut f32,
                ch,
                cw,
                g.output_height,
                g.output_width,
                p,
                desc.d_stride,
            )
        }
        .map_err(launch_err)?;
        let dw = self.alloc(desc.weight_count())?;
        let db = self.alloc(p)?;
        // SAFETY: all storage/geometry checked. Fresh dw/db are zeroed; dx
        // already holds aggregation adjoints. Same stream orders the scatters;
        // no input/cache alias is modified. All writers are atomic.
        unsafe {
            self.dcnv4_projection.dcnv4_project_backward(
                &self.stream,
                launch::elementwise(coarse_len, config),
                &input.buf,
                &weights.buf,
                &coarse_grad,
                input_grad.buf.cu_deviceptr() as *mut f32,
                dw.cu_deviceptr() as *mut f32,
                db.cu_deviceptr() as *mut f32,
                g.channels,
                g.input_height,
                g.input_width,
                ch,
                cw,
                p,
                g.size,
                g.stride_y * desc.d_stride,
                g.stride_x * desc.d_stride,
                g.padding,
                g.dilation,
            )
        }
        .map_err(launch_err)?;
        Ok(Dcnv4NativeGradients {
            input: input_grad,
            offset_weights: OxideTensorF32::new(dw, &[p, g.channels, g.size, g.size]),
            offset_biases: OxideTensorF32::new(db, &[p]),
        })
    }

    fn check_dcnv4_composition(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        desc: Dcnv4Composition,
        op: &str,
    ) -> Result<(), OxideError> {
        desc.checked().map_err(|msg| shape_err(op, msg))?;
        let g = desc.geometry;
        if g.kernel_points() > crate::kernels::dcnv4::MAX_POINTS {
            return Err(shape_err(
                op,
                "too many kernel points for device mask buffer".to_owned(),
            ));
        }
        if input.len() != g.batch * g.channels * g.input_height * g.input_width
            || weights.len() != desc.weight_count()
        {
            return Err(shape_err(
                op,
                "input/offset weight length mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}
