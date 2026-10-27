//! Native f32 DCNv4 math, before runtime bias/BN/activation. No host tensor hops.

use lightnet_cuda::ElementwiseLaunchConfig;
use lightnet_cuda::host_ops::dcnv4::Dcnv4Composition;

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
    /// Device DCNv4 aggregation (offsets given).
    #[allow(clippy::too_many_arguments)]
    pub fn dcnv4_aggregate_native(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::dcnv4::Dcnv4Geometry,
        offset_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dcnv4_aggregate_native";
        geometry.checked().map_err(|msg| shape_err(op, msg))?;
        let points = geometry.kernel_points();
        let plane = geometry.input_height * geometry.input_width;
        let spatial = geometry.spatial();
        let offset_dim = geometry.offset_dim();
        if input.len() != geometry.batch * geometry.channels * plane
            || offsets.len() != geometry.batch * spatial * offset_dim
        {
            return Err(shape_err(op, "operand length mismatch".to_owned()));
        }
        let out_len = geometry.batch * geometry.channels * spatial;
        // Zeroed: the kernel accumulates into each owned output cell.
        let mut out = self.alloc(out_len)?;
        let threads = geometry.batch * geometry.groups * spatial;
        let scratch = self.alloc(threads * points)?;
        // SAFETY: one thread per (batch, group, cell); each writes only its own
        // group's channels, and every tap is bounds-checked in the kernel.
        unsafe {
            self.dcnv4.dcnv4_aggregate(
                &self.stream,
                launch::elementwise(threads, config),
                &input.buf,
                &offsets.buf,
                &mut out,
                scratch.cu_deviceptr() as *mut f32,
                geometry.channels,
                geometry.groups,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(geometry.softmax),
                u32::from(geometry.remove_center),
                offset_scale,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            out,
            &[
                geometry.batch,
                geometry.channels,
                geometry.output_height,
                geometry.output_width,
            ],
        ))
    }

    /// Device adjoint of [`Self::dcnv4_aggregate_native`]: `(d_input, d_offsets)`.
    #[allow(clippy::too_many_arguments)]
    pub fn dcnv4_aggregate_backward_native(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::dcnv4::Dcnv4Geometry,
        offset_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dcnv4_aggregate_backward_native";
        geometry.checked().map_err(|msg| shape_err(op, msg))?;
        let points = geometry.kernel_points();
        let plane = geometry.input_height * geometry.input_width;
        let spatial = geometry.spatial();
        let offset_dim = geometry.offset_dim();
        if input.len() != geometry.batch * geometry.channels * plane
            || offsets.len() != geometry.batch * spatial * offset_dim
            || output_grad.len() != geometry.batch * geometry.channels * spatial
        {
            return Err(shape_err(op, "operand length mismatch".to_owned()));
        }
        let d_input = self.alloc(geometry.batch * geometry.channels * plane)?;
        let d_offsets = self.alloc(geometry.batch * spatial * offset_dim)?;
        let threads = geometry.batch * geometry.groups * spatial;
        let scratch = self.alloc(threads * points * 2)?;
        // SAFETY: one thread per (batch, group, cell); `d_offsets` slots are
        // owned by that thread, `d_input` is written only by atomic adds into a
        // zeroed buffer of the validated extent.
        unsafe {
            self.dcnv4_backward.dcnv4_aggregate_backward(
                &self.stream,
                launch::elementwise(threads, config),
                &input.buf,
                &offsets.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_offsets.cu_deviceptr() as *mut f32,
                scratch.cu_deviceptr() as *mut f32,
                geometry.channels,
                geometry.groups,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(geometry.softmax),
                u32::from(geometry.remove_center),
                offset_scale,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(
                d_input,
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
            ),
            OxideTensorF32::new(
                d_offsets,
                &[geometry.batch, geometry.spatial(), geometry.offset_dim()],
            ),
        ))
    }
}
