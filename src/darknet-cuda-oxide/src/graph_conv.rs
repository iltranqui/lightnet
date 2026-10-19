//! Native uniform graph-neighbor aggregation primitives.

use darknet_cuda::{ElementwiseLaunchConfig, host_ops::graph_conv::GraphConvAggregationGeometry};

use crate::conv::{expect_len, expect_shape};
use crate::launch;
use crate::{OxideBackend, OxideError, OxideTensorF32, launch_err};

impl OxideBackend {
    pub(crate) fn graph_conv_aggregate_forward_native(
        &self,
        input: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "graph_conv_aggregate_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        expect_shape(OP, "input", input.shape(), &input_shape)?;
        expect_len(OP, "input", input.len(), g.input_len())?;
        let mut output = self.alloc_uninit(g.output_len())?;
        // SAFETY: checked geometry establishes all tensor lengths and the
        // kernel assigns one unique output element to every in-range thread.
        unsafe {
            self.graph_conv.graph_aggregate_fwd(
                &self.stream,
                launch::elementwise(g.output_len(), config),
                &input.buf,
                &mut output,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &output_shape))
    }

    pub(crate) fn graph_conv_reference_forward_native(
        &self,
        input: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "graph_conv_reference_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        expect_shape(OP, "input", input.shape(), &input_shape)?;
        expect_len(OP, "input", input.len(), g.input_len())?;
        let mut reference = self.alloc_uninit(g.output_len())?;
        // SAFETY: checked geometry bounds coordinates and lengths; each output
        // location has one owning thread, including zeroed invalid centers.
        unsafe {
            self.graph_conv.graph_reference_fwd(
                &self.stream,
                launch::elementwise(g.output_len(), config),
                &input.buf,
                &mut reference,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(reference, &output_shape))
    }

    pub(crate) fn graph_conv_reference_backward_native(
        &self,
        output_gradient: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "graph_conv_reference_backward_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        expect_shape(
            OP,
            "output gradient",
            output_gradient.shape(),
            &output_shape,
        )?;
        expect_len(OP, "output gradient", output_gradient.len(), g.output_len())?;
        let mut input_gradient = self.alloc_uninit(g.input_len())?;
        // SAFETY: checked geometry bounds coordinates and lengths; each input
        // owns at most one center-reference output contribution.
        unsafe {
            self.graph_conv.graph_reference_bwd(
                &self.stream,
                launch::elementwise(g.input_len(), config),
                &output_gradient.buf,
                &mut input_gradient,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_gradient, &input_shape))
    }

    pub(crate) fn graph_conv_learned_aggregate_forward_native(
        &self,
        input: &OxideTensorF32,
        edge_kernel: &OxideTensorF32,
        edge_biases: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "graph_conv_learned_aggregate_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        if groups == 0 || g.channels % groups != 0 {
            return Err(OxideError::Shape(
                "graph_conv learned aggregation has invalid groups".to_owned(),
            ));
        }
        let channels_per_group = g.channels / groups;
        let kernel_area = g.size * g.size;
        let edge_count = groups.checked_mul(kernel_area).ok_or_else(|| {
            OxideError::Shape("graph_conv learned edge count overflows usize".to_owned())
        })?;
        let expected_kernel = edge_count
            .checked_mul(channels_per_group)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                OxideError::Shape("graph_conv learned parameter length overflows usize".to_owned())
            })?;
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        expect_shape(OP, "input", input.shape(), &input_shape)?;
        expect_len(OP, "input", input.len(), g.input_len())?;
        expect_len(OP, "edge kernel", edge_kernel.len(), expected_kernel)?;
        expect_len(OP, "edge biases", edge_biases.len(), edge_count)?;
        let mut output = self.alloc_uninit(g.output_len())?;
        // SAFETY: validated geometry and exact tensor lengths bound every
        // input/parameter access; each output index has one owning thread.
        unsafe {
            self.graph_conv.graph_learned_aggregate_fwd(
                &self.stream,
                launch::elementwise(g.output_len(), config),
                &input.buf,
                &edge_kernel.buf,
                &edge_biases.buf,
                &mut output,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
                groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &output_shape))
    }

    pub(crate) fn graph_conv_learned_aggregate_backward_native(
        &self,
        input: &OxideTensorF32,
        edge_kernel: &OxideTensorF32,
        edge_biases: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        const OP: &str = "graph_conv_learned_aggregate_backward_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        if groups == 0 || g.channels % groups != 0 {
            return Err(OxideError::Shape(
                "graph_conv learned aggregation has invalid groups".to_owned(),
            ));
        }
        let channels_per_group = g.channels / groups;
        let kernel_area = g.size * g.size;
        let edge_count = groups.checked_mul(kernel_area).ok_or_else(|| {
            OxideError::Shape("graph_conv learned edge count overflows usize".to_owned())
        })?;
        let kernel_len = edge_count
            .checked_mul(channels_per_group)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                OxideError::Shape("graph_conv learned parameter length overflows usize".to_owned())
            })?;
        let sites = g
            .batch
            .checked_mul(groups)
            .and_then(|n| n.checked_mul(g.output_height))
            .and_then(|n| n.checked_mul(g.output_width))
            .ok_or_else(|| {
                OxideError::Shape("graph_conv learned site count overflows usize".to_owned())
            })?;
        let alpha_len = sites.checked_mul(kernel_area).ok_or_else(|| {
            OxideError::Shape("graph_conv learned scratch length overflows usize".to_owned())
        })?;
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        expect_shape(OP, "input", input.shape(), &input_shape)?;
        expect_len(OP, "input", input.len(), g.input_len())?;
        expect_len(OP, "edge kernel", edge_kernel.len(), kernel_len)?;
        expect_len(OP, "edge biases", edge_biases.len(), edge_count)?;
        expect_shape(
            OP,
            "output gradient",
            output_gradient.shape(),
            &output_shape,
        )?;
        expect_len(OP, "output gradient", output_gradient.len(), g.output_len())?;

        let mut alpha = self.alloc_uninit(alpha_len)?;
        let mut logits_gradient = self.alloc_uninit(alpha_len)?;
        // SAFETY: checked extents bound all scratch and tensor accesses. Each
        // edge thread owns one alpha and one later thread owns each dlogit.
        unsafe {
            self.graph_conv.graph_learned_alpha_fwd(
                &self.stream,
                launch::elementwise(alpha_len, config),
                &input.buf,
                &edge_kernel.buf,
                &edge_biases.buf,
                &mut alpha,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
                groups,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.graph_conv.graph_learned_logits_bwd(
                &self.stream,
                launch::elementwise(alpha_len, config),
                &input.buf,
                &output_gradient.buf,
                &alpha,
                &mut logits_gradient,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
                groups,
            )
        }
        .map_err(launch_err)?;
        let mut input_gradient = self.alloc_uninit(g.input_len())?;
        unsafe {
            self.graph_conv.graph_learned_input_bwd(
                &self.stream,
                launch::elementwise(g.input_len(), config),
                &output_gradient.buf,
                &edge_kernel.buf,
                &alpha,
                &logits_gradient,
                &mut input_gradient,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
                groups,
            )
        }
        .map_err(launch_err)?;
        let mut edge_kernel_gradient = self.alloc_uninit(kernel_len)?;
        unsafe {
            self.graph_conv.graph_learned_kernel_bwd(
                &self.stream,
                launch::elementwise(kernel_len, config),
                &input.buf,
                &logits_gradient,
                &mut edge_kernel_gradient,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
                groups,
            )
        }
        .map_err(launch_err)?;
        let mut edge_bias_gradient = self.alloc_uninit(edge_count)?;
        unsafe {
            self.graph_conv.graph_learned_bias_bwd(
                &self.stream,
                launch::elementwise(edge_count, config),
                &logits_gradient,
                &mut edge_bias_gradient,
                g.batch,
                groups,
                g.output_height,
                g.output_width,
                kernel_area,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_gradient, &input_shape),
            OxideTensorF32::new(edge_kernel_gradient, &[kernel_len]),
            OxideTensorF32::new(edge_bias_gradient, &[edge_count]),
        ))
    }

    pub(crate) fn graph_conv_aggregate_backward_native(
        &self,
        output_gradient: &OxideTensorF32,
        geometry: GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "graph_conv_aggregate_backward_f32";
        let g = geometry.checked().map_err(OxideError::Shape)?;
        let output_shape = [g.batch, g.channels, g.output_height, g.output_width];
        let input_shape = [g.batch, g.channels, g.input_height, g.input_width];
        expect_shape(
            OP,
            "output gradient",
            output_gradient.shape(),
            &output_shape,
        )?;
        expect_len(OP, "output gradient", output_gradient.len(), g.output_len())?;
        let mut input_gradient = self.alloc_uninit(g.input_len())?;
        // SAFETY: shape checks above establish the buffer extents; every
        // in-range thread owns exactly one input-gradient element.
        unsafe {
            self.graph_conv.graph_aggregate_bwd(
                &self.stream,
                launch::elementwise(g.input_len(), config),
                &output_gradient.buf,
                &mut input_gradient,
                g.batch,
                g.channels,
                g.input_height,
                g.input_width,
                g.output_height,
                g.output_width,
                g.size,
                g.stride_x,
                g.stride_y,
                g.dilation,
                g.padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_gradient, &input_shape))
    }
}
