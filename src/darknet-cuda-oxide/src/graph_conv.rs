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
