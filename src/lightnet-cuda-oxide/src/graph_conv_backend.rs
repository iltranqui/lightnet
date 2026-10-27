//! Graph convolution backend method delegates.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn graph_conv_backend_aggregate_f32(
        &self,
        input: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.graph_conv_aggregate_forward_native(input, geometry, config)
    }

    pub(crate) fn graph_conv_backend_reference_f32(
        &self,
        input: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.graph_conv_reference_forward_native(input, geometry, config)
    }

    pub(crate) fn graph_conv_backend_reference_backward_f32(
        &self,
        output_gradient: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.graph_conv_reference_backward_native(output_gradient, geometry, config)
    }

    pub(crate) fn graph_conv_backend_learned_aggregate_f32(
        &self,
        input: &OxideTensorF32,
        edge_kernel: &OxideTensorF32,
        edge_biases: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.graph_conv_learned_aggregate_forward_native(
            input,
            edge_kernel,
            edge_biases,
            geometry,
            groups,
            config,
        )
    }

    pub(crate) fn graph_conv_backend_learned_aggregate_backward_f32(
        &self,
        input: &OxideTensorF32,
        edge_kernel: &OxideTensorF32,
        edge_biases: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.graph_conv_learned_aggregate_backward_native(
            input,
            edge_kernel,
            edge_biases,
            output_gradient,
            geometry,
            groups,
            config,
        )
    }

    pub(crate) fn graph_conv_backend_aggregate_backward_f32(
        &self,
        output_gradient: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.graph_conv_aggregate_backward_native(output_gradient, geometry, config)
    }
}
