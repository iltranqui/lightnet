//! Native Swin backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn swin_backend_window_partition_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.swin_window_partition_native(
            input,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
            config,
        )
    }

    pub(crate) fn swin_backend_window_unpartition_f32(
        &self,
        windows: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.swin_window_unpartition_native(
            windows,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
            clip,
            config,
        )
    }

    pub(crate) fn swin_backend_mhc_residual_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.swin_mhc_residual_native(skip, branch, skip_scale, branch_scale, config)
    }

    pub(crate) fn swin_backend_mhc_residual_backward_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        grad: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.swin_mhc_residual_backward_native(
            skip,
            branch,
            grad,
            skip_scale,
            branch_scale,
            mix,
            post_skip,
            post_branch,
            config,
        )
    }

    pub(crate) fn swin_backend_window_attention_f32(
        &self,
        qkv: &OxideTensorF32,
        relative_position_bias: &OxideTensorF32,
        mask: &OxideTensorF32,
        total_windows: usize,
        spatial_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.swin_window_attention_native(
            qkv,
            relative_position_bias,
            mask,
            total_windows,
            spatial_windows,
            window,
            channels,
            heads,
            config,
        )
    }

    pub(crate) fn swin_backend_window_attention_backward_f32(
        &self,
        qkv: &OxideTensorF32,
        probabilities: &OxideTensorF32,
        active: &OxideTensorF32,
        context_grad: &OxideTensorF32,
        total_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        self.swin_window_attention_backward_native(
            qkv,
            probabilities,
            active,
            context_grad,
            total_windows,
            window,
            channels,
            heads,
            config,
        )
    }
}
