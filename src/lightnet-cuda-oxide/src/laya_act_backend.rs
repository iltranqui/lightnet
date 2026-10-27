//! Laya act-head backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn laya_act_head_backward_backend_f32(
        &self,
        inputs: &lightnet_cuda::LayaActBackwardInputs<'_, OxideTensorF32>,
        layout: &lightnet_cuda::host_ops::laya_act::MlpLayout,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::LayaActHeadGrad<OxideTensorF32>, OxideError> {
        let op = "laya_act_head_backward_f32";
        let d = layout.features();
        if inputs.head.len() != d.head_len
            || inputs.scorer.len() != d.logits_len
            || inputs.markers.len() != d.marker_len
            || inputs.fc1_w.len() != layout.fc1_len()
            || inputs.fc1_b.len() != layout.fc_dim()
            || inputs.fc2_w.len() != layout.fc2_len()
            || inputs.seeds.len() != layout.seed_len()
        {
            return Err(shape_err(
                op,
                "act-head backward operand storage mismatch".to_owned(),
            ));
        }
        if inputs.fc1_w.shape() != [layout.fc_dim(), layout.inputs()]
            || inputs.fc1_b.shape() != [layout.fc_dim()]
            || inputs.fc2_w.shape() != [layout.n_act(), layout.fc_dim()]
            || (inputs.seeds.shape() != [d.batch, layout.n_act()]
                && inputs.seeds.shape() != [d.batch, layout.n_act(), 1, 1])
        {
            return Err(shape_err(
                op,
                "act-head backward matrix/seed metadata mismatch".to_owned(),
            ));
        }
        // Shared composition dispatches back into native Oxide overrides;
        // all sources are checked before any math, and no data is downloaded.
        lightnet_cuda::laya_act::backward_composed(self, inputs, layout, config)
    }

    pub(crate) fn laya_act_cls_scatter_backend_f32(
        &self,
        dcls: &OxideTensorF32,
        head_grad: &OxideTensorF32,
        batch: usize,
        hidden: usize,
        tokens: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_act_cls_scatter_f32";
        let cls_len = batch
            .checked_mul(hidden)
            .filter(|&len| batch > 0 && hidden > 0 && len <= i32::MAX as usize)
            .ok_or_else(|| shape_err(op, "invalid batch/hidden dimensions".to_owned()))?;
        let full_len = cls_len
            .checked_mul(tokens)
            .filter(|&len| tokens > 0 && len <= i32::MAX as usize)
            .ok_or_else(|| shape_err(op, "invalid token count/storage".to_owned()))?;
        if dcls.len() != cls_len || head_grad.len() != full_len {
            return Err(shape_err(
                op,
                format!(
                    "storage mismatch: expected dcls={cls_len}, head_grad={full_len}; got dcls={}, head_grad={}",
                    dcls.len(),
                    head_grad.len()
                ),
            ));
        }
        if (dcls.shape() != [batch, hidden] && dcls.shape() != [batch, hidden, 1, 1])
            || head_grad.shape() != [batch, hidden, 1, tokens]
        {
            return Err(shape_err(
                op,
                "expected dcls [B,H] and head gradient [B,H,1,T]".to_owned(),
            ));
        }
        let dcls_f = self.f32v(dcls)?;
        let head_grad_f = self.f32v(head_grad)?;
        let mut output = self.alloc_uninit(full_len)?;
        // SAFETY: dimensions/storage are checked, and each thread writes one
        // unique output element while reading only in-bounds source values.
        unsafe {
            self.laya.laya_act_cls_scatter(
                &self.stream,
                launch::elementwise(full_len, config),
                &dcls_f.buf,
                &head_grad_f.buf,
                &mut output,
                tokens,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, hidden, 1, tokens]))
    }

    pub(crate) fn laya_act_features_backend_f32(
        &self,
        head: &OxideTensorF32,
        scorer: &OxideTensorF32,
        markers: &OxideTensorF32,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_act_features_f32";
        let d =
            lightnet_cuda::host_ops::laya_act::Layout::new(head_shape, scorer_shape, marker_shape)
                .map_err(|err| shape_err(op, err))?;
        if head.len() != d.head_len || scorer.len() != d.logits_len || markers.len() != d.marker_len
        {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let head_f = self.f32v(head)?;
        let scorer_f = self.f32v(scorer)?;
        let markers_f = self.f32v(markers)?;
        let mut output = self.alloc_uninit(d.output_len)?;
        // SAFETY: positive checked shapes/products bound all source reads;
        // each sample thread owns an entire disjoint output row.
        unsafe {
            self.laya.laya_act_features(
                &self.stream,
                launch::elementwise(d.batch, config),
                &head_f.buf,
                &scorer_f.buf,
                &markers_f.buf,
                &mut output,
                d.hidden,
                d.options,
                d.marker_channels,
                d.head_width,
                d.marker_width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &d.output_shape()))
    }
}
