//! Native window-local Tucker attention; projections use the connected ops.
use lightnet_cuda::{ElementwiseLaunchConfig, host_ops::tucker_attention::AttentionGeometry};
use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

impl OxideBackend {
    pub(crate) fn tucker_attention_native(
        &self, q: &OxideTensorF32, k: &OxideTensorF32, v: &OxideTensorF32,
        g: AttentionGeometry, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let (features, scores_len) = g.lengths().map_err(OxideError::Shape)?;
        let q = self.f32v(q)?;
        let k = self.f32v(k)?;
        let v = self.f32v(v)?;
        if q.len() != features || k.len() != features || v.len() != features {
            return Err(OxideError::Shape("Tucker attention: Q/K/V length mismatch".into()));
        }
        let (sparse, block, topk) = g.sparse();
        let blocks = g.tokens.div_ceil(block);
        let block_len = scores_len / g.tokens * blocks;
        let mut scores = self.alloc_uninit(scores_len)?;
        let mut p = self.alloc_uninit(scores_len)?;
        let mut active = self.alloc_uninit(scores_len)?;
        let mut maxima = self.alloc_uninit(if sparse { block_len } else { 1 })?;
        let mut selected = self.alloc_uninit(if sparse { block_len } else { 1 })?;
        let mut context = self.alloc_uninit(features)?;
        // SAFETY: geometry and every operand length are validated above. Each
        // output element has one writer; launches are ordered on this stream.
        unsafe {
            self.tucker_attention.tucker_scores(&self.stream, launch::elementwise(scores_len, config),
                &q.buf, &k.buf, &mut scores, g.tokens, g.channels, g.heads)
        }.map_err(launch_err)?;
        if sparse {
            unsafe {
                self.tucker_attention.tucker_block_maxima(&self.stream, launch::elementwise(block_len, config),
                    &scores, &mut maxima, g.tokens, block, blocks)
            }.map_err(launch_err)?;
            unsafe {
                self.tucker_attention.tucker_select_blocks(&self.stream, launch::elementwise(block_len, config),
                    &maxima, &mut selected, g.tokens, block, blocks, topk)
            }.map_err(launch_err)?;
        }
        unsafe {
            self.tucker_attention.tucker_probabilities(&self.stream, launch::elementwise(scores_len, config),
                &scores, &selected, &mut p, g.tokens, sparse, block, blocks)
        }.map_err(launch_err)?;
        unsafe {
            self.tucker_attention.tucker_score_activity(&self.stream, launch::elementwise(scores_len, config),
                &scores, &mut active)
        }.map_err(launch_err)?;
        unsafe {
            self.tucker_attention.tucker_context(&self.stream, launch::elementwise(features, config),
                &v.buf, &p, &mut context, g.tokens, g.channels, g.heads)
        }.map_err(launch_err)?;
        let score_shape = [g.windows, g.heads, g.tokens, g.tokens];
        Ok((OxideTensorF32::new(context, &[g.windows * g.tokens, g.channels]),
            OxideTensorF32::new(p, &score_shape), OxideTensorF32::new(active, &score_shape)))
    }

    pub(crate) fn tucker_attention_backward_native(
        &self, q: &OxideTensorF32, k: &OxideTensorF32, v: &OxideTensorF32,
        p: &OxideTensorF32, active: &OxideTensorF32, grad: &OxideTensorF32,
        g: AttentionGeometry, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let (features, scores_len) = g.lengths().map_err(OxideError::Shape)?;
        let q = self.f32v(q)?;
        let k = self.f32v(k)?;
        let v = self.f32v(v)?;
        let p = self.f32v(p)?;
        let active = self.f32v(active)?;
        let grad = self.f32v(grad)?;
        if q.len() != features || k.len() != features || v.len() != features || grad.len() != features
            || p.len() != scores_len || active.len() != scores_len
        {
            return Err(OxideError::Shape("Tucker attention backward: operand length mismatch".into()));
        }
        let mut dp = self.alloc_uninit(scores_len)?;
        let mut ds = self.alloc_uninit(scores_len)?;
        let mut dq = self.alloc_uninit(features)?;
        let mut dk = self.alloc_uninit(features)?;
        let mut dv = self.alloc_uninit(features)?;
        // SAFETY: all lengths match the checked geometry; gather reductions
        // assign one writer to each gradient element without atomics.
        unsafe {
            self.tucker_attention.tucker_probability_grad(&self.stream, launch::elementwise(scores_len, config),
                &v.buf, &grad.buf, &mut dp, g.tokens, g.channels, g.heads)
        }.map_err(launch_err)?;
        unsafe {
            self.tucker_attention.tucker_score_grad(&self.stream, launch::elementwise(scores_len, config),
                &dp, &p.buf, &active.buf, &mut ds, g.tokens, g.sparse().0)
        }.map_err(launch_err)?;
        for (kind, output) in [(0, &mut dq), (1, &mut dk), (2, &mut dv)] {
            unsafe {
                self.tucker_attention.tucker_qkv_grad(&self.stream, launch::elementwise(features, config),
                    &q.buf, &k.buf, &grad.buf, &p.buf, &ds, output, g.tokens, g.channels, g.heads, kind)
            }.map_err(launch_err)?;
        }
        let shape = [g.windows * g.tokens, g.channels];
        Ok((OxideTensorF32::new(dq, &shape), OxideTensorF32::new(dk, &shape),
            OxideTensorF32::new(dv, &shape)))
    }
}
