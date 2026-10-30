//! Flash attention backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn vit_msa_forward_backend_f32(&self, qkv: &OxideTensorF32,
        batch: usize, tokens: usize, channels: usize, heads: usize, block: usize, topk: usize,
        config: ElementwiseLaunchConfig) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "vit_msa_forward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 || block == 0 || topk == 0 {
            return Err(shape_err(op, "invalid MSA geometry".into()));
        }
        let features = batch.checked_mul(tokens).and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| shape_err(op, "feature size overflow".into()))?;
        let qkv_len = features.checked_mul(3).ok_or_else(|| shape_err(op, "QKV size overflow".into()))?;
        let scores_len = batch.checked_mul(heads).and_then(|n| n.checked_mul(tokens))
            .and_then(|n| n.checked_mul(tokens)).ok_or_else(|| shape_err(op, "score size overflow".into()))?;
        let block = block.min(tokens);
        let blocks = tokens.div_ceil(block);
        let block_len = (scores_len / tokens).checked_mul(blocks)
            .ok_or_else(|| shape_err(op, "block size overflow".into()))?;
        let qkv = self.f32v(qkv)?;
        if qkv.len() != qkv_len { return Err(shape_err(op, "QKV length mismatch".into())); }
        let mut scores = self.alloc_uninit(scores_len)?;
        let mut active = self.alloc_uninit(scores_len)?;
        let mut probabilities = self.alloc_uninit(scores_len)?;
        let mut maxima = self.alloc_uninit(block_len)?;
        let mut selected = self.alloc_uninit(block_len)?;
        let mut context = self.alloc_uninit(features)?;
        // SAFETY: checked lengths and one writer per output element; launches
        // are ordered on the same stream.
        unsafe { self.attention.vit_msa_scores(&self.stream, launch::elementwise(scores_len, config),
            &qkv.buf, &mut scores, &mut active, tokens, channels, heads) }.map_err(launch_err)?;
        // Scores already have ViT's finite [-10, 10] normalization, so the
        // Tucker block selector and sparse softmax preserve ViT's tie order.
        unsafe { self.tucker_attention.tucker_block_maxima(&self.stream, launch::elementwise(block_len, config),
            &scores, &mut maxima, tokens, block, blocks) }.map_err(launch_err)?;
        unsafe { self.tucker_attention.tucker_select_blocks(&self.stream, launch::elementwise(block_len, config),
            &maxima, &mut selected, tokens, block, blocks, topk.min(blocks)) }.map_err(launch_err)?;
        unsafe { self.tucker_attention.tucker_probabilities(&self.stream, launch::elementwise(scores_len, config),
            &scores, &selected, &mut probabilities, tokens, true, block, blocks) }.map_err(launch_err)?;
        unsafe { self.attention.vit_msa_context(&self.stream, launch::elementwise(features, config),
            &qkv.buf, &probabilities, &mut context, tokens, channels, heads) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(context, &[batch, tokens, channels]),
            OxideTensorF32::new(probabilities, &[batch, heads, tokens, tokens]),
            OxideTensorF32::new(active, &[batch, heads, tokens, tokens])))
    }

    pub(crate) fn vit_msa_backward_backend_f32(&self, qkv: &OxideTensorF32,
        probabilities: &OxideTensorF32, active: &OxideTensorF32, context: &OxideTensorF32,
        grad: &OxideTensorF32, batch: usize, tokens: usize, channels: usize, heads: usize,
        config: ElementwiseLaunchConfig) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_msa_backward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(op, "invalid MSA geometry".into()));
        }
        let features = batch.checked_mul(tokens).and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| shape_err(op, "feature size overflow".into()))?;
        let qkv_len = features.checked_mul(3).ok_or_else(|| shape_err(op, "QKV size overflow".into()))?;
        let scores = batch.checked_mul(heads).and_then(|n| n.checked_mul(tokens))
            .and_then(|n| n.checked_mul(tokens)).ok_or_else(|| shape_err(op, "score size overflow".into()))?;
        let qkv = self.f32v(qkv)?; let p = self.f32v(probabilities)?;
        let active = self.f32v(active)?; let context = self.f32v(context)?; let grad = self.f32v(grad)?;
        if qkv.len() != qkv_len || p.len() != scores || active.len() != scores
            || context.len() != features || grad.len() != features {
            return Err(shape_err(op, "operand length mismatch".into()));
        }
        let mut result = self.alloc_uninit(qkv_len)?;
        // SAFETY: validated shapes and each thread writes one fused QKV gradient.
        unsafe { self.attention.vit_msa_backward(&self.stream, launch::elementwise(qkv_len, config),
            &qkv.buf, &p.buf, &active.buf, &context.buf, &grad.buf, &mut result,
            tokens, channels, heads) }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(result, &[batch, tokens, 3 * channels]))
    }

    pub(crate) fn flash_attention_forward_backend_f32(
        &self,
        qkv: &OxideTensorF32,
        output_shape: &[usize],
        lse_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "flash_attention_forward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let lse_len = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "LSE length overflows usize".to_owned()))?;
        if output_shape != [batch, tokens, channels] {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} must be [batch, tokens, channels]"),
            ));
        }
        if lse_shape != [batch, heads, tokens] {
            return Err(shape_err(
                op,
                format!("LSE shape {lse_shape:?} must be [batch, heads, tokens]"),
            ));
        }
        let qkv_f = self.f32v(qkv)?;
        let qkv = &*qkv_f;
        if qkv.len() != qkv_len {
            return Err(shape_err(
                op,
                format!("QKV has {} values, expected {qkv_len}", qkv.len()),
            ));
        }
        let head_dim = channels / heads;
        if head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} exceeds the kernel limit 128"),
            ));
        }
        let mut out = self.alloc_uninit(output_len)?;
        let mut lse = self.alloc_uninit(lse_len)?;
        // SAFETY: one query thread writes its unique head-channel interval in
        // `out` and its unique flattened query index in `lse`.
        unsafe {
            self.attention.flash_attention_forward(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv.buf,
                &mut out,
                &mut lse,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(out, output_shape),
            OxideTensorF32::new(lse, lse_shape),
        ))
    }

    pub(crate) fn flash_attention_backward_backend_f32(
        &self,
        qkv: &OxideTensorF32,
        context: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        lse: &OxideTensorF32,
        qkv_grad_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "flash_attention_backward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "attention size overflows usize".to_owned()))?;
        let qkv_len = elements
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let lse_len = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "LSE length overflows usize".to_owned()))?;
        if qkv_grad_shape != [batch, tokens, 3 * channels] {
            return Err(shape_err(
                op,
                format!("dQKV shape {qkv_grad_shape:?} must be [batch, tokens, 3*channels]"),
            ));
        }
        let qkv_f = self.f32v(qkv)?;
        let context_f = self.f32v(context)?;
        let grad_f = self.f32v(output_grad)?;
        let lse_f = self.f32v(lse)?;
        if qkv_f.len() != qkv_len
            || context_f.len() != elements
            || grad_f.len() != elements
            || lse_f.len() != lse_len
        {
            return Err(shape_err(
                op,
                format!(
                    "QKV/context/gradient/LSE lengths are {}/{}/{}/{}, expected {qkv_len}/{elements}/{elements}/{lse_len}",
                    qkv_f.len(),
                    context_f.len(),
                    grad_f.len(),
                    lse_f.len()
                ),
            ));
        }
        let head_dim = channels / heads;
        if head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} exceeds the kernel limit 128"),
            ));
        }
        let mut dqkv = self.alloc_uninit(qkv_len)?;
        let mut row_dot = self.alloc_uninit(lse_len)?;
        // SAFETY: each query/head thread writes its Q row and D_i scalar.
        unsafe {
            self.attention.flash_attention_backward_dq(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv_f.buf,
                &context_f.buf,
                &grad_f.buf,
                &lse_f.buf,
                &mut dqkv,
                &mut row_dot,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: each key/head thread writes its K and V rows, disjoint from dQ
        // and from every other key/head row in the fused gradient buffer.
        unsafe {
            self.attention.flash_attention_backward_dkdv(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv_f.buf,
                &grad_f.buf,
                &lse_f.buf,
                &row_dot,
                &mut dqkv,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dqkv, qkv_grad_shape))
    }

}
