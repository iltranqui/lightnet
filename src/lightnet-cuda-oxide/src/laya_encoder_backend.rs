//! Laya encoder backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn laya_encoder_embed_backend_f32(
        &self,
        input: &OxideTensorF32,
        embeddings: &OxideTensorF32,
        input_shape: &[usize],
        hidden: usize,
        vocab: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "laya_encoder_embed_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] != 2
            || input_shape[2] != 1
            || input_shape[3] == 0
            || hidden == 0
            || vocab == 0
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}"),
            ));
        }
        let batch = input_shape[0];
        let sequence = input_shape[3];
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "embedding length overflows usize".to_owned()))?;
        let input_len = batch
            .checked_mul(2)
            .and_then(|n| n.checked_mul(sequence))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let input = self.f32v(input)?;
        let embeddings = self.f32v(embeddings)?;
        if input.shape() != input_shape
            || input.len() != input_len
            || embeddings.shape() != [vocab, hidden]
            || embeddings.len() != vocab * hidden
        {
            return Err(shape_err(
                op,
                "input/embedding extents do not match".to_owned(),
            ));
        }
        let mut tokens = self.alloc_uninit(len)?;
        let mut mask = self.alloc_uninit(rows)?;
        // SAFETY: exact checked lengths bound each clamped id and mask read.
        unsafe {
            self.laya.laya_encoder_embed(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &embeddings.buf,
                &mut tokens,
                &mut mask,
                hidden,
                sequence,
                vocab,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(tokens, &[rows, hidden]),
            OxideTensorF32::new(mask, &[batch, sequence]),
        ))
    }

    pub(crate) fn laya_encoder_attention_backend_f32(
        &self,
        qkv: &OxideTensorF32,
        mask: &OxideTensorF32,
        cos: &OxideTensorF32,
        sin: &OxideTensorF32,
        batch: usize,
        sequence: usize,
        hidden: usize,
        heads: usize,
        sliding: bool,
        window_radius: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_encoder_attention_f32";
        if batch == 0 || sequence == 0 || hidden == 0 || heads == 0 || hidden % heads != 0 {
            return Err(shape_err(
                op,
                "invalid batch/sequence/hidden/head dimensions".to_owned(),
            ));
        }
        let head_dim = hidden / heads;
        if head_dim % 2 != 0 || head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} must be even and <=128"),
            ));
        }
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if output_shape != [rows, hidden] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let mask_len = rows;
        let table_len = sequence
            .checked_mul(head_dim / 2)
            .ok_or_else(|| shape_err(op, "RoPE table length overflows usize".to_owned()))?;
        let qkv = self.f32v(qkv)?;
        let mask = self.f32v(mask)?;
        let cos = self.f32v(cos)?;
        let sin = self.f32v(sin)?;
        if qkv.shape() != [rows, 3 * hidden]
            || qkv.len() != qkv_len
            || mask.shape() != [batch, sequence]
            || mask.len() != mask_len
            || cos.shape() != [sequence, head_dim / 2]
            || cos.len() != table_len
            || sin.shape() != [sequence, head_dim / 2]
            || sin.len() != table_len
        {
            return Err(shape_err(
                op,
                "QKV/mask/RoPE extents do not match".to_owned(),
            ));
        }
        let query_heads = rows
            .checked_mul(heads)
            .ok_or_else(|| shape_err(op, "query count overflows usize".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: validated dimensions bound every QKV, mask, RoPE and output access.
        unsafe {
            self.laya.laya_encoder_attention(
                &self.stream,
                launch::elementwise(query_heads, config),
                &qkv.buf,
                &mask.buf,
                &cos.buf,
                &sin.buf,
                &mut output,
                sequence,
                hidden,
                heads,
                window_radius,
                sliding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn laya_encoder_geglu_backend_f32(
        &self,
        input: &OxideTensorF32,
        rows: usize,
        intermediate: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_encoder_geglu_f32";
        let input_len = rows
            .checked_mul(intermediate)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(intermediate)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if rows == 0 || intermediate == 0 || output_shape != [rows, intermediate] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let input = self.f32v(input)?;
        if input.shape() != [rows, 2 * intermediate] || input.len() != input_len {
            return Err(shape_err(op, "GeGLU input extent mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: checked 2I input and I output rows bound every chunk access.
        unsafe {
            self.laya.laya_encoder_geglu(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                intermediate,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn laya_encoder_pack_backend_f32(
        &self,
        tokens: &OxideTensorF32,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_encoder_pack_f32";
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if batch == 0
            || sequence == 0
            || hidden == 0
            || output_shape != [batch, hidden, 1, sequence]
        {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let tokens = self.f32v(tokens)?;
        if tokens.shape() != [rows, hidden] || tokens.len() != len {
            return Err(shape_err(op, "token tensor extent mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: equal checked extents bound the token/channel-major permutation.
        unsafe {
            self.laya.laya_encoder_pack(
                &self.stream,
                launch::elementwise(len, config),
                &tokens.buf,
                &mut output,
                sequence,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

}
