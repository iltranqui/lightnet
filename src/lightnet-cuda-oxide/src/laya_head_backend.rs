//! Laya head backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn laya_head_prepare_backend_f32(
        &self,
        input: &OxideTensorF32,
        input_shape: &[usize],
        type_embedding: &[f32],
        hidden: usize,
        qtype: i32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "laya_head_prepare_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] != 1
            || input_shape[3] == 0
            || hidden == 0
            || !((0..=2).contains(&qtype) || qtype == -1)
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}, hidden={hidden}, qtype={qtype}"),
            ));
        }
        let (batch, channels, sequence) = (input_shape[0], input_shape[1], input_shape[3]);
        let expected_channels = hidden
            .checked_add(if qtype == -1 { 2 } else { 1 })
            .ok_or_else(|| shape_err(op, "channel count overflows".to_owned()))?;
        if channels != expected_channels || type_embedding.len() != 3 * hidden {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_channels} input channels and {} type embedding values",
                    3 * hidden
                ),
            ));
        }
        let input = self.f32v(input)?;
        if input.shape() != input_shape {
            return Err(shape_err(
                op,
                format!(
                    "input tensor shape {:?} does not match {input_shape:?}",
                    input.shape()
                ),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(sequence))
            .ok_or_else(|| shape_err(op, "input length overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(sequence)
            .and_then(|n| n.checked_mul(hidden))
            .ok_or_else(|| shape_err(op, "token output length overflows".to_owned()))?;
        let mask_len = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "mask length overflows".to_owned()))?;
        if input.len() != input_len {
            return Err(shape_err(
                op,
                format!("input has {} values, expected {input_len}", input.len()),
            ));
        }
        let type_embedding = self.upload_f32(&[3, hidden], type_embedding)?;
        let mut tokens = self.alloc_uninit(output_len)?;
        let mut mask = self.alloc_uninit(mask_len)?;
        // SAFETY: validated NCHW extents bound each input/type-embedding read;
        // each thread writes a distinct token feature and h=0 writes one mask.
        unsafe {
            self.laya.laya_head_prepare(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &type_embedding.buf,
                &mut tokens,
                &mut mask,
                hidden,
                channels,
                sequence,
                qtype,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(tokens, &[batch * sequence, hidden]),
            OxideTensorF32::new(mask, &[batch, sequence]),
        ))
    }

    pub(crate) fn laya_head_pack_backend_f32(
        &self,
        tokens: &OxideTensorF32,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_head_pack_f32";
        let expected_len = batch
            .checked_mul(sequence)
            .and_then(|n| n.checked_mul(hidden))
            .ok_or_else(|| shape_err(op, "tensor length overflows".to_owned()))?;
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
        if tokens.shape() != [batch * sequence, hidden] || tokens.len() != expected_len {
            return Err(shape_err(
                op,
                format!(
                    "token tensor has {} values, expected {expected_len}",
                    tokens.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(expected_len)?;
        // SAFETY: checked equal lengths make the channel-major index map valid.
        unsafe {
            self.laya.laya_head_pack(
                &self.stream,
                launch::elementwise(expected_len, config),
                &tokens.buf,
                &mut output,
                sequence,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn laya_head_attention_backend_f32(
        &self,
        qkv: &OxideTensorF32,
        mask: &OxideTensorF32,
        output_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_head_attention_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        if output_shape != [rows, channels] {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} must be [batch*tokens, channels]"),
            ));
        }
        let output_len = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let qkv_f = self.f32v(qkv)?;
        let mask_f = self.f32v(mask)?;
        if qkv_f.shape() != [rows, 3 * channels]
            || mask_f.shape() != [batch, tokens]
            || qkv_f.len() != qkv_len
            || mask_f.len() != rows
        {
            return Err(shape_err(
                op,
                format!(
                    "QKV/mask lengths are {}/{}, expected {qkv_len}/{rows}",
                    qkv_f.len(),
                    mask_f.len()
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
        let query_count = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "query count overflows usize".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: checked dimensions bound every Q/K/V/mask access, and one
        // query/head thread exclusively writes its output channel interval.
        unsafe {
            self.laya.laya_head_attention_forward(
                &self.stream,
                launch::elementwise(query_count, config),
                &qkv_f.buf,
                &mask_f.buf,
                &mut output,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }
}
