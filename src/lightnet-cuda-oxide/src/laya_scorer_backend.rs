//! Laya scorer backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn laya_scorer_unpack_backend_f32(
        &self,
        input: &OxideTensorF32,
        input_shape: &[usize],
        hidden: usize,
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_scorer_unpack_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] != hidden
            || input_shape[2] != 1
            || input_shape[3] != options
            || hidden == 0
            || options == 0
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}"),
            ));
        }
        let batch = input_shape[0];
        let len = batch
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(options))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let rows = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != input_shape || input.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length mismatch: {:?}/{}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: exact tensor extents bound the channel-major gather indices.
        unsafe {
            self.laya.laya_scorer_unpack(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                hidden,
                options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[rows, hidden]))
    }

    pub(crate) fn laya_scorer_pack_backend_f32(
        &self,
        logits: &OxideTensorF32,
        batch: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_scorer_pack_f32";
        let len = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if batch == 0 || options == 0 || output_shape != [batch, 1, 1, options] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let logits = self.f32v(logits)?;
        if logits.shape() != [len, 1] || logits.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "logit tensor shape/length mismatch: {:?}/{}",
                    logits.shape(),
                    logits.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: logits/output lengths are equal and every thread owns one slot.
        unsafe {
            self.laya.laya_scorer_pack(
                &self.stream,
                launch::elementwise(len, config),
                &logits.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn laya_scorer_seed_rows_backend_f32(
        &self,
        seed: &OxideTensorF32,
        batch: usize,
        options: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_scorer_seed_rows_f32";
        let len = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "seed length overflows usize".to_owned()))?;
        if batch == 0 || options == 0 {
            return Err(shape_err(op, "batch/options must be positive".to_owned()));
        }
        let seed = self.f32v(seed)?;
        if seed.shape() != [batch, 1, 1, options] || seed.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "seed tensor shape/length mismatch: {:?}/{}",
                    seed.shape(),
                    seed.len()
                ),
            ));
        }
        Ok(OxideTensorF32 {
            buf: seed.buf.clone(),
            shape: vec![len, 1],
            half: None,
        })
    }

    pub(crate) fn laya_scorer_pack_input_grad_backend_f32(
        &self,
        rows: &OxideTensorF32,
        batch: usize,
        hidden: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "laya_scorer_pack_input_grad_f32";
        let len = batch
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(options))
            .ok_or_else(|| shape_err(op, "gradient length overflows usize".to_owned()))?;
        let rows_count = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        if batch == 0 || hidden == 0 || options == 0 || output_shape != [batch, hidden, 1, options]
        {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let rows = self.f32v(rows)?;
        if rows.shape() != [rows_count, hidden] || rows.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "row tensor shape/length mismatch: {:?}/{}",
                    rows.shape(),
                    rows.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: equal extents bound all row/channel/option transpose indices.
        unsafe {
            self.laya.laya_scorer_pack_input_grad(
                &self.stream,
                launch::elementwise(len, config),
                &rows.buf,
                &mut output,
                hidden,
                options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

}
