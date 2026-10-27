//! WMHF backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    /// Native f32 WMHF four-direction scan backward and parameter gradients.
    pub(crate) fn wmhf_backend_scan4_backward_f32(
        &self,
        input: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        weights: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "wmhf_scan4_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let total = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "tensor element count overflows".to_owned()))?;
        let direction_len = total
            .checked_mul(4)
            .ok_or_else(|| shape_err(op, "direction scratch size overflows".to_owned()))?;
        let sequences_per_channel = height
            .checked_mul(2)
            .and_then(|value| value.checked_add(width.checked_mul(2)?))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let sequence_count = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(sequences_per_channel))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let output_grad = self.f32v(output_grad)?;
        let weights = self.f32v(weights)?;
        if input.shape() != [batch, channels, height, width]
            || input.len() != total
            || output_grad.shape() != [batch, channels, height, width]
            || output_grad.len() != total
        {
            return Err(shape_err(
                op,
                "input/output-gradient shape mismatch".to_owned(),
            ));
        }
        if weights.shape() != [4, channels] || weights.len() != 4 * channels {
            return Err(shape_err(op, "scan-weight shape mismatch".to_owned()));
        }
        let states = self.alloc_uninit(direction_len)?;
        let input_parts = self.alloc_uninit(direction_len)?;
        let da_parts = self.alloc_uninit(direction_len)?;
        let db_parts = self.alloc_uninit(direction_len)?;
        let dc_parts = self.alloc_uninit(direction_len)?;
        let dd_parts = self.alloc_uninit(direction_len)?;
        let mut input_grad = self.alloc_uninit(total)?;
        let mut weight_grad = self.alloc_uninit(4 * channels)?;
        // SAFETY: one row/column sequence owns each state and partial entry in
        // its direction plane. Follow-up gathers reduce four-way overlaps and
        // channel parameter contributions.
        unsafe {
            self.wmhf.wmhf_scan4_backward_states(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &weights.buf,
                states.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_sequences(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &output_grad.buf,
                &weights.buf,
                &states,
                input_parts.cu_deviceptr() as *mut f32,
                da_parts.cu_deviceptr() as *mut f32,
                db_parts.cu_deviceptr() as *mut f32,
                dc_parts.cu_deviceptr() as *mut f32,
                dd_parts.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_input(
                &self.stream,
                launch::elementwise(total, config),
                &input_parts,
                &mut input_grad,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_weights(
                &self.stream,
                launch::elementwise(4 * channels, config),
                &da_parts,
                &db_parts,
                &dc_parts,
                &dd_parts,
                &mut weight_grad,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_grad, &[batch, channels, height, width]),
            OxideTensorF32::new(weight_grad, &[4, channels]),
        ))
    }

    /// Native f32 WMHF high-frequency magnitude/nearest-upsample branch.
    pub(crate) fn wmhf_backend_high_frequency_forward_f32(
        &self,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_high_frequency_forward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let expected_coeff_shape = [batch, channels, height2, width2];
        let lh = self.f32v(lh)?;
        let hl = self.f32v(hl)?;
        let hh = self.f32v(hh)?;
        for (name, tensor) in [("LH", &*lh), ("HL", &*hl), ("HH", &*hh)] {
            if tensor.shape() != expected_coeff_shape || tensor.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!(
                        "{name} shape/length {:?}/{}, expected {expected_coeff_shape:?}/{coeff_len}",
                        tensor.shape(),
                        tensor.len()
                    ),
                ));
            }
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: one thread writes each output pixel and reads its checked
        // nearest-upsampled coefficient cell.
        unsafe {
            self.wmhf.wmhf_high_frequency_forward(
                &self.stream,
                launch::elementwise(output_len, config),
                &lh.buf,
                &hl.buf,
                &hh.buf,
                &mut output,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, channels, height, width],
        ))
    }

    /// Native gather adjoint for the high-frequency magnitude/upsample branch.
    pub(crate) fn wmhf_backend_high_frequency_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "wmhf_high_frequency_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let coeff_shape = [batch, channels, height2, width2];
        let output_grad = self.f32v(output_grad)?;
        let lh = self.f32v(lh)?;
        let hl = self.f32v(hl)?;
        let hh = self.f32v(hh)?;
        if output_grad.shape() != [batch, channels, height, width]
            || output_grad.len() != output_len
        {
            return Err(shape_err(
                op,
                "output gradient shape/length mismatch".to_owned(),
            ));
        }
        for (name, tensor) in [("LH", &*lh), ("HL", &*hl), ("HH", &*hh)] {
            if tensor.shape() != coeff_shape || tensor.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!("{name} coefficient shape/length mismatch"),
                ));
            }
        }
        let mut lh_grad = self.alloc_uninit(coeff_len)?;
        let mut hl_grad = self.alloc_uninit(coeff_len)?;
        let mut hh_grad = self.alloc_uninit(coeff_len)?;
        // SAFETY: one thread owns each coefficient gradient and gathers all
        // high-resolution cells mapped to that coefficient.
        unsafe {
            self.wmhf.wmhf_high_frequency_backward(
                &self.stream,
                launch::elementwise(coeff_len, config),
                &output_grad.buf,
                &lh.buf,
                &hl.buf,
                &hh.buf,
                &mut lh_grad,
                &mut hl_grad,
                &mut hh_grad,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(lh_grad, &coeff_shape),
            OxideTensorF32::new(hl_grad, &coeff_shape),
            OxideTensorF32::new(hh_grad, &coeff_shape),
        ))
    }

    /// Gather a contiguous NCHW channel range for a WMHF branch.
    pub(crate) fn wmhf_backend_split_channels_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        begin: usize,
        count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_split_channels_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || count == 0 {
            return Err(shape_err(
                op,
                "dimensions and channel count must be positive".to_owned(),
            ));
        }
        let end = begin
            .checked_add(count)
            .ok_or_else(|| shape_err(op, "channel range overflows".to_owned()))?;
        if end > channels {
            return Err(shape_err(op, "channel range exceeds input".to_owned()));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(count)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != [batch, channels, height, width] || input.len() != input_len {
            return Err(shape_err(op, "input shape/length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: each output index gathers one valid source channel/spatial index.
        unsafe {
            self.wmhf.wmhf_split_channels(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                channels,
                height,
                width,
                begin,
                count,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, count, height, width]))
    }

    pub(crate) fn wmhf_backend_concat_local_f32(
        &self,
        branch3: &OxideTensorF32,
        branch5: &OxideTensorF32,
        branch7: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_concat_local_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let branch_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "branch element count overflows".to_owned()))?;
        let output_channels = channels
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "output channel count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(output_channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let expected_shape = [batch, channels, height, width];
        let branch3 = self.f32v(branch3)?;
        let branch5 = self.f32v(branch5)?;
        let branch7 = self.f32v(branch7)?;
        for (name, branch) in [("3x", &*branch3), ("5x", &*branch5), ("7x", &*branch7)] {
            if branch.shape() != expected_shape || branch.len() != branch_len {
                return Err(shape_err(
                    op,
                    format!("{name} branch shape/length mismatch"),
                ));
            }
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: each output position selects one valid element from one validated branch.
        unsafe {
            self.wmhf.wmhf_concat_local(
                &self.stream,
                launch::elementwise(output_len, config),
                &branch3.buf,
                &branch5.buf,
                &branch7.buf,
                &mut output,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, output_channels, height, width],
        ))
    }

    /// Native WMHF gate fusion: `fuse + freq_scale * gate * projected + shortcut`.
    pub(crate) fn wmhf_backend_apply_gate_f32(
        &self,
        fuse: &OxideTensorF32,
        gate: &OxideTensorF32,
        projected: &OxideTensorF32,
        shortcut: Option<&OxideTensorF32>,
        freq_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_apply_gate_f32";
        let fuse = self.f32v(fuse)?;
        let gate = self.f32v(gate)?;
        let projected = self.f32v(projected)?;
        let shortcut = shortcut.map(|tensor| self.f32v(tensor)).transpose()?;
        if fuse.is_empty()
            || gate.shape() != fuse.shape()
            || gate.len() != fuse.len()
            || projected.shape() != fuse.shape()
            || projected.len() != fuse.len()
        {
            return Err(shape_err(
                op,
                "fuse/gate/projected shape mismatch".to_owned(),
            ));
        }
        if let Some(shortcut) = &shortcut {
            if shortcut.shape() != fuse.shape() || shortcut.len() != fuse.len() {
                return Err(shape_err(op, "shortcut shape mismatch".to_owned()));
            }
        }
        let mut output = self.alloc_uninit(fuse.len())?;
        // SAFETY: each output element has one owner; all inputs have identical validated lengths.
        unsafe {
            self.wmhf.wmhf_apply_gate(
                &self.stream,
                launch::elementwise(fuse.len(), config),
                &fuse.buf,
                &gate.buf,
                &projected.buf,
                shortcut.as_ref().map_or(&fuse.buf, |value| &value.buf),
                &mut output,
                freq_scale,
                shortcut.is_some(),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, fuse.shape()))
    }

    /// Native exact adjoint of WMHF gate fusion.
    pub(crate) fn wmhf_backend_apply_gate_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        gate: &OxideTensorF32,
        projected: &OxideTensorF32,
        freq_scale: f32,
        use_shortcut: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "wmhf_apply_gate_backward_f32";
        let output_grad = self.f32v(output_grad)?;
        let gate = self.f32v(gate)?;
        let projected = self.f32v(projected)?;
        if output_grad.is_empty()
            || gate.shape() != output_grad.shape()
            || gate.len() != output_grad.len()
            || projected.shape() != output_grad.shape()
            || projected.len() != output_grad.len()
        {
            return Err(shape_err(
                op,
                "gradient/gate/projected shape mismatch".to_owned(),
            ));
        }
        let mut fuse_grad = self.alloc_uninit(output_grad.len())?;
        let mut gate_grad = self.alloc_uninit(output_grad.len())?;
        let mut projected_grad = self.alloc_uninit(output_grad.len())?;
        let mut shortcut_grad = self.alloc_uninit(output_grad.len())?;
        // SAFETY: each gradient element has one owner and all input extents match.
        unsafe {
            self.wmhf.wmhf_apply_gate_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &output_grad.buf,
                &gate.buf,
                &projected.buf,
                &mut fuse_grad,
                &mut gate_grad,
                &mut projected_grad,
                &mut shortcut_grad,
                freq_scale,
                use_shortcut,
            )
        }
        .map_err(launch_err)?;
        let shape = output_grad.shape();
        Ok((
            OxideTensorF32::new(fuse_grad, shape),
            OxideTensorF32::new(gate_grad, shape),
            OxideTensorF32::new(projected_grad, shape),
            OxideTensorF32::new(shortcut_grad, shape),
        ))
    }

    /// Native f32 WMHF four-direction EMA scan for NCHW tensors and `[4,C]` weights.
    pub(crate) fn wmhf_backend_scan4_forward_f32(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_scan4_forward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let total = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "tensor element count overflows".to_owned()))?;
        let direction_len = total
            .checked_mul(4)
            .ok_or_else(|| shape_err(op, "direction scratch size overflows".to_owned()))?;
        let sequences_per_channel = height
            .checked_mul(2)
            .and_then(|value| value.checked_add(width.checked_mul(2)?))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let sequence_count = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(sequences_per_channel))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let weights = self.f32v(weights)?;
        if input.shape() != [batch, channels, height, width] || input.len() != total {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected [{batch},{channels},{height},{width}]/{total}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        if weights.shape() != [4, channels] || weights.len() != 4 * channels {
            return Err(shape_err(
                op,
                format!(
                    "weight shape/length are {:?}/{}, expected [{},{}]/{}",
                    weights.shape(),
                    weights.len(),
                    4,
                    channels,
                    4 * channels
                ),
            ));
        }
        let directions = self.alloc_uninit(direction_len)?;
        let mut output = self.alloc_uninit(total)?;
        // SAFETY: each sequence has a unique (batch,channel,line,direction)
        // and writes every element of its disjoint direction plane exactly once.
        unsafe {
            self.wmhf.wmhf_scan4_forward(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &weights.buf,
                directions.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_combine(
                &self.stream,
                launch::elementwise(total, config),
                &directions,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, channels, height, width],
        ))
    }

    /// Native f32 WMHF Haar decomposition for `[B,C,H,W]` input.
    pub(crate) fn wmhf_backend_haar_dwt_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "wmhf_haar_dwt_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.len() != input_len || input.shape() != [batch, channels, height, width] {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected [{batch},{channels},{height},{width}]/{input_len}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut ll = self.alloc_uninit(output_len)?;
        let mut lh = self.alloc_uninit(output_len)?;
        let mut hl = self.alloc_uninit(output_len)?;
        let mut hh = self.alloc_uninit(output_len)?;
        // SAFETY: one thread writes the same valid coefficient index in each
        // sub-band; all input and output extents were checked above.
        unsafe {
            self.wmhf.wmhf_haar_dwt(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut ll,
                &mut lh,
                &mut hl,
                &mut hh,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        let output_shape = [batch, channels, height2, width2];
        Ok((
            OxideTensorF32::new(ll, &output_shape),
            OxideTensorF32::new(lh, &output_shape),
            OxideTensorF32::new(hl, &output_shape),
            OxideTensorF32::new(hh, &output_shape),
        ))
    }

    /// Native Haar synthesis using the exact gather adjoint of DWT.
    pub(crate) fn wmhf_backend_haar_iwt_f32(
        &self,
        ll: &OxideTensorF32,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.wmhf_backend_haar_dwt_backward_f32(ll, lh, hl, hh, batch, channels, height, width, config)
    }

    /// Native adjoint of Haar synthesis; its sub-bands are exactly the DWT.
    pub(crate) fn wmhf_backend_haar_iwt_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        self.wmhf_backend_haar_dwt_f32(output_grad, batch, channels, height, width, config)
    }

    /// Native gather adjoint for four WMHF Haar sub-band gradients.
    pub(crate) fn wmhf_backend_haar_dwt_backward_f32(
        &self,
        ll_grad: &OxideTensorF32,
        lh_grad: &OxideTensorF32,
        hl_grad: &OxideTensorF32,
        hh_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_haar_dwt_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input gradient count overflows".to_owned()))?;
        let expected_shape = [batch, channels, height2, width2];
        let ll_grad = self.f32v(ll_grad)?;
        let lh_grad = self.f32v(lh_grad)?;
        let hl_grad = self.f32v(hl_grad)?;
        let hh_grad = self.f32v(hh_grad)?;
        for (name, gradient) in [
            ("LL", &*ll_grad),
            ("LH", &*lh_grad),
            ("HL", &*hl_grad),
            ("HH", &*hh_grad),
        ] {
            if gradient.shape() != expected_shape || gradient.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!(
                        "{name} gradient shape/length are {:?}/{}, expected {expected_shape:?}/{coeff_len}",
                        gradient.shape(),
                        gradient.len()
                    ),
                ));
            }
        }
        let mut input_grad = self.alloc_uninit(input_len)?;
        // SAFETY: one thread owns each input-gradient element and only reads
        // checked sub-band elements from its corresponding Haar tile.
        unsafe {
            self.wmhf.wmhf_haar_dwt_backward(
                &self.stream,
                launch::elementwise(input_len, config),
                &ll_grad.buf,
                &lh_grad.buf,
                &hl_grad.buf,
                &hh_grad.buf,
                &mut input_grad,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            input_grad,
            &[batch, channels, height, width],
        ))
    }

}
