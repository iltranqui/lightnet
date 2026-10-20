//! Softmax, reorg, dropout and scale_channels (the connected layer goes
//! through the GEMM and bias kernels instead).
//!
//! Every kernel owns its output elements exclusively (no atomics), so results
//! are deterministic.  Semantics follow `src-lib/{softmax,reorg,dropout,
//! scale_channels}_layer.*` and the `darknet-core` CPU runtime; the host
//! references live in `darknet_cuda::host_ops`.

use super::activations::exact_gelu_value;
use super::convert::{f16_bits_to_f32, f32_to_f16_bits};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Device twin of `darknet_cuda::dropout_unit`: must stay bit-identical.
#[device]
pub fn dropout_unit(seed: u64, index: u64) -> f32 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0
}

/// Match CPU half-away rounding without f32's (x+0.5) halfway rounding error.
/// Clamp before converting so NaN/infinities cannot poison integer indexing.
#[device]
pub fn marker_position(value: f32, sequence: usize) -> usize {
    let value = value as f64;
    if !(value >= 0.0) {
        0
    } else if value >= (sequence - 1) as f64 {
        sequence - 1
    } else {
        (value + 0.5) as usize
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Source ah_features: one thread per sample, serial option reductions.
    /// Masked slots retain finite -1e4 logits, even if all slots are masked.
    /// Four-direction WMHF EMA scan. One thread owns one whole row/column
    /// sequence and writes each sequence position to a disjoint direction plane.
    #[kernel]
    pub fn wmhf_scan4_forward(
        input: &[f32],
        weights: &[f32],
        directions: *mut f32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let sequence = thread::index_1d().get();
        let per_channel = 2 * height + 2 * width;
        if sequence >= batch * channels * per_channel {
            return;
        }
        let channel_batch = sequence / per_channel;
        let channel = channel_batch % channels;
        let plane = channel_batch * height * width;
        let local = sequence % per_channel;
        let horizontal = local < 2 * height;
        let within_line = if horizontal {
            local
        } else {
            local - 2 * height
        };
        let reverse = within_line % 2 == 1;
        let line = within_line / 2;
        let direction = if horizontal {
            if reverse { 1 } else { 0 }
        } else if reverse {
            3
        } else {
            2
        };
        let length = if horizontal { width } else { height };
        let raw_a = weights[channel];
        let sigmoid = if raw_a >= 0.0 {
            1.0 / (1.0 + (-raw_a).exp())
        } else {
            let z = raw_a.exp();
            z / (1.0 + z)
        };
        let a = 0.98 * sigmoid;
        let b_weight = weights[channels + channel];
        let c_weight = weights[2 * channels + channel];
        let d_weight = weights[3 * channels + channel];
        let total = batch * channels * height * width;
        let mut state = 0.0;
        let mut step = 0;
        while step < length {
            let position = if reverse { length - 1 - step } else { step };
            let spatial = if horizontal {
                line * width + position
            } else {
                position * width + line
            };
            let input_index = plane + spatial;
            let value = input[input_index];
            state = a * state + b_weight * value;
            let output_index = direction * total + input_index;
            unsafe {
                directions
                    .add(output_index)
                    .write(0.25 * (c_weight * state + d_weight * value));
            }
            step += 1;
        }
    }

    /// Combine the four non-overlapping direction planes into the WMHF output.
    #[kernel]
    pub fn wmhf_scan4_combine(directions: &[f32], mut output: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let index = i.get();
        if index >= output.len() {
            return;
        }
        let total = output.len();
        let value = directions[index]
            + directions[total + index]
            + directions[2 * total + index]
            + directions[3 * total + index];
        if let Some(dst) = output.get_mut(i) {
            *dst = value;
        }
    }

    /// One thread per row/column sequence writes every forward state to a
    /// disjoint direction plane for the sequence-owned backward pass.
    #[kernel]
    pub fn wmhf_scan4_backward_states(
        input: &[f32],
        weights: &[f32],
        states: *mut f32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let sequence = thread::index_1d().get();
        let per_channel = 2 * height + 2 * width;
        if sequence >= batch * channels * per_channel {
            return;
        }
        let channel_batch = sequence / per_channel;
        let channel = channel_batch % channels;
        let plane = channel_batch * height * width;
        let local = sequence % per_channel;
        let horizontal = local < 2 * height;
        let within_line = if horizontal {
            local
        } else {
            local - 2 * height
        };
        let reverse = within_line % 2 == 1;
        let line = within_line / 2;
        let direction = if horizontal {
            if reverse { 1 } else { 0 }
        } else if reverse {
            3
        } else {
            2
        };
        let length = if horizontal { width } else { height };
        let raw_a = weights[channel];
        let sigmoid = if raw_a >= 0.0 {
            1.0 / (1.0 + (-raw_a).exp())
        } else {
            let z = raw_a.exp();
            z / (1.0 + z)
        };
        let a = 0.98 * sigmoid;
        let b_weight = weights[channels + channel];
        let total = batch * channels * height * width;
        let mut state = 0.0;
        let mut step = 0;
        while step < length {
            let position = if reverse { length - 1 - step } else { step };
            let spatial = if horizontal {
                line * width + position
            } else {
                position * width + line
            };
            let input_index = plane + spatial;
            state = a * state + b_weight * input[input_index];
            unsafe {
                states.add(direction * total + input_index).write(state);
            }
            step += 1;
        }
    }

    /// One thread per sequence walks the adjoint recurrence backward and writes
    /// disjoint input/parameter contribution planes, avoiding per-pixel prefix
    /// and suffix recomputation.
    #[kernel]
    pub fn wmhf_scan4_backward_sequences(
        input: &[f32],
        output_grad: &[f32],
        weights: &[f32],
        states: &[f32],
        input_parts: *mut f32,
        da_parts: *mut f32,
        db_parts: *mut f32,
        dc_parts: *mut f32,
        dd_parts: *mut f32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let sequence = thread::index_1d().get();
        let per_channel = 2 * height + 2 * width;
        if sequence >= batch * channels * per_channel {
            return;
        }
        let channel_batch = sequence / per_channel;
        let channel = channel_batch % channels;
        let plane = channel_batch * height * width;
        let local = sequence % per_channel;
        let horizontal = local < 2 * height;
        let within_line = if horizontal {
            local
        } else {
            local - 2 * height
        };
        let reverse = within_line % 2 == 1;
        let line = within_line / 2;
        let direction = if horizontal {
            if reverse { 1 } else { 0 }
        } else if reverse {
            3
        } else {
            2
        };
        let length = if horizontal { width } else { height };
        let raw_a = weights[channel];
        let sigmoid = if raw_a >= 0.0 {
            1.0 / (1.0 + (-raw_a).exp())
        } else {
            let z = raw_a.exp();
            z / (1.0 + z)
        };
        let a = 0.98 * sigmoid;
        let b_weight = weights[channels + channel];
        let c_weight = weights[2 * channels + channel];
        let d_weight = weights[3 * channels + channel];
        let total = batch * channels * height * width;
        let mut ds_next = 0.0;
        let mut step = length;
        while step > 0 {
            let sequence_pos = step - 1;
            let position = if reverse {
                length - 1 - sequence_pos
            } else {
                sequence_pos
            };
            let spatial = if horizontal {
                line * width + position
            } else {
                position * width + line
            };
            let input_index = plane + spatial;
            let partial_index = direction * total + input_index;
            let dy = 0.25 * output_grad[input_index];
            let ds = ds_next + dy * c_weight;
            let state_prev = if sequence_pos == 0 {
                0.0
            } else {
                let previous = if reverse {
                    length - sequence_pos
                } else {
                    sequence_pos - 1
                };
                let previous_spatial = if horizontal {
                    line * width + previous
                } else {
                    previous * width + line
                };
                states[direction * total + plane + previous_spatial]
            };
            let value = input[input_index];
            unsafe {
                input_parts
                    .add(partial_index)
                    .write(dy * d_weight + ds * b_weight);
                da_parts
                    .add(partial_index)
                    .write(ds * state_prev * 0.98 * sigmoid * (1.0 - sigmoid));
                db_parts.add(partial_index).write(ds * value);
                dc_parts
                    .add(partial_index)
                    .write(dy * states[partial_index]);
                dd_parts.add(partial_index).write(dy * value);
            }
            ds_next = ds * a;
            step -= 1;
        }
    }

    /// Legacy per-pixel reference implementation retained for comparison.
    #[kernel]
    pub fn wmhf_scan4_backward_partials(
        input: &[f32],
        output_grad: &[f32],
        weights: &[f32],
        mut input_parts: DisjointSlice<f32>,
        mut da_parts: DisjointSlice<f32>,
        mut db_parts: DisjointSlice<f32>,
        mut dc_parts: DisjointSlice<f32>,
        mut dd_parts: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let i = thread::index_1d().get();
        let total = batch * channels * height * width;
        if i >= 4 * total {
            return;
        }
        let direction = i / total;
        let input_index = i % total;
        let x = input_index % width;
        let y = (input_index / width) % height;
        let channel = (input_index / (width * height)) % channels;
        let batch_index = input_index / (width * height * channels);
        let plane = (batch_index * channels + channel) * height * width;
        let horizontal = direction < 2;
        let line = if horizontal { y } else { x };
        let length = if horizontal { width } else { height };
        let target = match direction {
            0 => x,
            1 => width - 1 - x,
            2 => y,
            _ => height - 1 - y,
        };
        let raw_a = weights[channel];
        let sigmoid = if raw_a >= 0.0 {
            1.0 / (1.0 + (-raw_a).exp())
        } else {
            let z = raw_a.exp();
            z / (1.0 + z)
        };
        let a = 0.98 * sigmoid;
        let b_weight = weights[channels + channel];
        let c_weight = weights[2 * channels + channel];
        let d_weight = weights[3 * channels + channel];
        let mut state_prev = 0.0;
        let mut step = 0;
        while step < target {
            let position = if direction == 1 {
                width - 1 - step
            } else if direction == 3 {
                height - 1 - step
            } else {
                step
            };
            let index = if horizontal {
                plane + line * width + position
            } else {
                plane + position * width + line
            };
            state_prev = a * state_prev + b_weight * input[index];
            step += 1;
        }
        let state = a * state_prev + b_weight * input[input_index];
        let mut ds_next = 0.0;
        let mut ds_target = 0.0;
        step = length;
        while step > target {
            let sequence_pos = step - 1;
            let position = if direction == 1 {
                width - 1 - sequence_pos
            } else if direction == 3 {
                height - 1 - sequence_pos
            } else {
                sequence_pos
            };
            let index = if horizontal {
                plane + line * width + position
            } else {
                plane + position * width + line
            };
            let dy = 0.25 * output_grad[index];
            let ds = ds_next + dy * c_weight;
            if sequence_pos == target {
                ds_target = ds;
            }
            ds_next = ds * a;
            step -= 1;
        }
        let dy = 0.25 * output_grad[input_index];
        let value = input[input_index];
        if let Some(dst) = input_parts.get_mut(thread::index_1d()) {
            *dst = dy * d_weight + ds_target * b_weight;
        }
        if let Some(dst) = da_parts.get_mut(thread::index_1d()) {
            *dst = ds_target * state_prev * 0.98 * sigmoid * (1.0 - sigmoid);
        }
        if let Some(dst) = db_parts.get_mut(thread::index_1d()) {
            *dst = ds_target * value;
        }
        if let Some(dst) = dc_parts.get_mut(thread::index_1d()) {
            *dst = dy * state;
        }
        if let Some(dst) = dd_parts.get_mut(thread::index_1d()) {
            *dst = dy * value;
        }
    }

    #[kernel]
    pub fn wmhf_scan4_backward_input(input_parts: &[f32], mut input_grad: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input_grad.len() {
            return;
        }
        let total = input_grad.len();
        let value = input_parts[i]
            + input_parts[total + i]
            + input_parts[2 * total + i]
            + input_parts[3 * total + i];
        if let Some(dst) = input_grad.get_mut(idx) {
            *dst = value;
        }
    }

    #[kernel]
    pub fn wmhf_scan4_backward_weights(
        da_parts: &[f32],
        db_parts: &[f32],
        dc_parts: &[f32],
        dd_parts: &[f32],
        mut weight_grad: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= weight_grad.len() {
            return;
        }
        let parameter = i / channels;
        let channel = i % channels;
        let total = batch * channels * height * width;
        let area = height * width;
        let mut sum = 0.0;
        let mut direction = 0;
        while direction < 4 {
            let mut b = 0;
            while b < batch {
                let base = direction * total + (b * channels + channel) * area;
                let mut spatial = 0;
                while spatial < area {
                    let index = base + spatial;
                    sum += if parameter == 0 {
                        da_parts[index]
                    } else if parameter == 1 {
                        db_parts[index]
                    } else if parameter == 2 {
                        dc_parts[index]
                    } else {
                        dd_parts[index]
                    };
                    spatial += 1;
                }
                b += 1;
            }
            direction += 1;
        }
        if let Some(dst) = weight_grad.get_mut(idx) {
            *dst = sum;
        }
    }

    /// WMHF high-frequency magnitude sum with nearest 2x tile upsampling.
    #[kernel]
    pub fn wmhf_high_frequency_forward(
        lh: &[f32],
        hl: &[f32],
        hh: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        height2: usize,
        width2: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = i % width;
        let y = (i / width) % height;
        let channel_batch = i / (width * height);
        let coeff = channel_batch * height2 * width2 + (y / 2) * width2 + (x / 2);
        let value = lh[coeff].abs() + hl[coeff].abs() + hh[coeff].abs();
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
        let _ = channels;
    }

    /// Gather adjoint for high-frequency magnitude and nearest upsampling.
    #[kernel]
    pub fn wmhf_high_frequency_backward(
        output_grad: &[f32],
        lh: &[f32],
        hl: &[f32],
        hh: &[f32],
        mut lh_grad: DisjointSlice<f32>,
        mut hl_grad: DisjointSlice<f32>,
        mut hh_grad: DisjointSlice<f32>,
        height: usize,
        width: usize,
        height2: usize,
        width2: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= lh_grad.len() {
            return;
        }
        let x2 = i % width2;
        let y2 = (i / width2) % height2;
        let input_plane = height * width;
        let channel_batch = i / (height2 * width2);
        let base = channel_batch * input_plane;
        let mut sum = 0.0;
        let mut y = 2 * y2;
        while y < height && y < 2 * y2 + 2 {
            let mut x = 2 * x2;
            while x < width && x < 2 * x2 + 2 {
                sum += output_grad[base + y * width + x];
                x += 1;
            }
            y += 1;
        }
        let sign_lh = if lh[i] > 0.0 {
            1.0
        } else if lh[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        let sign_hl = if hl[i] > 0.0 {
            1.0
        } else if hl[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        let sign_hh = if hh[i] > 0.0 {
            1.0
        } else if hh[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        if let Some(dst) = lh_grad.get_mut(thread::index_1d()) {
            *dst = sum * sign_lh;
        }
        if let Some(dst) = hl_grad.get_mut(thread::index_1d()) {
            *dst = sum * sign_hl;
        }
        if let Some(dst) = hh_grad.get_mut(thread::index_1d()) {
            *dst = sum * sign_hh;
        }
    }

    /// NCHW channel-range gather used by the WMHF local/global branch split.
    #[kernel]
    pub fn wmhf_split_channels(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        begin: usize,
        count: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let area = height * width;
        let output_channel = (i / area) % count;
        let batch = i / (area * count);
        let spatial = i % area;
        let source = (batch * channels + begin + output_channel) * area + spatial;
        if let Some(dst) = output.get_mut(idx) {
            *dst = input[source];
        }
    }

    /// Concatenate 3x/5x/7x local depthwise branch outputs channel-major.
    #[kernel]
    pub fn wmhf_concat_local(
        branch3: &[f32],
        branch5: &[f32],
        branch7: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let area = height * width;
        let output_channels = 3 * channels;
        let batch = i / (area * output_channels);
        let output_channel = (i / area) % output_channels;
        let branch = output_channel / channels;
        let channel = output_channel % channels;
        let spatial = i % area;
        let source = (batch * channels + channel) * area + spatial;
        let value = if branch == 0 {
            branch3[source]
        } else if branch == 1 {
            branch5[source]
        } else {
            branch7[source]
        };
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// WMHF residual/high-frequency gate fusion.
    #[kernel]
    pub fn wmhf_apply_gate(
        fuse: &[f32],
        gate: &[f32],
        projected: &[f32],
        shortcut: &[f32],
        mut output: DisjointSlice<f32>,
        freq_scale: f32,
        use_shortcut: bool,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let residual = if use_shortcut { shortcut[i] } else { 0.0 };
        let value = fuse[i] + freq_scale * gate[i] * projected[i] + residual;
        if let Some(dst) = output.get_mut(idx) {
            *dst = value;
        }
    }

    /// Exact adjoint of WMHF residual/high-frequency gate fusion.
    #[kernel]
    pub fn wmhf_apply_gate_backward(
        output_grad: &[f32],
        gate: &[f32],
        projected: &[f32],
        mut fuse_grad: DisjointSlice<f32>,
        mut gate_grad: DisjointSlice<f32>,
        mut projected_grad: DisjointSlice<f32>,
        mut shortcut_grad: DisjointSlice<f32>,
        freq_scale: f32,
        use_shortcut: bool,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output_grad.len() {
            return;
        }
        let dy = output_grad[i];
        if let Some(dst) = fuse_grad.get_mut(thread::index_1d()) {
            *dst = dy;
        }
        if let Some(dst) = gate_grad.get_mut(thread::index_1d()) {
            *dst = freq_scale * projected[i] * dy;
        }
        if let Some(dst) = projected_grad.get_mut(thread::index_1d()) {
            *dst = freq_scale * gate[i] * dy;
        }
        if let Some(dst) = shortcut_grad.get_mut(thread::index_1d()) {
            *dst = if use_shortcut { dy } else { 0.0 };
        }
    }

    /// WMHF Haar decomposition: one thread per NCHW low-resolution tile writes
    /// each sub-band in the source's separate LL/LH/HL/HH buffer order.
    #[kernel]
    pub fn wmhf_haar_dwt(
        input: &[f32],
        mut ll: DisjointSlice<f32>,
        mut lh: DisjointSlice<f32>,
        mut hl: DisjointSlice<f32>,
        mut hh: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        height2: usize,
        width2: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= ll.len() {
            return;
        }
        let x = i % width2;
        let y = (i / width2) % height2;
        let channel = (i / (width2 * height2)) % channels;
        let batch = i / (width2 * height2 * channels);
        let y0 = (2 * y).min(height - 1);
        let y1 = (y0 + 1).min(height - 1);
        let x0 = (2 * x).min(width - 1);
        let x1 = (x0 + 1).min(width - 1);
        let plane = (batch * channels + channel) * height * width;
        let v00 = input[plane + y0 * width + x0];
        let v01 = input[plane + y0 * width + x1];
        let v10 = input[plane + y1 * width + x0];
        let v11 = input[plane + y1 * width + x1];
        if let Some(dst) = ll.get_mut(thread::index_1d()) {
            *dst = 0.5 * (v00 + v01 + v10 + v11);
        }
        if let Some(dst) = lh.get_mut(thread::index_1d()) {
            *dst = 0.5 * (v00 - v01 + v10 - v11);
        }
        if let Some(dst) = hl.get_mut(thread::index_1d()) {
            *dst = 0.5 * (v00 + v01 - v10 - v11);
        }
        if let Some(dst) = hh.get_mut(thread::index_1d()) {
            *dst = 0.5 * (v00 - v01 - v10 + v11);
        }
    }

    /// Exact gather adjoint of `wmhf_haar_dwt`: one thread per input pixel
    /// combines all row/column corners duplicated at odd replicated edges.
    #[kernel]
    pub fn wmhf_haar_dwt_backward(
        ll_grad: &[f32],
        lh_grad: &[f32],
        hl_grad: &[f32],
        hh_grad: &[f32],
        mut input_grad: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
        height2: usize,
        width2: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= input_grad.len() {
            return;
        }
        let x = i % width;
        let y = (i / width) % height;
        let channel = (i / (width * height)) % channels;
        let batch = i / (width * height * channels);
        let tile_y = y / 2;
        let tile_x = x / 2;
        let y0 = (2 * tile_y).min(height - 1);
        let y1 = (y0 + 1).min(height - 1);
        let x0 = (2 * tile_x).min(width - 1);
        let x1 = (x0 + 1).min(width - 1);
        let coeff_index =
            (batch * channels + channel) * height2 * width2 + tile_y * width2 + tile_x;
        let a = 0.5 * ll_grad[coeff_index];
        let b = 0.5 * lh_grad[coeff_index];
        let c = 0.5 * hl_grad[coeff_index];
        let d = 0.5 * hh_grad[coeff_index];
        let mut gradient = 0.0;
        if y == y0 && x == x0 {
            gradient += a + b + c + d;
        }
        if y == y0 && x == x1 {
            gradient += a - b + c - d;
        }
        if y == y1 && x == x0 {
            gradient += a + b - c - d;
        }
        if y == y1 && x == x1 {
            gradient += a - b - c + d;
        }
        if let Some(dst) = input_grad.get_mut(thread::index_1d()) {
            *dst = gradient;
        }
    }

    #[kernel]
    pub fn laya_act_features(
        head: &[f32],
        logits: &[f32],
        markers: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
        marker_channels: usize,
        head_width: usize,
        marker_width: usize,
    ) {
        let b = thread::index_1d().get();
        if b < output.len() / (hidden + 4) {
            let first_active = markers[(b * marker_channels + 1) * marker_width] > 0.5;
            let mut max = if first_active {
                logits[b * options]
            } else {
                -1e4
            };
            let mut valid = 0usize;
            let mut j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                if active {
                    valid += 1;
                }
                max = max.max(z);
                j += 1;
            }
            let mut sum = 0.0f32;
            j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                sum += (z - max).exp();
                j += 1;
            }
            let mut top1 = -1.0f32;
            let mut top2 = -1.0f32;
            let mut entropy = 0.0f32;
            j = 0;
            while j < options {
                let active = markers[(b * marker_channels + 2 * j + 1) * marker_width] > 0.5;
                let z = if active {
                    logits[b * options + j]
                } else {
                    -1e4
                };
                let p = (z - max).exp() / sum;
                entropy -= p * p.max(1e-9).ln();
                if p > top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
                j += 1;
            }
            let count = if valid < 2 { 2.0 } else { valid as f32 };
            let base = b * (hidden + 4);
            // SAFETY: layout/storage are checked by the host. Threads own
            // disjoint sample rows and write every CLS/summary element once.
            unsafe {
                let mut c = 0;
                while c < hidden {
                    *output.as_mut_ptr().add(base + c) = head[(b * hidden + c) * head_width];
                    c += 1;
                }
                *output.as_mut_ptr().add(base + hidden) = top1;
                *output.as_mut_ptr().add(base + hidden + 1) = top1 - top2;
                *output.as_mut_ptr().add(base + hidden + 2) = entropy / count.ln();
                *output.as_mut_ptr().add(base + hidden + 3) = count / 255.0;
            }
        }
    }

    /// Detached LAYA act-head features from half-storage sources.
    #[kernel]
    pub fn laya_act_features_f16(
        head: &[u16],
        logits: &[u16],
        markers: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: usize,
        options: usize,
        marker_channels: usize,
        head_width: usize,
        marker_width: usize,
    ) {
        let b = thread::index_1d().get();
        if b < output.len() / (hidden + 4) {
            let mut max = f32::NEG_INFINITY;
            let mut valid = 0usize;
            let mut j = 0usize;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                if active {
                    valid += 1;
                }
                max = max.max(z);
                j += 1;
            }
            let mut sum = 0.0f32;
            j = 0;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                sum += (z - max).exp();
                j += 1;
            }
            let mut top1 = -1.0f32;
            let mut top2 = -1.0f32;
            let mut entropy = 0.0f32;
            j = 0;
            while j < options {
                let active =
                    f16_bits_to_f32(markers[(b * marker_channels + 2 * j + 1) * marker_width])
                        > 0.5;
                let z = if active {
                    f16_bits_to_f32(logits[b * options + j])
                } else {
                    -1.0e4
                };
                let p = (z - max).exp() / sum;
                entropy -= p * p.max(1.0e-9).ln();
                if p > top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
                j += 1;
            }
            let count = if valid < 2 { 2.0 } else { valid as f32 };
            let base = b * (hidden + 4);
            // SAFETY: layout/storage are validated and each thread owns one row.
            unsafe {
                let mut c = 0usize;
                while c < hidden {
                    *output.as_mut_ptr().add(base + c) = head[(b * hidden + c) * head_width];
                    c += 1;
                }
                *output.as_mut_ptr().add(base + hidden) = f32_to_f16_bits(top1, 0);
                *output.as_mut_ptr().add(base + hidden + 1) = f32_to_f16_bits(top1 - top2, 0);
                *output.as_mut_ptr().add(base + hidden + 2) =
                    f32_to_f16_bits(entropy / count.ln(), 0);
                *output.as_mut_ptr().add(base + hidden + 3) = f32_to_f16_bits(count / 255.0, 0);
            }
        }
    }

    /// Scatter the pooled CLS adjoint into token zero of each NCHW channel.
    /// One thread owns each output element; summary channels are detached.
    #[kernel]
    pub fn laya_act_cls_scatter(
        dcls: &[f32],
        head_grad: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: usize,
    ) {
        let i = thread::index_1d();
        let idx = i.get();
        if idx < output.len() {
            let value = head_grad[idx]
                + if idx % tokens == 0 {
                    dcls[idx / tokens]
                } else {
                    0.0
                };
            // SAFETY: host validation proves the source extents and each thread
            // writes exactly one output element.
            unsafe {
                *output.as_mut_ptr().add(idx) = value;
            }
        }
    }

    /// One thread owns each gathered output; both inputs stay device-resident.
    #[kernel]
    pub fn marker_gather(
        features: &[f32],
        markers: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < output.len() {
            let j = i % options;
            let channel = i / options;
            let b = channel / hidden;
            let pos = marker_position(markers[(b * marker_channels + 2 * j) * sequence], sequence);
            // SAFETY: validated NCHW extents bound channel/pos and each thread owns i.
            unsafe {
                *output.as_mut_ptr().add(i) = features[channel * sequence + pos];
            }
        }
    }

    /// Half-storage inference: decode only marker metadata; feature bits are
    /// copied without floating arithmetic (including NaN payloads/subnormals).
    #[kernel]
    pub fn marker_gather_f16(
        features: &[u16],
        markers: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < output.len() {
            let j = i % options;
            let channel = i / options;
            let b = channel / hidden;
            let marker = f16_bits_to_f32(markers[(b * marker_channels + 2 * j) * sequence]);
            let pos = marker_position(marker, sequence);
            // SAFETY: checked extents bound the read; one thread owns each output.
            unsafe {
                *output.as_mut_ptr().add(i) = features[channel * sequence + pos];
            }
        }
    }

    /// One thread per input feature. Ascending option reduction is deterministic
    /// even for duplicate markers, without atomics or cross-thread races.
    #[kernel]
    pub fn marker_gather_backward(
        markers: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        marker_channels: usize,
        options: usize,
    ) {
        let i = thread::index_1d().get();
        if i < dx.len() {
            let channel = i / sequence;
            let b = channel / hidden;
            let pos = i % sequence;
            let mut sum = 0.0f32;
            let mut j = 0;
            while j < options {
                let selected =
                    marker_position(markers[(b * marker_channels + 2 * j) * sequence], sequence);
                if selected == pos {
                    sum += dy[channel * options + j];
                }
                j += 1;
            }
            // SAFETY: the host validates storage; each thread writes one input gradient.
            unsafe {
                *dx.as_mut_ptr().add(i) = sum;
            }
        }
    }

    /// Copy one contiguous sequence chunk to an independent output tensor.
    #[kernel]
    pub fn sequence_slice_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < output.len() {
            // SAFETY: the host validates the timestep range against input.
            unsafe {
                *output.as_mut_ptr().add(i) = input[offset + i];
            }
        }
    }

    /// Copy one contiguous recurrent timestep into its disjoint output range.
    #[kernel]
    pub fn sequence_concat_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            // SAFETY: host-provided offsets partition the output by timestep.
            unsafe {
                *output.as_mut_ptr().add(offset + i) = input[i];
            }
        }
    }

    /// Exact LSTM cell/gate derivative for one BPTT step. Each thread owns one
    /// hidden element and writes all gate deltas plus the cell gradient for the
    /// preceding step.
    #[kernel]
    pub fn lstm_gate_backward(
        hidden_gradient: &[f32],
        cell: &[f32],
        previous_cell: &[f32],
        forget: &[f32],
        input_gate: &[f32],
        candidate: &[f32],
        output_gate: &[f32],
        next_cell_gradient: &[f32],
        mut forget_gradient: DisjointSlice<f32>,
        mut input_gate_gradient: DisjointSlice<f32>,
        mut candidate_gradient: DisjointSlice<f32>,
        mut output_gate_gradient: DisjointSlice<f32>,
        mut cell_gradient_out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= hidden_gradient.len() {
            return;
        }
        let tanh_cell = cell[i].tanh();
        let output_delta = hidden_gradient[i] * tanh_cell;
        let cell_delta = hidden_gradient[i] * output_gate[i] * (1.0 - tanh_cell * tanh_cell)
            + next_cell_gradient[i];
        let mut forget_delta = cell_delta * previous_cell[i];
        let mut input_delta = cell_delta * candidate[i];
        let mut candidate_delta = cell_delta * input_gate[i];
        forget_delta *= forget[i] * (1.0 - forget[i]);
        input_delta *= input_gate[i] * (1.0 - input_gate[i]);
        candidate_delta *= 1.0 - candidate[i] * candidate[i];
        let mut output_delta = output_delta;
        output_delta *= output_gate[i] * (1.0 - output_gate[i]);
        // SAFETY: each thread writes the matching element in each output.
        unsafe {
            *forget_gradient.as_mut_ptr().add(i) = forget_delta;
            *input_gate_gradient.as_mut_ptr().add(i) = input_delta;
            *candidate_gradient.as_mut_ptr().add(i) = candidate_delta;
            *output_gate_gradient.as_mut_ptr().add(i) = output_delta;
            *cell_gradient_out.as_mut_ptr().add(i) = cell_delta * forget[i];
        }
    }

    /// Global-average pool followed by the MoE router's affine projection. One
    /// output thread computes one (sample, expert) logit; spatial reductions
    /// follow the same sequential f32 order as the CPU reference.
    #[kernel]
    pub fn moe_router_logits(
        input: &[f32],
        router_weights: &[f32],
        router_biases: &[f32],
        mut logits: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        experts: usize,
        batch: usize,
        temperature: f32,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= batch * experts {
            return;
        }
        let b = linear / experts;
        let e = linear % experts;
        let inv_spatial = 1.0 / spatial as f32;
        let temperature = if temperature > 1e-6 {
            temperature
        } else {
            1e-6
        };
        let mut value = router_biases[e];
        let mut c = 0usize;
        while c < channels {
            let base = (b * channels + c) * spatial;
            let mut sum = 0.0f32;
            let mut s = 0usize;
            while s < spatial {
                sum += input[base + s];
                s += 1;
            }
            let pooled = sum * inv_spatial;
            value += router_weights[e * channels + c] * pooled;
            c += 1;
        }
        if let Some(out) = logits.get_mut(idx) {
            *out = value / temperature;
        }
    }

    /// Top-k masked softmax. A single thread owns each row; counting strictly
    /// greater logits reproduces C++'s `v < kth` mask, including ties.
    #[kernel]
    pub fn moe_router_softmax_topk(
        logits: &[f32],
        mut gates: DisjointSlice<f32>,
        batch: usize,
        experts: usize,
        top_k: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batch {
            return;
        }
        let base = b * experts;
        let mut maximum = -f32::MAX;
        let mut e = 0usize;
        while e < experts {
            let value = logits[base + e];
            let mut keep = true;
            if top_k > 0 && top_k < experts {
                let mut greater = 0usize;
                let mut j = 0usize;
                while j < experts {
                    if logits[base + j] > value {
                        greater += 1;
                    }
                    j += 1;
                }
                keep = greater < top_k;
            }
            let masked = if keep { value } else { -f32::MAX };
            if masked > maximum {
                maximum = masked;
            }
            e += 1;
        }

        let mut sum = 0.0f32;
        e = 0;
        while e < experts {
            let value = logits[base + e];
            let mut keep = true;
            if top_k > 0 && top_k < experts {
                let mut greater = 0usize;
                let mut j = 0usize;
                while j < experts {
                    if logits[base + j] > value {
                        greater += 1;
                    }
                    j += 1;
                }
                keep = greater < top_k;
            }
            let masked = if keep { value } else { -f32::MAX };
            let exp = if masked <= -f32::MAX * 0.5 {
                0.0
            } else {
                (masked - maximum).exp()
            };
            // SAFETY: this thread uniquely owns the `experts`-element gate row.
            unsafe {
                *gates.as_mut_ptr().add(base + e) = exp;
            }
            sum += exp;
            e += 1;
        }

        if sum <= 0.0 || !sum.is_finite() {
            let uniform = 1.0 / experts as f32;
            e = 0;
            while e < experts {
                // SAFETY: this thread uniquely owns the `experts`-element gate row.
                unsafe {
                    *gates.as_mut_ptr().add(base + e) = uniform;
                }
                e += 1;
            }
        } else {
            e = 0;
            while e < experts {
                // SAFETY: this thread uniquely owns the `experts`-element gate row.
                unsafe {
                    *gates.as_mut_ptr().add(base + e) /= sum;
                }
                e += 1;
            }
        }
    }

    /// Global-average pool in NCHW storage to `[batch, channels]`; one thread
    /// sums the spatial plane for one sample/channel in CPU reference order.
    #[kernel]
    pub fn moe_router_pool(
        input: &[f32],
        mut pooled: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        batch: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= batch * channels {
            return;
        }
        let base = linear * spatial;
        let mut sum = 0.0f32;
        let mut s = 0usize;
        while s < spatial {
            sum += input[base + s];
            s += 1;
        }
        if let Some(out) = pooled.get_mut(idx) {
            *out = sum * (1.0 / spatial as f32);
        }
    }

    /// Broadcast the derivative of a global-average pooled descriptor.
    #[kernel]
    pub fn moe_router_pool_backward(
        pooled_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = input_gradient.get_mut(idx) {
            *out = pooled_gradient[i / spatial] * (1.0 / spatial as f32);
        }
    }

    /// Extract one expert's gate from the row-major `[batch, experts]` tensor.
    #[kernel]
    pub fn moe_gate_column(
        gates: &[f32],
        mut column: DisjointSlice<f32>,
        experts: usize,
        expert: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        if let Some(out) = column.get_mut(idx) {
            *out = gates[b * experts + expert];
        }
    }

    /// Add the exact auxiliary load-balance derivative and top-k softmax Jacobian.
    /// One thread owns a batch row, preserving the CPU's ordered expert reduction.
    #[kernel]
    pub fn moe_router_gate_backward(
        gates: &[f32],
        gate_dots: &[f32],
        mut dlogit: DisjointSlice<f32>,
        batch: usize,
        experts: usize,
        load_balance: f32,
        temperature: f32,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        if b >= batch {
            return;
        }
        let base = b * experts;
        let balance_scale = if load_balance <= 0.0 {
            0.0
        } else {
            load_balance * 2.0 * experts as f32 / batch as f32
        };
        let temperature = if temperature > 1e-6 {
            temperature
        } else {
            1e-6
        };
        let mut weighted = 0.0f32;
        let mut e = 0usize;
        while e < experts {
            let mut mean_gate = 0.0f32;
            let mut sample = 0usize;
            while sample < batch {
                mean_gate += gates[sample * experts + e];
                sample += 1;
            }
            mean_gate /= batch as f32;
            let raw = gate_dots[base + e] + balance_scale * mean_gate;
            weighted += gates[base + e] * raw;
            e += 1;
        }
        e = 0;
        while e < experts {
            let mut mean_gate = 0.0f32;
            let mut sample = 0usize;
            while sample < batch {
                mean_gate += gates[sample * experts + e];
                sample += 1;
            }
            mean_gate /= batch as f32;
            let raw = gate_dots[base + e] + balance_scale * mean_gate;
            // SAFETY: this thread uniquely owns the expert row for batch item `b`.
            unsafe {
                *dlogit.as_mut_ptr().add(base + e) =
                    gates[base + e] * (raw - weighted) / temperature;
            }
            e += 1;
        }
    }

    /// Stage 1 of per-sample gate dot: 256 strided partials per sample.
    #[kernel]
    pub fn moe_gate_dot_partial(
        output_gradient: &[f32],
        expert_output: &[f32],
        mut partial: DisjointSlice<f32>,
        per_image: usize,
        batch: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let sample = linear / 256;
        let lane = linear % 256;
        if sample >= batch {
            return;
        }
        let base = sample * per_image;
        let mut sum = 0.0f32;
        let mut i = lane;
        while i < per_image {
            sum += output_gradient[base + i] * expert_output[base + i];
            i += 256;
        }
        if let Some(out) = partial.get_mut(idx) {
            *out = sum;
        }
    }

    /// Stage 2: fold 256 partials into one gate derivative per sample.
    #[kernel]
    pub fn moe_gate_dot_reduce(partial: &[f32], mut dots: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let sample = idx.get();
        if let Some(out) = dots.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut lane = 0usize;
            while lane < 256 {
                sum += partial[sample * 256 + lane];
                lane += 1;
            }
            *out = sum;
        }
    }

    /// Fold 256 gate-dot partials into one expert column of `[batch, experts]`.
    #[kernel]
    pub fn moe_gate_dot_reduce_column(
        partial: &[f32],
        mut dots: DisjointSlice<f32>,
        experts: usize,
        expert: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        let output_index = b * experts + expert;
        if output_index >= dots.len() {
            return;
        }
        let mut sum = 0.0f32;
        let mut lane = 0usize;
        while lane < 256 {
            sum += partial[b * 256 + lane];
            lane += 1;
        }
        // SAFETY: this launch owns one column element per batch row; callers
        // launch experts serially and each launch writes a disjoint matrix column.
        unsafe {
            *dots.as_mut_ptr().add(output_index) = sum;
        }
    }

    /// `out[b * to_len + i] = i < min(from_len, to_len) ? x[b * from_len + i] : 0`
    /// (flat per-sample crop / zero-pad, the Darknet mismatched shortcut).
    #[kernel]
    pub fn shortcut_fit(x: &[f32], mut out: DisjointSlice<f32>, from_len: usize, to_len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let b = g / to_len;
            let i = g - b * to_len;
            *o = if i < from_len {
                x[b * from_len + i]
            } else {
                0.0
            };
        }
    }

    /// One query row per thread. Online max/sum and the value accumulator keep
    /// the attention workspace linear in sequence length; the host rejects
    /// head dimensions above the fixed local accumulator size.
    #[kernel]
    pub fn flash_attention_forward(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut acc = [0.0f32; 128];
        let q_base = query_row * 3 * channels + head_start;
        let mut d = 0usize;
        while d < head_dim {
            let raw = qkv[q_base + d];
            q[d] = if raw.is_finite() {
                raw.max(-128.0).min(128.0)
            } else {
                0.0
            };
            d += 1;
        }

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let mut dot = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw = qkv[k_base + d];
                let k = if raw.is_finite() {
                    raw.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                d += 1;
            }
            let raw_score = dot * scale;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let new_max = running_max.max(score);
            let correction = (running_max - new_max).exp();
            let p = (score - new_max).exp();
            running_sum = running_sum * correction + p;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            d = 0;
            while d < head_dim {
                acc[d] = acc[d] * correction + p * qkv[v_base + d];
                d += 1;
            }
            running_max = new_max;
            tk += 1;
        }

        let output_base = query_row * channels + head_start;
        d = 0;
        while d < head_dim {
            // SAFETY: each (batch, head, query) thread owns this head's disjoint
            // output channel interval for one query row.
            unsafe {
                *out.as_mut_ptr().add(output_base + d) = acc[d] / running_sum;
            }
            d += 1;
        }
        // LSE is flattened as `[batch, heads, tokens]`, matching `query`.
        unsafe {
            *lse.as_mut_ptr().add(query) = running_max + running_sum.ln();
        }
    }

    /// Reorder and type-condition channel-major LayaHead source features.
    #[kernel]
    pub fn laya_head_prepare(
        input: &[f32],
        type_embedding: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        hidden: usize,
        channels: usize,
        sequence: usize,
        qtype: i32,
    ) {
        let index = thread::index_1d().get();
        if index >= tokens.len() {
            return;
        }
        let b = index / (sequence * hidden);
        let rem = index - b * sequence * hidden;
        let t = rem / hidden;
        let h = rem - t * hidden;
        let selected = if qtype >= 0 {
            qtype as usize
        } else {
            let raw = input[(b * channels + hidden + 1) * sequence];
            (raw.round() as i32).max(0).min(2) as usize
        };
        let value =
            input[(b * channels + h) * sequence + t] + type_embedding[selected * hidden + h];
        // SAFETY: input shape and metadata are checked by the host. Each
        // thread owns one token feature; h=0 owns the per-token mask write.
        unsafe {
            *tokens.as_mut_ptr().add(index) = value;
            if h == 0 {
                *mask.as_mut_ptr().add(b * sequence + t) =
                    input[(b * channels + hidden) * sequence + t];
            }
        }
    }

    /// Pack token-major `[B,T,H]` states into NCHW channel-major `[B,H,1,T]`.
    #[kernel]
    pub fn laya_head_pack(
        tokens: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let b = index / (hidden * sequence);
            let rem = index - b * hidden * sequence;
            let h = rem / sequence;
            let t = rem - h * sequence;
            let source = (b * sequence + t) * hidden + h;
            // SAFETY: equal element counts and the inverse NCHW permutation
            // guarantee source and destination indices are in range.
            unsafe {
                *output.as_mut_ptr().add(index) = tokens[source];
            }
        }
    }

    /// Gather token embeddings and copy the channel-major attention mask.
    #[kernel]
    pub fn laya_encoder_embed(
        input: &[f32],
        embeddings: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        hidden: usize,
        sequence: usize,
        vocab: usize,
    ) {
        let index = thread::index_1d().get();
        if index < tokens.len() {
            let h = index % hidden;
            let row = index / hidden;
            let t = row % sequence;
            let b = row / sequence;
            let raw_id = input[(b * 2) * sequence + t];
            let id = (raw_id.round() as i32).max(0).min(vocab as i32 - 1) as usize;
            // SAFETY: host shape checks validate the input and embedding extents.
            unsafe {
                *tokens.as_mut_ptr().add(index) = embeddings[id * hidden + h];
                if h == 0 {
                    *mask.as_mut_ptr().add(b * sequence + t) = input[(b * 2 + 1) * sequence + t];
                }
            }
        }
    }

    /// Rotary, key-masked online attention with optional local-window restriction.
    #[kernel]
    pub fn laya_encoder_attention(
        qkv: &[f32],
        mask: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
        heads: usize,
        window_radius: usize,
        sliding: bool,
    ) {
        let query_head = thread::index_1d().get();
        let query_heads = output.len() / (hidden / heads);
        if query_head >= query_heads {
            return;
        }
        let head_dim = hidden / heads;
        let half = head_dim / 2;
        let head = query_head % heads;
        let row = query_head / heads;
        let query = row % sequence;
        let batch = row / sequence;
        let head_start = head * head_dim;
        let qrow = row * 3 * hidden + head_start;
        let out_base = row * hidden + head_start;
        // SAFETY: each query/head thread owns this non-overlapping output span.
        unsafe {
            let out_ptr = output.as_mut_ptr().add(out_base);
            let mut d = 0;
            while d < head_dim {
                *out_ptr.add(d) = 0.0;
                d += 1;
            }
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut key = 0;
            while key < sequence {
                if mask[batch * sequence + key] > 0.5
                    && (!sliding
                        || ((query as isize - key as isize).abs() as usize) <= window_radius)
                {
                    let krow = (batch * sequence + key) * 3 * hidden + hidden + head_start;
                    let mut dot = 0.0f32;
                    let mut j = 0;
                    while j < half {
                        let c_q = cos[query * half + j];
                        let s_q = sin[query * half + j];
                        let qa = qkv[qrow + j];
                        let qb = qkv[qrow + j + half];
                        let q0 = qa * c_q - qb * s_q;
                        let q1 = qb * c_q + qa * s_q;
                        let c_k = cos[key * half + j];
                        let s_k = sin[key * half + j];
                        let ka = qkv[krow + j];
                        let kb = qkv[krow + j + half];
                        let k0 = ka * c_k - kb * s_k;
                        let k1 = kb * c_k + ka * s_k;
                        dot += q0 * k0 + q1 * k1;
                        j += 1;
                    }
                    let score = dot * scale;
                    let next_max = running_max.max(score);
                    let alpha = if running_sum == 0.0 {
                        0.0
                    } else {
                        (running_max - next_max).exp()
                    };
                    let probability = (score - next_max).exp();
                    let vrow = (batch * sequence + key) * 3 * hidden + 2 * hidden + head_start;
                    d = 0;
                    while d < head_dim {
                        let old = *out_ptr.add(d);
                        *out_ptr.add(d) = old * alpha + probability * qkv[vrow + d];
                        d += 1;
                    }
                    running_sum = running_sum * alpha + probability;
                    running_max = next_max;
                }
                key += 1;
            }
            if running_sum > 0.0 {
                d = 0;
                while d < head_dim {
                    *out_ptr.add(d) /= running_sum;
                    d += 1;
                }
            }
        }
    }

    /// First-chunk exact-erf GELU multiplied by the raw second GeGLU chunk.
    #[kernel]
    pub fn laya_encoder_geglu(input: &[f32], mut output: DisjointSlice<f32>, intermediate: usize) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let column = index % intermediate;
            let row = index / intermediate;
            let base = row * 2 * intermediate;
            let x = input[base + column];
            let gate = input[base + intermediate + column];
            let value = exact_gelu_value(x);
            // SAFETY: all rows have the validated doubled intermediate extent.
            unsafe {
                *output.as_mut_ptr().add(index) = value * gate;
            }
        }
    }

    /// Pack token-major `[B*T,H]` into channel-major `[B,H,1,T]`.
    #[kernel]
    pub fn laya_encoder_pack(
        tokens: &[f32],
        mut output: DisjointSlice<f32>,
        sequence: usize,
        hidden: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let batch = index / (hidden * sequence);
            let rem = index - batch * hidden * sequence;
            let h = rem / sequence;
            let t = rem % sequence;
            let source = (batch * sequence + t) * hidden + h;
            // SAFETY: input/output element counts are equal.
            unsafe {
                *output.as_mut_ptr().add(index) = tokens[source];
            }
        }
    }

    /// Unpack channel-major scorer vectors `[B,H,1,K]` into `[B*K,H]` rows.
    #[kernel]
    pub fn laya_scorer_unpack(
        input: &[f32],
        mut rows: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
    ) {
        let index = thread::index_1d().get();
        if index < rows.len() {
            let row = index / hidden;
            let channel = index % hidden;
            let batch = row / options;
            let option = row % options;
            let source = (batch * hidden + channel) * options + option;
            // SAFETY: validated shape extents make both layouts equal-sized.
            unsafe {
                *rows.as_mut_ptr().add(index) = input[source];
            }
        }
    }

    /// Scatter option rows `[B*K,H]` to channel-major `[B,H,1,K]`.
    #[kernel]
    pub fn laya_scorer_pack_input_grad(
        rows: &[f32],
        mut output: DisjointSlice<f32>,
        hidden: usize,
        options: usize,
    ) {
        let index = thread::index_1d().get();
        if index < output.len() {
            let batch = index / (hidden * options);
            let rem = index - batch * hidden * options;
            let channel = rem / options;
            let option = rem % options;
            let source = (batch * options + option) * hidden + channel;
            // SAFETY: validated equal element counts bound each transpose index.
            unsafe {
                *output.as_mut_ptr().add(index) = rows[source];
            }
        }
    }

    /// Pack `[B*K]` option logits into `[B,1,1,K]` (identity in storage).
    #[kernel]
    pub fn laya_scorer_pack(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        if index < output.len() {
            // SAFETY: host validation guarantees equal input/output lengths.
            unsafe {
                *output.as_mut_ptr().add(index) = input[index];
            }
        }
    }

    /// LayaHead's masked attention. Skips invalid keys and writes a zero
    /// context for queries whose batch has no valid key. One thread owns one
    /// (batch, head, query) output slice and uses online softmax state.
    #[kernel]
    pub fn laya_head_attention_forward(
        qkv: &[f32],
        mask: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let head_start = h * head_dim;
        let query_row = b * tokens + tq;
        let q_base = query_row * 3 * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut acc = [0.0f32; 128];
        let mut d = 0usize;
        while d < head_dim {
            q[d] = qkv[q_base + d];
            d += 1;
        }
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut any_valid = false;
        let mut key = 0usize;
        while key < tokens {
            if mask[b * tokens + key] > 0.5 {
                any_valid = true;
                let key_row = b * tokens + key;
                let k_base = key_row * 3 * channels + channels + head_start;
                let mut dot = 0.0f32;
                d = 0;
                while d < head_dim {
                    dot += q[d] * qkv[k_base + d];
                    d += 1;
                }
                let score = dot * scale;
                let new_max = running_max.max(score);
                let correction = (running_max - new_max).exp();
                let probability = (score - new_max).exp();
                running_sum = running_sum * correction + probability;
                let v_base = key_row * 3 * channels + 2 * channels + head_start;
                d = 0;
                while d < head_dim {
                    acc[d] = acc[d] * correction + probability * qkv[v_base + d];
                    d += 1;
                }
                running_max = new_max;
            }
            key += 1;
        }
        let output_base = query_row * channels + head_start;
        d = 0;
        while d < head_dim {
            let value = if any_valid { acc[d] / running_sum } else { 0.0 };
            // SAFETY: every (batch, head, query) thread owns disjoint channels.
            unsafe {
                *output.as_mut_ptr().add(output_base + d) = value;
            }
            d += 1;
        }
    }

    /// One query row per thread computes dQ and D_i = dot(context_i, dO_i).
    /// The following key-row kernel consumes D_i to form dK and dV without atomics.
    #[kernel]
    pub fn flash_attention_backward_dq(
        qkv: &[f32],
        context: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        mut dqkv: DisjointSlice<f32>,
        mut row_dot: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let q_base = query_row * 3 * channels + head_start;
        let row_base = query_row * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut dq = [0.0f32; 128];
        let mut row_product = 0.0f32;
        let mut d = 0usize;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            q[d] = if raw_q.is_finite() {
                raw_q.max(-128.0).min(128.0)
            } else {
                0.0
            };
            row_product += output_grad[row_base + d] * context[row_base + d];
            dq[d] = 0.0;
            d += 1;
        }

        let lse_i = lse[query];
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_k = qkv[k_base + d];
                let k = if raw_k.is_finite() {
                    raw_k.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                dprob += output_grad[row_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse_i).exp();
            if active {
                let dscore = p * (dprob - row_product);
                d = 0;
                while d < head_dim {
                    let raw_k = qkv[k_base + d];
                    let k = if raw_k.is_finite() {
                        raw_k.max(-128.0).min(128.0)
                    } else {
                        0.0
                    };
                    dq[d] += dscore * k * scale;
                    d += 1;
                }
            }
            tk += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            let grad = if raw_q.is_finite() && raw_q.abs() < 128.0 {
                dq[d]
            } else {
                0.0
            };
            // SAFETY: each query/head thread owns this Q channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(query_row * 3 * channels + head_start + d) = grad;
            }
            d += 1;
        }
        // SAFETY: one thread owns each flattened (batch, head, query) scalar.
        unsafe {
            *row_dot.as_mut_ptr().add(query) = row_product;
        }
    }

    /// One key row per thread accumulates dK/dV across all query rows; every output
    /// row is owned by one thread, so the recompute path requires no atomics.
    #[kernel]
    pub fn flash_attention_backward_dkdv(
        qkv: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        row_dot: &[f32],
        mut dqkv: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let key = thread::index_1d().get();
        let key_count = batch * heads * tokens;
        if key >= key_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = key / tokens;
        let tk = key - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let key_row = b * tokens + tk;
        let head_start = h * head_dim;
        let k_base = key_row * 3 * channels + channels + head_start;
        let v_base = key_row * 3 * channels + 2 * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut k = [0.0f32; 128];
        let mut dk = [0.0f32; 128];
        let mut dv = [0.0f32; 128];
        let mut d = 0usize;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            k[d] = if raw_k.is_finite() {
                raw_k.max(-128.0).min(128.0)
            } else {
                0.0
            };
            dk[d] = 0.0;
            dv[d] = 0.0;
            d += 1;
        }

        let mut tq = 0usize;
        while tq < tokens {
            let query_row = b * tokens + tq;
            let query = (b * heads + h) * tokens + tq;
            let q_base = query_row * 3 * channels + head_start;
            let dy_base = query_row * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q * k[d];
                dprob += output_grad[dy_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse[query]).exp();
            let dscore = p * (dprob - row_dot[query]);
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dv[d] += p * output_grad[dy_base + d];
                if active {
                    dk[d] += dscore * q * scale;
                }
                d += 1;
            }
            tq += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            let dkey = if raw_k.is_finite() && raw_k.abs() < 128.0 {
                dk[d]
            } else {
                0.0
            };
            // SAFETY: each key/head thread owns one K and one V channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + channels + head_start + d) = dkey;
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + 2 * channels + head_start + d) = dv[d];
            }
            d += 1;
        }
    }

    /// One thread per softmax group of `group_size` contiguous elements
    /// (`n_groups = batch * groups`): max, exp + sum, normalise.
    #[kernel]
    pub fn softmax_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut max = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < group_size {
            let v = x[base + i];
            if v > max {
                max = v;
            }
            i += 1;
        }
        let mut sum = 0.0f32;
        i = 0;
        while i < group_size {
            let e = (x[base + i] - max).exp();
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`;
            // the host checks `out` covers `n_groups * group_size`.
            unsafe {
                *out.as_mut_ptr().add(base + i) = e;
            }
            sum += e;
            i += 1;
        }
        if sum > 0.0 {
            i = 0;
            while i < group_size {
                // SAFETY: as above; this thread already wrote the element.
                unsafe {
                    *out.as_mut_ptr().add(base + i) /= sum;
                }
                i += 1;
            }
        }
    }

    /// `dx = y * (dy - sum(y * dy))`, one thread per group.
    #[kernel]
    pub fn softmax_backward(
        y: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut dot = 0.0f32;
        let mut i = 0usize;
        while i < group_size {
            dot += y[base + i] * dy[base + i];
            i += 1;
        }
        i = 0;
        while i < group_size {
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`.
            unsafe {
                *dx.as_mut_ptr().add(base + i) = y[base + i] * (dy[base + i] - dot);
            }
            i += 1;
        }
    }

    /// CenterNet forward. NCHW class planes receive sigmoid; the four
    /// regression planes remain linear. Non-finite post-activation values are
    /// sanitized, with +inf class logits naturally saturating to one.
    #[kernel]
    pub fn centernet_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let channel = (index / spatial) % channels;
        let mut value = input[index];
        if channel < classes {
            value = 1.0 / (1.0 + (-value).exp());
        }
        if !value.is_finite() {
            value = 0.0;
        }
        // SAFETY: one thread owns each output element.
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Shared modern-YOLO activation in NCHW layout. Each element is sanitized;
    /// only the trailing objectness/class channels receive the bounded sigmoid.
    #[kernel]
    pub fn modern_yolo_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let channel = (index / spatial) % channels;
        let class_begin = channels - classes - 1;
        let mut value = input[index];
        if !value.is_finite() {
            value = 0.0;
        }
        if channel >= class_begin {
            value = 1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp());
        }
        // SAFETY: one thread owns each output element.
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Per-pixel binary sigmoid or channel-strided softmax for segmentation.
    /// One thread owns a complete pixel's class row and writes disjoint NCHW
    /// slots across all classes.
    #[kernel]
    pub fn segmentation_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
        temperature: f32,
    ) {
        let pixel = thread::index_1d().get();
        let pixels = output.len() / channels;
        if pixel >= pixels {
            return;
        }
        let batch = pixel / spatial;
        let location = pixel % spatial;
        let base = batch * channels * spatial;
        if classes == 1 {
            let mut value = input[base + location];
            if !value.is_finite() {
                value = 0.0;
            }
            value = 1.0 / (1.0 + (-value).exp());
            // SAFETY: this pixel thread owns this binary output slot.
            unsafe {
                *output.as_mut_ptr().add(base + location) = value;
            }
            return;
        }
        let temp = if temperature > 1.0e-6 {
            temperature
        } else {
            1.0e-6
        };
        let mut maximum = f32::NEG_INFINITY;
        let mut class = 0usize;
        while class < classes {
            let value = input[base + class * spatial + location] / temp;
            if value > maximum {
                maximum = value;
            }
            class += 1;
        }
        let mut sum = 0.0f32;
        class = 0;
        while class < classes {
            let index = base + class * spatial + location;
            let value = (input[index] / temp - maximum).exp();
            unsafe {
                *output.as_mut_ptr().add(index) = value;
            }
            sum += value;
            class += 1;
        }
        class = 0;
        while class < classes {
            let index = base + class * spatial + location;
            unsafe {
                *output.as_mut_ptr().add(index) /= sum;
            }
            class += 1;
        }
    }

    /// One thread owns a complete `[jev_ce]` group and writes its softmax row.
    #[kernel]
    pub fn jev_ce_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        groups: usize,
        candidates: usize,
        temperature: f32,
        channel_major: u32,
    ) {
        let group_index = thread::index_1d().get();
        if group_index >= output.len() / candidates {
            return;
        }
        let batch = group_index / groups;
        let group = group_index % groups;
        let base = batch * groups * candidates;
        let temp = if temperature > 1.0e-6 {
            temperature
        } else {
            1.0e-6
        };
        let index = |candidate: usize| {
            if channel_major != 0 {
                base + candidate * groups + group
            } else {
                base + group * candidates + candidate
            }
        };
        let mut maximum = f32::NEG_INFINITY;
        let mut candidate = 0usize;
        while candidate < candidates {
            let value = input[index(candidate)];
            if value > maximum {
                maximum = value;
            }
            candidate += 1;
        }
        let mut sum = 0.0f32;
        candidate = 0;
        while candidate < candidates {
            let value = ((input[index(candidate)] - maximum) / temp).exp();
            unsafe {
                *output.as_mut_ptr().add(index(candidate)) = value;
            }
            sum += value;
            candidate += 1;
        }
        candidate = 0;
        while candidate < candidates {
            unsafe {
                *output.as_mut_ptr().add(index(candidate)) /= sum;
            }
            candidate += 1;
        }
    }

    /// One thread owns one sample row and writes its stable K-way softmax.
    #[kernel]
    pub fn laya_loss_forward(input: &[f32], mut output: DisjointSlice<f32>, max_options: usize) {
        let batch = thread::index_1d().get();
        let batches = output.len() / max_options;
        if batch >= batches {
            return;
        }
        let base = batch * max_options;
        let mut maximum = -1.0e300f64;
        let mut option = 0usize;
        while option < max_options {
            maximum = maximum.max(input[base + option] as f64);
            option += 1;
        }
        let mut sum = 0.0f64;
        option = 0;
        while option < max_options {
            sum += (input[base + option] as f64 - maximum).exp();
            option += 1;
        }
        option = 0;
        while option < max_options {
            let index = base + option;
            let probability = (input[index] as f64 - maximum).exp() / sum;
            unsafe {
                *output.as_mut_ptr().add(index) = probability as f32;
            }
            option += 1;
        }
    }

    /// Region forward. One thread owns a batch/location/anchor group and
    /// writes its interleaved entries after objectness sigmoid and optional
    /// class softmax.
    #[kernel]
    pub fn region_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        spatial: usize,
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: u32,
    ) {
        let group = thread::index_1d().get();
        let groups_per_batch = spatial * anchors;
        if group >= output.len() / entries {
            return;
        }
        let batch = group / groups_per_batch;
        let rem = group % groups_per_batch;
        let location = rem / anchors;
        let anchor = rem % anchors;
        let channels = anchors * entries;
        let source = |entry: usize| {
            input[(batch * channels + anchor * entries + entry) * spatial + location]
        };
        let destination = (batch * spatial + location) * channels + anchor * entries;
        let mut entry = 0usize;
        while entry < entries {
            // SAFETY: this group exclusively owns its `entries`-wide output row.
            unsafe {
                *output.as_mut_ptr().add(destination + entry) = source(entry);
            }
            entry += 1;
        }
        let objectness = source(4);
        let sigmoid = 1.0 / (1.0 + (-objectness).exp());
        unsafe {
            *output.as_mut_ptr().add(destination + 4) = sigmoid;
        }
        if softmax_classes != 0 {
            let mut max_value = f32::NEG_INFINITY;
            entry = 0;
            while entry < classes {
                let value = source(5 + entry);
                if value > max_value {
                    max_value = value;
                }
                entry += 1;
            }
            let mut sum = 0.0f32;
            entry = 0;
            while entry < classes {
                let value = (source(5 + entry) - max_value).exp();
                unsafe {
                    *output.as_mut_ptr().add(destination + 5 + entry) = value;
                }
                sum += value;
                entry += 1;
            }
            if sum > 0.0 {
                entry = 0;
                while entry < classes {
                    unsafe {
                        *output.as_mut_ptr().add(destination + 5 + entry) /= sum;
                    }
                    entry += 1;
                }
            }
        }
    }

    /// Exact adjoint of `region_forward`, writing channel-major NCHW deltas.
    #[kernel]
    pub fn region_backward(
        output: &[f32],
        output_grad: &[f32],
        mut input_grad: DisjointSlice<f32>,
        spatial: usize,
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: u32,
    ) {
        let group = thread::index_1d().get();
        let groups_per_batch = spatial * anchors;
        if group >= input_grad.len() / entries {
            return;
        }
        let batch = group / groups_per_batch;
        let rem = group % groups_per_batch;
        let location = rem / anchors;
        let anchor = rem % anchors;
        let channels = anchors * entries;
        let base = (batch * spatial + location) * channels + anchor * entries;
        let mut dot = 0.0f32;
        if softmax_classes != 0 {
            let mut c = 0usize;
            while c < classes {
                dot += output[base + 5 + c] * output_grad[base + 5 + c];
                c += 1;
            }
        }
        let mut entry = 0usize;
        while entry < entries {
            let value = if entry == 4 {
                let p = output[base + 4];
                output_grad[base + 4] * p * (1.0 - p)
            } else if softmax_classes != 0 && entry >= 5 && entry < 5 + classes {
                let p = output[base + entry];
                p * (output_grad[base + entry] - dot)
            } else {
                output_grad[base + entry]
            };
            let target = (batch * channels + anchor * entries + entry) * spatial + location;
            // SAFETY: each batch/location/anchor/entry maps to one unique input slot.
            unsafe {
                *input_grad.as_mut_ptr().add(target) = value;
            }
            entry += 1;
        }
    }

    /// Big `[b, cb, hb, wb]` -> small `[b, cb*s*s, hs, ws]`: one thread per
    /// small element gathers its big source.
    #[kernel]
    pub fn reorg_to_small(
        big: &[f32],
        mut small: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= small.len() {
            return;
        }
        let cs = cb * stride * stride;
        let plane = hs * ws;
        let bk = i / plane;
        let r = i % plane;
        let b = bk / cs;
        let k = bk % cs;
        let c2 = k % cb;
        let off = k / cb;
        let h2 = (r / ws) * stride + off / stride;
        let w2 = (r % ws) * stride + off % stride;
        if let Some(o) = small.get_mut(idx) {
            *o = big[((b * cb + c2) * hb + h2) * wb + w2];
        }
    }

    /// Small -> big: one thread per big element reads its small source, or
    /// writes 0 on the border not covered when `hb % stride != 0`.
    #[kernel]
    pub fn reorg_to_big(
        small: &[f32],
        mut big: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= big.len() {
            return;
        }
        let plane = hb * wb;
        let bc = i / plane;
        let r = i % plane;
        let b = bc / cb;
        let c2 = bc % cb;
        let h2 = r / wb;
        let w2 = r % wb;
        let j = h2 / stride;
        let x = w2 / stride;
        let value = if j >= hs || x >= ws {
            0.0
        } else {
            let off = (h2 % stride) * stride + w2 % stride;
            let k = off * cb + c2;
            small[((b * cb * stride * stride + k) * hs + j) * ws + x]
        };
        if let Some(o) = big.get_mut(idx) {
            *o = value;
        }
    }

    /// `out[b, j*rows + i, s] = x[b, i*cols + j, s]`: one thread per output element.
    #[kernel]
    pub fn channel_permute(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        len: usize,
    ) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let per = rows * cols * len;
            let b = g / per;
            let r = g % per;
            let p = r / len; // output channel j*rows + i
            let s = r % len;
            let j = p / rows;
            let i = p % rows;
            *o = x[b * per + (i * cols + j) * len + s];
        }
    }

    /// `mask[i] = 0` when `dropout_unit(seed, i) < probability` else `scale`;
    /// `out[i] = x[i] * mask[i]`.
    #[kernel]
    pub fn dropout_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        probability: f32,
        scale: f32,
        seed: u64,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = if dropout_unit(seed, i as u64) < probability {
            0.0
        } else {
            scale
        };
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }

    /// `dx[i] = dy[i] * mask[i]`.
    #[kernel]
    pub fn dropout_backward(dy: &[f32], mask: &[f32], mut dx: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = dx.get_mut(idx) {
            *o = dy[i] * mask[i];
        }
    }

    /// DropBlock stage 1, one thread per `(sample, channel)` plane: picks the
    /// cell with the lowest quantised random value below `prob`
    /// (`darknet_cuda::host_ops::dropblock_corner`), writes the plane of
    /// `keep` (`1`, `0` inside the block) and `dropped[plane]` (`bs^2` or 0).
    #[kernel]
    pub fn dropblock_plane(
        mut keep: DisjointSlice<f32>,
        mut dropped: DisjointSlice<f32>,
        seed: u64,
        w: usize,
        h: usize,
        prob: f32,
        bs: usize,
    ) {
        let idx = thread::index_1d();
        let plane = idx.get();
        let spatial = w * h;
        if plane >= dropped.len() {
            return;
        }
        let mut best = 1_000_000i32;
        let mut cell = usize::MAX;
        let mut i = 0usize;
        while i < spatial {
            let u = dropout_unit(seed, (plane * spatial + i) as u64);
            if u < prob {
                let q = (u * 1_000_000.0) as i32;
                if q < best {
                    best = q;
                    cell = i;
                }
            }
            i += 1;
        }
        let mut bx = 0usize;
        let mut by = 0usize;
        if cell != usize::MAX {
            bx = cell % w;
            by = cell / w;
            if bx > w - bs {
                bx -= w - bs;
            }
            if by > h - bs {
                by -= h - bs;
            }
            if bx > w - bs {
                bx = w - bs;
            }
            if by > h - bs {
                by = h - bs;
            }
        }
        // SAFETY: this thread owns plane `plane` of `keep` (`spatial` cells)
        // and slot `plane` of `dropped`; the host allocates both exactly.
        unsafe {
            let base = keep.as_mut_ptr().add(plane * spatial);
            let mut j = 0usize;
            while j < spatial {
                let y = j / w;
                let x = j % w;
                let inside = cell != usize::MAX && x >= bx && x < bx + bs && y >= by && y < by + bs;
                *base.add(j) = if inside { 0.0 } else { 1.0 };
                j += 1;
            }
            *dropped.as_mut_ptr().add(plane) = if cell != usize::MAX {
                (bs * bs) as f32
            } else {
                0.0
            };
        }
    }

    /// DropBlock stage 2, one thread per sample: sums `dropped` over the
    /// sample's `channels` planes in order and writes
    /// `host_ops::dropblock_scale(sum, outputs)`.
    #[kernel]
    pub fn dropblock_scales(
        dropped: &[f32],
        mut scales: DisjointSlice<f32>,
        channels: usize,
        outputs: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        let mut sum = 0.0f32;
        let mut c = 0usize;
        while c < channels && b * channels + c < dropped.len() {
            sum += dropped[b * channels + c];
            c += 1;
        }
        let keep = 1.0 - sum / outputs as f32;
        if let Some(o) = scales.get_mut(idx) {
            *o = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        }
    }

    /// DropBlock stage 3, one thread per element: `mask = keep * scale[b]`,
    /// `out = x * mask`; `per_batch = channels * h * w`.
    #[kernel]
    pub fn dropblock_apply(
        x: &[f32],
        keep: &[f32],
        scales: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        per_batch: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = keep[i] * scales[i / per_batch];
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }

    /// Darknet `constrain_ongpu`: `out = min(limit, max(-limit, x))`, NaN -> `-limit`.
    #[kernel]
    pub fn constrain(x: &[f32], mut out: DisjointSlice<f32>, limit: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            let mut v = x[i];
            if !(v > -limit) {
                v = -limit;
            }
            if v > limit {
                v = limit;
            }
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Darknet `gradient_centralization_gpu`, one thread per filter of
    /// `f_size` contiguous values (sequential sum, = the host reference); a
    /// plain copy when `enabled == 0`.
    #[kernel]
    pub fn gradient_centralize(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        f_size: usize,
        filters: usize,
        enabled: u32,
    ) {
        let f = thread::index_1d().get();
        if f >= filters {
            return;
        }
        let base = f * f_size;
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < f_size {
            sum += x[base + i];
            i += 1;
        }
        let mean = if enabled != 0 {
            sum / f_size as f32
        } else {
            0.0
        };
        // SAFETY: this thread owns the `f_size` values of filter `f`.
        unsafe {
            let dst = out.as_mut_ptr().add(base);
            let mut i = 0usize;
            while i < f_size {
                *dst.add(i) = x[base + i] - mean;
                i += 1;
            }
        }
    }

    /// `out[i] = scale[si(i)] * src[i]`, `si` = `[batch, hw]` (`scale_wh != 0`)
    /// or `[batch, channels]` index of element `i`.
    #[kernel]
    pub fn scale_channels_forward(
        scale: &[f32],
        src: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = out.get_mut(idx) {
            *o = scale[si] * src[i];
        }
    }

    /// `d_source[i] = delta[i] * scale[si(i)]`.
    #[kernel]
    pub fn scale_channels_dsource(
        delta: &[f32],
        scale: &[f32],
        mut d_src: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = d_src.get_mut(idx) {
            *o = delta[i] * scale[si];
        }
    }

    /// `d_scale[j] = sum delta * src` over the elements sharing scale `j`;
    /// one thread per scale element, fixed summation order.
    #[kernel]
    pub fn scale_channels_dscale(
        delta: &[f32],
        src: &[f32],
        mut d_scale: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= d_scale.len() {
            return;
        }
        let mut sum = 0.0f32;
        if scale_wh != 0 {
            let b = j / hw;
            let s = j % hw;
            let mut c = 0usize;
            while c < channels {
                let i = (b * channels + c) * hw + s;
                sum += delta[i] * src[i];
                c += 1;
            }
        } else {
            let base = j * hw;
            let mut s = 0usize;
            while s < hw {
                sum += delta[base + s] * src[base + s];
                s += 1;
            }
        }
        if let Some(o) = d_scale.get_mut(idx) {
            *o = sum;
        }
    }

    /// One thread per shortcut coefficient slot. Each thread traverses its
    /// deterministic batch/feature range and reduces `delta * source`; a short
    /// source contributes only over its available flat prefix.
    #[kernel]
    pub fn shortcut_weight_gradient(
        delta: &[f32],
        source: &[f32],
        mut gradient: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
    ) {
        let idx = thread::index_1d();
        let slot = idx.get();
        if slot >= layer_step {
            return;
        }
        let span = per_sample / layer_step;
        let start = slot * span;
        let end = (start + span).min(per_sample).min(source_per_sample);
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut e = start;
            while e < end {
                sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                e += 1;
            }
            b += 1;
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = sum;
        }
    }

    /// Update one source row of `[input][slot]` `delta * source` products.
    #[kernel]
    pub fn shortcut_weight_gradient_accumulate(
        accumulated: &[f32],
        delta: &[f32],
        source: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let mut value = accumulated[i];
        if i / layer_step == source_index && source_index < num_inputs {
            let slot = i % layer_step;
            let span = per_sample / layer_step;
            let start = slot * span;
            let end = (start + span).min(per_sample).min(source_per_sample);
            let mut sum = 0.0f32;
            let mut b = 0usize;
            while b < batch {
                let mut e = start;
                while e < end {
                    sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                    e += 1;
                }
                b += 1;
            }
            value += sum;
        }
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    /// Exact per-slot normalization chain for shortcut coefficient gradients.
    /// `normalization`: 0 none, 1 relu, 2 softmax; `cpp_grad` copies raw z.
    #[kernel]
    pub fn shortcut_weight_normalized_gradient(
        raw: &[f32],
        products: &[f32],
        mut gradient: DisjointSlice<f32>,
        num_inputs: usize,
        layer_step: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let slot = i % layer_step;
        let mut value = products[i];
        if normalization == 1 && cpp_grad == 0 {
            let mut positive_sum = 0.0f32;
            let mut dot = 0.0f32;
            let mut input = 0usize;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let positive = raw[j].max(0.0);
                positive_sum += positive;
                dot += products[j] * positive;
                input += 1;
            }
            let sum = 0.0001f32 + positive_sum;
            value = if raw[i] > 0.0 {
                products[i] / sum - dot / (sum * sum)
            } else {
                0.0
            };
        } else if normalization == 2 && cpp_grad == 0 {
            let mut maximum = f32::NEG_INFINITY;
            let mut input = 0usize;
            while input < num_inputs {
                let raw_value = raw[input * layer_step + slot];
                if raw_value > maximum || input == 0 {
                    maximum = raw_value;
                }
                input += 1;
            }
            let mut exp_sum = 0.0f32;
            input = 0;
            while input < num_inputs {
                exp_sum += (raw[input * layer_step + slot] - maximum).exp();
                input += 1;
            }
            let sum = 0.0001f32 + exp_sum;
            let mut dot = 0.0f32;
            input = 0;
            while input < num_inputs {
                let j = input * layer_step + slot;
                let effective = (raw[j] - maximum).exp() / sum;
                dot += products[j] * effective;
                input += 1;
            }
            let effective = (raw[i] - maximum).exp() / sum;
            value = effective * (products[i] - dot);
        }
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }

    /// NCHW channel normalization at one `(batch, spatial)` position per thread.
    /// `normalization`: 0 ReLU-normalize, 1 softmax with zero shift, 2 softmax
    /// with channel-max subtraction.
    #[kernel]
    pub fn normalize_channels(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let value = if normalization == 0 {
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                let x = input[base + k * spatial];
                if x > 0.0 {
                    sum += x;
                }
                k += 1;
            }
            let x = input[i];
            if x > 0.0 { x / sum } else { 0.0 }
        } else {
            let maximum = if normalization == 2 {
                let mut maximum = -f32::MAX;
                let mut k = 0usize;
                while k < channels {
                    let x = input[base + k * spatial];
                    if x > maximum || k == 0 {
                        maximum = x;
                    }
                    k += 1;
                }
                maximum
            } else {
                0.0
            };
            let mut sum = 0.0001f32;
            let mut k = 0usize;
            while k < channels {
                sum += (input[base + k * spatial] - maximum).exp();
                k += 1;
            }
            let value = (input[i] - maximum).exp() / sum;
            if value.is_finite() { value } else { 0.0 }
        };
        if let Some(o) = output.get_mut(idx) {
            *o = value;
        }
    }

    #[device]
    fn eml_softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            x.exp()
        } else {
            x.exp().ln_1p()
        }
    }

    #[device]
    fn eml_sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    /// EML convolution's elementwise combination of its parallel conv outputs.
    #[kernel]
    pub fn eml_convolutional_forward(
        x_branch: &[f32],
        self_branch: &[f32],
        residual: &[f32],
        mut output: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
        has_residual: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let x = x_branch[i].max(-clamp).min(clamp);
        let y = self_branch[i];
        let mut value = scale * (x.exp() - (eml_softplus(y) + eps).ln());
        if has_residual != 0 {
            value += residual[i];
        }
        if let Some(out) = output.get_mut(idx) {
            *out = value;
        }
    }

    /// Exact EML branch adjoints, with the source's strict clamp gate.
    #[kernel]
    pub fn eml_convolutional_backward(
        x_branch: &[f32],
        self_branch: &[f32],
        output_grad: &[f32],
        mut x_gradient: DisjointSlice<f32>,
        mut self_gradient: DisjointSlice<f32>,
        clamp: f32,
        eps: f32,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x_gradient.len() {
            return;
        }
        let x_value = x_branch[i];
        let dx = if x_value > -clamp && x_value < clamp {
            x_value.max(-clamp).min(clamp).exp()
        } else {
            0.0
        };
        let y = self_branch[i];
        let dy = -eml_sigmoid(y) / (eml_softplus(y) + eps);
        let scaled_delta = output_grad[i] * scale;
        // SAFETY: every thread writes only element `i` in both output arrays.
        unsafe {
            *x_gradient.as_mut_ptr().add(i) = scaled_delta * dx;
            *self_gradient.as_mut_ptr().add(i) = scaled_delta * dy;
        }
    }

    /// Exact channel-normalization adjoint or the C++ CUDA diagonal shortcut.
    #[kernel]
    pub fn normalize_channels_backward(
        activation_input: &[f32],
        output: &[f32],
        delta: &[f32],
        mut gradient: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        normalization: u32,
        cpp_grad: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= gradient.len() {
            return;
        }
        let position = i % spatial;
        let batch = i / (channels * spatial);
        let base = batch * channels * spatial + position;
        let y = output[i];
        let value = if cpp_grad != 0 {
            if normalization == 0 {
                if y > 0.0 { delta[i] * y } else { delta[i] }
            } else {
                let value = delta[i] * y * (1.0 - y);
                if value.is_finite() { value } else { 0.0 }
            }
        } else {
            let mut dot = 0.0f32;
            let mut k = 0usize;
            while k < channels {
                let j = base + k * spatial;
                dot += output[j] * delta[j];
                k += 1;
            }
            if normalization == 0 {
                let mut sum = 0.0001f32;
                k = 0;
                while k < channels {
                    let x = activation_input[base + k * spatial];
                    if x > 0.0 {
                        sum += x;
                    }
                    k += 1;
                }
                if activation_input[i] > 0.0 {
                    (delta[i] - dot) / sum
                } else {
                    0.0
                }
            } else if normalization == 2 {
                // epsilon outside exp(x-max) leaves a max-chain term.
                // First maximum is the selected subgradient at exact ties.
                let mut maximum = activation_input[base];
                let mut first_max = 0usize;
                let mut correction = 0.0f32;
                k = 0;
                while k < channels {
                    let j = base + k * spatial;
                    if activation_input[j] > maximum {
                        maximum = activation_input[j];
                        first_max = k;
                    }
                    correction += output[j] * (delta[j] - dot);
                    k += 1;
                }
                let g = y * (delta[i] - dot);
                if (i / spatial) % channels == first_max {
                    g - correction
                } else {
                    g
                }
            } else {
                y * (delta[i] - dot)
            }
        };
        if let Some(o) = gradient.get_mut(idx) {
            *o = value;
        }
    }
}

/// Fused residual + bias epilogue for a transformer block:
/// `out[i] = residual[i] + y[i] + bias[i % cols]`.
///
/// The bias is per **column**: the row index is the token, so it must not shift
/// the offset.  `has_bias = 0` makes it a plain residual add.
#[cuda_module]
pub mod residual {
    use super::*;
    use cuda_device::{kernel, thread};

    #[kernel]
    pub fn residual_bias(
        residual: &[f32],
        y: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        cols: usize,
        has_bias: u32,
    ) {
        let i = thread::index_1d();
        let idx = i.get();
        if idx >= out.len() {
            return;
        }
        let b = if has_bias != 0 { bias[idx % cols] } else { 0.0 };
        if let Some(slot) = out.get_mut(i) {
            *slot = residual[idx] + y[idx] + b;
        }
    }
}
