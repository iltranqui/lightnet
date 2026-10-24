//! WMHF kernels: four-direction EMA scan, high-frequency branch, split/concat, gate and Haar DWT.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

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
}
