//! Dense-head loss and decode kernels: CenterNet, modern YOLO, segmentation, JevCe, LayaLoss and region.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

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

    /// One thread owns one segmentation sample: computes the per-element
    /// cross-entropy and `truth - prediction` delta, applies `lambda_ce` and
    /// the correct-pixel gate weight, and writes per-sample loss and suppressed
    /// pixel counts for the host reduction.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn segmentation_loss(
        output: &[f32],
        truth: &[f32],
        mut delta: DisjointSlice<f32>,
        mut partial_loss: DisjointSlice<f32>,
        mut suppressed: DisjointSlice<f32>,
        batch: usize,
        classes: usize,
        spatial: usize,
        softmax_loss: u32,
        scale: f32,
        gate_strength: f32,
        t_correct: f32,
    ) {
        let sample = thread::index_1d().get();
        if sample >= batch {
            return;
        }
        let base = sample * classes * spatial;
        let mut sample_loss = 0.0f32;
        let mut suppressed_count = 0usize;
        let mut location = 0usize;
        while location < spatial {
            let (is_match, p_y) = if classes == 1 {
                let p = output[base + location];
                let y = truth[base + location];
                ((p > 0.5) == (y > 0.5), if y > 0.5 { p } else { 1.0 - p })
            } else {
                let mut y_class = 0usize;
                let mut pred_class = 0usize;
                let mut max_truth = truth[base + location];
                let mut max_pred = output[base + location];
                let mut class = 1usize;
                while class < classes {
                    let index = base + class * spatial + location;
                    if truth[index] > max_truth {
                        max_truth = truth[index];
                        y_class = class;
                    }
                    if output[index] > max_pred {
                        max_pred = output[index];
                        pred_class = class;
                    }
                    class += 1;
                }
                (pred_class == y_class, output[base + y_class * spatial + location])
            };
            let gated = gate_strength > 0.0 && is_match && p_y >= t_correct;
            let weight = if gated { 1.0 - gate_strength } else { 1.0 };
            if gated {
                suppressed_count += 1;
            }
            let mut class = 0usize;
            while class < classes {
                let index = base + class * spatial + location;
                let t = truth[index];
                let p = output[index];
                let (error, d) = if softmax_loss != 0 {
                    let pc = if p < 1.0e-12 {
                        1.0e-12
                    } else if p > 1.0 {
                        1.0
                    } else {
                        p
                    };
                    (if t != 0.0 { -pc.ln() } else { 0.0 }, t - p)
                } else {
                    let pc = if p < 1.0e-12 {
                        1.0e-12
                    } else if p > 1.0 - 1.0e-12 {
                        1.0 - 1.0e-12
                    } else {
                        p
                    };
                    (
                        -t * pc.ln() - (1.0 - t) * (1.0 - pc).ln(),
                        t - p,
                    )
                };
                let mut l = error * scale * weight;
                if !l.is_finite() {
                    l = 0.0;
                }
                let d = d * scale * weight;
                unsafe {
                    *delta.as_mut_ptr().add(index) = d;
                }
                sample_loss += l;
                class += 1;
            }
            location += 1;
        }
        if let Some(slot) = partial_loss.get_mut(thread::index_1d()) {
            *slot = sample_loss;
        }
        if let Some(slot) = suppressed.get_mut(thread::index_1d()) {
            *slot = suppressed_count as f32;
        }
    }

    /// Applies the gate compensation and `max_delta` clip to a segmentation
    /// delta buffer in place.
    #[kernel]
    pub fn segmentation_scale_delta(
        mut delta: DisjointSlice<f32>,
        compensation: f32,
        max_delta: f32,
        clipping: u32,
    ) {
        let idx = thread::index_1d();
        if let Some(slot) = delta.get_mut(idx) {
            let mut value = *slot * compensation;
            if !value.is_finite() {
                value = 0.0;
            }
            if clipping != 0 {
                if value > max_delta {
                    value = max_delta;
                } else if value < -max_delta {
                    value = -max_delta;
                }
            }
            *slot = value;
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

    /// One thread owns one `[jev_ce]` group: computes its softmax,
    /// soft-target cross-entropy, and descent-direction delta.  `losses` and
    /// `active` have one slot per sample/group; masked empty groups leave the
    /// delta zero and mark `active` zero.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn jev_ce_loss(
        input: &[f32],
        truth: &[f32],
        mut delta: DisjointSlice<f32>,
        mut losses: DisjointSlice<f32>,
        mut active: DisjointSlice<f32>,
        batch: usize,
        groups: usize,
        candidates: usize,
        temperature: f32,
        normalize: u32,
        ignore_empty: u32,
        label_smoothing: f32,
        scale: f32,
        channel_major: u32,
    ) {
        let group_index = thread::index_1d().get();
        if group_index >= batch * groups {
            return;
        }
        let sample = group_index / groups;
        let group = group_index % groups;
        let base = sample * groups * candidates;
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
        let mut target_sum = 0.0f32;
        let mut candidate = 0usize;
        while candidate < candidates {
            target_sum += truth[index(candidate)];
            candidate += 1;
        }
        let empty = target_sum <= 1.0e-8;
        if empty && ignore_empty != 0 {
            candidate = 0;
            while candidate < candidates {
                unsafe {
                    *delta.as_mut_ptr().add(index(candidate)) = 0.0;
                }
                candidate += 1;
            }
            if let Some(slot) = losses.get_mut(thread::index_1d()) {
                *slot = 0.0;
            }
            if let Some(slot) = active.get_mut(thread::index_1d()) {
                *slot = 0.0;
            }
            return;
        }

        let mut maximum = f32::NEG_INFINITY;
        candidate = 0;
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
            sum += ((input[index(candidate)] - maximum) / temp).exp();
            candidate += 1;
        }
        let eps = if label_smoothing > 0.0 {
            label_smoothing.min(0.999)
        } else {
            0.0
        };
        let uniform = 1.0 / candidates as f32;
        let mut loss = 0.0f32;
        candidate = 0;
        while candidate < candidates {
            let prob = ((input[index(candidate)] - maximum) / temp).exp() / sum;
            let mut target = if empty {
                uniform
            } else {
                truth[index(candidate)] / target_sum
            };
            if eps > 0.0 {
                target = (1.0 - eps) * target + eps * uniform;
            }
            loss += -target * prob.max(1.0e-12).ln();
            unsafe {
                *delta.as_mut_ptr().add(index(candidate)) = (target - prob) / temp;
            }
            candidate += 1;
        }
        if normalize != 0 {
            let mut sample_active = 0usize;
            let mut other_group = 0usize;
            while other_group < groups {
                let mut other_sum = 0.0f32;
                let mut other_candidate = 0usize;
                while other_candidate < candidates {
                    let other_index = if channel_major != 0 {
                        base + other_candidate * groups + other_group
                    } else {
                        base + other_group * candidates + other_candidate
                    };
                    other_sum += truth[other_index];
                    other_candidate += 1;
                }
                if other_sum > 1.0e-8 || ignore_empty == 0 {
                    sample_active += 1;
                }
                other_group += 1;
            }
            let inv = 1.0 / sample_active.max(1) as f32;
            candidate = 0;
            while candidate < candidates {
                unsafe {
                    *delta.as_mut_ptr().add(index(candidate)) *= inv;
                }
                candidate += 1;
            }
        }
        candidate = 0;
        while candidate < candidates {
            unsafe {
                *delta.as_mut_ptr().add(index(candidate)) *= scale;
            }
            candidate += 1;
        }
        if let Some(slot) = losses.get_mut(thread::index_1d()) {
            *slot = loss;
        }
        if let Some(slot) = active.get_mut(thread::index_1d()) {
            *slot = 1.0;
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
}
