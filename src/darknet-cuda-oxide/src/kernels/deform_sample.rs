//! Native deformable bilinear gather (wave 1 of the deformable layer family).
//!
//! Contract and semantics: `darknet_cuda::host_ops::bilinear_gather`, which is
//! itself derived from `darknet_core::deform_conv::bilinear_sample`. Two
//! conventions must be reproduced exactly:
//!
//! * far-outside samples (`y < -1 || y > H || x < -1 || x > W`) yield a zero
//!   value **and** zero coordinate gradients — the C++ early-out, not the
//!   mathematical bilinear extension;
//! * integer coordinates use the floor-selected (right-sided) cell.
//!
//! The forward owns each output element. The backward owns each sample cell's
//! coordinate gradients, but the input-feature gradient is a scatter with
//! arbitrary coordinates, so it uses float atomics — the same documented
//! trade-off as `resample::crop_bilinear_backward`: the result set is exact,
//! the summation order is not bit-for-bit deterministic.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// `floor(v)` as i64 without a libdevice `floor`.
#[device]
pub fn deform_floor_i64(v: f32) -> i64 {
    let t = v as i64;
    if (t as f32) > v { t - 1 } else { t }
}

#[device]
pub fn deform_in_bounds(v: i64, extent: usize) -> bool {
    v >= 0 && v < extent as i64
}

/// Relaxed device-scope `*p += v`.
///
/// # Safety
/// `p` must be a live, aligned device f32 that is only accessed atomically
/// while the kernel runs.
#[device]
pub unsafe fn deform_atomic_add_f32(p: *mut f32, v: f32) {
    let cell = unsafe { DeviceAtomicF32::from_ptr(p) };
    let _old = cell.fetch_add(v, AtomicOrdering::Relaxed);
}

/// Bilinear sample of one channel at `(y, x)`, replicating
/// `darknet_core::deform_conv::bilinear_sample`: far-outside samples are
/// zero, corners outside the feature map contribute nothing, and integer
/// coordinates use the floor-selected cell.
#[device]
pub fn deform_sample_at(
    input: &[f32],
    channel_base: usize,
    in_h: usize,
    in_w: usize,
    y: f32,
    x: f32,
) -> f32 {
    if y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32 {
        return 0.0;
    }
    let y0 = deform_floor_i64(y);
    let x0 = deform_floor_i64(x);
    let ly = y - y0 as f32;
    let lx = x - x0 as f32;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;
    let mut value = 0.0f32;
    if deform_in_bounds(y0, in_h) && deform_in_bounds(x0, in_w) {
        value += hy * hx * input[channel_base + y0 as usize * in_w + x0 as usize];
    }
    if deform_in_bounds(y0, in_h) && deform_in_bounds(x0 + 1, in_w) {
        value += hy * lx * input[channel_base + y0 as usize * in_w + (x0 + 1) as usize];
    }
    if deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0, in_w) {
        value += ly * hx * input[channel_base + (y0 + 1) as usize * in_w + x0 as usize];
    }
    if deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0 + 1, in_w) {
        value += ly * lx * input[channel_base + (y0 + 1) as usize * in_w + (x0 + 1) as usize];
    }
    value
}


/// Source `sanitize`: finite -> clamp to +/-10000, non-finite -> 0.
#[device]
pub fn deform_sanitize(v: f32) -> f32 {
    if v.is_finite() {
        if v > 10_000.0f32 {
            10_000.0f32
        } else if v < -10_000.0f32 {
            -10_000.0f32
        } else {
            v
        }
    } else {
        0.0
    }
}

/// Source `stable_sigmoid` probability. `value` is the sanitized logit, so
/// the non-finite branch is unreachable in practice but kept for fidelity.
#[device]
pub fn deform_stable_sigmoid(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    let clipped = if value > 80.0 {
        80.0
    } else if value < -80.0 {
        -80.0
    } else {
        value
    };
    if clipped >= 0.0 {
        1.0 / (1.0 + (-clipped).exp())
    } else {
        let z = clipped.exp();
        z / (1.0 + z)
    }
}

/// Source `sanitize` slope: 1 strictly inside the band, else 0.
#[device]
pub fn deform_sanitize_slope(value: f32) -> f32 {
if value.is_finite() && value > -10_000.0f32 && value < 10_000.0f32 {
    1.0
} else {
    0.0
}
}

/// Slope of `centered_lhtan` including its outer `sanitize`.
#[device]
pub fn deform_centered_lhtan_slope(value: f32, max_offset: f32) -> f32 {
let scaled = value / (2.0 * max_offset) + 0.5;
let mapped = if scaled < 0.0 {
    0.001 * scaled
} else if scaled > 1.0 {
    1.0 + 0.001 * (scaled - 1.0)
} else {
    scaled
};
let slope_scaled = if scaled < 0.0 || scaled > 1.0 { 0.001 } else { 1.0 };
deform_sanitize_slope((2.0 * mapped - 1.0) * max_offset) * slope_scaled
}

/// Slope of `stable_sigmoid(sanitize(logit))`.
#[device]
pub fn deform_mask_sigmoid_slope(logit: f32) -> f32 {
let sanitized = deform_sanitize(logit);
if !(sanitized > -80.0 && sanitized < 80.0) {
    return 0.0;
}
let p = deform_stable_sigmoid(sanitized);
p * (1.0 - p) * deform_sanitize_slope(logit)
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Main `[deform_conv]` accumulation: per `(b, filter, oy, ox)` sums
    /// `weight[f, c, k] * sample(input, base + offset) * mask` over the group's
    /// input channels and kernel points, then applies the source `sanitize`
    /// clamp (finite -> clamp to +/-10000, non-finite -> 0).
    ///
    /// Offsets/masks are *prepared* operands: `offsets` is `[B, 2*K, OH*OW]`
    /// with `k = ky*size + kx`, plane `2k` = dy and `2k+1` = dx; `masks` is
    /// `[B, K, OH*OW]` and is ignored when `use_mask == 0`. This is the
    /// deformable step only; the predictor columns, conditional normalization
    /// and lhtan/sigmoid stages are separate and not part of this kernel.
    #[kernel]
    pub fn deform_conv_accumulate(
        input: &[f32],
        offsets: &[f32],
        masks: &[f32],
        weights: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        filters: usize,
        channels_per_group: usize,
        filters_per_group: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        use_mask: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let plane = out_h * out_w;
        let bf = i / plane;
        let filter = bf % filters;
        let b = bf / filters;
        let pos = i % plane;
        let oy = pos / out_w;
        let ox = pos % out_w;
        let kernel_area = size * size;
        let offset_planes = 2 * kernel_area;
        let group = filter / filters_per_group;
        let offset_base = b * offset_planes * plane + pos;
        let mask_base = b * kernel_area * plane + pos;
        let mut sum = 0.0f32;
        let mut local_channel = 0usize;
        while local_channel < channels_per_group {
            let channel = group * channels_per_group + local_channel;
            let channel_base = (b * channels + channel) * in_h * in_w;
            let mut ky = 0usize;
            while ky < size {
                let mut kx = 0usize;
                while kx < size {
                    let k = ky * size + kx;
                    let dy = offsets[offset_base + (2 * k) * plane];
                    let dx = offsets[offset_base + (2 * k + 1) * plane];
                    let base_y = oy as f32 * stride_y as f32 - (padding * dilation) as f32
                        + (ky * dilation) as f32;
                    let base_x = ox as f32 * stride_x as f32 - (padding * dilation) as f32
                        + (kx * dilation) as f32;
                    let sample =
                        deform_sample_at(input, channel_base, in_h, in_w, base_y + dy, base_x + dx);
                    let mask = if use_mask != 0 {
                        masks[mask_base + k * plane]
                    } else {
                        1.0
                    };
                    let weight =
                        weights[(filter * channels_per_group + local_channel) * kernel_area + k];
                    sum += weight * sample * mask;
                    kx += 1;
                }
                ky += 1;
            }
            local_channel += 1;
        }
        // Source `sanitize`: finite -> clamp to +/-10000, non-finite -> 0.
        let value = if sum.is_finite() {
            if sum > 10_000.0f32 {
                10_000.0f32
            } else if sum < -10_000.0f32 {
                -10_000.0f32
            } else {
                sum
            }
        } else {
            0.0
        };
        if let Some(o) = out.get_mut(idx) {
            *o = value;
        }
    }


    /// Source `centered_lhtan` value (the derivative parts are not needed for
    /// a forward-only envelope). `max_offset` must be finite and positive.
    #[device]
    pub fn deform_centered_lhtan(value: f32, max_offset: f32) -> f32 {
        let scaled = value / (2.0 * max_offset) + 0.5;
        let mapped = if scaled < 0.0 {
            0.001 * scaled
        } else if scaled > 1.0 {
            1.0 + 0.001 * (scaled - 1.0)
        } else {
            scaled
        };
        deform_sanitize((2.0 * mapped - 1.0) * max_offset)
    }


    /// Per-batch offset statistics over the **sanitized** stage-1 offsets,
    /// matching the source f64 accumulation:
    /// `mean = sum/n`, `variance = max(sum_sq/n - mean^2, 0)`, `stddev = sqrt`,
    /// `normalize = stddev > 1e-12 && (stddev > max_offset || |mean| > max_offset)`,
    /// `inv_std = normalize ? 1/stddev : 1`. One thread per batch does the whole
    /// serial reduction (correctness-first; no cross-thread reduction).
    #[kernel]
    pub fn deform_offset_stats(
        raw: &[f32],
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        mut normalize_out: DisjointSlice<f32>,
        per_batch: usize,
        max_offset: f64,
    ) {
        let idx = thread::index_1d();
        let idx_inv = thread::index_1d();
        let idx_norm = thread::index_1d();
        let b = idx.get();
        if b >= mean_out.len() {
            return;
        }
        let start = b * per_batch;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut i = 0usize;
        while i < per_batch {
            let v = deform_sanitize(raw[start + i]) as f64;
            sum += v;
            sum_sq += v * v;
            i += 1;
        }
        let n = per_batch as f64;
        let mean = sum / n;
        let variance = if sum_sq / n - mean * mean > 0.0 {
            sum_sq / n - mean * mean
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let normalize = stddev > 1.0e-12 && (stddev > max_offset || mean.abs() > max_offset);
        let inv_std = if normalize { 1.0 / stddev } else { 1.0 };
        if let Some(o) = mean_out.get_mut(idx) {
            *o = mean as f32;
        }
        if let Some(o) = inv_std_out.get_mut(idx_inv) {
            *o = inv_std as f32;
        }
        if let Some(o) = normalize_out.get_mut(idx_norm) {
            *o = if normalize { 1.0 } else { 0.0 };
        }
    }




/// `[deform_conv]` offset envelope, one thread per offset element:
    /// `sanitize -> optional per-batch normalization -> sanitize -> centered
    /// lhtan`. `normalize`/`mean`/`inv_std` are per-batch f32 tensors.
    #[kernel]
    pub fn deform_offset_envelope(
        raw: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        normalize: &[f32],
        mut out: DisjointSlice<f32>,
        planes: usize,
        spatial: usize,
        max_offset: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let per_batch = planes * spatial;
        let b = i / per_batch;
        let mut value = deform_sanitize(raw[i]);
        if normalize[b] != 0.0 {
            value = deform_sanitize((value - mean[b]) * inv_std[b]);
        }
        if let Some(o) = out.get_mut(idx) {
            *o = deform_centered_lhtan(value, max_offset);
        }
    }

    /// Elementwise source `sanitize` (±10000 clamp, non-finite -> 0).
    #[kernel]
    pub fn deform_sanitize_tensor(raw: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = deform_sanitize(raw[i]);
        }
    }

    /// Mask probabilities: `stable_sigmoid(sanitize(logit))`, elementwise.
    #[kernel]
    pub fn deform_mask_sigmoid(raw: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = deform_stable_sigmoid(deform_sanitize(raw[i]));
        }
    }

    /// `out[b,s,c,oy,ox] = bilinear(input[b,c], y[b,s,oy,ox], x[b,s,oy,ox])`.
    /// One thread per output element, no atomics.
    #[kernel]
    pub fn deform_gather(
        input: &[f32],
        sample_y: &[f32],
        sample_x: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        in_h: usize,
        in_w: usize,
        samples: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let plane = out_h * out_w;
        let bc = i / plane;
        let c = bc % channels;
        let bs = bc / channels;
        let b = bs / samples;
        let p = i % plane;
        let coord = bs * plane + p;
        let y = sample_y[coord];
        let x = sample_x[coord];
        let mut value = 0.0f32;
        if !(y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32) {
            let y0 = deform_floor_i64(y);
            let x0 = deform_floor_i64(x);
            let ly = y - y0 as f32;
            let lx = x - x0 as f32;
            let hy = 1.0 - ly;
            let hx = 1.0 - lx;
            let base = (b * channels + c) * in_h * in_w;
            if deform_in_bounds(y0, in_h) && deform_in_bounds(x0, in_w) {
                value += hy * hx * input[base + y0 as usize * in_w + x0 as usize];
            }
            if deform_in_bounds(y0, in_h) && deform_in_bounds(x0 + 1, in_w) {
                value += hy * lx * input[base + y0 as usize * in_w + (x0 + 1) as usize];
            }
            if deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0, in_w) {
                value += ly * hx * input[base + (y0 + 1) as usize * in_w + x0 as usize];
            }
            if deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0 + 1, in_w) {
                value += ly * lx * input[base + (y0 + 1) as usize * in_w + (x0 + 1) as usize];
            }
        }
        if let Some(o) = out.get_mut(idx) {
            *o = value;
        }
    }

    /// Adjoint of `deform_gather`: one thread per sample cell owns
    /// `d_sample_y`/`d_sample_x` (summed over channels) and scatters
    /// `d_input` atomically. `d_input` must be zeroed before the launch.
    #[kernel]
    pub fn deform_gather_backward(
        input: &[f32],
        sample_y: &[f32],
        sample_x: &[f32],
        dy: &[f32],
        d_input: *mut f32,
        mut d_y: DisjointSlice<f32>,
        mut d_x: DisjointSlice<f32>,
        channels: usize,
        in_h: usize,
        in_w: usize,
        samples: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let idx_x = thread::index_1d();
        let i = idx.get();
        if i >= d_y.len() {
            return;
        }
        let plane = out_h * out_w;
        let bs = i / plane;
        let b = bs / samples;
        let p = i % plane;
        let y = sample_y[i];
        let x = sample_x[i];
        if y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32 {
            // Far outside: zero value and zero coordinate gradients.
            if let Some(o) = d_y.get_mut(idx) {
                *o = 0.0;
            }
            if let Some(o) = d_x.get_mut(idx_x) {
                *o = 0.0;
            }
            return;
        }
        let y0 = deform_floor_i64(y);
        let x0 = deform_floor_i64(x);
        let ly = y - y0 as f32;
        let lx = x - x0 as f32;
        let hy = 1.0 - ly;
        let hx = 1.0 - lx;
        let has00 = deform_in_bounds(y0, in_h) && deform_in_bounds(x0, in_w);
        let has01 = deform_in_bounds(y0, in_h) && deform_in_bounds(x0 + 1, in_w);
        let has10 = deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0, in_w);
        let has11 = deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0 + 1, in_w);
        // Clamp before the `usize` cast: a negative floor would wrap. The
        // offsets are only read under the matching `has*` guard.
        let r0 = y0.max(0) as usize * in_w;
        let r1 = (y0 + 1).max(0) as usize * in_w;
        let c0 = x0.max(0) as usize;
        let c1 = (x0 + 1).max(0) as usize;
        let mut acc_y = 0.0f32;
        let mut acc_x = 0.0f32;
        let mut c = 0usize;
        while c < channels {
            let base = (b * channels + c) * in_h * in_w;
            let mut v00 = 0.0f32;
            let mut v01 = 0.0f32;
            let mut v10 = 0.0f32;
            let mut v11 = 0.0f32;
            if has00 {
                v00 = input[base + r0 + c0];
            }
            if has01 {
                v01 = input[base + r0 + c1];
            }
            if has10 {
                v10 = input[base + r1 + c0];
            }
            if has11 {
                v11 = input[base + r1 + c1];
            }
            let grad = dy[(bs * channels + c) * plane + p];
            if grad != 0.0 {
                // SAFETY: every offset is a validated in-image tap of the
                // `batch*channels*in_h*in_w` buffer the host checked; all
                // writers to those addresses are atomic.
                unsafe {
                    if has00 {
                        deform_atomic_add_f32(d_input.add(base + r0 + c0), grad * hy * hx);
                    }
                    if has01 {
                        deform_atomic_add_f32(d_input.add(base + r0 + c1), grad * hy * lx);
                    }
                    if has10 {
                        deform_atomic_add_f32(d_input.add(base + r1 + c0), grad * ly * hx);
                    }
                    if has11 {
                        deform_atomic_add_f32(d_input.add(base + r1 + c1), grad * ly * lx);
                    }
                }
                acc_y += grad * (-hx * v00 - lx * v01 + hx * v10 + lx * v11);
                acc_x += grad * (-hy * v00 + hy * v01 - ly * v10 + ly * v11);
            }
            c += 1;
        }
        let _ = samples;
        if let Some(o) = d_y.get_mut(idx) {
            *o = acc_y;
        }
        if let Some(o) = d_x.get_mut(idx_x) {
            *o = acc_x;
        }
    }
}

/// Adjoint of `deform_conv_accumulate`: one thread per `(b, filter, pos)`
/// recomputes the output slope, then scatters into `d_input`, `d_offsets`,
/// `d_masks` and `d_weights` with float atomics. All four targets must be
/// zeroed by the caller; the result set is exact, summation order is not.
#[cuda_module]
pub mod backward {
    use super::*;

    /// Adjoint of the dense predictor **and** the column gather: one thread per
    /// `(b, column, pos)`. Scatters `d_input` (geometry gather), `d_weights` and
    /// `d_biases` atomically. The input `sanitize` slope is the caller's job.
    /// Pre-normalization envelope adjoint: `d_out * lhtan_slope * sanitize_slope`.
    #[kernel]
    pub fn deform_envelope_pre_backward(
        raw: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        normalize: &[f32],
        d_out: &[f32],
        mut out: DisjointSlice<f32>,
        per_batch: usize,
        max_offset: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let b = i / per_batch;
        let v1 = deform_sanitize(raw[i]);
        let normalized = normalize[b] != 0.0;
        let pre = if normalized { (v1 - mean[b]) * inv_std[b] } else { v1 };
        let v2 = if normalized { deform_sanitize(pre) } else { pre };
        let slope = if normalized { deform_sanitize_slope(pre) } else { 1.0 };
        if let Some(o) = out.get_mut(idx) {
            *o = d_out[i] * deform_centered_lhtan_slope(v2, max_offset) * slope;
        }
    }

    /// Multiply the summed input gradient by the raw input's `sanitize` slope.
    #[kernel]
    pub fn deform_scale_input_gradient(
        raw: &[f32],
        grad: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = grad[i] * deform_sanitize_slope(raw[i]);
        }
    }

    /// Mask sigmoid adjoint: `d * sigmoid_slope(sanitize(logit))`.
    #[kernel]
    pub fn deform_mask_sigmoid_backward(
        logits: &[f32],
        d_out: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = d_out[i] * deform_mask_sigmoid_slope(logits[i]);
        }
    }

    #[kernel]
    pub fn deform_predictor_backward(
        input: &[f32],
        weights: &[f32],
        d_logits: &[f32],
        d_input: *mut f32,
        d_weights: *mut f32,
        d_biases: *mut f32,
        channels: usize,
        filters: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        batch: usize,
    ) {
        let column_count = channels * size * size;
        let plane = out_h * out_w;
        let i = thread::index_1d().get();
        // Thread space is (b, column, pos); `d_logits` is the *inner* loop over
        // filters, so it must not bound the launch.
        if i >= batch * column_count * plane {
            return;
        }
        let bcp = i / plane;
        let column = bcp % column_count;
        let b = bcp / column_count;
        let pos = i % plane;
        let c = column / (size * size);
        let k = column % (size * size);
        let ky = k / size;
        let kx = k % size;
        // Column value straight from the gather: no materialised columns buffer.
        let oy_col = pos / out_w;
        let ox_col = pos % out_w;
        let gy =
            (oy_col * stride_y) as isize - (padding * dilation) as isize + (ky * dilation) as isize;
        let gx =
            (ox_col * stride_x) as isize - (padding * dilation) as isize + (kx * dilation) as isize;
        let col_value = if gy >= 0 && gy < in_h as isize && gx >= 0 && gx < in_w as isize {
            input[(b * channels + c) * in_h * in_w + gy as usize * in_w + gx as usize]
        } else {
            0.0
        };
        let mut d_col = 0.0f32;
        let mut f = 0usize;
        while f < filters {
            let g = d_logits[(b * filters + f) * plane + pos];
            if g != 0.0 {
                let w = weights[f * column_count + column];
                d_col += g * w;
                // SAFETY: zeroed caller buffers; atomic adds only.
                unsafe {
                    deform_atomic_add_f32(d_weights.add(f * column_count + column), g * col_value);
                    // The bias gradient must be counted once per (b, f, pos),
                    // not once per column: designate column 0's thread.
                    if column == 0 {
                        deform_atomic_add_f32(d_biases.add(f), g);
                    }
                }
            }
            f += 1;
        }
        if d_col == 0.0 {
            return;
        }
        let oy = pos / out_w;
        let ox = pos % out_w;
        let y = (oy * stride_y) as isize - (padding * dilation) as isize + (ky * dilation) as isize;
        let x = (ox * stride_x) as isize - (padding * dilation) as isize + (kx * dilation) as isize;
        if y >= 0 && y < in_h as isize && x >= 0 && x < in_w as isize {
            let index = (b * channels + c) * in_h * in_w + y as usize * in_w + x as usize;
            // SAFETY: the gather tap is in-image, so the address is inside the
            // caller's zeroed `batch*channels*in_h*in_w` buffer.
            unsafe { deform_atomic_add_f32(d_input.add(index), d_col) };
        }
    }

    #[kernel]
    pub fn deform_conv_accumulate_backward(
        input: &[f32],
        offsets: &[f32],
        masks: &[f32],
        weights: &[f32],
        dy: &[f32],
        d_input: *mut f32,
        d_offsets: *mut f32,
        d_masks: *mut f32,
        d_weights: *mut f32,
        channels: usize,
        filters: usize,
        channels_per_group: usize,
        filters_per_group: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        use_mask: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= dy.len() {
            return;
        }
        let plane = out_h * out_w;
        let bf = i / plane;
        let filter = bf % filters;
        let b = bf / filters;
        let pos = i % plane;
        let oy = pos / out_w;
        let ox = pos % out_w;
        let kernel_area = size * size;
        let group = filter / filters_per_group;
        let offset_base = b * 2 * kernel_area * plane + pos;
        let mask_base = b * kernel_area * plane + pos;
        // Pass 1: the unsanitized sum, for the output slope.
        let mut raw_sum = 0.0f32;
        let mut lc = 0usize;
        while lc < channels_per_group {
            let channel = group * channels_per_group + lc;
            let channel_base = (b * channels + channel) * in_h * in_w;
            let mut ky = 0usize;
            while ky < size {
                let mut kx = 0usize;
                while kx < size {
                    let k = ky * size + kx;
                    let y = (oy * stride_y) as f32 - (padding * dilation) as f32
                        + (ky * dilation) as f32
                        + offsets[offset_base + (2 * k) * plane];
                    let x = (ox * stride_x) as f32 - (padding * dilation) as f32
                        + (kx * dilation) as f32
                        + offsets[offset_base + (2 * k + 1) * plane];
                    let sample = deform_sample_at(input, channel_base, in_h, in_w, y, x);
                    let mask = if use_mask != 0 {
                        masks[mask_base + k * plane]
                    } else {
                        1.0
                    };
                    raw_sum += weights[(filter * channels_per_group + lc) * kernel_area + k]
                        * sample
                        * mask;
                    kx += 1;
                }
                ky += 1;
            }
            lc += 1;
        }
        let slope = if raw_sum.is_finite()
            && raw_sum > -10_000.0f32
            && raw_sum < 10_000.0f32
        {
            1.0f32
        } else {
            0.0f32
        };
        let g_out = dy[i] * slope;
        if g_out == 0.0 {
            return;
        }
        // Pass 2: scatter.
        lc = 0;
        while lc < channels_per_group {
            let channel = group * channels_per_group + lc;
            let channel_base = (b * channels + channel) * in_h * in_w;
            let mut ky = 0usize;
            while ky < size {
                let mut kx = 0usize;
                while kx < size {
                    let k = ky * size + kx;
                    let dy_off = offsets[offset_base + (2 * k) * plane];
                    let dx_off = offsets[offset_base + (2 * k + 1) * plane];
                    let y = (oy * stride_y) as f32 - (padding * dilation) as f32
                        + (ky * dilation) as f32
                        + dy_off;
                    let x = (ox * stride_x) as f32 - (padding * dilation) as f32
                        + (kx * dilation) as f32
                        + dx_off;
                    let mask_index = mask_base + k * plane;
                    let mask = if use_mask != 0 { masks[mask_index] } else { 1.0 };
                    let w_index = (filter * channels_per_group + lc) * kernel_area + k;
                    let weight = weights[w_index];
                    let sample = deform_sample_at(input, channel_base, in_h, in_w, y, x);
                    // SAFETY: all targets are zeroed caller buffers sized for
                    // their full extents; every offset below is in range.
                    unsafe {
                        deform_atomic_add_f32(d_weights.add(w_index), g_out * sample * mask);
                        if use_mask != 0 {
                            deform_atomic_add_f32(d_masks.add(mask_index), g_out * weight * sample);
                        }
                    }
                    if !(y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32) {
                        let y0 = deform_floor_i64(y);
                        let x0 = deform_floor_i64(x);
                        let ly = y - y0 as f32;
                        let lx = x - x0 as f32;
                        let hy = 1.0 - ly;
                        let hx = 1.0 - lx;
                        let scale = g_out * weight * mask;
                        let mut v00 = 0.0f32;
                        let mut v01 = 0.0f32;
                        let mut v10 = 0.0f32;
                        let mut v11 = 0.0f32;
                        let has00 = deform_in_bounds(y0, in_h) && deform_in_bounds(x0, in_w);
                        let has01 = deform_in_bounds(y0, in_h) && deform_in_bounds(x0 + 1, in_w);
                        let has10 = deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0, in_w);
                        let has11 =
                            deform_in_bounds(y0 + 1, in_h) && deform_in_bounds(x0 + 1, in_w);
                        let r0 = (if y0 > 0 { y0 } else { 0 }) as usize * in_w;
                        let r1 = (if y0 + 1 > 0 { y0 + 1 } else { 0 }) as usize * in_w;
                        let c0 = (if x0 > 0 { x0 } else { 0 }) as usize;
                        let c1 = (if x0 + 1 > 0 { x0 + 1 } else { 0 }) as usize;
                        if has00 {
                            v00 = input[channel_base + r0 + c0];
                        }
                        if has01 {
                            v01 = input[channel_base + r0 + c1];
                        }
                        if has10 {
                            v10 = input[channel_base + r1 + c0];
                        }
                        if has11 {
                            v11 = input[channel_base + r1 + c1];
                        }
                        // SAFETY: as above; only in-image taps are touched.
                        unsafe {
                            if has00 {
                                deform_atomic_add_f32(d_input.add(channel_base + r0 + c0), scale * hy * hx);
                            }
                            if has01 {
                                deform_atomic_add_f32(d_input.add(channel_base + r0 + c1), scale * hy * lx);
                            }
                            if has10 {
                                deform_atomic_add_f32(d_input.add(channel_base + r1 + c0), scale * ly * hx);
                            }
                            if has11 {
                                deform_atomic_add_f32(d_input.add(channel_base + r1 + c1), scale * ly * lx);
                            }
                            let dy_slope =
                                -hx * v00 - lx * v01 + hx * v10 + lx * v11;
                            let dx_slope =
                                -hy * v00 + hy * v01 - ly * v10 + ly * v11;
                            deform_atomic_add_f32(
                                d_offsets.add(offset_base + (2 * k) * plane),
                                scale * dy_slope,
                            );
                            deform_atomic_add_f32(
                                d_offsets.add(offset_base + (2 * k + 1) * plane),
                                scale * dx_slope,
                            );
                        }
                    }
                    kx += 1;
                }
                ky += 1;
            }
            lc += 1;
        }
    }
}
