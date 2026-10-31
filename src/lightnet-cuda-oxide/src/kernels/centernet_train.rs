//! CenterNet training kernels: per-cell heatmap focal-loss delta and per-cell sparse
//! regression deltas. Port of `darknet raptor-fp4:src-lib/centernet_layer.cpp`
//! (`forward_centernet_layer`, training half). The host builds the Gaussian targets,
//! tiny-object weights and regression records (`layers/centernet/train.rs`). Top-K
//! decode stays on the CPU.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// C++ `finite_or_zero`.
#[device]
fn ct_finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// C++ `clamp_delta`.
#[device]
fn ct_clamp_delta(value: f32, max_delta: f32) -> f32 {
    if value.is_finite() { value.max(-max_delta).min(max_delta) } else { 0.0 }
}

/// C++ `safe_log` (1e-9 floor).
#[device]
fn ct_safe_log(value: f32) -> f32 {
    value.max(1e-9).ln()
}

/// C++ `smooth_l1_loss`.
#[device]
fn ct_smooth_l1(error: f32) -> f32 {
    let abs_error = error.abs();
    if abs_error < 1.0 { 0.5 * error * error } else { abs_error - 0.5 }
}

/// C++ `smooth_l1_grad`.
#[device]
fn ct_smooth_l1_grad(error: f32) -> f32 {
    if error > 1.0 {
        1.0
    } else if error < -1.0 {
        -1.0
    } else {
        error
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Heatmap focal loss, one thread per `(batch, class, cell)`; `g` indexes
    /// `[batch][class][spatial]`, matching the host `targets` and `weights` buffers.
    ///
    /// `output` is the sigmoid-activated head in full layout (`batch * outputs`).
    /// `p = [focal_alpha, focal_beta, hm_normalizer, max_delta]`.
    /// Writes the heatmap delta at its head index (no other thread writes it) and
    /// the cell loss to `cell_loss[g]`. The host sums `cell_loss`.
    #[kernel]
    pub fn centernet_hm_focal_delta(
        output: &[f32],
        targets: &[f32],
        weights: &[f32],
        mut delta: DisjointSlice<f32>,
        mut cell_loss: DisjointSlice<f32>,
        batch: usize,
        classes: usize,
        spatial: usize,
        outputs: usize,
        p: &[f32],
    ) {
        let g = thread::index_1d().get();
        if g >= batch * classes * spatial {
            return;
        }
        let loc = g % spatial;
        let c = (g / spatial) % classes;
        let b = g / (spatial * classes);
        let out_index = b * outputs + c * spatial + loc;

        let y = targets[g].clamp(0.0, 1.0);
        let pr = output[out_index].clamp(1e-6, 1.0 - 1e-6);
        let w = weights[g].max(0.0);
        let alpha = p[0].max(0.0);
        let beta = p[1].max(0.0);
        let norm = p[2];
        let max_delta = p[3];

        // TODO(verify): as in the host `heatmap_focal_cell`, the delta is w.r.t. the sigmoid
        // output with no p(1-p) factor. Literal C++ order.
        let (grad, loss) = if y >= 0.999 {
            let focal = (1.0 - pr).powf(alpha);
            let grad = w * focal * (1.0 - pr);
            (norm * grad, norm * w * focal * (-ct_safe_log(pr)))
        } else {
            let neg_weight = (1.0 - y).powf(beta);
            let focal = pr.powf(alpha);
            let grad = -w * neg_weight * focal * pr;
            (norm * grad, norm * w * neg_weight * focal * (-ct_safe_log(1.0 - pr)))
        };
        // SAFETY: one thread owns each `(batch, class, loc)` head slot and each `cell_loss[g]`.
        unsafe {
            *delta.as_mut_ptr().add(out_index) = ct_clamp_delta(grad, max_delta);
            *cell_loss.as_mut_ptr().add(g) = loss;
        }
    }

    /// Sparse regression, one thread per `(batch, cell)`. Each thread scans every packed
    /// positive record (stride 8: `[batch, loc, weight, tgt_log_w, tgt_log_h, tgt_off_x,
    /// tgt_off_y, pad]`) in order. It accumulates the four regression deltas only for its
    /// own cell, so overlapping centres sum in the C++ `+=` order without atomics.
    ///
    /// `p = [wh_normalizer, off_normalizer, max_delta]`. Writes the four head deltas
    /// (`log_w`, `log_h`, `off_x`, `off_y`), plus `wh_loss[g]` and `off_loss[g]` per cell.
    /// O(cells x records), which is fine for the small `max_boxes` used here.
    #[kernel]
    pub fn centernet_regression_delta(
        output: &[f32],
        records: &[f32],
        mut delta: DisjointSlice<f32>,
        mut wh_loss: DisjointSlice<f32>,
        mut off_loss: DisjointSlice<f32>,
        batch: usize,
        classes: usize,
        spatial: usize,
        outputs: usize,
        record_count: usize,
        p: &[f32],
    ) {
        let g = thread::index_1d().get();
        if g >= batch * spatial {
            return;
        }
        let b = g / spatial;
        let loc = g % spatial;
        let wh_norm = p[0];
        let off_norm = p[1];
        let max_delta = p[2];
        // Regression entries for this cell: `base + k * spatial`, k = 0..4.
        let base = b * outputs + classes * spatial + loc;

        let mut acc = [0.0f32; 4];
        let mut wh = 0.0f32;
        let mut off = 0.0f32;
        let mut r = 0;
        while r < record_count {
            let rec = r * 8;
            if records[rec] as usize == b && records[rec + 1] as usize == loc {
                let weight = records[rec + 2];
                let ew = ct_finite(output[base]) - records[rec + 3];
                let eh = ct_finite(output[base + spatial]) - records[rec + 4];
                let eox = ct_finite(output[base + 2 * spatial]) - records[rec + 5];
                let eoy = ct_finite(output[base + 3 * spatial]) - records[rec + 6];
                acc[0] += ct_clamp_delta(-wh_norm * weight * ct_smooth_l1_grad(ew), max_delta);
                acc[1] += ct_clamp_delta(-wh_norm * weight * ct_smooth_l1_grad(eh), max_delta);
                acc[2] += ct_clamp_delta(-off_norm * weight * ct_smooth_l1_grad(eox), max_delta);
                acc[3] += ct_clamp_delta(-off_norm * weight * ct_smooth_l1_grad(eoy), max_delta);
                wh += wh_norm * weight * (ct_smooth_l1(ew) + ct_smooth_l1(eh));
                off += off_norm * weight * (ct_smooth_l1(eox) + ct_smooth_l1(eoy));
            }
            r += 1;
        }
        // SAFETY: one thread owns each `(batch, cell)` regression slot and its loss cells.
        unsafe {
            let del = delta.as_mut_ptr();
            *del.add(base) = ct_finite(acc[0]);
            *del.add(base + spatial) = ct_finite(acc[1]);
            *del.add(base + 2 * spatial) = ct_finite(acc[2]);
            *del.add(base + 3 * spatial) = ct_finite(acc[3]);
            *wh_loss.as_mut_ptr().add(g) = wh;
            *off_loss.as_mut_ptr().add(g) = off;
        }
    }
}
