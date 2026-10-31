//! `[centernet]` training loss and decode, CPU side.
//!
//! Port of `darknet raptor-fp4:src-lib/centernet_layer.cpp`:
//!
//! * `forward_centernet_layer` (training half): heatmap targets, heatmap focal
//!   loss, sparse regression at true centres. Split here into
//!   [`heatmap_targets`], [`heatmap_focal_cell`] / [`heatmap_focal`],
//!   [`regression_rows`] / [`regression_terms`], and [`train_loss_cpu`].
//! * `is_local_peak`, `get_centernet_detections_batch`, `correct_centernet_boxes`
//!   (decode): [`is_peak`], [`decode_peaks`], [`correct_box`].
//!
//! Shared helpers (layout, `ct_index`, Gaussian draw, radii, tiny-object weight,
//! box decode) come from the sibling `cpu` module and match the C++ already.
//! The per-cell functions here are the contract for the GPU kernels in
//! `src/lightnet-cuda-oxide/src/kernels/centernet_train.rs`.
//!
//! Record layout for the GPU regression kernel: see [`pack_regression_records`].

use super::{
    CenterNetError, CenterNetLayout, CenterNetSettings, adaptive_gaussian_radii,
    continuous_small_weight, ct_index, decode_centernet_box, draw_center_gaussian,
};
use crate::model::CenterNetSpec;

/// Floats per packed regression record (see [`pack_regression_records`]).
pub const RECORD_STRIDE: usize = 8;

// ---- private helpers (C++ anonymous-namespace equivalents) ----

/// C++ `finite_or_zero`.
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// C++ `clamp_delta`. `max_delta = f32::MAX` is the identity for finite values.
fn clamp_delta(value: f32, max_delta: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(-max_delta, max_delta)
}

/// C++ `safe_log` (1e-9 floor).
fn safe_log(value: f32) -> f32 {
    value.max(1e-9).ln()
}

/// C++ `smooth_l1_loss`. TODO(verify): the task brief says "L1"; the reference is
/// smooth-L1 (Huber, beta = 1), so that is what is implemented.
fn smooth_l1_loss(error: f32) -> f32 {
    let abs_error = error.abs();
    if abs_error < 1.0 {
        0.5 * error * error
    } else {
        abs_error - 0.5
    }
}

/// C++ `smooth_l1_grad`.
fn smooth_l1_grad(error: f32) -> f32 {
    if error > 1.0 {
        1.0
    } else if error < -1.0 {
        -1.0
    } else {
        error
    }
}

/// One validated truth row after the C++ parsing (`float_to_box_stride` + checks).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Row {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    /// Class id after `class_map` (C++ `l.map`).
    class_id: usize,
}

/// Parse image `b`'s truth rows. Mirrors the C++ target loop: stop at the first
/// `x == 0` sentinel, fail on a bad class or box, then drop rows outside the
/// `scale_min_px..scale_max_px` gate (C++ `continue`).
///
/// TODO(verify): C++ casts the class with `static_cast<int>(float)`, which truncates
/// toward zero; `as i64` does the same here.
fn parse_rows(
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
    b: usize,
) -> Result<Vec<Row>, CenterNetError> {
    let truths = spec.max_boxes * settings.truth_size;
    if truth.len() < truths * l.batch {
        return Err(CenterNetError(format!(
            "[centernet] truth needs {} entries, got {}",
            truths * l.batch,
            truth.len()
        )));
    }
    let net_w = settings.net_width as f32;
    let net_h = settings.net_height as f32;
    let mut rows = Vec::new();
    for t in 0..spec.max_boxes {
        let base = b * truths + t * settings.truth_size;
        let (x, y, w, h) = (truth[base], truth[base + 1], truth[base + 2], truth[base + 3]);
        if x == 0.0 {
            break; // C++ `if (!truth.x) break;`
        }
        let raw_class = truth[base + 4] as i64;
        if raw_class < 0 || raw_class >= l.classes as i64 {
            return Err(CenterNetError(format!(
                "[centernet] invalid class id {raw_class} outside [0,{})",
                l.classes
            )));
        }
        let class_id = match &settings.class_map {
            Some(map) => map[raw_class as usize],
            None => raw_class as usize,
        };
        let finite = x.is_finite() && y.is_finite() && w.is_finite() && h.is_finite();
        if !finite || x <= 0.0 || x > 1.0 || y <= 0.0 || y > 1.0 || w <= 0.0 || h <= 0.0 {
            return Err(CenterNetError(format!(
                "[centernet] invalid truth box x={x} y={y} w={w} h={h}"
            )));
        }
        let max_side_px = (w * net_w).max(h * net_h);
        if max_side_px < spec.scale_min_px || max_side_px >= spec.scale_max_px {
            continue;
        }
        rows.push(Row { x, y, w, h, class_id });
    }
    Ok(rows)
}

/// Grid cell of a row's centre, clamped to the map. Returns `(ix, iy, gx, gy)`
/// where `gx, gy` are the unclamped centre in cell units.
fn center_cell(row: &Row, l: &CenterNetLayout) -> (i64, i64, f32, f32) {
    let gx = row.x * l.w as f32;
    let gy = row.y * l.h as f32;
    let ix = (gx.floor() as i64).clamp(0, l.w as i64 - 1);
    let iy = (gy.floor() as i64).clamp(0, l.h as i64 - 1);
    (ix, iy, gx, gy)
}

/// C++ `std::min(truth.w * net.w, truth.h * net.h)`.
fn min_side_px(row: &Row, settings: &CenterNetSettings) -> f32 {
    (row.w * settings.net_width as f32).min(row.h * settings.net_height as f32)
}

// ---- target generation ----

/// Heatmap targets and per-cell tiny-object weights, both `[batch][class][H*W]`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeatmapTargets {
    pub target: Vec<f32>,
    pub weight: Vec<f32>,
}

/// C++ target build in `forward_centernet_layer`: one max-blended anisotropic
/// Gaussian per truth (`draw_center_gaussian`), weight starts at 1.
///
/// TODO(verify): `weight` is max-blended with the tiny-object multiplier only where
/// the Gaussian wins (`value > target`), exactly as C++ does. Cells never drawn keep 1.
pub fn heatmap_targets(
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
) -> Result<HeatmapTargets, CenterNetError> {
    let n = l.batch * l.classes * l.spatial();
    let mut target = vec![0.0f32; n];
    let mut weight = vec![1.0f32; n];
    for b in 0..l.batch {
        for row in parse_rows(truth, l, spec, settings, b)? {
            let (ix, iy, _, _) = center_cell(&row, l);
            let w_cells = (row.w * l.w as f32).max(1e-6);
            let h_cells = (row.h * l.h as f32).max(1e-6);
            let small = continuous_small_weight(spec, min_side_px(&row, settings));
            let (radius_x, radius_y) = adaptive_gaussian_radii(spec, w_cells, h_cells);
            draw_center_gaussian(
                l,
                &mut target,
                &mut weight,
                b,
                row.class_id,
                ix,
                iy,
                radius_x,
                radius_y,
                small,
            );
        }
    }
    Ok(HeatmapTargets { target, weight })
}

// ---- heatmap focal loss ----

/// One heatmap cell of the focal loss. `p_raw` is the sigmoid output, `y_raw` the
/// target, `w_raw` the weight. Returns `(delta, loss)`.
///
/// Literal C++ order. TODO(verify): the delta is taken w.r.t. the sigmoid output `p`
/// with no `p * (1 - p)` factor, because the C++ backward is a pure `axpy` of `delta`.
/// Confirm this against the consumer of `l.delta` before relying on it.
///
/// TODO(verify): positives use `y >= 0.999`, a literal reference threshold.
pub fn heatmap_focal_cell(p_raw: f32, y_raw: f32, w_raw: f32, spec: &CenterNetSpec) -> (f32, f32) {
    let y = y_raw.clamp(0.0, 1.0);
    let p = p_raw.clamp(1e-6, 1.0 - 1e-6);
    let alpha = spec.focal_alpha.max(0.0);
    let beta = spec.focal_beta.max(0.0);
    let w = w_raw.max(0.0);
    let norm = spec.hm_normalizer;
    if y >= 0.999 {
        let focal = (1.0 - p).powf(alpha);
        let grad = w * focal * (1.0 - p);
        (
            clamp_delta(norm * grad, spec.max_delta),
            norm * w * focal * (-safe_log(p)),
        )
    } else {
        let neg_weight = (1.0 - y).powf(beta);
        let focal = p.powf(alpha);
        let grad = -w * neg_weight * focal * p;
        (
            clamp_delta(norm * grad, spec.max_delta),
            norm * w * neg_weight * focal * (-safe_log(1.0 - p)),
        )
    }
}

/// Focal loss over every heatmap cell, in C++ order (batch, class, loc).
/// `output` is the sigmoid-activated forward. Returns `(delta, hm_loss)` with
/// `delta` the full `batch * outputs` buffer (only heatmap entries written).
pub fn heatmap_focal(
    output: &[f32],
    targets: &HeatmapTargets,
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
) -> (Vec<f32>, f32) {
    let spatial = l.spatial();
    let mut delta = vec![0.0f32; l.len()];
    let mut loss = 0.0f32;
    for b in 0..l.batch {
        for c in 0..l.classes {
            for loc in 0..spatial {
                let hti = (b * l.classes + c) * spatial + loc;
                let outi = ct_index(l, b, c, loc);
                let (d, cell_loss) =
                    heatmap_focal_cell(output[outi], targets.target[hti], targets.weight[hti], spec);
                delta[outi] = d;
                loss += cell_loss;
            }
        }
    }
    (delta, loss)
}

// ---- sparse regression ----

/// A positive centre: which cell it claims and what it should regress to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionRow {
    pub batch: usize,
    pub loc: usize,
    /// `small_weight * class_multiplier` (C++ `weight`).
    pub weight: f32,
    /// `[log_w, log_h, off_x, off_y]` targets (C++ `target_log_w`, ...).
    pub target: [f32; 4],
}

/// Build the positive regression rows for every image, in C++ order (batch, then
/// truth). Later rows that land on the same cell add to it, as C++ `+=` does.
pub fn regression_rows(
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
) -> Result<Vec<RegressionRow>, CenterNetError> {
    let mut rows_out = Vec::new();
    for b in 0..l.batch {
        for row in parse_rows(truth, l, spec, settings, b)? {
            let (ix, iy, gx, gy) = center_cell(&row, l);
            let loc = (iy * l.w as i64 + ix) as usize;
            let class_multiplier = settings
                .class_multipliers
                .as_ref()
                .map(|m| m[row.class_id])
                .unwrap_or(1.0);
            let weight = continuous_small_weight(spec, min_side_px(&row, settings)) * class_multiplier;
            rows_out.push(RegressionRow {
                batch: b,
                loc,
                weight,
                target: [
                    safe_log((row.w * l.w as f32).max(1e-6)),
                    safe_log((row.h * l.h as f32).max(1e-6)),
                    gx - ix as f32 - 0.5,
                    gy - iy as f32 - 0.5,
                ],
            });
        }
    }
    Ok(rows_out)
}

/// Per-positive result. `delta` is in head order `[log_w, log_h, off_x, off_y]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionTerms {
    pub delta: [f32; 4],
    pub wh_loss: f32,
    pub off_loss: f32,
}

/// C++ regression body for one positive. `pred` is the raw head at that cell in the
/// same `[log_w, log_h, off_x, off_y]` order.
pub fn regression_terms(pred: [f32; 4], row: &RegressionRow, spec: &CenterNetSpec) -> RegressionTerms {
    let weight = row.weight;
    let ew = finite_or_zero(pred[0]) - row.target[0];
    let eh = finite_or_zero(pred[1]) - row.target[1];
    let eox = finite_or_zero(pred[2]) - row.target[2];
    let eoy = finite_or_zero(pred[3]) - row.target[3];
    let wh = spec.wh_normalizer;
    let off = spec.off_normalizer;
    RegressionTerms {
        delta: [
            clamp_delta(-wh * weight * smooth_l1_grad(ew), spec.max_delta),
            clamp_delta(-wh * weight * smooth_l1_grad(eh), spec.max_delta),
            clamp_delta(-off * weight * smooth_l1_grad(eox), spec.max_delta),
            clamp_delta(-off * weight * smooth_l1_grad(eoy), spec.max_delta),
        ],
        wh_loss: wh * weight * (smooth_l1_loss(ew) + smooth_l1_loss(eh)),
        off_loss: off * weight * (smooth_l1_loss(eox) + smooth_l1_loss(eoy)),
    }
}

/// Pack regression rows for the GPU kernel. Stride [`RECORD_STRIDE`]:
/// `[batch, loc, weight, target_log_w, target_log_h, target_off_x, target_off_y, 0]`.
/// Indices are stored as `f32`, exact for values below 2^24.
pub fn pack_regression_records(rows: &[RegressionRow]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * RECORD_STRIDE);
    for r in rows {
        out.extend_from_slice(&[
            r.batch as f32,
            r.loc as f32,
            r.weight,
            r.target[0],
            r.target[1],
            r.target[2],
            r.target[3],
            0.0,
        ]);
    }
    out
}

// ---- full CPU training pass ----

/// Result of [`train_loss_cpu`].
#[derive(Clone, Debug, PartialEq)]
pub struct CenterNetTrainCpu {
    /// Full `batch * outputs` delta (heatmap and regression entries).
    pub delta: Vec<f32>,
    pub hm_loss: f32,
    pub wh_loss: f32,
    pub off_loss: f32,
    /// `(hm + wh + off) / batch`, as C++ reports `*l.cost`.
    pub cost: f32,
    /// Number of positive centres.
    pub positives: usize,
}

/// Full training pass: targets, heatmap focal, sparse regression, sanitise.
/// `output` must be the sigmoid-activated forward.
pub fn train_loss_cpu(
    output: &[f32],
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
) -> Result<CenterNetTrainCpu, CenterNetError> {
    let targets = heatmap_targets(truth, l, spec, settings)?;
    let (mut delta, hm_loss) = heatmap_focal(output, &targets, l, spec);

    let rows = regression_rows(truth, l, spec, settings)?;
    let (mut wh_loss, mut off_loss) = (0.0f32, 0.0f32);
    for row in &rows {
        let mut pred = [0.0f32; 4];
        for (k, p) in pred.iter_mut().enumerate() {
            *p = output[ct_index(l, row.batch, l.classes + k, row.loc)];
        }
        let terms = regression_terms(pred, row, spec);
        for k in 0..4 {
            delta[ct_index(l, row.batch, l.classes + k, row.loc)] += terms.delta[k];
        }
        wh_loss += terms.wh_loss;
        off_loss += terms.off_loss;
    }

    for d in delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    let cost = (hm_loss + wh_loss + off_loss) / l.batch.max(1) as f32;
    Ok(CenterNetTrainCpu {
        delta,
        hm_loss,
        wh_loss,
        off_loss,
        cost,
        positives: rows.len(),
    })
}

// ---- decode ----

/// One detection candidate: a heatmap peak plus its decoded box.
/// `bbox` is `(x, y, w, h)`, normalised to the network input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenterNetPeak {
    pub batch: usize,
    pub class_id: usize,
    pub loc: usize,
    /// Raw sigmoid score (C++ `dets.objectness`).
    pub score: f32,
    pub bbox: (f32, f32, f32, f32),
}

/// C++ `is_local_peak`: a 3x3 max-pool test. The centre is a peak when no in-bounds
/// neighbour is strictly greater. `peak_nms = 0` accepts every cell.
///
/// Note: the sibling `cpu::is_local_peak` does not check `peak_nms`. This function does.
pub fn is_peak(
    output: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    batch: usize,
    class_id: usize,
    loc: usize,
) -> bool {
    if spec.peak_nms == 0 {
        return true;
    }
    let row = (loc / l.w) as i64;
    let col = (loc % l.w) as i64;
    let center = output[ct_index(l, batch, class_id, loc)];
    for dy in -1i64..=1 {
        let y = row + dy;
        if y < 0 || y >= l.h as i64 {
            continue;
        }
        for dx in -1i64..=1 {
            let x = col + dx;
            if x < 0 || x >= l.w as i64 || (dx == 0 && dy == 0) {
                continue;
            }
            let neighbor = output[ct_index(l, batch, class_id, (y * l.w as i64 + x) as usize)];
            if neighbor > center {
                return false;
            }
        }
    }
    true
}

/// C++ `get_centernet_detections_batch`, peak list only.
///
/// Scans class-major, then cell, keeping `score > thresh` peaks. Without a cap the
/// output keeps that enumeration order.
///
/// TODO(verify): the reference has NO top-K. It emits every peak above `thresh`.
/// Top-K here applies only when `top_k < peaks.len()`: it does a stable sort by score
/// (descending), so ties keep class-major order, then truncates. Pass `usize::MAX`
/// to match the reference exactly. This function stays on the CPU by design.
pub fn decode_peaks(
    output: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    batch: usize,
    thresh: f32,
    top_k: usize,
) -> Vec<CenterNetPeak> {
    let spatial = l.spatial();
    let mut peaks = Vec::new();
    for c in 0..l.classes {
        for loc in 0..spatial {
            let score = output[ct_index(l, batch, c, loc)];
            // Literal C++ `score <= thresh || !is_local_peak` skip: NaN passes, as in C++.
            if score <= thresh || !is_peak(output, l, spec, batch, c, loc) {
                continue;
            }
            peaks.push(CenterNetPeak {
                batch,
                class_id: c,
                loc,
                score,
                bbox: decode_centernet_box(output, l, batch, loc),
            });
        }
    }
    if peaks.len() > top_k {
        peaks.sort_by(|a, b| b.score.total_cmp(&a.score));
        peaks.truncate(top_k);
    }
    peaks
}

/// C++ `correct_centernet_boxes`, for one box. Maps network-relative `(x, y, w, h)` to
/// image coordinates, undoing letterbox padding when `letter` is set.
///
/// TODO(verify): the letterbox `new_h` / `new_w` use integer division, as in C++.
/// `relative = true` keeps the result in `[0, 1]`.
pub fn correct_box(
    bbox: (f32, f32, f32, f32),
    w: i32,
    h: i32,
    netw: i32,
    neth: i32,
    relative: bool,
    letter: bool,
) -> (f32, f32, f32, f32) {
    let mut new_w = netw;
    let mut new_h = neth;
    if letter {
        if (netw as f32 / w as f32) < (neth as f32 / h as f32) {
            new_h = (h * netw) / w;
        } else {
            new_w = (w * neth) / h;
        }
    }
    let deltaw = (netw - new_w) as f32;
    let deltah = (neth - new_h) as f32;
    let ratiow = new_w as f32 / netw as f32;
    let ratioh = new_h as f32 / neth as f32;

    let (mut x, mut y, mut bw, mut bh) = bbox;
    x = (x - deltaw * 0.5 / netw as f32) / ratiow;
    y = (y - deltah * 0.5 / neth as f32) / ratioh;
    bw *= 1.0 / ratiow;
    bh *= 1.0 / ratioh;
    if !relative {
        x *= w as f32;
        bw *= w as f32;
        y *= h as f32;
        bh *= h as f32;
    }
    (x, y, bw, bh)
}
