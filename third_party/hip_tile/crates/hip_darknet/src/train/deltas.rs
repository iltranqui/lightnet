//! Training in pure Rust: Darknet's box math, deltas and cost.
//!
//! Every formula here is a port of the C++ documented in `_shared/darknet-semantics.md`
//! (Training loss / Training deltas); the tests at the bottom re-derive the box gradients
//! numerically, which is the only external check available (`rules.md`: never run the Darknet
//! binary).
//!
//! All arithmetic is f32 by decision (`ROADMAP.md`).

use crate::cfg::{IouLoss, YoloCfg};
use crate::data::BoxLabel;
use crate::model::Shape;

/// A box in the same space as Darknet's `Box`: center + size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Darknet `box_iou`.
pub fn box_iou(a: Box, b: Box) -> f32 {
    let i = intersection(a, b);
    if i == 0.0 {
        return 0.0;
    }
    i / (a.w * a.h + b.w * b.h - i)
}

fn intersection(a: Box, b: Box) -> f32 {
    let w = (a.x + a.w / 2.0).min(b.x + b.w / 2.0) - (a.x - a.w / 2.0).max(b.x - b.w / 2.0);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (a.y + a.h / 2.0).min(b.y + b.h / 2.0) - (a.y - a.h / 2.0).max(b.y - b.h / 2.0);
    if h <= 0.0 {
        return 0.0;
    }
    w * h
}

/// Darknet `box_giou` (`box.cpp:1040`).
pub fn box_giou(a: Box, b: Box) -> f32 {
    let cw = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let ch = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let c = cw * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let u = a.w * a.h + b.w * b.h - intersection(a, b);
    iou - (c - u) / c
}

/// Darknet `box_diou` (`box.cpp:1063`), note the hard-coded `0.6` exponent.
pub fn box_diou(a: Box, b: Box) -> f32 {
    let cw = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let ch = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let c = cw * cw + ch * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    iou - (d / c).powf(0.6)
}

/// Darknet `box_ciou` (`box.cpp:1088`).
pub fn box_ciou(a: Box, b: Box) -> f32 {
    let cw = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let ch = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let c = cw * cw + ch * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    let d = u / c;
    let ar_gt = b.w / b.h;
    let ar_pred = a.w / a.h;
    let ar_loss = 4.0 / (std::f32::consts::PI * std::f32::consts::PI) * (ar_gt.atan() - ar_pred.atan()).powi(2);
    let alpha = ar_loss / (1.0 - iou + ar_loss + 0.000001);
    iou - (d + alpha * ar_loss)
}

/// Darknet `box_iou_kind` (`box.cpp:658`).
pub fn box_iou_kind(a: Box, b: Box, kind: IouLoss) -> f32 {
    match kind {
        IouLoss::Giou => box_giou(a, b),
        IouLoss::Diou => box_diou(a, b),
        IouLoss::Ciou => box_ciou(a, b),
        _ => box_iou(a, b),
    }
}

/// Darknet `dx_box_iou` (`box.cpp:1410`): the gradient of the IoU-family score w.r.t. the
/// predicted box's `(top, bot, left, right)` edges, reduced to `(x, y, w, h)` slots.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DxRep {
    /// Gradient slot for x (from `p_dl + p_dr`).
    pub dt: f32,
    /// Gradient slot for y (from `p_dt + p_db`).
    pub db: f32,
    /// Gradient slot for w (from `p_dr - p_dl`).
    pub dl: f32,
    /// Gradient slot for h (from `p_db - p_dt`).
    pub dr: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn dx_box_iou(pred: Box, truth: Box, kind: IouLoss) -> DxRep {
    let (pred_t, pred_b) = (pred.y - pred.h / 2.0, pred.y + pred.h / 2.0);
    let (pred_l, pred_r) = (pred.x - pred.w / 2.0, pred.x + pred.w / 2.0);
    let (truth_t, truth_b) = (truth.y - truth.h / 2.0, truth.y + truth.h / 2.0);
    let (truth_l, truth_r) = (truth.x - truth.w / 2.0, truth.x + truth.w / 2.0);

    let x_area = (pred_b - pred_t) * (pred_r - pred_l);
    let xhat_area = (truth_b - truth_t) * (truth_r - truth_l);
    let ih = pred_b.min(truth_b) - pred_t.max(truth_t);
    let iw = pred_r.min(truth_r) - pred_l.max(truth_l);
    let i = iw * ih;
    let u = x_area + xhat_area - i;
    let s = (pred.x - truth.x).powi(2) + (pred.y - truth.y).powi(2);
    let giou_cw = pred_r.max(truth_r) - pred_l.min(truth_l);
    let giou_ch = pred_b.max(truth_b) - pred_t.min(truth_t);

    let dx_dt = -(pred_r - pred_l);
    let dx_db = pred_r - pred_l;
    let dx_dl = -(pred_b - pred_t);
    let dx_dr = pred_b - pred_t;

    let di_dt = if pred_t > truth_t { -iw } else { 0.0 };
    let di_db = if pred_b < truth_b { iw } else { 0.0 };
    let di_dl = if pred_l > truth_l { -ih } else { 0.0 };
    let di_dr = if pred_r < truth_r { ih } else { 0.0 };

    let du_dt = dx_dt - di_dt;
    let du_db = dx_db - di_db;
    let du_dl = dx_dl - di_dl;
    let du_dr = dx_dr - di_dr;

    let dc_dt = if pred_t < truth_t { -giou_cw } else { 0.0 };
    let dc_db = if pred_b > truth_b { giou_cw } else { 0.0 };
    let dc_dl = if pred_l < truth_l { -giou_ch } else { 0.0 };
    let dc_dr = if pred_r > truth_r { giou_ch } else { 0.0 };

    let (mut p_dt, mut p_db, mut p_dl, mut p_dr) = (0.0, 0.0, 0.0, 0.0);
    if u > 0.0 {
        let uu = u * u;
        p_dt = ((u * di_dt) - (i * du_dt)) / uu;
        p_db = ((u * di_db) - (i * du_db)) / uu;
        p_dl = ((u * di_dl) - (i * du_dl)) / uu;
        p_dr = ((u * di_dr) - (i * du_dr)) / uu;
    }
    // Corner bookkeeping, exactly as Darknet swaps it.
    let (top_lt_bot, left_lt_right) = (pred_t < pred_b, pred_l < pred_r);
    let (t, b) = (p_dt, p_db);
    p_dt = if top_lt_bot { t } else { b };
    p_db = if top_lt_bot { b } else { t };
    let (l, r) = (p_dl, p_dr);
    p_dl = if left_lt_right { l } else { r };
    p_dr = if left_lt_right { r } else { l };

    if kind == IouLoss::Giou {
        let gg = giou_cw * giou_ch;
        if gg > 0.0 {
            p_dt += ((gg * du_dt) - (u * dc_dt)) / (gg * gg);
            p_db += ((gg * du_db) - (u * dc_db)) / (gg * gg);
            p_dl += ((gg * du_dl) - (u * dc_dl)) / (gg * gg);
            p_dr += ((gg * du_dr) - (u * dc_dr)) / (gg * gg);
        }
        if iw <= 0.0 || ih <= 0.0 {
            p_dt = ((gg * du_dt) - (u * dc_dt)) / (gg * gg);
            p_db = ((gg * du_db) - (u * dc_db)) / (gg * gg);
            p_dl = ((gg * du_dl) - (u * dc_dl)) / (gg * gg);
            p_dr = ((gg * du_dr) - (u * dc_dr)) / (gg * gg);
        }
    }

    // Enclosing box for the DIoU/CIoU distance term.
    let ct = (pred.y - pred.h / 2.0).min(truth.y - truth.h / 2.0);
    let cb = (pred.y + pred.h / 2.0).max(truth.y + truth.h / 2.0);
    let cl = (pred.x - pred.w / 2.0).min(truth.x - truth.w / 2.0);
    let cr = (pred.x + pred.w / 2.0).max(truth.x + truth.w / 2.0);
    let cw = cr - cl;
    let ch = cb - ct;
    let c = cw * cw + ch * ch;

    let dct_dy = if pred_t < truth_t { 1.0 } else { 0.0 };
    let dct_dh = if pred_t < truth_t { -0.5 } else { 0.0 };
    let dcb_dy = if pred_b > truth_b { 1.0 } else { 0.0 };
    let dcb_dh = if pred_b > truth_b { 0.5 } else { 0.0 };
    let dcl_dx = if pred_l < truth_l { 1.0 } else { 0.0 };
    let dcl_dw = if pred_l < truth_l { -0.5 } else { 0.0 };
    let dcr_dx = if pred_r > truth_r { 1.0 } else { 0.0 };
    let dcr_dw = if pred_r > truth_r { 0.5 } else { 0.0 };

    // Width depends only on `x`, height only on `y` (`box.cpp:1512-1546`).
    let (dcw_dx, dcw_dy, dcw_dw, dcw_dh) = (dcr_dx - dcl_dx, 0.0, dcr_dw - dcl_dw, 0.0);
    let (dch_dx, dch_dy, dch_dw, dch_dh) = (0.0, dcb_dy - dct_dy, 0.0, dcb_dh - dct_dh);

    let mut p_dx = p_dl + p_dr;
    let mut p_dy = p_dt + p_db;
    let mut p_dw = p_dr - p_dl;
    let mut p_dh = p_db - p_dt;

    let distance_terms = |p_dx: &mut f32, p_dy: &mut f32, p_dw: &mut f32, p_dh: &mut f32| {
        if c > 0.0 {
            let cc = c * c;
            *p_dx += (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx + 2.0 * ch * dch_dx) * s) / cc;
            *p_dy += (2.0 * (truth.y - pred.y) * c - (2.0 * cw * dcw_dy + 2.0 * ch * dch_dy) * s) / cc;
            *p_dw += (2.0 * cw * dcw_dw + 2.0 * ch * dch_dw) * s / cc;
            *p_dh += (2.0 * cw * dcw_dh + 2.0 * ch * dch_dh) * s / cc;
        }
        if iw <= 0.0 || ih <= 0.0 {
            let cc = c * c;
            *p_dx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx + 2.0 * ch * dch_dx) * s) / cc;
            *p_dy = (2.0 * (truth.y - pred.y) * c - (2.0 * cw * dcw_dy + 2.0 * ch * dch_dy) * s) / cc;
            *p_dw = (2.0 * cw * dcw_dw + 2.0 * ch * dch_dw) * s / cc;
            *p_dh = (2.0 * cw * dcw_dh + 2.0 * ch * dch_dh) * s / cc;
        }
    };

    if kind == IouLoss::Diou {
        distance_terms(&mut p_dx, &mut p_dy, &mut p_dw, &mut p_dh);
    }
    if kind == IouLoss::Ciou {
        let ar_gt = truth.w / truth.h;
        let ar_pred = pred.w / pred.h;
        let ar_loss =
            4.0 / (std::f32::consts::PI * std::f32::consts::PI) * (ar_gt.atan() - ar_pred.atan()).powi(2);
        let alpha = ar_loss / (1.0 - i / u + ar_loss + 0.000001);
        let ar_dw = 8.0 / (std::f32::consts::PI * std::f32::consts::PI) * (ar_gt.atan() - ar_pred.atan()) * pred.h;
        let ar_dh = -8.0 / (std::f32::consts::PI * std::f32::consts::PI) * (ar_gt.atan() - ar_pred.atan()) * pred.w;
        if c > 0.0 {
            let cc = c * c;
            p_dx += (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx + 2.0 * ch * dch_dx) * s) / cc;
            p_dy += (2.0 * (truth.y - pred.y) * c - (2.0 * cw * dcw_dy + 2.0 * ch * dch_dy) * s) / cc;
            p_dw += (2.0 * cw * dcw_dw + 2.0 * ch * dch_dw) * s / cc + alpha * ar_dw;
            p_dh += (2.0 * cw * dcw_dh + 2.0 * ch * dch_dh) * s / cc + alpha * ar_dh;
        }
        if iw <= 0.0 || ih <= 0.0 {
            let cc = c * c;
            p_dx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx + 2.0 * ch * dch_dx) * s) / cc;
            p_dy = (2.0 * (truth.y - pred.y) * c - (2.0 * cw * dcw_dy + 2.0 * ch * dch_dy) * s) / cc;
            p_dw = (2.0 * cw * dcw_dw + 2.0 * ch * dch_dw) * s / cc + alpha * ar_dw;
            p_dh = (2.0 * cw * dcw_dh + 2.0 * ch * dch_dh) * s / cc + alpha * ar_dh;
        }
    }

    DxRep { dt: p_dx, db: p_dy, dl: p_dw, dr: p_dh }
}

fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

fn clip(x: f32, max: f32) -> f32 {
    x.clamp(-max, max)
}

/// `get_yolo_box` (`yolo_layer.cpp:54-88`) reading the *activated* head tensor.
#[allow(clippy::too_many_arguments)]
pub fn get_yolo_box(
    out: &[f32],
    plane: usize,
    entry: usize,
    cell: usize,
    bias: (f32, f32),
    lw: usize,
    lh: usize,
    net_w: usize,
    net_h: usize,
) -> Box {
    let at = |e: usize| out[entry + e * plane + cell];
    let (col, row) = (cell % lw, cell / lw);
    let mut tw = finite(at(2)).clamp(-10.0, 10.0);
    let mut th = finite(at(3)).clamp(-10.0, 10.0);
    tw = tw.clamp(-10.0, 10.0);
    th = th.clamp(-10.0, 10.0);
    Box {
        x: finite((col as f32 + at(0)) / lw as f32),
        y: finite((row as f32 + at(1)) / lh as f32),
        w: finite((tw.exp() * bias.0 / net_w as f32).max(1e-6)),
        h: finite((th.exp() * bias.1 / net_h as f32).max(1e-6)),
    }
}

/// Per-head statistics that feed the reported cost (`yolo_layer.cpp:1231-1291`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HeadStats {
    pub count: usize,
    pub tot_iou: f32,
    pub tot_iou_loss: f32,
    pub tot_giou_loss: f32,
}

/// Apply Darknet's `[yolo]` delta for one image and one head.
///
/// `out` is the head's activated output (`mask.len() * (classes + 5)` planes of `h*w`);
/// `delta` is written in the same layout; `truths` are this image's labels.
#[allow(clippy::too_many_arguments)]
pub fn yolo_delta(
    out: &[f32],
    shape: Shape,
    y: &YoloCfg,
    truths: &[BoxLabel],
    net_w: usize,
    net_h: usize,
    delta: &mut [f32],
) -> HeadStats {
    let t = &y.train;
    let (lw, lh) = (shape.w, shape.h);
    let plane = lw * lh;
    let n_anchors = y.mask.len();
    let coords = 4;
    let stride = plane;
    let mut stats = HeadStats::default();

    // Pass A: every cell/anchor, objectness + (rarely) the truth_thresh assignment.
    for j in 0..lh {
        for i in 0..lw {
            for n in 0..n_anchors {
                let cell = j * lw + i;
                let entry = n * (y.classes + 5) * plane;
                let anchor = y.mask[n];
                let pred = get_yolo_box(out, plane, entry, cell, (y.anchors[2 * anchor], y.anchors[2 * anchor + 1]), lw, lh, net_w, net_h);
                let obj_index = entry + coords * plane + cell;
                let class_index = entry + (coords + 1) * plane + cell;

                // `compare_yolo_class` (`yolo_layer.cpp:441`): any class prob above 0.25.
                let class_seen = (0..y.classes).any(|c| out[class_index + stride * c] > 0.25);
                let mut best_iou = 0.0f32;
                let mut best_match_iou = 0.0f32;
                let mut best_t = usize::MAX;
                for (ti, truth) in truths.iter().enumerate() {
                    let tb = Box { x: truth.x, y: truth.y, w: truth.w, h: truth.h };
                    let iou = box_iou(pred, tb);
                    if iou > best_match_iou && class_seen {
                        best_match_iou = iou;
                    }
                    if iou > best_iou {
                        best_iou = iou;
                        best_t = ti;
                    }
                }

                delta[obj_index] = t.obj_normalizer * (0.0 - out[obj_index]);
                if best_match_iou > t.ignore_thresh {
                    delta[obj_index] = 0.0;
                }
                if best_iou > t.truth_thresh && best_t != usize::MAX {
                    let truth = &truths[best_t];
                    let tb = Box { x: truth.x, y: truth.y, w: truth.w, h: truth.h };
                    delta[obj_index] = t.obj_normalizer * (1.0 - out[obj_index]);
                    delta_class(out, delta, class_index, truth.class, y.classes, stride);
                    delta_box(tb, out, delta, plane, entry, cell, anchor, y, lw, lh, net_w, net_h, t.iou_normalizer, true, &mut stats);
                }
            }
        }
    }

    // Pass B: each truth is assigned to the best-matching anchor of this head
    // (`yolo_layer.cpp:795-880`).
    for truth in truths {
        let tb = Box { x: truth.x, y: truth.y, w: truth.w, h: truth.h };
        let mut best_iou = 0.0f32;
        let mut best_anchor = 0usize;
        for a in 0..y.anchors.len() / 2 {
            let anchor_box = Box {
                x: 0.0,
                y: 0.0,
                w: y.anchors[2 * a] / net_w as f32,
                h: y.anchors[2 * a + 1] / net_h as f32,
            };
            let shifted = Box { x: 0.0, y: 0.0, ..tb };
            let iou = box_iou(anchor_box, shifted);
            if iou > best_iou {
                best_iou = iou;
                best_anchor = a;
            }
        }
        let Some(pos) = y.mask.iter().position(|&m| m == best_anchor) else {
            continue;
        };
        let i = ((truth.x * lw as f32) as usize).min(lw - 1);
        let j = ((truth.y * lh as f32) as usize).min(lh - 1);
        let cell = j * lw + i;
        let entry = pos * (y.classes + 5) * plane;
        let obj_index = entry + coords * plane + cell;
        let class_index = entry + (coords + 1) * plane + cell;
        delta_box(tb, out, delta, plane, entry, cell, best_anchor, y, lw, lh, net_w, net_h, t.iou_normalizer, true, &mut stats);
        delta[obj_index] = t.obj_normalizer * (1.0 - out[obj_index]);
        delta_class(out, delta, class_index, truth.class, y.classes, stride);
        stats.count += 1;
    }
    stats
}

/// `delta_yolo_class` (`yolo_layer.cpp:352`). A cell whose truth-class delta is already set only
/// has that one refreshed; otherwise every class gets `(n == class) - output`, which is what
/// pushes the other classes of a positive cell toward 0. `cls_normalizer` only scales the delta
/// together with `classes_multipliers` there, which this port does not have.
fn delta_class(out: &[f32], delta: &mut [f32], class_index: usize, class: usize, classes: usize, stride: usize) {
    let ti = class_index + stride * class;
    if delta[ti] != 0.0 {
        delta[ti] = 1.0 - out[ti];
        return;
    }
    for n in 0..classes {
        let ci = class_index + stride * n;
        let d = (if n == class { 1.0 } else { 0.0 }) - out[ci];
        if d.is_finite() {
            delta[ci] = d;
        }
    }
}

/// `delta_yolo_box` (`yolo_layer.cpp:154-310`) for one anchor cell.
#[allow(clippy::too_many_arguments)]
fn delta_box(
    truth: Box,
    out: &[f32],
    delta: &mut [f32],
    plane: usize,
    entry: usize,
    cell: usize,
    anchor: usize,
    y: &YoloCfg,
    lw: usize,
    lh: usize,
    net_w: usize,
    net_h: usize,
    iou_normalizer: f32,
    accumulate: bool,
    stats: &mut HeadStats,
) {
    let t = &y.train;
    let idx = entry + cell;
    let pred = get_yolo_box(out, plane, entry, cell, (y.anchors[2 * anchor], y.anchors[2 * anchor + 1]), lw, lh, net_w, net_h);
    let iou = box_iou(pred, truth);
    let giou = box_giou(pred, truth);
    stats.tot_iou += finite(iou);
    stats.tot_iou_loss += 1.0 - finite(iou);
    stats.tot_giou_loss += 1.0 - finite(giou);

    if t.iou_loss == IouLoss::Mse {
        if !accumulate {
            for c in 0..4 {
                delta[idx + c * plane] = 0.0;
            }
        }
        let tx = truth.x * lw as f32 - cell as f32 % lw as f32;
        let ty = truth.y * lh as f32 - (cell / lw) as f32;
        let tw = (truth.w * net_w as f32 / y.anchors[2 * anchor]).ln();
        let th = (truth.h * net_h as f32 / y.anchors[2 * anchor + 1]).ln();
        delta[idx] += iou_normalizer * (tx - out[idx]);
        delta[idx + plane] += iou_normalizer * (ty - out[idx + plane]);
        delta[idx + 2 * plane] += iou_normalizer * (tw - out[idx + 2 * plane]);
        delta[idx + 3 * plane] += iou_normalizer * (th - out[idx + 3 * plane]);
        return;
    }

    let dxrep = dx_box_iou(pred, truth, t.iou_loss);
    let (mut dx, mut dy, mut dw, mut dh) = (dxrep.dt, dxrep.db, dxrep.dl, dxrep.dr);
    let tw = finite(out[idx + 2 * plane]).clamp(-10.0, 10.0);
    let th = finite(out[idx + 3 * plane]).clamp(-10.0, 10.0);
    dw *= tw.exp();
    dh *= th.exp();
    dx *= iou_normalizer;
    dy *= iou_normalizer;
    dw *= iou_normalizer;
    dh *= iou_normalizer;
    let (dx, dy, dw, dh) = (finite(dx), finite(dy), finite(dw), finite(dh));
    let clip_if = |v: f32| if t.max_delta != f32::MAX { clip(v, t.max_delta) } else { v };
    let (dx, dy, dw, dh) = (clip_if(dx), clip_if(dy), clip_if(dw), clip_if(dh));
    if !accumulate {
        for c in 0..4 {
            delta[idx + c * plane] = 0.0;
        }
    }
    delta[idx] += dx;
    delta[idx + plane] += dy;
    delta[idx + 2 * plane] += dw;
    delta[idx + 3 * plane] += dh;
}

/// `Σδ²` of one image's head delta, optionally skipping the four box entries (`x, y, w, h`).
pub fn delta_sq(delta: &[f32], shape: Shape, y: &YoloCfg, skip_box: bool) -> f32 {
    let plane = shape.w * shape.h;
    let mut acc = 0.0f32;
    for n in 0..y.mask.len() {
        let entry = n * (y.classes + 5) * plane;
        for c in 0..(y.classes + 5) {
            if skip_box && c < 4 {
                continue;
            }
            let base = entry + c * plane;
            for v in &delta[base..base + plane] {
                acc += v * v;
            }
        }
    }
    acc
}

/// The reported cost of one head from its summed statistics (`yolo_layer.cpp:1255-1291`):
/// `sq_all` / `sq_nobox` are `Σδ²` over the whole batch with and without the box entries. `mse`
/// reports `sq_all`; otherwise the IoU-loss average plus the normalised classification cost.
pub fn head_cost_from_sums(y: &YoloCfg, stats: &HeadStats, sq_all: f32, sq_nobox: f32) -> f32 {
    let t = &y.train;
    if t.iou_loss == IouLoss::Mse {
        return sq_all;
    }
    let count = stats.count.max(1) as f32;
    let iou_loss_avg = if t.iou_loss == IouLoss::Giou {
        t.iou_normalizer * (stats.tot_giou_loss / count)
    } else {
        t.iou_normalizer * (stats.tot_iou_loss / count)
    };
    iou_loss_avg + t.obj_normalizer * sq_nobox
}

/// The reported cost of one head for a single-image `delta` (see [`head_cost_from_sums`]).
pub fn head_cost(delta: &[f32], shape: Shape, y: &YoloCfg, stats: &HeadStats) -> f32 {
    head_cost_from_sums(y, stats, delta_sq(delta, shape, y, false), delta_sq(delta, shape, y, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::YoloTrainCfg;

    /// Numeric derivative of the score w.r.t. (x, y, w, h).
    fn fd_box(kind: IouLoss, pred: Box, truth: Box, h: f32) -> [f32; 4] {
        let score = |b: Box| box_iou_kind(b, truth, kind);
        let mut g = [0f32; 4];
        let fields: [fn(&mut Box) -> &mut f32; 4] =
            [|b| &mut b.x, |b| &mut b.y, |b| &mut b.w, |b| &mut b.h];
        for (i, f) in fields.iter().enumerate() {
            let mut a = pred;
            let mut b = pred;
            *f(&mut a) -= h;
            *f(&mut b) += h;
            g[i] = (score(b) - score(a)) / (2.0 * h);
        }
        g
    }

    /// `dx`/`dy` are the exact score gradient for IoU and GIoU: this pins the corner bookkeeping,
    /// the intersection/union partials and Darknet's min/max swap.
    #[test]
    fn dx_dy_exact_for_iou_and_giou() {
        // Corners are strictly inside/outside the truth box: no ties, so the piecewise-linear IoU
        // is differentiable at every probe.
        let truth = Box { x: 0.5, y: 0.5, w: 0.3, h: 0.2 };
        let preds = [
            Box { x: 0.44, y: 0.46, w: 0.22, h: 0.26 },
            Box { x: 0.56, y: 0.53, w: 0.34, h: 0.24 },
            Box { x: 0.60, y: 0.48, w: 0.16, h: 0.22 },
        ];
        for kind in [IouLoss::Iou, IouLoss::Giou] {
            for pred in preds {
                let g = dx_box_iou(pred, truth, kind);
                let n = fd_box(kind, pred, truth, 1e-4);
                for (name, a, x) in [("dx", g.dt, n[0]), ("dy", g.db, n[1])] {
                    let rel = (a - x).abs() / a.abs().max(x.abs()).max(1e-3);
                    assert!(rel < 2e-3, "{kind:?} {name}: {a} vs {x} (rel {rel})");
                }
            }
        }
    }

    /// Darknet does **not** halve the w/h slots ("For dw and dh, we do not divided by 2"), so the
    /// ported values are twice the derivative of the score w.r.t. the box width/height.
    #[test]
    fn dw_dh_are_twice_the_exact_gradient() {
        let truth = Box { x: 0.5, y: 0.5, w: 0.3, h: 0.2 };
        let pred = Box { x: 0.44, y: 0.46, w: 0.22, h: 0.26 };
        for kind in [IouLoss::Iou, IouLoss::Giou] {
            let g = dx_box_iou(pred, truth, kind);
            let n = fd_box(kind, pred, truth, 1e-4);
            for (name, a, x) in [("dw", g.dl, n[2]), ("dh", g.dr, n[3])] {
                let ratio = a / x;
                assert!((ratio - 2.0).abs() < 1e-2, "{kind:?} {name}: ratio {ratio}");
            }
        }
    }

    /// The CIoU jacobian is an approximation by design: the `alpha` factor's own dependence on the
    /// box is dropped and the aspect-ratio derivative omits the `1/(w^2+h^2)` denominator. So the
    /// check splits it: (a) `dx`/`dy` are the exact score gradient, (b) the `w`/`h` slots equal
    /// *twice* the numerical derivative of `iou - centre_distance` plus exactly `alpha * ar_d*`,
    /// with `alpha` and `ar_d*` computed here from their definitions.
    #[test]
    fn ciou_jacobian_is_the_documented_approximation() {
        let truth = Box { x: 0.5, y: 0.4, w: 0.25, h: 0.35 };
        let iou = |b: Box| box_iou(b, truth);
        let centre = |b: Box| {
            let cw = (b.x + b.w / 2.0).max(truth.x + truth.w / 2.0) - (b.x - b.w / 2.0).min(truth.x - truth.w / 2.0);
            let ch = (b.y + b.h / 2.0).max(truth.y + truth.h / 2.0) - (b.y - b.h / 2.0).min(truth.y - truth.h / 2.0);
            ((b.x - truth.x).powi(2) + (b.y - truth.y).powi(2)) / (cw * cw + ch * ch)
        };
        // The IoU term enters the w/h slots doubled ("we do not divide by 2"), while the enclosing-box
        // distance term enters at its true scale, so the two are differenced separately.
        let iou_part = iou;
        let dist_part = |b: Box| -centre(b);
        let k = 4.0 / (std::f32::consts::PI * std::f32::consts::PI);
        for pred in [
            Box { x: 0.47, y: 0.43, w: 0.22, h: 0.3 },
            Box { x: 0.54, y: 0.37, w: 0.3, h: 0.4 },
            Box { x: 0.58, y: 0.46, w: 0.2, h: 0.24 },
        ] {
            let g = dx_box_iou(pred, truth, IouLoss::Ciou);
            let n_full = fd_box(IouLoss::Ciou, pred, truth, 1e-4);
            // Dropping `alpha`'s dependence on the centre is Darknet's own approximation, so x/y
            // deviate from the exact CIoU gradient by up to ~7% on these samples.
            for (name, a, x) in [("dx", g.dt, n_full[0]), ("dy", g.db, n_full[1])] {
                let rel = (a - x).abs() / a.abs().max(x.abs()).max(1e-3);
                assert!(rel < 1.2e-1, "ciou {name}: {a} vs {x} (rel {rel})");
            }
            // h is large enough that f32 cancellation in the central difference stays below the
            // tolerance; the box pairs above keep every corner comparison strict, so no kink.
            let h = 1e-3;
            let d = |f: &dyn Fn(Box) -> f32, dx: f32, dy: f32| -> f32 {
                let a = f(Box { w: pred.w + dx, h: pred.h + dy, ..pred });
                let b = f(Box { w: pred.w - dx, h: pred.h - dy, ..pred });
                (a - b) / (2.0 * if dx != 0.0 { dx } else { dy })
            };
            let dn_dw = 2.0 * d(&iou_part, h, 0.0) + d(&dist_part, h, 0.0);
            let dn_dh = 2.0 * d(&iou_part, 0.0, h) + d(&dist_part, 0.0, h);
            let ar_gt = truth.w / truth.h;
            let ar_pred = pred.w / pred.h;
            let ar_loss = k * (ar_gt.atan() - ar_pred.atan()).powi(2);
            let alpha = ar_loss / (1.0 - iou(pred) + ar_loss + 0.000001);
            let ar_dw = 2.0 * k * (ar_gt.atan() - ar_pred.atan()) * pred.h;
            let ar_dh = -2.0 * k * (ar_gt.atan() - ar_pred.atan()) * pred.w;
            for (name, a, expect) in [
                ("dw", g.dl, dn_dw + alpha * ar_dw),
                ("dh", g.dr, dn_dh + alpha * ar_dh),
            ] {
                let rel = (a - expect).abs() / a.abs().max(expect.abs()).max(1e-3);
                assert!(rel < 2e-2, "ciou {name}: {a} vs {expect} (rel {rel})");
            }
        }
    }

    /// `iou_loss=mse`: the classic `(target - x) * normalizer` on all four entries.
    #[test]
    fn mse_delta_is_hand_computable() {
        let y = head(1, vec![0], vec![8.0, 4.0]);
        let shape = Shape { c: 6, h: 1, w: 1 };
        // Activated head: x = 0.5, y = 0.5, w = ln(2) (so exp = 2), h = 0, obj = 0.5, cls = 0.5.
        let mut out = vec![0.0f32; 6];
        out[0] = 0.5;
        out[1] = 0.5;
        out[2] = 2.0f32.ln();
        out[3] = 0.0;
        out[4] = 0.5;
        out[5] = 0.5;
        let mut ym = y.clone();
        ym.train.iou_loss = IouLoss::Mse;
        ym.train.iou_normalizer = 1.0;
        let truths = [BoxLabel { class: 0, x: 0.5, y: 0.5, w: 0.5, h: 0.5 }];
        let mut delta = vec![0.0f32; 6];
        yolo_delta(&out, shape, &ym, &truths, 16, 16, &mut delta);
        // tx = 0.5*1 - 0 = 0.5, so delta_x = (0.5 - 0.5) = 0.
        assert!(delta[0].abs() < 1e-6, "{}", delta[0]);
        assert!(delta[1].abs() < 1e-6);
        // tw = ln(0.5*16/8) = ln(1) = 0 -> delta_w = 0 - ln(2).
        assert!((delta[2] - -2.0f32.ln()).abs() < 1e-6, "{}", delta[2]);
        // th = ln(0.5*16/4) = ln(2) -> delta_h = ln(2) - 0.
        assert!((delta[3] - 2.0f32.ln()).abs() < 1e-6, "{}", delta[3]);
        assert!((delta[4] - 0.5).abs() < 1e-6, "{}", delta[4]);
    }

    fn head(classes: usize, mask: Vec<usize>, anchors: Vec<f32>) -> YoloCfg {
        YoloCfg {
            classes,
            mask,
            anchors,
            scale_x_y: 1.0,
            nms_kind: crate::cfg::NmsKind::Greedy,
            beta_nms: 0.6,
            train: YoloTrainCfg {
                jitter: 0.0,
                resize: 1.0,
                iou_loss: IouLoss::Ciou,
                iou_normalizer: 0.07,
                obj_normalizer: 1.0,
                cls_normalizer: 1.0,
                ignore_thresh: 0.7,
                truth_thresh: 1.0,
                max_delta: f32::MAX,
            },
        }
    }

    #[test]
    fn delta_assigns_one_responsible_anchor() {
        // 2x2 grid, 1 class, one anchor of 10x10 px on a 20x20 net -> cell 0,0..1,1.
        let y = head(1, vec![0], vec![10.0, 10.0]);
        let shape = Shape { c: 6, h: 2, w: 2 };
        // An activated head, one anchor and one class: planes x, y, w, h, obj, cls of 4 cells.
        // x,y = 0.5 (cell centre), w,h raw 0 (exp(0) = 1), obj = cls = 0.5.
        let mut out = vec![0.0f32; 6 * 4];
        for plane_idx in [0usize, 1, 4, 5] {
            for cell in 0..4 {
                out[plane_idx * 4 + cell] = 0.5;
            }
        }
        let mut delta = vec![0.0f32; out.len()];
        let truths = [BoxLabel { class: 0, x: 0.25, y: 0.25, w: 0.5, h: 0.5 }];
        let stats = yolo_delta(&out, shape, &y, &truths, 20, 20, &mut delta);
        assert_eq!(stats.count, 1);
        // The responsible cell is (0,0): objectness delta = 1 - 0.5 = 0.5.
        let plane = 4;
        let obj = 4 * plane; // entry 0, objectness plane, cell 0
        assert!((delta[obj] - 0.5).abs() < 1e-6, "{}", delta[obj]);
        // Class 0 delta = 1 - 0.5.
        assert!((delta[5 * plane] - 0.5).abs() < 1e-6);
        // The probe sits exactly on the truth, so the box gradient is ~0 — but it must be finite.
        for c in 0..4 {
            assert!(delta[c * plane].is_finite(), "box slot {c} not finite");
            assert!(delta[c * plane].abs() < 1.0);
        }
    }

    #[test]
    fn positive_cell_pushes_the_other_classes_toward_zero() {
        // 3 classes: planes x, y, w, h, obj, c0, c1, c2 of a 2x2 grid; every class prob is 0.4.
        let y = head(3, vec![0], vec![10.0, 10.0]);
        let shape = Shape { c: 8, h: 2, w: 2 };
        let mut out = vec![0.0f32; 8 * 4];
        for plane_idx in [0usize, 1, 4, 5, 6, 7] {
            for cell in 0..4 {
                out[plane_idx * 4 + cell] = if plane_idx >= 5 { 0.4 } else { 0.5 };
            }
        }
        let mut delta = vec![0.0f32; out.len()];
        let truths = [BoxLabel { class: 1, x: 0.25, y: 0.25, w: 0.5, h: 0.5 }];
        yolo_delta(&out, shape, &y, &truths, 20, 20, &mut delta);
        let plane = 4;
        // The responsible cell is 0: the truth class goes to 1, the others to 0.
        assert!((delta[5 * plane] + 0.4).abs() < 1e-6, "class 0: {}", delta[5 * plane]);
        assert!((delta[6 * plane] - 0.6).abs() < 1e-6, "class 1: {}", delta[6 * plane]);
        assert!((delta[7 * plane] + 0.4).abs() < 1e-6, "class 2: {}", delta[7 * plane]);
        // Cells without a truth get no class delta.
        assert_eq!(delta[5 * plane + 1], 0.0);
    }

    #[test]
    fn ignore_thresh_zeroes_objectness_of_a_matching_anchor() {
        let mut y = head(1, vec![0], vec![10.0, 10.0]);
        y.train.ignore_thresh = 0.0; // everything above IoU 0 is "ignored"
        let shape = Shape { c: 6, h: 2, w: 2 };
        let mut out = vec![0.0f32; 6 * 4];
        for plane_idx in [0usize, 1, 4, 5] {
            for cell in 0..4 {
                out[plane_idx * 4 + cell] = 0.5;
            }
        }
        let mut delta = vec![0.0f32; out.len()];
        let truths = [BoxLabel { class: 0, x: 0.25, y: 0.25, w: 0.5, h: 0.5 }];
        yolo_delta(&out, shape, &y, &truths, 20, 20, &mut delta);
        // Pass B still sets the responsible anchor's objectness after pass A zeroed it.
        let plane = 4;
        assert!((delta[4 * plane] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn head_cost_is_batch_sum_of_deltas_squared() {
        let y = head(1, vec![0], vec![10.0, 10.0]);
        let shape = Shape { c: 6, h: 2, w: 2 };
        let mut delta = vec![0.0f32; 6 * 4];
        delta[2] = 2.0; // box plane
        delta[16] = -1.0; // objectness plane (plane 4)
        let stats = HeadStats::default();
        let _ = ();
        // mse: full sum of squares.
        let mut ym = y.clone();
        ym.train.iou_loss = IouLoss::Mse;
        assert_eq!(head_cost(&delta, shape, &ym, &stats), 5.0);
        // ciou: the classification part skips the four box entries.
        assert_eq!(head_cost(&delta, shape, &y, &stats), 1.0);
    }
}
