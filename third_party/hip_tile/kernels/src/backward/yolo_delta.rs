//! The `[yolo]` delta tensor on device — a port of `yolo_layer.cpp`'s `process_batch`,
//! `delta_yolo_box` and `delta_yolo_class` (see `_shared/darknet-semantics.md`, Training deltas).
//!
//! Two passes, one thread per unit, because that is how Darknet's data dependencies run:
//!
//! 1. **Every cell and anchor** (`cell` kernel): objectness is assigned first, zeroed when the
//!    best class-matching IoU exceeds `ignore_thresh`, and the (normally unreachable, since
//!    `truth_thresh = 1`) "best IoU beats truth_thresh" branch assigns objectness + class and
//!    accumulates box deltas.
//! 2. **Every ground-truth box** (`truth` kernel): the box is assigned to the best-matching anchor
//!    by shape IoU; if that anchor is in this head's `mask`, the box delta accumulates, the
//!    objectness is set to `1 − σ(obj)` and the matched class entry to `1 − σ(cls)`.
//!
//! Objectness/class entries are **assigned**, box entries are **accumulated**, and the caller
//! zeroes the delta buffer once per iteration — exactly Darknet's ordering. Per-truth statistics
//! (`count`, `Σiou`, `Σ(1−iou)`, `Σ(1−giou)`) go to a scratch row per truth so the host can form
//! the reported cost without atomics.
//!
//! Device `no_std` has no `atan`, so the CIoU aspect-ratio term uses a polynomial `atan`
//! (`atan_poly`); the host implementation this is checked against is `hip_darknet::train::deltas`.

use hip_tile::prelude::*;
use hip_tile::KernelArg;
use hip_tile_macros::kernel;

/// `iou_loss` selector, matching `hip_darknet::cfg::IouLoss`.
pub const IOU_LOSS_MSE: u32 = 0;
pub const IOU_LOSS_IOU: u32 = 1;
pub const IOU_LOSS_GIOU: u32 = 2;
pub const IOU_LOSS_DIOU: u32 = 3;
pub const IOU_LOSS_CIOU: u32 = 4;

pub const MAX_MASK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct YoloDeltaParams {
    pub classes: u32,
    /// Anchor pairs (`num` of them).
    pub num: u32,
    pub n_mask: u32,
    pub mask: [u32; MAX_MASK],
    pub scale_x_y: f32,
    pub iou_normalizer: f32,
    pub obj_normalizer: f32,
    pub cls_normalizer: f32,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub max_delta: f32,
    pub iou_loss: u32,
    pub lw: u32,
    pub lh: u32,
    pub net_w: u32,
    pub net_h: u32,
    pub batch: u32,
    pub max_truths: u32,
}

// SAFETY: `repr(C)` struct of `u32`/`f32`/`[u32; 8]`, all `Copy`.
unsafe impl KernelArg for YoloDeltaParams {}

impl YoloDeltaParams {
    /// Entries per anchor: `x, y, w, h, obj, class_0..`.
    pub fn entries(&self) -> u32 {
        self.classes + 5
    }
    /// Elements of one head plane.
    pub fn plane(&self) -> u32 {
        self.lw * self.lh
    }
}

#[derive(Clone, Copy)]
struct Bx {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[allow(dead_code)]
#[inline(always)]
fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[allow(dead_code)]
#[inline(always)]
fn inter(a: Bx, b: Bx) -> f32 {
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

#[allow(dead_code)]
#[inline(always)]
fn box_iou(a: Bx, b: Bx) -> f32 {
    let i = inter(a, b);
    if i == 0.0 {
        return 0.0;
    }
    i / (a.w * a.h + b.w * b.h - i)
}

#[allow(dead_code)]
#[inline(always)]
fn enclosing(a: Bx, b: Bx) -> (f32, f32) {
    let cw = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let ch = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    (cw, ch)
}

#[allow(dead_code)]
#[inline(always)]
fn box_giou(a: Bx, b: Bx) -> f32 {
    let (cw, ch) = enclosing(a, b);
    let c = cw * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let u = a.w * a.h + b.w * b.h - inter(a, b);
    iou - (c - u) / c
}

/// Polynomial `atan` (Devienne-style, |error| ≲ 1e-4 on the reduced range) — the device has no
/// `atan`, and CIoU needs `atan(w/h)`.
#[allow(dead_code)]
#[inline(always)]
fn atan_poly(x: f32) -> f32 {
    let (ax, neg, inv) = if x < 0.0 {
        (-x, true, false)
    } else if x > 1.0 {
        (x, false, true)
    } else {
        (x, false, false)
    };
    let mut a = if inv { 1.0 / ax } else { ax };
    let z = a * a;
    let mut r = a
        * (0.999_866_f32
            + z * (-0.330_299_5_f32
                + z * (0.180_141_f32 + z * (-0.085_133_f32 + z * 0.020_835_1_f32))));
    if inv {
        r = core::f32::consts::FRAC_PI_2 - r;
    }
    let _ = &mut a;
    if neg { -r } else { r }
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn box_diou(a: Bx, b: Bx) -> f32 {
    let (cw, ch) = enclosing(a, b);
    let c = cw * cw + ch * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    iou - pow_06(d / c)
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn pow_06(v: f32) -> f32 {
    // `(d/c)^0.6` on the device: `exp(0.6 ln v)` without `powf`.
    if v <= 0.0 {
        return 0.0;
    }
    expf(0.6 * lnf(v))
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn box_ciou(a: Bx, b: Bx) -> f32 {
    let (cw, ch) = enclosing(a, b);
    let c = cw * cw + ch * ch;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    let d = u / c;
    let ar_gt = b.w / b.h;
    let ar_pred = a.w / a.h;
    let ar_loss = 4.0 / 9.869_604 * (atan_poly(ar_gt) - atan_poly(ar_pred)) * (atan_poly(ar_gt) - atan_poly(ar_pred));
    let alpha = ar_loss / (1.0 - iou + ar_loss + 0.000001);
    iou - (d + alpha * ar_loss)
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub(crate) fn expf(x: f32) -> f32 {
    core::intrinsics::exp::<f32>(x)
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn lnf(x: f32) -> f32 {
    core::intrinsics::log::<f32>(x)
}

/// `get_yolo_box` on the activated head tensor (`yolo_layer.cpp:54-88`).
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn get_box(head: *const f32, entry: u32, cell: u32, ax: f32, ay: f32, p: &YoloDeltaParams) -> Bx {
    let plane = p.plane();
    let at = |e: u32| unsafe { *head.add((entry + e * plane + cell) as usize) };
    let (col, row) = (cell % p.lw, cell / p.lw);
    let tw = finite(at(2)).clamp(-10.0, 10.0);
    let th = finite(at(3)).clamp(-10.0, 10.0);
    Bx {
        x: finite((col as f32 + at(0)) / p.lw as f32),
        y: finite((row as f32 + at(1)) / p.lh as f32),
        w: finite((expf(tw) * ax / p.net_w as f32).max(1e-6)),
        h: finite((expf(th) * ay / p.net_h as f32).max(1e-6)),
    }
}

/// `delta_yolo_class` (`yolo_layer.cpp:352`). A cell whose truth-class delta is already set only
/// has that one refreshed; otherwise every class gets `(n == class) - output`, which pushes the
/// other classes of a positive cell toward 0. (`cls_normalizer` only matters there together with
/// `classes_multipliers`, which this port does not have.)
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn delta_class(head: *const f32, delta: *mut f32, class_index: u32, class: u32, classes: u32, plane: u32) {
    let ti = (class_index + class * plane) as usize;
    unsafe {
        if *delta.add(ti) != 0.0 {
            *delta.add(ti) = 1.0 - *head.add(ti);
            return;
        }
        for n in 0..classes {
            let ci = (class_index + n * plane) as usize;
            let target = if n == class { 1.0 } else { 0.0 };
            let d = target - *head.add(ci);
            if d == d && d.abs() < f32::MAX {
                *delta.add(ci) = d;
            }
        }
    }
}

/// `delta_yolo_box` (`yolo_layer.cpp:154-310`). `accumulate` follows the caller's convention:
/// Darknet accumulates box entries in both passes.
#[cfg(target_arch = "amdgpu")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn delta_box(
    truth: Bx,
    head: *const f32,
    delta: *mut f32,
    entry: u32,
    cell: u32,
    ax: f32,
    ay: f32,
    p: &YoloDeltaParams,
) -> (f32, f32) {
    let plane = p.plane();
    let idx = entry + cell;
    let pred = get_box(head, entry, cell, ax, ay, p);
    let iou = box_iou(pred, truth);
    let giou = box_giou(pred, truth);
    if p.iou_loss == IOU_LOSS_MSE {
        let (col, row) = ((cell % p.lw) as f32, (cell / p.lw) as f32);
        let tx = truth.x * p.lw as f32 - col;
        let ty = truth.y * p.lh as f32 - row;
        let tw = lnf(truth.w * p.net_w as f32 / ax);
        let th = lnf(truth.h * p.net_h as f32 / ay);
        unsafe {
            *delta.add(idx as usize) += p.iou_normalizer * (tx - *head.add(idx as usize));
            *delta.add((idx + plane) as usize) += p.iou_normalizer * (ty - *head.add((idx + plane) as usize));
            *delta.add((idx + 2 * plane) as usize) += p.iou_normalizer * (tw - *head.add((idx + 2 * plane) as usize));
            *delta.add((idx + 3 * plane) as usize) += p.iou_normalizer * (th - *head.add((idx + 3 * plane) as usize));
        }
        return (iou, giou);
    }
    let (mut dx, mut dy, mut dw, mut dh) = dx_box_iou(pred, truth, p.iou_loss);
    let tw = finite(unsafe { *head.add((idx + 2 * plane) as usize) }).clamp(-10.0, 10.0);
    let th = finite(unsafe { *head.add((idx + 3 * plane) as usize) }).clamp(-10.0, 10.0);
    dw *= expf(tw);
    dh *= expf(th);
    dx = finite(dx * p.iou_normalizer);
    dy = finite(dy * p.iou_normalizer);
    dw = finite(dw * p.iou_normalizer);
    dh = finite(dh * p.iou_normalizer);
    if p.max_delta != f32::MAX {
        dx = dx.clamp(-p.max_delta, p.max_delta);
        dy = dy.clamp(-p.max_delta, p.max_delta);
        dw = dw.clamp(-p.max_delta, p.max_delta);
        dh = dh.clamp(-p.max_delta, p.max_delta);
    }
    unsafe {
        *delta.add(idx as usize) += dx;
        *delta.add((idx + plane) as usize) += dy;
        *delta.add((idx + 2 * plane) as usize) += dw;
        *delta.add((idx + 3 * plane) as usize) += dh;
    }
    (iou, giou)
}

/// `dx_box_iou` (`box.cpp:1410-1595`), with the same slot meanings (`dt, db, dl, dr` = `dx, dy,
/// dw, dh` in `delta_yolo_box`).
#[cfg(target_arch = "amdgpu")]
fn dx_box_iou(pred: Bx, truth: Bx, iou_loss: u32) -> (f32, f32, f32, f32) {
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
    let s = (pred.x - truth.x) * (pred.x - truth.x) + (pred.y - truth.y) * (pred.y - truth.y);
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

    let (mut p_dt, mut p_db, mut p_dl, mut p_dr) = (0.0f32, 0.0, 0.0, 0.0);
    if u > 0.0 {
        let uu = u * u;
        p_dt = (u * di_dt - i * du_dt) / uu;
        p_db = (u * di_db - i * du_db) / uu;
        p_dl = (u * di_dl - i * du_dl) / uu;
        p_dr = (u * di_dr - i * du_dr) / uu;
    }
    let (t_lt_b, l_lt_r) = (pred_t < pred_b, pred_l < pred_r);
    let (t, b) = (p_dt, p_db);
    p_dt = if t_lt_b { t } else { b };
    p_db = if t_lt_b { b } else { t };
    let (l, r) = (p_dl, p_dr);
    p_dl = if l_lt_r { l } else { r };
    p_dr = if l_lt_r { r } else { l };

    if iou_loss == IOU_LOSS_GIOU {
        let gg = giou_cw * giou_ch;
        if gg > 0.0 {
            let g2 = gg * gg;
            p_dt += (gg * du_dt - u * dc_dt) / g2;
            p_db += (gg * du_db - u * dc_db) / g2;
            p_dl += (gg * du_dl - u * dc_dl) / g2;
            p_dr += (gg * du_dr - u * dc_dr) / g2;
        }
        if iw <= 0.0 || ih <= 0.0 {
            let g2 = gg * gg;
            p_dt = (gg * du_dt - u * dc_dt) / g2;
            p_db = (gg * du_db - u * dc_db) / g2;
            p_dl = (gg * du_dl - u * dc_dl) / g2;
            p_dr = (gg * du_dr - u * dc_dr) / g2;
        }
    }

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

    let dcw_dx = dcr_dx - dcl_dx;
    let dcw_dw = dcr_dw - dcl_dw;
    let dch_dy = dcb_dy - dct_dy;
    let dch_dh = dcb_dh - dct_dh;

    let mut p_dx = p_dl + p_dr;
    let mut p_dy = p_dt + p_db;
    let mut p_dw = p_dr - p_dl;
    let mut p_dh = p_db - p_dt;

    let distance = |p_dx: &mut f32, p_dy: &mut f32, p_dw: &mut f32, p_dh: &mut f32| {
        let cc = c * c;
        let tdx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / cc;
        let tdy = (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / cc;
        let tdw = (2.0 * cw * dcw_dw) * s / cc;
        let tdh = (2.0 * ch * dch_dh) * s / cc;
        if c > 0.0 {
            *p_dx += tdx;
            *p_dy += tdy;
            *p_dw += tdw;
            *p_dh += tdh;
        }
        if iw <= 0.0 || ih <= 0.0 {
            *p_dx = tdx;
            *p_dy = tdy;
            *p_dw = tdw;
            *p_dh = tdh;
        }
    };

    if iou_loss == IOU_LOSS_DIOU {
        distance(&mut p_dx, &mut p_dy, &mut p_dw, &mut p_dh);
    }
    if iou_loss == IOU_LOSS_CIOU {
        let k = 4.0 / 9.869_604;
        let ar_gt = truth.w / truth.h;
        let ar_pred = pred.w / pred.h;
        let d = atan_poly(ar_gt) - atan_poly(ar_pred);
        let ar_loss = k * d * d;
        let alpha = ar_loss / (1.0 - i / u + ar_loss + 0.000001);
        let ar_dw = 2.0 * k * d * pred.h;
        let ar_dh = -2.0 * k * d * pred.w;
        let cc = c * c;
        let tdx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / cc;
        let tdy = (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / cc;
        let tdw = (2.0 * cw * dcw_dw) * s / cc + alpha * ar_dw;
        let tdh = (2.0 * ch * dch_dh) * s / cc + alpha * ar_dh;
        if c > 0.0 {
            p_dx += tdx;
            p_dy += tdy;
            p_dw += tdw;
            p_dh += tdh;
        }
        if iw <= 0.0 || ih <= 0.0 {
            p_dx = tdx;
            p_dy = tdy;
            p_dw = tdw;
            p_dh = tdh;
        }
    }
    (p_dx, p_dy, p_dw, p_dh)
}

#[cfg(target_arch = "amdgpu")]
fn flat() -> u32 {
    block_id().1 * BLOCK_THREADS + thread_id()
}

/// Pass A: objectness/class assignment for every cell and anchor of one head.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn yolo_delta_cell(
    head: Tensor<f32>,
    anchors: Tensor<f32>,
    truths: Tensor<f32>,
    n_truths: Tensor<f32>,
    delta: TensorMut<f32>,
    p: YoloDeltaParams,
) {
    let plane = p.plane();
    let entries = p.entries();
    let cells = plane;
    let i = flat();
    if i >= p.batch * cells {
        return;
    }
    let b = i / cells;
    let cell = i - b * cells;
    let count = unsafe { *n_truths.ptr().add(b as usize) };
    // One image's head is `n_mask` anchors of `entries` planes: that is the batch stride.
    let per_image = (p.n_mask * entries * plane) as usize;
    let hbase = unsafe { head.ptr().add(b as usize * per_image) };
    let dbase = unsafe { delta.ptr().add(b as usize * per_image) };
    let tbase = unsafe { truths.ptr().add((b * p.max_truths) as usize * 6) };
    let (col, row) = (cell % p.lw, cell / p.lw);

    for n in 0..p.n_mask {
        let entry = n * entries * plane;
        let ax = unsafe { *anchors.ptr().add((2 * p.mask[n as usize]) as usize) };
        let ay = unsafe { *anchors.ptr().add((2 * p.mask[n as usize] + 1) as usize) };
        let pred = get_box(hbase, entry, cell, ax, ay, &p);
        let obj_index = entry + 4 * plane + cell;
        let class_index = entry + 5 * plane + cell;

        // `compare_yolo_class`: any class probability above 0.25.
        let mut class_seen = false;
        for c in 0..p.classes {
            if unsafe { *hbase.add((class_index + c * plane) as usize) } > 0.25 {
                class_seen = true;
            }
        }
        let mut best_iou = 0.0f32;
        let mut best_match = 0.0f32;
        let mut best_t = u32::MAX;
        for t in 0..count as u32 {
            let tb = Bx {
                x: unsafe { *tbase.add((t * 6 + 1) as usize) },
                y: unsafe { *tbase.add((t * 6 + 2) as usize) },
                w: unsafe { *tbase.add((t * 6 + 3) as usize) },
                h: unsafe { *tbase.add((t * 6 + 4) as usize) },
            };
            let iou = box_iou(pred, tb);
            if iou > best_match && class_seen {
                best_match = iou;
            }
            if iou > best_iou {
                best_iou = iou;
                best_t = t;
            }
        }

        unsafe { *dbase.add(obj_index as usize) = p.obj_normalizer * (0.0 - *hbase.add(obj_index as usize)) };
        if best_match > p.ignore_thresh {
            unsafe { *dbase.add(obj_index as usize) = 0.0 };
        }
        if best_iou > p.truth_thresh && best_t != u32::MAX {
            let class = unsafe { *tbase.add((best_t * 6) as usize) } as u32;
            let tb = Bx {
                x: unsafe { *tbase.add((best_t * 6 + 1) as usize) },
                y: unsafe { *tbase.add((best_t * 6 + 2) as usize) },
                w: unsafe { *tbase.add((best_t * 6 + 3) as usize) },
                h: unsafe { *tbase.add((best_t * 6 + 4) as usize) },
            };
            unsafe { *dbase.add(obj_index as usize) = p.obj_normalizer * (1.0 - *hbase.add(obj_index as usize)) };
            if class < p.classes {
                delta_class(hbase, dbase, class_index, class, p.classes, plane);
            }
            delta_box(tb, hbase, dbase, entry, cell, ax, ay, &p);
        }
    }
}

/// Pass B: every ground-truth box lands on its responsible anchor.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn yolo_delta_truth(
    head: Tensor<f32>,
    anchors: Tensor<f32>,
    truths: Tensor<f32>,
    n_truths: Tensor<f32>,
    delta: TensorMut<f32>,
    stats: TensorMut<f32>,
    p: YoloDeltaParams,
) {
    let plane = p.plane();
    let entries = p.entries();
    let i = flat();
    if i >= p.batch * p.max_truths {
        return;
    }
    let b = i / p.max_truths;
    let t = i - b * p.max_truths;
    let count = unsafe { *n_truths.ptr().add(b as usize) };
    if t >= count as u32 {
        return;
    }
    let tbase = unsafe { truths.ptr().add((b * p.max_truths) as usize * 6) };
    let class = unsafe { *tbase.add((t * 6) as usize) } as u32;
    let tb = Bx {
        x: unsafe { *tbase.add((t * 6 + 1) as usize) },
        y: unsafe { *tbase.add((t * 6 + 2) as usize) },
        w: unsafe { *tbase.add((t * 6 + 3) as usize) },
        h: unsafe { *tbase.add((t * 6 + 4) as usize) },
    };
    if class >= p.classes {
        return;
    }

    // Best anchor by shape IoU (`yolo_anchor_match_score`).
    let mut best_iou = 0.0f32;
    let mut best_anchor = 0u32;
    for a in 0..p.num {
        let anchor = Bx {
            x: 0.0,
            y: 0.0,
            w: unsafe { *anchors.ptr().add((2 * a) as usize) } / p.net_w as f32,
            h: unsafe { *anchors.ptr().add((2 * a + 1) as usize) } / p.net_h as f32,
        };
        let shifted = Bx { x: 0.0, y: 0.0, ..tb };
        let iou = box_iou(anchor, shifted);
        if iou > best_iou {
            best_iou = iou;
            best_anchor = a;
        }
    }
    let mut pos = u32::MAX;
    for n in 0..p.n_mask {
        if p.mask[n as usize] == best_anchor {
            pos = n;
        }
    }
    if pos == u32::MAX {
        return;
    }

    let col = ((tb.x * p.lw as f32) as u32).min(p.lw - 1);
    let row = ((tb.y * p.lh as f32) as u32).min(p.lh - 1);
    let cell = row * p.lw + col;
    let entry = pos * entries * plane;
    let obj_index = entry + 4 * plane + cell;
    let class_index = entry + 5 * plane + cell;
    // One image's head is `n_mask` anchors of `entries` planes: that is the batch stride.
    let per_image = (p.n_mask * entries * plane) as usize;
    let hbase = unsafe { head.ptr().add(b as usize * per_image) };
    let dbase = unsafe { delta.ptr().add(b as usize * per_image) };
    let ax = unsafe { *anchors.ptr().add((2 * best_anchor) as usize) };
    let ay = unsafe { *anchors.ptr().add((2 * best_anchor + 1) as usize) };
    let (iou, giou) = delta_box(tb, hbase, dbase, entry, cell, ax, ay, &p);
    unsafe {
        *dbase.add(obj_index as usize) = p.obj_normalizer * (1.0 - *hbase.add(obj_index as usize));
        delta_class(hbase, dbase, class_index, class, p.classes, plane);
        // Per-truth statistics for the reported cost: count, Σiou, Σ(1−iou), Σ(1−giou).
        let sbase = ((b * p.max_truths + t) * 4) as usize;
        *stats.ptr().add(sbase) = 1.0;
        *stats.ptr().add(sbase + 1) = finite(iou);
        *stats.ptr().add(sbase + 2) = 1.0 - finite(iou);
        *stats.ptr().add(sbase + 3) = 1.0 - finite(giou);
    }
}

/// `truths` is `[batch*max_truths x 6]` as `(class, x, y, w, h, 0)` in normalized coordinates,
/// `n_truths` is `[batch]`, `stats` is `[batch*max_truths x 4]` (written, not accumulated).
/// `delta` must be zeroed by the caller.
#[cfg(not(target_arch = "amdgpu"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_yolo_delta(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    head: hip_tile::Tensor<f32>,
    anchors: hip_tile::Tensor<f32>,
    truths: hip_tile::Tensor<f32>,
    n_truths: hip_tile::Tensor<f32>,
    delta: hip_tile::TensorMut<f32>,
    stats: hip_tile::TensorMut<f32>,
    p: &YoloDeltaParams,
) -> hip_rt::Result<()> {
    // The head holds `n_mask` anchors of `classes + 5` planes each.
    let planes = p.n_mask * p.entries();
    assert_eq!(head.rows(), planes, "yolo_delta: bad head rows");
    assert_eq!(head.cols(), p.plane(), "yolo_delta: bad head cols");
    assert_eq!(delta.rows(), planes, "yolo_delta: bad delta rows");
    assert_eq!(delta.cols(), p.plane(), "yolo_delta: bad delta cols");
    assert_eq!(anchors.cols(), 2 * p.num, "yolo_delta: bad anchors");
    assert_eq!(truths.cols(), 6, "yolo_delta: bad truths");
    assert_eq!(n_truths.cols(), p.batch, "yolo_delta: bad counts");
    assert_eq!(stats.cols(), 4, "yolo_delta: bad stats");
    assert!((p.n_mask as usize) <= MAX_MASK, "yolo_delta: mask too long");
    let cells = p.batch * p.plane();
    yolo_delta_cell::launch(module, stream, (cells.div_ceil(BLOCK_THREADS), 1), head, anchors, truths, n_truths, delta, *p)?;
    let units = p.batch * p.max_truths;
    yolo_delta_truth::launch(
        module,
        stream,
        (units.div_ceil(BLOCK_THREADS), 1),
        head,
        anchors,
        truths,
        n_truths,
        delta,
        stats,
        *p,
    )
}
