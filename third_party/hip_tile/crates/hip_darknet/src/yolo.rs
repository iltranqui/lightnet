//! YOLO head decode and NMS, ported from Darknet (`yolo_layer.cpp`, `box.cpp`).

use crate::cfg::{NmsKind, YoloCfg};
use crate::model::{LayerKind, Model, Shape};

/// Box with normalized center and size (fractions of the image).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// One surviving (box, class) pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class: usize,
    /// `objectness * class_prob`.
    pub prob: f32,
    pub objectness: f32,
    pub bbox: BBox,
}

impl Detection {
    /// `(left_x, top_y, width, height)` in pixels of a `w` x `h` image.
    pub fn pixel_rect(&self, w: usize, h: usize) -> (f32, f32, f32, f32) {
        let b = self.bbox;
        ((b.x - b.w / 2.0) * w as f32, (b.y - b.h / 2.0) * h as f32, b.w * w as f32, b.h * h as f32)
    }
}

/// A candidate box with per-class probabilities (zeroed below threshold / when suppressed).
#[derive(Debug, Clone)]
pub struct RawDetection {
    pub bbox: BBox,
    pub objectness: f32,
    pub probs: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectOptions {
    pub thresh: f32,
    /// Darknet's `detector test` default is 0.45.
    pub nms_thresh: f32,
    /// Overrides the `[yolo]` heads' `nms_kind` when `Some`.
    pub nms_kind: Option<NmsKind>,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions { thresh: 0.25, nms_thresh: 0.45, nms_kind: None }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Decode one `[yolo]` input tensor (raw conv output, CHW with `mask.len() * (classes + 5)` channels).
/// Layout per anchor `n`: planes `x, y, w, h, objectness, class_0..`. Mirrors `get_yolo_detections`.
pub fn decode_layer(
    raw: &[f32],
    shape: Shape,
    y: &YoloCfg,
    net_w: usize,
    net_h: usize,
    thresh: f32,
    out: &mut Vec<RawDetection>,
) {
    let (lw, lh) = (shape.w, shape.h);
    let plane = lw * lh;
    let per_anchor = (y.classes + 5) * plane;
    let s = y.scale_x_y;
    for loc in 0..plane {
        let (row, col) = (loc / lw, loc % lw);
        for (n, &m) in y.mask.iter().enumerate() {
            let at = |entry: usize| raw[n * per_anchor + entry * plane + loc];
            let objectness = sigmoid(at(4));
            if objectness <= thresh {
                continue;
            }
            // scal_add: x' = sigmoid(x) * s - 0.5 * (s - 1)
            let tx = sigmoid(at(0)) * s - 0.5 * (s - 1.0);
            let ty = sigmoid(at(1)) * s - 0.5 * (s - 1.0);
            let tw = finite(at(2)).clamp(-10.0, 10.0);
            let th = finite(at(3)).clamp(-10.0, 10.0);
            let bbox = BBox {
                x: finite((col as f32 + tx) / lw as f32),
                y: finite((row as f32 + ty) / lh as f32),
                w: finite(tw.exp() * y.anchors[2 * m] / net_w as f32).max(1e-6),
                h: finite(th.exp() * y.anchors[2 * m + 1] / net_h as f32).max(1e-6),
            };
            let probs = (0..y.classes)
                .map(|j| {
                    let p = objectness * sigmoid(at(5 + j));
                    if p > thresh { p } else { 0.0 }
                })
                .collect();
            out.push(RawDetection { bbox, objectness, probs });
        }
    }
}

fn overlap(x1: f32, w1: f32, x2: f32, w2: f32) -> f32 {
    let left = (x1 - w1 / 2.0).max(x2 - w2 / 2.0);
    let right = (x1 + w1 / 2.0).min(x2 + w2 / 2.0);
    right - left
}

fn intersection(a: &BBox, b: &BBox) -> f32 {
    let w = overlap(a.x, a.w, b.x, b.w);
    if w <= 0.0 {
        return 0.0;
    }
    let h = overlap(a.y, a.h, b.y, b.h);
    if h <= 0.0 {
        return 0.0;
    }
    w * h
}

/// Intersection over union (`box_iou`).
pub fn box_iou(a: &BBox, b: &BBox) -> f32 {
    let i = intersection(a, b);
    if i == 0.0 {
        return 0.0;
    }
    i / (a.w * a.h + b.w * b.h - i)
}

/// IoU minus `(center distance^2 / enclosing diagonal^2)^beta` (`box_diou` uses beta 0.6, `box_diounms` the cfg's).
pub fn box_diou(a: &BBox, b: &BBox, beta: f32) -> f32 {
    let w = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let h = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let c = w * w + h * h;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    iou - (d / c).powf(beta)
}

/// Complete IoU (`box_ciou`, `box.cpp:1088`): IoU minus the squared centre distance over the
/// enclosing diagonal, minus the aspect-ratio term.
pub fn box_ciou(a: &BBox, b: &BBox) -> f32 {
    let w = (a.x + a.w / 2.0).max(b.x + b.w / 2.0) - (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let h = (a.y + a.h / 2.0).max(b.y + b.h / 2.0) - (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let c = w * w + h * h;
    let iou = box_iou(a, b);
    if c == 0.0 {
        return iou;
    }
    let d = ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)) / c;
    let pi2 = std::f32::consts::PI * std::f32::consts::PI;
    let ar_loss = 4.0 / pi2 * ((b.w / b.h).atan() - (a.w / a.h).atan()).powi(2);
    let alpha = ar_loss / (1.0 - iou + ar_loss + 0.000001);
    iou - (d + alpha * ar_loss)
}

/// Per-class greedy NMS in place (`do_nms_sort` / `diounms_sort`): sort by class prob, then zero the
/// probability of every later box whose overlap with a kept box exceeds `thresh`.
pub fn nms(dets: &mut Vec<RawDetection>, classes: usize, thresh: f32, kind: NmsKind, beta: f32) {
    dets.retain(|d| d.objectness != 0.0);
    for k in 0..classes {
        dets.sort_by(|a, b| b.probs[k].total_cmp(&a.probs[k]));
        for i in 0..dets.len() {
            if dets[i].probs[k] == 0.0 {
                continue;
            }
            let a = dets[i].bbox;
            for d in &mut dets[i + 1..] {
                let ov = match kind {
                    NmsKind::Default => box_iou(&a, &d.bbox),
                    NmsKind::Greedy => box_diou(&a, &d.bbox, 0.6),
                    NmsKind::Diou => box_diou(&a, &d.bbox, beta),
                    NmsKind::Ciou => box_ciou(&a, &d.bbox),
                };
                if ov > thresh {
                    d.probs[k] = 0.0;
                }
            }
        }
    }
}

/// Decode + NMS given the raw inputs of the model's `[yolo]` layers (in `model.yolo_layers()` order).
/// Boxes are normalized to the image (no letterboxing).
pub fn detect_heads(model: &Model, heads: &[&[f32]], opts: &DetectOptions) -> Vec<Detection> {
    let ylayers = model.yolo_layers();
    assert_eq!(heads.len(), ylayers.len(), "one tensor per [yolo] layer");
    let mut raw = vec![];
    let mut last = None;
    for (&li, head) in ylayers.iter().zip(heads) {
        let layer = &model.layers[li];
        let LayerKind::Yolo(y) = &layer.kind else { unreachable!() };
        decode_layer(head, layer.out_shape, y, model.net.width, model.net.height, opts.thresh, &mut raw);
        last = Some(y);
    }
    let Some(y) = last else { return vec![] };
    nms(&mut raw, y.classes, opts.nms_thresh, opts.nms_kind.unwrap_or(y.nms_kind), y.beta_nms);
    let mut out = vec![];
    for d in &raw {
        for (class, &prob) in d.probs.iter().enumerate() {
            if prob > opts.thresh {
                out.push(Detection { class, prob, objectness: d.objectness, bbox: d.bbox });
            }
        }
    }
    out.sort_by(|a, b| b.prob.total_cmp(&a.prob));
    out
}

/// `detect_heads` on the full per-layer outputs of `reference::forward`.
pub fn detect(model: &Model, layer_outputs: &[Vec<f32>], opts: &DetectOptions) -> Vec<Detection> {
    let heads: Vec<&[f32]> = model.yolo_layers().iter().map(|&i| layer_outputs[i].as_slice()).collect();
    detect_heads(model, &heads, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(x: f32, y: f32, w: f32, h: f32) -> BBox {
        BBox { x, y, w, h }
    }

    #[test]
    fn iou_hand_computed() {
        // inter = 0.3 * 0.4, union = 2 * 0.16 - 0.12.
        let v = box_iou(&bb(0.5, 0.5, 0.4, 0.4), &bb(0.6, 0.5, 0.4, 0.4));
        assert!((v - 0.6).abs() < 1e-6);
        assert_eq!(box_iou(&bb(0.2, 0.2, 0.1, 0.1), &bb(0.8, 0.8, 0.1, 0.1)), 0.0);
        assert!((box_iou(&bb(0.5, 0.5, 0.3, 0.3), &bb(0.5, 0.5, 0.3, 0.3)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn diou_penalty() {
        // Boxes at x=.5/.65: iou = .1/.22; enclosing 0.55x0.4 -> c=.4625; d=.0225.
        let (a, b) = (bb(0.5, 0.5, 0.4, 0.4), bb(0.65, 0.5, 0.4, 0.4));
        let want = 0.1 / 0.22 - (0.0225f32 / 0.4625).powf(0.6);
        assert!((box_diou(&a, &b, 0.6) - want).abs() < 1e-5);
        assert!(want < 0.45 && box_iou(&a, &b) > 0.45);
    }

    fn det(b: BBox, p: &[f32]) -> RawDetection {
        RawDetection { bbox: b, objectness: p.iter().cloned().fold(0.0, f32::max), probs: p.to_vec() }
    }

    #[test]
    fn nms_per_class() {
        let mut d = vec![
            det(bb(0.5, 0.5, 0.4, 0.4), &[0.6, 0.0]),
            det(bb(0.52, 0.5, 0.4, 0.4), &[0.8, 0.0]), // higher score, overlaps the first
            det(bb(0.52, 0.5, 0.4, 0.4), &[0.0, 0.7]), // other class: untouched
            det(bb(0.9, 0.9, 0.1, 0.1), &[0.3, 0.0]),  // far away
        ];
        nms(&mut d, 2, 0.45, NmsKind::Default, 0.6);
        let c0: Vec<f32> = d.iter().map(|x| x.probs[0]).filter(|&p| p > 0.0).collect();
        assert_eq!(c0, vec![0.8, 0.3]);
        assert_eq!(d.iter().filter(|x| x.probs[1] > 0.0).count(), 1);
    }

    #[test]
    fn nms_greedy_uses_diou() {
        let mk = || vec![det(bb(0.5, 0.5, 0.4, 0.4), &[0.9]), det(bb(0.65, 0.5, 0.4, 0.4), &[0.8])];
        let mut a = mk();
        nms(&mut a, 1, 0.45, NmsKind::Default, 0.6);
        assert_eq!(a.iter().filter(|x| x.probs[0] > 0.0).count(), 1); // iou .4545 > .45
        let mut b = mk();
        nms(&mut b, 1, 0.45, NmsKind::Greedy, 0.6);
        assert_eq!(b.iter().filter(|x| x.probs[0] > 0.0).count(), 2); // diou .29 < .45
    }

    fn head_cfg(scale: f32) -> YoloCfg {
        YoloCfg {
            classes: 2,
            mask: vec![1],
            anchors: vec![10.0, 10.0, 32.0, 16.0],
            scale_x_y: scale,
            nms_kind: NmsKind::Greedy,
            beta_nms: 0.6,
            train: crate::cfg::YoloTrainCfg::default(),
        }
    }

    /// 2x2 grid, one anchor (mask index 1), classes 2 -> 7 planes of 4.
    fn tensor(cell: usize, tx: f32, obj: f32, c0: f32, c1: f32) -> Vec<f32> {
        let mut t = vec![0.0; 7 * 4];
        for loc in 0..4 {
            t[4 * 4 + loc] = -10.0; // objectness off
        }
        t[cell] = tx;
        t[4 + cell] = 0.0;
        t[4 * 4 + cell] = obj;
        t[5 * 4 + cell] = c0;
        t[6 * 4 + cell] = c1;
        t
    }

    #[test]
    fn decode_hand_made() {
        let shape = Shape { c: 7, h: 2, w: 2 };
        let t = tensor(1, 0.0, 10.0, 0.0, -10.0); // cell col=1,row=0
        let mut out = vec![];
        decode_layer(&t, shape, &head_cfg(1.05), 64, 32, 0.25, &mut out);
        assert_eq!(out.len(), 1);
        let d = &out[0];
        // sigmoid(0)*1.05 - 0.025 = 0.5 -> x = (1 + .5)/2, y = (0 + .5)/2.
        assert!((d.bbox.x - 0.75).abs() < 1e-6 && (d.bbox.y - 0.25).abs() < 1e-6);
        // tw = th = 0 -> w = anchor(mask 1) / net: 32/64, 16/32.
        assert!((d.bbox.w - 0.5).abs() < 1e-6 && (d.bbox.h - 0.5).abs() < 1e-6);
        assert!((d.probs[0] - sigmoid(10.0) * 0.5).abs() < 1e-6);
        assert_eq!(d.probs[1], 0.0); // below threshold
    }

    #[test]
    fn decode_scale_x_y_and_clamp() {
        let shape = Shape { c: 7, h: 2, w: 2 };
        let mut t = tensor(0, 30.0, 10.0, 10.0, 10.0);
        t[2 * 4] = 50.0; // tw clamped to 10
        let mut out = vec![];
        decode_layer(&t, shape, &head_cfg(1.05), 64, 32, 0.25, &mut out);
        // sigmoid -> 1: 1.05 - 0.025 = 1.025 -> x = 1.025 / 2 (x center can exceed the cell).
        assert!((out[0].bbox.x - 0.5125).abs() < 1e-5);
        assert!((out[0].bbox.w - 10f32.exp() * 32.0 / 64.0).abs() < 1e-1);
        // scale 1 leaves plain sigmoid.
        let mut out = vec![];
        decode_layer(&t, shape, &head_cfg(1.0), 64, 32, 0.25, &mut out);
        assert!((out[0].bbox.x - 0.5).abs() < 1e-5);
    }
}
