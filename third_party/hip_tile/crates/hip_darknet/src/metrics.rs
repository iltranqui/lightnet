//! mAP evaluation on a labelled list, through the verified inference path.
//!
//! The detector output comes from [`crate::yolo`] (decode + NMS) after folding the trained BN
//! statistics into a [`crate::model::Model`], so evaluation uses exactly the same code as
//! `hip_darknet --gpu` inference (the caller passes the CPU or GPU detector). Matching is per class, greedy by score, VOC-style average precision.
//!
//! Darknet's `map.csv` logs `iou = 0.5`; a 0.5:0.95 sweep is computed next to it because the
//! reference file has no column for it (`ROADMAP.md`).

use crate::data::{BoxLabel, Sample};
use crate::error::Result;
use crate::image::Image;
use crate::yolo::Detection;

/// Result of one evaluation pass.
#[derive(Debug, Clone, Default)]
pub struct MapResult {
    /// Mean AP over the classes that have ground truth, at IoU 0.5.
    pub map_50: f32,
    /// Same, averaged over IoU thresholds 0.5, 0.55, …, 0.95.
    pub map_50_95: f32,
    /// Per-class AP at IoU 0.5 (0 when the class has no ground truth).
    pub per_class: Vec<f32>,
    pub n_images: usize,
    pub n_gt: usize,
    pub n_det: usize,
    /// Per-class detection counts at [`REPORT_CONFIDENCE`] and IoU 0.5.
    pub counts: Vec<ClassCounts>,
}

/// Confidence above which a detection counts for the TP / FP / FN table (Darknet's `thresh_calc_avg_iou`).
pub const REPORT_CONFIDENCE: f32 = 0.25;

/// True / false positives and false negatives of one class. A detection-only task has no
/// meaningful true-negative count (every empty box position would be one), so there is none.
#[derive(Debug, Clone, Default)]
pub struct ClassCounts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub gt: usize,
    /// Mean IoU of the matched detections (0 when there are none).
    pub avg_iou: f32,
}

impl ClassCounts {
    pub fn precision(&self) -> f32 {
        if self.tp + self.fp == 0 { 0.0 } else { self.tp as f32 / (self.tp + self.fp) as f32 }
    }

    pub fn recall(&self) -> f32 {
        if self.gt == 0 { 0.0 } else { self.tp as f32 / self.gt as f32 }
    }

    pub fn f1(&self) -> f32 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

/// Greedy match of one class's detections (sorted by score) at IoU 0.5, above [`REPORT_CONFIDENCE`].
fn class_counts(dets: &[&Det], gts: &[&(usize, BoxLabel)]) -> ClassCounts {
    let mut matched = vec![false; gts.len()];
    let (mut tp, mut fp, mut iou_sum) = (0usize, 0usize, 0.0f32);
    for d in dets.iter().filter(|d| d.score >= REPORT_CONFIDENCE) {
        let mut best = 0.5f32;
        let mut best_i = usize::MAX;
        for (i, (image, gt)) in gts.iter().enumerate() {
            if *image != d.image || matched[i] {
                continue;
            }
            let v = iou(d.bbox, *gt);
            if v >= best {
                best = v;
                best_i = i;
            }
        }
        if best_i == usize::MAX {
            fp += 1;
        } else {
            matched[best_i] = true;
            tp += 1;
            iou_sum += best;
        }
    }
    ClassCounts { tp, fp, fn_: gts.len() - tp, gt: gts.len(), avg_iou: if tp == 0 { 0.0 } else { iou_sum / tp as f32 } }
}

fn iou(a: BoxLabel, b: BoxLabel) -> f32 {
    let (al, ar, at, ab) = (a.x - a.w / 2.0, a.x + a.w / 2.0, a.y - a.h / 2.0, a.y + a.h / 2.0);
    let (bl, br, bt, bb) = (b.x - b.w / 2.0, b.x + b.w / 2.0, b.y - b.h / 2.0, b.y + b.h / 2.0);
    let w = ar.min(br) - al.max(bl);
    let h = ab.min(bb) - at.max(bt);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    inter / (a.w * a.h + b.w * b.h - inter)
}

/// One detection flattened for matching: which image, which class, its score and box.
struct Det {
    image: usize,
    class: usize,
    score: f32,
    bbox: BoxLabel,
}

impl From<&Detection> for Det {
    fn from(d: &Detection) -> Det {
        Det {
            image: 0,
            class: d.class,
            score: d.prob,
            bbox: BoxLabel { class: d.class, x: d.bbox.x, y: d.bbox.y, w: d.bbox.w, h: d.bbox.h },
        }
    }
}

/// Average precision for one class at one IoU threshold, VOC max-envelope style.
fn average_precision(dets: &[&Det], gts: &[&(usize, BoxLabel)], thr: f32) -> f32 {
    if gts.is_empty() {
        return 0.0;
    }
    let mut matched = vec![false; gts.len()];
    let mut cum_tp = 0.0f32;
    let mut cum_fp = 0.0f32;
    let mut precisions: Vec<f32> = vec![];
    let mut recalls: Vec<f32> = vec![];
    for d in dets {
        // Greedy match to the best unmatched ground truth of the same image and class.
        let mut best = thr;
        let mut best_i = usize::MAX;
        for (i, (image, gt)) in gts.iter().enumerate() {
            if *image != d.image || matched[i] {
                continue;
            }
            let iou = iou(d.bbox, *gt);
            if iou >= best {
                best = iou;
                best_i = i;
            }
        }
        if best_i != usize::MAX {
            matched[best_i] = true;
            cum_tp += 1.0;
        } else {
            cum_fp += 1.0;
        }
        precisions.push(cum_tp / (cum_tp + cum_fp));
        recalls.push(cum_tp / gts.len() as f32);
    }
    // Max envelope, then the area under it.
    for i in (0..precisions.len().saturating_sub(1)).rev() {
        if precisions[i] < precisions[i + 1] {
            precisions[i] = precisions[i + 1];
        }
    }
    let mut ap = 0.0f32;
    let mut prev_r = 0.0f32;
    for (p, r) in precisions.iter().zip(&recalls) {
        ap += *p * (*r - prev_r).max(0.0);
        prev_r = *r;
    }
    ap
}

/// Evaluate `model` on `samples` (labels are normalized to their own image, as are detections).
pub fn evaluate(
    classes: usize,
    samples: &[Sample],
    load: impl Fn(&Sample) -> Result<Image>,
    detect: impl Fn(&Image) -> Result<Vec<Detection>>,
) -> Result<MapResult> {
    let mut dets: Vec<Det> = vec![];
    let mut gt: Vec<(usize, BoxLabel)> = vec![];
    for (image_idx, sample) in samples.iter().enumerate() {
        let img = load(sample)?;
        let found = detect(&img)?;
        for d in &found {
            let mut det = Det::from(d);
            det.image = image_idx;
            dets.push(det);
        }
        for l in &sample.labels {
            gt.push((image_idx, *l));
        }
    }
    let n_gt = gt.len();
    let n_det = dets.len();
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut per_class = vec![0.0f32; classes];
    let mut have = vec![false; classes];
    for (_, l) in &gt {
        if l.class < classes {
            have[l.class] = true;
        }
    }
    let mut map_at = |thr: f32| -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for c in 0..classes {
            if !have[c] {
                continue;
            }
            let cd: Vec<&Det> = dets.iter().filter(|d| d.class == c).collect();
            let cg: Vec<&(usize, BoxLabel)> = gt.iter().filter(|(_, l)| l.class == c).collect();
            let ap = average_precision(&cd, &cg, thr);
            if thr == 0.5 {
                per_class[c] = ap;
            }
            sum += ap;
            n += 1;
        }
        if n == 0 { 0.0 } else { sum / n as f32 }
    };
    let map_50 = map_at(0.5);
    let mut sweep = 0.0f32;
    let mut n_thr = 0;
    for k in 0..10 {
        sweep += map_at(0.5 + 0.05 * k as f32);
        n_thr += 1;
    }
    let counts = (0..classes)
        .map(|c| {
            let cd: Vec<&Det> = dets.iter().filter(|d| d.class == c).collect();
            let cg: Vec<&(usize, BoxLabel)> = gt.iter().filter(|(_, l)| l.class == c).collect();
            class_counts(&cd, &cg)
        })
        .collect();
    Ok(MapResult {
        counts,
        map_50,
        map_50_95: sweep / n_thr as f32,
        per_class,
        n_images: samples.len(),
        n_gt,
        n_det,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt(class: usize, x: f32, y: f32, w: f32, h: f32) -> (usize, BoxLabel) {
        (0, BoxLabel { class, x, y, w, h })
    }

    #[test]
    fn average_precision_perfect_and_empty() {
        let g = gt(0, 0.5, 0.5, 0.2, 0.2);
        let det = Det { image: 0, class: 0, score: 0.9, bbox: g.1 };
        assert!((average_precision(&[&det], &[&g], 0.5) - 1.0).abs() < 1e-6);
        // A detection that misses gives zero precision but recall 0 -> AP 0.
        let miss = Det { image: 0, class: 0, score: 0.9, bbox: BoxLabel { class: 0, x: 0.1, y: 0.1, w: 0.05, h: 0.05 } };
        assert_eq!(average_precision(&[&miss], &[&g], 0.5), 0.0);
        // No detections, one GT -> AP 0.
        assert_eq!(average_precision(&[], &[&g], 0.5), 0.0);
        // No GT -> AP is defined as 0 here (the caller skips classes without GT).
        assert_eq!(average_precision(&[&det], &[], 0.5), 0.0);
    }

    #[test]
    fn average_precision_ranks_by_score() {
        // Two GTs, three detections: one hit, one hit, one miss interleaved by score.
        let g1 = gt(0, 0.3, 0.3, 0.2, 0.2);
        let g2 = gt(0, 0.7, 0.7, 0.2, 0.2);
        let hit1 = Det { image: 0, class: 0, score: 0.9, bbox: g1.1 };
        let miss = Det { image: 0, class: 0, score: 0.8, bbox: BoxLabel { class: 0, x: 0.5, y: 0.1, w: 0.05, h: 0.05 } };
        let hit2 = Det { image: 0, class: 0, score: 0.7, bbox: g2.1 };
        let dets = [&hit1, &miss, &hit2];
        let gts = [&g1, &g2];
        // P/R: (1, .5), (.5, .5), (.667, 1) -> max envelope (1, 1, .667).
        // AP = 1*(.5-0) + .667*(1-.5) = 0.8333.
        let ap = average_precision(&dets, &gts, 0.5);
        assert!((ap - 0.833_333_3).abs() < 1e-4, "{ap}");
    }

    #[test]
    fn iou_is_symmetric_and_bounded() {
        let a = BoxLabel { class: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 };
        let b = BoxLabel { class: 0, x: 0.5, y: 0.5, w: 0.2, h: 0.2 };
        assert!((iou(a, b) - 1.0).abs() < 1e-6);
        let c = BoxLabel { class: 0, x: 0.6, y: 0.5, w: 0.2, h: 0.2 };
        assert!((iou(a, c) - iou(c, a)).abs() < 1e-6);
        assert!(iou(a, c) < 0.5);
        let d = BoxLabel { class: 0, x: 0.9, y: 0.9, w: 0.05, h: 0.05 };
        assert_eq!(iou(a, d), 0.0);
    }
}
