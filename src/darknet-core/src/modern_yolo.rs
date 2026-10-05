//! `[yolox]`: the anchor-free detection head (Phase 2A of `tasks/rust/layers.md`),
//! ported from `Jev_laya:src-lib/modern_yolo_layer.cpp` and
//! `yolox_layer.cpp`, with the cfg keys from
//! `darknet_cfg.cpp::parse_yolox_section`.
//!
//! The layer has **no weights and no kernels**: it is a pass-through forward plus
//! a host-side assignment and delta computation, exactly as in C++.
//!
//! Output layout of the preceding `[convolutional]` (`coords = 4` for YOLOX):
//!
//! ```text
//! filters = classes + 5
//! [ tx | ty | tw | th | obj | cls_0 .. cls_{classes-1} ]   (per feature cell)
//! ```
//!
//! with every entry laid out channel-major and strided by `H * W`
//! (`modern_entry_index`).  The forward copies the input, replaces non-finite
//! values by 0 and applies a sigmoid to the last `1 + classes` entries only;
//! the four box entries stay raw.
//!
//! Decode (`decode_yolox_box`), in normalised coordinates:
//!
//! ```text
//! cx = (col + tx) / W      cy = (row + ty) / H
//! w  = exp(clamp(tw, -10, 10)) / W      h = exp(clamp(th, -10, 10)) / H
//! ```
//!
//! Assignment is YOLOX SimOTA on one feature map.  For every truth box, the
//! candidate cells are those whose centre is within
//! [`SIMOTA_CENTER_RADIUS`] feature cells of the box centre (YOLOX uses the
//! centre test, not the "inside the box" test the DFL variants use).  Each
//! candidate gets `cost = -log(max(sqrt(cls*obj), eps)) + 3 * -log(iou + eps)`
//! and `score = 1 / (1 + cost)`.  The number of positives is dynamic:
//! `k = max(1, int(sum of the top-10 IoUs))`, capped by the candidate count.
//! The `k` cheapest candidates are taken, and a cell keeps the cheapest truth
//! that claims it (C++ stores `-cost` in `quality` as the conflict key and
//! overwrites it with the hard label `1.0` afterwards, because YOLOX trains
//! with hard class targets).
//!
//! Delta, per cell (`net`-level normalizers, class multipliers and the
//! `label_smooth_eps` / `objectness_smooth` / `ignore_thresh` switches):
//!
//! * unassigned: the best IoU against any truth decides.  Above
//!   `ignore_thresh` the cell is left alone unless `objectness_smooth`, which
//!   targets the observed IoU; otherwise the objectness target is 0.
//! * assigned: objectness target 1, class target 1 for the assigned class and 0
//!   for the rest, and the four box targets
//!   `(truth.x*W - col, truth.y*H - row, log(truth.w*W), log(truth.h*H))`
//!   scaled by `iou_normalizer`.
//!
//! `cost` is `||delta||^2` (C++ `pow(mag_array(delta), 2)`).

use crate::model::ModernYoloSpec;

/// C++ `k_eps`: the floor on every logarithm and product.
pub const EPS: f32 = 1e-8;
/// C++ `k_simota_center_radius`, in feature cells.
pub const SIMOTA_CENTER_RADIUS: f32 = 2.5;
/// C++ `k_simota_candidate_topk`: how many IoUs feed the dynamic `k`.
pub const SIMOTA_CANDIDATE_TOPK: usize = 10;
/// Largest finite `f32`, C++ `FLT_MAX`: the default that disables clipping.
pub const FLT_MAX: f32 = f32::MAX;

/// Normalised `[0, 1]` box with a centre and a size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModernBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl ModernBox {
    pub fn left(&self) -> f32 {
        self.x - 0.5 * self.w
    }
    pub fn right(&self) -> f32 {
        self.x + 0.5 * self.w
    }
    pub fn top(&self) -> f32 {
        self.y - 0.5 * self.h
    }
    pub fn bottom(&self) -> f32 {
        self.y + 0.5 * self.h
    }
}

/// The strides of one `[yolox]` feature map, derived from the layer output
/// shape (`[batch, classes + 5, H, W]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModernLayout {
    pub batch: usize,
    /// Feature maps per image (`n`); 1 for YOLOX.
    pub n: usize,
    pub w: usize,
    pub h: usize,
    pub classes: usize,
    /// Box entries per cell (4 for YOLOX cxcywh).
    pub coords: usize,
}

/// Rejected configuration or truth, matching C++'s `darknet_fatal_error`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModernYoloError(pub String);

impl std::fmt::Display for ModernYoloError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl ModernLayout {
    /// Number of output entries per image.
    pub fn outputs(&self) -> usize {
        self.n * self.spatial() * (self.coords + self.classes + 1)
    }

    /// Entries per feature cell across all heads.
    pub fn spatial(&self) -> usize {
        self.w * self.h
    }

    /// Total output entries including the batch.
    pub fn len(&self) -> usize {
        self.batch * self.outputs()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// C++ `modern_entry_index`: the flat index of `entry` of cell `location`
    /// (`n * H * W + loc`) for image `batch`.
    pub fn entry_index(&self, batch: usize, location: usize, entry: usize) -> usize {
        let n = location / self.spatial();
        let loc = location % self.spatial();
        batch * self.outputs()
            + n * self.spatial() * (self.coords + self.classes + 1)
            + entry * self.spatial()
            + loc
    }

    /// C++ `obj_index_for`.
    pub fn obj_index(&self, b: usize, n: usize, loc: usize) -> usize {
        self.entry_index(b, n * self.spatial() + loc, self.coords)
    }

    /// C++ `class_index_for`.
    pub fn class_index(&self, b: usize, n: usize, loc: usize, class_id: usize) -> usize {
        self.entry_index(b, n * self.spatial() + loc, self.coords + 1 + class_id)
    }

    /// C++ `box_index_for`.
    pub fn box_index(&self, b: usize, n: usize, loc: usize) -> usize {
        self.entry_index(b, n * self.spatial() + loc, 0)
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn safe_log(value: f32) -> f32 {
    value.max(EPS).ln()
}

fn safe_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-finite_or_zero(x).clamp(-30.0, 30.0)).exp())
}

/// C++ `sanitize_box`.
fn sanitize(b: &mut ModernBox) {
    b.x = finite_or_zero(b.x);
    b.y = finite_or_zero(b.y);
    b.w = finite_or_zero(b.w).max(1e-6);
    b.h = finite_or_zero(b.h).max(1e-6);
}

/// Intersection-over-union, matching C++ `box_iou`.
pub fn box_iou(a: &ModernBox, b: &ModernBox) -> f32 {
    let iw = (a.right().min(b.right()) - a.left().max(b.left())).max(0.0);
    let ih = (a.bottom().min(b.bottom()) - a.top().max(b.top())).max(0.0);
    let intersection = iw * ih;
    if intersection == 0.0 {
        return 0.0;
    }
    let union = a.w.max(0.0) * a.h.max(0.0) + b.w.max(0.0) * b.h.max(0.0) - intersection;
    if union > 0.0 { intersection / union } else { 0.0 }
}

/// C++ `valid_truth`: finite, centre inside `[0, 1]` and a positive size.
fn valid_truth(t: &ModernBox) -> bool {
    t.x.is_finite()
        && t.y.is_finite()
        && t.w.is_finite()
        && t.h.is_finite()
        && t.x > 0.0
        && t.x <= 1.0
        && t.y >= 0.0
        && t.y <= 1.0
        && t.w > 0.0
        && t.h > 0.0
}

/// C++ `float_to_box_stride` for the detection truth layout
/// (`t * truth_size + b * truths + i`).
pub fn truth_box(truth: &[f32], truth_size: usize, truths: usize, t: usize, b: usize) -> ModernBox {
    let base = t * truth_size + b * truths;
    ModernBox {
        x: truth[base],
        y: truth[base + 1],
        w: truth[base + 2],
        h: truth[base + 3],
    }
}

/// The class id of truth row `t`, image `b`.
pub fn truth_class(truth: &[f32], truth_size: usize, truths: usize, t: usize, b: usize) -> usize {
    truth[t * truth_size + b * truths + 4] as usize
}

/// C++ `activate_modern_output`: sanitise everything, then sigmoid the
/// `1 + classes` trailing entries of every cell.  Mutates `output`.
pub fn activate_modern_output(output: &mut [f32], l: &ModernLayout) {
    for v in output.iter_mut() {
        *v = finite_or_zero(*v);
    }
    for b in 0..l.batch {
        for n in 0..l.n {
            let obj_index = l.entry_index(b, n * l.spatial(), l.coords);
            for i in 0..(1 + l.classes) * l.spatial() {
                output[obj_index + i] = safe_sigmoid(output[obj_index + i]);
            }
        }
    }
}

/// C++ `decode_yolox_box`.
pub fn decode_yolox_box(x: &[f32], l: &ModernLayout, b: usize, n: usize, loc: usize) -> ModernBox {
    let row = loc / l.w;
    let col = loc % l.w;
    let stride = l.spatial();
    let idx = l.box_index(b, n, loc);
    let mut box_ = ModernBox {
        x: (col as f32 + finite_or_zero(x[idx])) / l.w as f32,
        y: (row as f32 + finite_or_zero(x[idx + stride])) / l.h as f32,
        w: finite_or_zero(x[idx + 2 * stride]).clamp(-10.0, 10.0).exp() / l.w as f32,
        h: finite_or_zero(x[idx + 3 * stride]).clamp(-10.0, 10.0).exp() / l.h as f32,
    };
    sanitize(&mut box_);
    box_
}

/// C++ `point_inside_center_radius`, in feature cells.
fn point_inside_center_radius(l: &ModernLayout, loc: usize, truth: &ModernBox, radius: f32) -> bool {
    let col = loc % l.w;
    let row = loc / l.w;
    let gx = col as f32 + 0.5;
    let gy = row as f32 + 0.5;
    let tx = truth.x * l.w as f32;
    let ty = truth.y * l.h as f32;
    (gx - tx).abs() <= radius && (gy - ty).abs() <= radius
}

fn objectness_for(x: &[f32], l: &ModernLayout, b: usize, n: usize, loc: usize) -> f32 {
    finite_or_zero(x[l.obj_index(b, n, loc)]).clamp(0.0, 1.0)
}

fn class_score_for(x: &[f32], l: &ModernLayout, b: usize, n: usize, loc: usize, class_id: usize) -> f32 {
    if class_id >= l.classes {
        return 0.0;
    }
    finite_or_zero(x[l.class_index(b, n, loc, class_id)]).clamp(0.0, 1.0)
}

/// One candidate cell for one truth, as C++'s `Candidate`.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub anchor: usize,
    pub loc: usize,
    pub truth_index: usize,
    pub class_id: usize,
    pub iou: f32,
    pub score: f32,
    pub cost: f32,
    pub pred: ModernBox,
    pub truth: ModernBox,
}

/// C++ `collect_candidates_for_truth` for YOLOX.
fn collect_candidates(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    truth_index: usize,
    class_id: usize,
    truth: &ModernBox,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for n in 0..l.n {
        for loc in 0..l.spatial() {
            if !point_inside_center_radius(l, loc, truth, SIMOTA_CENTER_RADIUS) {
                continue;
            }
            let pred = decode_yolox_box(x, l, b, n, loc);
            let iou = finite_or_zero(box_iou(&pred, truth)).clamp(0.0, 1.0);
            let obj = objectness_for(x, l, b, n, loc);
            let cls = class_score_for(x, l, b, n, loc, class_id);
            let cls_obj = (cls * obj).max(EPS).sqrt();
            let cost = -safe_log(cls_obj) + 3.0 * -safe_log(iou + EPS);
            let score = 1.0 / (1.0 + cost);
            if score.is_finite() && cost.is_finite() {
                out.push(Candidate {
                    anchor: n,
                    loc,
                    truth_index,
                    class_id,
                    iou,
                    score,
                    cost,
                    pred,
                    truth: *truth,
                });
            }
        }
    }
    out
}

/// C++ `simota_dynamic_k`: the mean of the best IoUs, floored at 1.
fn simota_dynamic_k(candidates: &[Candidate]) -> usize {
    if candidates.is_empty() {
        return 0;
    }
    let mut by_iou: Vec<f32> = candidates.iter().map(|c| c.iou).collect();
    by_iou.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let k = SIMOTA_CANDIDATE_TOPK.min(by_iou.len());
    let sum_iou: f32 = by_iou[..k].iter().sum();
    (sum_iou as usize).max(1)
}

/// One truth's claim on one cell, as C++'s `Assignment`.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub truth_index: usize,
    pub class_id: usize,
    /// Conflict key during assignment, then the hard class target `1.0`.
    pub quality: f32,
    pub iou: f32,
    pub truth: ModernBox,
}

/// Net-level knobs the head reads (C++ takes them from `state.net` / the
/// `[net]` section).
#[derive(Clone, Debug, PartialEq)]
pub struct ModernYoloSettings {
    pub cls_normalizer: f32,
    pub obj_normalizer: f32,
    pub iou_normalizer: f32,
    pub label_smooth_eps: f32,
    pub ignore_thresh: f32,
    pub max_delta: f32,
    pub objectness_smooth: bool,
    pub classes_multipliers: Option<Vec<f32>>,
    pub class_map: Option<Vec<usize>>,
    pub truth_size: usize,
}

impl Default for ModernYoloSettings {
    /// C++ `set_default_training_params`: every normalizer that parsed as 0
    /// becomes 1, `ignore_thresh` becomes 0.7, `truth_thresh` 1 and `max_delta`
    /// `FLT_MAX`.
    fn default() -> Self {
        Self {
            cls_normalizer: 1.0,
            obj_normalizer: 1.0,
            iou_normalizer: 1.0,
            label_smooth_eps: 0.0,
            ignore_thresh: 0.7,
            max_delta: FLT_MAX,
            objectness_smooth: false,
            classes_multipliers: None,
            class_map: None,
            truth_size: 6,
        }
    }
}

/// C++ `assign_yolox`: SimOTA over one image's cells.
pub fn assign_yolox(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    b: usize,
) -> Result<Vec<Assignment>, ModernYoloError> {
    let truths = spec.max_boxes * settings.truth_size;
    let mut assignments: Vec<Assignment> = (0..l.n * l.spatial())
        .map(|_| Assignment {
            truth_index: usize::MAX,
            class_id: usize::MAX,
            quality: 0.0,
            iou: 0.0,
            truth: ModernBox::default(),
        })
        .collect();
    for t in 0..spec.max_boxes {
        if truth.len() < truths * l.batch {
            return Err(ModernYoloError(format!(
                "yolox truth needs {} entries for {} images x {} boxes x {} fields, got {}",
                truths * l.batch,
                l.batch,
                spec.max_boxes,
                settings.truth_size,
                truth.len()
            )));
        }
        let box_ = truth_box(truth, settings.truth_size, truths, t, b);
        if box_.x == 0.0 {
            break;
        }
        if !valid_truth(&box_) {
            return Err(ModernYoloError(format!(
                "invalid YOLOX truth box (x={}, y={}, w={}, h={})",
                box_.x, box_.y, box_.w, box_.h
            )));
        }
        let mut class_id = truth_class(truth, settings.truth_size, truths, t, b);
        if class_id >= l.classes {
            return Err(ModernYoloError(format!("invalid class ID #{class_id}")));
        }
        if let Some(map) = &settings.class_map {
            class_id = map[class_id];
        }
        let mut candidates = collect_candidates(x, l, b, t, class_id, &box_);
        if candidates.is_empty() {
            continue;
        }
        let dynamic_k = simota_dynamic_k(&candidates).min(candidates.len());
        candidates.sort_by(|a, c| a.cost.partial_cmp(&c.cost).unwrap_or(std::cmp::Ordering::Equal));
        for c in candidates.iter().take(dynamic_k) {
            let slot = c.anchor * l.spatial() + c.loc;
            let a = &mut assignments[slot];
            if a.truth_index == usize::MAX || c.cost < -a.quality {
                *a = Assignment {
                    truth_index: c.truth_index,
                    class_id: c.class_id,
                    // C++ keeps -cost as the conflict key here.
                    quality: -c.cost,
                    iou: c.iou,
                    truth: c.truth,
                };
            }
        }
    }
    // YOLOX trains with hard class labels.
    for a in assignments.iter_mut() {
        if a.truth_index != usize::MAX {
            a.quality = 1.0;
        }
    }
    Ok(assignments)
}

/// C++ `best_truth_iou_for_prediction`: the best IoU of this cell against any
/// truth box of the image.
fn best_truth_iou(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    b: usize,
    n: usize,
    loc: usize,
) -> f32 {
    let pred = decode_yolox_box(x, l, b, n, loc);
    let truths = spec.max_boxes * settings.truth_size;
    let mut best = 0.0f32;
    for t in 0..spec.max_boxes {
        let box_ = truth_box(truth, settings.truth_size, truths, t, b);
        if box_.x == 0.0 {
            break;
        }
        best = best.max(finite_or_zero(box_iou(&pred, &box_)).clamp(0.0, 1.0));
    }
    best
}

/// The head's whole training contribution for one image: the delta for every
/// cell, plus the mean IoU / IoU loss / positive count C++ reports.
#[derive(Clone, Debug, PartialEq)]
pub struct ModernYoloBatch {
    pub delta: Vec<f32>,
    pub sum_iou: f32,
    pub sum_iou_loss: f32,
    pub positives: usize,
    pub class_ids: Vec<usize>,
}

/// C++ `train_batch_modern` for YOLOX: assignment, then per-cell deltas.
pub fn yolox_train_batch(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    b: usize,
) -> Result<ModernYoloBatch, ModernYoloError> {
    let assignments = assign_yolox(x, truth, l, spec, settings, b)?;
    let stride = l.spatial();
    // `entry_index` is a whole-batch index, so the delta buffer is the whole
    // batch too; the caller slices out this image's `outputs` afterwards.
    let mut delta = vec![0.0f32; l.len()];
    let mut class_ids = vec![usize::MAX; l.n * stride];
    let (mut sum_iou, mut sum_iou_loss, mut positives) = (0.0f32, 0.0f32, 0usize);

    for n in 0..l.n {
        for loc in 0..stride {
            let slot = n * stride + loc;
            let a = &assignments[slot];
            let obj_idx = l.obj_index(b, n, loc);
            let obj_out = finite_or_zero(x[obj_idx]);
            let push_obj = |delta: &mut Vec<f32>, target: f32| {
                let d = settings.obj_normalizer * (target.clamp(0.0, 1.0) - obj_out);
                if d.is_finite() {
                    delta[obj_idx] += d;
                }
            };
            if a.truth_index == usize::MAX {
                let best = best_truth_iou(x, truth, l, spec, settings, b, n, loc);
                if best > settings.ignore_thresh {
                    if settings.objectness_smooth {
                        push_obj(&mut delta, best);
                    }
                    continue;
                }
                push_obj(&mut delta, 0.0);
                continue;
            }
            let class_multiplier = settings
                .classes_multipliers
                .as_ref()
                .map(|m| m[a.class_id])
                .unwrap_or(1.0);
            let quality = a.quality.clamp(0.0, 1.0);
            push_obj(&mut delta, 1.0);
            // class targets
            for c in 0..l.classes {
                let mut target = if c == a.class_id { quality } else { 0.0 };
                if settings.label_smooth_eps != 0.0 {
                    target = target * (1.0 - settings.label_smooth_eps) + 0.5 * settings.label_smooth_eps;
                }
                let idx = l.class_index(b, n, loc, c);
                let mut d = settings.cls_normalizer * (target - finite_or_zero(x[idx]));
                if let Some(m) = &settings.classes_multipliers {
                    if c == a.class_id {
                        d *= m[a.class_id];
                    }
                }
                if d.is_finite() {
                    delta[idx] += d;
                }
            }
            // box targets
            let idx = l.box_index(b, n, loc);
            let col = loc % l.w;
            let row = loc / l.w;
            let targets = [
                a.truth.x * l.w as f32 - col as f32,
                a.truth.y * l.h as f32 - row as f32,
                safe_log(a.truth.w * l.w as f32),
                safe_log(a.truth.h * l.h as f32),
            ];
            let normalizer = settings.iou_normalizer * class_multiplier;
            for (c, target) in targets.iter().enumerate() {
                let mut d = normalizer * (target - finite_or_zero(x[idx + c * stride]));
                if settings.max_delta < FLT_MAX {
                    d = d.clamp(-settings.max_delta, settings.max_delta);
                }
                if d.is_finite() {
                    delta[idx + c * stride] += d;
                }
            }
            class_ids[slot] = a.class_id;
            sum_iou += a.iou;
            sum_iou_loss += 1.0 - a.iou;
            positives += 1;
        }
    }
    Ok(ModernYoloBatch { delta, sum_iou, sum_iou_loss, positives, class_ids })
}

/// The head's training contribution for the whole batch.  `output` must already
/// be activated (see [`activate_modern_output`]); the returned delta has the
/// same layout.  `cost` is `||delta||^2` over the whole batch, as C++ reports.
pub fn yolox_training_delta(
    output: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
) -> Result<(Vec<f32>, f32, usize), ModernYoloError> {
    let mut delta = vec![0.0f32; l.len()];
    let (mut sum_iou, mut sum_iou_loss, mut positives) = (0.0f32, 0.0f32, 0usize);
    for b in 0..l.batch {
        let batch = yolox_train_batch(output, truth, l, spec, settings, b)?;
        let out = l.outputs();
        for (dst, src) in delta[b * out..(b + 1) * out].iter_mut().zip(&batch.delta) {
            *dst += src;
        }
        sum_iou += batch.sum_iou;
        sum_iou_loss += batch.sum_iou_loss;
        positives += batch.positives;
    }
    for d in delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    let cost = delta.iter().map(|d| d * d).sum::<f32>();
    Ok((delta, cost, positives))
}

/// One detection decoded from a `[yolox]` head, for the inference path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloxDetection {
    pub class_id: usize,
    pub score: f32,
    pub box_: ModernBox,
}

/// C++-equivalent decode of the best detection per class: every cell is scored
/// as `obj * cls`, kept above `threshold`, and NMS-free (the caller applies the
/// network-wide NMS, exactly like the yolo head's decode path).
pub fn decode_yolox(
    output: &[f32],
    l: &ModernLayout,
    threshold: f32,
) -> Vec<YoloxDetection> {
    let mut out = Vec::new();
    for b in 0..l.batch {
        for n in 0..l.n {
            for loc in 0..l.spatial() {
                let obj = objectness_for(output, l, b, n, loc);
                for c in 0..l.classes {
                    let cls = class_score_for(output, l, b, n, loc, c);
                    let score = obj * cls;
                    if score > threshold && score.is_finite() {
                        out.push(YoloxDetection {
                            class_id: c,
                            score,
                            box_: decode_yolox_box(output, l, b, n, loc),
                        });
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ModernYoloSpec {
        ModernYoloSpec { classes: 2, max_boxes: 3, jitter: 0.2, resize: 1.0 }
    }

    /// 2 classes + 5 channels, 2x2 cells, one image.
    fn layout() -> ModernLayout {
        ModernLayout { batch: 1, n: 1, w: 2, h: 2, classes: 2, coords: 4 }
    }

    /// Fill a head with explicit per-cell values: tx, ty, tw, th, obj, cls0, cls1.
    fn head(cells: &[[f32; 7]]) -> Vec<f32> {
        let mut out = vec![0.0f32; cells.len() * 7];
        for (i, cell) in cells.iter().enumerate() {
            for (e, v) in cell.iter().enumerate() {
                out[e * cells.len() + i] = *v;
            }
        }
        out
    }

    #[test]
    fn entry_index_matches_the_cpp_channel_major_layout() {
        let l = layout();
        // `filters = classes + 5`, so a cell is 4 box + 1 obj + 2 class entries.
        assert_eq!(l.spatial(), 4);
        assert_eq!(l.outputs(), 1 * 4 * (4 + 2 + 1));
        assert_eq!(l.len(), 28);
        // Entry e of cell `loc` on the only map: e * spatial + loc, batch 0.
        for e in 0..7 {
            for loc in 0..4 {
                assert_eq!(l.entry_index(0, loc, e), e * 4 + loc);
            }
        }
        assert_eq!(l.obj_index(0, 0, 3), 4 * 4 + 3);
        assert_eq!(l.class_index(0, 0, 3, 0), 5 * 4 + 3);
        assert_eq!(l.class_index(0, 0, 3, 1), 6 * 4 + 3);
        assert_eq!(l.box_index(0, 0, 3), 3);
    }

    #[test]
    fn activation_sigmoids_only_the_trailing_entries() {
        let l = layout();
        let cells = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mut x = head(&cells);
        activate_modern_output(&mut x, &l);
        // The four box entries of every cell survive untouched.
        for loc in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    x[l.box_index(0, 0, loc) + c * 4],
                    cells[loc][c],
                    "box entry {c} of cell {loc}"
                );
            }
        }
        // Cell 1's obj and class entries are the sigmoid of 5, 6, 7.
        let s = |v: f32| 1.0 / (1.0 + (-v).exp());
        assert!((x[l.obj_index(0, 0, 1)] - s(5.0)).abs() < 1e-6);
        assert!((x[l.class_index(0, 0, 1, 0)] - s(6.0)).abs() < 1e-6);
        assert!((x[l.class_index(0, 0, 1, 1)] - s(7.0)).abs() < 1e-6);
        // NaN became 0 before the sigmoid, so its cell reads 0.5, and a raw 0
        // sigmoid also reads 0.5.
        assert!((x[l.obj_index(0, 0, 2)] - 0.5).abs() < 1e-6);
        assert!((x[l.obj_index(0, 0, 0)] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn decode_is_the_hand_computed_centre_and_size() {
        let l = layout();
        // Cell 1 is (row 0, col 1): tx = 0, ty = 0, tw = ln(2), th = ln(2)
        // -> cx = 1/2, cy = 0, w = 2/2, h = 2/2.
        let x = head(&[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2f32.ln(), 2f32.ln(), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let b = decode_yolox_box(&x, &l, 0, 0, 1);
        assert!((b.x - 0.5).abs() < 1e-6, "cx {}", b.x);
        assert!(b.y.abs() < 1e-6, "cy {}", b.y);
        assert!((b.w - 1.0).abs() < 1e-6, "w {}", b.w);
        assert!((b.h - 1.0).abs() < 1e-6, "h {}", b.h);
        // The logit clamp is [-10, 10].
        let clamped = head(&[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 50.0, -50.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let b = decode_yolox_box(&clamped, &l, 0, 0, 1);
        assert!((b.w - 10f32.exp() / 2.0).abs() < 1e-6, "w {}", b.w);
        // -50 clamps to -10, so h = exp(-10)/2 (the 1e-6 floor in
        // `sanitize_box` only applies at or below zero width).
        assert!((b.h - (-10f32).exp() / 2.0).abs() < 1e-9, "h {}", b.h);
    }

    #[test]
    fn box_iou_matches_the_cpp_shape() {
        let a = ModernBox { x: 0.5, y: 0.5, w: 0.5, h: 0.5 };
        assert!((box_iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = ModernBox { x: 1.0, y: 0.5, w: 0.5, h: 0.5 };
        // Boxes [0.25, 0.75] and [0.75, 1.25] touch, so the intersection is 0.
        assert_eq!(box_iou(&a, &b), 0.0);
        let c = ModernBox { x: 0.6, y: 0.5, w: 0.5, h: 0.5 };
        // overlap width = 0.75 - 0.35 = 0.4, union = 0.25 + 0.25 - 0.2 = 0.3.
        assert!((box_iou(&a, &c) - 0.2f32 / 0.3).abs() < 1e-6, "{}", box_iou(&a, &c));
        assert_eq!(box_iou(&a, &ModernBox::default()), 0.0, "zero-area box");
    }

    #[test]
    fn simota_picks_the_cheapest_cell_and_hard_targets() {
        let l = ModernLayout { batch: 1, n: 1, w: 4, h: 4, classes: 2, coords: 4 };
        let s = spec();
        let settings = ModernYoloSettings::default();
        // One truth covering the top-left quadrant, class 0.
        let mut truth = vec![0.0f32; s.max_boxes * settings.truth_size];
        truth[0] = 0.25;
        truth[1] = 0.25;
        truth[2] = 0.5;
        truth[3] = 0.5;
        truth[4] = 0.0;
        // Make cell (0,0) predict the truth exactly and the other nearby cells
        // worse, so the cheapest candidate is the exact one.
        let cells = 16;
        let mut raw = vec![0.0f32; cells * 7];
        for loc in 0..cells {
            let (col, row) = (loc % 4, loc / 4);
            let mut e = |i: usize, v: f32| raw[i * cells + loc] = v;
            if loc == 0 {
                // Cell (0,0): cx = 0.25 -> tx = 0.25*4 - 0 = 1 px, and
                // w = 0.5 -> tw = ln(0.5 * 4) = ln 2.
                e(0, 1.0);
                e(1, 1.0);
                e(2, 2f32.ln());
                e(3, 2f32.ln());
                e(4, 10.0);
                e(5, 10.0);
                e(6, 0.0);
            } else {
                e(0, 0.0);
                e(1, 0.0);
                e(2, 0.0);
                e(3, 0.0);
                e(4, if (col + row) % 2 == 0 { 6.0 } else { -6.0 });
                e(5, 6.0);
                e(6, 0.0);
            }
        }
        let mut activated = raw.clone();
        activate_modern_output(&mut activated, &l);
        let assignments = assign_yolox(&activated, &truth, &l, &s, &settings, 0).unwrap();
        let assigned: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, a)| a.truth_index != usize::MAX)
            .map(|(i, _)| i)
            .collect();
        assert!(assigned.contains(&0), "the exact cell must be taken: {assigned:?}");
        for slot in assigned {
            assert_eq!(assignments[slot].truth_index, 0);
            assert_eq!(assignments[slot].class_id, 0);
            // YOLOX uses hard class targets.
            assert!((assignments[slot].quality - 1.0).abs() < 1e-6);
            assert!(assignments[slot].iou > 0.5, "iou {}", assignments[slot].iou);
        }
    }

    #[test]
    fn assignment_rejects_invalid_truth_and_classes() {
        let l = layout();
        let s = spec();
        let settings = ModernYoloSettings::default();
        let x = head(&[[0.0; 7]; 4]);
        // A truth box that escapes [0, 1].
        let mut truth = vec![0.0f32; s.max_boxes * settings.truth_size];
        truth[0] = 1.5;
        truth[1] = 0.25;
        truth[2] = 0.5;
        truth[3] = 0.5;
        let err = assign_yolox(&x, &truth, &l, &s, &settings, 0).unwrap_err();
        assert!(format!("{err}").contains("invalid YOLOX truth box"), "{err}");
        // A class id past the end of the head.
        let mut truth = vec![0.0f32; s.max_boxes * settings.truth_size];
        truth[0] = 0.25;
        truth[1] = 0.25;
        truth[2] = 0.5;
        truth[3] = 0.5;
        truth[4] = 7.0;
        let err = assign_yolox(&x, &truth, &l, &s, &settings, 0).unwrap_err();
        assert!(format!("{err}").contains("invalid class ID"), "{err}");
        // A short truth buffer is reported, not read out of bounds.
        let err = assign_yolox(&x, &[0.0f32; 3], &l, &s, &settings, 0).unwrap_err();
        assert!(format!("{err}").contains("truth needs"), "{err}");
    }

    #[test]
    fn an_unassigned_cell_gets_a_zero_objectness_target() {
        let l = ModernLayout { batch: 1, n: 1, w: 4, h: 4, classes: 1, coords: 4 };
        let s = ModernYoloSpec { classes: 1, max_boxes: 1, jitter: 0.2, resize: 1.0 };
        // `ignore_thresh` above any achievable IoU keeps every unassigned cell
        // on the "push objectness to 0" branch.
        let settings = ModernYoloSettings { ignore_thresh: 1.1, ..ModernYoloSettings::default() };
        let cells = 16;
        // obj = cls = 0 before activation, so sigmoid(0) = 0.5 everywhere.
        let mut x = vec![0.0f32; cells * 6];
        activate_modern_output(&mut x, &l);
        // No truth at all: every cell is unassigned.
        let truth = vec![0.0f32; 6];
        let (delta, cost, positives) = yolox_training_delta(&x, &truth, &l, &s, &settings).unwrap();
        assert_eq!(positives, 0);
        for loc in 0..cells {
            assert!(
                (delta[l.obj_index(0, 0, loc)] + 0.5).abs() < 1e-6,
                "cell {loc} objectness delta {} (target 0, prediction 0.5)",
                delta[l.obj_index(0, 0, loc)]
            );
        }
        // cost = ||delta||^2 over the whole batch.
        let want_cost: f32 = delta.iter().map(|d| d * d).sum();
        assert!((cost - want_cost).abs() < 1e-3, "{cost} vs {want_cost}");
    }

    #[test]
    fn the_seeded_cell_gets_objectness_class_and_box_targets() {
        let l = ModernLayout { batch: 1, n: 1, w: 4, h: 4, classes: 1, coords: 4 };
        let s = ModernYoloSpec { classes: 1, max_boxes: 2, jitter: 0.2, resize: 1.0 };
        let settings = ModernYoloSettings::default();
        let mut truth = vec![0.0f32; s.max_boxes * settings.truth_size];
        truth[0] = 0.5;
        truth[1] = 0.5;
        truth[2] = 1.0;
        truth[3] = 1.0;
        truth[4] = 0.0;
        // All cells predict obj = cls = 0 before activation, so the positive
        // terms dominate and every delta has a known sign.
        let cells = 16;
        let mut raw = vec![0.0f32; cells * 6];
        for loc in 0..cells {
            for e in 0..6 {
                raw[e * cells + loc] = 0.0;
            }
        }
        let mut x = raw.clone();
        activate_modern_output(&mut x, &l);
        let (delta, _cost, positives) = yolox_training_delta(&x, &truth, &l, &s, &settings).unwrap();
        assert!(positives >= 1);
        // Find an assigned cell and check its three target groups.
        let assigned = (0..cells)
            .find(|loc| {
                // assigned cells move objectness up from sigmoid(0) = 0.5
                delta[l.obj_index(0, 0, *loc)] > 0.0
            })
            .expect("some cell is assigned");
        assert!(delta[l.obj_index(0, 0, assigned)] > 0.0, "objectness up");
        assert!(delta[l.class_index(0, 0, assigned, 0)] > 0.0, "class up");
        let idx = l.box_index(0, 0, assigned);
        let stride = l.spatial();
        // Truth covers the whole image, so cx*W - col = 2 - col <= 2 and the
        // log size is ln(4).
        for c in 0..4 {
            assert!(delta[idx + c * stride].is_finite());
        }
        let col = assigned % 4;
        let row = assigned / 4;
        assert!((delta[idx] - (0.5 * 4.0 - col as f32)).abs() < 1e-4, "tx delta {}", delta[idx]);
        assert!((delta[idx + stride] - (0.5 * 4.0 - row as f32)).abs() < 1e-4, "ty delta {}", delta[idx + stride]);
        assert!((delta[idx + 2 * stride] - 4f32.ln()).abs() < 1e-4, "tw delta {}", delta[idx + 2 * stride]);
    }

    #[test]
    fn class_map_and_multipliers_are_applied() {
        let l = ModernLayout { batch: 1, n: 1, w: 4, h: 4, classes: 2, coords: 4 };
        let s = ModernYoloSpec { classes: 2, max_boxes: 2, jitter: 0.2, resize: 1.0 };
        let mut truth = vec![0.0f32; s.max_boxes * 6];
        truth[0] = 0.5;
        truth[1] = 0.5;
        truth[2] = 1.0;
        truth[3] = 1.0;
        truth[4] = 0.0;
        let cells = 16;
        let mut x = vec![0.0f32; cells * 7];
        activate_modern_output(&mut x, &l);
        // Without a map, class 0; with map [1, 0] the truth becomes class 1.
        let plain = assign_yolox(&x, &truth, &l, &s, &ModernYoloSettings::default(), 0).unwrap();
        let plain_class = plain.iter().find(|a| a.truth_index != usize::MAX).map(|a| a.class_id);
        assert_eq!(plain_class, Some(0));
        let settings =
            ModernYoloSettings { class_map: Some(vec![1, 0]), ..ModernYoloSettings::default() };
        let mapped = assign_yolox(&x, &truth, &l, &s, &settings, 0).unwrap();
        let mapped_class = mapped.iter().find(|a| a.truth_index != usize::MAX).map(|a| a.class_id);
        assert_eq!(mapped_class, Some(1), "the class map rewrote the label");
    }

    #[test]
    fn no_truth_trains_every_cell_toward_zero_objectness() {
        let l = ModernLayout { batch: 1, n: 1, w: 4, h: 4, classes: 2, coords: 4 };
        let s = ModernYoloSpec { classes: 2, max_boxes: 1, jitter: 0.2, resize: 1.0 };
        let truth = vec![0.0f32; 6]; // no boxes
        let cells = 16;
        let mut x = vec![0.0f32; cells * 7];
        activate_modern_output(&mut x, &l);
        let (delta, _cost, positives) =
            yolox_training_delta(&x, &truth, &l, &s, &ModernYoloSettings::default()).unwrap();
        assert_eq!(positives, 0);
        // obj = sigmoid(0) = 0.5 and the target is 0, so the objectness delta is
        // -0.5 in every cell and nothing else moves.
        for loc in 0..cells {
            assert!((delta[l.obj_index(0, 0, loc)] + 0.5).abs() < 1e-6, "cell {loc}");
        }
        assert_eq!(
            delta.iter().filter(|d| **d != 0.0).count(),
            cells,
            "only objectness entries move when nothing is assigned"
        );
    }

    #[test]
    fn decode_yolox_reports_objectness_times_class_above_the_threshold() {
        let l = layout();
        let mut x = head(&[
            [0.0, 0.0, 2f32.ln(), 2f32.ln(), 10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        activate_modern_output(&mut x, &l);
        let dets = decode_yolox(&x, &l, 0.5);
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert_eq!(dets[0].class_id, 0);
        let s = 1.0 / (1.0 + (-10f32).exp());
        assert!((dets[0].score - s * s).abs() < 1e-6, "score {}", dets[0].score);
        assert!(dets[0].score > 0.5);
    }

    #[test]
    fn default_settings_match_the_cpp_training_defaults() {
        let s = ModernYoloSettings::default();
        assert_eq!((s.cls_normalizer, s.obj_normalizer, s.iou_normalizer), (1.0, 1.0, 1.0));
        assert_eq!(s.ignore_thresh, 0.7);
        assert_eq!(s.max_delta, FLT_MAX);
        assert_eq!(s.truth_size, 6);
    }
}
