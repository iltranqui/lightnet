//! `[omd_loss]`: the observable-matrix / metric-geometry auxiliary loss
//! (Phase 1b of `tasks/rust/layers.md`), ported from
//! `Jev_laya:src-lib/omd_loss_layer.cpp` and its cfg parser.
//!
//! The layer is a **pass-through** for the forward (`output == input`) and for
//! the backward (`previous_delta += 1.0 * delta`).  What it does is sample
//! feature vectors from the previous layer during training, build a small
//! supervised pair set from the box (or class) truth, and inject the resulting
//! metric-learning deltas into that same layer.  Because it is a pass-through,
//! the ordinary seeded-backward machinery already routes the delta to the
//! previous layer; unlike the Task 10 heads this one therefore never has to
//! touch two layers at once.
//!
//! Math.  Every sample is L2-normalised as `z = raw / ||raw||` with the norm
//! floored at `sqrt(max(eps, 0))` so a zero vector normalises to zero.  Over
//! every pair `i < j` that is not background/background:
//!
//! * **positive** (both objects, same class): `loss += w * (1 - cos)`, with
//!   `grad_z_i -= w * z_j`, `grad_z_j -= w * z_i`.
//! * **negative** (background/object, or two different object classes) with
//!   `violation = margin - (1 - cos)`: nothing when `violation <= 0`,
//!   otherwise `loss += w * violation^2` and `grad_z_i += 2 w violation z_j`
//!   (the `1 - cos` chain rule turns the squared violation into `2 w v`).
//!
//! `w` is the positive / negative weight times `max(w_i, w_j)`, doubled by
//! `background_weight` when either side is a background sample.  The reported
//! cost is `total_loss * cls_normalizer / active_pairs`, and `accuracy` is the
//! percentage of pairs already inside the margin.
//!
//! Scattering the gradient back is the chain rule through the normalisation:
//! with `dot = <grad_z, z>` and `||raw|| = norm`, `d norm / d raw = z`, so
//! `grad_raw = (grad_z - z * dot) / norm`.  Darknet deltas are negative
//! gradients, hence the leading minus, and each element is clipped to
//! `max_delta` (default 10, not `FLT_MAX` as in `[segmentation]`).
//!
//! Sampling (only when training and a truth tensor is present):
//!
//! * **detection** (`max_boxes > 0`, `truth_size >= 5`, spatial input): one
//!   sample per valid box at the pixel the box centre falls in, with a
//!   `small_boost` weight for boxes whose smaller side is at most
//!   `small_threshold` pixels.
//! * **background** (`background_samples > 0`): pseudo-random pixels that fall
//!   outside every box of their image, weight `max(background_weight, 0)`,
//!   class -2.  The scan order is C++'s LCG so the choice is reproducible.
//! * **classification** (only when the two above produce nothing, i.e.
//!   `max_boxes == 0` and `classes > 0`): one sample per image at the argmax
//!   truth class, either per-channel global pooling or the whole image.

use crate::model::{OmdLossSpec, TensorShape};

/// Largest finite `f32`, C++ `FLT_MAX`: the sentinel that disables clipping.
pub const FLT_MAX: f32 = f32::MAX;

/// One sampled feature vector plus everything the pair loss accumulates.
#[derive(Clone, Debug, PartialEq)]
pub struct OmdSample {
    /// Image the sample comes from.
    pub batch: usize,
    /// Pixel column (detection / background samples only).
    pub col: usize,
    /// Pixel row (detection / background samples only).
    pub row: usize,
    /// Class label; `-2` marks a background sample, `-1` an unknown one.
    pub cls: i32,
    pub background: bool,
    /// `pool=1`: the sample is a per-channel global average.
    pub pooled: bool,
    /// The sample is a whole-image vector (classification without `pool`).
    pub vector: bool,
    /// Per-sample loss weight (`small_boost`, `background_weight`).
    pub weight: f32,
    /// `||raw||` after the eps floor.
    pub norm: f32,
    /// The unnormalised feature vector.
    pub raw: Vec<f32>,
    /// `raw / norm`.
    pub z: Vec<f32>,
    /// Accumulated gradient with respect to `z`.
    pub grad_z: Vec<f32>,
}

/// One truth box, normalised coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TruthBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    cls: i32,
}

/// Everything `[omd_loss]` computes during its forward.
#[derive(Clone, Debug, PartialEq)]
pub struct OmdLossResult {
    /// `total_loss * cls_normalizer / active_pairs`.
    pub cost: f32,
    /// `100 * good_pairs / active_pairs`, or `-1` when nothing was paired.
    pub accuracy: f32,
    /// The delta to inject into the previous layer (already sanitised).
    pub delta: Vec<f32>,
    pub samples: Vec<OmdSample>,
    pub object_samples: usize,
    pub background_samples: usize,
    pub classification_samples: usize,
    pub positive_pairs: usize,
    pub negative_pairs: usize,
    pub good_pairs: usize,
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn clip_delta(value: f32, max_delta: f32) -> f32 {
    if max_delta >= FLT_MAX {
        return value;
    }
    value.clamp(-max_delta, max_delta)
}

/// C++ `omd_has_spatial_layout`: an `[batch, c, h, w]` input.
fn has_spatial_layout(shape: &TensorShape) -> bool {
    shape.width > 0
        && shape.height > 0
        && shape.channels > 0
        && shape.element_count() == shape.width * shape.height * shape.channels
}

fn spatial_index(shape: &TensorShape, b: usize, c: usize, row: usize, col: usize) -> usize {
    b * shape.element_count() + c * shape.width * shape.height + row * shape.width + col
}

fn point_inside_any_box(boxes: &[TruthBox], x: f32, y: f32) -> bool {
    boxes.iter().any(|box_| {
        x >= box_.x - 0.5 * box_.w
            && x <= box_.x + 0.5 * box_.w
            && y >= box_.y - 0.5 * box_.h
            && y <= box_.y + 0.5 * box_.h
    })
}

fn extract_object_vector(input: &[f32], shape: &TensorShape, sample: &mut OmdSample) {
    sample.raw = (0..shape.channels)
        .map(|c| {
            finite_or_zero(input[spatial_index(shape, sample.batch, c, sample.row, sample.col)])
        })
        .collect();
}

fn extract_pooled_vector(input: &[f32], shape: &TensorShape, sample: &mut OmdSample) {
    let plane = shape.width * shape.height;
    let inv_plane = if plane > 0 { 1.0 / plane as f32 } else { 0.0 };
    sample.raw = (0..shape.channels)
        .map(|c| {
            let base = sample.batch * shape.element_count() + c * plane;
            (0..plane)
                .map(|p| finite_or_zero(input[base + p]))
                .sum::<f32>()
                * inv_plane
        })
        .collect();
}

fn extract_image_vector(input: &[f32], shape: &TensorShape, sample: &mut OmdSample) {
    let base = sample.batch * shape.element_count();
    sample.raw = (0..shape.element_count())
        .map(|i| finite_or_zero(input[base + i]))
        .collect();
}

/// C++ `normalize_sample`: `z = raw / ||raw||` with the squared norm floored.
fn normalize_sample(sample: &mut OmdSample, eps: f32) {
    let mut norm2 = eps.max(0.0);
    for v in &sample.raw {
        norm2 += v * v;
    }
    sample.norm = norm2.sqrt();
    if !sample.norm.is_finite() || sample.norm <= 0.0 {
        sample.norm = 1.0;
    }
    sample.z = sample.raw.iter().map(|v| v / sample.norm).collect();
    sample.grad_z = vec![0.0; sample.raw.len()];
}

/// Box-truth samples: one per valid box, at the pixel its centre falls in.
fn collect_detection_samples(
    input: &[f32],
    truth: &[f32],
    spec: &OmdLossSpec,
    shape: &TensorShape,
    net_w: usize,
    net_h: usize,
    samples: &mut Vec<OmdSample>,
    boxes_by_batch: &mut Vec<Vec<TruthBox>>,
) -> usize {
    if truth.len() < shape.batch * spec.max_boxes * spec.truth_size || !has_spatial_layout(shape) {
        return 0;
    }
    let mut object_count = 0usize;
    boxes_by_batch.clear();
    boxes_by_batch.resize(shape.batch, Vec::new());
    let max_samples = spec.max_samples.max(1);
    let truths = spec.max_boxes * spec.truth_size;
    for b in 0..shape.batch {
        let truth_b = &truth[b * truths..(b + 1) * truths];
        for t in 0..spec.max_boxes {
            let row = &truth_b[t * spec.truth_size..(t + 1) * spec.truth_size];
            if row[0] == 0.0 {
                break; // C++ `if (x == 0) break;`: a zero x ends this image's boxes
            }
            let box_ = TruthBox {
                x: finite_or_zero(row[0]),
                y: finite_or_zero(row[1]),
                w: finite_or_zero(row[2]),
                h: finite_or_zero(row[3]),
                cls: row[4] as i32,
            };
            if spec.classes > 0 && (box_.cls < 0 || box_.cls as usize >= spec.classes) {
                continue;
            }
            if !(box_.w > 0.0) || !(box_.h > 0.0) {
                continue;
            }
            if !(0.0..=1.0).contains(&box_.x) || !(0.0..=1.0).contains(&box_.y) {
                continue;
            }
            boxes_by_batch[b].push(box_);
            if samples.len() >= max_samples {
                continue;
            }
            let mut sample = OmdSample {
                batch: b,
                col: ((box_.x * shape.width as f32) as usize).min(shape.width - 1),
                row: ((box_.y * shape.height as f32) as usize).min(shape.height - 1),
                cls: box_.cls,
                background: false,
                pooled: false,
                vector: false,
                weight: 1.0,
                norm: 1.0,
                raw: Vec::new(),
                z: Vec::new(),
                grad_z: Vec::new(),
            };
            if spec.small_threshold > 0.0 && spec.small_boost > 1.0 {
                let small_side_px = (box_.w * net_w as f32).min(box_.h * net_h as f32);
                if small_side_px > 0.0 && small_side_px <= spec.small_threshold {
                    sample.weight = spec.small_boost;
                }
            }
            extract_object_vector(input, shape, &mut sample);
            normalize_sample(&mut sample, spec.eps);
            samples.push(sample);
            object_count += 1;
        }
    }
    object_count
}

/// Pseudo-random pixels outside every box of their image.  The scan order is
/// C++'s LCG so the same truth gives the same background samples.
fn collect_background_samples(
    input: &[f32],
    spec: &OmdLossSpec,
    shape: &TensorShape,
    samples: &mut Vec<OmdSample>,
    boxes_by_batch: &[Vec<TruthBox>],
) -> usize {
    if spec.background_samples == 0 || !has_spatial_layout(shape) || boxes_by_batch.is_empty() {
        return 0;
    }
    let max_samples = spec.max_samples.max(1);
    let plane = shape.width * shape.height;
    let mut count = 0usize;
    for b in 0..shape.batch {
        if boxes_by_batch[b].is_empty() {
            continue;
        }
        for s in 0..spec.background_samples {
            if samples.len() >= max_samples {
                break;
            }
            let base = (b as u32)
                .wrapping_mul(1_103_515_245)
                .wrapping_add((s as u32).wrapping_mul(2_654_435_761))
                .wrapping_add(1_013_904_223);
            for tries in 0..plane as u32 {
                let loc = (base.wrapping_add(tries.wrapping_mul(9973)) % plane as u32) as usize;
                let row = loc / shape.width;
                let col = loc % shape.width;
                let x = (col as f32 + 0.5) / shape.width as f32;
                let y = (row as f32 + 0.5) / shape.height as f32;
                if point_inside_any_box(&boxes_by_batch[b], x, y) {
                    continue;
                }
                let mut sample = OmdSample {
                    batch: b,
                    col,
                    row,
                    cls: -2,
                    background: true,
                    pooled: false,
                    vector: false,
                    weight: spec.background_weight.max(0.0),
                    norm: 1.0,
                    raw: Vec::new(),
                    z: Vec::new(),
                    grad_z: Vec::new(),
                };
                extract_object_vector(input, shape, &mut sample);
                normalize_sample(&mut sample, spec.eps);
                samples.push(sample);
                count += 1;
                break;
            }
        }
    }
    count
}

/// One sample per image at the argmax truth class, pooled or whole-image.
/// Only used when there is no box truth (`max_boxes == 0`, `classes > 0`).
fn collect_classification_samples(
    input: &[f32],
    truth: &[f32],
    spec: &OmdLossSpec,
    shape: &TensorShape,
    samples: &mut Vec<OmdSample>,
) -> usize {
    if truth.len() < shape.batch * spec.classes || spec.max_boxes > 0 || spec.classes == 0 {
        return 0;
    }
    for b in 0..shape.batch {
        let row = &truth[b * spec.classes..(b + 1) * spec.classes];
        let mut best_cls = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (c, v) in row.iter().enumerate() {
            if *v > best_score {
                best_score = *v;
                best_cls = c;
            }
        }
        if best_score <= 0.0 {
            continue;
        }
        let mut sample = OmdSample {
            batch: b,
            col: 0,
            row: 0,
            cls: best_cls as i32,
            background: false,
            pooled: spec.pool != 0 && has_spatial_layout(shape),
            vector: spec.pool == 0 || !has_spatial_layout(shape),
            weight: 1.0,
            norm: 1.0,
            raw: Vec::new(),
            z: Vec::new(),
            grad_z: Vec::new(),
        };
        if sample.pooled {
            extract_pooled_vector(input, shape, &mut sample);
        } else {
            extract_image_vector(input, shape, &mut sample);
        }
        normalize_sample(&mut sample, spec.eps);
        samples.push(sample);
    }
    samples.len()
}

/// C++ `compute_omd_pair_loss`: fills `grad_z` and returns the pair counts.
fn compute_pair_loss(spec: &OmdLossSpec, samples: &mut [OmdSample]) -> (f32, usize, usize, usize) {
    let mut total_loss = 0.0f32;
    let (mut positive, mut negative, mut good) = (0usize, 0usize, 0usize);
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            // Copy the two unit vectors out so the mutable `grad_z` update below
            // does not fight the immutable borrow of the samples.
            let (za, zb) = (samples[i].z.clone(), samples[j].z.clone());
            let (a, b) = (&samples[i], &samples[j]);
            if za.len() != zb.len() || za.is_empty() || (a.background && b.background) {
                continue;
            }
            let mut cos_sim = 0.0f32;
            for d in 0..za.len() {
                cos_sim += za[d] * zb[d];
            }
            let cos_sim = finite_or_zero(cos_sim).clamp(-1.0, 1.0);
            let distance = 1.0 - cos_sim;
            let pair_boost = a.weight.max(b.weight);

            let is_positive = !a.background && !b.background && a.cls >= 0 && a.cls == b.cls;
            if is_positive {
                let weight = spec.positive_weight.max(0.0) * pair_boost;
                if weight <= 0.0 {
                    continue;
                }
                total_loss += weight * distance;
                for d in 0..za.len() {
                    let (add_a, add_b) = (-weight * zb[d], -weight * za[d]);
                    samples[i].grad_z[d] += add_a;
                    samples[j].grad_z[d] += add_b;
                }
                positive += 1;
                if distance < spec.margin {
                    good += 1;
                }
                continue;
            }

            let is_negative = (a.background != b.background)
                || (!a.background && !b.background && a.cls >= 0 && b.cls >= 0 && a.cls != b.cls);
            if !is_negative {
                continue;
            }
            let mut weight = spec.negative_weight.max(0.0) * pair_boost;
            if a.background || b.background {
                weight *= spec.background_weight.max(0.0);
            }
            if weight <= 0.0 {
                continue;
            }
            let violation = spec.margin - distance;
            negative += 1;
            if violation <= 0.0 {
                good += 1;
                continue;
            }
            total_loss += weight * violation * violation;
            let factor = 2.0 * weight * violation; // dL / dcos
            for d in 0..za.len() {
                let (add_a, add_b) = (factor * zb[d], factor * za[d]);
                samples[i].grad_z[d] += add_a;
                samples[j].grad_z[d] += add_b;
            }
        }
    }
    (total_loss, positive, negative, good)
}

/// C++ `scatter_sample_delta`: chain rule through the L2 normalisation,
/// negative gradient (Darknet delta), clipped, into the previous layer's
/// delta buffer.
fn scatter_sample_delta(
    shape: &TensorShape,
    sample: &OmdSample,
    normalizer: f32,
    max_delta: f32,
    delta: &mut [f32],
) {
    if sample.raw.is_empty() || sample.grad_z.len() != sample.raw.len() {
        return;
    }
    let mut dot_gz_z = 0.0f32;
    for d in 0..sample.grad_z.len() {
        dot_gz_z += sample.grad_z[d] * sample.z[d];
    }
    let plane = shape.width * shape.height;
    let spatial = has_spatial_layout(shape);
    for (d, grad_z) in sample.grad_z.iter().enumerate() {
        let mut grad_raw = (grad_z - sample.z[d] * dot_gz_z) / sample.norm;
        if !grad_raw.is_finite() {
            grad_raw = 0.0;
        }
        // Darknet deltas are negative gradients.
        let mut darknet_delta = -normalizer * grad_raw;
        darknet_delta = clip_delta(darknet_delta, max_delta);
        if sample.pooled && spatial {
            let share = if plane > 0 {
                darknet_delta / plane as f32
            } else {
                0.0
            };
            let base = sample.batch * shape.element_count() + d * plane;
            for p in 0..plane {
                delta[base + p] += share;
            }
        } else if sample.vector || !spatial {
            delta[sample.batch * shape.element_count() + d] += darknet_delta;
        } else {
            delta[spatial_index(shape, sample.batch, d, sample.row, sample.col)] += darknet_delta;
        }
    }
}

/// The layer's whole forward: returns the pass-through output plus, when
/// training with a truth tensor, the metric loss and the injected delta.
///
/// `net_w` / `net_h` are the network's input size, needed only for the
/// `small_threshold` boost (which is in pixels of the *input* image).
pub fn omd_loss_forward(
    input: &[f32],
    truth: Option<&[f32]>,
    spec: &OmdLossSpec,
    shape: &TensorShape,
    net_w: usize,
    net_h: usize,
    training: bool,
) -> (Vec<f32>, OmdLossResult) {
    let total = shape.batch * shape.element_count();
    let output = input[..total.min(input.len())].to_vec();
    let mut result = OmdLossResult {
        cost: 0.0,
        accuracy: -1.0,
        delta: vec![0.0; total],
        samples: Vec::new(),
        object_samples: 0,
        background_samples: 0,
        classification_samples: 0,
        positive_pairs: 0,
        negative_pairs: 0,
        good_pairs: 0,
    };
    let Some(truth) = truth else {
        return (output, result);
    };
    if !training || spec.cls_normalizer <= 0.0 || spec.max_samples <= 1 || truth.is_empty() {
        return (output, result);
    }

    let mut samples: Vec<OmdSample> = Vec::new();
    let mut boxes_by_batch: Vec<Vec<TruthBox>> = Vec::new();
    result.object_samples = collect_detection_samples(
        input,
        truth,
        spec,
        shape,
        net_w,
        net_h,
        &mut samples,
        &mut boxes_by_batch,
    );
    // C++ runs the background sampler unconditionally (it is the *pairing* that
    // needs both kinds, not the sampling); only the classification fallback is
    // conditional on finding nothing at all.
    result.background_samples =
        collect_background_samples(input, spec, shape, &mut samples, &boxes_by_batch);
    if samples.is_empty() {
        result.classification_samples =
            collect_classification_samples(input, truth, spec, shape, &mut samples);
    }
    result.samples = samples;
    if result.samples.len() < 2 {
        return (output, result);
    }

    let (total_loss, positive, negative, good) = compute_pair_loss(spec, &mut result.samples);
    let active_pairs = positive + negative;
    let normalizer = if active_pairs > 0 {
        spec.cls_normalizer / active_pairs as f32
    } else {
        0.0
    };
    result.cost = total_loss * normalizer;
    result.accuracy = if active_pairs > 0 {
        100.0 * good as f32 / active_pairs as f32
    } else {
        -1.0
    };
    result.positive_pairs = positive;
    result.negative_pairs = negative;
    result.good_pairs = good;

    for sample in &result.samples {
        scatter_sample_delta(shape, sample, normalizer, spec.max_delta, &mut result.delta);
    }
    for d in result.delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    (output, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> OmdLossSpec {
        OmdLossSpec {
            classes: 2,
            max_boxes: 8,
            truth_size: 6,
            cls_normalizer: 1.0,
            margin: 0.5,
            positive_weight: 1.0,
            negative_weight: 1.0,
            background_weight: 1.0,
            small_threshold: 0.0,
            small_boost: 1.0,
            max_samples: 8,
            background_samples: 0,
            pool: 0,
            eps: 1e-6,
            max_delta: 10.0,
            show_details: false,
        }
    }

    fn shape(batch: usize, channels: usize, h: usize, w: usize) -> TensorShape {
        TensorShape::new(w, h, channels, batch)
    }

    #[test]
    fn omd_axis_layout_pins_cpp_nchw_sample_rows_pairs_and_scatter() {
        // C++ extracts each object vector through `spatial_index`, keeps
        // samples in batch/truth order, forms a pair in that order, and scatters
        // the normalized delta back to the same NCHW pixels (`omd_loss_layer.cpp:86-95,143-185,189-260`).
        let s = spec();
        let sh = shape(2, 2, 2, 3);
        let mut input = Vec::with_capacity(sh.element_count() * sh.batch);
        for b in 0..sh.batch {
            for c in 0..sh.channels {
                for y in 0..sh.height {
                    for x in 0..sh.width {
                        input.push((1 + b * 100 + c * 10 + y * sh.width + x) as f32);
                    }
                }
            }
        }
        let mut truth = vec![0.0f32; sh.batch * s.max_boxes * s.truth_size];
        let stride = s.max_boxes * s.truth_size;
        truth[0..5].copy_from_slice(&[0.5, 0.75, 0.2, 0.2, 0.0]);
        truth[stride..stride + 5].copy_from_slice(&[0.8, 0.25, 0.2, 0.2, 0.0]);
        let (output, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(output, input, "[omd_loss] forward is an identity");
        assert_eq!(result.object_samples, 2);
        assert_eq!(result.positive_pairs, 1);
        assert_eq!(result.samples.len(), 2);
        assert_eq!(
            (
                result.samples[0].batch,
                result.samples[0].row,
                result.samples[0].col
            ),
            (0, 1, 1)
        );
        assert_eq!(
            (
                result.samples[1].batch,
                result.samples[1].row,
                result.samples[1].col
            ),
            (1, 0, 2)
        );
        assert_eq!(result.samples[0].raw, vec![5.0, 15.0]);
        assert_eq!(result.samples[1].raw, vec![103.0, 113.0]);
        for b in 0..sh.batch {
            for c in 0..sh.channels {
                for y in 0..sh.height {
                    for x in 0..sh.width {
                        let index = spatial_index(&sh, b, c, y, x);
                        let sampled = (b == 0 && y == 1 && x == 1) || (b == 1 && y == 0 && x == 2);
                        if sampled {
                            assert!(
                                result.delta[index].abs() > 1.0e-6,
                                "sampled NCHW delta at {index}"
                            );
                        } else {
                            assert_eq!(
                                result.delta[index], 0.0,
                                "non-sampled NCHW delta at {index}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn forward_is_a_pass_through_without_truth() {
        let s = spec();
        let sh = shape(2, 3, 2, 2);
        let input: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
        let (output, result) = omd_loss_forward(&input, None, &s, &sh, 8, 8, true);
        assert_eq!(output, input);
        assert_eq!(result.cost, 0.0);
        assert_eq!(result.accuracy, -1.0);
        assert!(result.delta.iter().all(|d| *d == 0.0));
    }

    #[test]
    fn a_single_object_per_image_gives_no_pairs_and_no_delta() {
        // Two images, one box each: two samples but they are a positive pair
        // (same class), so the loss is non-zero.  With one image and one box
        // there is nothing to pair.
        let s = spec();
        let sh = shape(1, 2, 2, 2);
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let mut truth = vec![0.0f32; s.max_boxes * s.truth_size];
        truth[0] = 0.25; // x
        truth[1] = 0.25; // y
        truth[2] = 0.5; // w
        truth[3] = 0.5; // h
        truth[4] = 1.0; // cls
        let (output, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(output, input);
        assert_eq!(result.object_samples, 1);
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.cost, 0.0, "a single sample has no pairs");
        assert!(result.delta.iter().all(|d| *d == 0.0));
    }

    #[test]
    fn positive_pair_cost_and_sign_follow_the_hand_computation() {
        // Two images with identical feature vectors at their box pixels, same
        // class: cos = 1, distance = 0, so the pair contributes nothing and is
        // already "good".
        let mut s = spec();
        s.cls_normalizer = 1.0;
        let sh = shape(2, 2, 2, 2);
        // Both images have the same values everywhere (batch 2 x c 2 x 2 x 2 = 16).
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        for b in 0..2 {
            let base = b * s.max_boxes * s.truth_size;
            truth[base] = 0.25;
            truth[base + 1] = 0.25;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = 0.0; // class 0 for both
        }
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(result.object_samples, 2);
        assert_eq!(result.positive_pairs, 1);
        assert_eq!(result.negative_pairs, 0);
        assert_eq!(result.good_pairs, 1, "cos = 1 is inside the margin");
        assert!(result.cost.abs() < 1e-6, "cost {}", result.cost);
        assert!((result.accuracy - 100.0).abs() < 1e-6);
        assert!(
            result.delta.iter().all(|d| d.abs() < 1e-6),
            "a satisfied pair has no pull: {:?}",
            result.delta
        );
    }

    #[test]
    fn negative_pair_is_pulled_apart_with_the_squared_margin_violation() {
        // Two images, different classes, opposite feature vectors: cos = -1,
        // distance = 2, violation = margin - 2 < 0 for the default margin 0.5,
        // so nothing happens.  Use a margin above 2 to make it fire.
        let mut s = spec();
        s.classes = 2;
        s.margin = 3.0;
        s.cls_normalizer = 1.0;
        let sh = shape(2, 1, 1, 1);
        let input = vec![1.0f32, -1.0];
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        for (b, cls) in [(0usize, 0.0f32), (1, 1.0)] {
            let base = b * s.max_boxes * s.truth_size;
            truth[base] = 0.5;
            truth[base + 1] = 0.5;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = cls;
        }
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(result.negative_pairs, 1);
        assert_eq!(result.good_pairs, 0, "violation = 3 - 2 = 1 > 0");
        // cos = -1, distance = 2, violation = 1: cost = w * 1^2 / 1 pair = 1.
        assert!((result.cost - 1.0).abs() < 1e-5, "cost {}", result.cost);
        // The two pixels are pushed apart: the delta is a negative gradient of
        // L w.r.t. the raw value, and each image owns one of them.
        assert!(
            result.delta[0] > 0.0 && result.delta[1] < 0.0,
            "delta {:?}",
            result.delta
        );
    }

    #[test]
    fn classification_pairs_use_pooled_or_whole_image_vectors() {
        let mut s = spec();
        s.max_boxes = 0;
        s.classes = 2;
        s.cls_normalizer = 1.0;
        s.margin = 0.5;
        s.pool = 1;
        let sh = shape(2, 2, 2, 2);
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // One-hot class truth, both images class 0.
        let truth = vec![1.0f32, 0.0, 1.0, 0.0];
        let (_, pooled) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(pooled.classification_samples, 2);
        assert!(pooled.samples.iter().all(|smp| smp.pooled));
        assert_eq!(
            pooled.samples[0].raw.len(),
            2,
            "pooled over H*W per channel"
        );
        // Whole-image variant.
        let mut s2 = s.clone();
        s2.pool = 0;
        let (_, vector) = omd_loss_forward(&input, Some(&truth), &s2, &sh, 8, 8, true);
        assert!(vector.samples.iter().all(|smp| smp.vector));
        assert_eq!(
            vector.samples[0].raw.len(),
            8,
            "one entry per input element"
        );
    }

    #[test]
    fn the_delta_is_the_exact_derivative_of_the_pair_loss() {
        // The injected delta is the *negative* gradient of the scalar pair
        // loss with respect to the previous layer's input, so central finite
        // differences of `cost` must reproduce `-delta`.
        let mut s = spec();
        s.classes = 2;
        s.max_boxes = 4;
        s.cls_normalizer = 1.0;
        s.margin = 1.5;
        s.background_samples = 2;
        let sh = shape(2, 2, 2, 2);
        let input: Vec<f32> = (0..16).map(|i| ((i * 5 % 9) as f32 - 4.0) / 4.0).collect();
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        for (b, cls) in [(0usize, 0.0f32), (1, 1.0)] {
            let base = b * s.max_boxes * s.truth_size;
            truth[base] = 0.25;
            truth[base + 1] = 0.25;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = cls;
        }
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert!(result.object_samples > 0 && result.samples.len() >= 2);
        assert!(
            result.delta.iter().any(|d| d.abs() > 1e-6),
            "the test needs a non-zero delta"
        );

        let eps = 1e-3f32;
        for i in 0..input.len() {
            let mut hi = input.clone();
            hi[i] += eps;
            let mut lo = input.clone();
            lo[i] -= eps;
            let (_, r_hi) = omd_loss_forward(&hi, Some(&truth), &s, &sh, 8, 8, true);
            let (_, r_lo) = omd_loss_forward(&lo, Some(&truth), &s, &sh, 8, 8, true);
            let numeric = (r_hi.cost - r_lo.cost) / (2.0 * eps);
            let analytic = -result.delta[i];
            let slack = 5e-2 * analytic.abs().max(1e-2);
            assert!(
                (numeric - analytic).abs() <= slack,
                "element {i}: numeric {numeric} analytic {analytic}"
            );
        }
    }

    #[test]
    fn max_delta_clips_the_injected_delta() {
        let mut s = spec();
        s.classes = 2;
        s.max_boxes = 4;
        s.cls_normalizer = 10.0;
        s.margin = 2.0;
        s.max_delta = 0.05;
        let sh = shape(2, 2, 1, 1);
        let input = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        for (b, cls) in [(0usize, 0.0f32), (1, 1.0)] {
            let base = b * s.max_boxes * s.truth_size;
            truth[base] = 0.5;
            truth[base + 1] = 0.5;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = cls;
        }
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert!(
            result.delta.iter().all(|d| d.abs() <= s.max_delta + 1e-6),
            "{:?}",
            result.delta
        );
        assert!(
            result.delta.iter().any(|d| d.abs() > 0.0),
            "and something was injected"
        );

        // C++ FLT_MAX disables clipping rather than acting as a large finite cap.
        s.max_delta = FLT_MAX;
        let (_, unlimited) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert!(
            unlimited.delta.iter().any(|d| d.abs() > 0.05),
            "FLT_MAX sentinel must bypass the preceding 0.05 clip: {:?}",
            unlimited.delta
        );
    }

    #[test]
    fn small_object_boost_raises_the_sample_weight() {
        let mut s = spec();
        s.classes = 2;
        s.max_boxes = 4;
        s.small_threshold = 4.0;
        s.small_boost = 3.0;
        s.margin = 2.0;
        let sh = shape(2, 1, 2, 2);
        let input = vec![1.0f32, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        // Both boxes are tiny at an 8x8 network input: 0.5 * 8 = 4 px <= 4.
        for (b, cls) in [(0usize, 0.0f32), (1, 1.0)] {
            let base = b * s.max_boxes * s.truth_size;
            truth[base] = 0.25;
            truth[base + 1] = 0.25;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = cls;
        }
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(result.samples.len(), 2);
        assert!(
            result
                .samples
                .iter()
                .all(|smp| (smp.weight - 3.0).abs() < 1e-6),
            "{:?}",
            result.samples.iter().map(|s| s.weight).collect::<Vec<_>>()
        );
        // Without the boost the weight stays 1.
        s.small_boost = 1.0;
        let (_, plain) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert!(
            plain
                .samples
                .iter()
                .all(|smp| (smp.weight - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn background_samples_land_outside_every_box() {
        let mut s = spec();
        s.classes = 2;
        s.max_boxes = 4;
        s.background_samples = 2;
        let sh = shape(1, 1, 2, 2);
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut truth = vec![0.0f32; 1 * s.max_boxes * s.truth_size];
        // One box covering the top-left pixel centre (0.25, 0.25) with w=h=0.5,
        // i.e. the whole [0, 0.5] x [0, 0.5] quadrant.
        truth[0] = 0.25;
        truth[1] = 0.25;
        truth[2] = 0.5;
        truth[3] = 0.5;
        truth[4] = 0.0;
        let (_, result) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, true);
        assert_eq!(result.object_samples, 1);
        assert_eq!(result.background_samples, 2);
        let bg: Vec<usize> = result
            .samples
            .iter()
            .filter(|s| s.background)
            .map(|s| s.row * 2 + s.col)
            .collect();
        assert_eq!(bg.len(), 2);
        for loc in bg {
            let row = loc / 2;
            let col = loc % 2;
            let x = (col as f32 + 0.5) / 2.0;
            let y = (row as f32 + 0.5) / 2.0;
            let inside = x >= 0.0 && x <= 0.5 && y >= 0.0 && y <= 0.5;
            assert!(
                !inside,
                "background sample at ({row},{col}) = ({x},{y}) is inside the box"
            );
        }
    }

    #[test]
    fn training_false_and_zero_normalizer_skip_the_loss() {
        let s = spec();
        let sh = shape(2, 1, 1, 1);
        let input = vec![1.0f32, 1.0];
        let mut truth = vec![0.0f32; 2 * s.max_boxes * s.truth_size];
        truth[0] = 0.5;
        truth[1] = 0.5;
        truth[2] = 0.5;
        truth[3] = 0.5;
        truth[4] = 0.0;
        let base = s.max_boxes * s.truth_size;
        truth[base] = 0.5;
        truth[base + 1] = 0.5;
        truth[base + 2] = 0.5;
        truth[base + 3] = 0.5;
        truth[base + 4] = 1.0;
        let (_, not_training) = omd_loss_forward(&input, Some(&truth), &s, &sh, 8, 8, false);
        assert_eq!(not_training.samples.len(), 0);
        assert_eq!(not_training.cost, 0.0);
        let mut zero = s.clone();
        zero.cls_normalizer = 0.0;
        let (_, no_norm) = omd_loss_forward(&input, Some(&truth), &zero, &sh, 8, 8, true);
        assert_eq!(no_norm.samples.len(), 0);
        assert_eq!(no_norm.cost, 0.0);
    }
}
