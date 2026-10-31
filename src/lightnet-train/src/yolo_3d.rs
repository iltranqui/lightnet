//! Pure projected cuboid-head layout and activation math.
//!
//! The CPU runtime stores raw logits. These helpers activate a separate tensor
//! for future loss/decode callers; they do not implement assignment, losses,
//! detection, geometry, or metric consumption. Backward differentiates this
//! activation exactly, not C++'s shortcut training deltas.
//!
//! Hand-derived from `depth_3d@9c9e0b4b`, `src-lib/yolo_3d.cpp:102-109`
//! (indexing) and `:1501-1535` (activation); no compiled-C++ golden.

use crate::error::{Error, Result};
use crate::model::{TensorShape, Yolo3dSpec};
use crate::runtime::Tensor;

fn invalid(message: &str) -> Error {
    Error::InvalidConfig(format!("[yolo_3d] {message}"))
}

/// Checked, projected-only channel-major layout. Anchor indices denote mask
/// slots, not global anchor IDs. All dimensions are frozen at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Yolo3dLayout {
    shape: TensorShape,
    anchors: usize,
    entries: usize,
    classes: usize,
    keypoints: usize,
    spatial: usize,
    elements: usize,
}

impl Yolo3dLayout {
    /// Validate nonempty dimensions and bounded storage before indexing.
    /// Metric channels remain explicitly unsupported in this increment.
    pub fn new(spec: &Yolo3dSpec, shape: TensorShape) -> Result<Self> {
        if spec.use_3d {
            return Err(invalid("metric math is not implemented"));
        }
        if spec.classes == 0 || spec.keypoints == 0 {
            return Err(invalid("classes and keypoints must be positive"));
        }
        spec.validate_shape(shape)?;
        // validate_shape checked every product, including the batch, <= i32::MAX.
        let spatial = shape.width * shape.height;
        Ok(Self {
            shape,
            anchors: spec.mask.len(),
            entries: spec.entries_per_anchor()?,
            classes: spec.classes,
            keypoints: spec.keypoints,
            spatial,
            elements: shape.batch * shape.channels * spatial,
        })
    }

    pub fn entries_per_anchor(&self) -> usize {
        self.entries
    }

    /// Total storage including all batches.
    pub fn storage_len(&self) -> usize {
        self.elements
    }

    /// Index `batch -> anchor slot -> entry plane -> row -> column`.
    /// Returns an error for any out-of-range coordinate, without clamping.
    pub fn entry_index(
        &self,
        batch: usize,
        anchor: usize,
        entry: usize,
        row: usize,
        col: usize,
    ) -> Result<usize> {
        if batch >= self.shape.batch
            || anchor >= self.anchors
            || entry >= self.entries
            || row >= self.shape.height
            || col >= self.shape.width
        {
            return Err(invalid("output coordinate is out of range"));
        }
        Ok(
            ((batch * self.anchors + anchor) * self.entries + entry) * self.spatial
                + row * self.shape.width
                + col,
        )
    }

    /// Entry for a keypoint's u (0), v (1), or visibility (2) plane.
    pub fn vertex_entry(&self, vertex: usize, component: usize) -> Result<usize> {
        if vertex >= self.keypoints || component >= 3 {
            return Err(invalid("vertex or triplet component is out of range"));
        }
        Ok(5 + self.classes + 3 * vertex + component)
    }
}

fn validate_raw(spec: &Yolo3dSpec, raw: &Tensor) -> Result<Yolo3dLayout> {
    let layout = Yolo3dLayout::new(spec, raw.shape)?;
    if !spec.scale_x_y.is_finite() || spec.scale_x_y <= 0.0 {
        return Err(invalid("scale_x_y must be finite and positive"));
    }
    if raw.data.len() != layout.elements || raw.data.iter().any(|v| !v.is_finite()) {
        return Err(invalid("raw tensor must have complete, finite storage"));
    }
    Ok(layout)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

// The diagonal Jacobian is with respect to the original raw storage.
fn activate_entry(spec: &Yolo3dSpec, entry: usize, raw: f32) -> Result<(f32, f32)> {
    let logistic = || {
        let p = sigmoid(raw);
        (p, p * (1.0 - p))
    };
    let (value, slope) = if entry < 2 {
        let (p, derivative) = if spec.new_coords {
            (raw, 1.0)
        } else {
            logistic()
        };
        (
            p * spec.scale_x_y - 0.5 * (spec.scale_x_y - 1.0),
            derivative * spec.scale_x_y,
        )
    } else if entry < 4 || (spec.new_coords && entry < 5 + spec.classes) {
        (raw, 1.0)
    } else {
        logistic()
    };
    if !value.is_finite() || !slope.is_finite() {
        return Err(invalid("activation arithmetic overflowed"));
    }
    Ok((value, slope))
}

/// Activate raw projected output without mutating runtime storage.
///
/// Classic coordinates sigmoid x/y, objectness/classes and all vertex
/// triplets; new_coords only sigmoids vertex triplets. Both modes apply the
/// x/y affine scale. Width/height remain raw (box decode is a separate step).
/// Malformed/nonfinite storage, invalid scale and overflow return errors.
pub fn activate_yolo_3d_output(spec: &Yolo3dSpec, raw: &Tensor) -> Result<Tensor> {
    let layout = validate_raw(spec, raw)?;
    let values = raw
        .data
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            activate_entry(spec, (i / layout.spatial) % layout.entries, value).map(|pair| pair.0)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_data(raw.shape, values)
}

/// Apply the exact activation Jacobian transpose to activated-output seeds.
///
/// This includes the sigmoid and scale_x_y chain derivatives. It deliberately
/// has no GradMode switch: future loss code must implement C++ delta shortcuts
/// explicitly, rather than changing the derivative of this Rust forward.
/// Seed shape/storage must exactly match raw and contain only finite values.
pub fn backward_yolo_3d_activation(
    spec: &Yolo3dSpec,
    raw: &Tensor,
    seeds: &Tensor,
) -> Result<Tensor> {
    let layout = validate_raw(spec, raw)?;
    if seeds.shape != raw.shape
        || seeds.data.len() != layout.elements
        || seeds.data.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "activation seeds must match raw shape and have complete, finite storage",
        ));
    }
    let values = raw
        .data
        .iter()
        .zip(&seeds.data)
        .enumerate()
        .map(|(i, (&value, &seed))| {
            let (_, slope) = activate_entry(spec, (i / layout.spatial) % layout.entries, value)?;
            let gradient = seed * slope;
            if !gradient.is_finite() {
                return Err(invalid("activation seed arithmetic overflowed"));
            }
            Ok(gradient)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_data(raw.shape, values)
}

#[cfg(test)]
mod tests;

/// Projected MSE-box and score loss. Seeds are descent directions in raw layout.
/// Source-derived from depth_3d@9c9e0b4b; no compiled-C++ golden.
#[derive(Clone, Debug)]
pub struct Yolo3dProjectedLoss {
    pub box_loss: f32,
    pub objectness_loss: f32,
    pub class_loss: f32,
    pub seeds: Tensor,
}

impl Yolo3dProjectedLoss {
    pub fn loss(&self) -> f32 {
        self.box_loss + self.objectness_loss + self.class_loss
    }
}

fn projected_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let w = ((a[0] + a[2] * 0.5).min(b[0] + b[2] * 0.5)
        - (a[0] - a[2] * 0.5).max(b[0] - b[2] * 0.5))
    .max(0.0);
    let h = ((a[1] + a[3] * 0.5).min(b[1] + b[3] * 0.5)
        - (a[1] - a[3] * 0.5).max(b[1] - b[3] * 0.5))
    .max(0.0);
    let intersection = w * h;
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Compute the independently usable projected slice. IoU box variants and focal
/// classes are rejected until their source delta contracts have been resolved.
pub fn yolo_3d_projected_loss(
    spec: &Yolo3dSpec,
    raw: &Tensor,
    truths: &crate::cuboid_labels::Yolo3dTruthBatch,
    net_width: usize,
    net_height: usize,
    mode: crate::runtime::GradMode,
) -> Result<Yolo3dProjectedLoss> {
    use crate::model::YoloIouLoss;
    let layout = validate_raw(spec, raw)?;
    if spec.iou_loss != YoloIouLoss::Mse {
        return Err(invalid("projected IoU box losses are not implemented"));
    }
    if spec.focal_loss {
        return Err(invalid(
            "focal class source deltas have no identified scalar loss",
        ));
    }
    if spec.map_file.is_some() {
        return Err(invalid("class map is not resolved"));
    }
    if net_width == 0
        || net_height == 0
        || truths.batch != raw.shape.batch
        || truths.max_boxes != spec.max_boxes
        || truths.truth_size != 11 + 3 * spec.keypoints + usize::from(truths.mixed)
        || truths.counts.len() != truths.batch
        || truths.truths.len()
            != truths
                .batch
                .checked_mul(truths.max_boxes)
                .and_then(|v| v.checked_mul(truths.truth_size))
                .unwrap_or(usize::MAX)
        || truths.truths.iter().any(|v| !v.is_finite())
        || spec.anchors.len() != 2 * spec.num
        || spec.mask.iter().any(|&a| a >= spec.num)
        || spec.anchors.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(invalid("invalid truth, anchors, or network dimensions"));
    }
    let output = activate_yolo_3d_output(spec, raw)?;
    let mut result = Yolo3dProjectedLoss {
        box_loss: 0.0,
        objectness_loss: 0.0,
        class_loss: 0.0,
        seeds: Tensor::zeros(raw.shape),
    };
    let mut records = vec![Vec::<([f32; 4], usize)>::new(); truths.batch];
    for (b, list) in records.iter_mut().enumerate() {
        for t in 0..truths.max_boxes {
            let row = truths.record(b, t)?;
            if row[0] == 0.0 {
                break;
            }
            if truths.mixed && row[truths.truth_size - 1] != 2.0 {
                continue;
            }
            if row[0] < 0.0
                || row[0] > 1.0
                || row[1] < 0.0
                || row[1] > 1.0
                || row[2] <= 0.0
                || row[3] <= 0.0
                || row[4] < 0.0
                || row[4].fract() != 0.0
                || row[4] >= spec.classes as f32
            {
                return Err(invalid("invalid projected truth"));
            }
            list.push(([row[0], row[1], row[2], row[3]], row[4] as usize));
        }
    }
    let mut obj_targets = vec![Some(0.0f32); raw.shape.batch * layout.anchors * layout.spatial];
    let mut class_targets = vec![None; obj_targets.len()];
    let slot = |b: usize, a: usize, y: usize, x: usize| {
        ((b * layout.anchors + a) * raw.shape.height + y) * raw.shape.width + x
    };
    for (b, list) in records.iter().enumerate() {
        for a in 0..layout.anchors {
            for y in 0..raw.shape.height {
                for x in 0..raw.shape.width {
                    let idx = |e| layout.entry_index(b, a, e, y, x).unwrap();
                    let anchor = spec.mask[a];
                    let wh = |e: usize, bias: f32, dim: usize| {
                        let v = output.data[idx(e)];
                        if spec.new_coords {
                            4.0 * v * v * bias / dim as f32
                        } else {
                            v.exp() * bias / dim as f32
                        }
                    };
                    let pred = [
                        (x as f32 + output.data[idx(0)]) / raw.shape.width as f32,
                        (y as f32 + output.data[idx(1)]) / raw.shape.height as f32,
                        wh(2, spec.anchors[2 * anchor], net_width),
                        wh(3, spec.anchors[2 * anchor + 1], net_height),
                    ];
                    if pred.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("box decode overflow"));
                    }
                    let class_match = (0..spec.classes).any(|c| output.data[idx(5 + c)] > 0.25);
                    let mut best = 0.0;
                    let mut best_match = 0.0;
                    let mut best_class = 0;
                    for &(truth, class) in list {
                        let iou = projected_iou(pred, truth);
                        if iou > best {
                            best = iou;
                            best_class = class;
                        }
                        if class_match && iou > best_match {
                            best_match = iou;
                        }
                    }
                    let s = slot(b, a, y, x);
                    if best_match > spec.ignore_thresh {
                        obj_targets[s] = if spec.objectness_smooth {
                            Some(best_match)
                        } else {
                            None
                        };
                    }
                    if best > spec.truth_thresh {
                        obj_targets[s] = Some(if spec.objectness_smooth {
                            best * best
                        } else {
                            1.0
                        });
                        class_targets[s] = Some(best_class);
                    }
                }
            }
        }
        for &(truth, class) in list {
            let mut best = 0;
            let mut best_iou = 0.0;
            for n in 0..spec.num {
                let iou = projected_iou(
                    [0.0, 0.0, truth[2], truth[3]],
                    [
                        0.0,
                        0.0,
                        spec.anchors[2 * n] / net_width as f32,
                        spec.anchors[2 * n + 1] / net_height as f32,
                    ],
                );
                if iou > best_iou {
                    best_iou = iou;
                    best = n;
                }
            }
            let Some(a) = spec.mask.iter().position(|&n| n == best) else {
                continue;
            };
            let x = ((truth[0] * raw.shape.width as f32) as usize).min(raw.shape.width - 1);
            let y = ((truth[1] * raw.shape.height as f32) as usize).min(raw.shape.height - 1);
            let s = slot(b, a, y, x);
            if !spec.objectness_smooth
                || obj_targets[s].is_none()
                || obj_targets[s].is_some_and(|v| {
                    (v - output.data[layout.entry_index(b, a, 4, y, x).unwrap()]).abs() == 0.0
                })
            {
                obj_targets[s] = Some(1.0);
            }
            class_targets[s] = Some(class);
            let tw = truth[2] * net_width as f32 / spec.anchors[2 * best];
            let th = truth[3] * net_height as f32 / spec.anchors[2 * best + 1];
            let targets = [
                truth[0] * raw.shape.width as f32 - x as f32,
                truth[1] * raw.shape.height as f32 - y as f32,
                if spec.new_coords {
                    (tw / 4.0).sqrt()
                } else {
                    tw.ln()
                },
                if spec.new_coords {
                    (th / 4.0).sqrt()
                } else {
                    th.ln()
                },
            ];
            let weight = (2.0 - truth[2] * truth[3]) * spec.iou_normalizer;
            for (e, target) in targets.into_iter().enumerate() {
                let i = layout.entry_index(b, a, e, y, x)?;
                let d = target - output.data[i];
                result.box_loss += 0.5 * weight * d * d;
                let slope = if mode == crate::runtime::GradMode::Cpp {
                    1.0
                } else {
                    activate_entry(spec, e, raw.data[i])?.1
                };
                result.seeds.data[i] += weight * d * slope;
            }
        }
    }
    for b in 0..raw.shape.batch {
        for a in 0..layout.anchors {
            for y in 0..raw.shape.height {
                for x in 0..raw.shape.width {
                    let s = slot(b, a, y, x);
                    for (e, target, weight) in
                        std::iter::once((4, obj_targets[s], spec.obj_normalizer)).chain(
                            (0..spec.classes).map(|c| {
                                (
                                    5 + c,
                                    class_targets[s].map(|class| {
                                        (if c == class { 1.0 } else { 0.0 })
                                            * (1.0 - spec.label_smooth_eps)
                                            + 0.5 * spec.label_smooth_eps
                                    }),
                                    1.0,
                                )
                            }),
                        )
                    {
                        if let Some(target) = target {
                            let i = layout.entry_index(b, a, e, y, x)?;
                            let d = target - output.data[i];
                            result.objectness_loss +=
                                if e == 4 { 0.5 * weight * d * d } else { 0.0 };
                            result.class_loss += if e != 4 { 0.5 * weight * d * d } else { 0.0 };
                            let slope = if mode == crate::runtime::GradMode::Cpp {
                                1.0
                            } else {
                                activate_entry(spec, e, raw.data[i])?.1
                            };
                            result.seeds.data[i] = weight * d * slope;
                        }
                    }
                }
            }
        }
    }
    if !result.loss().is_finite() || result.seeds.data.iter().any(|v| !v.is_finite()) {
        return Err(invalid("loss arithmetic overflow"));
    }
    Ok(result)
}
