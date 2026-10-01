//! Dataset loading, YOLO target assignment, detector loss, and training state.
//!
//! Data preparation is kept separate from graph execution.  A
//! [`TrainingBatch`] owns decoded images and labels; the runtime produces
//! tensors and gradients; this module creates the same head seeds and loss
//! bookkeeping used by Darknet's detector loop.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::model::{LayerKind, Network, TensorShape, YoloIouLoss, YoloSpec};
use crate::runtime::{ForwardResult, Tensor};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataConfig {
    pub source: PathBuf,
    pub classes: Option<usize>,
    pub train: Option<PathBuf>,
    pub valid: Option<PathBuf>,
    pub names: Option<PathBuf>,
    pub backup: Option<PathBuf>,
    pub options: BTreeMap<String, String>,
}

impl DataConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let source = path.to_path_buf();
        let parent = source
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut options = BTreeMap::new();

        for (index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }

            let Some((key, value)) = line.split_once('=') else {
                return Err(Error::Parse(format!(
                    "failed to parse {} line {}: expected key=value",
                    source.display(),
                    index + 1
                )));
            };
            options.insert(key.trim().to_owned(), value.trim().to_owned());
        }

        Ok(Self {
            source,
            classes: options
                .get("classes")
                .map(|value| {
                    value.parse::<usize>().map_err(|_| {
                        Error::Parse(format!("invalid classes value in {}", path.display()))
                    })
                })
                .transpose()?,
            train: resolve_optional_path(&options, &parent, "train"),
            valid: resolve_optional_path(&options, &parent, "valid"),
            names: resolve_optional_path(&options, &parent, "names"),
            backup: resolve_optional_path(&options, &parent, "backup"),
            options,
        })
    }

    pub fn required_train_list(&self) -> Result<&Path> {
        self.train.as_deref().ok_or_else(|| {
            Error::InvalidConfig(format!("{} is missing train=...", self.source.display()))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloLabel {
    pub class_id: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingBatch {
    pub input: Tensor,
    pub image_paths: Vec<PathBuf>,
    pub label_paths: Vec<PathBuf>,
    pub labels: Vec<Vec<YoloLabel>>,
}

impl TrainingBatch {
    pub fn label_count(&self) -> usize {
        self.labels.iter().map(Vec::len).sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloTrainingLoss {
    pub loss: f32,
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
    pub positive_matches: usize,
    pub ignored_objectness: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct YoloTrainingState {
    pub badlabels_reject_threshold: f32,
    pub delta_rolling_max: f32,
    pub delta_rolling_avg: f32,
    pub delta_rolling_std: f32,
}

const YOLO_TARGET_EPSILON: f32 = 1.0e-4;

pub fn compute_yolo_training_loss(
    network: &Network,
    forward: &ForwardResult,
    labels: &[Vec<YoloLabel>],
) -> Result<YoloTrainingLoss> {
    compute_yolo_training_loss_with_state(network, forward, labels, 0, None)
}

pub fn compute_yolo_training_loss_with_state(
    network: &Network,
    forward: &ForwardResult,
    labels: &[Vec<YoloLabel>],
    iteration: usize,
    state: Option<&mut YoloTrainingState>,
) -> Result<YoloTrainingLoss> {
    let mut seeds = vec![None; network.layers.len()];
    let batch_size = network.net.batch.max(1);
    if labels.len() != batch_size {
        return Err(Error::InvalidConfig(format!(
            "training label batch mismatch: expected {batch_size}, got {}",
            labels.len()
        )));
    }

    let mut heads = Vec::with_capacity(forward.yolo_heads.len());
    for head in &forward.yolo_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing layer #{} for YOLO head", head.layer_index))
        })?;
        let spec = match &layer.kind {
            LayerKind::Yolo(spec) | LayerKind::GaussianYolo(spec) => spec,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is not a YOLO head",
                    head.layer_index
                )));
            }
        };
        validate_yolo_head_shape(head.output.shape, spec)?;
        if head.output.shape.batch != batch_size {
            return Err(Error::InvalidConfig(format!(
                "YOLO head batch mismatch: expected {batch_size}, got {}",
                head.output.shape.batch
            )));
        }
        seeds[head.layer_index] = Some(Tensor::zeros(head.output.shape));
        heads.push((head, spec));
    }

    let mut positive_matches = 0usize;
    let mut ignored_objectness = 0usize;

    for (head, spec) in &heads {
        let seed = seeds[head.layer_index]
            .as_mut()
            .expect("YOLO seed should be initialized");
        let anchors_per_cell = spec.mask.len();
        let entries_per_anchor = spec.classes + 5;
        let grid_width = head.output.shape.width;
        let grid_height = head.output.shape.height;

        for (batch, batch_labels) in labels.iter().enumerate() {
            for anchor_offset in 0..anchors_per_cell {
                let channel_base = anchor_offset * entries_per_anchor;
                let anchor = anchor_pair(spec, spec.mask[anchor_offset])?;
                for row in 0..grid_height {
                    for col in 0..grid_width {
                        let predicted_bbox = decode_training_bbox(
                            head.output.get(batch, channel_base, row, col),
                            head.output.get(batch, channel_base + 1, row, col),
                            head.output.get(batch, channel_base + 2, row, col),
                            head.output.get(batch, channel_base + 3, row, col),
                            col,
                            row,
                            grid_width,
                            grid_height,
                            network.net.width,
                            network.net.height,
                            anchor,
                            spec.scale_x_y,
                            spec.new_coords,
                        );
                        let objectness = yolo_objectness_output(
                            head.output.get(batch, channel_base + 4, row, col),
                            spec.new_coords,
                        );
                        let class_base = channel_base + 5;
                        let class_match = compare_yolo_class_outputs(
                            &head.output,
                            batch,
                            class_base,
                            row,
                            col,
                            spec.classes,
                            spec.new_coords,
                            0.25,
                        );
                        let mut best_iou = 0.0f32;
                        let mut best_match_iou = 0.0f32;
                        let mut best_label: Option<(&YoloLabel, usize)> = None;

                        for label in batch_labels {
                            let class_id = remap_class_id(spec, label.class_id, head.layer_index)?;
                            let iou = bbox_iou(predicted_bbox, bbox_from_label(label));
                            if class_match && iou > best_match_iou {
                                best_match_iou = iou;
                            }
                            if iou > best_iou {
                                best_iou = iou;
                                best_label = Some((label, class_id));
                            }
                        }

                        let mut object_delta = spec.obj_normalizer * (0.0 - objectness);
                        if best_match_iou > spec.ignore_thresh {
                            ignored_objectness += 1;
                            if spec.objectness_smooth {
                                let smooth = spec.obj_normalizer * (best_match_iou - objectness);
                                if smooth > object_delta {
                                    object_delta = smooth;
                                }
                            } else {
                                object_delta = 0.0;
                            }
                        }
                        seed.set(batch, channel_base + 4, row, col, object_delta);

                        if best_iou > spec.truth_thresh {
                            let Some((label, class_id)) = best_label else {
                                continue;
                            };
                            positive_matches += 1;
                            let iou_multiplier = best_iou * best_iou;
                            let positive_objectness = if spec.objectness_smooth {
                                iou_multiplier
                            } else {
                                1.0
                            };
                            seed.set(
                                batch,
                                channel_base + 4,
                                row,
                                col,
                                spec.obj_normalizer * (positive_objectness - objectness),
                            );
                            apply_yolo_class_delta(
                                &head.output,
                                seed,
                                batch,
                                class_base,
                                row,
                                col,
                                class_id,
                                spec,
                            );
                            if spec.objectness_smooth {
                                let class_output = yolo_class_output(
                                    head.output.get(batch, class_base + class_id, row, col),
                                    spec.new_coords,
                                );
                                seed.set(
                                    batch,
                                    class_base + class_id,
                                    row,
                                    col,
                                    class_multiplier(spec, class_id)
                                        * (iou_multiplier - class_output),
                                );
                            }
                            apply_yolo_box_delta(
                                &head.output,
                                seed,
                                batch,
                                channel_base,
                                row,
                                col,
                                grid_width,
                                grid_height,
                                network.net.width,
                                network.net.height,
                                anchor,
                                bbox_from_label(label),
                                spec.iou_normalizer * class_multiplier(spec, class_id),
                                spec,
                            );
                        }
                    }
                }
            }
        }

        for (batch, image_labels) in labels.iter().enumerate() {
            for label in image_labels {
                let class_id = remap_class_id(spec, label.class_id, head.layer_index)?;
                let Some((best_head_index, best_anchor_offset, _)) =
                    best_anchor_assignment(network, &heads, label)?
                else {
                    continue;
                };
                if heads[best_head_index].0.layer_index != head.layer_index {
                    continue;
                }

                let grid_x = (label.x * grid_width as f32).clamp(0.0, grid_width as f32 - 1.0e-6);
                let grid_y = (label.y * grid_height as f32).clamp(0.0, grid_height as f32 - 1.0e-6);
                let col = grid_x.floor() as usize;
                let row = grid_y.floor() as usize;
                let channel_base = best_anchor_offset * entries_per_anchor;
                let anchor = anchor_pair(spec, spec.mask[best_anchor_offset])?;
                let objectness = yolo_objectness_output(
                    head.output.get(batch, channel_base + 4, row, col),
                    spec.new_coords,
                );
                let class_scale = spec.iou_normalizer * class_multiplier(spec, class_id);

                apply_yolo_box_delta(
                    &head.output,
                    seed,
                    batch,
                    channel_base,
                    row,
                    col,
                    grid_width,
                    grid_height,
                    network.net.width,
                    network.net.height,
                    anchor,
                    bbox_from_label(label),
                    class_scale,
                    spec,
                );
                apply_yolo_positive_objectness(
                    &head.output,
                    seed,
                    batch,
                    channel_base,
                    row,
                    col,
                    objectness,
                    class_id,
                    spec,
                );
                apply_yolo_class_delta(
                    &head.output,
                    seed,
                    batch,
                    channel_base + 5,
                    row,
                    col,
                    class_id,
                    spec,
                );
                positive_matches += 1;

                if spec.iou_thresh < 1.0 {
                    let truth_shift = YoloBox {
                        x: 0.0,
                        y: 0.0,
                        w: label.w,
                        h: label.h,
                    };
                    let best_mask_index = spec.mask[best_anchor_offset];
                    for (anchor_offset, &mask_index) in spec.mask.iter().enumerate() {
                        if mask_index == best_mask_index {
                            continue;
                        }
                        let extra_anchor = anchor_pair(spec, mask_index)?;
                        let anchor_box = YoloBox {
                            x: 0.0,
                            y: 0.0,
                            w: extra_anchor.0 / network.net.width as f32,
                            h: extra_anchor.1 / network.net.height as f32,
                        };
                        if bbox_iou_kind(anchor_box, truth_shift, spec.iou_thresh_kind)
                            <= spec.iou_thresh
                        {
                            continue;
                        }

                        let extra_channel_base = anchor_offset * entries_per_anchor;
                        let extra_objectness = yolo_objectness_output(
                            head.output.get(batch, extra_channel_base + 4, row, col),
                            spec.new_coords,
                        );
                        apply_yolo_box_delta(
                            &head.output,
                            seed,
                            batch,
                            extra_channel_base,
                            row,
                            col,
                            grid_width,
                            grid_height,
                            network.net.width,
                            network.net.height,
                            extra_anchor,
                            bbox_from_label(label),
                            class_scale,
                            spec,
                        );
                        apply_yolo_positive_objectness(
                            &head.output,
                            seed,
                            batch,
                            extra_channel_base,
                            row,
                            col,
                            extra_objectness,
                            class_id,
                            spec,
                        );
                        apply_yolo_class_delta(
                            &head.output,
                            seed,
                            batch,
                            extra_channel_base + 5,
                            row,
                            col,
                            class_id,
                            spec,
                        );
                        positive_matches += 1;
                    }
                }
            }
        }

        if spec.iou_thresh < 1.0 {
            for batch in 0..batch_size {
                for anchor_offset in 0..anchors_per_cell {
                    let channel_base = anchor_offset * entries_per_anchor;
                    for row in 0..grid_height {
                        for col in 0..grid_width {
                            if seed.get(batch, channel_base + 4, row, col) == 0.0 {
                                continue;
                            }
                            let positive_classes = count_positive_class_deltas(
                                seed,
                                batch,
                                channel_base + 5,
                                row,
                                col,
                                spec.classes,
                            );
                            if positive_classes > 0 {
                                let scale = positive_classes as f32;
                                for box_channel in 0..4 {
                                    let value =
                                        seed.get(batch, channel_base + box_channel, row, col)
                                            / scale;
                                    seed.set(batch, channel_base + box_channel, row, col, value);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(state) = state {
        apply_yolo_delta_filtering(network, &mut seeds, iteration, state);
    }

    let loss = seeds
        .iter()
        .flatten()
        .flat_map(|tensor| tensor.data.iter())
        .map(|delta| delta * delta)
        .sum();

    Ok(YoloTrainingLoss {
        loss,
        layer_output_gradient_seeds: seeds,
        positive_matches,
        ignored_objectness,
    })
}

fn apply_yolo_delta_filtering(
    network: &Network,
    seeds: &mut [Option<Tensor>],
    iteration: usize,
    state: &mut YoloTrainingState,
) {
    let start_point = network.net.max_batches.saturating_mul(3) / 4;
    let reject_by_percentage =
        network.net.badlabels_rejection_percentage != 0.0 && start_point < iteration;
    let reject_by_sigmas =
        network.net.num_sigmas_reject_badlabels != 0.0 && start_point < iteration;
    let reject_by_equidistant =
        network.net.equidistant_point != 0 && network.net.equidistant_point < iteration;

    if !reject_by_percentage && !reject_by_sigmas && !reject_by_equidistant {
        return;
    }

    for layer in &network.layers {
        let (LayerKind::Yolo(spec) | LayerKind::GaussianYolo(spec)) = &layer.kind else {
            continue;
        };
        let Some(seed) = seeds.get_mut(layer.index).and_then(Option::as_mut) else {
            continue;
        };
        apply_yolo_delta_filtering_to_seed(network, spec, seed, iteration, start_point, state);
    }
}

fn apply_yolo_delta_filtering_to_seed(
    network: &Network,
    spec: &YoloSpec,
    seed: &mut Tensor,
    iteration: usize,
    start_point: usize,
    state: &mut YoloTrainingState,
) {
    let nonzero: Vec<f32> = seed
        .data
        .iter()
        .copied()
        .filter(|delta| *delta != 0.0)
        .collect();
    if nonzero.is_empty() {
        return;
    }

    let cur_max = nonzero
        .iter()
        .map(|delta| delta.abs())
        .fold(0.0f32, f32::max);
    let cur_avg = nonzero.iter().map(|delta| delta.abs()).sum::<f32>() / nonzero.len() as f32;

    if state.delta_rolling_max == 0.0 {
        state.delta_rolling_max = cur_max;
    }
    state.delta_rolling_max = state.delta_rolling_max * 0.99 + cur_max * 0.01;
    state.delta_rolling_avg = state.delta_rolling_avg * 0.99 + cur_avg * 0.01;

    let mut ep_loss_threshold = equidistant_loss_threshold(network, iteration, state);

    if network.net.num_sigmas_reject_badlabels != 0.0 && start_point < iteration {
        let rolling_std = state.delta_rolling_std;
        let rolling_max = state.delta_rolling_max;
        let rolling_avg = state.delta_rolling_avg;
        let progress_badlabels = if start_point == 0 {
            1.0
        } else {
            (iteration - start_point) as f32 / start_point as f32
        };

        let cur_std = (nonzero
            .iter()
            .map(|delta| (*delta - rolling_avg).powi(2))
            .sum::<f32>()
            / nonzero.len() as f32)
            .sqrt();
        state.delta_rolling_std = state.delta_rolling_std * 0.99 + cur_std * 0.01;

        let final_badlabels_threshold =
            rolling_avg + rolling_std * network.net.num_sigmas_reject_badlabels;
        let mut badlabels_threshold =
            rolling_max - progress_badlabels * (rolling_max - final_badlabels_threshold).abs();
        badlabels_threshold = badlabels_threshold.max(final_badlabels_threshold);
        zero_deltas_above(seed, badlabels_threshold);

        ep_loss_threshold =
            final_badlabels_threshold.min(rolling_avg) * equidistant_progress(network, iteration);
    }

    if network.net.badlabels_rejection_percentage != 0.0 && start_point < iteration {
        if state.badlabels_reject_threshold == 0.0 {
            state.badlabels_reject_threshold = state.delta_rolling_max;
        }

        let counter_all = seed.data.iter().filter(|delta| **delta != 0.0).count();
        let mut counter_reject = 0usize;
        if counter_all != 0 {
            for delta in &mut seed.data {
                if *delta != 0.0 && delta.abs() > state.badlabels_reject_threshold {
                    counter_reject += 1;
                    *delta = 0.0;
                }
            }

            let num_deltas_per_anchor = spec.classes + 5;
            let cur_percent =
                100.0 * (counter_reject * num_deltas_per_anchor) as f32 / counter_all as f32;
            if cur_percent > network.net.badlabels_rejection_percentage {
                state.badlabels_reject_threshold += 0.01;
            } else if state.badlabels_reject_threshold > 0.01 {
                state.badlabels_reject_threshold -= 0.01;
            }
        }
    }

    if network.net.equidistant_point != 0 && network.net.equidistant_point < iteration {
        zero_deltas_below(seed, ep_loss_threshold);
    }
}

fn equidistant_loss_threshold(
    network: &Network,
    iteration: usize,
    state: &YoloTrainingState,
) -> f32 {
    state.delta_rolling_avg * equidistant_progress(network, iteration) * 1.4
}

fn equidistant_progress(network: &Network, iteration: usize) -> f32 {
    let start = network.net.equidistant_point;
    let denominator = network.net.max_batches.saturating_sub(start);
    if denominator == 0 || iteration <= start {
        0.0
    } else {
        (iteration - start) as f32 / denominator as f32
    }
}

fn zero_deltas_above(seed: &mut Tensor, threshold: f32) {
    for delta in &mut seed.data {
        if delta.abs() > threshold {
            *delta = 0.0;
        }
    }
}

fn zero_deltas_below(seed: &mut Tensor, threshold: f32) {
    for delta in &mut seed.data {
        if delta.abs() < threshold {
            *delta = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_yolo_box_delta(
    output: &Tensor,
    seed: &mut Tensor,
    batch: usize,
    channel_base: usize,
    row: usize,
    col: usize,
    grid_width: usize,
    grid_height: usize,
    net_width: usize,
    net_height: usize,
    anchor: (f32, f32),
    truth_bbox: YoloBox,
    normalizer: f32,
    spec: &YoloSpec,
) {
    // C++ `delta_yolo_box`: the `(2 - w*h)` scale only enters the MSE branch; the
    // iou/giou/diou/ciou branches use `iou_normalizer * class_multiplier` alone.
    let scale = (2.0 - truth_bbox.w * truth_bbox.h) * normalizer;
    let tx_raw = output.get(batch, channel_base, row, col);
    let ty_raw = output.get(batch, channel_base + 1, row, col);
    let tw_raw = output.get(batch, channel_base + 2, row, col);
    let th_raw = output.get(batch, channel_base + 3, row, col);
    let tx_output = yolo_xy_output(tx_raw, spec.scale_x_y, spec.new_coords);
    let ty_output = yolo_xy_output(ty_raw, spec.scale_x_y, spec.new_coords);

    match spec.iou_loss {
        YoloIouLoss::Mse => {
            let tx_target = truth_bbox.x * grid_width as f32 - col as f32;
            let ty_target = truth_bbox.y * grid_height as f32 - row as f32;
            let tw_target =
                width_height_target(truth_bbox.w, net_width as f32, anchor.0, spec.new_coords);
            let th_target =
                width_height_target(truth_bbox.h, net_height as f32, anchor.1, spec.new_coords);
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                0,
                scale * (tx_target - tx_output),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                1,
                scale * (ty_target - ty_output),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                2,
                scale * (tw_target - tw_raw),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                3,
                scale * (th_target - th_raw),
            );
        }
        _ => {
            let mut predicted_bbox = decode_training_bbox(
                tx_raw,
                ty_raw,
                tw_raw,
                th_raw,
                col,
                row,
                grid_width,
                grid_height,
                net_width,
                net_height,
                anchor,
                spec.scale_x_y,
                spec.new_coords,
            );
            if predicted_bbox.w == 0.0 {
                predicted_bbox.w = 1.0;
            }
            if predicted_bbox.h == 0.0 {
                predicted_bbox.h = 1.0;
            }

            let mut box_grad = dx_box_iou(predicted_bbox, truth_bbox, spec.iou_loss);
            if !spec.new_coords {
                box_grad.dw *= tw_raw.exp();
                box_grad.dh *= th_raw.exp();
            }

            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                0,
                sanitize_and_clip_delta(box_grad.dx * normalizer, spec.max_delta),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                1,
                sanitize_and_clip_delta(box_grad.dy * normalizer, spec.max_delta),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                2,
                sanitize_and_clip_delta(box_grad.dw * normalizer, spec.max_delta),
            );
            add_box_delta(
                seed,
                batch,
                channel_base,
                row,
                col,
                3,
                sanitize_and_clip_delta(box_grad.dh * normalizer, spec.max_delta),
            );
        }
    }
}

fn add_box_delta(
    seed: &mut Tensor,
    batch: usize,
    channel_base: usize,
    row: usize,
    col: usize,
    box_channel: usize,
    delta: f32,
) {
    let channel = channel_base + box_channel;
    let value = seed.get(batch, channel, row, col) + delta;
    seed.set(batch, channel, row, col, value);
}

#[allow(clippy::too_many_arguments)]
fn apply_yolo_positive_objectness(
    _output: &Tensor,
    seed: &mut Tensor,
    batch: usize,
    channel_base: usize,
    row: usize,
    col: usize,
    objectness: f32,
    class_id: usize,
    spec: &YoloSpec,
) {
    let delta = class_multiplier(spec, class_id) * spec.obj_normalizer * (1.0 - objectness);
    let channel = channel_base + 4;
    if spec.objectness_smooth {
        if seed.get(batch, channel, row, col) == 0.0 {
            seed.set(batch, channel, row, col, delta);
        }
    } else {
        seed.set(batch, channel, row, col, delta);
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_yolo_class_delta(
    output: &Tensor,
    seed: &mut Tensor,
    batch: usize,
    class_base: usize,
    row: usize,
    col: usize,
    class_id: usize,
    spec: &YoloSpec,
) {
    if seed.get(batch, class_base + class_id, row, col) != 0.0 {
        let result_delta = smoothed_target(true, spec.label_smooth_eps)
            - yolo_class_output(
                output.get(batch, class_base + class_id, row, col),
                spec.new_coords,
            );
        if result_delta.is_finite() {
            let value = if spec.class_multipliers.is_some() {
                result_delta * class_multiplier(spec, class_id)
            } else {
                result_delta
            };
            seed.set(batch, class_base + class_id, row, col, value);
        }
        return;
    }

    if spec.focal_loss {
        let pt = yolo_class_output(
            output.get(batch, class_base + class_id, row, col),
            spec.new_coords,
        ) + 1.0e-15;
        let grad = -(1.0 - pt) * (2.0 * pt * pt.ln() + pt - 1.0);
        let alpha = 0.5;
        for class_offset in 0..spec.classes {
            let target = if class_offset == class_id { 1.0 } else { 0.0 };
            let class_output = yolo_class_output(
                output.get(batch, class_base + class_offset, row, col),
                spec.new_coords,
            );
            let delta: f32 = (target - class_output) * alpha * grad;
            if delta.is_finite() {
                seed.set(batch, class_base + class_offset, row, col, delta);
            }
        }
        return;
    }

    for class_offset in 0..spec.classes {
        let class_output = yolo_class_output(
            output.get(batch, class_base + class_offset, row, col),
            spec.new_coords,
        );
        let mut delta =
            smoothed_target(class_offset == class_id, spec.label_smooth_eps) - class_output;
        if !delta.is_finite() {
            continue;
        }
        if spec.class_multipliers.is_some() && class_offset == class_id {
            delta *= class_multiplier(spec, class_id) * spec.cls_normalizer;
        }
        seed.set(batch, class_base + class_offset, row, col, delta);
    }
}

fn remap_class_id(spec: &YoloSpec, raw_class_id: usize, layer_index: usize) -> Result<usize> {
    let class_id = if let Some(map) = &spec.class_map {
        *map.get(raw_class_id).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "label class {} is out of range for YOLO map on layer #{}",
                raw_class_id, layer_index
            ))
        })?
    } else {
        raw_class_id
    };

    if class_id >= spec.classes {
        return Err(Error::InvalidConfig(format!(
            "label class {} exceeds YOLO classes={} for layer #{}",
            class_id, spec.classes, layer_index
        )));
    }

    Ok(class_id)
}

fn class_multiplier(spec: &YoloSpec, class_id: usize) -> f32 {
    spec.class_multipliers
        .as_ref()
        .and_then(|multipliers| multipliers.get(class_id))
        .copied()
        .unwrap_or(1.0)
}

#[allow(clippy::too_many_arguments)]
fn compare_yolo_class_outputs(
    output: &Tensor,
    batch: usize,
    class_base: usize,
    row: usize,
    col: usize,
    classes: usize,
    new_coords: bool,
    threshold: f32,
) -> bool {
    (0..classes).any(|class_offset| {
        yolo_class_output(
            output.get(batch, class_base + class_offset, row, col),
            new_coords,
        ) > threshold
    })
}

fn count_positive_class_deltas(
    seed: &Tensor,
    batch: usize,
    class_base: usize,
    row: usize,
    col: usize,
    classes: usize,
) -> usize {
    (0..classes)
        .filter(|class_offset| seed.get(batch, class_base + class_offset, row, col) > 0.0)
        .count()
}

fn smoothed_target(is_positive: bool, epsilon: f32) -> f32 {
    let base = if is_positive { 1.0 } else { 0.0 };
    if epsilon == 0.0 {
        base
    } else {
        base * (1.0 - epsilon) + 0.5 * epsilon
    }
}

pub fn load_paths_file(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = PathBuf::from(trimmed);
        out.push(resolve_path(parent, &path));
    }

    if out.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "failed to read any paths from {}",
            path.display()
        )));
    }

    Ok(out)
}

pub fn replace_image_to_label(path: impl AsRef<Path>) -> PathBuf {
    let mut out = path.as_ref().to_path_buf();
    out.set_extension("txt");
    out
}

pub fn load_yolo_labels(path: impl AsRef<Path>, classes: Option<usize>) -> Result<Vec<YoloLabel>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let mut labels = Vec::new();

    for (index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 5 {
            return Err(Error::Parse(format!(
                "failed to parse {} line {}: expected 5 YOLO fields",
                path.display(),
                index + 1
            )));
        }

        let class_id = parts[0].parse::<usize>().map_err(|_| {
            Error::Parse(format!(
                "invalid class id in {} line {}",
                path.display(),
                index + 1
            ))
        })?;
        if let Some(classes) = classes
            && class_id >= classes
        {
            return Err(Error::InvalidConfig(format!(
                "class id {} exceeds classes={} in {}",
                class_id,
                classes,
                path.display()
            )));
        }

        let x = parse_unit_f32(parts[1], path, index + 1, "x")?;
        let y = parse_unit_f32(parts[2], path, index + 1, "y")?;
        let w = parse_unit_f32(parts[3], path, index + 1, "w")?;
        let h = parse_unit_f32(parts[4], path, index + 1, "h")?;
        if w <= 0.0 || h <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "invalid box size in {} line {}",
                path.display(),
                index + 1
            )));
        }

        labels.push(YoloLabel {
            class_id,
            x,
            y,
            w,
            h,
        });
    }

    Ok(labels)
}

pub fn load_training_batch(
    network: &Network,
    data: &DataConfig,
    start_index: usize,
) -> Result<TrainingBatch> {
    let train_list = data.required_train_list()?;
    let image_paths = load_paths_file(train_list)?;
    load_training_batch_from_paths(network, data, &image_paths, start_index)
}

pub fn load_training_batch_from_paths(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
) -> Result<TrainingBatch> {
    if image_paths.is_empty() {
        return Err(Error::InvalidConfig(
            "training image list is empty".to_owned(),
        ));
    }
    let batch_size = network.net.batch.max(1);
    let channels = network.net.channels;
    let width = network.net.width;
    let height = network.net.height;
    let shape = TensorShape::new(width, height, channels, batch_size);
    let per_image = shape.element_count();
    let mut input = Tensor::zeros(shape);
    let mut batch_image_paths = Vec::with_capacity(batch_size);
    let mut batch_label_paths = Vec::with_capacity(batch_size);
    let mut batch_labels = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let image_path = &image_paths[(start_index + batch_index) % image_paths.len()];
        let label_path = replace_image_to_label(image_path);
        if !label_path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing label file {}", label_path.display()),
            )));
        }

        let image = PlanarRgbImage::load(image_path)?;
        let prepared = image.resize(width, height)?;
        if prepared.shape.channels != channels {
            return Err(Error::InvalidConfig(format!(
                "image {} has {} channels, expected {}",
                image_path.display(),
                prepared.shape.channels,
                channels
            )));
        }

        let labels = load_yolo_labels(&label_path, data.classes)?;
        let dst = batch_index * per_image;
        input.data[dst..dst + per_image].copy_from_slice(&prepared.data);
        batch_image_paths.push(image_path.clone());
        batch_label_paths.push(label_path);
        batch_labels.push(labels);
    }

    Ok(TrainingBatch {
        input,
        image_paths: batch_image_paths,
        label_paths: batch_label_paths,
        labels: batch_labels,
    })
}

fn parse_unit_f32(value: &str, path: &Path, line: usize, field: &str) -> Result<f32> {
    let parsed = value.parse::<f32>().map_err(|_| {
        Error::Parse(format!(
            "invalid {} in {} line {}",
            field,
            path.display(),
            line
        ))
    })?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(Error::InvalidConfig(format!(
            "invalid {}={} in {} line {}",
            field,
            parsed,
            path.display(),
            line
        )));
    }
    Ok(parsed)
}

fn resolve_optional_path(
    options: &BTreeMap<String, String>,
    parent: &Path,
    key: &str,
) -> Option<PathBuf> {
    options
        .get(key)
        .map(|value| resolve_path(parent, Path::new(value)))
}

fn resolve_path(parent: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        parent.join(path)
    }
}

fn validate_yolo_head_shape(shape: TensorShape, spec: &YoloSpec) -> Result<()> {
    let expected_channels = spec.mask.len() * (spec.classes + 5);
    if shape.channels != expected_channels {
        return Err(Error::InvalidConfig(format!(
            "YOLO head channel mismatch: expected {expected_channels}, got {}",
            shape.channels
        )));
    }
    Ok(())
}

fn best_anchor_assignment(
    network: &Network,
    heads: &[(&crate::runtime::YoloHeadOutput, &YoloSpec)],
    label: &YoloLabel,
) -> Result<Option<(usize, usize, f32)>> {
    let label_w = label.w * network.net.width as f32;
    let label_h = label.h * network.net.height as f32;
    let mut best: Option<(usize, usize, f32)> = None;

    for (head_index, (_, spec)) in heads.iter().enumerate() {
        for (anchor_offset, &mask_idx) in spec.mask.iter().enumerate() {
            let anchor = anchor_pair(spec, mask_idx)?;
            let iou = wh_iou(label_w, label_h, anchor.0, anchor.1);
            if best.is_none_or(|(_, _, best_iou)| iou > best_iou) {
                best = Some((head_index, anchor_offset, iou));
            }
        }
    }

    Ok(best)
}

fn width_height_target(
    label_extent: f32,
    net_extent: f32,
    anchor_extent: f32,
    new_coords: bool,
) -> f32 {
    let scaled = (label_extent * net_extent / anchor_extent).max(YOLO_TARGET_EPSILON);
    if new_coords {
        (scaled / 4.0).sqrt()
    } else {
        scaled.ln()
    }
}

#[derive(Clone, Copy)]
struct YoloBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Clone, Copy)]
struct BoxAbs {
    top: f32,
    bot: f32,
    left: f32,
    right: f32,
}

#[derive(Clone, Copy)]
struct BoxGrad {
    dx: f32,
    dy: f32,
    dw: f32,
    dh: f32,
}

fn bbox_from_label(label: &YoloLabel) -> YoloBox {
    YoloBox {
        x: label.x,
        y: label.y,
        w: label.w,
        h: label.h,
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_training_bbox(
    tx: f32,
    ty: f32,
    tw: f32,
    th: f32,
    col: usize,
    row: usize,
    grid_width: usize,
    grid_height: usize,
    net_width: usize,
    net_height: usize,
    anchor: (f32, f32),
    scale_x_y: f32,
    new_coords: bool,
) -> YoloBox {
    let x = (col as f32 + yolo_xy_output(tx, scale_x_y, new_coords)) / grid_width as f32;
    let y = (row as f32 + yolo_xy_output(ty, scale_x_y, new_coords)) / grid_height as f32;
    let (w, h) = if new_coords {
        (
            tw * tw * 4.0 * anchor.0 / net_width as f32,
            th * th * 4.0 * anchor.1 / net_height as f32,
        )
    } else {
        (
            tw.clamp(-10.0, 10.0).exp() * anchor.0 / net_width as f32,
            th.clamp(-10.0, 10.0).exp() * anchor.1 / net_height as f32,
        )
    };
    YoloBox { x, y, w, h }
}

fn anchor_pair(spec: &YoloSpec, index: usize) -> Result<(f32, f32)> {
    let anchor_index = index
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidConfig("YOLO anchor index overflow".to_owned()))?;
    let width = *spec.anchors.get(anchor_index).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor width for YOLO mask index {index}"))
    })?;
    let height = *spec.anchors.get(anchor_index + 1).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor height for YOLO mask index {index}"))
    })?;
    Ok((width, height))
}

fn wh_iou(w1: f32, h1: f32, w2: f32, h2: f32) -> f32 {
    let intersection = w1.min(w2).max(0.0) * h1.min(h2).max(0.0);
    let union = (w1 * h1 + w2 * h2 - intersection).max(YOLO_TARGET_EPSILON);
    intersection / union
}

fn bbox_iou(lhs: YoloBox, rhs: YoloBox) -> f32 {
    let intersection = box_intersection(lhs, rhs);
    if intersection == 0.0 {
        return 0.0;
    }
    let union = box_union(lhs, rhs, intersection).max(YOLO_TARGET_EPSILON);
    intersection / union
}

fn overlap(center_a: f32, size_a: f32, center_b: f32, size_b: f32) -> f32 {
    let left = (center_a - size_a * 0.5).max(center_b - size_b * 0.5);
    let right = (center_a + size_a * 0.5).min(center_b + size_b * 0.5);
    right - left
}

fn box_intersection(lhs: YoloBox, rhs: YoloBox) -> f32 {
    let w = overlap(lhs.x, lhs.w, rhs.x, rhs.w);
    if w <= 0.0 {
        return 0.0;
    }
    let h = overlap(lhs.y, lhs.h, rhs.y, rhs.h);
    if h <= 0.0 {
        return 0.0;
    }
    w * h
}

fn box_union(lhs: YoloBox, rhs: YoloBox, intersection: f32) -> f32 {
    lhs.w * lhs.h + rhs.w * rhs.h - intersection
}

fn box_ciou(lhs: YoloBox, rhs: YoloBox) -> f32 {
    let box_c = enclosing_box(lhs, rhs);
    let w = box_c.right - box_c.left;
    let h = box_c.bot - box_c.top;
    let c = w * w + h * h;
    let iou = bbox_iou(lhs, rhs);
    if c == 0.0 {
        return iou;
    }
    let u = (lhs.x - rhs.x).powi(2) + (lhs.y - rhs.y).powi(2);
    let d = u / c;
    let ar_gt = rhs.w / rhs.h.max(YOLO_TARGET_EPSILON);
    let ar_pred = lhs.w / lhs.h.max(YOLO_TARGET_EPSILON);
    let ar_loss = 4.0 / (std::f32::consts::PI * std::f32::consts::PI)
        * (ar_gt.atan() - ar_pred.atan()).powi(2);
    let alpha = ar_loss / (1.0 - iou + ar_loss + 1.0e-6);
    iou - (d + alpha * ar_loss)
}

fn box_diou(lhs: YoloBox, rhs: YoloBox) -> f32 {
    let box_c = enclosing_box(lhs, rhs);
    let w = box_c.right - box_c.left;
    let h = box_c.bot - box_c.top;
    let c = w * w + h * h;
    let iou = bbox_iou(lhs, rhs);
    if c == 0.0 {
        return iou;
    }
    let d = (lhs.x - rhs.x).powi(2) + (lhs.y - rhs.y).powi(2);
    iou - (d / c).powf(0.6)
}

fn box_giou(lhs: YoloBox, rhs: YoloBox) -> f32 {
    let box_c = enclosing_box(lhs, rhs);
    let c = (box_c.right - box_c.left) * (box_c.bot - box_c.top);
    let iou = bbox_iou(lhs, rhs);
    if c == 0.0 {
        return iou;
    }
    let u = box_union(lhs, rhs, box_intersection(lhs, rhs));
    iou - (c - u) / c
}

fn bbox_iou_kind(lhs: YoloBox, rhs: YoloBox, iou_kind: YoloIouLoss) -> f32 {
    match iou_kind {
        YoloIouLoss::Iou | YoloIouLoss::Mse => bbox_iou(lhs, rhs),
        YoloIouLoss::Giou => box_giou(lhs, rhs),
        YoloIouLoss::Diou => box_diou(lhs, rhs),
        YoloIouLoss::Ciou => box_ciou(lhs, rhs),
    }
}

fn dx_box_iou(pred: YoloBox, truth: YoloBox, iou_loss: YoloIouLoss) -> BoxGrad {
    let pred_tblr = to_tblr(pred);
    let pred_t = pred_tblr.top.min(pred_tblr.bot);
    let pred_b = pred_tblr.top.max(pred_tblr.bot);
    let pred_l = pred_tblr.left.min(pred_tblr.right);
    let pred_r = pred_tblr.left.max(pred_tblr.right);
    let truth_tblr = to_tblr(truth);

    let x = (pred_b - pred_t) * (pred_r - pred_l);
    let xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    let ih = pred_b.min(truth_tblr.bot) - pred_t.max(truth_tblr.top);
    let iw = pred_r.min(truth_tblr.right) - pred_l.max(truth_tblr.left);
    let intersection = iw * ih;
    let union = x + xhat - intersection;
    let s = (pred.x - truth.x).powi(2) + (pred.y - truth.y).powi(2);
    let giou_cw = pred_r.max(truth_tblr.right) - pred_l.min(truth_tblr.left);
    let giou_ch = pred_b.max(truth_tblr.bot) - pred_t.min(truth_tblr.top);
    let giou_c = giou_cw * giou_ch;

    let dx_wrt_t = -(pred_r - pred_l);
    let dx_wrt_b = pred_r - pred_l;
    let dx_wrt_l = -(pred_b - pred_t);
    let dx_wrt_r = pred_b - pred_t;

    let di_wrt_t = if pred_t > truth_tblr.top { -iw } else { 0.0 };
    let di_wrt_b = if pred_b < truth_tblr.bot { iw } else { 0.0 };
    let di_wrt_l = if pred_l > truth_tblr.left { -ih } else { 0.0 };
    let di_wrt_r = if pred_r < truth_tblr.right { ih } else { 0.0 };

    let du_wrt_t = dx_wrt_t - di_wrt_t;
    let du_wrt_b = dx_wrt_b - di_wrt_b;
    let du_wrt_l = dx_wrt_l - di_wrt_l;
    let du_wrt_r = dx_wrt_r - di_wrt_r;

    let dc_wrt_t = if pred_t < truth_tblr.top {
        -giou_cw
    } else {
        0.0
    };
    let dc_wrt_b = if pred_b > truth_tblr.bot {
        giou_cw
    } else {
        0.0
    };
    let dc_wrt_l = if pred_l < truth_tblr.left {
        -giou_ch
    } else {
        0.0
    };
    let dc_wrt_r = if pred_r > truth_tblr.right {
        giou_ch
    } else {
        0.0
    };

    let mut p_dt = 0.0;
    let mut p_db = 0.0;
    let mut p_dl = 0.0;
    let mut p_dr = 0.0;
    if union > 0.0 {
        p_dt = (union * di_wrt_t - intersection * du_wrt_t) / (union * union);
        p_db = (union * di_wrt_b - intersection * du_wrt_b) / (union * union);
        p_dl = (union * di_wrt_l - intersection * du_wrt_l) / (union * union);
        p_dr = (union * di_wrt_r - intersection * du_wrt_r) / (union * union);
    }

    let (orig_p_dt, orig_p_db, orig_p_dl, orig_p_dr) = (p_dt, p_db, p_dl, p_dr);
    p_dt = if pred_tblr.top < pred_tblr.bot {
        orig_p_dt
    } else {
        orig_p_db
    };
    p_db = if pred_tblr.top < pred_tblr.bot {
        orig_p_db
    } else {
        orig_p_dt
    };
    p_dl = if pred_tblr.left < pred_tblr.right {
        orig_p_dl
    } else {
        orig_p_dr
    };
    p_dr = if pred_tblr.left < pred_tblr.right {
        orig_p_dr
    } else {
        orig_p_dl
    };

    if iou_loss == YoloIouLoss::Giou {
        if giou_c > 0.0 {
            p_dt += (giou_c * du_wrt_t - union * dc_wrt_t) / (giou_c * giou_c);
            p_db += (giou_c * du_wrt_b - union * dc_wrt_b) / (giou_c * giou_c);
            p_dl += (giou_c * du_wrt_l - union * dc_wrt_l) / (giou_c * giou_c);
            p_dr += (giou_c * du_wrt_r - union * dc_wrt_r) / (giou_c * giou_c);
        }
        if iw <= 0.0 || ih <= 0.0 {
            p_dt = (giou_c * du_wrt_t - union * dc_wrt_t) / (giou_c * giou_c);
            p_db = (giou_c * du_wrt_b - union * dc_wrt_b) / (giou_c * giou_c);
            p_dl = (giou_c * du_wrt_l - union * dc_wrt_l) / (giou_c * giou_c);
            p_dr = (giou_c * du_wrt_r - union * dc_wrt_r) / (giou_c * giou_c);
        }
    }

    let ct = (pred.y - pred.h * 0.5).min(truth.y - truth.h * 0.5);
    let cb = (pred.y + pred.h * 0.5).max(truth.y + truth.h * 0.5);
    let cl = (pred.x - pred.w * 0.5).min(truth.x - truth.w * 0.5);
    let cr = (pred.x + pred.w * 0.5).max(truth.x + truth.w * 0.5);
    let cw = cr - cl;
    let ch = cb - ct;
    let c = cw * cw + ch * ch;

    let dct_dy = if pred_t < truth_tblr.top { 1.0 } else { 0.0 };
    let dct_dh = if pred_t < truth_tblr.top { -0.5 } else { 0.0 };
    let dcb_dy = if pred_b > truth_tblr.bot { 1.0 } else { 0.0 };
    let dcb_dh = if pred_b > truth_tblr.bot { 0.5 } else { 0.0 };
    let dcl_dx = if pred_l < truth_tblr.left { 1.0 } else { 0.0 };
    let dcl_dw = if pred_l < truth_tblr.left { -0.5 } else { 0.0 };
    let dcr_dx = if pred_r > truth_tblr.right { 1.0 } else { 0.0 };
    let dcr_dw = if pred_r > truth_tblr.right { 0.5 } else { 0.0 };

    let dcw_dx = dcr_dx - dcl_dx;
    let dcw_dw = dcr_dw - dcl_dw;
    let dch_dy = dcb_dy - dct_dy;
    let dch_dh = dcb_dh - dct_dh;

    let mut p_dx = p_dl + p_dr;
    let mut p_dy = p_dt + p_db;
    let mut p_dw = p_dr - p_dl;
    let mut p_dh = p_db - p_dt;

    if iou_loss == YoloIouLoss::Diou {
        if c > 0.0 {
            p_dx += (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / (c * c);
            p_dy += (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / (c * c);
            p_dw += (2.0 * cw * dcw_dw) * s / (c * c);
            p_dh += (2.0 * ch * dch_dh) * s / (c * c);
        }
        if iw <= 0.0 || ih <= 0.0 {
            p_dx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / (c * c);
            p_dy = (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / (c * c);
            p_dw = (2.0 * cw * dcw_dw) * s / (c * c);
            p_dh = (2.0 * ch * dch_dh) * s / (c * c);
        }
    }

    if iou_loss == YoloIouLoss::Ciou {
        let ar_gt = truth.w / truth.h.max(YOLO_TARGET_EPSILON);
        let ar_pred = pred.w / pred.h.max(YOLO_TARGET_EPSILON);
        let ar_loss = 4.0 / (std::f32::consts::PI * std::f32::consts::PI)
            * (ar_gt.atan() - ar_pred.atan()).powi(2);
        let iou = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };
        let alpha = ar_loss / (1.0 - iou + ar_loss + 1.0e-6);
        let ar_dw = 8.0 / (std::f32::consts::PI * std::f32::consts::PI)
            * (ar_gt.atan() - ar_pred.atan())
            * pred.h;
        let ar_dh = -8.0 / (std::f32::consts::PI * std::f32::consts::PI)
            * (ar_gt.atan() - ar_pred.atan())
            * pred.w;

        if c > 0.0 {
            p_dx += (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / (c * c);
            p_dy += (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / (c * c);
            p_dw += (2.0 * cw * dcw_dw) * s / (c * c) + alpha * ar_dw;
            p_dh += (2.0 * ch * dch_dh) * s / (c * c) + alpha * ar_dh;
        }
        if iw <= 0.0 || ih <= 0.0 {
            p_dx = (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx) * s) / (c * c);
            p_dy = (2.0 * (truth.y - pred.y) * c - (2.0 * ch * dch_dy) * s) / (c * c);
            p_dw = (2.0 * cw * dcw_dw) * s / (c * c) + alpha * ar_dw;
            p_dh = (2.0 * ch * dch_dh) * s / (c * c) + alpha * ar_dh;
        }
    }

    BoxGrad {
        dx: p_dx,
        dy: p_dy,
        dw: p_dw,
        dh: p_dh,
    }
}

fn to_tblr(b: YoloBox) -> BoxAbs {
    BoxAbs {
        top: b.y - b.h * 0.5,
        bot: b.y + b.h * 0.5,
        left: b.x - b.w * 0.5,
        right: b.x + b.w * 0.5,
    }
}

fn enclosing_box(lhs: YoloBox, rhs: YoloBox) -> BoxAbs {
    BoxAbs {
        top: (lhs.y - lhs.h * 0.5).min(rhs.y - rhs.h * 0.5),
        bot: (lhs.y + lhs.h * 0.5).max(rhs.y + rhs.h * 0.5),
        left: (lhs.x - lhs.w * 0.5).min(rhs.x - rhs.w * 0.5),
        right: (lhs.x + lhs.w * 0.5).max(rhs.x + rhs.w * 0.5),
    }
}

fn sanitize_delta(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn sanitize_and_clip_delta(value: f32, max_delta: f32) -> f32 {
    let mut out = sanitize_delta(value);
    if max_delta != f32::MAX {
        out = out.clamp(-max_delta, max_delta);
    }
    out
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn yolo_xy_output(raw: f32, scale_x_y: f32, new_coords: bool) -> f32 {
    if new_coords {
        raw
    } else {
        scale_x_y * sigmoid(raw) - 0.5 * (scale_x_y - 1.0)
    }
}

fn yolo_class_output(raw: f32, new_coords: bool) -> f32 {
    yolo_objectness_output(raw, new_coords)
}

fn yolo_objectness_output(raw: f32, new_coords: bool) -> f32 {
    if new_coords { raw } else { sigmoid(raw) }
}

#[cfg(test)]
mod tests {
    use super::{
        YoloBox, apply_yolo_box_delta, DataConfig, YoloLabel, YoloTrainingState, apply_yolo_delta_filtering,
        compute_yolo_training_loss, compute_yolo_training_loss_with_state, load_training_batch,
        load_yolo_labels, replace_image_to_label,
    };
    use crate::cfg::ConfigFile;
    use crate::model::{Network, TensorShape, YoloIouLoss, YoloSpec};
    use crate::runtime::{ExecutableNetwork, Tensor};
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn parses_data_cfg_and_resolves_relative_paths() {
        let dir = temp_path("darknet-train-data-cfg");
        let _ = fs::create_dir_all(&dir);
        let data_path = dir.join("sample.data");
        fs::write(
            &data_path,
            "classes=2\ntrain=train.txt\nvalid=valid.txt\nnames=labels.names\nbackup=backup\n",
        )
        .expect("data cfg writes");

        let parsed = DataConfig::from_path(&data_path).expect("data cfg parses");
        assert_eq!(parsed.classes, Some(2));
        assert_eq!(
            parsed.train.as_deref(),
            Some(dir.join("train.txt").as_path())
        );
        assert_eq!(
            parsed.valid.as_deref(),
            Some(dir.join("valid.txt").as_path())
        );
        assert_eq!(
            parsed.names.as_deref(),
            Some(dir.join("labels.names").as_path())
        );
        assert_eq!(parsed.backup.as_deref(), Some(dir.join("backup").as_path()));
    }

    #[test]
    fn loads_yolo_labels_and_accepts_empty_files() {
        let path = temp_path("darknet-yolo-labels.txt");
        fs::write(&path, "0 0.5 0.25 0.1 0.2\n").expect("label writes");
        let labels = load_yolo_labels(&path, Some(1)).expect("labels load");
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].class_id, 0);

        fs::write(&path, "").expect("label truncates");
        let empty = load_yolo_labels(&path, Some(1)).expect("empty labels load");
        assert!(empty.is_empty());
    }

    #[test]
    fn replaces_image_extension_with_txt() {
        assert_eq!(
            replace_image_to_label("/tmp/example.jpg"),
            PathBuf::from("/tmp/example.txt")
        );
        assert_eq!(
            replace_image_to_label("/tmp/example.png"),
            PathBuf::from("/tmp/example.txt")
        );
    }

    #[test]
    fn loads_training_batch_from_supported_images() {
        let dir = temp_path("darknet-train-batch");
        let _ = fs::create_dir_all(&dir);
        let data_path = dir.join("sample.data");
        let train_list = dir.join("train.txt");
        let image_path = dir.join("sample.ppm");
        let label_path = dir.join("sample.txt");

        fs::write(&image_path, b"P6\n1 1\n255\n\xff\x80\x00").expect("ppm writes");
        fs::write(&label_path, "0 0.5 0.5 0.25 0.25\n").expect("labels write");
        fs::write(&train_list, format!("{}\n", image_path.display())).expect("train list writes");
        fs::write(
            &data_path,
            format!("classes=1\ntrain={}\n", train_list.display()),
        )
        .expect("data cfg writes");

        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=1
                height=1
                channels=3

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let data = DataConfig::from_path(&data_path).expect("data cfg parses");
        let batch = load_training_batch(&network, &data, 0).expect("batch loads");

        assert_eq!(batch.image_paths, vec![image_path]);
        assert_eq!(batch.label_paths, vec![label_path]);
        assert_eq!(batch.label_count(), 1);
        assert_eq!(batch.input.shape, network.net.input_shape());
    }

    #[test]
    fn computes_nonzero_yolo_training_loss_and_seeds() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=8
                height=8
                channels=3

                [convolutional]
                filters=6
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = executable.zero_input();
        let forward = executable.forward(&input).expect("forward runs");

        let training = compute_yolo_training_loss(
            &network,
            &forward,
            &[vec![YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 0.25,
                h: 0.25,
            }]],
        )
        .expect("training loss computes");

        assert!(training.loss.is_finite());
        assert!(training.loss > 0.0);
        assert_eq!(training.positive_matches, 1);
        let seed = training.layer_output_gradient_seeds[1]
            .as_ref()
            .expect("yolo seed exists");
        assert!(seed.data.iter().any(|value| *value != 0.0));
    }

    /// C++ `delta_yolo_box` iou branch ignores the `(2 - w*h)` scale: delta =
    /// d(IoU)/d(box) * iou_normalizer * class_multiplier (only MSE uses the scale).
    /// Hand computation: pred (.5,.5,.5,.5) vs truth (.6,.5,.5,.5): I=.2, U=.3,
    /// dx=2.7778, dy=0, dw=.5556, dh=-2.2222 (x exp(0) for w/h), times iou_normalizer .5.
    #[test]
    fn yolo_iou_box_delta_has_no_area_scale_like_cpp() {
        let spec = YoloSpec {
            classes: 1,
            num: 1,
            mask: vec![0],
            anchors: vec![50.0, 50.0],
            scale_x_y: 1.0,
            new_coords: false,
            cls_normalizer: 1.0,
            obj_normalizer: 1.0,
            iou_normalizer: 0.5,
            ignore_thresh: 0.5,
            truth_thresh: 1.0,
            iou_thresh: 1.0,
            iou_loss: YoloIouLoss::Iou,
            iou_thresh_kind: YoloIouLoss::Iou,
            focal_loss: false,
            label_smooth_eps: 0.0,
            objectness_smooth: false,
            max_delta: f32::MAX,
            class_multipliers: None,
            class_map: None,
        };
        let shape = TensorShape { width: 1, height: 1, channels: 6, batch: 1 };
        let output = Tensor::zeros(shape);
        let truth = YoloBox { x: 0.6, y: 0.5, w: 0.5, h: 0.5 };
        let mut seed = Tensor::zeros(shape);
        apply_yolo_box_delta(
            &output, &mut seed, 0, 0, 0, 0, 1, 1, 100, 100, (50.0, 50.0), truth,
            spec.iou_normalizer, &spec,
        );
        let want = [1.388_889, 0.0, 0.277_778, -1.111_111];
        for (c, w) in want.iter().enumerate() {
            let got = seed.get(0, c, 0, 0);
            assert!((got - w).abs() < 1.0e-4, "channel {c}: {got} vs {w}");
        }

        // MSE keeps the (2 - w*h) scale: (2 - .25) * .5 * (tx - sigmoid(0)), tx = .6*1-0.
        let mse = YoloSpec { iou_loss: YoloIouLoss::Mse, ..spec };
        let mut seed = Tensor::zeros(shape);
        apply_yolo_box_delta(
            &output, &mut seed, 0, 0, 0, 0, 1, 1, 100, 100, (50.0, 50.0), truth,
            mse.iou_normalizer, &mse,
        );
        assert!((seed.get(0, 0, 0, 0) - 1.75 * 0.5 * 0.1).abs() < 1.0e-5);
    }

    #[test]
    fn yolo_iou_thresh_can_activate_extra_anchors() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=32
                height=32
                channels=3

                [convolutional]
                filters=12
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=0,1
                anchors=10,12,12,14
                classes=1
                num=2
                truth_thresh=1.0
                iou_thresh=0.5
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = executable.zero_input();
        let forward = executable.forward(&input).expect("forward runs");

        let training = compute_yolo_training_loss(
            &network,
            &forward,
            &[vec![YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 10.0 / 32.0,
                h: 12.0 / 32.0,
            }]],
        )
        .expect("training loss computes");

        assert_eq!(training.positive_matches, 2);
        let seed = training.layer_output_gradient_seeds[1]
            .as_ref()
            .expect("yolo seed exists");
        assert_ne!(seed.get(0, 4, 0, 0), 0.0);
        assert_ne!(seed.get(0, 10, 0, 0), 0.0);
    }

    #[test]
    fn yolo_training_state_filters_low_deltas_after_equidistant_point() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=8
                height=8
                channels=3
                max_batches=4
                equidistant_point=1

                [convolutional]
                filters=6
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = executable.zero_input();
        let forward = executable.forward(&input).expect("forward runs");
        let labels = [vec![YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 0.25,
            h: 0.25,
        }]];
        let baseline = compute_yolo_training_loss(&network, &forward, &labels)
            .expect("baseline loss computes");
        assert!(baseline.loss > 0.0);

        let mut state = YoloTrainingState {
            delta_rolling_avg: 10.0,
            ..YoloTrainingState::default()
        };
        let filtered =
            compute_yolo_training_loss_with_state(&network, &forward, &labels, 2, Some(&mut state))
                .expect("filtered loss computes");

        assert_eq!(filtered.loss, 0.0);
        let seed = filtered.layer_output_gradient_seeds[1]
            .as_ref()
            .expect("yolo seed exists");
        assert!(seed.data.iter().all(|delta| *delta == 0.0));
        assert!(state.delta_rolling_avg > 0.0);
    }

    #[test]
    fn yolo_training_state_rejects_high_deltas_by_percentage() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=1
                height=1
                channels=6
                max_batches=4
                badlabels_rejection_percentage=0.2

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let mut seeds = vec![Some(
            crate::runtime::Tensor::from_data(
                network.layers[0].output_shape,
                vec![0.1, 0.2, 5.0, 0.0, 0.0, 0.0],
            )
            .expect("seed tensor"),
        )];
        let mut state = YoloTrainingState {
            badlabels_reject_threshold: 1.0,
            delta_rolling_max: 1.0,
            ..YoloTrainingState::default()
        };

        apply_yolo_delta_filtering(&network, &mut seeds, 4, &mut state);

        let seed = seeds[0].as_ref().expect("seed should remain present");
        assert_eq!(seed.data, vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0]);
        assert!((state.badlabels_reject_threshold - 1.01).abs() < 1.0e-6);
    }

    #[test]
    fn yolo_training_state_rejects_high_deltas_by_sigma_threshold() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=1
                height=1
                channels=6
                max_batches=4
                num_sigmas_reject_badlabels=1

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let mut seeds = vec![Some(
            crate::runtime::Tensor::from_data(
                network.layers[0].output_shape,
                vec![0.1, 0.2, 5.0, 0.0, 0.0, 0.0],
            )
            .expect("seed tensor"),
        )];
        let mut state = YoloTrainingState {
            delta_rolling_max: 5.0,
            delta_rolling_avg: 0.1,
            delta_rolling_std: 0.1,
            ..YoloTrainingState::default()
        };

        apply_yolo_delta_filtering(&network, &mut seeds, 4, &mut state);

        let seed = seeds[0].as_ref().expect("seed should remain present");
        assert_eq!(seed.data, vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0]);
        assert!(state.delta_rolling_std > 0.1);
    }
}
