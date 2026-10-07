//! Dataset loading, YOLO target assignment, detector loss, and training state.
//!
//! Data preparation is kept separate from graph execution.  A
//! [`TrainingBatch`] owns decoded images and labels; the runtime produces
//! tensors and gradients; this module creates the same head seeds and loss
//! bookkeeping used by Darknet's detector loop.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::model::{LayerKind, Network, TensorShape, YoloIouLoss, YoloSpec};
use crate::obb_labels::{ObbLabel, flip_obb_labels, load_obb_labels};
use crate::obb_training::{ObbYoloTruthBatch, encode_obb_yolo_truth_batch};
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

/// An image batch paired with OBB labels, keeping optional front-point
/// annotations distinct from the center placeholder used by five-field rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbTrainingBatch {
    pub input: Tensor,
    pub image_paths: Vec<PathBuf>,
    pub label_paths: Vec<PathBuf>,
    pub labels: Vec<Vec<ObbLabel>>,
}

impl ObbTrainingBatch {
    pub fn label_count(&self) -> usize {
        self.labels.iter().map(Vec::len).sum()
    }

    /// Encode labels into the source OBB YOLO `[batch][box][7]` truth layout.
    pub fn encode_truths(&self, max_boxes: usize, classes: usize) -> Result<ObbYoloTruthBatch> {
        encode_obb_yolo_truth_batch(&self.labels, max_boxes, classes)
    }

    /// Horizontally flip selected NCHW samples and their OBB labels together.
    ///
    /// `flips` must have one entry per batch sample. This operation acts on the
    /// already-loaded/resized batch; flip selection remains the caller's policy.
    pub fn apply_horizontal_flips(&mut self, flips: &[bool]) -> Result<()> {
        let shape = self.input.shape;
        if flips.len() != shape.batch || self.labels.len() != shape.batch {
            return Err(Error::InvalidConfig(format!(
                "OBB flip mask and labels must match batch size {} (mask {}, labels {})",
                shape.batch,
                flips.len(),
                self.labels.len()
            )));
        }
        if shape.width == 0 || shape.height == 0 || shape.channels == 0 || shape.batch == 0 {
            return Err(Error::InvalidConfig(
                "OBB horizontal flip requires non-zero NCHW dimensions".to_owned(),
            ));
        }
        let sample_len = shape
            .width
            .checked_mul(shape.height)
            .and_then(|area| area.checked_mul(shape.channels))
            .ok_or_else(|| Error::InvalidConfig("OBB flip tensor size overflows".to_owned()))?;
        let expected_len = sample_len
            .checked_mul(shape.batch)
            .ok_or_else(|| Error::InvalidConfig("OBB flip batch size overflows".to_owned()))?;
        if self.input.data.len() != expected_len {
            return Err(Error::InvalidConfig(format!(
                "OBB flip tensor length mismatch: expected {expected_len}, got {}",
                self.input.data.len()
            )));
        }

        let plane_len = shape.width * shape.height;
        for (sample, &flip) in flips.iter().enumerate() {
            if !flip {
                continue;
            }
            let sample_start = sample * sample_len;
            for channel in 0..shape.channels {
                for y in 0..shape.height {
                    let row_start = sample_start + channel * plane_len + y * shape.width;
                    for x in 0..shape.width / 2 {
                        self.input
                            .data
                            .swap(row_start + x, row_start + shape.width - 1 - x);
                    }
                }
            }
            flip_obb_labels(&mut self.labels[sample]);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloTrainingLoss {
    pub loss: f32,
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
    pub positive_matches: usize,
    pub ignored_objectness: usize,
    /// Mean IoU of the box-loss assignments (C++ `tot_iou / count`, summed over heads).
    pub avg_iou: f32,
    /// Number of box-loss assignments (C++ `count`, summed over heads).
    pub box_count: usize,
    /// Number of class-delta assignments (C++ `class_count`; incremented together with `count`).
    pub class_count: usize,
}

/// Per-head box-loss statistics (C++ `tot_iou`, `tot_iou_loss`, `tot_giou_loss`, `count`).
#[derive(Clone, Copy, Debug, Default)]
struct YoloHeadStats {
    tot_iou: f32,
    tot_iou_loss: f32,
    tot_giou_loss: f32,
    count: usize,
}

impl YoloHeadStats {
    fn add(&mut self, (iou, giou): (f32, f32)) {
        self.tot_iou += iou;
        self.tot_iou_loss += 1.0 - iou;
        self.tot_giou_loss += 1.0 - giou;
        self.count += 1;
    }
}

/// C++ `forward_yolo_layer` cost (yolo_layer.cpp:1062-1120) of one head from its final deltas:
/// MSE kind `||delta||^2`; otherwise `iou_normalizer * mean(1 - iou|giou)` plus
/// `obj_normalizer * ||delta without the 4 box channels||^2`.
fn yolo_head_cost(seed: &Tensor, spec: &YoloSpec, stats: &YoloHeadStats) -> f32 {
    let entries = spec.classes + 5;
    let mut total = 0.0f32;
    let mut no_box = 0.0f32;
    for batch in 0..seed.shape.batch {
        for channel in 0..seed.shape.channels {
            let is_box = channel % entries < 4;
            for row in 0..seed.shape.height {
                for col in 0..seed.shape.width {
                    let sq = seed.get(batch, channel, row, col).powi(2);
                    total += sq;
                    if !is_box {
                        no_box += sq;
                    }
                }
            }
        }
    }
    if spec.iou_loss == YoloIouLoss::Mse {
        return total;
    }
    let tot = if spec.iou_loss == YoloIouLoss::Giou {
        stats.tot_giou_loss
    } else {
        stats.tot_iou_loss
    };
    let avg_iou_loss = if stats.count > 0 {
        spec.iou_normalizer * (tot / stats.count as f32)
    } else {
        0.0
    };
    avg_iou_loss + spec.obj_normalizer * no_box
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
    if network.net.obb
        || network
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::ObbYolo(_)))
    {
        return Err(Error::InvalidConfig(
            "OBB YOLO training loss is not implemented; refusing axis-aligned YOLO assignment"
                .to_owned(),
        ));
    }
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
    let mut head_stats = vec![YoloHeadStats::default(); heads.len()];

    let mut positive_matches = 0usize;
    let mut ignored_objectness = 0usize;

    for (head_index, (head, spec)) in heads.iter().enumerate() {
        let stats = &mut head_stats[head_index];
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
                            let _ = apply_yolo_box_delta(
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

        // C++ `process_batch` label loop, run per head: the best anchor is searched over ALL
        // `total` anchors of the layer (every head carries the full list), the primary
        // assignment happens only when that anchor is in this head's mask, and the
        // `iou_thresh` extra anchors are taken from this head's mask independently of it.
        let total_anchors = spec.num.min(spec.anchors.len() / 2);
        for (batch, image_labels) in labels.iter().enumerate() {
            for label in image_labels {
                let class_id = remap_class_id(spec, label.class_id, head.layer_index)?;
                let truth_shift = YoloBox {
                    x: 0.0,
                    y: 0.0,
                    w: label.w,
                    h: label.h,
                };
                let best_n = best_anchor_index(
                    spec,
                    total_anchors,
                    network.net.width,
                    network.net.height,
                    truth_shift,
                )?;

                let grid_x = (label.x * grid_width as f32).clamp(0.0, grid_width as f32 - 1.0e-6);
                let grid_y = (label.y * grid_height as f32).clamp(0.0, grid_height as f32 - 1.0e-6);
                let col = grid_x.floor() as usize;
                let row = grid_y.floor() as usize;
                let class_scale = spec.iou_normalizer * class_multiplier(spec, class_id);

                if let Some(best_offset) = spec.mask.iter().position(|&m| m == best_n) {
                    assign_yolo_label_to_anchor(
                        network,
                        head,
                        spec,
                        seed,
                        stats,
                        batch,
                        best_offset,
                        row,
                        col,
                        label,
                        class_id,
                        class_scale,
                    )?;
                    positive_matches += 1;
                }

                if spec.iou_thresh < 1.0 {
                    for (anchor_offset, &mask_n) in spec.mask.iter().enumerate() {
                        if mask_n == best_n {
                            continue;
                        }
                        let extra_anchor = anchor_pair(spec, mask_n)?;
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
                        assign_yolo_label_to_anchor(
                            network,
                            head,
                            spec,
                            seed,
                            stats,
                            batch,
                            anchor_offset,
                            row,
                            col,
                            label,
                            class_id,
                            class_scale,
                        )?;
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

    // C++ `get_network_cost`: mean of the per-yolo-layer `cost`.
    let loss = if heads.is_empty() {
        0.0
    } else {
        heads
            .iter()
            .zip(&head_stats)
            .map(|((head, spec), stats)| {
                let seed = seeds[head.layer_index]
                    .as_ref()
                    .expect("YOLO seed should be initialized");
                yolo_head_cost(seed, spec, stats)
            })
            .sum::<f32>()
            / heads.len() as f32
    };
    let box_count: usize = head_stats.iter().map(|s| s.count).sum();
    let avg_iou = if box_count > 0 {
        head_stats.iter().map(|s| s.tot_iou).sum::<f32>() / box_count as f32
    } else {
        0.0
    };

    // C++ `backward_yolo_layer_gpu`: `delta_normalizer` scales the delta only when it is
    // propagated (after the cost above has been taken from the unscaled delta).
    for (head, spec) in &heads {
        if spec.delta_normalizer != 1.0 {
            if let Some(seed) = seeds[head.layer_index].as_mut() {
                for delta in &mut seed.data {
                    *delta *= spec.delta_normalizer;
                }
            }
        }
    }

    Ok(YoloTrainingLoss {
        loss,
        layer_output_gradient_seeds: seeds,
        positive_matches,
        ignored_objectness,
        avg_iou,
        box_count,
        class_count: box_count,
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

    // C++ takes `ep_loss_threshold` from the rolling average BEFORE this step's update.
    let mut ep_loss_threshold = equidistant_loss_threshold(network, iteration, state);

    if state.delta_rolling_max == 0.0 {
        state.delta_rolling_max = cur_max;
    }
    state.delta_rolling_max = state.delta_rolling_max * 0.99 + cur_max * 0.01;
    state.delta_rolling_avg = state.delta_rolling_avg * 0.99 + cur_avg * 0.01;

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

        // C++ has no empty guard: with no nonzero delta left (e.g. all rejected by the sigma
        // step) `cur_percent` is NaN, `NaN > pct` is false and the threshold decreases.
        let counter_all = seed.data.iter().filter(|delta| **delta != 0.0).count();
        let mut counter_reject = 0usize;
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
) -> (f32, f32) {
    // C++ `delta_yolo_box`: the `(2 - w*h)` scale only enters the MSE branch; the
    // iou/giou/diou/ciou branches use `iou_normalizer * class_multiplier` alone.
    let scale = (2.0 - truth_bbox.w * truth_bbox.h) * normalizer;
    let tx_raw = output.get(batch, channel_base, row, col);
    let ty_raw = output.get(batch, channel_base + 1, row, col);
    let tw_raw = output.get(batch, channel_base + 2, row, col);
    let th_raw = output.get(batch, channel_base + 3, row, col);
    let tx_output = yolo_xy_output(tx_raw, spec.scale_x_y, spec.new_coords);
    let ty_output = yolo_xy_output(ty_raw, spec.scale_x_y, spec.new_coords);
    // C++ `all_ious` (iou, giou) of the decoded prediction vs the truth, taken before the
    // `w == 0` fix-up, for the cost / avg_iou statistics.
    let decoded = decode_training_bbox(
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
    let ious = (bbox_iou(decoded, truth_bbox), box_giou(decoded, truth_bbox));

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
    ious
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
    load_training_batch_impl(network, data, image_paths, start_index, None)
}

/// Load the next configured training batch with OBB labels and direct image
/// resizing. Augmentation and OBB truth assignment are intentionally separate.
pub fn load_obb_training_batch(
    network: &Network,
    data: &DataConfig,
    start_index: usize,
) -> Result<ObbTrainingBatch> {
    let train_list = data.required_train_list()?;
    let image_paths = load_paths_file(train_list)?;
    load_obb_training_batch_from_paths(network, data, &image_paths, start_index)
}

/// Load an OBB batch from image paths, wrapping at the end of the list like the
/// ordinary detector loader. Class IDs are bounded by the smallest OBB head
/// and, when supplied, the data config's class count.
pub fn load_obb_training_batch_from_paths(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
) -> Result<ObbTrainingBatch> {
    let classes = obb_training_classes(network, data.classes)?;
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
    let total = image_paths.len();
    let prepared_entries: Vec<Result<(Vec<ObbLabel>, PathBuf, PathBuf)>> = input
        .data
        .par_chunks_mut(per_image)
        .enumerate()
        .map(
            |(batch_index, chunk)| -> Result<(Vec<ObbLabel>, PathBuf, PathBuf)> {
                let image_path = &image_paths[(start_index + batch_index) % total];
                let label_path = replace_image_to_label(image_path);
                if !label_path.exists() {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("missing label file {}", label_path.display()),
                    )));
                }

                let labels = load_obb_labels(&label_path, Some(classes))?;
                let image = PlanarRgbImage::load(image_path)?.resize(width, height)?;
                if image.shape.channels != channels {
                    return Err(Error::InvalidConfig(format!(
                        "image {} has {} channels, expected {}",
                        image_path.display(),
                        image.shape.channels,
                        channels
                    )));
                }
                if image.data.len() != chunk.len() {
                    return Err(Error::InvalidConfig(format!(
                        "resized image {} has {} values, expected {}",
                        image_path.display(),
                        image.data.len(),
                        chunk.len()
                    )));
                }
                chunk.copy_from_slice(&image.data);
                Ok((labels, image_path.clone(), label_path))
            },
        )
        .collect();

    let mut batch_image_paths = Vec::with_capacity(batch_size);
    let mut batch_label_paths = Vec::with_capacity(batch_size);
    let mut batch_labels = Vec::with_capacity(batch_size);
    for entry in prepared_entries {
        let (labels, image_path, label_path) = entry?;
        batch_image_paths.push(image_path);
        batch_label_paths.push(label_path);
        batch_labels.push(labels);
    }

    Ok(ObbTrainingBatch {
        input,
        image_paths: batch_image_paths,
        label_paths: batch_label_paths,
        labels: batch_labels,
    })
}

/// Load an OBB batch by direct resize, then apply configured, seeded
/// horizontal flips to each image and its labels together. Other augmentation
/// options are deliberately not applied by this OBB-specific wrapper.
pub fn load_obb_training_batch_with_flips_from_paths(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
    augmentation: &crate::augmentation::DetectorAugmentation,
    seed: u64,
    sample_start: u64,
) -> Result<ObbTrainingBatch> {
    let mut batch = load_obb_training_batch_from_paths(network, data, image_paths, start_index)?;
    let flips = augmentation.horizontal_flip_mask(seed, sample_start, batch.input.shape.batch);
    batch.apply_horizontal_flips(&flips)?;
    Ok(batch)
}

fn obb_training_classes(network: &Network, data_classes: Option<usize>) -> Result<usize> {
    if !network.net.obb {
        return Err(Error::InvalidConfig(
            "OBB batch loading requires [net] obb=1".to_owned(),
        ));
    }
    let head_classes = network
        .layers
        .iter()
        .filter_map(|layer| match &layer.kind {
            LayerKind::ObbYolo(spec) => Some(spec.classes),
            _ => None,
        })
        .min()
        .ok_or_else(|| {
            Error::InvalidConfig(
                "OBB batch loading requires at least one [obb_yolo] head".to_owned(),
            )
        })?;
    Ok(data_classes.map_or(head_classes, |classes| classes.min(head_classes)))
}

/// Training-only augmented loader. Absolute sample ordinals keep random draws
/// independent of subdivision and background-worker scheduling.
pub fn load_augmented_training_batch_from_paths(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
    augmentation: &crate::augmentation::DetectorAugmentation,
    seed: u64,
    sample_start: u64,
) -> Result<TrainingBatch> {
    load_training_batch_impl(
        network,
        data,
        image_paths,
        start_index,
        Some((augmentation, seed, sample_start)),
    )
}

fn load_training_batch_impl(
    network: &Network,
    data: &DataConfig,
    image_paths: &[PathBuf],
    start_index: usize,
    augmentation: Option<(&crate::augmentation::DetectorAugmentation, u64, u64)>,
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
    let classes = data.classes;
    let total = image_paths.len();
    let profiling = crate::loader_profile::enabled();

    // Decode and augment each image independently.  Each chunk of `input` is
    // written by exactly one task, and the augmentation RNG is keyed by the
    // absolute sample ordinal, so the bytes are identical to the sequential
    // loader (`augmented_loader_sample_stream_is_subdivision_independent`).
    let prepared_entries: Vec<Result<(Vec<YoloLabel>, PathBuf, PathBuf)>> = input
        .data
        .par_chunks_mut(per_image)
        .enumerate()
        .map(
            |(batch_index, chunk)| -> Result<(Vec<YoloLabel>, PathBuf, PathBuf)> {
                let image_path = &image_paths[(start_index + batch_index) % total];
                let label_path = replace_image_to_label(image_path);
                if !label_path.exists() {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("missing label file {}", label_path.display()),
                    )));
                }

                let image = PlanarRgbImage::load(image_path)?;
                let labels_start = std::time::Instant::now();
                let labels = load_yolo_labels(&label_path, classes)?;
                let labels_elapsed = labels_start.elapsed();
                let augment_start = std::time::Instant::now();
                let (prepared, labels) = match augmentation {
                    Some((options, seed, sample_start)) if options.mosaic => {
                        if crate::augmentation::mixup_active(seed, sample_start) {
                            // C++ `if (rand_bool()) use_mixup = 0;`: half the batches use
                            // the plain single-image transform.
                            options.apply(
                                &image,
                                &labels,
                                width,
                                height,
                                seed,
                                sample_start.wrapping_add(batch_index as u64),
                            )?
                        } else {
                            let mut members = vec![(image, labels)];
                            for member in 1..4 {
                                let idx = (start_index + batch_index + member * batch_size)
                                    % image_paths.len();
                                let path = &image_paths[idx];
                                let member_label_path = replace_image_to_label(path);
                                if !member_label_path.exists() {
                                    return Err(Error::Io(std::io::Error::new(
                                        std::io::ErrorKind::NotFound,
                                        format!(
                                            "missing label file {}",
                                            member_label_path.display()
                                        ),
                                    )));
                                }
                                members.push((
                                    PlanarRgbImage::load(path)?,
                                    load_yolo_labels(&member_label_path, data.classes)?,
                                ));
                            }
                            options.apply_mosaic(
                                &members,
                                width,
                                height,
                                seed,
                                sample_start.wrapping_add(batch_index as u64),
                            )?
                        }
                    }
                    Some((options, seed, sample_start)) => {
                        let (primary, primary_labels) = options.apply(
                            &image,
                            &labels,
                            width,
                            height,
                            seed,
                            sample_start.wrapping_add(batch_index as u64),
                        )?;
                        if options.mixup && crate::augmentation::mixup_active(seed, sample_start) {
                            let mix_path = &image_paths
                                [(start_index + batch_index + batch_size) % image_paths.len()];
                            let mix_label_path = replace_image_to_label(mix_path);
                            if !mix_label_path.exists() {
                                return Err(Error::Io(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
                                    format!("missing label file {}", mix_label_path.display()),
                                )));
                            }
                            let mix_image = PlanarRgbImage::load(mix_path)?;
                            let mix_labels = load_yolo_labels(&mix_label_path, data.classes)?;
                            let (mix_prepared, mix_labels) = options.apply(
                                &mix_image,
                                &mix_labels,
                                width,
                                height,
                                seed,
                                sample_start.wrapping_add((batch_size + batch_index) as u64),
                            )?;
                            crate::augmentation::blend_mixup(
                                &primary,
                                &primary_labels,
                                &mix_prepared,
                                &mix_labels,
                                options.max_boxes,
                            )?
                        } else {
                            (primary, primary_labels)
                        }
                    }
                    None => (image.resize(width, height)?, labels),
                };
                if prepared.shape.channels != channels {
                    return Err(Error::InvalidConfig(format!(
                        "image {} has {} channels, expected {}",
                        image_path.display(),
                        prepared.shape.channels,
                        channels
                    )));
                }

                let augment_elapsed = augment_start.elapsed();
                let copy_start = std::time::Instant::now();
                chunk.copy_from_slice(&prepared.data);
                if profiling {
                    crate::loader_profile::add_sample();
                    crate::loader_profile::add_labels(labels_elapsed);
                    crate::loader_profile::add_augment(augment_elapsed);
                    crate::loader_profile::add_copy(copy_start.elapsed());
                }
                Ok((labels, image_path.clone(), label_path))
            },
        )
        .collect();

    let mut batch_image_paths = Vec::with_capacity(batch_size);
    let mut batch_label_paths = Vec::with_capacity(batch_size);
    let mut batch_labels = Vec::with_capacity(batch_size);
    for entry in prepared_entries {
        let (labels, image_path, label_path) = entry?;
        batch_image_paths.push(image_path);
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

/// C++ best-anchor search: over the first `total` anchors in cfg order, normalised sizes,
/// strict `>` from `best_iou = 0` / `best_n = 0`.
fn best_anchor_index(
    spec: &YoloSpec,
    total: usize,
    net_width: usize,
    net_height: usize,
    truth_shift: YoloBox,
) -> Result<usize> {
    let mut best_iou = 0.0f32;
    let mut best_n = 0usize;
    for n in 0..total {
        let anchor = anchor_pair(spec, n)?;
        let pred = YoloBox {
            x: 0.0,
            y: 0.0,
            w: anchor.0 / net_width as f32,
            h: anchor.1 / net_height as f32,
        };
        let iou = bbox_iou(pred, truth_shift);
        if iou > best_iou {
            best_iou = iou;
            best_n = n;
        }
    }
    Ok(best_n)
}

/// Box delta + positive objectness + class delta of one truth on one anchor of a head
/// (the shared body of the C++ primary and `iou_thresh` extra-anchor assignments).
#[allow(clippy::too_many_arguments)]
fn assign_yolo_label_to_anchor(
    network: &Network,
    head: &crate::runtime::YoloHeadOutput,
    spec: &YoloSpec,
    seed: &mut Tensor,
    stats: &mut YoloHeadStats,
    batch: usize,
    anchor_offset: usize,
    row: usize,
    col: usize,
    label: &YoloLabel,
    class_id: usize,
    class_scale: f32,
) -> Result<()> {
    let channel_base = anchor_offset * (spec.classes + 5);
    let anchor = anchor_pair(spec, spec.mask[anchor_offset])?;
    let objectness = yolo_objectness_output(
        head.output.get(batch, channel_base + 4, row, col),
        spec.new_coords,
    );
    stats.add(apply_yolo_box_delta(
        &head.output,
        seed,
        batch,
        channel_base,
        row,
        col,
        head.output.shape.width,
        head.output.shape.height,
        network.net.width,
        network.net.height,
        anchor,
        bbox_from_label(label),
        class_scale,
        spec,
    ));
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
    Ok(())
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

/// Source `obb_yolo_delta_front_point` needs an AABB CIoU value and the
/// configured box-IoU x/y derivatives for its fixed-size front-point boxes.
pub(crate) fn obb_front_point_ciou_gradient(
    pred: [f32; 4],
    truth: [f32; 4],
    iou_loss: YoloIouLoss,
) -> (f32, f32, f32) {
    let pred_box = YoloBox {
        x: pred[0],
        y: pred[1],
        w: pred[2],
        h: pred[3],
    };
    let truth_box = YoloBox {
        x: truth[0],
        y: truth[1],
        w: truth[2],
        h: truth[3],
    };
    let ciou = box_ciou(pred_box, truth_box);
    let derivative_kind = if iou_loss == YoloIouLoss::Mse {
        YoloIouLoss::Ciou
    } else {
        iou_loss
    };
    let gradient = dx_box_iou(pred_box, truth_box, derivative_kind);
    (ciou, gradient.dx, gradient.dy)
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
    // C++ `scal_add(scale_x_y, -0.5*(scale_x_y-1))` runs on x,y with and without new_coords
    // (with new_coords the logistic was already applied by the previous layer).
    let activated = if new_coords { raw } else { sigmoid(raw) };
    scale_x_y * activated - 0.5 * (scale_x_y - 1.0)
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
        DataConfig, ObbTrainingBatch, YoloBox, YoloHeadStats, YoloLabel, YoloTrainingState,
        apply_yolo_box_delta, apply_yolo_class_delta, apply_yolo_delta_filtering,
        apply_yolo_positive_objectness, bbox_iou_kind, compute_yolo_training_loss,
        compute_yolo_training_loss_with_state, dx_box_iou, load_obb_training_batch,
        load_obb_training_batch_from_paths, load_obb_training_batch_with_flips_from_paths,
        load_training_batch, load_yolo_labels, replace_image_to_label, yolo_head_cost,
        yolo_xy_output,
    };
    use crate::cfg::ConfigFile;
    use crate::model::{Network, TensorShape, YoloIouLoss, YoloSpec};
    use crate::obb_labels::ObbLabel;
    use crate::runtime::{ExecutableNetwork, Tensor};
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    fn obb_test_network(batch: usize) -> Network {
        obb_test_network_with_classes(batch, 1)
    }

    fn obb_test_network_with_classes(batch: usize, classes: usize) -> Network {
        let cfg = ConfigFile::parse_str(
            "obb-batch.cfg",
            &format!(
                "[net]\nwidth=2\nheight=1\nchannels=3\nbatch={batch}\nobb=1\n[convolutional]\nfilters={}\nsize=1\nstride=1\nactivation=linear\n[obb_yolo]\nclasses={classes}\nnum=1\nanchors=1,1\n",
                classes + 7
            ),
        )
        .expect("cfg parses");
        Network::from_config(cfg).expect("OBB network builds")
    }

    fn empty_data_config(classes: Option<usize>) -> DataConfig {
        DataConfig {
            source: PathBuf::new(),
            classes,
            train: None,
            valid: None,
            names: None,
            backup: None,
            options: Default::default(),
        }
    }

    #[test]
    fn refuses_axis_aligned_training_loss_for_obb_networks() {
        let cfg = ConfigFile::parse_str(
            "obb-train.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("OBB network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = Tensor::from_data(network.net.input_shape(), vec![0.0; 8]).expect("input");
        let forward = executable.forward(&input).expect("forward");
        let error = compute_yolo_training_loss(&network, &forward, &[Vec::new()])
            .expect_err("OBB must not be assigned with AABB YOLO loss");
        assert!(
            error
                .to_string()
                .contains("OBB YOLO training loss is not implemented")
        );

        let hybrid = Network::from_config(
            ConfigFile::parse_str(
                "obb-hybrid-train.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=6\nobb=1\n[yolo]\nallow_obb_truth=1\nclasses=1\nnum=1\nanchors=1,1\n",
            )
            .expect("hybrid cfg parses"),
        )
        .expect("hybrid cfg builds");
        let executable = ExecutableNetwork::new(hybrid.clone());
        let input = Tensor::from_data(hybrid.net.input_shape(), vec![0.0; 6]).expect("input");
        let forward = executable.forward(&input).expect("forward");
        assert!(
            compute_yolo_training_loss(&hybrid, &forward, &[Vec::new()]).is_err(),
            "allow_obb_truth must not enable the unsupported AABB loss path"
        );
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
    fn obb_batch_horizontal_flip_changes_selected_nchw_images_and_labels_together() {
        let explicit = ObbLabel {
            class_id: 0,
            x: 0.25,
            y: 0.375,
            w: 0.25,
            h: 0.5,
            left: 0.125,
            right: 0.375,
            top: 0.125,
            bottom: 0.625,
            front_x: 0.125,
            front_y: 0.625,
            has_front: true,
        };
        let absent = ObbLabel {
            class_id: 1,
            x: 0.375,
            y: 0.625,
            w: 0.25,
            h: 0.5,
            left: 0.25,
            right: 0.5,
            top: 0.375,
            bottom: 0.875,
            front_x: 0.375,
            front_y: 0.625,
            has_front: false,
        };
        let untouched = ObbLabel {
            class_id: 2,
            x: 0.125,
            left: 0.0,
            right: 0.25,
            front_x: 0.125,
            ..absent
        };
        let mut batch = ObbTrainingBatch {
            input: Tensor::from_data(
                TensorShape::new(3, 1, 2, 3),
                vec![
                    1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0, 7.0, 8.0,
                    9.0, 70.0, 80.0, 90.0,
                ],
            )
            .expect("NCHW batch tensor"),
            image_paths: Vec::new(),
            label_paths: Vec::new(),
            labels: vec![vec![explicit], vec![absent], vec![untouched]],
        };

        let before = batch.clone();
        assert!(batch.apply_horizontal_flips(&[true]).is_err());
        assert_eq!(
            batch, before,
            "invalid mask must not partially mutate the batch"
        );

        batch
            .apply_horizontal_flips(&[true, true, false])
            .expect("per-image flips apply");
        assert_eq!(
            batch.input.data,
            vec![
                3.0, 2.0, 1.0, 30.0, 20.0, 10.0, 6.0, 5.0, 4.0, 60.0, 50.0, 40.0, 7.0, 8.0, 9.0,
                70.0, 80.0, 90.0
            ]
        );
        let flipped = batch.labels[0][0];
        assert_eq!(flipped.x, 0.75);
        assert_eq!(flipped.front_x, 0.875);
        assert_eq!(flipped.front_y, explicit.front_y);
        assert_eq!(flipped.left, 0.625);
        assert_eq!(flipped.right, 0.875);
        assert!(flipped.has_front);

        let flipped_absent = batch.labels[1][0];
        assert_eq!(flipped_absent.x, 0.625);
        assert_eq!(flipped_absent.front_x, flipped_absent.x);
        assert_eq!(flipped_absent.front_y, flipped_absent.y);
        assert!(!flipped_absent.has_front);
        assert_eq!(
            batch.labels[2][0], untouched,
            "unselected labels stay unchanged"
        );
    }

    #[test]
    fn obb_batch_loader_resizes_images_and_preserves_front_presence() {
        let dir = temp_path("darknet-obb-training-batch");
        let _ = fs::create_dir_all(&dir);
        let red = dir.join("red.ppm");
        let blue = dir.join("blue.ppm");
        let train_list = dir.join("train.txt");
        let data_path = dir.join("sample.data");
        fs::write(&red, b"P6\n1 1\n255\n\xff\x00\x00").expect("red image writes");
        fs::write(&blue, b"P6\n1 1\n255\n\x00\x00\xff").expect("blue image writes");
        fs::write(dir.join("red.txt"), "0 0.25 0.5 0.5 0.5\n").expect("AABB label writes");
        fs::write(dir.join("blue.txt"), "0 0.75 0.5 0.25 0.5 0.875 0.5\n")
            .expect("oriented label writes");
        fs::write(
            &train_list,
            format!("{}\n{}\n", red.display(), blue.display()),
        )
        .expect("train list writes");
        fs::write(
            &data_path,
            format!("classes=1\ntrain={}\n", train_list.display()),
        )
        .expect("data cfg writes");

        let network = obb_test_network(2);
        let data = DataConfig::from_path(&data_path).expect("data cfg parses");
        let batch = load_obb_training_batch(&network, &data, 1).expect("OBB batch loads");

        assert_eq!(batch.image_paths, vec![blue.clone(), red.clone()]);
        assert_eq!(
            batch.label_paths,
            vec![dir.join("blue.txt"), dir.join("red.txt")]
        );
        assert_eq!(batch.input.shape, network.net.input_shape());
        assert_eq!(batch.label_count(), 2);
        let truths = batch
            .encode_truths(1, 1)
            .expect("labels encode to truth rows");
        assert_eq!(truths.batch_size, 2);
        assert_eq!(truths.max_boxes, 1);
        assert_eq!(
            &truths.values[..7],
            &[0.75, 0.5, 0.25, 0.5, 0.0, 0.875, 0.5]
        );
        assert_eq!(&truths.values[7..], &[0.25, 0.5, 0.5, 0.5, 0.0, 0.25, 0.5]);
        assert!(batch.labels[0][0].has_front);
        assert!(!batch.labels[1][0].has_front);
        assert_eq!(batch.labels[1][0].front_x, batch.labels[1][0].x);
        assert_eq!(batch.labels[1][0].front_y, batch.labels[1][0].y);

        let expected = crate::image::PlanarRgbImage::load(&blue)
            .expect("image loads")
            .resize(network.net.width, network.net.height)
            .expect("image resizes");
        let image_size = network.net.input_shape().element_count();
        assert_eq!(
            &batch.input.data[..image_size],
            expected.data.as_slice(),
            "wrapped first sample should contain the resized blue image"
        );
    }

    #[test]
    fn obb_batch_loader_applies_configured_seeded_flips_to_images_and_labels() {
        let dir = temp_path("darknet-obb-flip-loader");
        let _ = fs::create_dir_all(&dir);
        let first = dir.join("first.ppm");
        let second = dir.join("second.ppm");
        fs::write(&first, b"P6\n2 1\n255\n\xff\x00\x00\x00\x00\xff").expect("first image writes");
        fs::write(&second, b"P6\n2 1\n255\n\x00\xff\x00\xff\xff\x00").expect("second image writes");
        fs::write(dir.join("first.txt"), "0 0.25 0.5 0.25 0.5 0.125 0.5\n")
            .expect("explicit-front label writes");
        fs::write(dir.join("second.txt"), "0 0.75 0.5 0.25 0.5\n")
            .expect("absent-front label writes");

        let paths = [first, second];
        let network = obb_test_network(2);
        let data = empty_data_config(Some(1));
        let augmentation_cfg = ConfigFile::parse_str(
            "obb-flip.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=3\nobb=1\nflip=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\n",
        )
        .expect("OBB augmentation cfg parses");
        let augmentation =
            crate::augmentation::DetectorAugmentation::from_config(&augmentation_cfg)
                .expect("OBB augmentation options parse");
        let seed = 77;
        let sample_start = (0..128)
            .find(|&sample| {
                augmentation
                    .horizontal_flip_mask(seed, sample, 2)
                    .iter()
                    .any(|flip| *flip)
            })
            .expect("some deterministic sample includes a flip");
        let flips = augmentation.horizontal_flip_mask(seed, sample_start, 2);
        assert!(flips.iter().any(|flip| *flip));

        let baseline = load_obb_training_batch_from_paths(&network, &data, &paths, 0)
            .expect("unaugmented OBB batch loads");
        let mut expected = baseline.clone();
        expected
            .apply_horizontal_flips(&flips)
            .expect("reference mask applies");
        let actual = load_obb_training_batch_with_flips_from_paths(
            &network,
            &data,
            &paths,
            0,
            &augmentation,
            seed,
            sample_start,
        )
        .expect("flipped OBB batch loads");
        assert_eq!(actual, expected);
        let sample_len = network.net.input_shape().element_count();
        for (sample, flip) in flips.iter().copied().enumerate() {
            let range = sample * sample_len..(sample + 1) * sample_len;
            let actual_sample = &actual.input.data[range.clone()];
            let baseline_sample = &baseline.input.data[range];
            if flip {
                assert_ne!(actual_sample, baseline_sample);
            } else {
                assert_eq!(actual_sample, baseline_sample);
            }
        }
        assert!(actual.labels[0][0].has_front);
        assert!(!actual.labels[1][0].has_front);
    }

    #[test]
    fn obb_batch_loader_requires_obb_head_and_valid_labels() {
        let standard = Network::from_config(
            ConfigFile::parse_str(
                "standard-batch.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=3\n[yolo]\nclasses=1\nnum=1\nanchors=1,1\n",
            )
            .expect("cfg parses"),
        )
        .expect("standard network builds");
        assert!(
            load_obb_training_batch_from_paths(&standard, &empty_data_config(None), &[], 0)
                .is_err(),
            "standard detector graphs must not use the OBB batch API"
        );

        let network = obb_test_network(1);
        let dir = temp_path("darknet-obb-training-invalid");
        let _ = fs::create_dir_all(&dir);
        let image = dir.join("invalid.ppm");
        fs::write(dir.join("invalid.txt"), "1 0.5 0.5 0.2 0.2\n")
            .expect("out-of-range label writes");
        let data = empty_data_config(Some(1));
        let error = load_obb_training_batch_from_paths(&network, &data, &[image], 0)
            .expect_err("class IDs are bounded by the OBB head");
        assert!(error.to_string().contains("exceeds classes=1"));

        let missing = dir.join("missing.ppm");
        let error = load_obb_training_batch_from_paths(&network, &data, &[missing], 0)
            .expect_err("missing labels must fail the OBB loader");
        assert!(error.to_string().contains("missing label file"));
    }

    #[test]
    fn obb_batch_loader_also_respects_data_class_limit() {
        let network = obb_test_network_with_classes(1, 2);
        let dir = temp_path("darknet-obb-training-data-class-limit");
        let _ = fs::create_dir_all(&dir);
        let image = dir.join("class-one.ppm");
        fs::write(dir.join("class-one.txt"), "1 0.5 0.5 0.2 0.2\n")
            .expect("out-of-range label writes");
        let data = empty_data_config(Some(1));
        let error = load_obb_training_batch_from_paths(&network, &data, &[image], 0)
            .expect_err("data class limit applies even when the head has more classes");
        assert!(error.to_string().contains("exceeds classes=1"));
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
            delta_normalizer: 1.0,
            class_multipliers: None,
            class_map: None,
        };
        let shape = TensorShape {
            width: 1,
            height: 1,
            channels: 6,
            batch: 1,
        };
        let output = Tensor::zeros(shape);
        let truth = YoloBox {
            x: 0.6,
            y: 0.5,
            w: 0.5,
            h: 0.5,
        };
        let mut seed = Tensor::zeros(shape);
        let (iou, giou) = apply_yolo_box_delta(
            &output,
            &mut seed,
            0,
            0,
            0,
            0,
            1,
            1,
            100,
            100,
            (50.0, 50.0),
            truth,
            spec.iou_normalizer,
            &spec,
        );
        // I = .2, U = .3, enclosing box = union box -> giou == iou == 2/3.
        assert!((iou - 2.0 / 3.0).abs() < 1.0e-5 && (giou - 2.0 / 3.0).abs() < 1.0e-5);
        let want = [1.388_889, 0.0, 0.277_778, -1.111_111];
        for (c, w) in want.iter().enumerate() {
            let got = seed.get(0, c, 0, 0);
            assert!((got - w).abs() < 1.0e-4, "channel {c}: {got} vs {w}");
        }

        // MSE keeps the (2 - w*h) scale: (2 - .25) * .5 * (tx - sigmoid(0)), tx = .6*1-0.
        let mse = YoloSpec {
            iou_loss: YoloIouLoss::Mse,
            ..spec
        };
        let mut seed = Tensor::zeros(shape);
        apply_yolo_box_delta(
            &output,
            &mut seed,
            0,
            0,
            0,
            0,
            1,
            1,
            100,
            100,
            (50.0, 50.0),
            truth,
            mse.iou_normalizer,
            &mse,
        );
        assert!((seed.get(0, 0, 0, 0) - 1.75 * 0.5 * 0.1).abs() < 1.0e-5);
    }

    #[test]
    fn yolo_objectness_and_class_deltas_match_bce_finite_differences() {
        let spec = YoloSpec {
            classes: 3,
            num: 1,
            mask: vec![0],
            anchors: vec![1.0, 1.0],
            scale_x_y: 1.0,
            new_coords: false,
            cls_normalizer: 1.0,
            obj_normalizer: 1.7,
            iou_normalizer: 1.0,
            ignore_thresh: 0.5,
            truth_thresh: 1.0,
            iou_thresh: 1.0,
            iou_loss: YoloIouLoss::Iou,
            iou_thresh_kind: YoloIouLoss::Iou,
            focal_loss: false,
            label_smooth_eps: 0.0,
            objectness_smooth: false,
            max_delta: f32::MAX,
            delta_normalizer: 1.0,
            class_multipliers: None,
            class_map: None,
        };
        let shape = TensorShape {
            width: 1,
            height: 1,
            channels: 5 + spec.classes,
            batch: 1,
        };
        let step = 1e-3f32;
        let bce = |raw: f32, target: f32| {
            let p = 1.0 / (1.0 + (-raw).exp());
            -target * p.ln() - (1.0 - target) * (1.0 - p).ln()
        };

        // Positive objectness delta is the negative derivative of BCE-with-logits.
        let raw_objectness = 0.37f32;
        let objectness = 1.0 / (1.0 + (-raw_objectness).exp());
        let mut output = Tensor::zeros(shape);
        output.set(0, 4, 0, 0, raw_objectness);
        let mut seed = Tensor::zeros(shape);
        apply_yolo_positive_objectness(&output, &mut seed, 0, 0, 0, 0, objectness, 1, &spec);
        let numeric = spec.obj_normalizer
            * (bce(raw_objectness + step, 1.0) - bce(raw_objectness - step, 1.0))
            / (2.0 * step);
        let delta = seed.get(0, 4, 0, 0);
        assert!(
            (delta + numeric).abs() < 2e-4,
            "objectness delta {delta}, finite difference {numeric}"
        );

        // One positive and the other negative classes each use target-output,
        // the negative derivative of their independent sigmoid BCE terms.
        let logits = [0.2f32, -0.4, 0.7];
        for (class, &logit) in logits.iter().enumerate() {
            output.set(0, 5 + class, 0, 0, logit);
        }
        seed = Tensor::zeros(shape);
        apply_yolo_class_delta(&output, &mut seed, 0, 5, 0, 0, 1, &spec);
        for (class, &logit) in logits.iter().enumerate() {
            let target = if class == 1 { 1.0 } else { 0.0 };
            let numeric = (bce(logit + step, target) - bce(logit - step, target)) / (2.0 * step);
            let delta = seed.get(0, 5 + class, 0, 0);
            assert!(
                (delta + numeric).abs() < 2e-4,
                "class {class} delta {delta}, finite difference {numeric}"
            );
        }
    }

    #[test]
    fn yolo_iou_and_giou_box_deltas_match_finite_differences() {
        let predicted = YoloBox {
            x: 0.48,
            y: 0.47,
            w: 0.36,
            h: 0.28,
        };
        let truth = YoloBox {
            x: 0.52,
            y: 0.51,
            w: 0.30,
            h: 0.24,
        };
        let h = 1e-3;
        for kind in [YoloIouLoss::Iou, YoloIouLoss::Giou] {
            let gradient = dx_box_iou(predicted, truth, kind);
            let analytic = [gradient.dx, gradient.dy, gradient.dw, gradient.dh];
            for (channel, grad) in analytic.into_iter().enumerate() {
                let perturb = |step: f32| {
                    let mut box_value = predicted;
                    match channel {
                        0 => box_value.x += step,
                        1 => box_value.y += step,
                        2 => box_value.w += step,
                        3 => box_value.h += step,
                        _ => unreachable!(),
                    }
                    bbox_iou_kind(box_value, truth, kind)
                };
                let numeric = (perturb(h) - perturb(-h)) / (2.0 * h);
                // Darknet's legacy delta uses the full opposing-edge difference for
                // width/height; its own grad check converts those two deltas by 0.5.
                let finite_difference_scale = if channel >= 2 { 0.5 } else { 1.0 };
                let comparable = grad * finite_difference_scale;
                assert!(
                    (comparable - numeric).abs() < 2e-3,
                    "{kind:?} box channel {channel}: scaled delta {comparable}, numeric {numeric}"
                );
            }
        }
    }

    #[test]
    fn yolo_diou_and_ciou_deltas_match_the_legacy_cpp_formula() {
        // Keep the prediction strictly inside the target: the enclosing box is
        // fixed by truth, making C++ box.cpp's centre/aspect contributions
        // independently hand-checkable without crossing min/max boundaries.
        let predicted = YoloBox {
            x: 0.48,
            y: 0.47,
            w: 0.25,
            h: 0.20,
        };
        let truth = YoloBox {
            x: 0.50,
            y: 0.50,
            w: 0.80,
            h: 0.70,
        };
        let enclosing_c = truth.w * truth.w + truth.h * truth.h;
        let iou = dx_box_iou(predicted, truth, YoloIouLoss::Iou);
        let center_x = 2.0 * (truth.x - predicted.x) / enclosing_c;
        let center_y = 2.0 * (truth.y - predicted.y) / enclosing_c;

        // C++ box.cpp's DIOU branch differentiates d/c in dx_box_iou even
        // though box_diou's forward score uses pow(d/c, 0.6).
        let diou = dx_box_iou(predicted, truth, YoloIouLoss::Diou);
        assert!((diou.dx - (iou.dx + center_x)).abs() < 1e-6);
        assert!((diou.dy - (iou.dy + center_y)).abs() < 1e-6);
        assert!((diou.dw - iou.dw).abs() < 1e-6);
        assert!((diou.dh - iou.dh).abs() < 1e-6);

        // C++ CIOU adds the same legacy centre derivative and alpha * d(ar_loss)
        // to the IoU derivative; alpha itself is not differentiated.
        let iou_score = bbox_iou_kind(predicted, truth, YoloIouLoss::Iou);
        let ar_gt = truth.w / truth.h;
        let ar_pred = predicted.w / predicted.h;
        let ar_diff = ar_gt.atan() - ar_pred.atan();
        let ar_loss = 4.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff.powi(2);
        let alpha = ar_loss / (1.0 - iou_score + ar_loss + 1e-6);
        let ar_dw = 8.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff * predicted.h;
        let ar_dh = -8.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff * predicted.w;
        let ciou = dx_box_iou(predicted, truth, YoloIouLoss::Ciou);
        assert!((ciou.dx - (iou.dx + center_x)).abs() < 1e-6);
        assert!((ciou.dy - (iou.dy + center_y)).abs() < 1e-6);
        assert!((ciou.dw - (iou.dw + alpha * ar_dw)).abs() < 1e-6);
        assert!((ciou.dh - (iou.dh + alpha * ar_dh)).abs() < 1e-6);
    }

    #[test]
    fn yolo_diou_ciou_edge_and_disjoint_branches_match_cpp_source() {
        // Reproduce the box.cpp DIoU/CIOU center term's min/max edge selection;
        // this intentionally does not finite-difference the legacy surrogate.
        fn legacy_center_gradient(pred: YoloBox, truth: YoloBox) -> [f32; 4] {
            let ct = (pred.y - pred.h * 0.5).min(truth.y - truth.h * 0.5);
            let cb = (pred.y + pred.h * 0.5).max(truth.y + truth.h * 0.5);
            let cl = (pred.x - pred.w * 0.5).min(truth.x - truth.w * 0.5);
            let cr = (pred.x + pred.w * 0.5).max(truth.x + truth.w * 0.5);
            let cw = cr - cl;
            let ch = cb - ct;
            let c = cw * cw + ch * ch;
            let pred_t = pred.y - pred.h * 0.5;
            let pred_b = pred.y + pred.h * 0.5;
            let pred_l = pred.x - pred.w * 0.5;
            let pred_r = pred.x + pred.w * 0.5;
            let truth_t = truth.y - truth.h * 0.5;
            let truth_b = truth.y + truth.h * 0.5;
            let truth_l = truth.x - truth.w * 0.5;
            let truth_r = truth.x + truth.w * 0.5;
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
            let s = (pred.x - truth.x).powi(2) + (pred.y - truth.y).powi(2);
            [
                (2.0 * (truth.x - pred.x) * c - (2.0 * cw * dcw_dx + 2.0 * ch * 0.0) * s) / (c * c),
                (2.0 * (truth.y - pred.y) * c - (2.0 * cw * 0.0 + 2.0 * ch * dch_dy) * s) / (c * c),
                (2.0 * cw * dcw_dw + 2.0 * ch * 0.0) * s / (c * c),
                (2.0 * cw * 0.0 + 2.0 * ch * dch_dh) * s / (c * c),
            ]
        }

        fn legacy_ciou_aspect_gradient(pred: YoloBox, truth: YoloBox) -> [f32; 2] {
            let iw = (pred.x + pred.w * 0.5).min(truth.x + truth.w * 0.5)
                - (pred.x - pred.w * 0.5).max(truth.x - truth.w * 0.5);
            let ih = (pred.y + pred.h * 0.5).min(truth.y + truth.h * 0.5)
                - (pred.y - pred.h * 0.5).max(truth.y - truth.h * 0.5);
            let intersection = iw * ih;
            let union = pred.w * pred.h + truth.w * truth.h - intersection;
            let ar_gt = truth.w / truth.h;
            let ar_pred = pred.w / pred.h;
            let ar_diff = ar_gt.atan() - ar_pred.atan();
            let ar_loss = 4.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff.powi(2);
            // box.cpp uses the unclamped I/U from dx_box_iou and does not
            // differentiate alpha itself.
            let alpha = ar_loss / (1.0 - intersection / union + ar_loss + 1.0e-6);
            [
                alpha * 8.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff * pred.h,
                alpha * -8.0 / (std::f32::consts::PI * std::f32::consts::PI) * ar_diff * pred.w,
            ]
        }

        let cases = [
            // Prediction crosses the enclosing box on the right and both y edges.
            (
                YoloBox {
                    x: 0.56,
                    y: 0.47,
                    w: 0.70,
                    h: 0.60,
                },
                YoloBox {
                    x: 0.50,
                    y: 0.50,
                    w: 0.60,
                    h: 0.50,
                },
                false,
            ),
            // A tied left edge follows box.cpp's strict `<` comparison.
            (
                YoloBox {
                    x: 0.50,
                    y: 0.50,
                    w: 0.60,
                    h: 0.50,
                },
                YoloBox {
                    x: 0.40,
                    y: 0.50,
                    w: 0.40,
                    h: 0.50,
                },
                false,
            ),
            // Horizontal and vertical separation exercise each `Iw <= 0 || Ih <= 0` override.
            (
                YoloBox {
                    x: 0.20,
                    y: 0.50,
                    w: 0.20,
                    h: 0.40,
                },
                YoloBox {
                    x: 0.70,
                    y: 0.50,
                    w: 0.40,
                    h: 0.20,
                },
                true,
            ),
            (
                YoloBox {
                    x: 0.50,
                    y: 0.20,
                    w: 0.40,
                    h: 0.20,
                },
                YoloBox {
                    x: 0.50,
                    y: 0.70,
                    w: 0.20,
                    h: 0.40,
                },
                true,
            ),
        ];
        for (case_index, (pred, truth, disjoint)) in cases.into_iter().enumerate() {
            let overlap_iou = dx_box_iou(pred, truth, YoloIouLoss::Iou);
            let center = legacy_center_gradient(pred, truth);
            let aspect = legacy_ciou_aspect_gradient(pred, truth);
            let diou = dx_box_iou(pred, truth, YoloIouLoss::Diou);
            let ciou = dx_box_iou(pred, truth, YoloIouLoss::Ciou);
            let expected_diou = if disjoint {
                center
            } else {
                [
                    overlap_iou.dx + center[0],
                    overlap_iou.dy + center[1],
                    overlap_iou.dw + center[2],
                    overlap_iou.dh + center[3],
                ]
            };
            let expected_ciou = if disjoint {
                [
                    center[0],
                    center[1],
                    center[2] + aspect[0],
                    center[3] + aspect[1],
                ]
            } else {
                [
                    overlap_iou.dx + center[0],
                    overlap_iou.dy + center[1],
                    overlap_iou.dw + center[2] + aspect[0],
                    overlap_iou.dh + center[3] + aspect[1],
                ]
            };
            for (name, actual, expected) in [
                ("DIoU", [diou.dx, diou.dy, diou.dw, diou.dh], expected_diou),
                ("CIoU", [ciou.dx, ciou.dy, ciou.dw, ciou.dh], expected_ciou),
            ] {
                for channel in 0..4 {
                    assert!(
                        (actual[channel] - expected[channel]).abs() < 2e-6,
                        "{name} delta case {case_index}, channel {channel}: {} vs {}",
                        actual[channel],
                        expected[channel]
                    );
                }
            }
        }
    }

    /// C++ cost (yolo_layer.cpp:1062-1120), hand computed. 1 anchor, 1 class (6 entries),
    /// deltas box (.3,.4,.5,.6), obj .2, cls -.1; count 2, tot(1-iou) .6, tot(1-giou) 1.0,
    /// iou_normalizer .5, obj_normalizer 2.
    /// iou: .5 * (.6/2) + 2 * (.2^2 + .1^2) = .15 + .1 = .25; giou: .5 * (1/2) + .1 = .35;
    /// mse: .09+.16+.25+.36+.04+.01 = .91; no assignments: avg term 0 -> .1.
    #[test]
    fn yolo_head_cost_matches_cpp_formula() {
        let spec = YoloSpec {
            classes: 1,
            num: 1,
            mask: vec![0],
            anchors: vec![50.0, 50.0],
            scale_x_y: 1.0,
            new_coords: false,
            cls_normalizer: 1.0,
            obj_normalizer: 2.0,
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
            delta_normalizer: 1.0,
            class_multipliers: None,
            class_map: None,
        };
        let shape = TensorShape {
            width: 1,
            height: 1,
            channels: 6,
            batch: 1,
        };
        let mut seed = Tensor::zeros(shape);
        for (c, v) in [0.3, 0.4, 0.5, 0.6, 0.2, -0.1].iter().enumerate() {
            seed.set(0, c, 0, 0, *v);
        }
        let stats = YoloHeadStats {
            tot_iou: 1.4,
            tot_iou_loss: 0.6,
            tot_giou_loss: 1.0,
            count: 2,
        };
        assert!((yolo_head_cost(&seed, &spec, &stats) - 0.25).abs() < 1.0e-6);
        let giou = YoloSpec {
            iou_loss: YoloIouLoss::Giou,
            ..spec.clone()
        };
        assert!((yolo_head_cost(&seed, &giou, &stats) - 0.35).abs() < 1.0e-6);
        let mse = YoloSpec {
            iou_loss: YoloIouLoss::Mse,
            ..spec.clone()
        };
        assert!((yolo_head_cost(&seed, &mse, &stats) - 0.91).abs() < 1.0e-6);
        assert!((yolo_head_cost(&seed, &spec, &YoloHeadStats::default()) - 0.1).abs() < 1.0e-6);
    }

    /// C++ applies `scale_x_y * x - 0.5 * (scale_x_y - 1)` to x,y also with new_coords.
    #[test]
    fn yolo_xy_output_scales_with_new_coords_like_cpp() {
        assert!((yolo_xy_output(0.5, 1.1, true) - 0.5).abs() < 1.0e-6);
        assert!((yolo_xy_output(0.2, 1.1, true) - 0.17).abs() < 1.0e-6);
        assert_eq!(yolo_xy_output(0.2, 1.0, true), 0.2);
        let s = 1.0 / (1.0 + (-0.3f32).exp());
        assert!((yolo_xy_output(0.3, 1.2, false) - (1.2 * s - 0.1)).abs() < 1.0e-6);
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

    fn forward_two_head_zero_input(yolo_extra: &str) -> (Network, crate::runtime::ForwardResult) {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            &format!(
                r#"
                [net]
                batch=1
                width=32
                height=32
                channels=3

                [convolutional]
                filters=6
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,12,12,14,40,48
                classes=1
                num=3
                truth_thresh=1.0
                {yolo_extra}

                [route]
                layers=0

                [convolutional]
                filters=12
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=1,2
                anchors=10,12,12,14,40,48
                classes=1
                num=3
                truth_thresh=1.0
                {yolo_extra}
            "#
            ),
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = executable.zero_input();
        let forward = executable.forward(&input).expect("forward runs");
        (network, forward)
    }

    /// #3/#4 vs C++ `process_batch`: best anchor over ALL `num` anchors (cfg order), primary
    /// assignment only in the head whose mask holds it, `iou_thresh` extras in every head from
    /// its own mask. Anchors (10,12) (12,14) (40,48), net 32, truth 10x12 px at the grid
    /// centre: best_n = 0 (iou 1) -> head A (mask 0) primary; anchor 1 iou = 120/168 = .714 > .5
    /// -> head B (mask 1,2, offset 0) gets an extra although best_n is not in its mask;
    /// anchor 2 iou = 120/1920 = .0625 -> no. Zero input -> raw 0 -> objectness .5, so the
    /// positive objectness delta is obj_normalizer * (1 - .5) = .5 and the negative one -.5.
    #[test]
    fn yolo_iou_thresh_extra_anchor_in_other_head_like_cpp() {
        let (network, forward) = forward_two_head_zero_input("iou_thresh=0.5");
        let label = YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 10.0 / 32.0,
            h: 12.0 / 32.0,
        };
        let training = compute_yolo_training_loss(&network, &forward, &[vec![label]])
            .expect("training loss computes");
        assert_eq!(training.positive_matches, 2);
        let a = training.layer_output_gradient_seeds[1]
            .as_ref()
            .expect("head A seed");
        let b = training.layer_output_gradient_seeds[4]
            .as_ref()
            .expect("head B seed");
        assert!((a.get(0, 4, 16, 16) - 0.5).abs() < 1.0e-6);
        assert!(
            (b.get(0, 4, 16, 16) - 0.5).abs() < 1.0e-6,
            "extra anchor objectness"
        );
        assert!(
            (b.get(0, 5, 16, 16) - 0.5).abs() < 1.0e-6,
            "extra anchor class delta"
        );
        assert!(
            (b.get(0, 10, 16, 16) + 0.5).abs() < 1.0e-6,
            "anchor 2 stays negative"
        );
        // without iou_thresh only the primary head is assigned
        let (network, forward) = forward_two_head_zero_input("");
        let label = YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 10.0 / 32.0,
            h: 12.0 / 32.0,
        };
        let training = compute_yolo_training_loss(&network, &forward, &[vec![label]])
            .expect("training loss computes");
        assert_eq!(training.positive_matches, 1);
        let b = training.layer_output_gradient_seeds[4]
            .as_ref()
            .expect("head B seed");
        // anchor 1's decoded box (12x14 px) has iou .714 > ignore_thresh with the truth and the
        // class output .5 > .25 matches -> C++ ignores it (delta 0), nothing positive is written
        assert_eq!(b.get(0, 4, 16, 16), 0.0);
        assert_eq!(b.get(0, 5, 16, 16), 0.0);
    }

    /// #4: the best anchor is searched over all `num` anchors, not the head's mask. Single head
    /// mask=1 (anchor 40x48); truth 10x12 -> best_n = 0, not in the mask -> C++ assigns nothing
    /// (a union-of-masks search would have picked anchor 1).
    #[test]
    fn yolo_best_anchor_search_covers_all_anchors_like_cpp() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=32
                height=32
                channels=3

                [convolutional]
                filters=6
                size=1
                stride=1
                pad=1
                activation=linear

                [yolo]
                mask=1
                anchors=10,12,40,48
                classes=1
                num=2
                truth_thresh=1.0
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let input = executable.zero_input();
        let forward = executable.forward(&input).expect("forward runs");
        let label = YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 10.0 / 32.0,
            h: 12.0 / 32.0,
        };
        let training = compute_yolo_training_loss(&network, &forward, &[vec![label]])
            .expect("training loss computes");
        assert_eq!(training.positive_matches, 0);
        let seed = training.layer_output_gradient_seeds[1]
            .as_ref()
            .expect("seed");
        assert!((seed.get(0, 4, 16, 16) + 0.5).abs() < 1.0e-6);
        assert_eq!(seed.get(0, 5, 16, 16), 0.0);
        assert_eq!(seed.get(0, 0, 16, 16), 0.0);
        assert_eq!(training.box_count, 0);
    }

    /// #10: `delta_normalizer` scales only the propagated delta (backward axpy), not the cost.
    #[test]
    fn yolo_delta_normalizer_scales_seeds_not_loss() {
        let (network, forward) = forward_two_head_zero_input("");
        let (network2, forward2) = forward_two_head_zero_input("delta_normalizer=2.5");
        let label = YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 10.0 / 32.0,
            h: 12.0 / 32.0,
        };
        let base = compute_yolo_training_loss(&network, &forward, &[vec![label.clone()]]).unwrap();
        let scaled = compute_yolo_training_loss(&network2, &forward2, &[vec![label]]).unwrap();
        assert_eq!(base.loss, scaled.loss);
        for head in [1, 4] {
            let a = base.layer_output_gradient_seeds[head].as_ref().unwrap();
            let b = scaled.layer_output_gradient_seeds[head].as_ref().unwrap();
            assert!(a.data.iter().any(|v| *v != 0.0));
            for (x, y) in a.data.iter().zip(&b.data) {
                assert!((x * 2.5 - y).abs() < 1.0e-6);
            }
        }
        assert!(
            (base.layer_output_gradient_seeds[1]
                .as_ref()
                .unwrap()
                .get(0, 4, 16, 16)
                - 0.5)
                .abs()
                < 1.0e-6
        );
        assert_eq!(base.class_count, base.box_count);
    }

    /// #11: C++ takes `ep_loss_threshold = delta_rolling_avg * progress * 1.4` from the rolling
    /// average BEFORE the update. progress = (3 - 1) / (5 - 1) = .5, avg 2.0 -> threshold 1.4,
    /// so 1.395 is zeroed and 3.0 kept (the post-update avg 2.001975 would give 1.4014, but a
    /// post-update ordering with a lower avg, e.g. 1.985, would give 1.3895 and keep 1.395).
    #[test]
    fn yolo_equidistant_threshold_uses_pre_update_rolling_avg() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=1
                height=1
                channels=6
                max_batches=5
                equidistant_point=1

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
            Tensor::from_data(
                network.layers[0].output_shape,
                vec![1.395, 3.0, 0.0, 0.0, 0.0, 0.0],
            )
            .expect("seed tensor"),
        )];
        let mut state = YoloTrainingState {
            delta_rolling_avg: 2.0,
            ..YoloTrainingState::default()
        };
        apply_yolo_delta_filtering(&network, &mut seeds, 3, &mut state);
        assert_eq!(
            seeds[0].as_ref().unwrap().data,
            vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
        );
        // rolling update: avg = 2*.99 + mean(1.395, 3)*.01, max = 3 (first step) * .99 + 3 * .01
        assert!((state.delta_rolling_avg - 2.001_975).abs() < 1.0e-5);
        assert!((state.delta_rolling_max - 3.0).abs() < 1.0e-6);
    }

    /// #11: sigma step rejects the only delta, then the percentage step sees no deltas:
    /// C++ `cur_percent = 0 / 0 = NaN`, `NaN > pct` is false -> threshold decreases by .01.
    /// Hand: rolling max 5, avg .1 -> .099 + .05 = .149; final = .149 + .1 * 1 = .249;
    /// progress_badlabels = (4 - 3) / 3; threshold = 5 - (1/3) * 4.751 = 3.416 < 5 -> zeroed.
    #[test]
    fn yolo_badlabels_percentage_decreases_when_all_deltas_rejected() {
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
            Tensor::from_data(
                network.layers[0].output_shape,
                vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .expect("seed tensor"),
        )];
        let mut state = YoloTrainingState {
            badlabels_reject_threshold: 1.0,
            delta_rolling_max: 5.0,
            delta_rolling_avg: 0.1,
            delta_rolling_std: 0.1,
        };
        apply_yolo_delta_filtering(&network, &mut seeds, 4, &mut state);
        assert!(seeds[0].as_ref().unwrap().data.iter().all(|v| *v == 0.0));
        assert!((state.badlabels_reject_threshold - 0.99).abs() < 1.0e-6);
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
