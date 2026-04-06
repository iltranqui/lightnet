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
            Error::InvalidConfig(format!(
                "{} is missing train=...",
                self.source.display()
            ))
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

const YOLO_TARGET_EPSILON: f32 = 1.0e-4;

pub fn compute_yolo_training_loss(
    network: &Network,
    forward: &ForwardResult,
    labels: &[Vec<YoloLabel>],
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

    let mut assignments =
        BTreeMap::<(usize, usize, usize, usize, usize), Vec<YoloLabel>>::new();
    for (batch, image_labels) in labels.iter().enumerate() {
        for label in image_labels {
            let Some((head_index, anchor_offset, _match_iou)) =
                best_anchor_assignment(network, &heads, label)?
            else {
                continue;
            };
            let head = heads[head_index].0;
            let grid_x = (label.x * head.output.shape.width as f32)
                .clamp(0.0, head.output.shape.width as f32 - 1.0e-6);
            let grid_y = (label.y * head.output.shape.height as f32)
                .clamp(0.0, head.output.shape.height as f32 - 1.0e-6);
            let col = grid_x.floor() as usize;
            let row = grid_y.floor() as usize;
            let key = (batch, head.layer_index, anchor_offset, row, col);
            assignments.entry(key).or_default().push(label.clone());
        }
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

        for batch in 0..batch_size {
            for anchor_offset in 0..anchors_per_cell {
                let channel_base = anchor_offset * entries_per_anchor;
                let anchor = anchor_pair(spec, spec.mask[anchor_offset])?;
                for row in 0..grid_height {
                    for col in 0..grid_width {
                        let slot = (batch, head.layer_index, anchor_offset, row, col);
                        if assignments.contains_key(&slot) {
                            continue;
                        }

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
                        let best_iou = labels[batch]
                            .iter()
                            .map(|label| bbox_iou(predicted_bbox, bbox_from_label(label)))
                            .fold(0.0f32, f32::max);
                        if best_iou > spec.ignore_thresh {
                            ignored_objectness += 1;
                            continue;
                        }

                        let objectness = yolo_objectness_output(
                            head.output.get(batch, channel_base + 4, row, col),
                            spec.new_coords,
                        );
                        seed.set(
                            batch,
                            channel_base + 4,
                            row,
                            col,
                            spec.obj_normalizer * (0.0 - objectness),
                        );
                    }
                }
            }
        }

        for ((batch, layer_index, anchor_offset, row, col), slot_labels) in &assignments {
            if *layer_index != head.layer_index {
                continue;
            }
            if slot_labels.is_empty() {
                continue;
            }

            let channel_base = anchor_offset * entries_per_anchor;
            let anchor = anchor_pair(spec, spec.mask[*anchor_offset])?;
            let tx_raw = head.output.get(*batch, channel_base, *row, *col);
            let ty_raw = head.output.get(*batch, channel_base + 1, *row, *col);
            let tw_raw = head.output.get(*batch, channel_base + 2, *row, *col);
            let th_raw = head.output.get(*batch, channel_base + 3, *row, *col);
            let objectness = yolo_objectness_output(
                head.output.get(*batch, channel_base + 4, *row, *col),
                spec.new_coords,
            );
            let tx_output = yolo_xy_output(tx_raw, spec.scale_x_y, spec.new_coords);
            let ty_output = yolo_xy_output(ty_raw, spec.scale_x_y, spec.new_coords);
            let predicted_bbox = decode_training_bbox(
                tx_raw,
                ty_raw,
                tw_raw,
                th_raw,
                *col,
                *row,
                grid_width,
                grid_height,
                network.net.width,
                network.net.height,
                anchor,
                spec.scale_x_y,
                spec.new_coords,
            );

            let mut acc_dx = 0.0f32;
            let mut acc_dy = 0.0f32;
            let mut acc_dw = 0.0f32;
            let mut acc_dh = 0.0f32;
            let mut class_targets = vec![false; spec.classes];

            for label in slot_labels {
                if label.class_id >= spec.classes {
                    return Err(Error::InvalidConfig(format!(
                        "label class {} exceeds YOLO classes={} for layer #{}",
                        label.class_id, spec.classes, head.layer_index
                    )));
                }
                positive_matches += 1;
                class_targets[label.class_id] = true;

                let grid_x =
                    (label.x * grid_width as f32).clamp(0.0, grid_width as f32 - 1.0e-6);
                let grid_y =
                    (label.y * grid_height as f32).clamp(0.0, grid_height as f32 - 1.0e-6);
                let tx_target = grid_x - *col as f32;
                let ty_target = grid_y - *row as f32;
                let box_scale = 2.0 - label.w * label.h;
                let truth_bbox = bbox_from_label(label);

                match spec.iou_loss {
                    YoloIouLoss::Mse => {
                        acc_dx += box_scale * (tx_target - tx_output) * spec.iou_normalizer;
                        acc_dy += box_scale * (ty_target - ty_output) * spec.iou_normalizer;

                        let tw_target = width_height_target(
                            label.w,
                            network.net.width as f32,
                            anchor.0,
                            spec.new_coords,
                        );
                        let th_target = width_height_target(
                            label.h,
                            network.net.height as f32,
                            anchor.1,
                            spec.new_coords,
                        );
                        acc_dw += box_scale * (tw_target - tw_raw) * spec.iou_normalizer;
                        acc_dh += box_scale * (th_target - th_raw) * spec.iou_normalizer;
                    }
                    _ => {
                        let mut box_grad = dx_box_iou(predicted_bbox, truth_bbox, spec.iou_loss);
                        if !spec.new_coords {
                            box_grad.dw *= tw_raw.exp();
                            box_grad.dh *= th_raw.exp();
                        }
                        acc_dx += sanitize_delta(box_grad.dx * spec.iou_normalizer);
                        acc_dy += sanitize_delta(box_grad.dy * spec.iou_normalizer);
                        acc_dw += sanitize_delta(box_grad.dw * spec.iou_normalizer);
                        acc_dh += sanitize_delta(box_grad.dh * spec.iou_normalizer);
                    }
                }
            }

            seed.set(*batch, channel_base, *row, *col, acc_dx);
            seed.set(*batch, channel_base + 1, *row, *col, acc_dy);
            seed.set(*batch, channel_base + 2, *row, *col, acc_dw);
            seed.set(*batch, channel_base + 3, *row, *col, acc_dh);
            seed.set(
                *batch,
                channel_base + 4,
                *row,
                *col,
                spec.obj_normalizer * (1.0 - objectness),
            );

            for class_id in 0..spec.classes {
                let class_output = yolo_objectness_output(
                    head.output.get(*batch, channel_base + 5 + class_id, *row, *col),
                    spec.new_coords,
                );
                let target = if class_targets[class_id] { 1.0 } else { 0.0 };
                seed.set(
                    *batch,
                    channel_base + 5 + class_id,
                    *row,
                    *col,
                    spec.cls_normalizer * (target - class_output),
                );
            }
        }
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

pub fn load_yolo_labels(
    path: impl AsRef<Path>,
    classes: Option<usize>,
) -> Result<Vec<YoloLabel>> {
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
        if let Some(classes) = classes && class_id >= classes {
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
    options.get(key).map(|value| resolve_path(parent, Path::new(value)))
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

fn width_height_target(label_extent: f32, net_extent: f32, anchor_extent: f32, new_coords: bool) -> f32 {
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

    let dc_wrt_t = if pred_t < truth_tblr.top { -giou_cw } else { 0.0 };
    let dc_wrt_b = if pred_b > truth_tblr.bot { giou_cw } else { 0.0 };
    let dc_wrt_l = if pred_l < truth_tblr.left { -giou_ch } else { 0.0 };
    let dc_wrt_r = if pred_r > truth_tblr.right { giou_ch } else { 0.0 };

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
    p_dt = if pred_tblr.top < pred_tblr.bot { orig_p_dt } else { orig_p_db };
    p_db = if pred_tblr.top < pred_tblr.bot { orig_p_db } else { orig_p_dt };
    p_dl = if pred_tblr.left < pred_tblr.right { orig_p_dl } else { orig_p_dr };
    p_dr = if pred_tblr.left < pred_tblr.right { orig_p_dr } else { orig_p_dl };

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
        let iou = if union > 0.0 { intersection / union } else { 0.0 };
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

fn yolo_objectness_output(raw: f32, new_coords: bool) -> f32 {
    if new_coords {
        raw
    } else {
        sigmoid(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DataConfig, YoloLabel, compute_yolo_training_loss, load_training_batch, load_yolo_labels,
        replace_image_to_label,
    };
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::ExecutableNetwork;
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
        assert_eq!(parsed.train.as_deref(), Some(dir.join("train.txt").as_path()));
        assert_eq!(parsed.valid.as_deref(), Some(dir.join("valid.txt").as_path()));
        assert_eq!(parsed.names.as_deref(), Some(dir.join("labels.names").as_path()));
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

        fs::write(
            &image_path,
            b"P6\n1 1\n255\n\xff\x80\x00",
        )
        .expect("ppm writes");
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
}
