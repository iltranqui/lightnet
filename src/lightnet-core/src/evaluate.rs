//! Detector validation: one inference per validation image and VOC-style AP.
use crate::obb_geometry::{BoxObb, box_obb_iou, box_obb_vertices, carpn_polygon_iou};
use crate::{
    DataConfig, Detection, DetectionBox, DetectionOptions, Error, ForwardResult, LayerKind,
    Network, ObbDetection, ObbDetectionGeometry, ObbLabel, PlanarRgbImage, Result, Tensor,
    YoloHeadOutput, YoloLabel, decode_network_detections, decode_obb_network_detections,
    load_obb_labels, load_paths_file, load_yolo_labels, replace_image_to_label,
};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct MapReport {
    /// AP in [0,1], one entry for every configured class (absent classes = 0).
    pub average_precision: Vec<f64>,
    pub ground_truth_counts: Vec<usize>,
    pub detection_counts: Vec<usize>,
    pub mean_average_precision: f64,
    /// Per-class TP/FP/FN counted only for detections with confidence >=
    /// [`REPORT_CONFIDENCE`] (C++ `thresh_calc_avg_iou`); AP above uses all.
    pub true_positives: Vec<usize>,
    pub false_positives: Vec<usize>,
    pub false_negatives: Vec<usize>,
    /// Mean IoU of the true positives at [`REPORT_CONFIDENCE`] (0 when none).
    pub average_iou: Vec<f64>,
}

/// Confidence at which the Darknet mAP table counts TP/FP/FN and F1.
pub const REPORT_CONFIDENCE: f32 = 0.25;

impl MapReport {
    /// F1 of one class at [`REPORT_CONFIDENCE`], 0 when undefined.
    pub fn f1(&self, class: usize) -> f64 {
        let (tp, fp) = (self.true_positives[class], self.false_positives[class]);
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if self.ground_truth_counts[class] > 0 {
            tp as f64 / self.ground_truth_counts[class] as f64
        } else {
            0.0
        };
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

/// Preflight validation paths and labels before training begins.
/// Pixels are decoded only during evaluation, once per image per check.
pub struct ValidationDataset {
    pub classes: usize,
    pub images: Vec<(PathBuf, Vec<YoloLabel>)>,
}
impl ValidationDataset {
    pub fn from_data(network: &Network, data: &DataConfig) -> Result<Self> {
        network.ensure_no_cuboid_heads("axis-aligned validation dataset")?;
        if network.net.obb
            || network
                .layers
                .iter()
                .any(|layer| matches!(layer.kind, LayerKind::ObbYolo(_)))
        {
            return Err(invalid(
                "OBB mAP is not implemented; refusing axis-aligned evaluation",
            ));
        }
        let heads: Vec<_> = network
            .layers
            .iter()
            .filter_map(|l| match &l.kind {
                LayerKind::Yolo(s) | LayerKind::GaussianYolo(s) => Some(s.classes),
                _ => None,
            })
            .collect();
        let classes = *heads
            .first()
            .ok_or_else(|| invalid("-map requires YOLO heads"))?;
        if classes == 0
            || heads.iter().any(|c| *c != classes)
            || data.classes.is_some_and(|c| c != classes)
        {
            return Err(invalid(
                "-map class counts must agree in cfg heads and data",
            ));
        }
        let valid = data
            .valid
            .as_ref()
            .ok_or_else(|| invalid("-map requires valid= in the data file"))?;
        let paths = load_paths_file(valid)?;
        if paths.is_empty() {
            return Err(invalid("-map validation list is empty"));
        }
        let mut images = Vec::with_capacity(paths.len());
        for path in paths {
            if !path.is_file() {
                return Err(invalid(&format!(
                    "missing validation image {}",
                    path.display()
                )));
            }
            let labels = load_yolo_labels(replace_image_to_label(&path), Some(classes))?;
            images.push((path, labels));
        }
        Ok(Self { classes, images })
    }

    /// `network` must be a batch-one inference graph with the CURRENT training
    /// parameters. The callback is invoked exactly once per image. No training,
    /// augmentation, checkpoint loading or parameter updates happen here.
    pub fn evaluate(
        &self,
        network: &Network,
        options: DetectionOptions,
        iou_threshold: f32,
        mut infer: impl FnMut(&Tensor) -> Result<Vec<YoloHeadOutput>>,
    ) -> Result<MapReport> {
        self.evaluate_with_forward(network, options, iou_threshold, |input| {
            Ok(ForwardResult::from_heads(infer(input)?, Vec::new()))
        })
    }

    /// Full-forward variant required when the prediction heads depend on
    /// hyperbolic-foveation parameters cached by the runtime.
    pub fn evaluate_with_forward(
        &self,
        network: &Network,
        options: DetectionOptions,
        iou_threshold: f32,
        mut infer: impl FnMut(&Tensor) -> Result<ForwardResult>,
    ) -> Result<MapReport> {
        validate_validation_options(network, self.classes, options, iou_threshold)?;
        let mut samples = Vec::with_capacity(self.images.len());
        for (path, labels) in &self.images {
            let image = PlanarRgbImage::load(path)?;
            let prepared = if options.letterbox {
                image.letterbox(network.net.width, network.net.height)?
            } else {
                // Same u8 preprocessing as detector map, not detector test.
                image.resize_linear_exact(network.net.width, network.net.height)?
            };
            let input = Tensor::from_data(prepared.shape, prepared.data)?;
            let forward = infer(&input)?;
            let mut detections = decode_network_detections(
                network,
                &forward,
                0,
                image.shape.width,
                image.shape.height,
                options,
            )?;
            for detection in &mut detections {
                detection.bbox.x /= image.shape.width as f32;
                detection.bbox.w /= image.shape.width as f32;
                detection.bbox.y /= image.shape.height as f32;
                detection.bbox.h /= image.shape.height as f32;
            }
            samples.push((labels.clone(), detections));
        }
        detector_map(&samples, self.classes, iou_threshold)
    }
}

/// OBB validation paths and labels, preflighted before evaluation starts.
pub struct ObbValidationDataset {
    pub classes: usize,
    pub images: Vec<(PathBuf, Vec<ObbLabel>)>,
}

impl ObbValidationDataset {
    pub fn from_data(network: &Network, data: &DataConfig) -> Result<Self> {
        network.ensure_no_cuboid_heads("OBB validation dataset")?;
        let classes = obb_validation_classes(network)?;
        if data.classes.is_some_and(|configured| configured != classes) {
            return Err(invalid(
                "-map class counts must agree in OBB cfg heads and data",
            ));
        }
        let valid = data
            .valid
            .as_ref()
            .ok_or_else(|| invalid("-map requires valid= in the data file"))?;
        let paths = load_paths_file(valid)?;
        if paths.is_empty() {
            return Err(invalid("-map validation list is empty"));
        }

        let mut images = Vec::with_capacity(paths.len());
        for path in paths {
            if !path.is_file() {
                return Err(invalid(&format!(
                    "missing validation image {}",
                    path.display()
                )));
            }
            let labels = load_obb_labels(replace_image_to_label(&path), Some(classes))?;
            images.push((path, labels));
        }
        Ok(Self { classes, images })
    }

    /// Evaluate with current parameters and exactly one inference callback per
    /// image. Predictions are normalized before orientation-aware AP matching.
    pub fn evaluate(
        &self,
        network: &Network,
        options: DetectionOptions,
        iou_threshold: f32,
        mut infer: impl FnMut(&Tensor) -> Result<Vec<YoloHeadOutput>>,
    ) -> Result<MapReport> {
        self.evaluate_with_forward(network, options, iou_threshold, |input| {
            Ok(ForwardResult::from_heads(infer(input)?, Vec::new()))
        })
    }

    /// Full-forward variant required for `[obb_refine]` graphs. Legacy OBB
    /// graphs still use the same Stage-1 decoder through the shared network
    /// decoder, while CARPN graphs explicitly select polygon mAP even when
    /// there are no surviving proposals.
    pub fn evaluate_with_forward(
        &self,
        network: &Network,
        options: DetectionOptions,
        iou_threshold: f32,
        mut infer: impl FnMut(&Tensor) -> Result<ForwardResult>,
    ) -> Result<MapReport> {
        validate_validation_options(network, self.classes, options, iou_threshold)?;
        if obb_validation_classes(network)? != self.classes {
            return Err(invalid(
                "validation dataset class count does not match the OBB network",
            ));
        }

        let mut samples = Vec::with_capacity(self.images.len());
        for (path, labels) in &self.images {
            let image = PlanarRgbImage::load(path)?;
            let prepared = if options.letterbox {
                image.letterbox(network.net.width, network.net.height)?
            } else {
                image.resize_linear_exact(network.net.width, network.net.height)?
            };
            let input = Tensor::from_data(prepared.shape, prepared.data)?;
            let forward = infer(&input)?;
            let mut detections = decode_obb_network_detections(
                network,
                &forward,
                image.shape.width,
                image.shape.height,
                options,
            )?;
            for detection in &mut detections {
                let bbox = &mut detection.detection.bbox;
                bbox.x /= image.shape.width as f32;
                bbox.w /= image.shape.width as f32;
                bbox.y /= image.shape.height as f32;
                bbox.h /= image.shape.height as f32;
                match &mut detection.geometry {
                    ObbDetectionGeometry::FrontPoint { x, y } => {
                        *x /= image.shape.width as f32;
                        *y /= image.shape.height as f32;
                    }
                    ObbDetectionGeometry::CarpnQuad(vertices) => {
                        for point in vertices.chunks_exact_mut(2) {
                            point[0] /= image.shape.width as f32;
                            point[1] /= image.shape.height as f32;
                        }
                    }
                }
            }
            samples.push((labels.clone(), detections));
        }
        if network
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
        {
            carpn_detector_map(&samples, self.classes, iou_threshold)
        } else {
            obb_detector_map(&samples, self.classes, iou_threshold)
        }
    }
}

fn obb_validation_classes(network: &Network) -> Result<usize> {
    if !network.net.obb {
        return Err(invalid("OBB validation requires [net] obb=1"));
    }

    let mut classes = None;
    for layer in &network.layers {
        match &layer.kind {
            LayerKind::ObbYolo(spec) => {
                if classes.is_some_and(|expected| expected != spec.classes) {
                    return Err(invalid("OBB validation head class counts do not agree"));
                }
                classes = Some(spec.classes);
            }
            LayerKind::Yolo(_)
            | LayerKind::GaussianYolo(_)
            | LayerKind::Region(_)
            | LayerKind::Yolox(_)
            | LayerKind::Ppyoloe(_)
            | LayerKind::Yolonas(_)
            | LayerKind::CenterNet(_)
            | LayerKind::Detection(_) => {
                return Err(invalid(
                    "OBB validation requires an OBB-only detector graph",
                ));
            }
            _ => {}
        }
    }

    classes
        .filter(|classes| *classes > 0)
        .ok_or_else(|| invalid("OBB validation requires at least one [obb_yolo] head"))
}

fn validate_validation_options(
    network: &Network,
    classes: usize,
    options: DetectionOptions,
    iou_threshold: f32,
) -> Result<()> {
    if network.net.batch != 1 {
        return Err(invalid("validation requires a batch-one network"));
    }
    if classes == 0 || !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(invalid("invalid mAP class count or IoU threshold"));
    }
    if !options.confidence_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.confidence_threshold)
        || !options.nms_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.nms_threshold)
    {
        return Err(invalid("invalid validation confidence/NMS threshold"));
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::InvalidConfig(message.into())
}

/// Rank detections by class across all images, match the highest-IoU truth
/// (strictly above the threshold) once, and integrate the precision envelope.
/// Both truth and detection boxes must be in the same normalized coordinates.
/// Duplicate detections stay false positives even if another truth is nearby,
/// matching tools/oxide_map.py and Darknet's best-truth matching policy.
pub fn detector_map(
    samples: &[(Vec<YoloLabel>, Vec<Detection>)],
    classes: usize,
    iou_threshold: f32,
) -> Result<MapReport> {
    detector_map_by(
        samples,
        classes,
        iou_threshold,
        |label| label.class_id,
        |detection| detection.class_id,
        |detection| detection.confidence,
        |detection, label| {
            iou(
                detection.bbox,
                DetectionBox {
                    x: label.x,
                    y: label.y,
                    w: label.w,
                    h: label.h,
                },
            )
        },
    )
}

/// Rank normalized CARPN quadrilateral detections using polygon IoU.
/// Ground-truth labels must include front points so their oriented rectangle
/// can be reconstructed in the same coordinate space as the predicted vertices.
pub fn carpn_detector_map(
    samples: &[(Vec<ObbLabel>, Vec<ObbDetection>)],
    classes: usize,
    iou_threshold: f32,
) -> Result<MapReport> {
    if samples
        .iter()
        .flat_map(|sample| &sample.1)
        .any(|detection| !matches!(detection.geometry, ObbDetectionGeometry::CarpnQuad(_)))
    {
        return Err(invalid("CARPN mAP requires four-vertex CARPN detections"));
    }
    if samples
        .iter()
        .flat_map(|sample| &sample.0)
        .any(|label| !label.has_front)
    {
        return Err(invalid(
            "CARPN mAP requires labels with front-point columns",
        ));
    }

    detector_map_by(
        samples,
        classes,
        iou_threshold,
        |label| label.class_id,
        |detection| detection.detection.class_id,
        |detection| detection.detection.confidence,
        |detection, label| {
            let ObbDetectionGeometry::CarpnQuad(predicted_vertices) = detection.geometry else {
                unreachable!("CARPN detections were validated above")
            };
            carpn_polygon_iou(predicted_vertices, carpn_truth_vertices(label))
        },
    )
}

/// Rebuild the CARPN truth quad using the source BoxOBB corner order. The
/// source CARPN encoder falls back to the axis-aligned label rectangle when a
/// front vector is degenerate or non-finite; preserve that fallback here.
fn carpn_truth_vertices(label: &ObbLabel) -> [f32; 8] {
    let boxobb = BoxObb {
        bbox: DetectionBox {
            x: label.x,
            y: label.y,
            w: label.w,
            h: label.h,
        },
        front_x: label.front_x,
        front_y: label.front_y,
    };
    if let Some(vertices) = box_obb_vertices(boxobb) {
        return std::array::from_fn(|index| {
            let point = vertices[index / 2];
            point[index % 2]
        });
    }

    let half_width = label.w * 0.5;
    let half_height = label.h * 0.5;
    [
        label.x + half_width,
        label.y - half_height,
        label.x + half_width,
        label.y + half_height,
        label.x - half_width,
        label.y + half_height,
        label.x - half_width,
        label.y - half_height,
    ]
}

/// Rank normalized OBB detections and report classwise AP. Front-point geometry
/// uses OBB IoU; CARPN quads use polygon IoU. Mixed geometry is rejected. With
/// no detections, the representation cannot be inferred and the legacy route is
/// retained; use [`carpn_detector_map`] when explicit CARPN mode is required.
pub fn obb_detector_map(
    samples: &[(Vec<ObbLabel>, Vec<ObbDetection>)],
    classes: usize,
    iou_threshold: f32,
) -> Result<MapReport> {
    let (mut has_front_point, mut has_carpn) = (false, false);
    for detection in samples.iter().flat_map(|sample| &sample.1) {
        match detection.geometry {
            ObbDetectionGeometry::FrontPoint { .. } => has_front_point = true,
            ObbDetectionGeometry::CarpnQuad(_) => has_carpn = true,
        }
    }
    if has_front_point && has_carpn {
        return Err(invalid(
            "OBB mAP cannot mix CARPN and front-point detections",
        ));
    }
    if has_carpn {
        return carpn_detector_map(samples, classes, iou_threshold);
    }

    detector_map_by(
        samples,
        classes,
        iou_threshold,
        |label| label.class_id,
        |detection| detection.detection.class_id,
        |detection| detection.detection.confidence,
        |detection, label| {
            let (front_x, front_y) = match detection.geometry {
                ObbDetectionGeometry::FrontPoint { x, y } => (x, y),
                ObbDetectionGeometry::CarpnQuad(_) => unreachable!("CARPN rejected above"),
            };
            let truth_front = if label.has_front {
                (label.front_x, label.front_y)
            } else {
                (label.x, label.y + label.h * 0.5)
            };
            box_obb_iou(
                BoxObb {
                    bbox: detection.detection.bbox,
                    front_x,
                    front_y,
                },
                BoxObb {
                    bbox: DetectionBox {
                        x: label.x,
                        y: label.y,
                        w: label.w,
                        h: label.h,
                    },
                    front_x: truth_front.0,
                    front_y: truth_front.1,
                },
            )
        },
    )
}

fn detector_map_by<L, D>(
    samples: &[(Vec<L>, Vec<D>)],
    classes: usize,
    iou_threshold: f32,
    truth_class: impl Fn(&L) -> usize,
    detection_class: impl Fn(&D) -> usize,
    confidence: impl Fn(&D) -> f32,
    overlap: impl Fn(&D, &L) -> f32,
) -> Result<MapReport> {
    if classes == 0 || !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(invalid("invalid mAP class count or IoU threshold"));
    }
    let mut counts = vec![0; classes];
    let mut predictions = vec![Vec::new(); classes];
    for (image, (truth, detections)) in samples.iter().enumerate() {
        for label in truth {
            let class = truth_class(label);
            if class >= classes {
                return Err(invalid("validation truth class out of range"));
            }
            counts[class] += 1;
        }
        for detection in detections {
            let class = detection_class(detection);
            if class >= classes || !confidence(detection).is_finite() {
                return Err(invalid("invalid validation detection class/score"));
            }
            predictions[class].push((image, detection));
        }
    }
    let mut report = MapReport {
        average_precision: vec![0.0; classes],
        ground_truth_counts: counts,
        detection_counts: predictions.iter().map(Vec::len).collect(),
        mean_average_precision: 0.0,
        true_positives: vec![0; classes],
        false_positives: vec![0; classes],
        false_negatives: vec![0; classes],
        average_iou: vec![0.0; classes],
    };
    for (class, ranked) in predictions.iter_mut().enumerate() {
        ranked.sort_by(|a, b| confidence(b.1).total_cmp(&confidence(a.1)));
        let mut used: Vec<Vec<bool>> = samples
            .iter()
            .map(|sample| vec![false; sample.0.len()])
            .collect();
        let mut points = Vec::with_capacity(ranked.len());
        let (mut tp, mut fp) = (0usize, 0usize);
        let (mut report_tp, mut report_fp, mut iou_sum) = (0usize, 0usize, 0.0f64);
        for (image, detection) in ranked {
            let mut best = None;
            let mut best_iou = iou_threshold;
            for (index, label) in samples[*image].0.iter().enumerate() {
                if truth_class(label) != class {
                    continue;
                }
                let candidate_iou = overlap(detection, label);
                if candidate_iou > best_iou {
                    best = Some(index);
                    best_iou = candidate_iou;
                }
            }
            let counted = confidence(detection) >= REPORT_CONFIDENCE;
            if let Some(index) = best.filter(|index| !used[*image][*index]) {
                used[*image][index] = true;
                tp += 1;
                if counted {
                    report_tp += 1;
                    iou_sum += best_iou as f64;
                }
            } else {
                fp += 1;
                report_fp += counted as usize;
            }
            let recall = tp as f64 / report.ground_truth_counts[class].max(1) as f64;
            points.push((recall, tp as f64 / (tp + fp) as f64));
        }
        let mut envelope = 0.0f64;
        let mut area = 0.0;
        for index in (0..points.len()).rev() {
            envelope = envelope.max(points[index].1);
            let prev_recall = if index == 0 { 0.0 } else { points[index - 1].0 };
            area += (points[index].0 - prev_recall) * envelope;
        }
        report.average_precision[class] = area;
        report.true_positives[class] = report_tp;
        report.false_positives[class] = report_fp;
        report.false_negatives[class] = report.ground_truth_counts[class].saturating_sub(report_tp);
        report.average_iou[class] = if report_tp > 0 {
            iou_sum / report_tp as f64
        } else {
            0.0
        };
    }
    report.mean_average_precision = report.average_precision.iter().sum::<f64>() / classes as f64;
    Ok(report)
}
fn iou(a: DetectionBox, b: DetectionBox) -> f32 {
    let intersection = (a.right().min(b.right()) - a.left().max(b.left())).max(0.0)
        * (a.bottom().min(b.bottom()) - a.top().max(b.top())).max(0.0);
    let union = a.w.max(0.0) * a.h.max(0.0) + b.w.max(0.0) * b.h.max(0.0) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn truth(class_id: usize) -> YoloLabel {
        YoloLabel {
            class_id,
            x: 0.5,
            y: 0.5,
            w: 0.4,
            h: 0.4,
        }
    }
    fn prediction(class_id: usize, confidence: f32, x: f32) -> Detection {
        Detection {
            class_id,
            confidence,
            objectness: confidence,
            bbox: DetectionBox {
                x,
                y: 0.5,
                w: 0.4,
                h: 0.4,
            },
        }
    }
    fn obb_truth(class_id: usize, front: Option<[f32; 2]>) -> ObbLabel {
        ObbLabel {
            class_id,
            x: 0.5,
            y: 0.5,
            w: 0.6,
            h: 0.1,
            left: 0.2,
            right: 0.8,
            top: 0.45,
            bottom: 0.55,
            front_x: front.map_or(0.5, |point| point[0]),
            front_y: front.map_or(0.5, |point| point[1]),
            has_front: front.is_some(),
        }
    }
    fn obb_prediction(class_id: usize, confidence: f32, front: [f32; 2]) -> ObbDetection {
        ObbDetection {
            detection: Detection {
                class_id,
                confidence,
                objectness: confidence,
                bbox: DetectionBox {
                    x: 0.5,
                    y: 0.5,
                    w: 0.6,
                    h: 0.1,
                },
            },
            geometry: ObbDetectionGeometry::FrontPoint {
                x: front[0],
                y: front[1],
            },
        }
    }

    fn obb_validation_network(classes: usize, batch: usize) -> Network {
        let cfg = crate::cfg::ConfigFile::parse_str(
            "obb-validation.cfg",
            &format!(
                "[net]\nwidth=2\nheight=2\nchannels=3\nbatch={batch}\nobb=1\n[convolutional]\nfilters={}\nsize=1\nstride=1\nactivation=linear\n[obb_yolo]\nclasses={classes}\nnum=1\nanchors=1,1\n",
                classes + 7
            ),
        )
        .expect("OBB validation cfg parses");
        Network::from_config(cfg).expect("OBB validation graph builds")
    }

    fn carpn_validation_fixture(tag: &str) -> (PathBuf, Network, ObbValidationDataset) {
        let dir = std::env::temp_dir().join(format!(
            "darknet-carpn-validation-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("sample.ppm"),
            b"P6\n2 1\n255\n\x80\x40\x20\x80\x40\x20",
        )
        .expect("validation image writes");
        std::fs::write(
            dir.join("sample.txt"),
            "0 0.25 0.5 0.15811388 0.15811388 0.15 0.4\n",
        )
        .expect("CARPN front-point label writes");
        std::fs::write(dir.join("valid.txt"), "sample.ppm\n").expect("valid list writes");
        std::fs::write(dir.join("sample.data"), "classes=1\nvalid=valid.txt\n")
            .expect("data cfg writes");
        let cfg = crate::cfg::ConfigFile::parse_str(
            "carpn-validation.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=3
                batch=1
                obb=1
                [convolutional]
                filters=2
                size=1
                batch_normalize=0
                activation=linear
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                activation=linear
                [obb_yolo]
                classes=1
                num=1
                anchors=0.2,0.2
                new_coords=1
                max=1
                [obb_refine]
                from=0
                proposal_from=2
                proposal_topk_train=1
                proposal_topk_infer=1
                proposal_capacity=1
                sample_grid=1
                front_range=1
            "#,
        )
        .expect("CARPN cfg parses");
        let network = Network::from_config(cfg)
            .expect("CARPN validation graph builds")
            .with_batch(1);
        let data = DataConfig::from_path(dir.join("sample.data")).expect("data cfg parses");
        let dataset = ObbValidationDataset::from_data(&network, &data)
            .expect("CARPN validation data preflights");
        (dir, network, dataset)
    }

    fn carpn_validation_executable(
        network: Network,
        objectness: f32,
        refine_biases: [f32; 5],
    ) -> crate::runtime::ExecutableNetwork {
        use crate::runtime::{ExecutableNetwork, LayerParams};

        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("CARPN feature convolution");
        };
        feature.weights.fill(0.0);
        feature.biases.fill(0.0);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("CARPN proposal convolution");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, objectness, 0.8]);
        let LayerParams::Connected(refine) = &mut executable.params[3] else {
            panic!("CARPN refine parameters");
        };
        refine.biases.copy_from_slice(&refine_biases);
        refine.weights.fill(0.0);
        executable
    }

    #[test]
    fn obb_validation_dataset_runs_one_decode_per_image_and_normalizes_predictions() {
        let dir =
            std::env::temp_dir().join(format!("darknet-obb-validation-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        for name in ["a", "b"] {
            std::fs::write(
                dir.join(format!("{name}.ppm")),
                b"P6\n2 2\n255\n\x80\x40\x20\x80\x40\x20\x80\x40\x20\x80\x40\x20",
            )
            .expect("validation image writes");
            std::fs::write(
                dir.join(format!("{name}.txt")),
                "0 0.25 0.25 0.5 0.5 0.25 0.375\n",
            )
            .expect("OBB validation label writes");
        }
        std::fs::write(dir.join("valid.txt"), "a.ppm\nb.ppm\n").expect("validation list writes");
        std::fs::write(dir.join("sample.data"), "classes=1\nvalid=valid.txt\n")
            .expect("data cfg writes");

        let network = obb_validation_network(1, 1);
        let data = DataConfig::from_path(dir.join("sample.data")).expect("data cfg parses");
        let dataset = ObbValidationDataset::from_data(&network, &data)
            .expect("OBB validation dataset preflights");
        assert_eq!(dataset.classes, 1);
        assert_eq!(dataset.images.len(), 2);
        assert!(dataset.images[0].1[0].has_front);

        let mut calls = 0;
        let report = dataset
            .evaluate(
                &network,
                DetectionOptions {
                    confidence_threshold: 0.3,
                    nms_threshold: 0.45,
                    letterbox: false,
                },
                0.5,
                |input| {
                    calls += 1;
                    assert_eq!(input.shape, crate::TensorShape::new(2, 2, 3, 1));
                    let stride = 4;
                    let mut output = vec![0.0; 8 * stride];
                    output[4 * stride] = 0.5;
                    output[5 * stride] = 0.75;
                    output[6 * stride] = 2.0;
                    output[7 * stride] = 2.0;
                    Ok(vec![YoloHeadOutput {
                        layer_index: 1,
                        output: Tensor::from_data(crate::TensorShape::new(2, 2, 8, 1), output)?,
                    }])
                },
            )
            .expect("OBB validation evaluates");
        assert_eq!(calls, 2);
        assert_eq!(report.mean_average_precision, 1.0);
        assert_eq!(report.ground_truth_counts, vec![2]);
        assert_eq!(report.detection_counts, vec![2]);
        std::fs::remove_dir_all(dir).expect("temporary validation data is removed");
    }

    #[test]
    fn carpn_map_network_validation_executes_full_network_and_scores_polygon_ap() {
        let (dir, network, dataset) = carpn_validation_fixture("polygon-ap");
        let executable = carpn_validation_executable(network, 0.9, [0.0; 5]);
        let mut calls = 0;
        let report = dataset
            .evaluate_with_forward(
                &executable.network,
                DetectionOptions {
                    confidence_threshold: 0.1,
                    nms_threshold: 1.0,
                    letterbox: false,
                },
                0.5,
                |input| {
                    calls += 1;
                    executable.forward(input)
                },
            )
            .expect("full CARPN validation pass");
        assert_eq!(calls, 1);
        assert_eq!(report.average_precision, vec![1.0]);
        assert_eq!(report.ground_truth_counts, vec![1]);
        assert_eq!(report.detection_counts, vec![1]);
        assert_eq!(report.true_positives, vec![1]);
        assert_eq!(report.false_positives, vec![0]);
        assert_eq!(report.false_negatives, vec![0]);
        assert!((report.average_iou[0] - 1.0).abs() < 1e-5);
        std::fs::remove_dir_all(dir).expect("temporary validation data is removed");
    }

    #[test]
    fn carpn_map_network_validation_uses_polygon_iou_when_aabb_would_match() {
        let (dir, network, dataset) = carpn_validation_fixture("polygon-not-aabb");
        let diagonal_square_raw = [
            0.0,
            0.0,
            2.0f32.sqrt().ln(),
            1.0 / 2.0f32.sqrt(),
            1.0 / 2.0f32.sqrt(),
        ];
        let executable = carpn_validation_executable(network, 0.9, diagonal_square_raw);
        let image = PlanarRgbImage::load(&dataset.images[0].0).expect("image");
        let prepared = image
            .resize_linear_exact(executable.network.net.width, executable.network.net.height)
            .expect("preprocess");
        let input = Tensor::from_data(prepared.shape, prepared.data).expect("input tensor");
        let forward = executable.forward(&input).expect("network forward");
        let mut detection = decode_obb_network_detections(
            &executable.network,
            &forward,
            image.shape.width,
            image.shape.height,
            DetectionOptions {
                confidence_threshold: 0.1,
                nms_threshold: 1.0,
                letterbox: false,
            },
        )
        .expect("CARPN network decode")
        .pop()
        .expect("one proposal");
        let truth = &dataset.images[0].1[0];
        let truth_vertices = carpn_truth_vertices(truth);
        let mut min_x = truth_vertices[0];
        let mut max_x = truth_vertices[0];
        let mut min_y = truth_vertices[1];
        let mut max_y = truth_vertices[1];
        for vertex in 1..4 {
            min_x = min_x.min(truth_vertices[vertex * 2]);
            max_x = max_x.max(truth_vertices[vertex * 2]);
            min_y = min_y.min(truth_vertices[vertex * 2 + 1]);
            max_y = max_y.max(truth_vertices[vertex * 2 + 1]);
        }
        let truth_aabb = DetectionBox {
            x: 0.5 * (min_x + max_x),
            y: 0.5 * (min_y + max_y),
            w: max_x - min_x,
            h: max_y - min_y,
        };
        detection.detection.bbox.x /= image.shape.width as f32;
        detection.detection.bbox.w /= image.shape.width as f32;
        detection.detection.bbox.y /= image.shape.height as f32;
        detection.detection.bbox.h /= image.shape.height as f32;
        let ObbDetectionGeometry::CarpnQuad(vertices) = &mut detection.geometry else {
            panic!("CARPN output geometry");
        };
        for point in vertices.chunks_exact_mut(2) {
            point[0] /= image.shape.width as f32;
            point[1] /= image.shape.height as f32;
        }
        assert!((iou(detection.detection.bbox, truth_aabb) - 1.0).abs() < 1e-5);
        let ObbDetectionGeometry::CarpnQuad(predicted_vertices) = detection.geometry else {
            unreachable!();
        };
        assert!(carpn_polygon_iou(predicted_vertices, truth_vertices) < 0.6);

        let aabb_fallback = detector_map(
            &[(
                vec![YoloLabel {
                    class_id: 0,
                    x: truth_aabb.x,
                    y: truth_aabb.y,
                    w: truth_aabb.w,
                    h: truth_aabb.h,
                }],
                vec![Detection {
                    class_id: 0,
                    confidence: detection.detection.confidence,
                    objectness: detection.detection.objectness,
                    bbox: detection.detection.bbox,
                }],
            )],
            1,
            0.6,
        )
        .expect("AABB fallback report");
        assert_eq!(aabb_fallback.average_precision, vec![1.0]);

        let mut calls = 0;
        let polygon_report = dataset
            .evaluate_with_forward(
                &executable.network,
                DetectionOptions {
                    confidence_threshold: 0.1,
                    nms_threshold: 1.0,
                    letterbox: false,
                },
                0.6,
                |input| {
                    calls += 1;
                    executable.forward(input)
                },
            )
            .expect("polygon CARPN validation");
        assert_eq!(calls, 1);
        assert_eq!(polygon_report.average_precision, vec![0.0]);
        assert_eq!(polygon_report.false_positives, vec![1]);
        assert_eq!(polygon_report.false_negatives, vec![1]);
        std::fs::remove_dir_all(dir).expect("temporary validation data is removed");
    }

    #[test]
    fn carpn_map_network_validation_requires_front_labels_even_for_zero_proposals() {
        let (dir, network, _) = carpn_validation_fixture("empty-without-front");
        std::fs::write(dir.join("sample.txt"), "0 0.25 0.5 0.15811388 0.15811388\n")
            .expect("front-less label writes");
        let data = DataConfig::from_path(dir.join("sample.data")).expect("data cfg");
        let dataset = ObbValidationDataset::from_data(&network, &data).expect("dataset");
        let executable = carpn_validation_executable(network, 0.005, [0.0; 5]);
        let mut calls = 0;
        let error = dataset
            .evaluate_with_forward(
                &executable.network,
                DetectionOptions {
                    confidence_threshold: 0.001,
                    nms_threshold: 1.0,
                    letterbox: false,
                },
                0.5,
                |input| {
                    calls += 1;
                    executable.forward(input)
                },
            )
            .expect_err("empty CARPN output still requires front labels");
        assert_eq!(calls, 1);
        assert!(error.to_string().contains("front-point"), "{error}");
        std::fs::remove_dir_all(dir).expect("temporary validation data is removed");
    }

    #[test]
    fn obb_validation_dataset_rejects_non_obb_graphs_and_class_mismatch() {
        let network = obb_validation_network(1, 1);
        let data = DataConfig {
            source: PathBuf::new(),
            classes: Some(2),
            train: None,
            valid: None,
            names: None,
            backup: None,
            options: Default::default(),
        };
        assert!(ObbValidationDataset::from_data(&network, &data).is_err());

        let standard = Network::from_config(
            crate::cfg::ConfigFile::parse_str(
                "standard-validation.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=3\n[convolutional]\nfilters=6\nsize=1\n[yolo]\nclasses=1\nnum=1\nanchors=1,1\n",
            )
            .expect("standard cfg parses"),
        )
        .expect("standard graph builds");
        assert!(ObbValidationDataset::from_data(&standard, &data).is_err());

        let hybrid = Network::from_config(
            crate::cfg::ConfigFile::parse_str(
                "hybrid-validation.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=3\nobb=1\n[yolo]\nallow_obb_truth=1\nclasses=1\nnum=1\nanchors=1,1\n",
            )
            .expect("hybrid cfg parses"),
        )
        .expect("hybrid graph builds");
        assert!(ObbValidationDataset::from_data(&hybrid, &data).is_err());
    }

    #[test]
    fn obb_map_uses_orientation_and_preserves_report_matching_rules() {
        let oriented = vec![(
            vec![obb_truth(0, Some([0.5, 0.55]))],
            vec![
                obb_prediction(0, 0.9, [0.8, 0.5]),
                obb_prediction(0, 0.8, [0.5, 0.55]),
            ],
        )];
        let report = obb_detector_map(&oriented, 1, 0.5).expect("oriented mAP");
        assert_eq!(report.average_precision, vec![0.5]);
        assert_eq!(report.true_positives, vec![1]);
        assert_eq!(report.false_positives, vec![1]);
        assert_eq!(report.false_negatives, vec![0]);
        assert_eq!(report.detection_counts, vec![2]);
        assert!((report.average_iou[0] - 1.0).abs() < 1e-6);

        let axis_aligned = vec![(
            vec![obb_truth(0, None)],
            vec![
                obb_prediction(0, 0.9, [0.5, 0.55]),
                obb_prediction(0, 0.8, [0.5, 0.55]),
            ],
        )];
        let report = obb_detector_map(&axis_aligned, 1, 0.5).expect("AABB fallback mAP");
        assert_eq!(report.average_precision, vec![1.0]);
        assert_eq!(report.true_positives, vec![1]);
        assert_eq!(report.false_positives, vec![1]);
        assert_eq!(report.false_negatives, vec![0]);

        let low_confidence = vec![(
            vec![obb_truth(0, None)],
            vec![obb_prediction(0, 0.1, [0.5, 0.55])],
        )];
        let report = obb_detector_map(&low_confidence, 1, 0.5).expect("low-confidence mAP");
        assert_eq!(report.average_precision, vec![1.0]);
        assert_eq!(report.true_positives, vec![0]);
        assert_eq!(report.false_positives, vec![0]);
        assert_eq!(report.false_negatives, vec![1]);
        assert_eq!(report.average_iou, vec![0.0]);
    }

    #[test]
    fn obb_map_rejects_carpn_without_front_and_invalid_classes_or_thresholds() {
        let mut carpn = obb_prediction(0, 0.9, [0.5, 0.55]);
        carpn.geometry = ObbDetectionGeometry::CarpnQuad([0.0; 8]);
        assert!(obb_detector_map(&[(vec![obb_truth(0, None)], vec![carpn])], 1, 0.5).is_err());

        let mut bad_truth = obb_truth(0, Some([0.5, 0.55]));
        bad_truth.class_id = 1;
        assert!(obb_detector_map(&[(vec![bad_truth], vec![])], 1, 0.5).is_err());
        assert!(obb_detector_map(&[], 1, f32::NAN).is_err());
    }

    fn carpn_prediction(class_id: usize, confidence: f32, vertices: [f32; 8]) -> ObbDetection {
        ObbDetection {
            detection: Detection {
                class_id,
                confidence,
                objectness: confidence,
                bbox: DetectionBox {
                    x: 0.5,
                    y: 0.5,
                    w: 0.6,
                    h: 0.1,
                },
            },
            geometry: ObbDetectionGeometry::CarpnQuad(vertices),
        }
    }

    #[test]
    fn carpn_map_matches_polygon_quads_and_reuses_report_matching_rules() {
        let label = obb_truth(0, Some([0.5, 0.55]));
        let vertices = carpn_truth_vertices(&label);
        let expected_vertices = [0.2, 0.55, 0.8, 0.55, 0.8, 0.45, 0.2, 0.45];
        for (actual, expected) in vertices.into_iter().zip(expected_vertices) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let degenerate = obb_truth(0, Some([0.5, 0.5]));
        let fallback = carpn_truth_vertices(&degenerate);
        let expected_fallback = [0.8, 0.45, 0.8, 0.55, 0.2, 0.55, 0.2, 0.45];
        for (actual, expected) in fallback.into_iter().zip(expected_fallback) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let report = carpn_detector_map(
            &[(
                vec![label],
                vec![
                    carpn_prediction(0, 0.9, vertices),
                    carpn_prediction(0, 0.8, vertices),
                ],
            )],
            1,
            0.5,
        )
        .expect("CARPN polygon mAP");
        assert_eq!(report.average_precision, vec![1.0]);
        assert_eq!(report.true_positives, vec![1]);
        assert_eq!(report.false_positives, vec![1]);
        assert_eq!(report.false_negatives, vec![0]);
        assert_eq!(report.detection_counts, vec![2]);
        assert!((report.average_iou[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn obb_map_dispatches_carpn_samples_to_polygon_map() {
        let label = obb_truth(0, Some([0.5, 0.55]));
        let vertices = carpn_truth_vertices(&label);
        let rotated_label = obb_truth(0, Some([0.55, 0.55]));
        let rotated_vertices = carpn_truth_vertices(&rotated_label);
        let min_x = rotated_vertices
            .chunks_exact(2)
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min);
        let max_x = rotated_vertices
            .chunks_exact(2)
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = rotated_vertices
            .chunks_exact(2)
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = rotated_vertices
            .chunks_exact(2)
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let rotated_aabb = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y];
        assert!(carpn_polygon_iou(rotated_aabb, rotated_vertices) < 0.5);
        let samples = vec![
            (
                vec![label],
                vec![
                    carpn_prediction(0, 0.9, vertices),
                    carpn_prediction(0, 0.8, vertices),
                ],
            ),
            (
                vec![rotated_label],
                vec![carpn_prediction(0, 0.7, rotated_aabb)],
            ),
        ];
        let direct = carpn_detector_map(&samples, 1, 0.5).expect("direct CARPN mAP");
        let dispatched = obb_detector_map(&samples, 1, 0.5).expect("dispatched CARPN mAP");

        assert_eq!(dispatched.average_precision, direct.average_precision);
        assert_eq!(dispatched.ground_truth_counts, direct.ground_truth_counts);
        assert_eq!(dispatched.detection_counts, direct.detection_counts);
        assert_eq!(
            dispatched.mean_average_precision,
            direct.mean_average_precision
        );
        assert_eq!(dispatched.true_positives, direct.true_positives);
        assert_eq!(dispatched.false_positives, direct.false_positives);
        assert_eq!(dispatched.false_negatives, direct.false_negatives);
        assert_eq!(dispatched.average_iou, direct.average_iou);
        assert_eq!(dispatched.true_positives, vec![1]);
        assert_eq!(dispatched.false_positives, vec![2]);
        assert_eq!(dispatched.false_negatives, vec![1]);
    }

    #[test]
    fn obb_map_rejects_mixed_front_point_and_carpn_geometry() {
        let label = obb_truth(0, Some([0.5, 0.55]));
        let vertices = carpn_truth_vertices(&label);
        let carpn = carpn_prediction(0, 0.9, vertices);
        let front_point = obb_prediction(0, 0.8, [0.5, 0.55]);

        let mixed_within_image = obb_detector_map(
            &[(vec![label], vec![carpn.clone(), front_point.clone()])],
            1,
            0.5,
        )
        .expect_err("mixed geometry in one image must fail");
        assert!(mixed_within_image.to_string().contains("cannot mix"));

        let mixed_across_images = obb_detector_map(
            &[
                (vec![obb_truth(0, Some([0.5, 0.55]))], vec![carpn]),
                (vec![obb_truth(0, Some([0.5, 0.55]))], vec![front_point]),
            ],
            1,
            0.5,
        )
        .expect_err("mixed geometry across images must fail");
        assert!(mixed_across_images.to_string().contains("cannot mix"));
    }

    #[test]
    fn obb_map_empty_detections_keep_legacy_mode_but_carpn_entry_is_explicit() {
        let with_front = obb_truth(0, Some([0.5, 0.55]));
        let legacy = obb_detector_map(&[(vec![with_front], vec![])], 1, 0.5)
            .expect("empty predictions use legacy mode");
        let explicit_carpn = carpn_detector_map(&[(vec![with_front], vec![])], 1, 0.5)
            .expect("empty predictions can explicitly use CARPN mode");
        assert_eq!(legacy.false_negatives, vec![1]);
        assert_eq!(legacy.average_precision, explicit_carpn.average_precision);
        assert!(carpn_detector_map(&[(vec![obb_truth(0, None)], vec![])], 1, 0.5).is_err());

        let legacy_without_front = obb_detector_map(&[(vec![obb_truth(0, None)], vec![])], 1, 0.5)
            .expect("legacy empty predictions retain five-field label behavior");
        assert_eq!(legacy_without_front.false_negatives, vec![1]);
    }

    #[test]
    fn carpn_map_uses_polygon_iou_for_rotated_truths() {
        let label = obb_truth(0, Some([0.55, 0.55]));
        let truth_vertices = carpn_truth_vertices(&label);
        let min_x = truth_vertices
            .chunks_exact(2)
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min);
        let max_x = truth_vertices
            .chunks_exact(2)
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = truth_vertices
            .chunks_exact(2)
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = truth_vertices
            .chunks_exact(2)
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let aabb = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y];
        assert!(carpn_polygon_iou(aabb, truth_vertices) < 0.5);

        let report = carpn_detector_map(
            &[(vec![label], vec![carpn_prediction(0, 0.9, aabb)])],
            1,
            0.5,
        )
        .expect("polygon matching rejects the AABB-only overlap");
        assert_eq!(report.average_precision, vec![0.0]);
        assert_eq!(report.true_positives, vec![0]);
        assert_eq!(report.false_positives, vec![1]);
        assert_eq!(report.false_negatives, vec![1]);
    }

    #[test]
    fn carpn_map_requires_quads_and_front_point_labels() {
        let label = obb_truth(0, Some([0.5, 0.55]));
        let wrong_geometry = obb_prediction(0, 0.9, [0.5, 0.55]);
        assert!(carpn_detector_map(&[(vec![label], vec![wrong_geometry])], 1, 0.5).is_err());

        let no_front = obb_truth(0, None);
        let prediction = carpn_prediction(0, 0.9, [0.0; 8]);
        assert!(carpn_detector_map(&[(vec![no_front], vec![prediction])], 1, 0.5).is_err());
    }

    #[test]
    fn obb_networks_are_not_silently_evaluated_with_aabb_map() {
        let network = Network::from_config(
            crate::cfg::ConfigFile::parse_str(
                "obb-map.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
            )
            .expect("cfg parses"),
        )
        .expect("network builds");
        let data = crate::DataConfig {
            source: PathBuf::new(),
            classes: Some(1),
            train: None,
            valid: None,
            names: None,
            backup: None,
            options: Default::default(),
        };
        let error = ValidationDataset::from_data(&network, &data)
            .err()
            .expect("OBB mAP must not fall through to AABB evaluation");
        assert!(error.to_string().contains("OBB mAP is not implemented"));

        let hybrid = Network::from_config(
            crate::cfg::ConfigFile::parse_str(
                "obb-hybrid-map.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=6\nobb=1\n[yolo]\nallow_obb_truth=1\nclasses=1\nnum=1\nanchors=1,1\n",
            )
            .expect("cfg parses"),
        )
        .expect("hybrid network builds");
        assert!(ValidationDataset::from_data(&hybrid, &data).is_err());
    }

    #[test]
    fn ap_precision_envelope_duplicates_background_and_missing_classes() {
        // FP then TP then duplicate then another TP: AP=.5*.5+.5*.5=.5.
        let samples = vec![
            (
                vec![truth(0)],
                vec![
                    prediction(0, 0.9, 0.1),
                    prediction(0, 0.8, 0.5),
                    prediction(0, 0.7, 0.5),
                ],
            ),
            (vec![truth(0)], vec![prediction(0, 0.6, 0.5)]),
            (vec![], vec![prediction(1, 0.9, 0.5)]),
        ];
        let r = detector_map(&samples, 2, 0.5).unwrap();
        assert_eq!(r.average_precision, vec![0.5, 0.0]);
        assert_eq!(r.mean_average_precision, 0.25);
        assert_eq!(r.ground_truth_counts, vec![2, 0]);
        // At conf >= 0.25: class 0 has 2 TP (the 0.8 and 0.6 hits), FP = the
        // 0.9 miss + the 0.7 duplicate; class 1 has one FP and no truth.
        assert_eq!(r.true_positives, vec![2, 0]);
        assert_eq!(r.false_positives, vec![2, 1]);
        assert_eq!(r.false_negatives, vec![0, 0]);
        assert!((r.f1(0) - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(r.detection_counts, vec![4, 1]);
    }
    #[test]
    fn precision_envelope_integrates_all_recall_steps() {
        let samples = vec![
            (
                vec![truth(0)],
                vec![prediction(0, 0.9, 0.5), prediction(0, 0.8, 0.1)],
            ),
            (vec![truth(0)], vec![prediction(0, 0.7, 0.5)]),
        ];
        let report = detector_map(&samples, 1, 0.5).unwrap();
        assert!((report.mean_average_precision - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn validates_all_images_once_with_current_parameter_snapshot() {
        use crate::{ConfigFile, ExecutableNetwork, LayerParams};
        let dir = std::env::temp_dir().join(format!("darknet-map-once-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["a", "b"] {
            std::fs::write(
                dir.join(format!("{name}.ppm")),
                b"P6\n1 1\n255\n\xff\x80\x00",
            )
            .unwrap();
            std::fs::write(dir.join(format!("{name}.txt")), "0 0.5 0.5 0.5 0.5\n").unwrap();
        }
        std::fs::write(dir.join("valid.txt"), "a.ppm\nb.ppm\n").unwrap();
        std::fs::write(dir.join("sample.data"), "classes=1\nvalid=valid.txt\n").unwrap();
        let cfg = ConfigFile::parse_str("map.cfg","[net]\nbatch=2\nwidth=8\nheight=8\nchannels=3\n[convolutional]\nfilters=6\nsize=1\nactivation=linear\n[avgpool]\n[yolo]\nmask=0\nanchors=4,4\nclasses=1\nnum=1\n").unwrap();
        let mut training = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
        if let LayerParams::Convolutional(p) = &mut training.params[0] {
            p.biases[4] = 10.0;
            p.biases[5] = 10.0;
        }
        let before = training.params.clone();
        let mut inference = ExecutableNetwork::new(training.network.with_batch(1));
        inference.params = training.params.clone();
        let data = DataConfig::from_path(dir.join("sample.data")).unwrap();
        let dataset = ValidationDataset::from_data(&training.network, &data).unwrap();
        let mut calls = 0;
        let report = dataset
            .evaluate(
                &inference.network,
                DetectionOptions {
                    confidence_threshold: 0.005,
                    nms_threshold: 0.45,
                    letterbox: false,
                },
                0.5,
                |input| {
                    calls += 1;
                    Ok(inference.forward(input)?.yolo_heads)
                },
            )
            .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(report.mean_average_precision, 1.0);
        assert_eq!(report.ground_truth_counts, vec![2]);
        assert_eq!(report.detection_counts, vec![2]);
        assert_eq!(training.params, before);
        assert_eq!(training.network.net.batch, 2);
        assert_eq!(inference.params, before);
        std::fs::write(dir.join("valid.txt"), "").unwrap();
        assert!(ValidationDataset::from_data(&training.network, &data).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_predictions_and_iou_threshold_are_respected() {
        let samples = vec![
            (vec![truth(0)], vec![prediction(0, 0.9, 0.5)]),
            (vec![truth(0)], vec![]),
        ];
        assert_eq!(
            detector_map(&samples, 1, 0.5)
                .unwrap()
                .mean_average_precision,
            0.5
        );
        assert_eq!(
            detector_map(&samples, 1, 1.0)
                .unwrap()
                .mean_average_precision,
            0.0
        );
        assert!(detector_map(&samples, 1, f32::NAN).is_err());
    }
}
