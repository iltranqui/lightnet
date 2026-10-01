//! Detector validation: one inference per validation image and VOC-style AP.
use crate::{
    DataConfig, Detection, DetectionBox, DetectionOptions, Error, LayerKind, Network,
    PlanarRgbImage, Result, Tensor, YoloHeadOutput, YoloLabel, decode_yolo_detections,
    load_paths_file, load_yolo_labels, replace_image_to_label,
};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct MapReport {
    /// AP in [0,1], one entry for every configured class (absent classes = 0).
    pub average_precision: Vec<f64>,
    pub ground_truth_counts: Vec<usize>,
    pub detection_counts: Vec<usize>,
    pub mean_average_precision: f64,
}

/// Preflight validation paths and labels before training begins.
/// Pixels are decoded only during evaluation, once per image per check.
pub struct ValidationDataset {
    pub classes: usize,
    pub images: Vec<(PathBuf, Vec<YoloLabel>)>,
}
impl ValidationDataset {
    pub fn from_data(network: &Network, data: &DataConfig) -> Result<Self> {
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
        if network.net.batch != 1 {
            return Err(invalid("validation requires a batch-one network"));
        }
        if self.classes == 0 || !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold)
        {
            return Err(invalid("invalid mAP class count or IoU threshold"));
        }
        if !options.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.confidence_threshold)
            || !options.nms_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.nms_threshold)
        {
            return Err(invalid("invalid validation confidence/NMS threshold"));
        }
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
            let heads = infer(&input)?;
            let mut detections = decode_yolo_detections(
                network,
                &heads,
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
    if classes == 0 || !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(invalid("invalid mAP class count or IoU threshold"));
    }
    let mut counts = vec![0; classes];
    let mut predictions = vec![Vec::new(); classes];
    for (image, (truth, detections)) in samples.iter().enumerate() {
        for label in truth {
            if label.class_id >= classes {
                return Err(invalid("validation truth class out of range"));
            }
            counts[label.class_id] += 1;
        }
        for detection in detections {
            if detection.class_id >= classes || !detection.confidence.is_finite() {
                return Err(invalid("invalid validation detection class/score"));
            }
            predictions[detection.class_id].push((image, detection));
        }
    }
    let mut report = MapReport {
        average_precision: vec![0.0; classes],
        ground_truth_counts: counts,
        detection_counts: predictions.iter().map(Vec::len).collect(),
        mean_average_precision: 0.0,
    };
    for (class, ranked) in predictions.iter_mut().enumerate() {
        ranked.sort_by(|a, b| b.1.confidence.total_cmp(&a.1.confidence));
        let mut used: Vec<Vec<bool>> = samples.iter().map(|s| vec![false; s.0.len()]).collect();
        let mut points = Vec::with_capacity(ranked.len());
        let (mut tp, mut fp) = (0usize, 0usize);
        for (image, detection) in ranked {
            let mut best = None;
            let mut best_iou = iou_threshold;
            for (index, label) in samples[*image].0.iter().enumerate() {
                if label.class_id != class {
                    continue;
                }
                let overlap = iou(
                    detection.bbox,
                    DetectionBox {
                        x: label.x,
                        y: label.y,
                        w: label.w,
                        h: label.h,
                    },
                );
                if overlap > best_iou {
                    best = Some(index);
                    best_iou = overlap;
                }
            }
            if let Some(index) = best.filter(|i| !used[*image][*i]) {
                used[*image][index] = true;
                tp += 1;
            } else {
                fp += 1;
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
