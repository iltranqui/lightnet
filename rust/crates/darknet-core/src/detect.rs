//! YOLO output decoding and CPU non-maximum suppression.
//!
//! The runtime leaves YOLO heads in Darknet's flat channel-major layout.  This
//! module converts those tensors to image-space detections and keeps
//! post-processing separate from neural-network execution, like the C++
//! `get_yolo_detections()` and NMS helpers.

use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, YoloSpec};
use crate::runtime::YoloHeadOutput;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl DetectionBox {
    pub fn left(self) -> f32 {
        self.x - self.w * 0.5
    }

    pub fn top(self) -> f32 {
        self.y - self.h * 0.5
    }

    pub fn right(self) -> f32 {
        self.x + self.w * 0.5
    }

    pub fn bottom(self) -> f32 {
        self.y + self.h * 0.5
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub objectness: f32,
    pub bbox: DetectionBox,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionOptions {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub letterbox: bool,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            nms_threshold: 0.45,
            letterbox: true,
        }
    }
}

pub fn decode_yolo_detections(
    network: &Network,
    yolo_heads: &[YoloHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    for head in yolo_heads {
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
        decode_yolo_head(
            &head.output.data,
            head.output.shape.width,
            head.output.shape.height,
            spec,
            network.net.width,
            network.net.height,
            image_width,
            image_height,
            options,
            &mut detections,
        )?;
    }

    apply_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| rhs.confidence.total_cmp(&lhs.confidence));
    Ok(detections)
}

#[allow(clippy::too_many_arguments)]
fn decode_yolo_head(
    data: &[f32],
    grid_width: usize,
    grid_height: usize,
    spec: &YoloSpec,
    net_width: usize,
    net_height: usize,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
    out: &mut Vec<Detection>,
) -> Result<()> {
    let anchors_per_cell = spec.mask.len();
    let entries_per_anchor = spec.classes + 5;
    let expected = grid_width
        .checked_mul(grid_height)
        .and_then(|v| v.checked_mul(anchors_per_cell))
        .and_then(|v| v.checked_mul(entries_per_anchor))
        .ok_or_else(|| Error::InvalidConfig("YOLO head size overflow".to_owned()))?;
    if data.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "YOLO head tensor length mismatch: expected {expected}, got {}",
            data.len()
        )));
    }

    let stride = grid_width * grid_height;
    for (anchor_offset, &mask_idx) in spec.mask.iter().enumerate() {
        let anchor = anchor_pair(spec, mask_idx)?;
        let anchor_base = anchor_offset * entries_per_anchor;
        for index in 0..stride {
            let row = index / grid_width;
            let col = index % grid_width;
            let tx = data[anchor_base * stride + index];
            let ty = data[(anchor_base + 1) * stride + index];
            let tw = data[(anchor_base + 2) * stride + index];
            let th = data[(anchor_base + 3) * stride + index];
            let objectness = sigmoid(data[(anchor_base + 4) * stride + index]);

            if objectness <= options.confidence_threshold {
                continue;
            }

            let bbox = decode_bbox(
                tx,
                ty,
                tw,
                th,
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
            let bbox = correct_bbox(
                bbox,
                image_width as f32,
                image_height as f32,
                net_width as f32,
                net_height as f32,
                options.letterbox,
            );

            for class_id in 0..spec.classes {
                let class_prob = sigmoid(data[(anchor_base + 5 + class_id) * stride + index]);
                let confidence = objectness * class_prob;
                if confidence > options.confidence_threshold {
                    out.push(Detection {
                        class_id,
                        confidence,
                        objectness,
                        bbox,
                    });
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_bbox(
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
) -> DetectionBox {
    let bx = (col as f32 + scale_x_y * sigmoid(tx) - 0.5 * (scale_x_y - 1.0)) / grid_width as f32;
    let by = (row as f32 + scale_x_y * sigmoid(ty) - 0.5 * (scale_x_y - 1.0)) / grid_height as f32;
    let (bw, bh) = if new_coords {
        (
            tw * tw * 4.0 * anchor.0 / net_width as f32,
            th * th * 4.0 * anchor.1 / net_height as f32,
        )
    } else {
        (
            tw.exp() * anchor.0 / net_width as f32,
            th.exp() * anchor.1 / net_height as f32,
        )
    };

    DetectionBox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    }
}

fn correct_bbox(
    mut bbox: DetectionBox,
    image_width: f32,
    image_height: f32,
    net_width: f32,
    net_height: f32,
    letterbox: bool,
) -> DetectionBox {
    let (new_w, new_h) = if letterbox {
        if net_width / image_width < net_height / image_height {
            (net_width, image_height * net_width / image_width)
        } else {
            (image_width * net_height / image_height, net_height)
        }
    } else {
        (net_width, net_height)
    };

    let delta_w = net_width - new_w;
    let delta_h = net_height - new_h;
    let ratio_w = new_w / net_width;
    let ratio_h = new_h / net_height;

    bbox.x = (bbox.x - delta_w / 2.0 / net_width) / ratio_w;
    bbox.y = (bbox.y - delta_h / 2.0 / net_height) / ratio_h;
    bbox.w /= ratio_w;
    bbox.h /= ratio_h;

    bbox.x *= image_width;
    bbox.y *= image_height;
    bbox.w *= image_width;
    bbox.h *= image_height;
    bbox
}

fn apply_nms(detections: &mut Vec<Detection>, threshold: f32) {
    if threshold <= 0.0 {
        return;
    }

    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if suppressed[j] || detections[i].class_id != detections[j].class_id {
                continue;
            }
            let (keep_idx, drop_idx) = if detections[i].confidence >= detections[j].confidence {
                (i, j)
            } else {
                (j, i)
            };
            if bbox_iou(detections[keep_idx].bbox, detections[drop_idx].bbox) > threshold {
                suppressed[drop_idx] = true;
            }
        }
    }

    let mut keep = 0usize;
    for (idx, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if keep != idx {
                detections.swap(keep, idx);
            }
            keep += 1;
        }
    }
    detections.truncate(keep);
}

fn bbox_iou(lhs: DetectionBox, rhs: DetectionBox) -> f32 {
    let inter_left = lhs.left().max(rhs.left());
    let inter_top = lhs.top().max(rhs.top());
    let inter_right = lhs.right().min(rhs.right());
    let inter_bottom = lhs.bottom().min(rhs.bottom());

    let inter_w = (inter_right - inter_left).max(0.0);
    let inter_h = (inter_bottom - inter_top).max(0.0);
    let inter_area = inter_w * inter_h;
    if inter_area <= 0.0 {
        return 0.0;
    }

    let lhs_area = lhs.w.max(0.0) * lhs.h.max(0.0);
    let rhs_area = rhs.w.max(0.0) * rhs.h.max(0.0);
    inter_area / (lhs_area + rhs_area - inter_area)
}

fn anchor_pair(spec: &YoloSpec, index: usize) -> Result<(f32, f32)> {
    let anchor_index = index
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidConfig("anchor index overflow".to_owned()))?;
    let width = *spec.anchors.get(anchor_index).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor width for YOLO mask index {index}"))
    })?;
    let height = *spec.anchors.get(anchor_index + 1).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor height for YOLO mask index {index}"))
    })?;
    Ok((width, height))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{Tensor, YoloHeadOutput};

    use super::{DetectionOptions, decode_yolo_detections};

    #[test]
    fn decodes_single_yolo_detection() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=32
                height=32
                channels=6

                [maxpool]
                size=32
                stride=32

                [yolo]
                mask=0
                anchors=10,12
                classes=1
                num=1
                scale_x_y=1.0
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(
                network.layers[1].output_shape,
                vec![0.0, 0.0, 0.0, 0.0, 8.0, 8.0],
            )
            .expect("tensor"),
        };

        let detections = decode_yolo_detections(
            &network,
            &[head],
            32,
            32,
            DetectionOptions {
                confidence_threshold: 0.25,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .expect("decode should succeed");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 0);
        assert!(detections[0].confidence > 0.99);
        assert!((detections[0].bbox.x - 16.0).abs() < 0.01);
        assert!((detections[0].bbox.w - 10.0).abs() < 0.01);
    }

    #[test]
    fn suppresses_overlapping_boxes_with_nms() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=32
                height=32
                channels=12

                [maxpool]
                size=32
                stride=32

                [yolo]
                mask=0,1
                anchors=10,12,10,12
                classes=1
                num=2
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut data = vec![0.0; 12];
        data[4] = 8.0;
        data[5] = 8.0;
        data[10] = 7.0;
        data[11] = 7.0;
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(network.layers[1].output_shape, data).expect("tensor"),
        };

        let detections =
            decode_yolo_detections(&network, &[head], 32, 32, DetectionOptions::default())
                .expect("decode should succeed");

        assert_eq!(detections.len(), 1);
    }
}
