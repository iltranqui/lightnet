//! YOLO output decoding and CPU non-maximum suppression.
//!
//! The runtime leaves YOLO heads in Darknet's flat channel-major layout.  This
//! module converts those tensors to image-space detections and keeps
//! post-processing separate from neural-network execution, like the C++
//! `get_yolo_detections()` and NMS helpers.

use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, YoloSpec};
use crate::runtime::{ModernHeadOutput, YoloHeadOutput};

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
            let objectness = head_activation(data[(anchor_base + 4) * stride + index], spec.new_coords);

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
                let class_prob = head_activation(
                    data[(anchor_base + 5 + class_id) * stride + index],
                    spec.new_coords,
                );
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
/// Decode the modern anchor-free heads (`[yolox]`, `[ppyoloe]`, `[yolonas]`,
/// `[centernet]`) into the same [`Detection`] shape as the yolo head, so
/// `detect-image` and `evaluate.rs` work unchanged on a modern cfg.
///
/// * YOLOX scores `obj * cls` and regresses cxcywh directly.
/// * PP-YOLOE / YOLO-NAS score `obj * cls` on the DFL-decoded box.
/// * CenterNet scores the sigmoid heatmap against its peak (`peak_nms`) and uses
///   the clamped sub-cell offset.
///
/// Letterbox correction, NMS and sorting run once over the merged list, exactly
/// as for the yolo head.
pub fn decode_modern_detections(
    network: &Network,
    modern_heads: &[ModernHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    for head in modern_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing layer #{} for a modern head", head.layer_index))
        })?;
        let (w, h, classes) = (head.output.shape.width, head.output.shape.height, head.output.shape.channels);
        let grid_w = w as f32;
        let grid_h = h as f32;
        match &layer.kind {
            LayerKind::Yolox(spec) => {
                let layout = crate::modern_yolo::ModernLayout::for_spec(
                    head.output.shape.batch, w, h, spec,
                );
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "yolox head tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.n {
                    for loc in 0..stride {
                        let obj = crate::modern_yolo::modern_objectness(
                            &head.output.data, &layout, 0, n, loc,
                        );
                        for c in 0..layout.classes {
                            let cls = crate::modern_yolo::modern_class_score(
                                &head.output.data, &layout, 0, n, loc, c,
                            );
                            let score = obj * cls;
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let box_ = crate::modern_yolo::decode_modern_box(
                                &head.output.data, &layout, 0, n, loc,
                            );
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: obj,
                                bbox: correct_bbox(
                                    DetectionBox { x: box_.x, y: box_.y, w: box_.w, h: box_.h },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            LayerKind::Ppyoloe(spec) | LayerKind::Yolonas(spec) => {
                let layout = crate::modern_yolo::ModernLayout::for_spec(
                    head.output.shape.batch, w, h, spec,
                );
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "distribution head tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.n {
                    for loc in 0..stride {
                        let obj = crate::modern_yolo::modern_objectness(
                            &head.output.data, &layout, 0, n, loc,
                        );
                        for c in 0..layout.classes {
                            let cls = crate::modern_yolo::modern_class_score(
                                &head.output.data, &layout, 0, n, loc, c,
                            );
                            // The distribution heads gate on the class score, not
                            // on the product, because the objectness logit is
                            // trained towards a soft quality target that is 0 for
                            // most of the map.
                            let score = cls.max(obj * cls);
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let box_ = crate::modern_yolo::decode_modern_box(
                                &head.output.data, &layout, 0, n, loc,
                            );
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: obj,
                                bbox: correct_bbox(
                                    DetectionBox { x: box_.x, y: box_.y, w: box_.w, h: box_.h },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            LayerKind::CenterNet(spec) => {
                let layout = crate::centernet::CenterNetLayout::for_spec(
                    head.output.shape.batch, w, h, spec,
                );
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "centernet tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.batch {
                    for c in 0..layout.classes {
                        for loc in 0..stride {
                            if spec.peak_nms != 0
                                && !crate::centernet::is_local_peak(
                                    &head.output.data, &layout, n, c, loc,
                                )
                            {
                                continue;
                            }
                            let score = crate::centernet::heatmap_score(
                                &head.output.data, &layout, n, c, loc,
                            );
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let (bx, by, bw, bh) =
                                crate::centernet::decode_centernet_box(&head.output.data, &layout, n, loc);
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: score,
                                bbox: correct_bbox(
                                    DetectionBox { x: bx, y: by, w: bw, h: bh },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            other => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ({}) is not a modern anchor-free head",
                    head.layer_index,
                    other.name()
                )));
            }
        }
        let _ = (grid_w, grid_h, classes);
    }
    apply_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| rhs.confidence.total_cmp(&lhs.confidence));
    Ok(detections)
}

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
    let bx = (col as f32 + scale_x_y * head_activation(tx, new_coords) - 0.5 * (scale_x_y - 1.0))
        / grid_width as f32;
    let by = (row as f32 + scale_x_y * head_activation(ty, new_coords) - 0.5 * (scale_x_y - 1.0))
        / grid_height as f32;
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

/// The runtime leaves the head tensor as the previous layer wrote it. Without `new_coords`
/// the C++ yolo layer applies the logistic to x, y, objectness and classes in forward;
/// with `new_coords` it does not (the preceding layer already activated them), so
/// `get_yolo_detections` reads the values as they are.
fn head_activation(raw: f32, new_coords: bool) -> f32 {
    if new_coords { raw } else { sigmoid(raw) }
}

#[cfg(test)]
mod tests {
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{Tensor, YoloHeadOutput};

    use super::{DetectionOptions, decode_yolo_detections};

    /// C++ `get_yolo_detections` + `get_yolo_box` with `new_coords=1`: head values are used as
    /// stored (already activated), x,y get `scale_x_y*v - 0.5*(scale_x_y-1)`, w = 4*tw^2*anchor/net.
    /// Grid 1x1, net 32, anchor 10x12, scale_x_y 1.2, tx=.5 ty=.25 tw=.5 th=.25, obj .9, cls .8:
    /// bx = 1.2*.5 - .1 = .5 -> 16 px; by = 1.2*.25 - .1 = .2 -> 6.4 px;
    /// w = 4*.25*10 = 10 px; h = 4*.0625*12 = 3 px; confidence = .9*.8 = .72.
    #[test]
    fn new_coords_head_is_decoded_without_sigmoid() {
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
                scale_x_y=1.2
                new_coords=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(
                network.layers[1].output_shape,
                vec![0.5, 0.25, 0.5, 0.25, 0.9, 0.8],
            )
            .expect("tensor"),
        };
        let detections = decode_yolo_detections(
            &network,
            &[head],
            32,
            32,
            DetectionOptions { confidence_threshold: 0.25, nms_threshold: 0.45, letterbox: false },
        )
        .expect("decode should succeed");
        assert_eq!(detections.len(), 1);
        let d = &detections[0];
        assert!((d.confidence - 0.72).abs() < 1.0e-5, "{}", d.confidence);
        assert!((d.objectness - 0.9).abs() < 1.0e-6);
        assert!((d.bbox.x - 16.0).abs() < 1.0e-4, "{}", d.bbox.x);
        assert!((d.bbox.y - 6.4).abs() < 1.0e-4, "{}", d.bbox.y);
        assert!((d.bbox.w - 10.0).abs() < 1.0e-4, "{}", d.bbox.w);
        assert!((d.bbox.h - 3.0).abs() < 1.0e-4, "{}", d.bbox.h);
    }

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

#[cfg(test)]
mod modern_decode_tests {
    use super::*;
    use crate::model::{CenterNetSpec, ModernYoloKind, ModernYoloSpec};
    use crate::runtime::{ModernHeadOutput, Tensor};
    use crate::{ConfigFile, Network, TensorShape};

    fn net(text: &str) -> Network {
        Network::from_config(ConfigFile::parse_str("t.cfg", text).expect("cfg parses")).expect("builds")
    }

    fn yolox_cfg() -> String {
        // 4x4 grid, 2 classes -> filters = classes + 5 = 7.
        String::from(
            "[net]\nwidth=16\nheight=16\nchannels=7\n[convolutional]\nfilters=7\nsize=4\nstride=4\nactivation=linear\n\
             [yolox]\nclasses=2\nmax=4\n",
        )
    }

    #[test]
    fn yolox_decodes_a_confident_box() {
        let network = net(&yolox_cfg());
        let head = &network.layers[1];
        let classes = 2usize;
        let spatial = 16usize;
        let entries = classes + 5;
        let mut data = vec![0.0f32; spatial * entries];
        // Cell (row 2, col 1): obj = 1, class 0 = 1 -> a confident detection.
        let loc = 2 * 4 + 1;
        data[0 * spatial + loc] = 1.0f32; // tx: centre at col + 1 = 2 of 4
        data[1 * spatial + loc] = 2.0f32; // ty: row + 2 = 4 of 4
        data[2 * spatial + loc] = 4f32.ln(); // w = 4 / 4 = 1
        data[3 * spatial + loc] = 4f32.ln(); // h = 4 / 4 = 1
        data[4 * spatial + loc] = 10.0; // objectness
        data[5 * spatial + loc] = 10.0; // class 0
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions { confidence_threshold: 0.3, nms_threshold: 0.45, letterbox: false },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert_eq!(dets[0].class_id, 0);
        assert!(dets[0].confidence > 0.99, "{}", dets[0].confidence);
        // cx = 2/4, cy = 4/4, w = h = 1 in a 16x16 input.
        assert!((dets[0].bbox.x - 8.0).abs() < 1e-3, "x {}", dets[0].bbox.x);
        assert!((dets[0].bbox.y - 16.0).abs() < 1e-3, "y {}", dets[0].bbox.y);
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-3, "w {}", dets[0].bbox.w);
        assert!((dets[0].bbox.h - 16.0).abs() < 1e-3, "h {}", dets[0].bbox.h);
    }

    #[test]
    fn the_confidence_threshold_filters_and_nms_deduplicates() {
        let network = net(&yolox_cfg());
        let head = &network.layers[1];
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 7];
        // Two adjacent cells both fire on class 1: NMS must merge them.  The
        // objectness and class entries are already sigmoid-activated by the
        // head's forward, so the decode reads probabilities, not logits.
        for loc in [0usize, 1] {
            data[0 * spatial + loc] = 0.0;
            data[1 * spatial + loc] = 0.0;
            data[2 * spatial + loc] = 4f32.ln();
            data[3 * spatial + loc] = 4f32.ln();
            data[4 * spatial + loc] = 0.99; // objectness
            data[6 * spatial + loc] = 0.98; // class 1
        }
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            std::slice::from_ref(&out),
            16,
            16,
            DetectionOptions { confidence_threshold: 0.9, nms_threshold: 0.45, letterbox: false },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "NMS merged the overlapping pair: {dets:?}");
        // A high threshold keeps nothing.
        let none = decode_modern_detections(
            &network,
            &[out.clone()],
            16,
            16,
            DetectionOptions { confidence_threshold: 0.995, nms_threshold: 0.45, letterbox: false },
        )
        .unwrap();
        assert!(none.is_empty(), "{none:?}");
    }

    #[test]
    fn centernet_decodes_only_peaks() {
        // 4x4 grid, 1 class -> filters = 1 + 4 = 5.
        let cfg = String::from(
            "[net]\nwidth=16\nheight=16\nchannels=5\n[convolutional]\nfilters=5\nsize=4\nstride=4\nactivation=linear\n\
             [centernet]\nclasses=1\nmax=4\n",
        );
        let network = net(&cfg);
        let head = &network.layers[1];
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 5];
        // A ridge: cell 5 is the peak, its neighbours are lower.
        for (loc, v) in [(5usize, 10.0f32), (4, 6.0), (6, 6.0), (9, 6.0)] {
            data[loc] = v;
        }
        data[1 * spatial + 5] = 4f32.ln(); // log_w = 1 cell
        data[2 * spatial + 5] = 4f32.ln();
        data[3 * spatial + 5] = 0.0;
        data[4 * spatial + 5] = 0.0;
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions { confidence_threshold: 0.5, nms_threshold: 0.45, letterbox: false },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "only the strict peak survives: {dets:?}");
        assert_eq!(dets[0].class_id, 0);
        // cx = cy = (col + 0.5) / 4 = 0.375 -> 6 px of 16.
        assert!((dets[0].bbox.x - 6.0).abs() < 1e-3, "x {}", dets[0].bbox.x);
        assert!((dets[0].bbox.y - 6.0).abs() < 1e-3, "y {}", dets[0].bbox.y);
        // exp(ln 4) / 4 cells = the full 16 px width.
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-3, "w {}", dets[0].bbox.w);
    }

    #[test]
    fn the_distribution_heads_decode_through_the_dfl_expectation() {
        // 4x4 grid, 1 class, reg_max = 3 -> coords = 16, filters = 16 + 1 + 1 = 18.
        let cfg = String::from(
            "[net]\nwidth=16\nheight=16\nchannels=18\n[convolutional]\nfilters=18\nsize=4\nstride=4\nactivation=linear\n\
             [ppyoloe]\nclasses=1\nmax=4\nreg_max=3\n",
        );
        let network = net(&cfg);
        let head = &network.layers[1];
        let bins = 4usize;
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 18];
        let loc = 9usize;
        // Make every side collapse onto bin 2 (i.e. a 2-cell distance) and the
        // class score confident.
        for side in 0..4 {
            data[(side * bins + 2) * spatial + loc] = 40.0;
        }
        data[(4 * bins) * spatial + loc] = 40.0; // quality (objectness)
        data[(4 * bins + 1) * spatial + loc] = 40.0; // class 0
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions { confidence_threshold: 0.3, nms_threshold: 0.45, letterbox: false },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        // All four distances are 2 cells around cell (row 2, col 1), so the box
        // centre is the cell centre (1.5 / 4 -> 6 px) and both sides span 4
        // cells (16 px) in total.
        assert!((dets[0].bbox.x - 6.0).abs() < 1e-2, "x {}", dets[0].bbox.x);
        // row 2, col 1 -> cy = 2.5 / 4 = 0.625 -> 10 px.
        assert!((dets[0].bbox.y - 10.0).abs() < 1e-2, "y {}", dets[0].bbox.y);
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-2, "w {}", dets[0].bbox.w);
    }

    #[test]
    fn a_non_modern_layer_is_rejected() {
        let network = net(&yolox_cfg());
        let out = ModernHeadOutput {
            layer_index: 0, // the convolution, not a head
            output: Tensor::from_data(
                network.layers[0].output_shape,
                vec![0.0; network.layers[0].output_shape.batch * network.layers[0].output_shape.element_count()],
            )
            .unwrap(),
        };
        let err = decode_modern_detections(&network, &[out], 16, 16, DetectionOptions::default())
            .expect_err("a convolution is not a modern head");
        assert!(format!("{err}").contains("not a modern anchor-free head"), "{err}");
    }
}
