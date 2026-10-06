//! YOLO output decoding and CPU non-maximum suppression, with separate OBB
//! post-processing primitives.
//!
//! The runtime leaves YOLO heads in Darknet's flat channel-major layout.  This
//! module converts those tensors to image-space detections and keeps
//! post-processing separate from neural-network execution, like the C++
//! `get_yolo_detections()` and NMS helpers.

use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, YoloSpec};
use crate::obb_geometry::{BoxObb, box_obb_iou, carpn_polygon_iou};
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

/// Geometry carried by a detection from an OBB head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObbDetectionGeometry {
    /// Legacy Darknet OBB representation: one point on the height axis.
    FrontPoint { x: f32, y: f32 },
    /// CARPN representation: four interleaved vertex `(x, y)` pairs.
    CarpnQuad([f32; 8]),
}

/// A class-specific detection with OBB geometry, separate from the existing
/// axis-aligned [`Detection`] API.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbDetection {
    pub detection: Detection,
    pub geometry: ObbDetectionGeometry,
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
            LayerKind::ObbYolo(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is an OBB head; use decode_obb_yolo_detections",
                    head.layer_index
                )));
            }
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

/// Decode and merge the network's legacy `[obb_yolo]` heads, then run
/// orientation-aware NMS once across all heads. This API currently supports
/// batch size one; OBB training and mAP are separate, unimplemented paths.
pub fn decode_obb_yolo_detections(
    network: &Network,
    obb_heads: &[YoloHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<ObbDetection>> {
    let mut detections = Vec::new();
    for head in obb_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing layer #{} for OBB YOLO head",
                head.layer_index
            ))
        })?;
        let spec = match &layer.kind {
            LayerKind::ObbYolo(spec) => spec,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is not an OBB YOLO head",
                    head.layer_index
                )));
            }
        };
        if head.output.shape.batch != 1 {
            return Err(Error::InvalidConfig(format!(
                "OBB YOLO decode currently requires batch size one, got {}",
                head.output.shape.batch
            )));
        }
        detections.extend(decode_obb_yolo_head(
            &head.output.data,
            head.output.shape.width,
            head.output.shape.height,
            spec,
            network.net.width,
            network.net.height,
            image_width,
            image_height,
            options,
        )?);
    }

    apply_obb_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| {
        rhs.detection
            .confidence
            .total_cmp(&lhs.detection.confidence)
    });
    Ok(detections)
}

/// Decode one six-coordinate legacy OBB YOLO head into per-class detections.
///
/// `data` is one image's output in the existing Darknet channel-major layout
/// with six box channels, one objectness channel, and then `classes` channels
/// per anchor. The returned front point is letterbox-corrected and expressed in
/// image pixels, like the
/// bbox. This per-head helper does not run NMS; merge head outputs first, then
/// call [`apply_obb_nms`].
#[allow(clippy::too_many_arguments)]
pub fn decode_obb_yolo_head(
    data: &[f32],
    grid_width: usize,
    grid_height: usize,
    spec: &YoloSpec,
    net_width: usize,
    net_height: usize,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<ObbDetection>> {
    if grid_width == 0
        || grid_height == 0
        || net_width == 0
        || net_height == 0
        || image_width == 0
        || image_height == 0
        || spec.classes == 0
        || spec.mask.is_empty()
        || !spec.scale_x_y.is_finite()
    {
        return Err(Error::InvalidConfig(
            "invalid geometry or spec for OBB YOLO decoding".to_owned(),
        ));
    }

    let stride = grid_width
        .checked_mul(grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO grid size overflow".to_owned()))?;
    let entries_per_anchor = spec
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO entry count overflow".to_owned()))?;
    let expected = stride
        .checked_mul(spec.mask.len())
        .and_then(|size| size.checked_mul(entries_per_anchor))
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO head size overflow".to_owned()))?;
    if data.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB YOLO head tensor length mismatch: expected {expected}, got {}",
            data.len()
        )));
    }

    let correction = ObbImageCorrection::new(
        net_width,
        net_height,
        image_width,
        image_height,
        options.letterbox,
    )?;
    let mut detections = Vec::new();
    for index in 0..stride {
        let row = index / grid_width;
        let col = index % grid_width;
        for (anchor_offset, &mask_idx) in spec.mask.iter().enumerate() {
            let anchor = anchor_pair(spec, mask_idx)?;
            if !anchor.0.is_finite() || !anchor.1.is_finite() || anchor.0 <= 0.0 || anchor.1 <= 0.0
            {
                return Err(Error::InvalidConfig(format!(
                    "invalid OBB YOLO anchor for mask index {mask_idx}"
                )));
            }

            let anchor_base = anchor_offset * entries_per_anchor;
            let objectness = finite_or_zero(head_activation(
                data[(anchor_base + 6) * stride + index],
                spec.new_coords,
            ));
            if objectness <= options.confidence_threshold {
                continue;
            }

            let bbox = decode_obb_bbox(
                data[(anchor_base + 0) * stride + index],
                data[(anchor_base + 1) * stride + index],
                data[(anchor_base + 2) * stride + index],
                data[(anchor_base + 3) * stride + index],
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
            let bbox = correction.correct_bbox(bbox, image_width as f32, image_height as f32);

            // Source channels 4/5 are raw, linear, cell-relative front offsets.
            let front_x = finite_or_zero(data[(anchor_base + 4) * stride + index]);
            let front_y = finite_or_zero(data[(anchor_base + 5) * stride + index]);
            let front_x = (col as f32 + front_x) / grid_width as f32;
            let front_y = (row as f32 + front_y) / grid_height as f32;
            let [front_x, front_y] = correction.correct_front_point(
                [front_x, front_y],
                image_width as f32,
                image_height as f32,
            );

            for class_id in 0..spec.classes {
                let class_probability = finite_or_zero(head_activation(
                    data[(anchor_base + 7 + class_id) * stride + index],
                    spec.new_coords,
                ));
                let confidence = objectness * class_probability;
                if confidence > options.confidence_threshold {
                    detections.push(ObbDetection {
                        detection: Detection {
                            class_id,
                            confidence,
                            objectness,
                            bbox,
                        },
                        geometry: ObbDetectionGeometry::FrontPoint {
                            x: front_x,
                            y: front_y,
                        },
                    });
                }
            }
        }
    }
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

fn decode_obb_bbox(
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
    let scale_offset = 0.5 * (scale_x_y - 1.0);
    let tx = finite_or_zero(scale_x_y * head_activation(tx, new_coords) - scale_offset);
    let ty = finite_or_zero(scale_x_y * head_activation(ty, new_coords) - scale_offset);
    let tw = finite_or_zero(tw);
    let th = finite_or_zero(th);
    let (width, height) = if new_coords {
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
    DetectionBox {
        x: finite_or_zero((col as f32 + tx) / grid_width as f32),
        y: finite_or_zero((row as f32 + ty) / grid_height as f32),
        w: finite_or_zero(width).max(1e-6),
        h: finite_or_zero(height).max(1e-6),
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[derive(Clone, Copy, Debug)]
struct ObbImageCorrection {
    delta_width: f32,
    delta_height: f32,
    ratio_width: f32,
    ratio_height: f32,
    net_width: f32,
    net_height: f32,
}

impl ObbImageCorrection {
    fn new(
        net_width: usize,
        net_height: usize,
        image_width: usize,
        image_height: usize,
        letterbox: bool,
    ) -> Result<Self> {
        let (new_width, new_height) = if letterbox {
            if net_width as f32 / (image_width as f32) < net_height as f32 / (image_height as f32) {
                (
                    net_width,
                    image_height.checked_mul(net_width).ok_or_else(|| {
                        Error::InvalidConfig("OBB letterbox height overflow".to_owned())
                    })? / image_width,
                )
            } else {
                (
                    image_width.checked_mul(net_height).ok_or_else(|| {
                        Error::InvalidConfig("OBB letterbox width overflow".to_owned())
                    })? / image_height,
                    net_height,
                )
            }
        } else {
            (net_width, net_height)
        };
        if new_width == 0 || new_height == 0 {
            return Err(Error::InvalidConfig(
                "OBB letterbox dimensions collapse to zero".to_owned(),
            ));
        }

        let (net_width, net_height) = (net_width as f32, net_height as f32);
        let (new_width, new_height) = (new_width as f32, new_height as f32);
        Ok(Self {
            delta_width: net_width - new_width,
            delta_height: net_height - new_height,
            ratio_width: new_width / net_width,
            ratio_height: new_height / net_height,
            net_width,
            net_height,
        })
    }

    fn correct_bbox(
        self,
        mut bbox: DetectionBox,
        image_width: f32,
        image_height: f32,
    ) -> DetectionBox {
        bbox.x = (bbox.x - self.delta_width / 2.0 / self.net_width) / self.ratio_width;
        bbox.y = (bbox.y - self.delta_height / 2.0 / self.net_height) / self.ratio_height;
        bbox.w /= self.ratio_width;
        bbox.h /= self.ratio_height;
        bbox.x *= image_width;
        bbox.y *= image_height;
        bbox.w *= image_width;
        bbox.h *= image_height;
        bbox
    }

    fn correct_front_point(
        self,
        mut front: [f32; 2],
        image_width: f32,
        image_height: f32,
    ) -> [f32; 2] {
        front[0] = (front[0] - self.delta_width / 2.0 / self.net_width) / self.ratio_width;
        front[1] = (front[1] - self.delta_height / 2.0 / self.net_height) / self.ratio_height;
        front[0] *= image_width;
        front[1] *= image_height;
        front
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

/// IoU dispatch used by OBB post-processing.
///
/// Two CARPN quads use explicit polygon IoU. All other OBB pairs use the first
/// point pair as the legacy front point, matching `detection_obb_iou` when one
/// detection is CARPN and the other is not. This helper accepts only OBB records
/// and never silently falls back to AABB IoU.
pub fn detection_obb_iou(lhs: &ObbDetection, rhs: &ObbDetection) -> f32 {
    if let (
        ObbDetectionGeometry::CarpnQuad(lhs_vertices),
        ObbDetectionGeometry::CarpnQuad(rhs_vertices),
    ) = (lhs.geometry, rhs.geometry)
    {
        return carpn_polygon_iou(lhs_vertices, rhs_vertices);
    }

    let front_point = |geometry| match geometry {
        ObbDetectionGeometry::FrontPoint { x, y } => [x, y],
        ObbDetectionGeometry::CarpnQuad(vertices) => [vertices[0], vertices[1]],
    };
    let [lhs_front_x, lhs_front_y] = front_point(lhs.geometry);
    let [rhs_front_x, rhs_front_y] = front_point(rhs.geometry);
    box_obb_iou(
        BoxObb {
            bbox: lhs.detection.bbox,
            front_x: lhs_front_x,
            front_y: lhs_front_y,
        },
        BoxObb {
            bbox: rhs.detection.bbox,
            front_x: rhs_front_x,
            front_y: rhs_front_y,
        },
    )
}

/// Apply classwise greedy OBB NMS, sorting candidates by descending confidence.
///
/// `confidence` is the per-class probability for this class-specific record.
/// Exactly-zero objectness or confidence records remain in the output but do
/// not participate in suppression, matching `diounms_sort`. Suppression uses a
/// strict `IoU > threshold` comparison. Equal-score order is stable and defined
/// here because the C++ sorter does not specify tie order.
pub fn apply_obb_nms(detections: &mut Vec<ObbDetection>, threshold: f32) {
    detections.sort_by(|lhs, rhs| {
        let lhs_inactive = lhs.detection.objectness == 0.0;
        let rhs_inactive = rhs.detection.objectness == 0.0;
        lhs_inactive.cmp(&rhs_inactive).then_with(|| {
            rhs.detection
                .confidence
                .total_cmp(&lhs.detection.confidence)
        })
    });

    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        let keeper = &detections[i];
        if suppressed[i] || keeper.detection.objectness == 0.0 || keeper.detection.confidence == 0.0
        {
            continue;
        }

        for j in (i + 1)..detections.len() {
            let candidate = &detections[j];
            if suppressed[j]
                || keeper.detection.class_id != candidate.detection.class_id
                || candidate.detection.objectness == 0.0
                || candidate.detection.confidence == 0.0
            {
                continue;
            }
            if detection_obb_iou(keeper, candidate) > threshold {
                suppressed[j] = true;
            }
        }
    }

    let mut kept = 0usize;
    for (index, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if kept != index {
                detections.swap(kept, index);
            }
            kept += 1;
        }
    }
    detections.truncate(kept);
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
mod obb_decode_tests {
    use crate::model::{YoloIouLoss, YoloSpec};

    use super::{DetectionOptions, ObbDetectionGeometry, decode_obb_yolo_head};

    fn spec(new_coords: bool, scale_x_y: f32) -> YoloSpec {
        YoloSpec {
            classes: 2,
            num: 1,
            mask: vec![0],
            anchors: vec![10.0, 12.0],
            scale_x_y,
            new_coords,
            cls_normalizer: 1.0,
            obj_normalizer: 1.0,
            iou_normalizer: 1.0,
            ignore_thresh: 0.5,
            truth_thresh: 1.0,
            iou_thresh: 1.0,
            iou_loss: YoloIouLoss::Mse,
            iou_thresh_kind: YoloIouLoss::Iou,
            focal_loss: false,
            label_smooth_eps: 0.0,
            objectness_smooth: false,
            max_delta: 5.0,
            delta_normalizer: 1.0,
            class_multipliers: None,
            class_map: None,
        }
    }

    fn options(confidence_threshold: f32, letterbox: bool) -> DetectionOptions {
        DetectionOptions {
            confidence_threshold,
            nms_threshold: 0.45,
            letterbox,
        }
    }

    #[test]
    fn logistic_obb_head_decodes_box_front_and_strict_thresholds() {
        let outputs = decode_obb_yolo_head(
            &[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 2.0, -2.0],
            1,
            1,
            &spec(false, 1.2),
            32,
            32,
            32,
            32,
            options(0.1, false),
        )
        .expect("valid OBB head");
        assert_eq!(outputs.len(), 1, "only class 0 exceeds the score threshold");
        let detection = &outputs[0].detection;
        assert_eq!(detection.class_id, 0);
        assert!((detection.objectness - 0.5).abs() < 1e-6);
        assert!((detection.confidence - 0.5 * (1.0 / (1.0 + (-2.0f32).exp()))).abs() < 1e-6);
        assert!((detection.bbox.x - 16.0).abs() < 1e-6);
        assert!((detection.bbox.y - 16.0).abs() < 1e-6);
        assert!((detection.bbox.w - 10.0).abs() < 1e-6);
        assert!((detection.bbox.h - 12.0).abs() < 1e-6);
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 8.0).abs() < 1e-6);
        assert!((y - 16.0).abs() < 1e-6);

        let at_objectness_threshold = decode_obb_yolo_head(
            &[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 2.0, -2.0],
            1,
            1,
            &spec(false, 1.2),
            32,
            32,
            32,
            32,
            options(0.5, false),
        )
        .expect("valid OBB head");
        assert!(
            at_objectness_threshold.is_empty(),
            "objectness threshold is strict"
        );
    }

    #[test]
    fn new_coords_obb_head_corrects_front_point_for_letterbox() {
        let outputs = decode_obb_yolo_head(
            &[0.5, 0.5, 0.5, 0.5, 0.75, 0.25, 0.9, 0.8, 0.7],
            1,
            1,
            &spec(true, 1.2),
            32,
            32,
            64,
            32,
            options(0.7, true),
        )
        .expect("valid new-coordinates OBB head");
        assert_eq!(
            outputs.len(),
            1,
            "NMS is deferred and only class 0 meets threshold"
        );
        let detection = &outputs[0].detection;
        assert_eq!(detection.class_id, 0);
        assert!((detection.confidence - 0.72).abs() < 1e-6);
        assert!((detection.bbox.x - 32.0).abs() < 1e-6);
        assert!((detection.bbox.y - 16.0).abs() < 1e-6);
        assert!((detection.bbox.w - 20.0).abs() < 1e-6);
        assert!((detection.bbox.h - 24.0).abs() < 1e-6);
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 48.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn obb_letterbox_uses_integer_resized_dimensions() {
        let outputs = decode_obb_yolo_head(
            &[0.5, 0.5, 0.5, 0.5, 0.75, 0.25, 0.9, 0.8, 0.7],
            1,
            1,
            &spec(true, 1.0),
            32,
            32,
            5,
            3,
            options(0.7, true),
        )
        .expect("valid OBB head");
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 3.75).abs() < 1e-6);
        // C++ computes new_h=(3*32)/5 using integer division, so 19/32 is
        // used instead of the mathematically exact 19.2/32 scale.
        assert!((y - 0.236_842_1).abs() < 1e-6);
    }

    #[test]
    fn obb_yolo_decoder_rejects_bad_tensor_lengths_and_zero_grids() {
        let spec = spec(false, 1.0);
        let error =
            decode_obb_yolo_head(&[0.0; 8], 1, 1, &spec, 32, 32, 32, 32, options(0.1, false))
                .expect_err("the six-coordinate OBB layout needs nine entries");
        assert!(format!("{error}").contains("length mismatch"));

        let error = decode_obb_yolo_head(&[], 0, 1, &spec, 32, 32, 32, 32, options(0.1, false))
            .expect_err("zero grid dimensions are invalid");
        assert!(format!("{error}").contains("invalid geometry"));
    }
}

#[cfg(test)]
mod obb_postprocess_tests {
    use super::{
        Detection, DetectionBox, ObbDetection, ObbDetectionGeometry, apply_obb_nms,
        detection_obb_iou,
    };

    fn rect_detection(
        class_id: usize,
        confidence: f32,
        objectness: f32,
        front: [f32; 2],
    ) -> ObbDetection {
        ObbDetection {
            detection: Detection {
                class_id,
                confidence,
                objectness,
                bbox: DetectionBox {
                    x: 0.0,
                    y: 0.0,
                    w: 4.0,
                    h: 2.0,
                },
            },
            geometry: ObbDetectionGeometry::FrontPoint {
                x: front[0],
                y: front[1],
            },
        }
    }

    #[test]
    fn detection_iou_dispatches_carpon_and_legacy_front_points() {
        let horizontal = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let vertical = rect_detection(0, 0.8, 0.8, [1.0, 0.0]);
        assert!((detection_obb_iou(&horizontal, &vertical) - 1.0 / 3.0).abs() < 1e-6);

        let square = ObbDetection {
            detection: horizontal.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]),
        };
        let diamond = ObbDetection {
            detection: vertical.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0]),
        };
        assert!((detection_obb_iou(&square, &diamond) - 0.5).abs() < 1e-6);

        // C++ falls through to its legacy path for mixed OBB representations,
        // using the CARPN mask's first vertex as the front point.
        let mixed = ObbDetection {
            detection: vertical.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0]),
        };
        assert!((detection_obb_iou(&horizontal, &mixed) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn obb_nms_is_classwise_orientation_aware_and_uses_strict_thresholds() {
        let low = rect_detection(0, 0.8, 0.9, [1.0, 0.0]);
        let high = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let other_class = rect_detection(1, 0.7, 0.9, [1.0, 0.0]);
        let overlap = detection_obb_iou(&low, &high);
        assert!((overlap - 1.0 / 3.0).abs() < 1e-6);

        let mut orientation_safe = vec![low.clone(), high.clone()];
        apply_obb_nms(&mut orientation_safe, 0.5);
        assert_eq!(orientation_safe.len(), 2, "OBB IoU is below AABB IoU");

        let mut below_threshold = vec![low.clone(), other_class.clone(), high.clone()];
        apply_obb_nms(&mut below_threshold, overlap - 1e-4);
        assert_eq!(below_threshold.len(), 2);
        assert_eq!(
            below_threshold[0].detection.confidence,
            high.detection.confidence
        );
        assert!(below_threshold.iter().any(|d| d.detection.class_id == 1));

        let mut at_threshold = vec![low, high, other_class];
        apply_obb_nms(&mut at_threshold, overlap);
        assert_eq!(at_threshold.len(), 3, "IoU equal to threshold is retained");
    }

    #[test]
    fn zero_objectness_candidates_never_suppress_or_get_suppressed() {
        let inactive = rect_detection(0, 0.99, 0.0, [0.0, 1.0]);
        let active = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut detections = vec![active.clone(), inactive.clone()];
        apply_obb_nms(&mut detections, 0.0);
        assert_eq!(detections.len(), 2);
        assert!(detections.iter().any(|d| d == &inactive));
        assert!(detections.iter().any(|d| d == &active));

        let zero_probability = rect_detection(0, 0.0, 0.9, [0.0, 1.0]);
        let active = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut zero_score = vec![active.clone(), zero_probability.clone()];
        apply_obb_nms(&mut zero_score, 0.0);
        assert_eq!(zero_score.len(), 2);
        assert!(zero_score.iter().any(|d| d == &zero_probability));

        let duplicate = rect_detection(0, 0.7, 0.9, [0.0, 1.0]);
        let mut positive = vec![active, duplicate];
        apply_obb_nms(&mut positive, 0.0);
        assert_eq!(positive.len(), 1, "positive overlap exceeds zero threshold");
    }
}

#[cfg(test)]
mod obb_network_decode_tests {
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{ExecutableNetwork, Tensor};

    use super::{
        DetectionOptions, ObbDetectionGeometry, decode_obb_yolo_detections, decode_yolo_detections,
    };

    #[test]
    fn network_heads_merge_before_orientation_aware_nms() {
        let cfg = ConfigFile::parse_str(
            "obb-detect.cfg",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("OBB network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let stride = network.net.width * network.net.height;
        let mut data = vec![0.0; network.net.channels * stride];
        data[4 * stride] = 0.25; // Raw linear front-point x offset.
        data[6 * stride] = 2.0; // Objectness logit.
        data[7 * stride] = 2.0; // Class logit.
        let input = Tensor::from_data(network.net.input_shape(), data).expect("input tensor");
        let forward = executable.forward(&input).expect("forward pass");
        assert_eq!(forward.yolo_heads.len(), 2);

        let detections = decode_obb_yolo_detections(
            &network,
            &forward.yolo_heads,
            4,
            2,
            DetectionOptions::default(),
        )
        .expect("OBB heads decode");
        assert_eq!(detections.len(), 1, "duplicate heads should be suppressed");
        assert!((detections[0].detection.bbox.x - 0.5).abs() < 1e-6);
        assert!((detections[0].detection.bbox.y - 0.5).abs() < 1e-6);
        assert!((detections[0].detection.bbox.w - 4.0).abs() < 1e-6);
        assert!((detections[0].detection.bbox.h - 2.0).abs() < 1e-6);
        assert!(matches!(
            detections[0].geometry,
            ObbDetectionGeometry::FrontPoint { x, y }
                if (x - 0.25).abs() < 1e-6 && y.abs() < 1e-6
        ));
        assert!(
            decode_yolo_detections(
                &network,
                &forward.yolo_heads,
                4,
                2,
                DetectionOptions::default(),
            )
            .is_err(),
            "OBB heads must not enter axis-aligned decode"
        );
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
