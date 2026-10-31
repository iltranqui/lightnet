//! Shared YOLO detection geometry for detection and CARPN.

use crate::error::{Error, Result};
use crate::model::YoloSpec;

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

pub(crate) fn decode_obb_bbox(
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

pub(crate) fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

pub(crate) fn anchor_pair(spec: &YoloSpec, index: usize) -> Result<(f32, f32)> {
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
pub(crate) fn head_activation(raw: f32, new_coords: bool) -> f32 {
    if new_coords { raw } else { sigmoid(raw) }
}
