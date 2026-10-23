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
