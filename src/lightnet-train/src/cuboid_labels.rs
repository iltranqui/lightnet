//! Projected cuboid labels and deterministic CPU truth encoding.
//!
//! Hand-derived from `depth_3d` at `9c9e0b4b`, `data.cpp:500-1250`;
//! no compiled-C++ golden. Network execution remains unsupported.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::Yolo3dSpec;
use crate::train::YoloLabel;

/// Projected normalized vertices in source corner order, with visibility c.
/// c=0 is absent; c<0 is unknown but participates in the hull/coordinate loss.
#[derive(Clone, Debug, PartialEq)]
pub struct CuboidLabel {
    pub class_id: usize,
    pub track_id: u32,
    pub vertices: Vec<[f32; 3]>,
}

/// Mixed rows preserve the source's separate 2D/cuboid ownership. The 2D
/// payload also carries track_id: reusing YoloLabel alone would discard it.
#[derive(Clone, Debug, PartialEq)]
pub enum MixedLabel {
    Box2d { label: YoloLabel, track_id: u32 },
    Cuboid(CuboidLabel),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yolo3dCrop {
    /// Crop rectangle in original-image pixels; it may extend past the image.
    pub pleft: f32,
    pub ptop: f32,
    pub swidth: f32,
    pub sheight: f32,
    pub flip: bool,
    /// Marks an already composed mosaic: retain only boxes/class. Source
    /// blend_truth_mosaic also drops mixed ownership, leaving flags zero.
    /// This API does not perform mosaic sampling or tile placement.
    pub mosaic: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yolo3dImageMeta {
    pub original_width: usize,
    pub original_height: usize,
    /// Needed for the source's one-network-pixel minimum hull size.
    pub network_width: usize,
    pub network_height: usize,
    pub crop: Yolo3dCrop,
}

/// Batch-major, object-major truth records: 11-float prefix + 3*K vertices,
/// with one trailing ownership flag only in mixed mode (0=padding, 1=2D, 2=3D).
#[derive(Clone, Debug, PartialEq)]
pub struct Yolo3dTruthBatch {
    pub truths: Vec<f32>,
    pub truth_size: usize,
    pub max_boxes: usize,
    pub batch: usize,
    pub counts: Vec<usize>,
    pub mixed: bool,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InvalidConfig(format!("[yolo_3d] labels/truth: {message}"))
}

impl CuboidLabel {
    /// Hull of present/unknown vertices, or all vertices when every c is zero.
    pub fn hull(&self) -> Result<[f32; 4]> {
        validate_vertices(&self.vertices)?;
        let all_absent = self.vertices.iter().all(|v| v[2] == 0.0);
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for v in self.vertices.iter().filter(|v| all_absent || v[2] != 0.0) {
            bounds[0] = bounds[0].min(v[0]);
            bounds[1] = bounds[1].min(v[1]);
            bounds[2] = bounds[2].max(v[0]);
            bounds[3] = bounds[3].max(v[1]);
        }
        if bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
            return Err(invalid("degenerate cuboid hull"));
        }
        Ok(bounds)
    }
}

fn validate_vertices(vertices: &[[f32; 3]]) -> Result<()> {
    if !(1..=8).contains(&vertices.len())
        || vertices.iter().any(|v| {
            v.iter().any(|x| !x.is_finite())
                || !(0.0..=1.0).contains(&v[0])
                || !(0.0..=1.0).contains(&v[1])
        })
    {
        return Err(invalid(
            "expected 1..=8 finite vertices with normalized u,v",
        ));
    }
    Ok(())
}

/// Read mixed source rows, including the 5/7/9-field 2D precedence rule.
/// Missing files and non-UTF-8 path identities are errors, not empty labels.
pub fn load_cuboid_labels(
    path: impl AsRef<Path>,
    classes: Option<usize>,
    keypoints: usize,
) -> Result<Vec<MixedLabel>> {
    let path = path.as_ref();
    let identity = path
        .to_str()
        .ok_or_else(|| invalid("label path must be UTF-8"))?;
    parse_mixed_cuboid_labels(
        &std::fs::read_to_string(path)?,
        identity,
        classes,
        keypoints,
    )
}

/// Load a pure cuboid file without the mixed grammar's ambiguous-row rule.
pub fn load_pure_cuboid_labels(
    path: impl AsRef<Path>,
    classes: Option<usize>,
    keypoints: usize,
) -> Result<Vec<CuboidLabel>> {
    let path = path.as_ref();
    let identity = path
        .to_str()
        .ok_or_else(|| invalid("label path must be UTF-8"))?;
    parse_cuboid_labels(
        &std::fs::read_to_string(path)?,
        identity,
        classes,
        keypoints,
    )
}

/// Pure cuboid grammar: 3*K floats or leading class followed by 3*K floats.
/// Unlike mixed mode this disambiguates K=2/3 rows in favour of cuboids.
pub fn parse_cuboid_labels(
    text: &str,
    identity: &str,
    classes: Option<usize>,
    keypoints: usize,
) -> Result<Vec<CuboidLabel>> {
    parse_rows(text, identity, classes, keypoints, false)?
        .into_iter()
        .map(|label| match label {
            MixedLabel::Cuboid(cuboid) => Ok(cuboid),
            MixedLabel::Box2d { .. } => Err(invalid("unexpected 2D row in pure cuboid parser")),
        })
        .collect()
}

/// Source mixed grammar; 5/7/9-field rows are 2D before testing 3*K/1+3*K.
/// Every token must be finite and numeric (stricter than the C++ stream parser,
/// which may stop at malformed trailing text). Blank lines are skipped.
pub fn parse_mixed_cuboid_labels(
    text: &str,
    identity: &str,
    classes: Option<usize>,
    keypoints: usize,
) -> Result<Vec<MixedLabel>> {
    parse_rows(text, identity, classes, keypoints, true)
}

fn parse_rows(
    text: &str,
    identity: &str,
    classes: Option<usize>,
    keypoints: usize,
    mixed: bool,
) -> Result<Vec<MixedLabel>> {
    if !(1..=8).contains(&keypoints) || classes.is_some_and(|c| c == 0 || c > i32::MAX as usize) {
        return Err(invalid(
            "keypoints must be 1..=8 and classes must be positive source ints",
        ));
    }
    // Linux C++ unsigned long DJB2 with signed char bytes; path-based track
    // IDs have no loss effect but must survive the truth prefix unchanged.
    let hash = identity.bytes().fold(5381u64, |h, b| {
        h.wrapping_mul(33).wrapping_add(b as i8 as i64 as u64)
    });
    let base_track = ((hash % 4000) * 4000) as u32;
    let mut labels = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let context =
            |error: Error| invalid(format!("{identity}: line {}: {error}", line_index + 1));
        let values: Vec<f32> = line
            .split_whitespace()
            .map(|word| {
                word.parse::<f32>()
                    .map_err(|_| invalid(format!("invalid numeric token {word:?}")))
            })
            .collect::<Result<_>>()
            .map_err(context)?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(context(invalid("nonfinite annotation value")));
        }
        let offset = u32::try_from(labels.len())
            .ok()
            .and_then(|n| n.checked_add(u32::from(mixed)))
            .ok_or_else(|| context(invalid("track row count overflows")))?;
        let track_id = base_track
            .checked_add(offset)
            .ok_or_else(|| context(invalid("track ID overflows")))?;
        let parse_class = |v: f32| -> Result<usize> {
            let v = f64::from(v).trunc();
            if v < 0.0 || v > f64::from(i32::MAX) {
                return Err(invalid("class ID is outside source int range"));
            }
            let id = v as usize;
            if classes.is_some_and(|c| id >= c) {
                return Err(invalid("class ID exceeds class count"));
            }
            Ok(id)
        };
        if mixed && matches!(values.len(), 5 | 7 | 9) {
            let class_id = parse_class(values[0]).map_err(context)?;
            labels.push(MixedLabel::Box2d {
                label: YoloLabel {
                    class_id,
                    x: values[1].clamp(0.0, 1.0),
                    y: values[2].clamp(0.0, 1.0),
                    w: values[3].clamp(0.0, 1.0),
                    h: values[4].clamp(0.0, 1.0),
                },
                track_id,
            });
        } else {
            let (class_id, offset) = if values.len() == 3 * keypoints + 1 {
                (parse_class(values[0]).map_err(context)?, 1)
            } else if values.len() == 3 * keypoints {
                (0, 0)
            } else {
                return Err(context(invalid(format!(
                    "invalid field count {} for K={keypoints}",
                    values.len()
                ))));
            };
            let vertices = values[offset..]
                .chunks_exact(3)
                .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0), p[2]])
                .collect();
            let label = CuboidLabel {
                class_id,
                track_id,
                vertices,
            };
            label.hull().map_err(context)?;
            labels.push(MixedLabel::Cuboid(label));
        }
    }
    Ok(labels)
}

impl Yolo3dTruthBatch {
    /// Includes padded records. Invalid indices or inconsistent public storage
    /// return an error rather than slicing out of bounds.
    pub fn record(&self, batch: usize, object: usize) -> Result<&[f32]> {
        if batch >= self.batch || object >= self.max_boxes || self.truth_size == 0 {
            return Err(invalid("truth record index out of bounds"));
        }
        let start = batch
            .checked_mul(self.max_boxes)
            .and_then(|n| n.checked_add(object))
            .and_then(|n| n.checked_mul(self.truth_size))
            .ok_or_else(|| invalid("truth index overflows"))?;
        let end = start
            .checked_add(self.truth_size)
            .ok_or_else(|| invalid("truth index overflows"))?;
        self.truths
            .get(start..end)
            .ok_or_else(|| invalid("inconsistent truth storage"))
    }
}

/// Encode pure cuboid truth. Explicitly rejects 2D rows instead of dropping
/// supervision. Selection is deterministic caller order: take max_boxes first,
/// then remove subpixel/cropped hulls and pack survivors without refilling.
/// This mirrors source cap/filter ordering but does not reproduce its shuffle.
pub fn encode_yolo_3d_truths(
    labels: &[Vec<MixedLabel>],
    meta: &[Yolo3dImageMeta],
    spec: &Yolo3dSpec,
) -> Result<Yolo3dTruthBatch> {
    encode_truths(labels, meta, spec, false)
}

/// Encode combined truth with source ownership flags. Mosaic indicates an
/// already positioned crop; zero every field after class, including ownership,
/// matching source blend_truth_mosaic's box-only copying. Mixed mosaic rows are
/// therefore unsupervised by either head (source behavior). No random sampling.
pub fn encode_mixed_yolo_3d_truths(
    labels: &[Vec<MixedLabel>],
    meta: &[Yolo3dImageMeta],
    spec: &Yolo3dSpec,
) -> Result<Yolo3dTruthBatch> {
    encode_truths(labels, meta, spec, true)
}

fn encode_truths(
    labels: &[Vec<MixedLabel>],
    meta: &[Yolo3dImageMeta],
    spec: &Yolo3dSpec,
    mixed: bool,
) -> Result<Yolo3dTruthBatch> {
    if labels.is_empty()
        || labels.len() != meta.len()
        || !(1..=8).contains(&spec.keypoints)
        || spec.classes == 0
        || spec.classes > i32::MAX as usize
        || spec.max_boxes == 0
        || !spec.geometry.focal_35mm.is_finite()
        || spec.geometry.focal_35mm < 0.0
        || !matches!(spec.geometry.winding, -1 | 1)
    {
        return Err(invalid("invalid batch, spec or image metadata count"));
    }
    let truth_size = 11 + 3 * spec.keypoints + usize::from(mixed);
    let count = labels
        .len()
        .checked_mul(spec.max_boxes)
        .and_then(|n| n.checked_mul(truth_size))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| invalid("truth storage size overflows source dimensions"))?;
    // Validate public values even beyond the selection cap: errors must not
    // become silently accepted simply because the object was not selected.
    for (image, m) in labels.iter().zip(meta) {
        m.validate()?;
        for label in image {
            match label {
                MixedLabel::Cuboid(c) => {
                    if c.class_id >= spec.classes || c.vertices.len() != spec.keypoints {
                        return Err(invalid("cuboid class/keypoint count mismatch"));
                    }
                    c.hull()?;
                }
                MixedLabel::Box2d { label, .. } => {
                    if !mixed {
                        return Err(invalid("2D rows require encode_mixed_yolo_3d_truths"));
                    }
                    if label.class_id >= spec.classes
                        || [label.x, label.y, label.w, label.h]
                            .iter()
                            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
                    {
                        return Err(invalid("invalid normalized 2D label"));
                    }
                }
            }
        }
    }
    let mut truths = Vec::new();
    truths
        .try_reserve_exact(count)
        .map_err(|_| invalid("cannot allocate truth storage"))?;
    truths.resize(count, 0.0);
    let mut result = Yolo3dTruthBatch {
        truths,
        truth_size,
        max_boxes: spec.max_boxes,
        batch: labels.len(),
        counts: vec![0; labels.len()],
        mixed,
    };
    for (b, (image, m)) in labels.iter().zip(meta).enumerate() {
        for label in image.iter().take(spec.max_boxes) {
            // data.cpp:correct_boxes marks the (0,0) 2D center sentinel as
            // invalid even when its clipped box could otherwise have area.
            if matches!(label, MixedLabel::Box2d { label, .. } if label.x == 0.0 && label.y == 0.0)
            {
                continue;
            }
            let (bounds, class_id, track_id, cuboid) = match label {
                MixedLabel::Cuboid(c) => (c.hull()?, c.class_id, c.track_id, Some(c)),
                MixedLabel::Box2d { label, track_id } => (
                    [
                        (label.x - label.w / 2.0).clamp(0.0, 1.0),
                        (label.y - label.h / 2.0).clamp(0.0, 1.0),
                        (label.x + label.w / 2.0).clamp(0.0, 1.0),
                        (label.y + label.h / 2.0).clamp(0.0, 1.0),
                    ],
                    label.class_id,
                    *track_id,
                    None,
                ),
            };
            let transformed = m.transform_bounds(bounds)?;
            let w = transformed[2] - transformed[0];
            let h = transformed[3] - transformed[1];
            if w < 1.0 / m.network_width as f32 || h < 1.0 / m.network_height as f32 {
                continue;
            }
            let start = (b * spec.max_boxes + result.counts[b]) * truth_size;
            let row = &mut result.truths[start..start + truth_size];
            row[..6].copy_from_slice(&[
                (transformed[0] + transformed[2]) / 2.0,
                (transformed[1] + transformed[3]) / 2.0,
                w,
                h,
                class_id as f32,
                track_id as f32,
            ]);
            if let Some(c) = cuboid {
                let mut clipped = false;
                for (k, vertex) in c.vertices.iter().enumerate() {
                    let [u, v] = m.transform_point([vertex[0], vertex[1]])?;
                    clipped |= !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v);
                    row[11 + 3 * k..14 + 3 * k].copy_from_slice(&[
                        u.clamp(0.0, 1.0),
                        v.clamp(0.0, 1.0),
                        vertex[2],
                    ]);
                }
                if !clipped && !m.crop.mosaic && spec.geometry.focal_35mm > 0.0 {
                    let f_px = spec.geometry.focal_35mm / 43.27
                        * (m.original_width as f32).hypot(m.original_height as f32);
                    let px = (m.original_width as f32 / 2.0 - m.crop.pleft) / m.crop.swidth;
                    let intrinsics = [
                        f_px / m.crop.swidth,
                        f_px / m.crop.sheight,
                        if m.crop.flip { 1.0 - px } else { px },
                        (m.original_height as f32 / 2.0 - m.crop.ptop) / m.crop.sheight,
                        spec.geometry.winding as f32 * if m.crop.flip { -1.0 } else { 1.0 },
                    ];
                    if intrinsics.iter().any(|x| !x.is_finite())
                        || intrinsics[0] <= 0.0
                        || intrinsics[1] <= 0.0
                    {
                        return Err(invalid("intrinsics overflow or underflow"));
                    }
                    row[6..11].copy_from_slice(&intrinsics);
                }
            }
            if mixed {
                row[truth_size - 1] = if cuboid.is_some() { 2.0 } else { 1.0 };
            }
            if m.crop.mosaic {
                row[5..].fill(0.0);
            }
            result.counts[b] += 1;
        }
    }
    Ok(result)
}

impl Yolo3dImageMeta {
    fn validate(&self) -> Result<()> {
        if [
            self.original_width,
            self.original_height,
            self.network_width,
            self.network_height,
        ]
        .iter()
        .any(|&n| n == 0 || n > i32::MAX as usize)
            || [
                self.crop.pleft,
                self.crop.ptop,
                self.crop.swidth,
                self.crop.sheight,
            ]
            .iter()
            .any(|x| !x.is_finite())
            || self.crop.swidth <= 0.0
            || self.crop.sheight <= 0.0
        {
            return Err(invalid(
                "image/crop dimensions must be positive finite source dimensions",
            ));
        }
        Ok(())
    }

    fn transform_point(&self, [u, v]: [f32; 2]) -> Result<[f32; 2]> {
        let u = u * (self.original_width as f32 / self.crop.swidth)
            - self.crop.pleft / self.crop.swidth;
        let v = v * (self.original_height as f32 / self.crop.sheight)
            - self.crop.ptop / self.crop.sheight;
        let u = if self.crop.flip { 1.0 - u } else { u };
        if !u.is_finite() || !v.is_finite() {
            return Err(invalid("crop transform overflows"));
        }
        Ok([u, v])
    }

    fn transform_bounds(&self, [left, top, right, bottom]: [f32; 4]) -> Result<[f32; 4]> {
        let a = self.transform_point([left, top])?;
        let b = self.transform_point([right, bottom])?;
        let (left, right) = if self.crop.flip {
            (b[0], a[0])
        } else {
            (a[0], b[0])
        };
        Ok([
            left.clamp(0.0, 1.0),
            a[1].clamp(0.0, 1.0),
            right.clamp(0.0, 1.0),
            b[1].clamp(0.0, 1.0),
        ])
    }
}

#[cfg(test)]
mod tests;
