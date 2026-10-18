//! Source-backed configuration preparation for the Stage-1 cuboid head.
//! This does not enable network dispatch: the layer remains unsupported until
//! its loss, decode and runtime contracts are implemented.

use super::{TensorShape, YoloIouLoss};
use crate::cfg::RawSection;
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Yolo3dVertexLoss {
    SmoothL1,
    Wing,
}

/// Intrinsics used by truth encoding, plus staged depth/metric loss settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Yolo3dGeometrySpec {
    pub focal_35mm: f32,
    pub winding: i32,
    pub depth_layer: Option<usize>,
    pub tilt_normalizer: f32,
    pub ortho_normalizer: f32,
    pub oks_normalizer: f32,
    pub oks_kappa: f32,
    pub tilt_min_face_width: f32,
    pub geo_burn_in: usize,
}

/// Configuration only; constructing this value does not enable a runtime head.
///
/// Defaults are hand-derived from `depth_3d` at `9c9e0b4b`,
/// `src-lib/darknet_cfg.cpp:2556-2697`, no compiled-C++ golden.
/// Unlike C++, invalid sizes, nonfinite values, malformed arrays and unknown
/// keys are errors. Zero loss normalizers remain valid to disable terms.
#[derive(Clone, Debug, PartialEq)]
pub struct Yolo3dSpec {
    pub classes: usize,
    pub num: usize,
    pub mask: Vec<usize>,
    pub anchors: Vec<f32>,
    pub keypoints: usize,
    pub max_boxes: usize,
    pub use_3d: bool,
    pub max_delta: f32,
    pub counters_per_class: Vec<usize>,
    pub label_smooth_eps: f32,
    pub scale_x_y: f32,
    pub objectness_smooth: bool,
    pub new_coords: bool,
    pub iou_normalizer: f32,
    pub obj_normalizer: f32,
    pub cls_normalizer: f32,
    pub delta_normalizer: f32,
    pub loss_weight: f32,
    pub iou_loss: YoloIouLoss,
    pub iou_thresh_kind: YoloIouLoss,
    /// Parsed metadata: the source public decoder uses AABB NMS regardless.
    pub nms_kind: String,
    pub beta_nms: f32,
    /// Loader metadata; no augmentation sampler is enabled by this helper.
    pub jitter: f32,
    pub resize: f32,
    pub focal_loss: bool,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub iou_thresh: f32,
    pub random: f32,
    pub loss_kind: Yolo3dVertexLoss,
    pub wing_w: f32,
    pub vtx_normalizer: f32,
    pub vis_normalizer: f32,
    pub geom_normalizer: f32,
    pub use_geometry: bool,
    pub edge_weight: [f32; 3],
    pub loss_3d_normalizer: f32,
    pub shape_normalizer: f32,
    /// Unresolved filename, not a loaded class mapping. Network integration
    /// must resolve this relative to the cfg as it does for other YOLO heads.
    pub map_file: Option<String>,
    pub geometry: Yolo3dGeometrySpec,
}

fn invalid(section: &RawSection, message: impl std::fmt::Display) -> Error {
    Error::InvalidConfig(format!(
        "[{}] at line {}: {message}",
        section.name, section.line
    ))
}

impl Yolo3dSpec {
    /// Parse the projected-only config contract. Geometry/metric consumption
    /// remains rejected until the second CPU implementation increment.
    pub fn from_section(section: &RawSection, input: TensorShape) -> Result<Self> {
        if !matches!(
            section.name.as_str(),
            "yolo_3d" | "yolo3d" | "cuboid" | "cuboid_yolo"
        ) {
            return Err(invalid(section, "expected a yolo_3d section"));
        }
        const KEYS: &[&str] = &[
            "classes",
            "num",
            "max",
            "keypoints",
            "mask",
            "anchors",
            "use_3d",
            "max_delta",
            "counters_per_class",
            "label_smooth_eps",
            "scale_x_y",
            "objectness_smooth",
            "new_coords",
            "iou_normalizer",
            "obj_normalizer",
            "cls_normalizer",
            "delta_normalizer",
            "loss_weight",
            "iou_loss",
            "iou_thresh_kind",
            "beta_nms",
            "nms_kind",
            "jitter",
            "resize",
            "focal_loss",
            "ignore_thresh",
            "truth_thresh",
            "iou_thresh",
            "random",
            "loss_kind",
            "wing_w",
            "vtx_normalizer",
            "vis_normalizer",
            "geom_normalizer",
            "use_geometry",
            "loss_3d_normalizer",
            "shape_normalizer",
            "edge_weight",
            "focal_35mm",
            "winding",
            "depth_layer",
            "tilt_normalizer",
            "ortho_normalizer",
            "oks_normalizer",
            "geo_burn_in",
            "oks_kappa",
            "tilt_min_face_width",
            "map",
        ];
        for key in section.values.keys() {
            if !KEYS.contains(&key.as_str()) {
                return Err(invalid(section, format!("unknown key {key:?}")));
            }
        }
        let positive_size = |key, default| -> Result<usize> {
            let value = section.get_usize(key, default)?;
            if value == 0 || value > i32::MAX as usize {
                return Err(invalid(section, format!("{key} must be in 1..=i32::MAX")));
            }
            Ok(value)
        };
        let float = |key, default, min, strict| -> Result<f32> {
            let value = section.get_f32(key, default)?;
            if !value.is_finite() || value < min || (strict && value == min) {
                return Err(invalid(
                    section,
                    format!(
                        "{key} must be finite and {} {min}",
                        if strict { ">" } else { ">=" }
                    ),
                ));
            }
            Ok(value)
        };
        let unit = |key, default| -> Result<f32> {
            let value = float(key, default, 0.0, false)?;
            if value > 1.0 {
                return Err(invalid(section, format!("{key} must be <= 1")));
            }
            Ok(value)
        };
        let classes = positive_size("classes", 1)?;
        let num = positive_size("num", 1)?;
        let max_boxes = positive_size("max", 200)?;
        let keypoints = positive_size("keypoints", 8)?;
        if keypoints > 8 {
            return Err(invalid(section, "keypoints must be in 1..=8"));
        }
        let anchor_count = num
            .checked_mul(2)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| invalid(section, "anchor count overflows source dimensions"))?;
        let use_3d = section.get_bool_int("use_3d", false)?;
        let depth_layer = section.get_isize("depth_layer", -1)?;
        if depth_layer < -1 {
            return Err(invalid(
                section,
                "depth_layer must be -1 or an absolute layer index",
            ));
        }
        let geometry = Yolo3dGeometrySpec {
            focal_35mm: float("focal_35mm", 0.0, 0.0, false)?,
            winding: match section.get_i64("winding", 1)? {
                value @ (-1 | 1) => value as i32,
                _ => return Err(invalid(section, "winding must be 1 or -1")),
            },
            depth_layer: if depth_layer == -1 {
                None
            } else {
                Some(depth_layer as usize)
            },
            tilt_normalizer: float("tilt_normalizer", 0.0, 0.0, false)?,
            ortho_normalizer: float("ortho_normalizer", 0.0, 0.0, false)?,
            oks_normalizer: float("oks_normalizer", 0.0, 0.0, false)?,
            oks_kappa: float("oks_kappa", 0.0849, 0.0, true)?,
            tilt_min_face_width: float("tilt_min_face_width", 0.1, 0.0, false)?,
            geo_burn_in: section.get_usize("geo_burn_in", 0)?,
        };
        let loss_3d_normalizer = float(
            "loss_3d_normalizer",
            if use_3d { 1.0 } else { 0.0 },
            0.0,
            false,
        )?;
        let shape_normalizer = float(
            "shape_normalizer",
            if use_3d { 0.5 } else { 0.0 },
            0.0,
            false,
        )?;
        for (key, requested) in [
            ("use_3d", use_3d),
            ("depth_layer", geometry.depth_layer.is_some()),
            ("tilt_normalizer", geometry.tilt_normalizer != 0.0),
            ("ortho_normalizer", geometry.ortho_normalizer != 0.0),
            ("tilt_min_face_width", geometry.tilt_min_face_width != 0.1),
            ("geo_burn_in", geometry.geo_burn_in != 0),
            ("loss_3d_normalizer", loss_3d_normalizer != 0.0),
            ("shape_normalizer", shape_normalizer != 0.0),
        ] {
            if requested {
                return Err(invalid(
                    section,
                    format!("{key}: not implemented until the depth/metric CPU increment"),
                ));
            }
        }
        let mut mask = section.get_usize_list("mask")?;
        // Reject dimensions/truth storage overflow before allocating defaults.
        let entries = 5usize
            .checked_add(classes)
            .and_then(|n| n.checked_add(3 * keypoints))
            .ok_or_else(|| invalid(section, "entry count overflows"))?;
        let active = if mask.is_empty() { num } else { mask.len() };
        let channels = active
            .checked_mul(entries)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| invalid(section, "channel count overflows source dimensions"))?;
        let truth_count = max_boxes
            .checked_mul(11 + 3 * keypoints)
            .and_then(|n| n.checked_mul(input.batch))
            .filter(|&n| n <= i32::MAX as usize);
        if truth_count.is_none() {
            return Err(invalid(
                section,
                "truth storage overflows source dimensions",
            ));
        }
        validate_tensor_shape(input, channels).map_err(|error| invalid(section, error))?;
        if mask.is_empty() {
            mask = (0..num).collect();
        }
        let mut seen = std::collections::HashSet::new();
        for &index in &mask {
            if index >= num || !seen.insert(index) {
                return Err(invalid(
                    section,
                    "mask indices must be unique and less than num",
                ));
            }
        }
        let anchors = if section.get("anchors").is_none() {
            vec![0.5; anchor_count]
        } else {
            section.get_f32_list("anchors")?
        };
        if anchors.len() != anchor_count || anchors.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(invalid(
                section,
                "anchors must contain exactly 2*num finite positive values",
            ));
        }
        let weights = section.get_f32_list("edge_weight")?;
        let edge_weight = if section.get("edge_weight").is_none() {
            [1.0; 3]
        } else {
            weights
                .try_into()
                .map_err(|_| invalid(section, "edge_weight requires exactly three values"))?
        };
        if edge_weight.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(invalid(
                section,
                "edge_weight values must be finite and nonnegative",
            ));
        }
        let counters_per_class = section.get_usize_list("counters_per_class")?;
        if !counters_per_class.is_empty() && counters_per_class.len() != classes {
            return Err(invalid(
                section,
                "counters_per_class requires one count per class",
            ));
        }
        let nms_kind = section.get_str("nms_kind", "default");
        if !matches!(
            nms_kind.as_str(),
            "default" | "greedynms" | "diounms" | "cornersnms"
        ) {
            return Err(invalid(
                section,
                format!("unsupported nms_kind {nms_kind:?}"),
            ));
        }
        let loss_kind = match section.get_str("loss_kind", "smooth_l1").as_str() {
            "smooth_l1" | "smoothl1" | "l1" => Yolo3dVertexLoss::SmoothL1,
            "wing" => Yolo3dVertexLoss::Wing,
            other => return Err(invalid(section, format!("unsupported loss_kind {other:?}"))),
        };
        let map = section.get_str("map", "");
        Ok(Self {
            classes,
            num,
            mask,
            anchors,
            keypoints,
            max_boxes,
            use_3d,
            max_delta: float("max_delta", 10.0, 0.0, true)?,
            counters_per_class,
            label_smooth_eps: unit("label_smooth_eps", 0.0)?,
            scale_x_y: float("scale_x_y", 1.0, 0.0, true)?,
            objectness_smooth: section.get_bool_int("objectness_smooth", false)?,
            new_coords: section.get_bool_int("new_coords", false)?,
            iou_normalizer: float("iou_normalizer", 0.75, 0.0, false)?,
            obj_normalizer: float("obj_normalizer", 1.0, 0.0, false)?,
            cls_normalizer: float("cls_normalizer", 1.0, 0.0, false)?,
            delta_normalizer: float("delta_normalizer", 1.0, 0.0, false)?,
            loss_weight: float("loss_weight", 1.0, 0.0, true)?,
            iou_loss: YoloIouLoss::parse(&section.get_str("iou_loss", "mse"))?,
            iou_thresh_kind: YoloIouLoss::parse(&section.get_str("iou_thresh_kind", "iou"))?,
            nms_kind,
            beta_nms: float("beta_nms", 0.6, 0.0, false)?,
            jitter: unit("jitter", 0.2)?,
            resize: float("resize", 1.0, 0.0, true)?,
            focal_loss: section.get_bool_int("focal_loss", false)?,
            ignore_thresh: unit("ignore_thresh", 0.5)?,
            truth_thresh: unit("truth_thresh", 1.0)?,
            iou_thresh: unit("iou_thresh", 1.0)?,
            random: float("random", 0.0, 0.0, false)?,
            loss_kind,
            wing_w: float("wing_w", 0.02, 0.0, true)?,
            vtx_normalizer: float("vtx_normalizer", 2.0, 0.0, false)?,
            vis_normalizer: float("vis_normalizer", 0.5, 0.0, false)?,
            geom_normalizer: float("geom_normalizer", 0.5, 0.0, false)?,
            use_geometry: section.get_bool_int("use_geometry", true)?,
            edge_weight,
            loss_3d_normalizer,
            shape_normalizer,
            map_file: if map.is_empty() { None } else { Some(map) },
            geometry,
        })
    }

    /// The channel-major anchor block has box/objectness, classes and K
    /// (u,v,visibility) triplets, followed by seven optional metric entries.
    pub fn entries_per_anchor(&self) -> Result<usize> {
        self.keypoints
            .checked_mul(3)
            .and_then(|n| n.checked_add(self.classes))
            .and_then(|n| n.checked_add(5))
            .and_then(|n| n.checked_add(if self.use_3d { 7 } else { 0 }))
            .ok_or_else(|| Error::InvalidConfig("[yolo_3d] entry count overflows".into()))
    }

    /// Validate the identity output shape without allocating runtime tensors.
    pub fn validate_shape(&self, input: TensorShape) -> Result<TensorShape> {
        let channels = self
            .mask
            .len()
            .checked_mul(self.entries_per_anchor()?)
            .ok_or_else(|| Error::InvalidConfig("[yolo_3d] channel count overflows".into()))?;
        validate_tensor_shape(input, channels)?;
        Ok(input)
    }
}

fn validate_tensor_shape(input: TensorShape, channels: usize) -> Result<()> {
    let elements = input
        .width
        .checked_mul(input.height)
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul(input.batch));
    if channels == 0
        || input.width == 0
        || input.height == 0
        || input.batch == 0
        || input.channels != channels
        || elements.is_none_or(|n| n > i32::MAX as usize)
    {
        return Err(Error::InvalidConfig(format!(
            "[yolo_3d] expected a nonempty input with {channels} channels and bounded tensor storage; got {input}, batch {}",
            input.batch
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::ConfigFile;
    use crate::model::{Network, TensorShape, YoloIouLoss};

    fn parse(text: &str, channels: usize) -> crate::error::Result<Yolo3dSpec> {
        let cfg = ConfigFile::parse_str("yolo3d.cfg", &format!("[yolo_3d]\n{text}"))?;
        Yolo3dSpec::from_section(&cfg.sections[0], TensorShape::new(3, 5, channels, 2))
    }

    #[test]
    fn yolo_3d_cfg_defaults_match_source() {
        // depth_3d:darknet_cfg.cpp:2556-2697 and yolo_3d.cpp:1275-1394.
        // Hand-derived from source, no compiled-C++ golden.
        let s = parse("", 30).unwrap();
        assert_eq!((s.classes, s.num, s.keypoints, s.max_boxes), (1, 1, 8, 200));
        assert_eq!(s.mask, [0]);
        assert_eq!(s.anchors, [0.5, 0.5]);
        assert_eq!(s.entries_per_anchor().unwrap(), 30);
        assert_eq!(s.loss_kind, Yolo3dVertexLoss::SmoothL1);
        assert_eq!(s.edge_weight, [1.0; 3]);
        assert_eq!(
            (
                s.wing_w,
                s.vtx_normalizer,
                s.vis_normalizer,
                s.geom_normalizer
            ),
            (0.02, 2.0, 0.5, 0.5)
        );
        assert!(s.use_geometry);
        assert!(!s.use_3d);
        assert_eq!((s.loss_3d_normalizer, s.shape_normalizer), (0.0, 0.0));
        assert_eq!(
            (s.max_delta, s.scale_x_y, s.iou_normalizer),
            (10.0, 1.0, 0.75)
        );
        assert_eq!(
            (
                s.obj_normalizer,
                s.cls_normalizer,
                s.delta_normalizer,
                s.loss_weight
            ),
            (1.0, 1.0, 1.0, 1.0)
        );
        assert_eq!(
            (s.ignore_thresh, s.truth_thresh, s.iou_thresh),
            (0.5, 1.0, 1.0)
        );
        assert_eq!(
            (s.iou_loss, s.iou_thresh_kind),
            (YoloIouLoss::Mse, YoloIouLoss::Iou)
        );
        assert_eq!(
            (s.jitter, s.resize, s.random, s.beta_nms),
            (0.2, 1.0, 0.0, 0.6)
        );
        assert_eq!(s.nms_kind, "default");
        assert!(s.counters_per_class.is_empty());
        assert!(s.map_file.is_none());
        assert_eq!(s.geometry.focal_35mm, 0.0);
        assert_eq!(s.geometry.winding, 1);
        assert_eq!(s.geometry.depth_layer, None);
        assert_eq!(
            (
                s.geometry.tilt_normalizer,
                s.geometry.ortho_normalizer,
                s.geometry.oks_normalizer
            ),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(s.geometry.oks_kappa, 0.0849);
        assert_eq!(s.geometry.tilt_min_face_width, 0.1);
        assert_eq!(s.geometry.geo_burn_in, 0);
    }

    #[test]
    fn yolo_3d_cfg_aliases_follow_existing_section_normalization() {
        for alias in ["yolo_3d", "yolo3d", "cuboid", "cuboid_yolo", "YOLO_3D"] {
            let cfg = ConfigFile::parse_str("alias.cfg", &format!("[{alias}]")).unwrap();
            assert!(
                Yolo3dSpec::from_section(&cfg.sections[0], TensorShape::new(3, 5, 30, 2)).is_ok()
            );
        }
        let cfg = ConfigFile::parse_str("bad.cfg", "[yolo]").unwrap();
        assert!(Yolo3dSpec::from_section(&cfg.sections[0], TensorShape::new(3, 5, 30, 2)).is_err());
    }

    #[test]
    fn yolo_3d_cfg_projected_channels_and_identity_shape() {
        let s = parse(
            "classes=2\nnum=3\nmask=2,0\nanchors=1,2,3,4,5,6\nkeypoints=4",
            38,
        )
        .unwrap();
        assert_eq!(s.entries_per_anchor().unwrap(), 19);
        assert_eq!(s.mask, [2, 0]);
        let shape = TensorShape::new(3, 5, 38, 2);
        assert_eq!(s.validate_shape(shape).unwrap(), shape);
        assert!(s.validate_shape(TensorShape::new(3, 5, 37, 2)).is_err());
        for shape in [
            TensorShape::new(0, 5, 38, 2),
            TensorShape::new(3, 0, 38, 2),
            TensorShape::new(3, 5, 38, 0),
            TensorShape::new(usize::MAX, 5, 38, 2),
        ] {
            assert!(s.validate_shape(shape).is_err());
        }
        assert!(parse("", 29).is_err());
    }

    #[test]
    fn yolo_3d_cfg_rejects_unknown_key_with_section_and_line() {
        let error = parse("yolo_3d_keypoints=8", 30).unwrap_err().to_string();
        assert!(
            error.contains("yolo_3d_keypoints")
                && error.contains("yolo_3d")
                && error.contains("line 1"),
            "{error}"
        );
    }

    #[test]
    fn yolo_3d_cfg_rejects_invalid_values_without_source_fallbacks() {
        // Rust-only defensive validation where C++ falls back or leaves unchecked.
        for text in [
            "classes=0",
            "num=0",
            "max=0",
            "keypoints=0",
            "keypoints=9",
            "classes=-1",
            "mask=1",
            "num=2\nmask=0,0",
            "anchors=1",
            "anchors=1,2,3",
            "anchors=0,1",
            "anchors=NaN,1",
            "anchors=inf,1",
            "winding=0",
            "oks_kappa=0",
            "oks_kappa=NaN",
            "scale_x_y=0",
            "wing_w=0",
            "vtx_normalizer=-1",
            "obj_normalizer=NaN",
            "label_smooth_eps=1.1",
            "ignore_thresh=-1",
            "edge_weight=1,2",
            "edge_weight=1,-1,2",
            "counters_per_class=1,2",
            "counters_per_class=-1",
            "loss_kind=bogus",
            "iou_loss=bogus",
            "nms_kind=obb",
            "resize=0",
        ] {
            assert!(parse(text, 30).is_err(), "accepted {text}");
        }
    }

    #[test]
    fn yolo_3d_cfg_increment_two_keys_rejected_until_implemented() {
        for text in [
            "use_3d=1",
            "use_3d=-1",
            "depth_layer=0",
            "tilt_normalizer=1",
            "ortho_normalizer=1",
            "tilt_min_face_width=0.2",
            "geo_burn_in=1",
            "loss_3d_normalizer=1",
            "shape_normalizer=0.5",
        ] {
            let error = parse(text, 30).unwrap_err().to_string();
            assert!(error.contains("not implemented"), "{text}: {error}");
        }
        assert!(parse("use_3d=0\ndepth_layer=-1\ntilt_normalizer=0\northo_normalizer=0\ntilt_min_face_width=0.1\ngeo_burn_in=0\nloss_3d_normalizer=0\nshape_normalizer=0", 30).is_ok());
    }

    #[test]
    fn yolo_3d_cfg_truth_intrinsics_and_projected_options_are_not_staged_out() {
        let s = parse("focal_35mm=24\nwinding=-1\noks_normalizer=0.3\noks_kappa=0.1\nloss_kind=wing\nedge_weight=1,2,3\ncounters_per_class=7\nmap=classes.map\nrandom=0.25", 30).unwrap();
        assert_eq!(s.geometry.focal_35mm, 24.0);
        assert_eq!(s.geometry.winding, -1);
        assert_eq!(s.geometry.oks_normalizer, 0.3);
        assert_eq!(s.loss_kind, Yolo3dVertexLoss::Wing);
        assert_eq!(s.edge_weight, [1.0, 2.0, 3.0]);
        assert_eq!(s.counters_per_class, [7]);
        assert_eq!(s.map_file.as_deref(), Some("classes.map"));
        assert_eq!(s.random, 0.25); // source reads float, not integer.
        for loss in ["l1", "smoothl1", "smooth_l1"] {
            assert_eq!(
                parse(&format!("loss_kind={loss}"), 30).unwrap().loss_kind,
                Yolo3dVertexLoss::SmoothL1
            );
        }
    }

    #[test]
    fn yolo_3d_cfg_overflow_is_an_error_before_allocation() {
        for text in [
            format!("classes={}", usize::MAX),
            format!("num={}", usize::MAX),
            format!("max={}", usize::MAX),
        ] {
            assert!(parse(&text, 30).is_err());
        }
        let mut s = parse("", 30).unwrap();
        s.classes = usize::MAX;
        assert!(s.entries_per_anchor().is_err());
        s.classes = 1;
        s.keypoints = usize::MAX;
        assert!(s.entries_per_anchor().is_err());
    }

    #[test]
    fn yolo_3d_cfg_numeric_options_reject_nonfinite_values() {
        for key in [
            "max_delta",
            "label_smooth_eps",
            "scale_x_y",
            "iou_normalizer",
            "obj_normalizer",
            "cls_normalizer",
            "delta_normalizer",
            "loss_weight",
            "beta_nms",
            "jitter",
            "resize",
            "ignore_thresh",
            "truth_thresh",
            "iou_thresh",
            "random",
            "wing_w",
            "vtx_normalizer",
            "vis_normalizer",
            "geom_normalizer",
            "loss_3d_normalizer",
            "shape_normalizer",
            "focal_35mm",
            "tilt_normalizer",
            "ortho_normalizer",
            "oks_normalizer",
            "oks_kappa",
            "tilt_min_face_width",
        ] {
            for value in ["NaN", "inf", "-inf"] {
                assert!(
                    parse(&format!("{key}={value}"), 30).is_err(),
                    "{key}={value}"
                );
            }
        }
        for text in ["edge_weight=1,NaN,1", "depth_layer=-2", "geo_burn_in=-1"] {
            assert!(parse(text, 30).is_err(), "{text}");
        }
    }

    #[test]
    fn yolo_3d_cfg_retains_projected_and_loader_overrides() {
        let s = parse(
            concat!(
                "max_delta=4\nlabel_smooth_eps=0.2\nscale_x_y=1.2\n",
                "objectness_smooth=1\nnew_coords=1\nfocal_loss=1\nuse_geometry=0\n",
                "iou_normalizer=0.2\nobj_normalizer=0.3\ncls_normalizer=0.4\n",
                "delta_normalizer=0.5\nloss_weight=2\niou_loss=giou\niou_thresh_kind=diou\n",
                "nms_kind=diounms\nbeta_nms=0.7\njitter=0.1\nresize=1.5\n",
                "ignore_thresh=0.4\ntruth_thresh=0.9\niou_thresh=0.8\n",
                "wing_w=0.03\nvtx_normalizer=3\nvis_normalizer=0.7\ngeom_normalizer=0\n"
            ),
            30,
        )
        .unwrap();
        assert_eq!(
            (s.max_delta, s.label_smooth_eps, s.scale_x_y),
            (4.0, 0.2, 1.2)
        );
        assert!(s.objectness_smooth && s.new_coords && s.focal_loss);
        assert!(!s.use_geometry);
        assert_eq!(
            (s.iou_normalizer, s.obj_normalizer, s.cls_normalizer),
            (0.2, 0.3, 0.4)
        );
        assert_eq!((s.delta_normalizer, s.loss_weight), (0.5, 2.0));
        assert_eq!(
            (s.iou_loss, s.iou_thresh_kind),
            (YoloIouLoss::Giou, YoloIouLoss::Diou)
        );
        assert_eq!(s.nms_kind, "diounms");
        assert_eq!((s.beta_nms, s.jitter, s.resize), (0.7, 0.1, 1.5));
        assert_eq!(
            (s.ignore_thresh, s.truth_thresh, s.iou_thresh),
            (0.4, 0.9, 0.8)
        );
        assert_eq!(
            (
                s.wing_w,
                s.vtx_normalizer,
                s.vis_normalizer,
                s.geom_normalizer
            ),
            (0.03, 3.0, 0.7, 0.0)
        );
        assert!(parse("iou_normalizer=0\nobj_normalizer=0\ncls_normalizer=0\ndelta_normalizer=0\nvtx_normalizer=0\nvis_normalizer=0\ngeom_normalizer=0", 30).is_ok());
    }

    #[test]
    fn yolo_3d_cfg_partial_and_empty_arrays_do_not_silently_default() {
        for text in [
            "anchors=",
            "edge_weight=",
            "mask=-1",
            "mask=0,1",
            "anchors=-1,2",
        ] {
            assert!(parse(text, 30).is_err(), "{text}");
        }
        // C++ treats an empty mask as identity, including an explicit empty list.
        assert_eq!(parse("mask=", 30).unwrap().mask, [0]);
        for k in 1..=8 {
            assert_eq!(
                parse(&format!("keypoints={k}"), 6 + 3 * k)
                    .unwrap()
                    .keypoints,
                k
            );
        }
    }

    #[test]
    fn yolo_3d_cfg_preparation_does_not_enable_incomplete_runtime() {
        let cfg = ConfigFile::parse_str(
            "unwired.cfg",
            "[net]\nwidth=3\nheight=5\nchannels=30\n[yolo_3d]",
        )
        .unwrap();
        assert!(matches!(
            Network::from_config(cfg),
            Err(Error::UnsupportedLayer { layer_index: 0, name }) if name == "yolo_3d"
        ));
    }
}
