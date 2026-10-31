use super::*;
use crate::cfg::ConfigFile;
use crate::model::{TensorShape, Yolo3dSpec};
use crate::runtime::Tensor;

fn spec(options: &str) -> Yolo3dSpec {
    let cfg = ConfigFile::parse_str(
        "math.cfg",
        &format!("[yolo_3d]\nclasses=2\nnum=3\nmask=2,0\nanchors=2,3,5,7,11,13\n{options}"),
    )
    .unwrap();
    Yolo3dSpec::from_section(&cfg.sections[0], shape()).unwrap()
}

fn shape() -> TensorShape {
    TensorShape::new(3, 5, 62, 2)
}

#[test]
fn yolo_3d_math_layout_matches_source_across_batches_and_anchor_slots() {
    // depth_3d@9c9e0b4b yolo_3d.cpp:102-109. Hand-derived from source,
    // no compiled-C++ golden. This is not the complete decode A2 pin.
    let s = spec("");
    let layout = Yolo3dLayout::new(&s, shape()).unwrap();
    assert_eq!(layout.entries_per_anchor(), 31);
    assert_eq!(layout.storage_len(), 1860);
    assert_eq!(layout.entry_index(1, 0, 7, 2, 1).unwrap(), 1042);
    assert_eq!(layout.entry_index(1, 1, 30, 4, 2).unwrap(), 1859);
    let mut visited = vec![false; 1860];
    for b in 0..2 {
        for a in 0..2 {
            for e in 0..31 {
                for y in 0..5 {
                    for x in 0..3 {
                        let index = layout.entry_index(b, a, e, y, x).unwrap();
                        assert_eq!(index, b * 930 + a * 465 + e * 15 + y * 3 + x);
                        assert!(!visited[index]);
                        visited[index] = true;
                    }
                }
            }
        }
    }
    assert!(visited.into_iter().all(|x| x));
    // Anchor is a mask SLOT, not the global anchor ID 2 in the first slot.
    assert_eq!(layout.entry_index(0, 0, 0, 0, 0).unwrap(), 0);
    assert_eq!(layout.vertex_entry(0, 0).unwrap(), 7);
    assert_eq!(layout.vertex_entry(7, 2).unwrap(), 30);
}

#[test]
fn yolo_3d_math_layout_rejects_each_out_of_range_axis_and_overflow() {
    let s = spec("");
    let l = Yolo3dLayout::new(&s, shape()).unwrap();
    for args in [
        (2, 0, 0, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 0, 31, 0, 0),
        (0, 0, 0, 5, 0),
        (0, 0, 0, 0, 3),
    ] {
        assert!(
            l.entry_index(args.0, args.1, args.2, args.3, args.4)
                .is_err()
        );
    }
    assert!(l.vertex_entry(8, 0).is_err());
    assert!(l.vertex_entry(0, 3).is_err());
    for bad in [
        TensorShape::new(usize::MAX, 5, 62, 2),
        TensorShape::new(3, 5, 61, 2),
        TensorShape::new(0, 5, 62, 2),
        TensorShape::new(3, 5, 62, 0),
    ] {
        assert!(Yolo3dLayout::new(&s, bad).is_err());
    }
    let mut bad = s.clone();
    bad.keypoints = usize::MAX;
    assert!(Yolo3dLayout::new(&bad, shape()).is_err());
}

#[test]
fn yolo_3d_math_classic_activation_matches_hand_computed_planes() {
    // C++ yolo_3d.cpp:1501-1535: xy sigmoid+affine, wh raw,
    // objectness/classes sigmoid, interleaved u/v/visibility sigmoid.
    let s = spec("scale_x_y=1.6\n");
    let raw = Tensor::filled(shape(), 3.0_f32.ln());
    let original = raw.clone();
    let activated = activate_yolo_3d_output(&s, &raw).unwrap();
    assert_eq!(raw, original);
    for (channel, plane) in activated.data.chunks_exact(15).enumerate() {
        let expected = match channel % 31 {
            0 | 1 => 0.9,
            2 | 3 => 3.0_f32.ln(),
            _ => 0.75,
        };
        for &v in plane {
            assert!((v - expected).abs() < 1e-6, "channel {channel}: {v}");
        }
    }
}

#[test]
fn yolo_3d_math_new_coords_only_sigmoids_vertex_triplets() {
    let s = spec("scale_x_y=1.6\nnew_coords=1\n");
    let raw = Tensor::filled(shape(), 0.25);
    let activated = activate_yolo_3d_output(&s, &raw).unwrap();
    let gradient = backward_yolo_3d_activation(&s, &raw, &Tensor::filled(shape(), 1.0)).unwrap();
    for channel in 0..124 {
        let e = channel % 31;
        let expected = match e {
            0 | 1 => 0.1,
            2..=6 => 0.25,
            _ => 0.5621765,
        };
        let slope = match e {
            0 | 1 => 1.6,
            2..=6 => 1.0,
            _ => 0.24613409,
        };
        assert!((activated.data[channel * 15] - expected).abs() < 1e-6);
        assert!((gradient.data[channel * 15] - slope).abs() < 1e-6);
    }
}

#[test]
fn yolo_3d_math_activation_backward_is_exact_raw_chain_not_cpp_shortcut() {
    for options in ["scale_x_y=1.6", "scale_x_y=1.6\nnew_coords=1"] {
        let s = spec(options);
        let l = Yolo3dLayout::new(&s, shape()).unwrap();
        let raw = Tensor::from_data(
            shape(),
            (0..1860).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect(),
        )
        .unwrap();
        let seed = Tensor::from_data(
            shape(),
            (0..1860).map(|i| (i % 13) as f32 * 0.03 - 0.1).collect(),
        )
        .unwrap();
        let gradient = backward_yolo_3d_activation(&s, &raw, &seed).unwrap();
        for entry in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30] {
            let index = l.entry_index(1, 1, entry, 4, 2).unwrap();
            let score = |offset: f32| {
                let mut changed = raw.clone();
                changed.data[index] += offset;
                activate_yolo_3d_output(&s, &changed)
                    .unwrap()
                    .data
                    .iter()
                    .zip(&seed.data)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
            };
            let fd = (score(0.001) - score(-0.001)) / 0.002;
            assert!(
                (fd - f64::from(gradient.data[index])).abs() < 0.0001,
                "entry {entry}: {fd} vs {}",
                gradient.data[index]
            );
        }
    }
    let s = spec("scale_x_y=1.6");
    let gradient =
        backward_yolo_3d_activation(&s, &Tensor::zeros(shape()), &Tensor::filled(shape(), 1.0))
            .unwrap();
    assert!((gradient.data[0] - 0.4).abs() < 1e-6); // not C++'s omitted xy chain
    assert_eq!(gradient.data[30], 1.0); // raw width plane
    assert_eq!(gradient.data[60], 0.25); // objectness plane
}

#[test]
fn yolo_3d_math_activation_handles_extreme_finite_logits() {
    let s = spec("");
    for v in [-1000.0, 1000.0] {
        let raw = Tensor::filled(shape(), v);
        let output = activate_yolo_3d_output(&s, &raw).unwrap();
        let grad = backward_yolo_3d_activation(&s, &raw, &Tensor::filled(shape(), 1.0)).unwrap();
        assert!(output.data.iter().chain(&grad.data).all(|x| x.is_finite()));
        assert_eq!(output.data[0], if v < 0.0 { 0.0 } else { 1.0 });
        assert_eq!(grad.data[0], 0.0);
        assert_eq!(output.data[30], v);
        assert_eq!(grad.data[30], 1.0);
    }
}

#[test]
fn yolo_3d_math_activation_rejects_malformed_storage_nonfinite_and_metric_requests() {
    let s = spec("");
    let raw = Tensor::zeros(shape());
    let mut bad = raw.clone();
    bad.data.pop();
    assert!(activate_yolo_3d_output(&s, &bad).is_err());
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = raw.clone();
        bad.data[17] = v;
        assert!(activate_yolo_3d_output(&s, &bad).is_err());
        assert!(backward_yolo_3d_activation(&s, &raw, &bad).is_err());
    }
    let mut wrong = raw.clone();
    wrong.shape.width = 5;
    wrong.shape.height = 3;
    assert!(backward_yolo_3d_activation(&s, &raw, &wrong).is_err());
    let mut short = raw.clone();
    short.data.clear();
    assert!(backward_yolo_3d_activation(&s, &raw, &short).is_err());
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut bad = s.clone();
        bad.scale_x_y = scale;
        assert!(activate_yolo_3d_output(&bad, &raw).is_err());
    }
    let mut bad = s.clone();
    bad.use_3d = true;
    assert!(
        activate_yolo_3d_output(&bad, &raw)
            .unwrap_err()
            .to_string()
            .contains("metric")
    );
    let mut bad = s.clone();
    bad.classes = 0;
    assert!(Yolo3dLayout::new(&bad, shape()).is_err());
    let mut bad = s.clone();
    bad.keypoints = 0;
    assert!(Yolo3dLayout::new(&bad, shape()).is_err());
}

#[test]
fn yolo_3d_math_activation_reports_forward_and_seed_overflow() {
    let s = spec("new_coords=1\nscale_x_y=2");
    let raw = Tensor::filled(shape(), f32::MAX);
    assert!(activate_yolo_3d_output(&s, &raw).is_err());
    assert!(backward_yolo_3d_activation(&s, &raw, &Tensor::filled(shape(), 1.0)).is_err());
    assert!(
        backward_yolo_3d_activation(
            &s,
            &Tensor::zeros(shape()),
            &Tensor::filled(shape(), f32::MAX)
        )
        .is_err()
    );
}

fn loss_fixture(options: &str) -> (Yolo3dSpec, Tensor, crate::cuboid_labels::Yolo3dTruthBatch) {
    let shape = TensorShape::new(1, 1, 9, 1);
    let cfg = ConfigFile::parse_str("loss.cfg", &format!("[yolo_3d]\nclasses=1\nkeypoints=1\nnum=1\nmask=0\nanchors=1,1\nmax=1\niou_loss=mse\n{options}")).unwrap();
    let spec = Yolo3dSpec::from_section(&cfg.sections[0], shape).unwrap();
    let mut row = vec![0.0; 14];
    row[0..5].copy_from_slice(&[0.5, 0.5, 0.5, 0.5, 0.0]);
    (
        spec,
        Tensor::zeros(shape),
        crate::cuboid_labels::Yolo3dTruthBatch {
            truths: row,
            truth_size: 14,
            max_boxes: 1,
            batch: 1,
            counts: vec![1],
            mixed: false,
        },
    )
}

#[test]
fn yolo_3d_projected_loss_hand_computed() {
    let (s, raw, truth) = loss_fixture("");
    let r = yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
        .unwrap();
    assert!((r.box_loss - 0.0).abs() < 1e-6);
    assert!((r.objectness_loss - 0.125 * s.obj_normalizer).abs() < 1e-6);
    assert!((r.class_loss - 0.125).abs() < 1e-6);
    assert!((r.seeds.data[4] - 0.125 * s.obj_normalizer).abs() < 1e-6);
}

#[test]
fn yolo_3d_ignore_threshold_masks_objectness_hand_case() {
    let (mut s, raw, truth) = loss_fixture("");
    s.mask = vec![0];
    s.num = 2;
    s.anchors = vec![1.0, 1.0, 2.0, 2.0];
    let mut truth = truth;
    truth.truths[2] = 1.0;
    truth.truths[3] = 1.0;
    s.ignore_thresh = 0.2;
    let r = yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
        .unwrap();
    assert_eq!(r.objectness_loss, 0.0);
    assert_eq!(r.seeds.data[4], 0.0);
}

#[test]
fn yolo_3d_cpp_mode_reproduces_source_deltas() {
    // depth_3d yolo_3d.cpp:430-446, 546-566, 1236-1249; hand calculated, no C++ golden.
    let (s, mut raw, truth) = loss_fixture("");
    raw.data[0] = 3.0_f32.ln();
    raw.data[2] = 0.2;
    let r = yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Cpp).unwrap();
    assert!((r.seeds.data[0] - (1.75 * s.iou_normalizer * -0.25)).abs() < 1e-6);
    assert!((r.seeds.data[2] - (1.75 * s.iou_normalizer * -0.2)).abs() < 1e-6);
    assert!((r.seeds.data[4] - 0.5 * s.obj_normalizer).abs() < 1e-6);
    assert!((r.seeds.data[5] - 0.5).abs() < 1e-6);
}

#[test]
fn yolo_3d_projected_seed_matches_finite_difference() {
    let (s, mut raw, truth) = loss_fixture("");
    raw.data[0] = 0.2;
    raw.data[2] = 0.1;
    raw.data[4] = 0.3;
    raw.data[5] = -0.2;
    let r = yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
        .unwrap();
    for i in 0..raw.data.len() {
        let mut p = raw.clone();
        let mut m = raw.clone();
        p.data[i] += 0.001;
        m.data[i] -= 0.001;
        let f = |t: &Tensor| {
            yolo_3d_projected_loss(&s, t, &truth, 2, 2, crate::runtime::GradMode::Analytical)
                .unwrap()
                .loss()
        };
        let fd = (f(&p) - f(&m)) / 0.002;
        assert!(
            (fd + r.seeds.data[i]).abs() < 0.0001,
            "{i}: {fd} {}",
            r.seeds.data[i]
        );
    }
}

#[test]
fn yolo_3d_projected_loss_rejects_invalid_and_unsupported_inputs() {
    let (mut s, raw, mut truth) = loss_fixture("");
    truth.truths.pop();
    assert!(
        yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
            .is_err()
    );
    truth.truths.push(0.0);
    s.focal_loss = true;
    assert!(
        yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
            .is_err()
    );
    s.focal_loss = false;
    s.iou_loss = crate::model::YoloIouLoss::Giou;
    assert!(
        yolo_3d_projected_loss(&s, &raw, &truth, 2, 2, crate::runtime::GradMode::Analytical)
            .is_err()
    );
}
