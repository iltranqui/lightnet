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
