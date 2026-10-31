use super::*;
use crate::cfg::ConfigFile;
use crate::model::{TensorShape, Yolo3dSpec};

fn spec() -> Yolo3dSpec {
    let cfg = ConfigFile::parse_str(
        "t.cfg",
        "[yolo_3d]\nclasses=3\nkeypoints=4\nmax=3\nfocal_35mm=24",
    )
    .unwrap();
    Yolo3dSpec::from_section(&cfg.sections[0], TensorShape::new(3, 5, 20, 2)).unwrap()
}

fn cuboid() -> CuboidLabel {
    CuboidLabel {
        class_id: 2,
        track_id: 123,
        vertices: vec![
            [0.2, 0.3, 1.0],
            [0.6, 0.3, -1.0],
            [0.6, 0.9, 0.7],
            [0.2, 0.9, 0.0],
        ],
    }
}

fn meta() -> Yolo3dImageMeta {
    Yolo3dImageMeta {
        original_width: 1280,
        original_height: 960,
        network_width: 640,
        network_height: 480,
        crop: Yolo3dCrop {
            pleft: 0.0,
            ptop: 0.0,
            swidth: 1280.0,
            sheight: 960.0,
            flip: false,
            mosaic: false,
        },
    }
}

fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 2e-6, "{a} != {b}");
}

#[test]
fn yolo_3d_label_row_parses_source_layout() {
    let values = parse_cuboid_labels(
        "2 0.2 0.3 1 0.6 0.3 -1 0.6 0.9 0.7 0.2 0.9 0",
        "labels/a.txt",
        Some(3),
        4,
    )
    .unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].class_id, 2);
    assert_eq!(values[0].vertices, cuboid().vertices);
    let implicit = parse_cuboid_labels("0.1 0.2 1 0.7 0.8 1", "a", None, 2).unwrap();
    assert_eq!(implicit[0].class_id, 0);
    // C++ casts the leading float to int before checking its sign.
    assert_eq!(
        parse_cuboid_labels("-0.2 0.1 0.2 1 0.7 0.8 1", "a", Some(1), 2).unwrap()[0].class_id,
        0
    );
}

#[test]
fn yolo_3d_mixed_label_ambiguous_row_precedence_matches_source() {
    for (k, row) in [
        (2, "0 0.5 0.4 0.2 0.3 1 1"),
        (3, "0 0.5 0.4 0.2 0.3 1 1 1 1"),
    ] {
        let labels = parse_mixed_cuboid_labels(row, "a", Some(1), k).unwrap();
        assert!(
            matches!(&labels[0], MixedLabel::Box2d { label, .. } if label.class_id == 0 && label.x == 0.5)
        );
    }
    let labels = parse_mixed_cuboid_labels(
        "0 0.5 0.4 0.2 0.3\n2 0.2 0.3 1 0.6 0.3 -1 0.6 0.9 0.7 0.2 0.9 0",
        "a",
        Some(3),
        4,
    )
    .unwrap();
    assert!(matches!(&labels[1], MixedLabel::Cuboid(c) if c.class_id == 2));
}

#[test]
fn yolo_3d_label_unknown_visibility_is_not_absent() {
    let c = parse_cuboid_labels("0.1 0.2 -1 0.7 0.8 1 1 1 0", "a", None, 3)
        .unwrap()
        .remove(0);
    assert_eq!(c.hull().unwrap(), [0.1, 0.2, 0.7, 0.8]);
    assert_eq!(c.vertices[0][2], -1.0);
    let absent = parse_cuboid_labels("0.1 0.2 0 0.7 0.8 0", "a", None, 2)
        .unwrap()
        .remove(0);
    assert_eq!(absent.hull().unwrap(), [0.1, 0.2, 0.7, 0.8]);
}

#[test]
fn yolo_3d_label_clamps_coordinates_but_rejects_nonfinite_and_malformed_rows() {
    let c = parse_cuboid_labels("-1 -2 1 2 3 1", "a", None, 2)
        .unwrap()
        .remove(0);
    assert_eq!(c.vertices, [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]);
    for row in [
        "0 0",
        "a b c",
        "NaN 0 1 1 1 1",
        "0 0 inf 1 1 1",
        "-1 0 0 1 1 1 1",
        "3 0 0 1 1 1 1",
        "0.5 0.5 1 0.5 0.5 1",
        "2147483648 0 0 1 1 1 1",
    ] {
        assert!(parse_cuboid_labels(row, "a", Some(3), 2).is_err(), "{row}");
    }
    for k in [0, 9, usize::MAX] {
        assert!(parse_cuboid_labels("", "a", None, k).is_err());
    }
    assert!(parse_cuboid_labels("", "a", Some(0), 2).is_err());
    assert!(
        parse_cuboid_labels("\n  \n", "a", None, 2)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn yolo_3d_label_source_track_hash_and_pure_mixed_counter_offsets() {
    // Independently evaluated DJB2: hash("a") = 5381*33+97 = 177670.
    let pure = parse_cuboid_labels("0 0 1 1 1 1\n\n0 0 1 1 1 1", "a", None, 2).unwrap();
    assert_eq!(pure[0].track_id, 6_680_000);
    assert_eq!(pure[1].track_id, 6_680_001);
    let mixed = parse_mixed_cuboid_labels("0 0.5 0.5 0.3 0.4\n0 0 1 1 1 1", "a", None, 2).unwrap();
    assert!(matches!(
        &mixed[0],
        MixedLabel::Box2d {
            track_id: 6_680_001,
            ..
        }
    ));
    assert!(matches!(&mixed[1], MixedLabel::Cuboid(c) if c.track_id == 6_680_002));
}

#[test]
fn yolo_3d_truth_prefix_offsets_padding_and_independent_batches() {
    let s = spec();
    let mut b = cuboid();
    b.class_id = 1;
    b.track_id = 456;
    let out = encode_yolo_3d_truths(
        &[
            vec![MixedLabel::Cuboid(cuboid())],
            vec![MixedLabel::Cuboid(b)],
        ],
        &[meta(), meta()],
        &s,
    )
    .unwrap();
    assert_eq!((out.truth_size, out.max_boxes, out.batch), (23, 3, 2));
    assert_eq!(out.counts, [1, 1]);
    let row = out.record(0, 0).unwrap();
    for (actual, expected) in row[..6].iter().zip([0.4, 0.6, 0.4, 0.6, 2.0, 123.0]) {
        close(*actual, expected);
    }
    assert_eq!(&row[11..], cuboid().vertices.as_flattened());
    assert!(out.record(0, 1).unwrap().iter().all(|&v| v == 0.0));
    assert_eq!(&out.record(1, 0).unwrap()[4..6], [1.0, 456.0]);
    assert!(out.record(2, 0).is_err());
    assert!(out.record(0, 3).is_err());
}

#[test]
fn yolo_3d_truth_each_image_uses_its_own_crop_intrinsics_and_flip() {
    let a = meta();
    let mut b = meta();
    b.original_width = 640;
    b.crop.swidth = 640.0;
    b.crop.flip = true;
    let labels = vec![vec![MixedLabel::Cuboid(cuboid())]; 2];
    let out = encode_yolo_3d_truths(&labels, &[a, b], &spec()).unwrap();
    let first = out.record(0, 0).unwrap();
    let second = out.record(1, 0).unwrap();
    let f_px = 24.0_f64 / 43.27 * 640.0_f64.hypot(960.0);
    close(second[6], (f_px / 640.0) as f32);
    close(second[7], (f_px / 960.0) as f32);
    assert!((first[6] - second[6]).abs() > 0.1);
    assert_eq!((first[10], second[10]), (1.0, -1.0));
    close(first[11], 0.2);
    close(second[11], 0.8);
}

#[test]
fn yolo_3d_truth_intrinsics_hand_computed_and_crop() {
    let s = spec();
    let mut m = meta();
    m.crop.pleft = 128.0;
    m.crop.ptop = 96.0;
    m.crop.swidth = 1024.0;
    m.crop.sheight = 768.0;
    let out = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(cuboid())]], &[m], &s).unwrap();
    let row = out.record(0, 0).unwrap();
    // 1280x960 diagonal = 1600 pixels, f_px = 24/43.27*1600.
    close(row[6], 0.86665124);
    close(row[7], 1.155535);
    close(row[8], 0.5);
    close(row[9], 0.5);
    close(row[10], 1.0);
    close(row[11], 0.125);
    close(row[12], 0.25);
}

#[test]
fn yolo_3d_truth_flip_mirrors_px_and_negates_geo_without_reordering_vertices() {
    let s = spec();
    let mut m = meta();
    m.crop.pleft = 64.0;
    m.crop.swidth = 1280.0;
    let base = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(cuboid())]], &[m], &s).unwrap();
    m.crop.flip = true;
    let flip = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(cuboid())]], &[m], &s).unwrap();
    let a = base.record(0, 0).unwrap();
    let b = flip.record(0, 0).unwrap();
    close(b[8], 1.0 - a[8]);
    assert_eq!(b[10], -1.0);
    for k in 0..4 {
        close(b[11 + 3 * k], 1.0 - a[11 + 3 * k]);
        assert_eq!(b[12 + 3 * k], a[12 + 3 * k]);
        assert_eq!(b[13 + 3 * k], a[13 + 3 * k]);
    }
}

#[test]
fn yolo_3d_truth_clipped_absent_corner_zeroes_intrinsics_but_keeps_points() {
    let s = spec();
    let mut c = cuboid();
    c.vertices[3] = [0.0, 0.0, 0.0];
    let mut m = meta();
    m.crop.pleft = 128.0;
    let out = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(c)]], &[m], &s).unwrap();
    let row = out.record(0, 0).unwrap();
    assert_eq!(&row[6..11], [0.0; 5]);
    assert_eq!(&row[20..23], [0.0, 0.0, 0.0]);
    close(row[11], 0.1);
}

#[test]
fn yolo_3d_truth_mosaic_drops_fields_after_class() {
    let mut m = meta();
    m.crop.mosaic = true;
    let out = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(cuboid())]], &[m], &spec()).unwrap();
    let row = out.record(0, 0).unwrap();
    assert_eq!(row[4], 2.0);
    assert!(row[5..].iter().all(|&v| v == 0.0));
}

#[test]
fn yolo_3d_truth_mixed_ownership_routes_rows_to_heads() {
    let labels = vec![
        MixedLabel::Box2d {
            label: crate::train::YoloLabel {
                class_id: 1,
                x: 0.4,
                y: 0.5,
                w: 0.3,
                h: 0.2,
            },
            track_id: 77,
        },
        MixedLabel::Cuboid(cuboid()),
    ];
    assert!(encode_yolo_3d_truths(&[labels.clone()], &[meta()], &spec()).is_err());
    let out = encode_mixed_yolo_3d_truths(&[labels], &[meta()], &spec()).unwrap();
    assert_eq!(out.truth_size, 24);
    let box2d = out.record(0, 0).unwrap();
    let row_3d = out.record(0, 1).unwrap();
    assert_eq!(box2d[23], 1.0);
    assert_eq!(row_3d[23], 2.0);
    assert_eq!(box2d[5], 77.0);
    assert!(box2d[6..23].iter().all(|&v| v == 0.0));
    assert_eq!(&row_3d[11..23], cuboid().vertices.as_flattened());
    assert_eq!(out.record(0, 2).unwrap()[23], 0.0);
}

#[test]
fn yolo_3d_truth_deterministic_cap_before_tiny_filter_and_missing_focal() {
    let mut s = spec();
    s.max_boxes = 1;
    s.geometry.focal_35mm = 0.0;
    let mut tiny = cuboid();
    for v in &mut tiny.vertices {
        v[0] *= 0.0001;
        v[1] *= 0.0001;
    }
    let out = encode_yolo_3d_truths(
        &[vec![MixedLabel::Cuboid(tiny), MixedLabel::Cuboid(cuboid())]],
        &[meta()],
        &s,
    )
    .unwrap();
    assert_eq!(out.counts, [0]); // cap before filtering, no refill.
    let out = encode_yolo_3d_truths(
        &[vec![
            MixedLabel::Cuboid(cuboid()),
            MixedLabel::Cuboid(cuboid()),
        ]],
        &[meta()],
        &s,
    )
    .unwrap();
    assert_eq!(out.counts, [1]);
    assert_eq!(&out.record(0, 0).unwrap()[6..11], [0.0; 5]);
}

#[test]
fn yolo_3d_truth_mixed_mosaic_also_drops_ownership_like_source() {
    // data.cpp:1490-1505 copies only entries 0..4, not the flags slot.
    let mut m = meta();
    m.crop.mosaic = true;
    let labels = vec![
        MixedLabel::Cuboid(cuboid()),
        MixedLabel::Box2d {
            label: crate::train::YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 0.3,
                h: 0.3,
            },
            track_id: 7,
        },
    ];
    let out = encode_mixed_yolo_3d_truths(&[labels], &[m], &spec()).unwrap();
    assert_eq!(out.counts, [2]);
    for index in 0..2 {
        assert!(out.record(0, index).unwrap()[5..].iter().all(|&v| v == 0.0));
    }
}

#[test]
fn yolo_3d_truth_cropped_hull_does_not_rebuild_from_clamped_vertices() {
    let mut c = cuboid();
    // Absent far corner must not enlarge the hull after crop/flip.
    c.vertices[3] = [1.0, 1.0, 0.0];
    let mut m = meta();
    m.crop.pleft = 256.0;
    m.crop.swidth = 640.0;
    let out = encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(c)]], &[m], &spec()).unwrap();
    let row = out.record(0, 0).unwrap();
    close(row[0], 0.4);
    close(row[2], 0.8);
    assert_eq!(row[20], 1.0); // absent corner clipped, while hull ends at 0.8
    assert_eq!(&row[6..11], [0.0; 5]);
}

#[test]
fn yolo_3d_truth_two_dimensional_zero_center_is_source_sentinel() {
    let row = MixedLabel::Box2d {
        label: crate::train::YoloLabel {
            class_id: 0,
            x: 0.0,
            y: 0.0,
            w: 0.8,
            h: 0.8,
        },
        track_id: 1,
    };
    let out = encode_mixed_yolo_3d_truths(&[vec![row]], &[meta()], &spec()).unwrap();
    assert_eq!(out.counts, [0]);
    assert!(out.truths.iter().all(|&v| v == 0.0));
}

#[test]
fn yolo_3d_label_file_loaders_preserve_grammar_and_report_io_errors() {
    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = Fixture(
        std::env::temp_dir().join(format!("yolo3d-labels-{}-{unique}", std::process::id())),
    );
    std::fs::create_dir(&dir.0).unwrap();
    let path = dir.0.join("label.txt");
    std::fs::write(&path, "0 0.1 0.2 1 0.7 0.8 1").unwrap();
    assert_eq!(load_pure_cuboid_labels(&path, Some(1), 2).unwrap().len(), 1);
    assert!(matches!(
        &load_cuboid_labels(&path, Some(1), 2).unwrap()[0],
        MixedLabel::Box2d { .. }
    ));
    assert!(matches!(
        load_cuboid_labels(dir.0.join("missing.txt"), Some(1), 2),
        Err(Error::Io(_))
    ));
    std::fs::write(&path, "").unwrap();
    assert!(load_cuboid_labels(&path, None, 2).unwrap().is_empty());
    std::fs::write(&path, "malformed").unwrap();
    let message = load_pure_cuboid_labels(&path, None, 2)
        .unwrap_err()
        .to_string();
    assert!(message.contains("line 1") && message.contains("label.txt"));
}

#[test]
fn yolo_3d_truth_inconsistent_public_storage_is_an_error() {
    let mut out = encode_yolo_3d_truths(&[vec![]], &[meta()], &spec()).unwrap();
    out.truths.clear();
    assert!(out.record(0, 0).is_err());
    out.max_boxes = usize::MAX;
    out.batch = 3;
    assert!(out.record(2, 0).is_err());
}

#[test]
fn yolo_3d_truth_invalid_public_inputs_are_errors_not_panics() {
    let s = spec();
    let labels = vec![vec![MixedLabel::Cuboid(cuboid())]];
    assert!(encode_yolo_3d_truths(&labels, &[], &s).is_err());
    for m in [
        Yolo3dImageMeta {
            original_width: 0,
            ..meta()
        },
        Yolo3dImageMeta {
            network_height: 0,
            ..meta()
        },
        Yolo3dImageMeta {
            crop: Yolo3dCrop {
                swidth: 0.0,
                ..meta().crop
            },
            ..meta()
        },
        Yolo3dImageMeta {
            crop: Yolo3dCrop {
                pleft: f32::NAN,
                ..meta().crop
            },
            ..meta()
        },
    ] {
        assert!(encode_yolo_3d_truths(&labels, &[m], &s).is_err());
    }
    let mut bad = cuboid();
    bad.vertices[0][0] = f32::NAN;
    assert!(encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(bad)]], &[meta()], &s).is_err());
    let mut bad = cuboid();
    bad.class_id = 3;
    assert!(encode_yolo_3d_truths(&[vec![MixedLabel::Cuboid(bad)]], &[meta()], &s).is_err());
    let mut bad = s;
    bad.max_boxes = usize::MAX;
    assert!(encode_yolo_3d_truths(&labels, &[meta()], &bad).is_err());
}
