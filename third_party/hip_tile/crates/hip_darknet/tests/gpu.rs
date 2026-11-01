//! GPU executor vs the CPU reference: per-layer parity and detection parity + ground truth.
//! Skips when the DarkSuite dataset is not present.
#![cfg(feature = "gpu")]

use hip_darknet::gpu::GpuModel;
use hip_darknet::yolo::{BBox, box_iou};
use hip_darknet::{DetectOptions, Model, detect_cpu, detect_gpu, image::Image, reference};
use std::path::Path;

const DIR: &str = "/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2";

fn labels(path: &Path) -> Vec<(usize, BBox)> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            (v.len() == 5).then(|| {
                let f = |i: usize| v[i].parse::<f32>().unwrap();
                (v[0].parse().unwrap(), BBox { x: f(1), y: f(2), w: f(3), h: f(4) })
            })
        })
        .collect()
}

fn load() -> Option<(Model, Vec<std::path::PathBuf>)> {
    let dir = Path::new(DIR);
    let (cfg, weights) = (dir.join("LegoGears_v2.cfg"), dir.join("LegoGears_v2_last.weights"));
    if !cfg.exists() || !weights.exists() {
        eprintln!("SKIP: {DIR} not found");
        return None;
    }
    let model = Model::load(&cfg, &weights).unwrap();
    let list = std::fs::read_to_string(dir.join("LegoGears_v2_valid.txt")).unwrap();
    let images = list
        .lines()
        .map(Path::new)
        .filter(|p| p.extension().is_some_and(|e| e == "png") && !labels(&p.with_extension("txt")).is_empty())
        .map(Path::to_path_buf)
        .collect();
    Some((model, images))
}

#[test]
fn per_layer_parity() {
    let Some((model, images)) = load() else { return };
    let gpu = GpuModel::new(&model).unwrap();
    let n = model.layers.len();
    let mut max_abs = vec![0f64; n];
    let mut max_rel = vec![0f64; n];
    for p in images.iter().take(3) {
        let img = Image::load(p).unwrap().resize(model.net.width, model.net.height);
        let want = reference::forward(&model, &img.data).unwrap();
        let got = gpu.forward_all(&img.data).unwrap();
        for i in 0..n {
            assert_eq!(got[i].len(), want[i].len(), "layer {i} size");
            let scale = want[i].iter().fold(0f64, |m, v| m.max(f64::from(v.abs())));
            let floor = (scale * 0.01).max(1e-12);
            for (g, w) in got[i].iter().zip(&want[i]) {
                let d = (f64::from(*g) - f64::from(*w)).abs();
                max_abs[i] = max_abs[i].max(d);
                max_rel[i] = max_rel[i].max(d / f64::from(w.abs()).max(floor));
            }
        }
    }
    println!("{:>5} {:<8} {:>14} {:>10} {:>10}", "layer", "kind", "out CxHxW", "max abs", "max rel");
    for (i, l) in model.layers.iter().enumerate() {
        let kind = match l.kind {
            hip_darknet::LayerKind::Conv(_) => "conv",
            hip_darknet::LayerKind::Maxpool(_) => "maxpool",
            hip_darknet::LayerKind::Route { .. } => "route",
            hip_darknet::LayerKind::Upsample { .. } => "upsample",
            hip_darknet::LayerKind::Yolo(_) => "yolo",
        };
        let s = l.out_shape;
        println!("{i:>5} {kind:<8} {:>14} {:>10.2e} {:>10.2e}", format!("{}x{}x{}", s.c, s.h, s.w), max_abs[i], max_rel[i]);
    }
    for (i, r) in max_rel.iter().enumerate() {
        assert!(*r <= 1e-3, "layer {i}: max rel err {r:e}");
    }
}

#[test]
fn detection_parity_and_ground_truth() {
    let Some((model, images)) = load() else { return };
    let images: Vec<_> = images.into_iter().take(10).collect();
    assert!(images.len() >= 8);
    let gpu = GpuModel::new(&model).unwrap();
    let opts = DetectOptions::default();
    let (mut n_gt, mut n_det, mut n_tp, mut n_cls_err) = (0, 0, 0, 0);
    for p in &images {
        let img = Image::load(p).unwrap();
        let c = detect_cpu(&model, &img, &opts).unwrap();
        let g = detect_gpu(&model, &gpu, &img, &opts).unwrap();
        let stem = p.file_stem().unwrap().to_string_lossy();
        assert_eq!(c.len(), g.len(), "{stem}: detection count cpu {} gpu {}", c.len(), g.len());
        for (a, b) in c.iter().zip(&g) {
            assert_eq!(a.class, b.class, "{stem}: class");
            assert!(box_iou(&a.bbox, &b.bbox) >= 0.99, "{stem}: IoU {}", box_iou(&a.bbox, &b.bbox));
            assert!((a.prob - b.prob).abs() <= 1e-3, "{stem}: conf cpu {} gpu {}", a.prob, b.prob);
        }
        let gt = labels(&p.with_extension("txt"));
        n_gt += gt.len();
        n_det += g.len();
        let mut used = vec![false; g.len()];
        for (cls, t) in &gt {
            let best = g
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, d)| (i, box_iou(t, &d.bbox)))
                .filter(|&(_, iou)| iou >= 0.5)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((i, _)) = best {
                used[i] = true;
                n_tp += 1;
                n_cls_err += usize::from(g[i].class != *cls);
            }
        }
    }
    let recall = n_tp as f32 / n_gt as f32;
    let precision = n_tp as f32 / n_det.max(1) as f32;
    println!(
        "GPU: images={} gt={n_gt} det={n_det} matched@IoU0.5={n_tp} recall={recall:.3} precision={precision:.3} class_errors={n_cls_err}; GPU == CPU detections on all images",
        images.len()
    );
    assert!(recall >= 0.9, "recall {recall}");
    assert_eq!(n_cls_err, 0);
}
