//! Ground-truth check: CPU detections on validation images vs the YOLO-format label files.
//! Skips when the DarkSuite dataset is not present.

use hip_darknet::yolo::{BBox, box_iou};
use hip_darknet::{DetectOptions, Model, detect_cpu, image::Image, load_names};
use std::path::Path;
use std::time::Instant;

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

#[test]
fn detections_match_labels() {
    let dir = Path::new(DIR);
    let (cfg, weights) = (dir.join("LegoGears_v2.cfg"), dir.join("LegoGears_v2_last.weights"));
    if !cfg.exists() || !weights.exists() {
        eprintln!("SKIP: {DIR} not found");
        return;
    }
    let model = Model::load(&cfg, &weights).unwrap();
    let names = load_names(dir.join("LegoGears.names")).unwrap();
    assert_eq!(names.len(), 5);
    assert_eq!(model.seen, 48000);

    // Labelled 224x160 PNGs from the validation list (the JPEGs are not net-sized).
    let list = std::fs::read_to_string(dir.join("LegoGears_v2_valid.txt")).unwrap();
    let images: Vec<&Path> = list
        .lines()
        .map(Path::new)
        .filter(|p| p.extension().is_some_and(|e| e == "png") && !labels(&p.with_extension("txt")).is_empty())
        .take(10)
        .collect();
    assert!(images.len() >= 8, "found only {} labelled PNGs", images.len());

    let opts = DetectOptions::default();
    let (mut n_gt, mut n_det, mut n_tp, mut n_cls_err, mut total_ms) = (0, 0, 0, 0, 0.0);
    println!("{:<14} {:>9} {:>5}  {:<12} {:>6}  gt box (cx cy w h)  ->  det box   IoU", "image", "class", "", "pred", "conf");
    for p in &images {
        let img = Image::load(p).unwrap();
        assert_eq!((img.w, img.h), (224, 160));
        let t = Instant::now();
        let dets = detect_cpu(&model, &img, &opts).unwrap();
        total_ms += t.elapsed().as_secs_f64() * 1e3;
        let gt = labels(&p.with_extension("txt"));
        n_gt += gt.len();
        n_det += dets.len();
        let mut used = vec![false; dets.len()];
        let stem = p.file_stem().unwrap().to_string_lossy();
        for (cls, g) in &gt {
            // Best unused detection by IoU (any class), as in a class-aware error count.
            let best = dets
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, d)| (i, box_iou(g, &d.bbox)))
                .filter(|&(_, iou)| iou >= 0.5)
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, iou)) => {
                    used[i] = true;
                    n_tp += 1;
                    let d = &dets[i];
                    if d.class != *cls {
                        n_cls_err += 1;
                    }
                    println!(
                        "{stem:<14} {:>9} {:>5}  {:<12} {:>5.1}%  ({:.3} {:.3} {:.3} {:.3}) -> ({:.3} {:.3} {:.3} {:.3})  {iou:.2}",
                        names[*cls], "", names[d.class], d.prob * 100.0, g.x, g.y, g.w, g.h, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
                    );
                }
                None => println!("{stem:<14} {:>9} MISSED ({:.3} {:.3} {:.3} {:.3})", names[*cls], g.x, g.y, g.w, g.h),
            }
        }
        for (_, d) in dets.iter().enumerate().filter(|(i, _)| !used[*i]) {
            println!("{stem:<14} {:>9} EXTRA {:.1}% ({:.3} {:.3} {:.3} {:.3})", names[d.class], d.prob * 100.0, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
        }
    }
    let recall = n_tp as f32 / n_gt as f32;
    let precision = n_tp as f32 / n_det.max(1) as f32;
    println!(
        "images={} gt={n_gt} det={n_det} matched@IoU0.5={n_tp} recall={recall:.3} precision={precision:.3} class_errors={n_cls_err} cpu={:.0} ms/frame (incl. resize+decode)",
        images.len(),
        total_ms / images.len() as f64
    );
    assert!(recall >= 0.9, "recall {recall}");
    assert_eq!(n_cls_err, 0, "class mismatches on matched boxes");
}
