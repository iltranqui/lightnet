//! Opt-in measurement of detector training loader phases.
//!
//! Run with a real dataset (e.g. LegoGears_v2):
//!
//! ```text
//! DARKNET_LOADER_PROFILE_CFG=../datasets/LegoGears_v2/LegoGears_v2.cfg \
//! DARKNET_LOADER_PROFILE_DATA=../datasets/LegoGears_v2/LegoGears_v2.data \
//! cargo test -p darknet-core --release --test loader_profile -- --ignored --nocapture
//! ```
//!
//! `DARKNET_LOADER_PROFILE_ITERS` overrides the measured batch count (default 20).
//! The test is a no-op (passes) when the env vars are unset.

use std::time::Instant;

use darknet_core::loader_profile;

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[test]
#[ignore = "opt-in loader measurement; set DARKNET_LOADER_PROFILE_CFG and _DATA"]
fn measure_loader_phases() {
    let (Ok(cfg_path), Ok(data_path)) = (
        std::env::var("DARKNET_LOADER_PROFILE_CFG"),
        std::env::var("DARKNET_LOADER_PROFILE_DATA"),
    ) else {
        eprintln!(
            "skipping: set DARKNET_LOADER_PROFILE_CFG and DARKNET_LOADER_PROFILE_DATA to measure"
        );
        return;
    };
    let iters: usize = std::env::var("DARKNET_LOADER_PROFILE_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);

    let cfg = darknet_core::ConfigFile::parse_file(&cfg_path).expect("parse cfg");
    let augmentation = darknet_core::DetectorAugmentation::from_config(&cfg).expect("augmentation");
    let network = darknet_core::Network::from_config(cfg).expect("network");
    let data = darknet_core::DataConfig::from_path(&data_path).expect("data cfg");
    let paths = darknet_core::load_paths_file(data.required_train_list().expect("train list"))
        .expect("paths");

    let batch = network.net.batch.max(1);

    // Mirror the trainer's bounded loader pool (see `build_loader_pool` in the
    // CLI): leave a couple of cores for the GPU-feeding thread.
    let loader_threads = std::env::var("DARKNET_LOADER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|parallelism| parallelism.get().saturating_sub(2).clamp(1, 6))
                .unwrap_or(4)
        });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(loader_threads)
        .build()
        .expect("loader pool");
    eprintln!(
        "LOADER_PROFILE cfg={cfg_path} batch={batch} images={} mosaic={} mixup={} iters={iters} loader_threads={loader_threads}",
        paths.len(),
        augmentation.mosaic,
        augmentation.mixup
    );

    let run_batch = |iteration: usize| {
        let start_index = (iteration * batch) % paths.len();
        let sample_start = iteration as u64 * batch as u64;
        pool.install(|| {
            darknet_core::load_augmented_training_batch_from_paths(
                &network,
                &data,
                &paths,
                start_index,
                &augmentation,
                42,
                sample_start,
            )
        })
        .expect("batch")
    };

    // Warm the page cache / decode path before measuring.
    loader_profile::enable();
    for iteration in 0..3 {
        let _ = run_batch(iteration);
    }

    let mut per_image = vec![[0.0f64; 8]; iters];
    let mut per_augment = vec![[0.0f64; 5]; iters];
    let mut wall_ms = Vec::with_capacity(iters);
    let mut samples_seen = 0u64;
    let mut loads_seen = 0u64;
    for iteration in 0..iters {
        loader_profile::reset();
        let wall = Instant::now();
        let _batch = run_batch(iteration);
        wall_ms.push(wall.elapsed().as_secs_f64() * 1000.0);
        let snapshot = loader_profile::take();
        samples_seen += snapshot.samples;
        loads_seen += snapshot.loads;
        per_image[iteration] = snapshot.per_sample_ms();
        per_augment[iteration] = snapshot.per_sample_augment_ms();
    }

    let names = [
        "read", "decode", "planar", "labels", "augment", "copy", "load", "total",
    ];
    let mut summary = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let mut values: Vec<f64> = per_image.iter().map(|row| row[index]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = median(&mut values);
        summary.push(format!("{name}={median:.3}/{mean:.3}"));
    }
    eprintln!(
        "LOADER_PROFILE per-image ms median/mean: {}",
        summary.join(" ")
    );

    let augment_names = ["to_rgb8", "mean", "resize", "hsv", "encode"];
    let mut augment_summary = Vec::with_capacity(augment_names.len());
    for (index, name) in augment_names.iter().enumerate() {
        let mut values: Vec<f64> = per_augment.iter().map(|row| row[index]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = median(&mut values);
        augment_summary.push(format!("{name}={median:.3}/{mean:.3}"));
    }
    eprintln!(
        "LOADER_PROFILE augment-internal ms median/mean: {}",
        augment_summary.join(" ")
    );

    let mut wall = wall_ms.clone();
    let wall_median = median(&mut wall);
    let wall_mean = wall_ms.iter().sum::<f64>() / wall_ms.len() as f64;
    eprintln!(
        "LOADER_PROFILE batch wall ms median={wall_median:.3} mean={wall_mean:.3} | samples={samples_seen} loads={loads_seen}"
    );
}
