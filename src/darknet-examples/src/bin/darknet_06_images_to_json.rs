//! Rust equivalent of `darknet_06_images_to_json.cpp`.
//! Converts image detections into a machine-readable JSON report.

use std::fs;
use std::time::Instant;

use darknet_examples::{AppResult, Detector, json_record, parse_common, print_detections};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let output = args.output.clone().unwrap_or_else(|| "output.json".into());
    let started = Instant::now();
    let mut records = Vec::new();
    let mut objects = 0usize;
    for input in &args.inputs {
        print!("processing {}: ", input.display());
        let image = darknet_core::PlanarRgbImage::load(input)?;
        let image_started = Instant::now();
        let detections = detector.predict(&image)?;
        println!("{} objects", detections.len());
        print_detections(&detector, &detections);
        objects += detections.len();
        records.push(json_record(
            &detector,
            input,
            image.shape.width,
            image.shape.height,
            &detections,
            image_started.elapsed(),
        ));
    }
    let document = serde_json::json!({ "file": records });
    fs::write(&output, serde_json::to_vec_pretty(&document)?)?;
    println!("-> JSON results ....... {}", output.display());
    println!("-> images processed ... {}", args.inputs.len());
    println!("-> objects detected ... {objects}");
    println!(
        "-> time elapsed ....... {:.3} seconds",
        started.elapsed().as_secs_f64()
    );
    println!(
        "-> processed rate ..... {:.1} FPS",
        args.inputs.len() as f64 / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE)
    );
    Ok(())
}
