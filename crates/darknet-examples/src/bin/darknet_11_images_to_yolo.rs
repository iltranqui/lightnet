//! Rust equivalent of `darknet_11_images_to_yolo.cpp`.
//! Converts detections into normalized YOLO label files beside the images.

use std::fs;
use std::io::Write;

use darknet_examples::{AppResult, Detector, parse_common};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let mut missing = 0usize;
    let mut existing = 0usize;
    let mut processed = 0usize;
    let mut negative = 0usize;
    let mut objects = 0usize;

    for input in &args.inputs {
        if !input.exists() {
            missing += 1;
            println!("ERROR: image file does not exist: {}", input.display());
            continue;
        }
        let output = input.with_extension("txt");
        if output.exists() {
            existing += 1;
            println!(
                "WARNING: YOLO annotation file already exists: {}",
                output.display()
            );
            continue;
        }
        println!("processing {}", input.display());
        let image = darknet_core::PlanarRgbImage::load(input)?;
        let detections = detector.predict(&image)?;
        let mut file = fs::File::create(&output)?;
        if detections.is_empty() {
            negative += 1;
        } else {
            for detection in &detections {
                let width = image.shape.width as f32;
                let height = image.shape.height as f32;
                writeln!(
                    file,
                    "{} {:.10} {:.10} {:.10} {:.10}",
                    detection.class_id,
                    detection.bbox.x / width,
                    detection.bbox.y / height,
                    detection.bbox.w / width,
                    detection.bbox.h / height,
                )?;
            }
        }
        processed += 1;
        objects += detections.len();
    }

    println!("-> missing images or bad paths .... {missing}");
    println!("-> old YOLO annotations exist ..... {existing}");
    println!("-> new YOLO files saved to disk ... {processed}");
    println!("-> negative samples ............... {negative}");
    println!("-> total objects detected ......... {objects}");
    Ok(())
}
