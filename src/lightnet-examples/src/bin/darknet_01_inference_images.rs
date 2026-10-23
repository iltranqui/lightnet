//! Rust equivalent of `src-examples/darknet_01_inference_images.cpp`.
//! Runs inference for one or more images and prints detections.

use std::time::Instant;

use lightnet_examples::{
    AppResult, Detector, output_path, parse_common, print_detections, process_image,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("predicting {}:", input.display());
        let output = output_path(input, args.output.as_deref(), "_output.png", multiple);
        let started = Instant::now();
        let detections = process_image(&detector, input, &output, false)?;
        print_detections(&detector, &detections);
        println!("annotated output: {}", output.display());
        println!(
            "elapsed: {:.3} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
