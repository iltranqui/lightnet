//! Rust equivalent of `darknet_02_display_annotated_images.cpp`.
//! It writes annotated images and optionally opens them with the display helper.

use darknet_examples::{
    AppResult, Detector, display_image, output_path, parse_common, print_detections, process_image,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("processing {}", input.display());
        let output = output_path(input, args.output.as_deref(), "_annotated.png", multiple);
        let detections = process_image(&detector, input, &output, false)?;
        print_detections(&detector, &detections);
        println!("displaying {}", output.display());
        display_image(&output)?;
    }
    Ok(())
}
