//! Rust equivalent of `darknet_09_keypoints_images.cpp`, using the same
//! detector output but drawing keypoint-style annotations.

use lightnet_examples::{
    AppResult, Detector, display_image, output_path, parse_common, print_detections, process_image,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("processing {}", input.display());
        let output = output_path(input, args.output.as_deref(), "_keypoints.png", multiple);
        let detections = process_image(&detector, input, &output, true)?;
        print_detections(&detector, &detections);
        display_image(&output)?;
    }
    Ok(())
}
