//! Rust equivalent of the keypoint video example.  It applies keypoint
//! annotations frame-by-frame and writes the requested output stream.

use darknet_examples::{
    AppResult, Detector, display_video, output_path, parse_common, print_video_summary,
    process_video,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("processing {}:", input.display());
        let output = output_path(input, args.output.as_deref(), "_keypoints.mp4", multiple);
        let summary = process_video(&detector, input, &output, true, 1, args.seconds)?;
        print_video_summary(&summary);
        display_video(&output)?;
    }
    Ok(())
}
