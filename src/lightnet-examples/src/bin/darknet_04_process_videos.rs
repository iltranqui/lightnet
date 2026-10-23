//! Rust equivalent of `darknet_04_process_videos.cpp` for non-interactive
//! video processing and annotated output writing.

use lightnet_examples::{
    AppResult, Detector, output_path, parse_common, print_video_summary, process_video,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("processing {}:", input.display());
        let output = output_path(input, args.output.as_deref(), "_output.mp4", multiple);
        let summary = process_video(&detector, input, &output, false, 1, args.seconds)?;
        print_video_summary(&summary);
        println!("output filename .......... {}", output.display());
    }
    Ok(())
}
