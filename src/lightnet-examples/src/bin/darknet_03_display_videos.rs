//! Rust equivalent of `darknet_03_display_videos.cpp`: process a video,
//! annotate frames, and send the result to the display/output pipeline.

use lightnet_examples::{
    AppResult, Detector, display_video, output_path, parse_common, print_video_summary,
    process_video,
};

fn main() -> AppResult<()> {
    let args = parse_common(std::env::args().skip(1))?;
    let detector = Detector::load(&args)?;
    let workers = std::thread::available_parallelism()
        .map(|count| count.get().min(4))
        .unwrap_or(2);
    println!("threads for this video ... {workers}");
    let multiple = args.inputs.len() > 1;
    for input in &args.inputs {
        println!("processing {}:", input.display());
        let output = output_path(input, args.output.as_deref(), "_display.mp4", multiple);
        let summary = process_video(&detector, input, &output, false, workers, args.seconds)?;
        print_video_summary(&summary);
        display_video(&output)?;
    }
    Ok(())
}
