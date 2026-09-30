//! Rust equivalent of the RTSP display example.  The stream is decoded by the
//! external media helper while Darknet inference remains in Rust.

use darknet_examples::{
    AppResult, Detector, display_video, parse_common, print_video_summary, process_video,
};

fn main() -> AppResult<()> {
    let mut args = parse_common(std::env::args().skip(1))?;
    if args.inputs.len() != 1 {
        return Err("RTSP example expects exactly one stream URL".into());
    }
    let detector = Detector::load(&args)?;
    let input = args.inputs.remove(0);
    let output = args
        .output
        .take()
        .unwrap_or_else(|| "rtsp_output.mp4".into());
    println!("Connecting to stream {}...", input.display());
    let summary = process_video(&detector, &input, &output, false, 1, args.seconds)?;
    print_video_summary(&summary);
    display_video(&output)?;
    Ok(())
}
