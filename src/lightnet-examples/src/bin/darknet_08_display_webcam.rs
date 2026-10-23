//! Rust equivalent of the webcam display example.  The camera path/device is
//! passed through the common media pipeline and each frame is annotated.

use lightnet_examples::{
    AppResult, Detector, display_video, parse_common, print_video_summary, process_video,
};

fn main() -> AppResult<()> {
    let mut args = parse_common(std::env::args().skip(1))?;
    if args.inputs.len() != 1 {
        return Err("webcam example expects one camera device, such as /dev/video0".into());
    }
    let camera = args.inputs.remove(0);
    let detector = Detector::load(&args)?;
    let output = args
        .output
        .take()
        .unwrap_or_else(|| "webcam_output.mp4".into());
    let duration = args.seconds.or(Some(10));
    println!("Opening webcam {}...", camera.display());
    let summary = process_video(&detector, &camera, &output, false, 1, duration)?;
    print_video_summary(&summary);
    display_video(&output)?;
    Ok(())
}
