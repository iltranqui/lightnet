//! Rust equivalent of the multithreaded Darknet video example.  Workers share
//! decoded model state while preserving frame/output ordering in the helper.

use darknet_examples::{
    AppResult, Detector, output_path, parse_common, print_video_summary, process_video,
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
        let output = output_path(input, args.output.as_deref(), "_output.mp4", multiple);
        let summary = process_video(&detector, input, &output, false, workers, args.seconds)?;
        print_video_summary(&summary);
        println!("output filename .......... {}", output.display());
    }
    Ok(())
}
