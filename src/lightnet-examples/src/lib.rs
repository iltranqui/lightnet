//! Shared helpers for Rust equivalents of Darknet's `src-examples`.
//!
//! The binaries preserve the original example split while sharing model
//! loading, image annotation, ffmpeg/ffplay process handling, output naming,
//! and detection formatting here.

use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use lightnet_core::{
    Detection, DetectionBox, DetectionOptions, ExecutableNetwork, Network, PlanarRgbImage,
    decode_yolo_detections, load_names_file,
};
use image::RgbImage;
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;

pub type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Debug)]
pub struct CommonArgs {
    pub cfg: PathBuf,
    pub weights: Option<PathBuf>,
    pub names: Option<PathBuf>,
    pub inputs: Vec<PathBuf>,
    pub threshold: f32,
    pub nms: f32,
    pub letterbox: bool,
    pub output: Option<PathBuf>,
    pub seconds: Option<u64>,
}

pub fn parse_common<I>(mut args: I) -> AppResult<CommonArgs>
where
    I: Iterator<Item = String>,
{
    let cfg = PathBuf::from(required(&mut args, "cfg path")?);
    let weights = optional_path(required(&mut args, "weights path or '-'")?);
    let names = optional_path(required(&mut args, "names path or '-'")?);

    let mut result = CommonArgs {
        cfg,
        weights,
        names,
        inputs: Vec::new(),
        threshold: 0.25,
        nms: 0.45,
        letterbox: true,
        output: None,
        seconds: None,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threshold" => {
                result.threshold = parse_next(&mut args, "--threshold")?;
            }
            "--nms" => {
                result.nms = parse_next(&mut args, "--nms")?;
            }
            "--resize" => result.letterbox = false,
            "--letterbox" => result.letterbox = true,
            "--output" => {
                result.output = Some(PathBuf::from(required(&mut args, "--output path")?));
            }
            "--seconds" => {
                result.seconds = Some(parse_next(&mut args, "--seconds")?);
            }
            _ => result.inputs.push(PathBuf::from(arg)),
        }
    }

    if result.inputs.is_empty() {
        return Err("at least one input path or stream is required".into());
    }
    Ok(result)
}

fn required<I>(args: &mut I, name: &str) -> AppResult<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing required argument: {name}").into())
}

fn parse_next<T, I>(args: &mut I, name: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
    I: Iterator<Item = String>,
{
    required(args, name)?
        .parse()
        .map_err(|err| format!("invalid {name}: {err}").into())
}

fn optional_path(value: String) -> Option<PathBuf> {
    (value != "-").then(|| PathBuf::from(value))
}

#[derive(Clone, Debug)]
pub struct Detector {
    pub executable: ExecutableNetwork,
    pub names: Vec<String>,
    pub options: DetectionOptions,
}

impl Detector {
    pub fn load(args: &CommonArgs) -> AppResult<Self> {
        // Darknet's image/video prediction API uses a single image even when
        // the cfg's [net] batch value is its training batch.
        let network = Network::from_path(&args.cfg)?.with_batch(1);
        let mut executable = ExecutableNetwork::new(network);
        if let Some(weights) = &args.weights {
            let header = executable.load_weights_file(weights)?;
            println!(
                "loaded weights {}.{}.{} seen={}",
                header.major, header.minor, header.revision, header.seen
            );
        } else {
            println!("running with zero-initialized weights");
        }
        let names = match &args.names {
            Some(path) => load_names_file(path)?,
            None => Vec::new(),
        };
        Ok(Self {
            executable,
            names,
            options: DetectionOptions {
                confidence_threshold: args.threshold,
                nms_threshold: args.nms,
                letterbox: args.letterbox,
            },
        })
    }

    pub fn predict(&self, image: &PlanarRgbImage) -> AppResult<Vec<Detection>> {
        let prepared = if self.options.letterbox {
            image.letterbox(
                self.executable.network.net.width,
                self.executable.network.net.height,
            )?
        } else {
            image.resize(
                self.executable.network.net.width,
                self.executable.network.net.height,
            )?
        };
        let forward = self.executable.forward(&prepared.into())?;
        Ok(decode_yolo_detections(
            &self.executable.network,
            &forward.yolo_heads,
            image.shape.width,
            image.shape.height,
            self.options,
        )?)
    }

    pub fn class_name(&self, class_id: usize) -> &str {
        self.names
            .get(class_id)
            .map(String::as_str)
            .unwrap_or("<unknown>")
    }
}

pub fn output_path(
    input: &Path,
    requested: Option<&Path>,
    suffix: &str,
    multiple: bool,
) -> PathBuf {
    if let Some(requested) = requested {
        if !multiple {
            return requested.to_path_buf();
        }
        let _ = fs::create_dir_all(requested);
        return requested.join(
            input
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
                + suffix,
        );
    }
    input.with_file_name(
        input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
            + suffix,
    )
}

pub fn process_image(
    detector: &Detector,
    input: &Path,
    output: &Path,
    keypoints: bool,
) -> AppResult<Vec<Detection>> {
    let image = PlanarRgbImage::load(input)?;
    let detections = detector.predict(&image)?;
    let mut rgb = image.to_rgb8()?;
    if keypoints {
        annotate_keypoints(&mut rgb, &detections, detector.options.confidence_threshold);
    } else {
        annotate_detections(&mut rgb, &detections, detector.options.confidence_threshold);
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    rgb.save(output)?;
    Ok(detections)
}

pub fn print_detections(detector: &Detector, detections: &[Detection]) {
    println!("prediction results: {}", detections.len());
    for (index, detection) in detections.iter().enumerate() {
        println!(
            "-> {}/{}: #{} {} prob={:.6} x={:.1} y={:.1} w={:.1} h={:.1}",
            index + 1,
            detections.len(),
            detection.class_id,
            detector.class_name(detection.class_id),
            detection.confidence,
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.w,
            detection.bbox.h
        );
    }
}

pub fn annotate_detections(image: &mut RgbImage, detections: &[Detection], threshold: f32) {
    for detection in detections
        .iter()
        .filter(|detection| detection.confidence >= threshold)
    {
        draw_box(image, detection.bbox, color_for_class(detection.class_id));
    }
}

pub fn annotate_keypoints(image: &mut RgbImage, detections: &[Detection], threshold: f32) {
    let mut joints = [None; 17];
    for detection in detections
        .iter()
        .filter(|detection| detection.confidence >= threshold)
    {
        if detection.class_id < joints.len() {
            joints[detection.class_id] = Some((detection.bbox.x, detection.bbox.y));
        } else if detection.class_id == 17 {
            draw_box(image, detection.bbox, image::Rgb([255, 255, 0]));
        }
    }

    const BONES: &[(usize, usize)] = &[
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ];
    for &(from, to) in BONES {
        if let (Some((x1, y1)), Some((x2, y2))) = (joints[from], joints[to]) {
            draw_line_segment_mut(image, (x1, y1), (x2, y2), image::Rgb([0, 255, 0]));
        }
    }
    for (x, y) in joints.into_iter().flatten() {
        draw_filled_circle_mut(
            image,
            (x.round() as i32, y.round() as i32),
            3,
            image::Rgb([255, 0, 0]),
        );
    }
}

fn draw_box(image: &mut RgbImage, bbox: DetectionBox, color: image::Rgb<u8>) {
    let max_x = image.width().saturating_sub(1) as i32;
    let max_y = image.height().saturating_sub(1) as i32;
    let left = (bbox.left().round() as i32).clamp(0, max_x);
    let top = (bbox.top().round() as i32).clamp(0, max_y);
    let right = (bbox.right().round() as i32).clamp(0, max_x).max(left);
    let bottom = (bbox.bottom().round() as i32).clamp(0, max_y).max(top);
    draw_hollow_rect_mut(
        image,
        Rect::at(left, top).of_size((right - left).max(1) as u32, (bottom - top).max(1) as u32),
        color,
    );
}

fn color_for_class(class_id: usize) -> image::Rgb<u8> {
    let colors = [
        image::Rgb([255, 64, 64]),
        image::Rgb([64, 255, 64]),
        image::Rgb([64, 128, 255]),
        image::Rgb([255, 192, 64]),
        image::Rgb([224, 64, 255]),
        image::Rgb([64, 255, 224]),
    ];
    colors[class_id % colors.len()]
}

#[derive(Clone, Debug, Default)]
pub struct VideoSummary {
    pub frames: usize,
    pub objects: usize,
    pub elapsed: Duration,
    pub fps: f64,
}

pub fn process_video(
    detector: &Detector,
    input: &Path,
    output: &Path,
    keypoints: bool,
    workers: usize,
    seconds: Option<u64>,
) -> AppResult<VideoSummary> {
    let source = unique_temp_dir("lightnet-rs-video-source")?;
    let rendered = unique_temp_dir("lightnet-rs-video-rendered")?;
    let started = Instant::now();
    extract_frames(input, &source, seconds)?;
    let mut frames = fs::read_dir(&source)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
        .collect::<Vec<_>>();
    frames.sort();
    if frames.is_empty() {
        return Err(format!("no video frames decoded from {}", input.display()).into());
    }
    fs::create_dir_all(&rendered)?;

    let total_frames = frames.len();
    println!("-> decoding frames .......... {total_frames}");
    let workers = workers.max(1).min(frames.len());
    let frame_paths = Arc::new(frames);
    let completed = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(workers);
    for worker in 0..workers {
        let detector = detector.clone();
        let frame_paths = Arc::clone(&frame_paths);
        let rendered = rendered.clone();
        let completed = Arc::clone(&completed);
        handles.push(thread::spawn(move || -> AppResult<usize> {
            let mut objects = 0;
            for index in (worker..frame_paths.len()).step_by(workers) {
                let output = rendered.join(format!("frame_{index:08}.png"));
                objects += process_image(&detector, &frame_paths[index], &output, keypoints)?.len();
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 25 == 0 || done == total_frames {
                    print!("\r-> detecting frames ......... {done}/{total_frames}");
                    let _ = std::io::stdout().flush();
                }
            }
            Ok(objects)
        }));
    }
    let mut objects = 0;
    for handle in handles {
        objects += handle
            .join()
            .map_err(|_| "video worker thread panicked")??;
    }
    println!();

    let fps = probe_video_fps(input).unwrap_or(30.0);
    encode_frames(&rendered, fps, output)?;
    let elapsed = started.elapsed();
    let frames = frame_paths.len();
    let _ = fs::remove_dir_all(&source);
    let _ = fs::remove_dir_all(&rendered);
    Ok(VideoSummary {
        frames,
        objects,
        elapsed,
        fps: frames as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
    })
}

pub fn print_video_summary(summary: &VideoSummary) {
    println!("-> total frames processed ... {}", summary.frames);
    println!(
        "-> time to process video .... {:.3} seconds",
        summary.elapsed.as_secs_f64()
    );
    println!("-> processed frame rate ..... {:.3} FPS", summary.fps);
    println!("-> total objects found ...... {}", summary.objects);
    println!(
        "-> average objects/frame .... {:.6}",
        summary.objects as f64 / summary.frames.max(1) as f64
    );
}

pub fn display_image(path: &Path) -> AppResult<()> {
    run_status(
        Command::new("ffplay")
            .args(["-hide_banner", "-loglevel", "error", "-autoexit"])
            .arg(path),
        "ffplay image",
    )
}

pub fn display_video(path: &Path) -> AppResult<()> {
    run_status(
        Command::new("ffplay")
            .args(["-hide_banner", "-loglevel", "error", "-autoexit"])
            .arg(path),
        "ffplay video",
    )
}

fn extract_frames(input: &Path, output: &Path, seconds: Option<u64>) -> AppResult<()> {
    fs::create_dir_all(output)?;
    let mut command = Command::new("ffmpeg");
    command.args(["-hide_banner", "-loglevel", "error"]);
    if input.to_string_lossy().starts_with("/dev/video") {
        command.args(["-f", "v4l2", "-framerate", "30", "-video_size", "640x480"]);
    }
    command.arg("-i").arg(input);
    if let Some(seconds) = seconds {
        command.args(["-t", &seconds.to_string()]);
    }
    command
        .args(["-vsync", "0", "-start_number", "0"])
        .arg(output.join("frame_%08d.png"));
    run_status(&mut command, "ffmpeg frame extraction")
}

fn encode_frames(input: &Path, fps: f64, output: &Path) -> AppResult<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    run_status(
        Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-framerate", &format!("{fps:.6}")])
            .arg("-i")
            .arg(input.join("frame_%08d.png"))
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(output),
        "ffmpeg video encoding",
    )
}

fn probe_video_fps(input: &Path) -> AppResult<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(input)
        .output()?;
    if !output.status.success() {
        return Err("ffprobe could not read the video".into());
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if let Some((numerator, denominator)) = value.split_once('/') {
        return Ok(numerator.parse::<f64>()? / denominator.parse::<f64>()?);
    }
    Ok(value.parse()?)
}

fn run_status(command: &mut Command, description: &str) -> AppResult<()> {
    let output = command.output()?;
    if output.status.success() {
        return Ok(());
    }
    let details = String::from_utf8_lossy(&output.stderr);
    Err(format!("{description} failed: {details}").into())
}

fn unique_temp_dir(prefix: &str) -> AppResult<PathBuf> {
    let path = std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn json_record(
    detector: &Detector,
    input: &Path,
    image_width: usize,
    image_height: usize,
    detections: &[Detection],
    elapsed: Duration,
) -> serde_json::Value {
    serde_json::json!({
        "filename": input.display().to_string(),
        "count": detections.len(),
        "duration_ms": elapsed.as_secs_f64() * 1000.0,
        "predictions": detections.iter().map(|d| serde_json::json!({
            "class": d.class_id,
            "name": detector.class_name(d.class_id),
            "probability": d.confidence,
            "objectness": d.objectness,
            "rect": {
                "x": d.bbox.x,
                "y": d.bbox.y,
                "width": d.bbox.w,
                "height": d.bbox.h
            },
            "normalized": {
                "x": d.bbox.x / image_width.max(1) as f32,
                "y": d.bbox.y / image_height.max(1) as f32,
                "width": d.bbox.w / image_width.max(1) as f32,
                "height": d.bbox.h / image_height.max(1) as f32
            }
        })).collect::<Vec<_>>()
    })
}
