//! `darknet-rs`, the command-line front end for the Rust port.
//!
//! CLI code stays thin: it parses commands, loads models/data, and formats
//! Darknet-style summaries while `darknet-core` owns graph and training
//! semantics.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "oxide")]
use darknet_core::GpuExecutableNetwork;
use darknet_core::train::validate_obb_augmentation;
use darknet_core::{
    ConfigFile, DataConfig, Detection, DetectionOptions, DetectorAugmentation, ExecutableNetwork,
    ForwardResult, LayerKind, LayerParams, Network, ObbDetection, ObbDetectionGeometry,
    PlanarRgbImage, Tensor, TrainingBatch, YoloLabel, YoloTrainingState, compute_learning_rate,
    compute_yolo_training_loss_with_state, decode_network_detections,
    decode_obb_network_detections, initialize_training_params, inspect_weights_file,
    load_augmented_training_batch_from_paths, load_names_file, load_paths_file, save_weights,
};
use darknet_core::{
    compute_obb_yolo_training_deltas, load_obb_training_batch_with_flips_from_paths,
};
#[cfg(feature = "oxide")]
use darknet_cuda_oxide::OxideBackend;

mod colour;

fn main() {
    let handle = std::thread::Builder::new()
        .name("darknet-cli".to_owned())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            if let Err(err) = run() {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        })
        .expect("failed to spawn CLI thread");

    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

struct ExportOnnxArgs {
    cfg: String,
    weights: String,
    out: PathBuf,
    fuse: bool,
}

fn parse_export_onnx_args(args: Vec<String>) -> Result<ExportOnnxArgs, String> {
    let mut positional = Vec::new();
    let mut fuse = true;
    for arg in args {
        match arg.as_str() {
            "--no-fuse" | "-dontfuse" => fuse = false,
            other if other.starts_with('-') => {
                return Err(format!("unknown export-onnx flag: {other}"));
            }
            _ => positional.push(arg),
        }
    }
    if !(2..=3).contains(&positional.len()) {
        return Err(format!(
            "export-onnx requires cfg, weights, and optionally output path (got {} positional arguments)",
            positional.len()
        ));
    }
    let cfg = positional.remove(0);
    let weights = positional.remove(0);
    let out = positional
        .pop()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&cfg).with_extension("onnx"));
    Ok(ExportOnnxArgs {
        cfg,
        weights,
        out,
        fuse,
    })
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let _bin = args.next();
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "detector" => {
            let subcommand = required_arg(args.next(), "detector subcommand")?;
            match subcommand.as_str() {
                "train" => {
                    let first_path = required_arg(args.next(), "cfg or data path")?;
                    let second_path = required_arg(args.next(), "data or cfg path")?;
                    let (cfg, data) = classify_detector_train_paths(&first_path, &second_path)?;
                    let options = TrainOptions::parse(args)?;
                    run_detector_train(&cfg, &data, &options)?;
                }
                "map" => {
                    let cfg = required_arg(args.next(), "cfg path")?;
                    let data = required_arg(args.next(), "data path")?;
                    let options = MapOptions::parse(args)?;
                    run_detector_map(&cfg, &data, &options)?;
                }
                other => {
                    return Err(format!("unknown detector subcommand: {other}").into());
                }
            }
        }
        "version" => {
            println!("darknet-rs {}", env!("CARGO_PKG_VERSION"));
            println!("cutile reference {}", darknet_cuda::CUTILE_RS_REFERENCE);
        }
        "cfg-validate" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let network = Network::from_path(&cfg)?;
            println!(
                "validated {} with {} layers and {} parameters",
                cfg,
                network.layers.len(),
                network.total_parameter_count()
            );
        }
        "cfg-summary" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let network = Network::from_path(&cfg)?;
            print_network_summary(&network);
        }
        "names" => {
            let names_path = required_arg(args.next(), "names path")?;
            let names = load_names_file(&names_path)?;
            println!("loaded {} class names from {}", names.len(), names_path);
            for (idx, name) in names.iter().take(10).enumerate() {
                println!("{idx:>3}: {name}");
            }
        }
        "weights-inspect" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let weights = required_arg(args.next(), "weights path")?;
            let network = Network::from_path(&cfg)?;
            let inspection = inspect_weights_file(&network, &weights)?;
            println!(
                "weights version {}.{}.{}",
                inspection.header.major, inspection.header.minor, inspection.header.revision
            );
            println!("images seen {}", inspection.header.seen);
            println!(
                "accounted {} bytes of {} bytes (trailing {})",
                inspection.accounted_bytes, inspection.file_size, inspection.trailing_bytes
            );
            for layer in inspection.layers.iter().take(16) {
                let bytes: u64 = layer.arrays.iter().map(|array| array.byte_len).sum();
                println!(
                    "#{:>3} {:<18} {} bytes",
                    layer.layer_index, layer.layer_name, bytes
                );
            }
        }
        "forward-dummy" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let maybe_weights = args.next();
            let mut executable = ExecutableNetwork::from_path(&cfg)?;
            if let Some(weights) = maybe_weights {
                let header = executable.load_weights_file(&weights)?;
                println!(
                    "loaded weights {}.{}.{} seen={}",
                    header.major, header.minor, header.revision, header.seen
                );
            } else {
                println!("running with zero-initialized weights");
            }

            let input = executable.zero_input();
            let result = executable.forward(&input)?;
            let final_output = result
                .final_output()
                .ok_or("network produced no layer outputs")?;

            println!("layers executed {}", result.layer_outputs.len());
            println!("final output {}", final_output.shape);
            println!("yolo heads {}", result.yolo_heads.len());
            for head in &result.yolo_heads {
                println!("#{:>3} yolo {}", head.layer_index, head.output.shape);
            }
        }
        "forward-backward-dummy" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let next_arg = args.next();
            let (maybe_weights, seed_mode) = match next_arg.as_deref() {
                Some("auto" | "yolo" | "final") => (None, next_arg.expect("seed mode present")),
                Some("-") => (None, args.next().unwrap_or_else(|| "auto".to_owned())),
                Some(_) => (next_arg, args.next().unwrap_or_else(|| "auto".to_owned())),
                None => (None, "auto".to_owned()),
            };
            let mut executable = ExecutableNetwork::from_path(&cfg)?;
            if let Some(weights) = maybe_weights {
                let header = executable.load_weights_file(&weights)?;
                println!(
                    "loaded weights {}.{}.{} seen={}",
                    header.major, header.minor, header.revision, header.seen
                );
            } else {
                println!("running with zero-initialized weights");
            }

            let input = executable.zero_input();
            print_forward_backward_summary(&mut executable, &input, &seed_mode)?;
        }
        "bench-train" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let steps = args.next().map(|v| v.parse()).transpose()?.unwrap_or(10);
            let warmup = args.next().map(|v| v.parse()).transpose()?.unwrap_or(2);
            bench_train(&cfg, steps, warmup)?;
        }
        "detect-image" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let weights = required_arg(args.next(), "weights path")?;
            let names_path = required_arg(args.next(), "names path")?;
            let image_path = required_arg(args.next(), "image path")?;
            let threshold = optional_f32(args.next(), 0.25, "confidence threshold")?;
            let nms = optional_f32(args.next(), 0.45, "nms threshold")?;
            let preprocess = args.next().unwrap_or_else(|| "resize".to_owned());
            let letterbox = match preprocess.as_str() {
                "letterbox" => true,
                "resize" | "resize-exact" => false,
                other => {
                    return Err(format!(
                        "invalid preprocess mode: {other} (expected letterbox, resize or resize-exact)"
                    )
                    .into());
                }
            };

            let names = load_names_file(&names_path)?;
            let network = Network::from_path(&cfg)?.with_batch(1);
            network.ensure_no_cuboid_heads("detect-image")?;
            let (net_w, net_h) = (network.net.width, network.net.height);
            let original = PlanarRgbImage::load(&image_path)?;
            let prepared = if letterbox {
                original.letterbox(net_w, net_h)?
            } else if preprocess == "resize-exact" {
                original.resize_linear_exact(net_w, net_h)?
            } else {
                original.resize(net_w, net_h)?
            };
            let input: Tensor = prepared.clone().into();
            let result = match selected_backend()?.as_str() {
                "cpu" => {
                    let mut executable = ExecutableNetwork::new(network.clone());
                    let header = executable.load_weights_file(&weights)?;
                    println!(
                        "loaded weights {}.{}.{} seen={}",
                        header.major, header.minor, header.revision, header.seen
                    );
                    if let Some(iters) = std::env::var("DARKNET_BENCH_ITERS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                    {
                        let batch = bench_batch();
                        let mut bench_exec = ExecutableNetwork::new(network.with_batch(batch));
                        bench_exec.load_weights_file(&weights)?;
                        let bench_input = replicate_batch(&input, batch);
                        bench_exec.forward(&bench_input)?;
                        let mut ms = Vec::with_capacity(iters);
                        for _ in 0..iters {
                            let start = Instant::now();
                            bench_exec.forward(&bench_input)?;
                            ms.push(start.elapsed().as_secs_f64() * 1e3);
                        }
                        print_bench_json("cpu", batch, &ms, "\"threads\":0");
                    }
                    executable.forward(&input)?
                }
                #[cfg(feature = "oxide")]
                "oxide" => {
                    let backend = OxideBackend::new()?;
                    println!(
                        "gpu backend oxide device=0 conv={:?} gemm={:?}",
                        backend.conv_mode(),
                        backend.gemm_mode()
                    );
                    let mut executable = GpuExecutableNetwork::new(network.clone(), backend);
                    let header = executable.load_weights_file(&weights)?;
                    println!(
                        "loaded weights {}.{}.{} seen={}",
                        header.major, header.minor, header.revision, header.seen
                    );
                    if let Some(iters) = std::env::var("DARKNET_BENCH_ITERS")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                    {
                        // Latency benchmark (`tools/oxide_bench.py`): forward_compact returns
                        // the yolo heads on the host, so every iteration is fully synchronised.
                        // DARKNET_BENCH_BATCH replicates the image into a batch (default 1).
                        let batch = bench_batch();
                        let bench_net = network.with_batch(batch);
                        let bench_input = replicate_batch(&input, batch);
                        let mut bench_exec =
                            GpuExecutableNetwork::new(bench_net, OxideBackend::new()?);
                        bench_exec.load_weights_file(&weights)?;
                        let (free0, total) = bench_exec.backend().device_mem_info()?;
                        let mut peak_used = 0usize;
                        // DARKNET_OXIDE_GEMM=fp16 runs the f16-storage inference path.
                        let use_f16 = bench_exec.backend().f16_storage_enabled();
                        for _ in 0..5 {
                            if use_f16 {
                                bench_exec.forward_compact_f16(&bench_input)?;
                            } else {
                                bench_exec.forward_compact(&bench_input)?;
                            }
                        }
                        let mut ms = Vec::with_capacity(iters);
                        for _ in 0..iters {
                            let start = Instant::now();
                            if use_f16 {
                                bench_exec.forward_compact_f16(&bench_input)?;
                            } else {
                                bench_exec.forward_compact(&bench_input)?;
                            }
                            ms.push(start.elapsed().as_secs_f64() * 1e3);
                            let (free, _) = bench_exec.backend().device_mem_info()?;
                            peak_used = peak_used.max(free0.saturating_sub(free));
                        }
                        let b = bench_exec.backend();
                        print_bench_json(
                            "oxide",
                            batch,
                            &ms,
                            &format!(
                                "\"device_mem_delta_mb\":{:.1},\"workspace_mb\":{:.2},\"pool_cached_mb\":{:.1},\"device_total_mb\":{:.0},\"conv_mode\":\"{:?}\",\"gemm_mode\":\"{:?}\"",
                                peak_used as f64 / 1048576.0,
                                b.workspace_bytes() as f64 / 1048576.0,
                                b.pool_cached_bytes() as f64 / 1048576.0,
                                total as f64 / 1048576.0,
                                b.conv_mode(),
                                b.gemm_mode()
                            ),
                        );
                    }
                    if executable.backend().f16_storage_enabled() {
                        // f16 storage inference (DARKNET_OXIDE_GEMM=fp16): only the
                        // yolo heads are produced (detections need nothing else).
                        let summary = executable.forward_compact_f16(&input)?;
                        darknet_core::ForwardResult::from_heads(
                            summary.yolo_heads,
                            summary.modern_heads,
                        )
                    } else {
                        executable.forward(&input)?
                    }
                }
                other => {
                    return Err(format!(
                        "backend {other} is not available in this build (oxide needs `--features oxide` via cargo-oxide)"
                    )
                    .into());
                }
            };
            if let Some(dump_dir) = std::env::var_os("DARKNET_LAYER_DUMP_DIR") {
                dump_rust_layer_outputs(Path::new(&dump_dir), &network, &result.layer_outputs)?;
            }
            let detections = decode_image_detections(
                &network,
                &result,
                original.shape.width,
                original.shape.height,
                DetectionOptions {
                    confidence_threshold: threshold,
                    nms_threshold: nms,
                    letterbox,
                },
            )?;

            println!("image {}", image_path);
            println!(
                "input {}x{} -> network {}x{} ({})",
                original.shape.width,
                original.shape.height,
                prepared.shape.width,
                prepared.shape.height,
                preprocess.as_str()
            );
            println!("detections {}", detections.len());
            let tsv = std::env::var_os("DARKNET_DETECT_TSV").is_some();
            match detections {
                ImageDetections::AxisAligned(detections) => {
                    for detection in detections {
                        let class_name = names
                            .get(detection.class_id)
                            .map(String::as_str)
                            .unwrap_or("<unknown>");
                        if tsv {
                            // Full-precision machine-readable line for backend comparisons.
                            println!(
                                "det\t{class_name}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
                                detection.confidence,
                                detection.bbox.left(),
                                detection.bbox.top(),
                                detection.bbox.right(),
                                detection.bbox.bottom(),
                            );
                            continue;
                        }
                        println!(
                            "{:>6.2}% {:<24} left={:>7.2} top={:>7.2} right={:>7.2} bottom={:>7.2}",
                            detection.confidence * 100.0,
                            class_name,
                            detection.bbox.left(),
                            detection.bbox.top(),
                            detection.bbox.right(),
                            detection.bbox.bottom(),
                        );
                    }
                }
                ImageDetections::Oriented(detections) => {
                    for detection in detections {
                        let class_name = names
                            .get(detection.detection.class_id)
                            .map(String::as_str)
                            .unwrap_or("<unknown>");
                        println!("{}", format_obb_detection_line(&detection, class_name, tsv));
                    }
                }
            }
        }
        "detect-server" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let weights = required_arg(args.next(), "weights path")?;
            run_detect_server(&cfg, &weights)?;
        }
        "export-onnx" => {
            let parsed = parse_export_onnx_args(args.collect())?;
            let network = Network::from_path(&parsed.cfg)?.with_batch(1);
            let mut executable = ExecutableNetwork::new(network);
            executable.load_weights_file(&parsed.weights)?;
            let bytes = darknet_core::export_onnx(
                &executable.network,
                &executable.params,
                &darknet_core::OnnxOptions {
                    fuse_batch_norm: parsed.fuse,
                },
            )?;
            fs::write(&parsed.out, &bytes)?;
            println!(
                "wrote {} ({} bytes, batch_norm {})",
                parsed.out.display(),
                bytes.len(),
                if parsed.fuse { "fused" } else { "separate" }
            );
        }
        "help" | "--help" | "-h" => print_usage(),
        other => {
            return Err(format!("unknown command: {other}").into());
        }
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
enum ImageDetections {
    AxisAligned(Vec<Detection>),
    Oriented(Vec<ObbDetection>),
}

impl ImageDetections {
    fn len(&self) -> usize {
        match self {
            Self::AxisAligned(detections) => detections.len(),
            Self::Oriented(detections) => detections.len(),
        }
    }
}

/// `detect-server`: one JSON request per stdin line, one JSON reply per line.
///
/// Requests are `{"image": "<path>", "thresh": <f32>}`; replies are
/// `{"boxes":[{"class","conf","cx","cy","w","h"}]}` with normalized geometry, or
/// `{"error": "<message>"}`. A bad request or image never terminates the server.
fn run_detect_server(cfg: &str, weights: &str) -> Result<(), Box<dyn std::error::Error>> {
    let network = Network::from_path(cfg)?.with_batch(1);
    network.ensure_no_cuboid_heads("detect-server")?;
    if network.net.obb {
        return Err("detect-server supports detector mode only (OBB is out of scope)".into());
    }
    match selected_backend()?.as_str() {
        "cpu" => {
            let mut executable = ExecutableNetwork::new(network.clone());
            executable.load_weights_file(weights)?;
            serve_detections(&network, |input| executable.forward(input))
        }
        #[cfg(feature = "oxide")]
        "oxide" => {
            let backend = OxideBackend::new()?;
            let mut executable = GpuExecutableNetwork::new(network.clone(), backend);
            executable.load_weights_file(weights)?;
            serve_detections(&network, |input| executable.forward(input))
        }
        other => Err(format!("unsupported detect-server backend: {other}").into()),
    }
}

fn serve_detections<F>(network: &Network, mut forward: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&Tensor) -> darknet_core::Result<ForwardResult>,
{
    use std::io::BufRead as _;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = detect_server_request(network, &line, &mut forward);
        writeln!(stdout, "{reply}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn detect_server_request<F>(network: &Network, line: &str, forward: &mut F) -> serde_json::Value
where
    F: FnMut(&Tensor) -> darknet_core::Result<ForwardResult>,
{
    let request: serde_json::Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(error) => return serde_json::json!({ "error": format!("invalid request: {error}") }),
    };
    let Some(image) = request.get("image").and_then(|value| value.as_str()) else {
        return serde_json::json!({ "error": "request needs an image path" });
    };
    let thresh = request
        .get("thresh")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.25) as f32;

    let prepared = match PlanarRgbImage::load(image) {
        Ok(original) => {
            let (width, height) = (original.shape.width, original.shape.height);
            match original.letterbox(network.net.width, network.net.height) {
                Ok(input) => (input, width, height),
                Err(error) => return serde_json::json!({ "error": error.to_string() }),
            }
        }
        Err(error) => return serde_json::json!({ "error": error.to_string() }),
    };
    let (input, width, height) = prepared;
    let input: Tensor = input.into();
    let result = match forward(&input) {
        Ok(result) => result,
        Err(error) => return serde_json::json!({ "error": error.to_string() }),
    };
    let options = DetectionOptions {
        confidence_threshold: thresh.clamp(0.0, 1.0),
        nms_threshold: 0.45,
        letterbox: true,
    };
    match decode_network_detections(network, &result, 0, width, height, options) {
        Ok(detections) => {
            let boxes: Vec<serde_json::Value> = detections
                .iter()
                .map(|detection| {
                    serde_json::json!({
                        "class": detection.class_id,
                        "conf": detection.confidence,
                        "cx": detection.bbox.x / width.max(1) as f32,
                        "cy": detection.bbox.y / height.max(1) as f32,
                        "w": detection.bbox.w / width.max(1) as f32,
                        "h": detection.bbox.h / height.max(1) as f32,
                    })
                })
                .collect();
            serde_json::json!({ "boxes": boxes })
        }
        Err(error) => serde_json::json!({ "error": error.to_string() }),
    }
}

fn decode_image_detections(
    network: &Network,
    result: &ForwardResult,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> darknet_core::Result<ImageDetections> {
    if network.net.obb {
        if !result.modern_heads.is_empty() {
            return Err(darknet_core::Error::InvalidConfig(
                "detect-image cannot combine OBB YOLO and modern detection heads".to_owned(),
            ));
        }
        return Ok(ImageDetections::Oriented(decode_obb_network_detections(
            network,
            result,
            image_width,
            image_height,
            options,
        )?));
    }

    let detections =
        decode_network_detections(network, result, 0, image_width, image_height, options)?;
    Ok(ImageDetections::AxisAligned(detections))
}

fn format_obb_detection_line(detection: &ObbDetection, class_name: &str, tsv: bool) -> String {
    let detection_box = detection.detection.bbox;
    let confidence = detection.detection.confidence;
    let bounds = (
        detection_box.left(),
        detection_box.top(),
        detection_box.right(),
        detection_box.bottom(),
    );
    match (&detection.geometry, tsv) {
        (ObbDetectionGeometry::FrontPoint { x, y }, true) => format!(
            "det_obb\t{class_name}\t{confidence:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\tfront_point\t{x:.9}\t{y:.9}",
            bounds.0, bounds.1, bounds.2, bounds.3,
        ),
        (ObbDetectionGeometry::CarpnQuad(points), true) => format!(
            "det_obb\t{class_name}\t{confidence:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\tcarpn_quad\t{}",
            bounds.0,
            bounds.1,
            bounds.2,
            bounds.3,
            points
                .iter()
                .map(|point| format!("{point:.9}"))
                .collect::<Vec<_>>()
                .join("\t"),
        ),
        (ObbDetectionGeometry::FrontPoint { x, y }, false) => format!(
            "{:>6.2}% {:<24} left={:>7.2} top={:>7.2} right={:>7.2} bottom={:>7.2} front=({x:.2},{y:.2})",
            confidence * 100.0,
            class_name,
            bounds.0,
            bounds.1,
            bounds.2,
            bounds.3,
        ),
        (ObbDetectionGeometry::CarpnQuad(points), false) => {
            let vertices = points
                .chunks_exact(2)
                .map(|vertex| format!("({:.2},{:.2})", vertex[0], vertex[1]))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{:>6.2}% {:<24} left={:>7.2} top={:>7.2} right={:>7.2} bottom={:>7.2} quad=[{vertices}]",
                confidence * 100.0,
                class_name,
                bounds.0,
                bounds.1,
                bounds.2,
                bounds.3,
            )
        }
    }
}

fn dump_rust_layer_outputs(
    dump_dir: &Path,
    network: &Network,
    layer_outputs: &[Tensor],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dump_dir)?;
    let mut metadata = fs::File::create(dump_dir.join("layers.tsv"))?;
    writeln!(metadata, "index\ttype\tw\th\tc\tbatch\tcount")?;
    for (index, (layer, output)) in network.layers.iter().zip(layer_outputs).enumerate() {
        let mut raw = fs::File::create(dump_dir.join(format!("layer_{index:03}.f32")))?;
        for value in &output.data {
            raw.write_all(&value.to_le_bytes())?;
        }
        let shape = output.shape;
        writeln!(
            metadata,
            "{index}\t{:?}\t{}\t{}\t{}\t{}\t{}",
            layer.kind,
            shape.width,
            shape.height,
            shape.channels,
            shape.batch,
            output.data.len(),
        )?;
    }
    Ok(())
}

/// Execution backend for inference/training: `DARKNET_BACKEND=cpu|oxide`
/// (default `cpu`). `DARKNET_OXIDE_CONV` / `DARKNET_OXIDE_GEMM` tune oxide.
fn selected_backend() -> Result<String, Box<dyn std::error::Error>> {
    let backend = std::env::var("DARKNET_BACKEND").unwrap_or_else(|_| "cpu".to_owned());
    match backend.as_str() {
        "cpu" | "oxide" => Ok(backend),
        other => Err(format!("invalid DARKNET_BACKEND: {other} (expected cpu or oxide)").into()),
    }
}

/// Deterministic uniform [0, 1) stream for the synthetic benchmark data.
fn bench_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as u32) as f32 / 16_777_216.0
}

/// He-style random weights, unit BN scales/variances, zero biases.
/// All convolution weights, flattened (for the bench weight-delta dump).
#[cfg(feature = "oxide")]
fn conv_weight_snapshot(params: &[LayerParams]) -> Vec<f32> {
    let mut v = Vec::new();
    for p in params {
        if let LayerParams::Convolutional(conv) = p {
            v.extend_from_slice(&conv.weights);
        }
    }
    v
}

/// First-order loss surrogate sum(delta * yolo_output) over all heads.
fn surrogate(heads: &[darknet_core::YoloHeadOutput], seeds: &[Option<Tensor>]) -> f64 {
    heads
        .iter()
        .filter_map(|h| seeds[h.layer_index].as_ref().map(|d| (h, d)))
        .map(|(h, d)| {
            h.output
                .data
                .iter()
                .zip(&d.data)
                .map(|(o, d)| f64::from(*o) * f64::from(*d))
                .sum::<f64>()
        })
        .sum()
}

/// L2 norm of all yolo output-gradient seeds.
fn seeds_l2(seeds: &[Option<Tensor>]) -> f64 {
    seeds
        .iter()
        .flatten()
        .flat_map(|t| t.data.iter())
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt()
}

/// Parity dump: per conv layer biases, [bn biases, scales, rolling mean, rolling var], weights
/// (f32 LE, in that order) to `<prefix>.step<k>.bin`, lengths to `<prefix>.layers` (`layer:nb:nbn:nw`).
fn dump_parity(
    prefix: &str,
    step: usize,
    params: &[LayerParams],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    let mut lens = Vec::new();
    for (idx, p) in params.iter().enumerate() {
        if let LayerParams::Convolutional(conv) = p {
            let put = |b: &mut Vec<u8>, v: &[f32]| v.iter().for_each(|x| b.extend(x.to_le_bytes()));
            put(&mut bytes, &conv.biases);
            let nbn = conv.batch_norm.as_ref().map_or(0, |bn| bn.scales.len());
            if let Some(bn) = &conv.batch_norm {
                put(&mut bytes, &bn.biases);
                put(&mut bytes, &bn.scales);
                put(&mut bytes, &bn.rolling_mean);
                put(&mut bytes, &bn.rolling_variance);
            }
            put(&mut bytes, &conv.weights);
            lens.push(format!(
                "{idx}:{}:{nbn}:{}",
                conv.biases.len(),
                conv.weights.len()
            ));
        }
    }
    fs::write(format!("{prefix}.layers"), lens.join(","))?;
    fs::write(format!("{prefix}.step{step}.bin"), bytes)?;
    Ok(())
}

fn init_bench_params(params: &mut [LayerParams], state: &mut u64) {
    for p in params {
        if let LayerParams::Convolutional(conv) = p {
            let fan_in = (conv.weights.len() / conv.biases.len().max(1)).max(1);
            let bound = (6.0 / fan_in as f32).sqrt();
            for w in &mut conv.weights {
                *w = (bench_uniform(state) * 2.0 - 1.0) * bound;
            }
            if let Some(bn) = conv.batch_norm.as_mut() {
                bn.scales.fill(1.0);
                bn.rolling_variance.fill(1.0);
            }
        }
    }
}

/// Peak resident host memory of this process in MB (`VmHWM`).
fn peak_host_rss_mb() -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Synthetic training-step benchmark (no dataset): random input and random
/// output deltas on every yolo layer stand in for the CPU loss (C++ and Rust
/// both compute the yolo loss on the host), so a step is
/// forward-for-training + backward + SGD update, fully synchronised (the GPU
/// path downloads every layer output after the forward and the updated weights
/// after the update).  Prints one `BENCH_TRAIN_JSON` line.
fn bench_train(cfg: &str, steps: usize, warmup: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut network = Network::from_path(cfg)?;
    if let Some(batch) = std::env::var("DARKNET_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        network = network.with_batch(batch);
    }
    let backend = selected_backend()?;
    let batch = network.net.batch;
    let lr = std::env::var("DARKNET_BENCH_LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-4f32);
    let (momentum, decay) = (network.net.momentum, network.net.decay);
    let mut rng = 0x5eed_1234_5678_9abcu64;
    let shape = network.net.input_shape();
    let input = Tensor::from_data(
        shape,
        (0..batch * shape.element_count())
            .map(|_| bench_uniform(&mut rng))
            .collect(),
    )?;
    let mut seeds: Vec<Option<Tensor>> = vec![None; network.layers.len()];
    for layer in &network.layers {
        if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
            let shape = layer.output_shape;
            seeds[layer.index] = Some(Tensor::from_data(
                shape,
                (0..shape.batch * shape.element_count())
                    .map(|_| (bench_uniform(&mut rng) - 0.5) * 0.2)
                    .collect(),
            )?);
        }
    }

    // `DARKNET_BENCH_INPUT=<f32 LE file>` + `DARKNET_BENCH_LABELS=<text, lines "b cls x y w h">`:
    // fixed real batch + the REAL yolo loss every step (training-parity run, see
    // docs/OXIDE_BENCH.md section 8). `DARKNET_PARITY_DUMP=<prefix>`,
    // `DARKNET_PARITY_STEPS=0,1,5` dump all parameters after those steps.
    let mut input = input;
    let mut parity_labels: Option<Vec<Vec<YoloLabel>>> = None;
    let read_f32 = |path: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(fs::read(path)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    if let Ok(ip) = std::env::var("DARKNET_BENCH_INPUT") {
        input = Tensor::from_data(shape, read_f32(&ip)?)?;
    }
    // `DARKNET_BENCH_DELTAS=<prefix>`: fixed synthetic yolo output deltas from `<prefix>.<layer>.bin`
    // (identical files are read by the C++ parity probe); prints a first-order loss surrogate
    // sum(delta * yolo_output) per step instead of the real loss.
    let fixed_deltas = if let Ok(prefix) = std::env::var("DARKNET_BENCH_DELTAS") {
        for layer in &network.layers {
            if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
                seeds[layer.index] = Some(Tensor::from_data(
                    layer.output_shape,
                    read_f32(&format!("{prefix}.{}.bin", layer.index))?,
                )?);
            }
        }
        true
    } else {
        false
    };
    if let Ok(lp) = std::env::var("DARKNET_BENCH_LABELS") {
        let mut labels = vec![Vec::new(); batch];
        for line in fs::read_to_string(&lp)?.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() == 6 {
                labels[f[0].parse::<usize>()?].push(YoloLabel {
                    class_id: f[1].parse()?,
                    x: f[2].parse()?,
                    y: f[3].parse()?,
                    w: f[4].parse()?,
                    h: f[5].parse()?,
                });
            }
        }
        parity_labels = Some(labels);
    }
    let synthetic_labels = (std::env::var("DARKNET_BENCH_YOLO").as_deref() == Ok("1")).then(|| {
        vec![
            vec![YoloLabel {
                class_id: 0,
                x: 0.5,
                y: 0.5,
                w: 0.2,
                h: 0.2,
            }];
            batch
        ]
    });
    let parity_prefix = std::env::var("DARKNET_PARITY_DUMP").ok();
    let parity_steps: Vec<usize> = std::env::var("DARKNET_PARITY_STEPS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let mut yolo_state = YoloTrainingState::default();

    let (mut total, mut fwd, mut bwd, mut upd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    #[allow(unused_mut)]
    let mut peak_device = 0usize;
    #[allow(unused_mut)]
    let mut device_total = 0usize;
    match backend.as_str() {
        "cpu" => {
            let mut exec = ExecutableNetwork::new(network);
            init_bench_params(&mut exec.params, &mut rng);
            if let Ok(path) = std::env::var("DARKNET_BENCH_WEIGHTS") {
                exec.load_weights_file(&path)?;
            }
            if let Some(prefix) = &parity_prefix {
                if parity_steps.contains(&0) {
                    dump_parity(prefix, 0, &exec.params)?;
                }
            }
            for step in 0..warmup + steps {
                let t0 = Instant::now();
                let forward = exec.forward_training(&input)?;
                if let Some(labels) = &parity_labels {
                    let yolo = compute_yolo_training_loss_with_state(
                        &exec.network,
                        &forward,
                        labels,
                        step,
                        Some(&mut yolo_state),
                    )?;
                    println!(
                        "PARITY backend=cpu step={} loss={:.6} delta_l2={:.6}",
                        step + 1,
                        yolo.loss,
                        seeds_l2(&yolo.layer_output_gradient_seeds)
                    );
                    seeds = yolo.layer_output_gradient_seeds;
                } else if let Some(labels) = &synthetic_labels {
                    seeds =
                        darknet_core::compute_yolo_training_loss(&exec.network, &forward, labels)?
                            .layer_output_gradient_seeds;
                }
                if fixed_deltas {
                    println!(
                        "PARITY backend=cpu step={} surrogate={:.6}",
                        step + 1,
                        surrogate(&forward.yolo_heads, &seeds)
                    );
                }
                let t1 = Instant::now();
                if let Some(labels) = parity_labels.as_ref().or(synthetic_labels.as_ref()) {
                    exec.backward_accumulate_seeded_with_hf_auxiliary_gradients(
                        &input, &forward, &seeds, labels,
                    )?;
                } else {
                    exec.backward_accumulate_seeded(&input, &forward, &seeds)?;
                }
                let t2 = Instant::now();
                if exec.network.net.muon {
                    exec.update_weights_muon(lr, batch)?;
                } else if exec.network.net.adam {
                    exec.update_weights_adam(lr, batch)?;
                } else {
                    exec.update_weights(lr, momentum, decay, batch);
                }
                let t3 = Instant::now();
                if let Some(prefix) = &parity_prefix {
                    if parity_steps.contains(&(step + 1)) {
                        dump_parity(prefix, step + 1, &exec.params)?;
                    }
                }
                if step >= warmup {
                    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
                    fwd.push(ms(t0, t1));
                    bwd.push(ms(t1, t2));
                    upd.push(ms(t2, t3));
                    total.push(ms(t0, t3));
                }
            }
        }
        #[cfg(feature = "oxide")]
        "oxide" => {
            let oxide = OxideBackend::new()?;
            let (free0, total_bytes) = oxide.device_mem_info()?;
            device_total = total_bytes;
            let mut exec = GpuExecutableNetwork::new(network, oxide);
            init_bench_params(&mut exec.params, &mut rng);
            // Optional trained weights instead of the random init (A/B of precision
            // modes on a well-conditioned net).
            if let Ok(path) = std::env::var("DARKNET_BENCH_WEIGHTS") {
                exec.load_weights_file(&path)?;
            }
            let init_weights = conv_weight_snapshot(&exec.params);
            exec.init_train_buffers_gpu()?;
            if let Some(prefix) = &parity_prefix {
                if parity_steps.contains(&0) {
                    dump_parity(prefix, 0, &exec.params)?;
                }
            }
            for step in 0..warmup + steps {
                let t0 = Instant::now();
                let forward = exec.forward_training_device(&input)?;
                if let Some(labels) = &parity_labels {
                    let yolo = compute_yolo_training_loss_with_state(
                        &exec.network,
                        &forward.cpu_forward,
                        labels,
                        step,
                        Some(&mut yolo_state),
                    )?;
                    println!(
                        "PARITY backend=oxide step={} loss={:.6} delta_l2={:.6}",
                        step + 1,
                        yolo.loss,
                        seeds_l2(&yolo.layer_output_gradient_seeds)
                    );
                    seeds = yolo.layer_output_gradient_seeds;
                } else if let Some(labels) = &synthetic_labels {
                    seeds = darknet_core::compute_yolo_training_loss(
                        &exec.network,
                        &forward.cpu_forward,
                        labels,
                    )?
                    .layer_output_gradient_seeds;
                }
                if fixed_deltas {
                    println!(
                        "PARITY backend=oxide step={} surrogate={:.6}",
                        step + 1,
                        surrogate(&forward.cpu_forward.yolo_heads, &seeds)
                    );
                }
                if step == 0 {
                    if let Some(prefix) = &parity_prefix {
                        for h in &forward.cpu_forward.yolo_heads {
                            let b: Vec<u8> =
                                h.output.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                            fs::write(format!("{prefix}.out.{}.bin", h.layer_index), b)?;
                        }
                    }
                }
                let t1 = Instant::now();
                exec.backward_accumulate_seeded_gpu(&forward, &seeds)?;
                let (free_bwd, _) = exec.backend().device_mem_info()?;
                let t2 = Instant::now();
                drop(forward);
                exec.update_weights_gpu_deferred(lr, momentum, decay, batch)?;
                let t3 = Instant::now();
                if let Some(prefix) = &parity_prefix {
                    if parity_steps.contains(&(step + 1)) {
                        exec.sync_host_weights_from_device()?;
                        dump_parity(prefix, step + 1, &exec.params)?;
                    }
                }
                let (free_end, _) = exec.backend().device_mem_info()?;
                peak_device = peak_device
                    .max(free0.saturating_sub(free_bwd))
                    .max(free0.saturating_sub(free_end));
                if step >= warmup {
                    let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
                    fwd.push(ms(t0, t1));
                    bwd.push(ms(t1, t2));
                    upd.push(ms(t2, t3));
                    total.push(ms(t0, t3));
                }
            }
            // `DARKNET_BENCH_DUMP=<file>`: conv weight deltas (f32 LE) for A/B of
            // precision modes (same init / input / seeds), see tools / OXIDE_BENCH.md.
            if let Ok(path) = std::env::var("DARKNET_BENCH_DUMP") {
                exec.sync_host_weights_from_device()?;
                let now = conv_weight_snapshot(&exec.params);
                let mut bytes = Vec::new();
                for (a, b) in now.iter().zip(&init_weights) {
                    bytes.extend((a - b).to_le_bytes());
                }
                let lens: Vec<String> = exec
                    .params
                    .iter()
                    .filter_map(|p| match p {
                        LayerParams::Convolutional(conv) => Some(conv.weights.len().to_string()),
                        _ => None,
                    })
                    .collect();
                fs::write(format!("{path}.layers"), lens.join(","))?;
                fs::write(path, bytes)?;
            }
        }
        other => {
            return Err(format!("backend {other} is not available in this build").into());
        }
    }
    let list = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        s.get(s.len() / 2).copied().unwrap_or(0.0)
    };
    println!(
        "bench-train backend={backend} batch={batch} steps={steps}: step median {:.2} ms (fwd {:.2} / bwd {:.2} / update {:.2})",
        median(&total),
        median(&fwd),
        median(&bwd),
        median(&upd)
    );
    println!(
        "BENCH_TRAIN_JSON {{\"backend\":\"{backend}\",\"batch\":{batch},\"steps\":{steps},\"ms\":[{}],\"fwd_ms\":[{}],\"bwd_ms\":[{}],\"update_ms\":[{}],\"device_mem_peak_mb\":{:.1},\"device_total_mb\":{:.0},\"host_rss_peak_mb\":{:.1}}}",
        list(&total),
        list(&fwd),
        list(&bwd),
        list(&upd),
        peak_device as f64 / 1048576.0,
        device_total as f64 / 1048576.0,
        peak_host_rss_mb()
    );
    Ok(())
}

fn bench_batch() -> usize {
    std::env::var("DARKNET_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .max(1)
}

/// Tile a batch-1 input into `batch` identical images.
fn replicate_batch(input: &Tensor, batch: usize) -> Tensor {
    let mut shape = input.shape;
    shape.batch = batch;
    Tensor {
        shape,
        data: input.data.repeat(batch),
    }
}

/// One machine-readable line for `tools/oxide_bench.py`: per-iteration latencies in ms.
fn print_bench_json(backend: &str, batch: usize, ms: &[f64], extra: &str) {
    let list: Vec<String> = ms.iter().map(|v| format!("{v:.4}")).collect();
    println!(
        "BENCH_JSON {{\"backend\":\"{backend}\",\"batch\":{batch},\"iters\":{},\"ms\":[{}],{extra}}}",
        ms.len(),
        list.join(",")
    );
}

fn required_arg(value: Option<String>, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("missing required argument: {name}").into())
}

fn optional_f32(
    value: Option<String>,
    default: f32,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    match value {
        Some(value) => value
            .parse()
            .map_err(|_| format!("invalid {name}: {value}").into()),
        None => Ok(default),
    }
}

fn classify_detector_train_paths(
    first_path: &str,
    second_path: &str,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let first_ext = Path::new(first_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let second_ext = Path::new(second_path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();

    match (first_ext, second_ext) {
        ("cfg", "data") => Ok((first_path.to_owned(), second_path.to_owned())),
        ("data", "cfg") => Ok((second_path.to_owned(), first_path.to_owned())),
        _ => Err(format!(
            "detector train expects one .cfg path and one .data path, got {first_path} and {second_path}"
        )
        .into()),
    }
}

#[derive(Debug, PartialEq)]
struct TrainOptions {
    weights: Option<String>,
    seed: u64,
    clear: bool,
    dont_show: bool,
    gpu: bool,
    map: bool,
    map_every: Option<usize>,
    map_thresh: f32,
    map_iou: f32,
    /// Iterations between periodic checkpoints. Also controls `<name>_last.weights`.
    save_every: usize,
    /// Dynamic loss scaling on the oxide training path: halve on non-finite
    /// gradients (skipping the step) and double after a long finite streak.
    dynamic_loss_scale: bool,
    /// ANSI colour output: `Some(true)` = `-colour`, `Some(false)` = `-nocolour`,
    /// `None` = auto (terminal + no `NO_COLOR`).
    colour: Option<bool>,
}

impl TrainOptions {
    fn initial_seen(&self, loaded_seen: u64) -> u64 {
        if self.clear { 0 } else { loaded_seen }
    }

    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Self {
            weights: None,
            seed: 0,
            clear: false,
            dont_show: false,
            gpu: false,
            map: false,
            map_every: None,
            map_thresh: 0.005,
            map_iou: 0.5,
            save_every: 1000,
            dynamic_loss_scale: false,
            colour: None,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-clear" => options.clear = true,
                "-dont_show" => options.dont_show = true,
                "-colour" | "--colour" => options.colour = Some(true),
                "-nocolour" | "--nocolour" | "--no-colour" => options.colour = Some(false),
                "-seed" | "--seed" => {
                    let value = required_arg(args.next(), "seed value")?;
                    options.seed = value
                        .parse()
                        .map_err(|_| format!("invalid seed: {value} (expected u64)"))?;
                }
                "-gpus" => {
                    let value = required_arg(args.next(), "GPU list")?;
                    if value != "0" {
                        return Err("only -gpus 0 is supported; multi-GPU/device selection is not implemented".into());
                    }
                    options.gpu = true;
                }
                "-map" => options.map = true,
                "-dyn_scale" | "--dynamic_loss_scale" => options.dynamic_loss_scale = true,
                "-save_every" | "--save_every" => {
                    let value = required_arg(args.next(), "save interval")?;
                    let interval = value
                        .parse::<usize>()
                        .map_err(|_| "invalid -save_every interval")?;
                    if interval == 0 {
                        return Err("-save_every must be positive".into());
                    }
                    options.save_every = interval;
                }
                "-map_every" => {
                    let value = required_arg(args.next(), "validation interval")?;
                    let interval = value
                        .parse::<usize>()
                        .map_err(|_| "invalid -map_every interval")?;
                    if interval == 0 {
                        return Err("-map_every must be positive".into());
                    }
                    options.map_every = Some(interval);
                }
                "-thresh" | "-iou_thresh" => {
                    let value = required_arg(args.next(), "validation threshold")?;
                    let value: f32 = value.parse().map_err(|_| "invalid validation threshold")?;
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err("validation thresholds must be finite and in [0,1]".into());
                    }
                    if arg == "-thresh" {
                        options.map_thresh = value;
                    } else {
                        options.map_iou = value;
                    }
                }
                flag if flag.starts_with('-') => {
                    return Err(format!("unsupported detector train flag: {flag}").into());
                }
                _ if options.weights.is_none() => options.weights = Some(arg),
                _ => return Err(format!("unexpected detector train argument: {arg}").into()),
            }
        }
        if !options.map
            && (options.map_every.is_some()
                || options.map_thresh != 0.005
                || options.map_iou != 0.5)
        {
            return Err("validation options require -map".into());
        }
        Ok(options)
    }
}

#[derive(Debug, PartialEq)]
struct MapOptions {
    weights: Option<String>,
    confidence_threshold: f32,
    iou_threshold: f32,
    nms_threshold: f32,
    letterbox: bool,
}

impl MapOptions {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Self {
            weights: None,
            confidence_threshold: 0.005,
            iou_threshold: 0.5,
            nms_threshold: 0.45,
            letterbox: false,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-map" => {}
                "-letter_box" | "-letterbox" => options.letterbox = true,
                "-thresh" | "-iou_thresh" | "-nms" => {
                    let value = required_arg(args.next(), "mAP threshold")?;
                    let value = value
                        .parse::<f32>()
                        .map_err(|_| format!("invalid mAP threshold: {value}"))?;
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err("mAP thresholds must be finite and in [0,1]".into());
                    }
                    match arg.as_str() {
                        "-thresh" => options.confidence_threshold = value,
                        "-iou_thresh" => options.iou_threshold = value,
                        "-nms" => options.nms_threshold = value,
                        _ => unreachable!(),
                    }
                }
                flag if flag.starts_with('-') => {
                    return Err(format!("unsupported detector map flag: {flag}").into());
                }
                _ if options.weights.is_none() => options.weights = Some(arg),
                _ => return Err(format!("unexpected detector map argument: {arg}").into()),
            }
        }
        Ok(options)
    }
}

fn run_detector_map(
    cfg: &str,
    data_path: &str,
    options: &MapOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let (classes, image_count, names, report) = evaluate_detector_map(cfg, data_path, options)?;
    println!("mAP: {image_count} validation images, {classes} classes (CPU)");
    print!(
        "{}",
        format_map_report(&report, &names, options.iou_threshold)
    );
    Ok(())
}

fn evaluate_detector_map(
    cfg: &str,
    data_path: &str,
    options: &MapOptions,
) -> Result<(usize, usize, Vec<String>, darknet_core::MapReport), Box<dyn std::error::Error>> {
    let network = Network::from_path(cfg)?.with_batch(1);
    network.ensure_no_cuboid_heads("detector map")?;
    let mut executable = ExecutableNetwork::new(network);
    if let Some(weights) = &options.weights {
        let header = executable.load_weights_file(weights)?;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    }
    let data = DataConfig::from_path(data_path)?;
    let detection_options = DetectionOptions {
        confidence_threshold: options.confidence_threshold,
        nms_threshold: options.nms_threshold,
        letterbox: options.letterbox,
    };
    if executable.network.net.obb {
        let dataset = darknet_core::ObbValidationDataset::from_data(&executable.network, &data)?;
        let names = map_class_names(&data, dataset.classes)?;
        let report = dataset.evaluate_with_forward(
            &executable.network,
            detection_options,
            options.iou_threshold,
            |input| executable.forward(input),
        )?;
        Ok((dataset.classes, dataset.images.len(), names, report))
    } else {
        let dataset = darknet_core::ValidationDataset::from_data(&executable.network, &data)?;
        let names = map_class_names(&data, dataset.classes)?;
        let report = dataset.evaluate_with_forward(
            &executable.network,
            detection_options,
            options.iou_threshold,
            |input| executable.forward(input),
        )?;
        Ok((dataset.classes, dataset.images.len(), names, report))
    }
}

fn map_class_names(
    data: &DataConfig,
    classes: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let names = if let Some(path) = &data.names {
        load_names_file(path)?
    } else {
        (0..classes).map(|i| format!("class_{i}")).collect()
    };
    if names.len() != classes {
        return Err("-map names count differs from cfg classes".into());
    }
    Ok(names)
}

fn run_detector_train(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(cfg).exists() {
        return Err(format!("cfg path does not exist: {cfg}").into());
    }
    if !Path::new(data).exists() {
        return Err(format!("data path does not exist: {data}").into());
    }
    let network = Network::from_path(cfg)?;
    network.ensure_no_cuboid_heads("detector train")?;
    if network
        .layers
        .iter()
        .any(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
    {
        return Err("detector train does not support [obb_refine] until CARPN Stage-2 assignment, loss, backward, and updates are implemented".into());
    }

    // Colour must be configured before the first formatted training line.
    colour::configure(options.colour);

    if options.gpu && selected_backend()? != "oxide" {
        return Err("-gpus 0 requires DARKNET_BACKEND=oxide".into());
    }
    println!("detector train");
    println!("cfg {}", cfg);
    println!("data {}", data);
    if options.dont_show {
        println!("-dont_show: headless trainer (no GUI)");
    }
    println!("training seed {}", options.seed);
    run_detector_train_preferred_backend(cfg, data, options)
}

fn print_training_configuration(
    image_count: usize,
    batch: usize,
    subdivisions: usize,
    max_batches: usize,
    learning_rate: f32,
    momentum: f32,
    decay: f32,
) {
    println!("Learning Rate: {learning_rate:.8}");
    println!("Momentum: {momentum:.8}");
    println!("Decay: {decay:.8}");
    println!("Images: {image_count}");
    println!("Batch: {batch}");
    println!("Subdivisions: {subdivisions}");
    println!("Max Batches: {max_batches}");
}

struct TrainingProgress {
    started: Instant,
    first_batch: usize,
    avg_loss: Option<f32>,
}

impl TrainingProgress {
    fn new(first_batch: usize) -> Self {
        Self {
            started: Instant::now(),
            first_batch,
            avg_loss: None,
        }
    }

    fn print(
        &mut self,
        batch: usize,
        loss: f32,
        learning_rate: f32,
        images_seen: u64,
        max_batches: usize,
        iteration_duration: Duration,
        map_fields: &str,
    ) {
        let avg_loss = self
            .avg_loss
            .map_or(loss, |previous| previous * 0.9 + loss * 0.1);
        self.avg_loss = Some(avg_loss);

        let elapsed = self.started.elapsed().as_secs_f64();
        let completed = batch.saturating_sub(self.first_batch).max(1) as f64;
        let seconds_per_batch = elapsed / completed;
        let time_remaining = if max_batches > batch {
            format_seconds(seconds_per_batch * (max_batches - batch) as f64)
        } else {
            "unknown".to_owned()
        };

        // Keep the field names and ordering used by Darknet's detector trainer:
        // loss, avg loss, rate, elapsed training time, images, and ETA.  The
        // iteration is bright white and `loss` / `avg loss` use the C++
        // `format_loss` colouring (see src-lib/detector.cpp:608-619).
        println!(
            "{}",
            format_training_progress_line(
                batch,
                loss,
                avg_loss,
                learning_rate,
                images_seen,
                &time_remaining,
                iteration_duration.as_secs_f64() * 1000.0,
                map_fields,
            )
        );
    }
}

/// One `detector train` progress line, formatted like the C++ trainer: the
/// iteration in bright white and `loss` / `avg loss` coloured by
/// `Darknet::format_loss` (`src-lib/detector.cpp:608-619`).
fn format_training_progress_line(
    batch: usize,
    loss: f32,
    avg_loss: f32,
    learning_rate: f32,
    images_seen: u64,
    time_remaining: &str,
    train_milliseconds: f64,
    map_fields: &str,
) -> String {
    format!(
        "{}: loss={}, avg loss={}{map_fields}, rate={learning_rate:.8}, train={train_milliseconds:.3} milliseconds, {images_seen} images, time remaining={time_remaining}",
        colour::in_colour(colour::EColour::BrightWhite, &batch.to_string()),
        colour::format_loss(loss as f64),
        colour::format_loss(avg_loss as f64),
    )
}

fn format_seconds(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_owned();
    }
    if seconds < 60.0 {
        format!("{seconds:.0} seconds")
    } else {
        let minutes = (seconds / 60.0).floor();
        let remaining = (seconds - minutes * 60.0).round();
        format!("{minutes:.0} minutes {remaining:.0} seconds")
    }
}

#[cfg(feature = "oxide")]
fn run_detector_train_preferred_backend(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    if selected_backend()? != "oxide" {
        println!("cpu backend (set DARKNET_BACKEND=oxide for the GPU path)");
        return run_detector_train_cpu(cfg, data, options);
    }
    let backend =
        OxideBackend::new().map_err(|err| format!("requested oxide backend unavailable: {err}"))?;
    println!("gpu backend oxide device=0");
    run_detector_train_gpu(cfg, data, options, backend)
}

#[cfg(not(feature = "oxide"))]
fn run_detector_train_preferred_backend(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    if selected_backend()? == "oxide" {
        return Err(
            "requested oxide backend is not available; rebuild with --features oxide".into(),
        );
    }
    println!("cpu backend");
    run_detector_train_cpu(cfg, data, options)
}

#[derive(Debug)]
struct MapSchedule {
    interval: usize,
    burn_in: usize,
    last_iteration: usize,
}
impl MapSchedule {
    fn due(&self, iteration: usize, max_batches: usize) -> bool {
        (max_batches != 0 && iteration == max_batches)
            || (iteration >= self.burn_in
                && iteration >= self.last_iteration.saturating_add(self.interval))
    }
}

enum TrainingValidationDataset {
    AxisAligned(darknet_core::ValidationDataset),
    Oriented(darknet_core::ObbValidationDataset),
}

impl TrainingValidationDataset {
    fn classes(&self) -> usize {
        match self {
            Self::AxisAligned(dataset) => dataset.classes,
            Self::Oriented(dataset) => dataset.classes,
        }
    }

    fn image_count(&self) -> usize {
        match self {
            Self::AxisAligned(dataset) => dataset.images.len(),
            Self::Oriented(dataset) => dataset.images.len(),
        }
    }

    fn evaluate(
        &self,
        network: &Network,
        options: DetectionOptions,
        iou_threshold: f32,
        infer: impl FnMut(&Tensor) -> darknet_core::Result<ForwardResult>,
    ) -> darknet_core::Result<darknet_core::MapReport> {
        match self {
            Self::AxisAligned(dataset) => {
                dataset.evaluate_with_forward(network, options, iou_threshold, infer)
            }
            Self::Oriented(dataset) => {
                dataset.evaluate_with_forward(network, options, iou_threshold, infer)
            }
        }
    }
}

struct TrainingValidation {
    dataset: TrainingValidationDataset,
    names: Vec<String>,
    schedule: MapSchedule,
    detection_options: DetectionOptions,
    iou_threshold: f32,
    best_map: Option<f64>,
    last_map: Option<f64>,
    log: fs::File,
}
impl TrainingValidation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        options: &TrainOptions,
        network: &Network,
        data: &DataConfig,
        letterbox: bool,
        train_images: usize,
        images_per_iteration: usize,
        iteration: usize,
        backup: &Path,
        name: &str,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if !options.map {
            return Ok(None);
        }
        let dataset = if network.net.obb {
            TrainingValidationDataset::Oriented(darknet_core::ObbValidationDataset::from_data(
                network, data,
            )?)
        } else {
            TrainingValidationDataset::AxisAligned(darknet_core::ValidationDataset::from_data(
                network, data,
            )?)
        };
        let names = map_class_names(data, dataset.classes())?;
        let interval = options
            .map_every
            .unwrap_or_else(|| (train_images / images_per_iteration.max(1)).max(100));
        println!(
            "mAP: {} validation images, every {interval} iterations (after burn-in {}), and at final iteration",
            dataset.image_count(),
            network.net.burn_in
        );
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(backup.join(format!("{name}_map.csv")))?;
        if log.metadata()?.len() == 0 {
            writeln!(log, "iteration,seen,iou,threshold,map,best_map")?;
        }
        Ok(Some(Self {
            dataset,
            names,
            schedule: MapSchedule {
                interval,
                burn_in: network.net.burn_in,
                last_iteration: iteration,
            },
            detection_options: DetectionOptions {
                confidence_threshold: options.map_thresh,
                nms_threshold: 0.45,
                letterbox,
            },
            iou_threshold: options.map_iou,
            best_map: None,
            last_map: None,
            log,
        }))
    }

    fn record(
        &mut self,
        report: &darknet_core::MapReport,
        iteration: usize,
        seen: u64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let improved = self
            .best_map
            .is_none_or(|best| report.mean_average_precision >= best);
        if improved {
            self.best_map = Some(report.mean_average_precision);
        }
        self.last_map = Some(report.mean_average_precision);
        let best = self.best_map.unwrap_or(0.0);
        print!(
            "{}",
            format_map_report(report, &self.names, self.iou_threshold)
        );
        println!(
            "{iteration}: mAP@{:.2} = {}, best = {}",
            self.iou_threshold,
            colour::format_map_accuracy(report.mean_average_precision as f32),
            colour::format_map_accuracy(best as f32),
        );
        if improved {
            println!("New best mAP, saving weights!");
        }
        writeln!(
            self.log,
            "{iteration},{seen},{},{},{:.8},{:.8}",
            self.iou_threshold,
            self.detection_options.confidence_threshold,
            report.mean_average_precision,
            best
        )?;
        self.log.flush()?;
        self.schedule.last_iteration = iteration;
        Ok(improved)
    }
}

/// The `last=..., best=..., mAP=...` fields of the C++ training line.
fn map_progress_fields(validation: Option<&TrainingValidation>) -> String {
    let none = || colour::format_map_accuracy(-1.0);
    match validation {
        None => format!(", last={}, best={}, mAP=disabled", none(), none()),
        Some(v) => {
            let shown =
                |m: Option<f64>| m.map_or_else(none, |m| colour::format_map_accuracy(m as f32));
            let next = v
                .schedule
                .burn_in
                .max(v.schedule.last_iteration + v.schedule.interval);
            format!(
                ", last={}, best={}, mAP={next}",
                shown(v.last_map),
                shown(v.best_map)
            )
        }
    }
}

/// C++ `ostream` default float formatting (`%g`, 6 significant digits).
fn general(value: f64) -> String {
    if value == 0.0 || !value.is_finite() {
        return if value == 0.0 {
            "0".into()
        } else {
            value.to_string()
        };
    }
    let decimals = (5 - value.abs().log10().floor() as i32).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

/// Per-class table and summary lines of `validate_detector_map` (C++
/// `format_map_ap_row_values` and the `-> ...` summary).
fn format_map_report(report: &darknet_core::MapReport, names: &[String], iou: f32) -> String {
    use colour::EColour;
    let conf = darknet_core::REPORT_CONFIDENCE;
    let mut out = String::from(
        "\n\n AP=average precision, TP=true positive, TN=true negative,\n \
         FP=false positive, FN=false negative, GT=ground truth count\n\n  \
         Id Name                       AP      TP      FP      FN      GT       F1   AvgIoU\n  \
         -- -------------------- -------- ------- ------- ------- ------- -------- --------\n",
    );
    let (mut tp, mut fp, mut gt, mut iou_sum) = (0usize, 0usize, 0usize, 0.0f64);
    for class in 0..report.average_precision.len() {
        out += &format!(
            "  {} {} {} {} {} {} {} {} {}\n",
            colour::format_int(class, EColour::Normal, 2),
            colour::format_name(
                names.get(class).map_or("", String::as_str),
                EColour::BrightWhite,
                20
            ),
            colour::format_ratio(100.0 * report.average_precision[class] as f32, 8),
            colour::format_int(report.true_positives[class], EColour::Normal, 7),
            colour::format_int(report.false_positives[class], EColour::Normal, 7),
            colour::format_int(report.false_negatives[class], EColour::Normal, 7),
            colour::format_int(report.ground_truth_counts[class], EColour::Normal, 7),
            colour::format_ratio(100.0 * report.f1(class) as f32, 8),
            colour::format_ratio(100.0 * report.average_iou[class] as f32, 8),
        );
        tp += report.true_positives[class];
        fp += report.false_positives[class];
        gt += report.ground_truth_counts[class];
        iou_sum += report.average_iou[class] * report.true_positives[class] as f64;
    }
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if gt > 0 { tp as f64 / gt as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let avg_iou = if tp > 0 { iou_sum / tp as f64 } else { 0.0 };
    out += &format!(
        "\n-> for conf_thresh={conf}, precision={}, recall={}, F1 score={}\n\
         -> for conf_thresh={conf}, TP={tp}, FP={fp}, FN={}, average IoU={}%\n\
         -> IoU threshold={}%, used area-under-curve for each unique recall\n\
         -> mean average precision (mAP@{iou:.2})={}\n",
        general(precision),
        general(recall),
        general(f1),
        gt.saturating_sub(tp),
        general(avg_iou * 100.0),
        general(iou as f64 * 100.0),
        colour::format_map_accuracy(report.mean_average_precision as f32),
    );
    out
}

/// Appends `<name>_loss.csv` and keeps the points needed for `chart.png`.
struct TrainingLossLog {
    log: fs::File,
    chart_path: PathBuf,
    points: Vec<darknet_core::ChartPoint>,
}

impl TrainingLossLog {
    fn new(backup: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(backup.join(format!("{name}_loss.csv")))?;
        if log.metadata()?.len() == 0 {
            writeln!(
                log,
                "iteration,seen,loss,avg_loss,learning_rate,milliseconds"
            )?;
        }
        Ok(Self {
            log,
            chart_path: backup.join("chart.png"),
            points: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        iteration: usize,
        seen: u64,
        loss: f32,
        avg_loss: f32,
        learning_rate: f32,
        milliseconds: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(
            self.log,
            "{iteration},{seen},{loss:.6},{avg_loss:.6},{learning_rate:.8},{milliseconds:.3}"
        )?;
        self.log.flush()?;
        let map = self.points.last().and_then(|point| point.map);
        self.points.push(darknet_core::ChartPoint {
            iteration,
            avg_loss,
            map,
        });
        Ok(())
    }

    /// Attach the mAP measured for the most recent iteration to its chart point.
    fn set_map(&mut self, map: f64) {
        if let Some(point) = self.points.last_mut() {
            point.map = Some(map);
        }
    }

    fn write_chart(&self) -> Result<(), Box<dyn std::error::Error>> {
        darknet_core::write_training_chart(&self.chart_path, &self.points)?;
        Ok(())
    }

    /// Persist `_last.weights` (and the chart) after a periodic save.
    fn save_last(
        &self,
        network: &Network,
        params: &[LayerParams],
        backup: &Path,
        name: &str,
        seen: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let last = backup.join(format!("{name}_last.weights"));
        save_weights(network, params, &last, seen)?;
        println!("saved {}", last.display());
        self.write_chart()?;
        Ok(())
    }
}

#[derive(Default)]
struct ObbTrainingMetrics {
    anchor_class_output_sum: f64,
    anchor_class_count: usize,
    anchor_box_count: usize,
    iou_loss_sum: f64,
    giou_loss_sum: f64,
    truth_threshold_positive_cells: usize,
    total_bbox_count: usize,
    front_point_loss_sum: f64,
    front_point_count: usize,
    angle_sum_degrees: f64,
    angle_count: usize,
    angle_below_5: usize,
    angle_below_20: usize,
    angle_above_30: usize,
    classification_loss_per_image_sum: f64,
    source_cost_per_image_sum: f64,
}

impl ObbTrainingMetrics {
    fn add_batch(
        &mut self,
        batch: &darknet_core::ObbYoloTrainingDeltaBatch,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("OBB metrics require a nonempty subdivision batch".into());
        }
        let stats = &batch.stats;
        self.anchor_class_output_sum += f64::from(stats.anchor_class_stats.class_output_sum);
        self.iou_loss_sum += stats.iou_loss_sum;
        self.giou_loss_sum += stats.giou_loss_sum;
        self.front_point_loss_sum += f64::from(stats.front_point_loss.loss_sum);
        let per_image = 1.0 / batch_size as f64;
        self.classification_loss_per_image_sum +=
            f64::from(batch.classification_loss_component) * per_image;
        self.source_cost_per_image_sum += f64::from(batch.source_loss_cost) * per_image;

        for (count, increment, name) in [
            (
                &mut self.anchor_class_count,
                stats.anchor_class_stats.positive_count,
                "anchor class",
            ),
            (
                &mut self.anchor_box_count,
                stats.anchor_box_positive_count,
                "anchor box",
            ),
            (
                &mut self.truth_threshold_positive_cells,
                stats.truth_threshold_positive_cells,
                "truth-threshold positive",
            ),
            (
                &mut self.total_bbox_count,
                stats.total_bbox_count,
                "total bbox",
            ),
            (
                &mut self.front_point_count,
                stats.front_point_loss.count,
                "front-point",
            ),
            (
                &mut self.angle_count,
                stats.front_point_loss.angle_count,
                "angle",
            ),
            (
                &mut self.angle_below_5,
                stats.front_point_loss.angle_below_5,
                "angle below 5 degrees",
            ),
            (
                &mut self.angle_below_20,
                stats.front_point_loss.angle_below_20,
                "angle below 20 degrees",
            ),
            (
                &mut self.angle_above_30,
                stats.front_point_loss.angle_above_30,
                "angle above 30 degrees",
            ),
        ] {
            *count = count.checked_add(increment).ok_or_else(|| {
                std::io::Error::other(format!("OBB {name} metrics count overflow"))
            })?;
        }
        self.angle_sum_degrees += stats.front_point_loss.angle_sum_degrees;
        if [
            self.anchor_class_output_sum,
            self.iou_loss_sum,
            self.giou_loss_sum,
            self.front_point_loss_sum,
            self.classification_loss_per_image_sum,
            self.source_cost_per_image_sum,
            self.angle_sum_degrees,
        ]
        .into_iter()
        .any(|sum| !sum.is_finite())
        {
            return Err("OBB training metrics became non-finite".into());
        }
        Ok(())
    }

    fn average_iou(&self) -> Option<f64> {
        (self.anchor_box_count > 0).then(|| 1.0 - self.iou_loss_sum / self.anchor_box_count as f64)
    }

    fn average_giou(&self) -> Option<f64> {
        (self.anchor_box_count > 0).then(|| 1.0 - self.giou_loss_sum / self.anchor_box_count as f64)
    }

    fn average_class_output(&self) -> Option<f64> {
        (self.anchor_class_count > 0)
            .then(|| self.anchor_class_output_sum / self.anchor_class_count as f64)
    }

    fn average_front_point_loss(&self) -> Option<f64> {
        (self.front_point_count > 0)
            .then(|| self.front_point_loss_sum / self.front_point_count as f64)
    }

    fn angle_accuracy_below_20_percent(&self) -> Option<f64> {
        (self.angle_count > 0).then(|| 100.0 * self.angle_below_20 as f64 / self.angle_count as f64)
    }

    fn average_angle_error_degrees(&self) -> Option<f64> {
        (self.angle_count > 0).then(|| self.angle_sum_degrees / self.angle_count as f64)
    }
}

fn csv_metric(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(String::new, |value| format!("{value:.6}"))
}

struct ObbTrainingMetricsLog {
    log: fs::File,
}

impl ObbTrainingMetricsLog {
    fn new(backup: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(backup.join(format!("{name}_obb_metrics.csv")))?;
        if log.metadata()?.len() == 0 {
            writeln!(
                log,
                "iteration,seen,avg_obb_iou,avg_giou,anchor_box_count,truth_threshold_positive_cells,total_bbox_count,avg_cat,anchor_class_count,classification_loss_per_image,cost_minus_classification_per_image,front_point_loss,front_point_count,angle_accuracy_below_20_percent,mean_angle_error_degrees,angle_count,angle_below_5,angle_below_20,angle_above_30"
            )?;
        }
        Ok(Self { log })
    }

    fn record(
        &mut self,
        iteration: usize,
        seen: u64,
        metrics: &ObbTrainingMetrics,
        subdivisions: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let divisor = subdivisions.max(1) as f64;
        let classification_loss = metrics.classification_loss_per_image_sum / divisor;
        let source_cost = metrics.source_cost_per_image_sum / divisor;
        writeln!(
            self.log,
            "{iteration},{seen},{},{},{},{},{},{},{},{classification_loss:.6},{:.6},{},{},{},{},{},{},{},{}",
            csv_metric(metrics.average_iou()),
            csv_metric(metrics.average_giou()),
            metrics.anchor_box_count,
            metrics.truth_threshold_positive_cells,
            metrics.total_bbox_count,
            csv_metric(metrics.average_class_output()),
            metrics.anchor_class_count,
            source_cost - classification_loss,
            csv_metric(metrics.average_front_point_loss()),
            metrics.front_point_count,
            csv_metric(metrics.angle_accuracy_below_20_percent()),
            csv_metric(metrics.average_angle_error_degrees()),
            metrics.angle_count,
            metrics.angle_below_5,
            metrics.angle_below_20,
            metrics.angle_above_30,
        )?;
        self.log.flush()?;
        Ok(())
    }
}

/// Background batch loader: decodes/augments the next batch on a worker thread so
/// the training step does not wait on it.  The caller drives a one-deep
/// lookahead by requesting the next subdivision before processing the current.
enum PrefetchRequest {
    Load {
        network: std::sync::Arc<Network>,
        paths: std::sync::Arc<Vec<PathBuf>>,
        start_index: usize,
        sample_start: u64,
    },
    Stop,
}

struct BatchPrefetcher {
    requests: std::sync::mpsc::SyncSender<PrefetchRequest>,
    responses: std::sync::mpsc::Receiver<Result<TrainingBatch, String>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Bounded thread pool for the image loader.
///
/// C++ uses `6 * ngpus` loader threads; leave a couple of cores for the
/// trainer / GPU-feeding thread.  `DARKNET_LOADER_THREADS` overrides the size.
fn build_loader_pool() -> rayon::ThreadPool {
    let threads = std::env::var("DARKNET_LOADER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|parallelism| parallelism.get().saturating_sub(2).clamp(1, 6))
                .unwrap_or(4)
        });
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("darknet-load-{index}"))
        .build()
        .expect("build image loader thread pool")
}

impl BatchPrefetcher {
    fn new(data: DataConfig, augmentation: DetectorAugmentation, seed: u64) -> Self {
        let pool = build_loader_pool();
        let (requests, request_rx) = std::sync::mpsc::sync_channel(8);
        let (response_tx, responses) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("darknet-prefetch".to_owned())
            .spawn(move || {
                while let Ok(request) = request_rx.recv() {
                    match request {
                        PrefetchRequest::Stop => break,
                        PrefetchRequest::Load {
                            network,
                            paths,
                            start_index,
                            sample_start,
                        } => {
                            // Run the decode/augment fan-out on a bounded pool so
                            // the GPU-feeding thread keeps cores free.
                            let result = pool
                                .install(|| {
                                    load_augmented_training_batch_from_paths(
                                        network.as_ref(),
                                        &data,
                                        &paths,
                                        start_index,
                                        &augmentation,
                                        seed,
                                        sample_start,
                                    )
                                })
                                .map_err(|err| err.to_string());
                            if response_tx.send(result).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
            .expect("spawn prefetch worker");
        Self {
            requests,
            responses,
            handle: Some(handle),
        }
    }

    fn request(
        &self,
        network: std::sync::Arc<Network>,
        paths: std::sync::Arc<Vec<PathBuf>>,
        start_index: usize,
        sample_start: u64,
    ) {
        let _ = self.requests.send(PrefetchRequest::Load {
            network,
            paths,
            start_index,
            sample_start,
        });
    }

    fn take(&self) -> Result<TrainingBatch, Box<dyn std::error::Error>> {
        match self.responses.recv() {
            Ok(Ok(batch)) => Ok(batch),
            Ok(Err(message)) => Err(message.into()),
            Err(_) => Err("prefetch worker stopped".into()),
        }
    }
}

impl Drop for BatchPrefetcher {
    fn drop(&mut self) {
        let _ = self.requests.send(PrefetchRequest::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_detector_train_cpu(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_config = DataConfig::from_path(data)?;
    let augmentation = DetectorAugmentation::from_config(&ConfigFile::parse_file(cfg)?)?;
    let executable = ExecutableNetwork::from_path(cfg)?;
    if executable.network.net.obb {
        return run_detector_train_cpu_obb(cfg, executable, data_config, augmentation, options);
    }
    let mut executable = executable;
    initialize_training_params(&executable.network, &mut executable.params, options.seed);
    let mut seen = 0u64;
    if let Some(weights) = options.weights.as_deref() {
        let header = executable.load_weights_file(weights)?;
        seen = header.seen;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    } else {
        println!("running with seeded random weights");
    }
    seen = options.initial_seen(seen);
    if options.clear {
        println!("-clear: reset seen and iteration");
    }

    let train_images = load_paths_file(data_config.required_train_list()?)?;
    println!(
        "loaded {} images from {}",
        train_images.len(),
        data_config.required_train_list()?.display()
    );
    print_training_configuration(
        train_images.len(),
        executable.network.net.batch,
        executable.network.net.subdivisions,
        executable.network.net.max_batches,
        executable.network.net.learning_rate,
        executable.network.net.momentum,
        executable.network.net.decay,
    );
    if train_images.is_empty() {
        return Err("training list is empty".into());
    }

    let backup_dir = data_config
        .backup
        .clone()
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&backup_dir)?;
    let base_name = cfg_base_name(cfg);
    let optimizer_state_path = backup_dir.join(format!("{base_name}_optimizer.state"));
    if executable.network.net.muon && !options.clear && optimizer_state_path.exists() {
        executable.load_optimizer_state(&optimizer_state_path)?;
        println!(
            "resumed optimizer state (Adam step {}) from {}",
            executable.adam_step(),
            optimizer_state_path.display()
        );
    }
    let images_per_iteration =
        executable.network.net.batch.max(1) * executable.network.net.subdivisions.max(1);
    let mut batch_num = (seen as usize)
        .checked_div(images_per_iteration)
        .unwrap_or(0);
    let mut validation = TrainingValidation::new(
        options,
        &executable.network,
        &data_config,
        augmentation.letter_box,
        train_images.len(),
        images_per_iteration,
        batch_num,
        &backup_dir,
        &base_name,
    )?;
    let mut yolo_state = YoloTrainingState::default();
    let mut progress = TrainingProgress::new(batch_num);
    let mut epoch = if train_images.is_empty() {
        0usize
    } else {
        (seen as usize) / train_images.len()
    };
    let mut epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
    let mut image_cursor = if train_images.is_empty() {
        0usize
    } else {
        (seen as usize) % train_images.len()
    };
    let mut loss_log = TrainingLossLog::new(&backup_dir, &base_name)?;
    let mut prefetch_network = std::sync::Arc::new(executable.network.clone());
    let prefetcher = BatchPrefetcher::new(data_config.clone(), augmentation.clone(), options.seed);

    let base_width = executable.network.net.width;
    let base_height = executable.network.net.height;
    let mut multi_scale_count = 0usize;
    // Shape of the batch that has been requested but not yet consumed. `None`
    // until the lookahead pipeline is primed.
    let mut pending_shape: Option<(usize, usize, usize)> = None;

    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        if augmentation.random > 0.0 && multi_scale_count % 10 == 0 {
            let (dim_w, dim_h) = augmentation.multi_scale_dimensions(
                base_width,
                base_height,
                options.seed,
                batch_num,
                executable.network.net.max_batches,
            );
            if dim_w != executable.network.net.width || dim_h != executable.network.net.height {
                executable.network = executable.network.with_input_size(dim_w, dim_h)?;
            }
        }
        multi_scale_count += 1;
        let iteration_start = Instant::now();
        let lr = compute_learning_rate(&executable.network.net, batch_num);
        executable.set_train_iteration(batch_num);
        let mut iter_loss = 0.0f32;
        let current_shape = (
            executable.network.net.width,
            executable.network.net.height,
            executable.network.net.batch,
        );
        if prefetch_network.net.width != executable.network.net.width
            || prefetch_network.net.height != executable.network.net.height
            || prefetch_network.net.batch != executable.network.net.batch
        {
            prefetch_network = std::sync::Arc::new(executable.network.clone());
        }
        match pending_shape {
            None => {
                // Prime the pipeline: request this iteration's first batch.
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor,
                    seen,
                );
                pending_shape = Some(current_shape);
            }
            Some(shape) if shape != current_shape => {
                // A multi-scale resize invalidated the batch already in flight:
                // discard it and re-request at the new size (C++ joins and
                // restarts the loader thread here).
                let _ = prefetcher.take();
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor,
                    seen,
                );
                pending_shape = Some(current_shape);
            }
            Some(_) => {}
        }
        for subdivision in 0..executable.network.net.subdivisions.max(1) {
            let batch = prefetcher.take()?;
            if subdivision + 1 < executable.network.net.subdivisions.max(1) {
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor + (subdivision + 1) * executable.network.net.batch.max(1),
                    seen + ((subdivision + 1) * executable.network.net.batch.max(1)) as u64,
                );
            } else {
                // One-batch lookahead: begin loading the next iteration's first
                // batch while this iteration is still being trained.
                let (next_order, next_cursor) = advanced_epoch_images(
                    &train_images,
                    options.seed,
                    epoch,
                    &epoch_images,
                    image_cursor,
                    images_per_iteration,
                );
                prefetcher.request(
                    prefetch_network.clone(),
                    next_order,
                    next_cursor,
                    seen + images_per_iteration as u64,
                );
            }
            pending_shape = Some(current_shape);
            let forward = executable.forward_training(&batch.input)?;
            let yolo = compute_yolo_training_loss_with_state(
                &executable.network,
                &forward,
                &batch.labels,
                batch_num,
                Some(&mut yolo_state),
            )?;
            iter_loss += yolo.loss;
            executable.backward_accumulate_seeded_with_hf_auxiliary_gradients(
                &batch.input,
                &forward,
                &yolo.layer_output_gradient_seeds,
                &batch.labels,
            )?;
        }
        if executable.network.net.muon {
            executable.update_weights_muon(lr, images_per_iteration)?;
        } else {
            executable.update_weights(
                lr,
                executable.network.net.momentum,
                executable.network.net.decay,
                // C++ update_network: batch * subdivisions (gradients are summed
                // over every subdivision before the step).
                images_per_iteration,
            );
        }

        batch_num += 1;
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            epoch += 1;
            epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
            image_cursor %= train_images.len();
        }

        let loss = iter_loss / images_per_iteration.max(1) as f32;
        progress.print(
            batch_num,
            loss,
            lr,
            seen,
            executable.network.net.max_batches,
            iteration_start.elapsed(),
            &map_progress_fields(validation.as_ref()),
        );
        loss_log.record(
            batch_num,
            seen,
            loss,
            progress.avg_loss.unwrap_or(loss),
            lr,
            iteration_start.elapsed().as_secs_f64() * 1000.0,
        )?;

        if let Some(validation) = &mut validation {
            if validation
                .schedule
                .due(batch_num, executable.network.net.max_batches)
            {
                println!(
                    "Validating current training weights on all {} images...",
                    validation.dataset.image_count()
                );
                // Separate batch-one inference graph, same current parameters;
                // never alter the training graph, caches or momentum buffers.
                let mut inference = ExecutableNetwork::new(executable.network.with_batch(1));
                inference.params = executable.params.clone();
                let report = validation.dataset.evaluate(
                    &inference.network,
                    validation.detection_options,
                    validation.iou_threshold,
                    |input| inference.forward(input),
                )?;
                loss_log.set_map(report.mean_average_precision);
                if validation.record(&report, batch_num, seen)? {
                    let checkpoint = backup_dir.join(format!("{base_name}_best.weights"));
                    save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
                    if executable.network.net.muon {
                        executable.save_optimizer_state(&optimizer_state_path)?;
                    }
                    println!("saved {}", checkpoint.display());
                }
            }
        }

        if options.save_every != 0 && batch_num % options.save_every == 0 {
            let checkpoint = backup_dir.join(format!("{base_name}_{batch_num}.weights"));
            save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
            println!("saved {}", checkpoint.display());
            loss_log.save_last(
                &executable.network,
                &executable.params,
                &backup_dir,
                &base_name,
                seen,
            )?;
            if executable.network.net.muon {
                executable.save_optimizer_state(&optimizer_state_path)?;
            }
        }

        if executable.network.net.max_batches != 0
            && batch_num >= executable.network.net.max_batches
        {
            break;
        }
    }

    let final_checkpoint = backup_dir.join(format!("{base_name}_final.weights"));
    save_weights(
        &executable.network,
        &executable.params,
        &final_checkpoint,
        seen,
    )?;
    println!("saved {}", final_checkpoint.display());
    if executable.network.net.muon {
        executable.save_optimizer_state(&optimizer_state_path)?;
    }
    loss_log.save_last(
        &executable.network,
        &executable.params,
        &backup_dir,
        &base_name,
        seen,
    )?;
    Ok(())
}

fn run_detector_train_cpu_obb(
    cfg: &str,
    mut executable: ExecutableNetwork,
    data_config: DataConfig,
    augmentation: DetectorAugmentation,
    options: &TrainOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    if options.dynamic_loss_scale {
        return Err("OBB CLI dynamic loss scaling requires the Oxide training path".into());
    }
    validate_obb_augmentation(&augmentation)?;

    initialize_training_params(&executable.network, &mut executable.params, options.seed);
    let mut seen = 0u64;
    if let Some(weights) = options.weights.as_deref() {
        let header = executable.load_weights_file(weights)?;
        seen = header.seen;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    } else {
        println!("running with seeded random weights");
    }
    seen = options.initial_seen(seen);
    if options.clear {
        println!("-clear: reset seen and iteration");
    }
    let train_images = load_paths_file(data_config.required_train_list()?)?;
    println!(
        "loaded {} images from {}",
        train_images.len(),
        data_config.required_train_list()?.display()
    );
    print_training_configuration(
        train_images.len(),
        executable.network.net.batch,
        executable.network.net.subdivisions,
        executable.network.net.max_batches,
        executable.network.net.learning_rate,
        executable.network.net.momentum,
        executable.network.net.decay,
    );
    if train_images.is_empty() {
        return Err("training list is empty".into());
    }

    let backup_dir = data_config
        .backup
        .clone()
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&backup_dir)?;
    let base_name = cfg_base_name(cfg);
    let optimizer_state_path = backup_dir.join(format!("{base_name}_optimizer.state"));
    if (executable.network.net.adam || executable.network.net.muon)
        && !options.clear
        && optimizer_state_path.exists()
    {
        executable.load_optimizer_state(&optimizer_state_path)?;
        println!(
            "resumed optimizer state (Adam step {}) from {}",
            executable.adam_step(),
            optimizer_state_path.display()
        );
    }
    let batch_size = executable.network.net.batch.max(1);
    let subdivisions = executable.network.net.subdivisions.max(1);
    let images_per_iteration = batch_size
        .checked_mul(subdivisions)
        .ok_or_else(|| std::io::Error::other("OBB CLI images-per-iteration count overflow"))?;
    let mut batch_num = (seen as usize) / images_per_iteration;
    let mut validation = TrainingValidation::new(
        options,
        &executable.network,
        &data_config,
        augmentation.letter_box,
        train_images.len(),
        images_per_iteration,
        batch_num,
        &backup_dir,
        &base_name,
    )?;
    let mut progress = TrainingProgress::new(batch_num);
    let mut epoch = (seen as usize) / train_images.len();
    let mut epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
    let mut image_cursor = (seen as usize) % train_images.len();
    let mut loss_log = TrainingLossLog::new(&backup_dir, &base_name)?;
    let mut metrics_log = ObbTrainingMetricsLog::new(&backup_dir, &base_name)?;

    println!(
        "OBB CPU trainer: {} with direct resize and seeded horizontal flips",
        if executable.network.net.muon {
            "Muon/AdamW"
        } else if executable.network.net.adam {
            "Adam"
        } else {
            "SGD"
        }
    );
    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        let iteration_start = Instant::now();
        let learning_rate = compute_learning_rate(&executable.network.net, batch_num);
        executable.set_train_iteration(batch_num);
        executable.zero_train_buffers();
        let mut iteration_loss = 0.0f32;
        let mut iteration_metrics = ObbTrainingMetrics::default();
        for subdivision in 0..subdivisions {
            let sample_offset = subdivision.checked_mul(batch_size).ok_or_else(|| {
                std::io::Error::other("OBB CLI subdivision sample offset overflow")
            })?;
            let start_index = image_cursor
                .checked_add(sample_offset)
                .ok_or_else(|| std::io::Error::other("OBB CLI image cursor overflow"))?;
            let sample_start = seen
                .checked_add(u64::try_from(sample_offset)?)
                .ok_or_else(|| std::io::Error::other("OBB CLI sample ordinal overflow"))?;
            let batch = load_obb_training_batch_with_flips_from_paths(
                &executable.network,
                &data_config,
                &epoch_images,
                start_index,
                &augmentation,
                options.seed,
                sample_start,
            )?;
            let forward = executable.forward_training(&batch.input)?;
            let delta_batch =
                compute_obb_yolo_training_deltas(&executable.network, &forward, &batch.labels)?;
            iteration_metrics.add_batch(&delta_batch, batch_size)?;
            iteration_loss += delta_batch.source_loss_cost / batch_size as f32;
            if !iteration_loss.is_finite() {
                return Err("OBB CLI iteration loss became non-finite".into());
            }
            executable.backward_accumulate_seeded(
                &batch.input,
                &forward,
                &delta_batch.layer_output_gradient_seeds,
            )?;
        }
        if executable.network.net.muon {
            executable.update_weights_muon(learning_rate, images_per_iteration)?;
        } else if executable.network.net.adam {
            executable.update_weights_adam(learning_rate, images_per_iteration)?;
        } else {
            executable.update_weights(
                learning_rate,
                executable.network.net.momentum,
                executable.network.net.decay,
                images_per_iteration,
            );
        }

        batch_num += 1;
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            epoch += 1;
            epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
            image_cursor %= train_images.len();
        }
        let loss = iteration_loss / subdivisions as f32;
        progress.print(
            batch_num,
            loss,
            learning_rate,
            seen,
            executable.network.net.max_batches,
            iteration_start.elapsed(),
            &map_progress_fields(validation.as_ref()),
        );
        loss_log.record(
            batch_num,
            seen,
            loss,
            progress.avg_loss.unwrap_or(loss),
            learning_rate,
            iteration_start.elapsed().as_secs_f64() * 1000.0,
        )?;
        metrics_log.record(batch_num, seen, &iteration_metrics, subdivisions)?;

        if let Some(validation) = &mut validation {
            if validation
                .schedule
                .due(batch_num, executable.network.net.max_batches)
            {
                println!(
                    "Validating current training weights on all {} images...",
                    validation.dataset.image_count()
                );
                let mut inference = ExecutableNetwork::new(executable.network.with_batch(1));
                inference.params = executable.params.clone();
                let report = validation.dataset.evaluate(
                    &inference.network,
                    validation.detection_options,
                    validation.iou_threshold,
                    |input| inference.forward(input),
                )?;
                loss_log.set_map(report.mean_average_precision);
                if validation.record(&report, batch_num, seen)? {
                    let checkpoint = backup_dir.join(format!("{base_name}_best.weights"));
                    save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
                    if executable.network.net.muon {
                        executable.save_optimizer_state(&optimizer_state_path)?;
                    }
                    println!("saved {}", checkpoint.display());
                }
            }
        }

        if options.save_every != 0 && batch_num % options.save_every == 0 {
            let checkpoint = backup_dir.join(format!("{base_name}_{batch_num}.weights"));
            save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
            println!("saved {}", checkpoint.display());
            loss_log.save_last(
                &executable.network,
                &executable.params,
                &backup_dir,
                &base_name,
                seen,
            )?;
            if executable.network.net.adam || executable.network.net.muon {
                executable.save_optimizer_state(&optimizer_state_path)?;
            }
        }
        if executable.network.net.max_batches != 0
            && batch_num >= executable.network.net.max_batches
        {
            break;
        }
    }

    let final_checkpoint = backup_dir.join(format!("{base_name}_final.weights"));
    save_weights(
        &executable.network,
        &executable.params,
        &final_checkpoint,
        seen,
    )?;
    println!("saved {}", final_checkpoint.display());
    loss_log.save_last(
        &executable.network,
        &executable.params,
        &backup_dir,
        &base_name,
        seen,
    )?;
    if executable.network.net.adam || executable.network.net.muon {
        executable.save_optimizer_state(&optimizer_state_path)?;
    }
    Ok(())
}

#[cfg(feature = "oxide")]
fn print_train_profile(samples: &[[f64; 5]], iteration: usize, at_end: bool) {
    if samples.is_empty() {
        return;
    }
    let phases = [
        "data_wait",
        "forward_d2h",
        "yolo_host",
        "backward_h2d",
        "update",
    ];
    let summary = phases
        .iter()
        .enumerate()
        .map(|(phase, name)| {
            let mut values: Vec<f64> = samples.iter().map(|sample| sample[phase]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.sort_by(f64::total_cmp);
            let mid = values.len() / 2;
            let median = (values[(values.len() - 1) / 2] + values[mid]) / 2.0;
            format!("{name}={median:.3}/{mean:.3}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let end = if at_end { " end" } else { "" };
    println!(
        "TRAIN_PROFILE{end} iteration={iteration} window_iters={} ms median/mean: {summary}",
        samples.len()
    );
}

#[cfg(feature = "oxide")]
fn run_detector_train_gpu(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
    backend: OxideBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_config = DataConfig::from_path(data)?;
    let augmentation = DetectorAugmentation::from_config(&ConfigFile::parse_file(cfg)?)?;
    let mut executable = GpuExecutableNetwork::from_path(cfg, backend)?;
    if executable.network.net.obb {
        return Err("OBB detector CLI training is currently CPU-only".into());
    }
    initialize_training_params(&executable.network, &mut executable.params, options.seed);
    let mut seen = 0u64;
    if let Some(weights) = options.weights.as_deref() {
        let header = executable.load_weights_file(weights)?;
        seen = header.seen;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    } else {
        println!("running with seeded random weights");
    }
    seen = options.initial_seen(seen);
    if options.clear {
        println!("-clear: reset seen and iteration");
    }
    executable.init_train_buffers_gpu()?;
    println!("allocated GPU params and train buffers");

    let train_images = load_paths_file(data_config.required_train_list()?)?;
    println!(
        "loaded {} images from {}",
        train_images.len(),
        data_config.required_train_list()?.display()
    );
    print_training_configuration(
        train_images.len(),
        executable.network.net.batch,
        executable.network.net.subdivisions,
        executable.network.net.max_batches,
        executable.network.net.learning_rate,
        executable.network.net.momentum,
        executable.network.net.decay,
    );
    if train_images.is_empty() {
        return Err("training list is empty".into());
    }

    let backup_dir = data_config
        .backup
        .clone()
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&backup_dir)?;
    let base_name = cfg_base_name(cfg);
    let optimizer_state_path = backup_dir.join(format!("{base_name}_optimizer.state"));
    if !options.clear && optimizer_state_path.exists() {
        executable.load_optimizer_state(&optimizer_state_path)?;
        println!(
            "resumed optimizer state (Adam step {}) from {}",
            executable.adam_step(),
            optimizer_state_path.display()
        );
    }
    let mut loss_scale_growth = 0u32;
    let mut profile = (std::env::var("DARKNET_TRAIN_PROFILE").as_deref() == Ok("1"))
        .then(|| Vec::<[f64; 5]>::with_capacity(20));
    if profile.is_some() {
        executable.backend().synchronize()?;
    }
    let images_per_iteration =
        executable.network.net.batch.max(1) * executable.network.net.subdivisions.max(1);
    let mut batch_num = if images_per_iteration == 0 {
        0usize
    } else {
        (seen as usize) / images_per_iteration
    };
    let mut validation = TrainingValidation::new(
        options,
        &executable.network,
        &data_config,
        augmentation.letter_box,
        train_images.len(),
        images_per_iteration,
        batch_num,
        &backup_dir,
        &base_name,
    )?;
    let mut yolo_state = YoloTrainingState::default();
    let mut progress = TrainingProgress::new(batch_num);
    let mut epoch = if train_images.is_empty() {
        0usize
    } else {
        (seen as usize) / train_images.len()
    };
    let mut epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
    let mut image_cursor = if train_images.is_empty() {
        0usize
    } else {
        (seen as usize) % train_images.len()
    };
    let mut loss_log = TrainingLossLog::new(&backup_dir, &base_name)?;
    let mut prefetch_network = std::sync::Arc::new(executable.network.clone());
    let prefetcher = BatchPrefetcher::new(data_config.clone(), augmentation.clone(), options.seed);

    let base_width = executable.network.net.width;
    let base_height = executable.network.net.height;
    let mut multi_scale_count = 0usize;
    // Shape of the batch that has been requested but not yet consumed. `None`
    // until the lookahead pipeline is primed.
    let mut pending_shape: Option<(usize, usize, usize)> = None;

    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        if augmentation.random > 0.0 && multi_scale_count % 10 == 0 {
            let (dim_w, dim_h) = augmentation.multi_scale_dimensions(
                base_width,
                base_height,
                options.seed,
                batch_num,
                executable.network.net.max_batches,
            );
            if dim_w != executable.network.net.width || dim_h != executable.network.net.height {
                executable.network = executable.network.with_input_size(dim_w, dim_h)?;
            }
        }
        multi_scale_count += 1;
        let iteration_start = Instant::now();
        let lr = compute_learning_rate(&executable.network.net, batch_num);
        executable.set_train_iteration(batch_num);
        let mut iter_loss = 0.0f32;
        let mut phase_ms = [0.0; 5];
        if let Some(samples) = &mut profile {
            if samples.len() == 20 {
                samples.clear();
            }
        }
        // NOTE: do NOT zero train buffers here. After update_weights_gpu, the buffers
        // hold momentum * previous_gradient (C++ weight_updates *= momentum). Zeroing
        // would destroy the momentum carry. init_train_buffers_gpu() above initialises
        // them to zero only once at startup, which is correct.
        let current_shape = (
            executable.network.net.width,
            executable.network.net.height,
            executable.network.net.batch,
        );
        if prefetch_network.net.width != executable.network.net.width
            || prefetch_network.net.height != executable.network.net.height
            || prefetch_network.net.batch != executable.network.net.batch
        {
            prefetch_network = std::sync::Arc::new(executable.network.clone());
        }
        match pending_shape {
            None => {
                // Prime the pipeline: request this iteration's first batch.
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor,
                    seen,
                );
                pending_shape = Some(current_shape);
            }
            Some(shape) if shape != current_shape => {
                // A multi-scale resize invalidated the batch already in flight:
                // discard it and re-request at the new size (C++ joins and
                // restarts the loader thread here).
                let _ = prefetcher.take();
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor,
                    seen,
                );
                pending_shape = Some(current_shape);
            }
            Some(_) => {}
        }
        for subdivision in 0..executable.network.net.subdivisions.max(1) {
            let start = profile.as_ref().map(|_| Instant::now());
            let batch = prefetcher.take()?;
            if let Some(start) = start {
                phase_ms[0] += start.elapsed().as_secs_f64() * 1000.0;
            }
            if subdivision + 1 < executable.network.net.subdivisions.max(1) {
                prefetcher.request(
                    prefetch_network.clone(),
                    epoch_images.clone(),
                    image_cursor + (subdivision + 1) * executable.network.net.batch.max(1),
                    seen + ((subdivision + 1) * executable.network.net.batch.max(1)) as u64,
                );
            } else {
                // One-batch lookahead: begin loading the next iteration's first
                // batch while this iteration is still being trained.
                let (next_order, next_cursor) = advanced_epoch_images(
                    &train_images,
                    options.seed,
                    epoch,
                    &epoch_images,
                    image_cursor,
                    images_per_iteration,
                );
                prefetcher.request(
                    prefetch_network.clone(),
                    next_order,
                    next_cursor,
                    seen + images_per_iteration as u64,
                );
            }
            pending_shape = Some(current_shape);
            let start = profile.as_ref().map(|_| Instant::now());
            let forward = executable.forward_training_device(&batch.input)?;
            if let Some(start) = start {
                // Head D2H is included; profiling-only sync also covers any trailing GPU work.
                executable.backend().synchronize()?;
                phase_ms[1] += start.elapsed().as_secs_f64() * 1000.0;
            }
            let start = profile.as_ref().map(|_| Instant::now());
            let yolo = compute_yolo_training_loss_with_state(
                &executable.network,
                &forward.cpu_forward,
                &batch.labels,
                batch_num,
                Some(&mut yolo_state),
            )?;
            if let Some(start) = start {
                phase_ms[2] += start.elapsed().as_secs_f64() * 1000.0;
            }
            iter_loss += yolo.loss;
            let start = profile.as_ref().map(|_| Instant::now());
            executable
                .backward_accumulate_seeded_gpu(&forward, &yolo.layer_output_gradient_seeds)?;
            if let Some(start) = start {
                // Backward (including seed H2D) and deferred update are asynchronous:
                // synchronize only in profiling mode so host timings include GPU execution.
                executable.backend().synchronize()?;
                phase_ms[3] += start.elapsed().as_secs_f64() * 1000.0;
            }
        }
        let start = profile.as_ref().map(|_| Instant::now());
        if options.dynamic_loss_scale && executable.gradients_non_finite()? {
            // Overflow: drop the accumulated gradients, skip the step and
            // halve the scale so the next backward stays representable.
            executable.zero_train_buffers_gpu()?;
            let next = (executable.loss_scale() * 0.5).max(1.0);
            executable.set_loss_scale(next);
            loss_scale_growth = 0;
            println!("{batch_num}: non-finite gradients; loss scale -> {next}");
        } else {
            executable.update_weights_gpu_deferred(
                lr,
                executable.network.net.momentum,
                executable.network.net.decay,
                // C++ update_network: batch * subdivisions (gradients are summed
                // over every subdivision before the step).
                images_per_iteration,
            )?;
            if options.dynamic_loss_scale {
                loss_scale_growth += 1;
                if loss_scale_growth >= 1000 && executable.loss_scale() < 65536.0 {
                    let next = executable.loss_scale() * 2.0;
                    executable.set_loss_scale(next);
                    loss_scale_growth = 0;
                }
            }
        }
        if let Some(start) = start {
            executable.backend().synchronize()?;
            phase_ms[4] += start.elapsed().as_secs_f64() * 1000.0;
        }

        batch_num += 1;
        if let Some(samples) = &mut profile {
            samples.push(phase_ms);
            if samples.len() == 20 {
                print_train_profile(samples, batch_num, false);
            }
        }
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            epoch += 1;
            epoch_images = std::sync::Arc::new(epoch_order(&train_images, options.seed, epoch));
            image_cursor %= train_images.len();
        }

        let loss = iter_loss / images_per_iteration.max(1) as f32;
        progress.print(
            batch_num,
            loss,
            lr,
            seen,
            executable.network.net.max_batches,
            iteration_start.elapsed(),
            &map_progress_fields(validation.as_ref()),
        );
        loss_log.record(
            batch_num,
            seen,
            loss,
            progress.avg_loss.unwrap_or(loss),
            lr,
            iteration_start.elapsed().as_secs_f64() * 1000.0,
        )?;

        if let Some(validation) = &mut validation {
            if validation
                .schedule
                .due(batch_num, executable.network.net.max_batches)
            {
                println!(
                    "Validating current training weights on all {} images...",
                    validation.dataset.image_count()
                );
                // Deferred updates make device parameters authoritative. Sync
                // weights AND BN state, then snapshot for batch-one inference.
                executable.sync_host_weights_from_device()?;
                let mut inference = GpuExecutableNetwork::new(
                    executable.network.with_batch(1),
                    OxideBackend::new()?,
                );
                inference.params = executable.params.clone();
                let report = validation.dataset.evaluate(
                    &inference.network,
                    validation.detection_options,
                    validation.iou_threshold,
                    |input| {
                        let summary = inference.forward_compact(input)?;
                        Ok(ForwardResult::from_heads(
                            summary.yolo_heads,
                            summary.modern_heads,
                        ))
                    },
                )?;
                loss_log.set_map(report.mean_average_precision);
                if validation.record(&report, batch_num, seen)? {
                    let checkpoint = backup_dir.join(format!("{base_name}_best.weights"));
                    save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
                    println!("saved {}", checkpoint.display());
                }
            }
        }

        if options.save_every != 0 && batch_num % options.save_every == 0 {
            let checkpoint = backup_dir.join(format!("{base_name}_{batch_num}.weights"));
            executable.sync_host_weights_from_device()?;
            save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
            println!("saved {}", checkpoint.display());
            loss_log.save_last(
                &executable.network,
                &executable.params,
                &backup_dir,
                &base_name,
                seen,
            )?;
            executable.save_optimizer_state(&optimizer_state_path)?;
        }

        if executable.network.net.max_batches != 0
            && batch_num >= executable.network.net.max_batches
        {
            break;
        }
    }

    if let Some(samples) = &profile {
        print_train_profile(samples, batch_num, true);
    }
    let final_checkpoint = backup_dir.join(format!("{base_name}_final.weights"));
    executable.sync_host_weights_from_device()?;
    save_weights(
        &executable.network,
        &executable.params,
        &final_checkpoint,
        seen,
    )?;
    println!("saved {}", final_checkpoint.display());
    loss_log.save_last(
        &executable.network,
        &executable.params,
        &backup_dir,
        &base_name,
        seen,
    )?;
    executable.save_optimizer_state(&optimizer_state_path)?;
    Ok(())
}

fn print_forward_backward_summary(
    executable: &mut ExecutableNetwork,
    input: &Tensor,
    seed_mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let forward = executable.forward(input)?;
    let seeds = make_backward_seeds(&executable.network, &forward, seed_mode)?;
    executable.zero_train_buffers();
    let backward = executable.backward_accumulate_seeded(input, &forward, &seeds)?;
    let input_grad_l1 = backward
        .input_gradient
        .data
        .iter()
        .map(|value| value.abs())
        .sum::<f32>();
    let active_seed_layers = seeds.iter().flatten().count();
    let param_grad_layers = executable
        .train_buffers
        .iter()
        .filter(|params| match params {
            darknet_core::LayerTrainBuffers::Convolutional(conv) => {
                conv.bias_updates.iter().any(|value| *value != 0.0)
                    || conv.weight_updates.iter().any(|value| *value != 0.0)
                    || conv
                        .batch_norm
                        .as_ref()
                        .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
            }
            darknet_core::LayerTrainBuffers::Connected(connected) => {
                connected.bias_updates.iter().any(|value| *value != 0.0)
                    || connected.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::BatchNorm(batch_norm) => {
                batch_norm.bias_updates.iter().any(|value| *value != 0.0)
                    || batch_norm.scale_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Local(local) => {
                local.bias_updates.iter().any(|value| *value != 0.0)
                    || local.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Shortcut(shortcut) => {
                shortcut.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Moe(moe) => {
                moe.router_bias_updates.iter().any(|value| *value != 0.0)
                    || moe.router_weight_updates.iter().any(|value| *value != 0.0)
                    || moe.experts.iter().any(|expert| {
                        expert.bias_updates.iter().any(|value| *value != 0.0)
                            || expert.weight_updates.iter().any(|value| *value != 0.0)
                    })
            }
            darknet_core::LayerTrainBuffers::EmlConvolutional(eml) => {
                [&eml.input, &eml.self_params].iter().any(|conv| {
                    conv.bias_updates.iter().any(|value| *value != 0.0)
                        || conv.weight_updates.iter().any(|value| *value != 0.0)
                        || conv
                            .batch_norm
                            .as_ref()
                            .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
                })
            }
            darknet_core::LayerTrainBuffers::Vit(vit) => {
                vit.updates.to_blob().iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Swin(swin) => swin
                .updates
                .to_blob()
                .is_ok_and(|blob| blob.iter().any(|value| *value != 0.0)),
            darknet_core::LayerTrainBuffers::TuckerAttention(tucker) => {
                tucker.bias_updates.iter().any(|value| *value != 0.0)
                    || tucker.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::MambaVision(mamba) => {
                mamba.updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::KimiKda(kimi) => {
                kimi.bias_updates.iter().any(|value| *value != 0.0)
                    || kimi.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::GraphConv(graph) => {
                graph
                    .projection
                    .bias_updates
                    .iter()
                    .any(|value| *value != 0.0)
                    || graph
                        .projection
                        .weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || graph
                        .projection
                        .batch_norm
                        .as_ref()
                        .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
                    || graph
                        .self_weight_updates
                        .as_ref()
                        .is_some_and(|updates| updates.iter().any(|value| *value != 0.0))
                    || graph
                        .edge_kernel_updates
                        .as_ref()
                        .is_some_and(|updates| updates.iter().any(|value| *value != 0.0))
                    || graph
                        .edge_bias_updates
                        .as_ref()
                        .is_some_and(|updates| updates.iter().any(|value| *value != 0.0))
            }
            darknet_core::LayerTrainBuffers::Dcnv4(dcnv4) => {
                dcnv4.bias_updates.iter().any(|value| *value != 0.0)
                    || dcnv4.offset_bias_updates.iter().any(|value| *value != 0.0)
                    || dcnv4
                        .offset_weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || dcnv4
                        .batch_norm
                        .as_ref()
                        .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
            }
            darknet_core::LayerTrainBuffers::DsConv(dsconv) => {
                dsconv
                    .collapse
                    .bias_updates
                    .iter()
                    .any(|value| *value != 0.0)
                    || dsconv
                        .collapse
                        .weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || dsconv
                        .offset_weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || dsconv.offset_bias_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Wmhf(wmhf) => {
                [
                    &wmhf.pre,
                    &wmhf.local_dw3,
                    &wmhf.local_dw5,
                    &wmhf.local_dw7,
                    &wmhf.local_mix,
                    &wmhf.hf_gate,
                    &wmhf.fuse,
                ]
                .into_iter()
                .any(|conv| {
                    conv.bias_updates.iter().any(|value| *value != 0.0)
                        || conv.weight_updates.iter().any(|value| *value != 0.0)
                        || conv
                            .batch_norm
                            .as_ref()
                            .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
                }) || wmhf.scan_weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::Clifford(clifford) => {
                let branch_has_updates = |branch: &darknet_core::CliffordBranchTrainBuffers| {
                    branch
                        .projection_weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                        || branch
                            .projection_bias_updates
                            .iter()
                            .any(|value| *value != 0.0)
                        || branch.gate_weight_updates.iter().any(|value| *value != 0.0)
                        || branch.gate_bias_updates.iter().any(|value| *value != 0.0)
                };
                let convolution_has_updates = |conv: &darknet_core::ConvolutionalTrainBuffers| {
                    conv.bias_updates.iter().any(|value| *value != 0.0)
                        || conv.weight_updates.iter().any(|value| *value != 0.0)
                        || conv.batch_norm.as_ref().is_some_and(|bn| {
                            bn.scale_updates.iter().any(|value| *value != 0.0)
                                || bn.mean_delta.iter().any(|value| *value != 0.0)
                                || bn.variance_delta.iter().any(|value| *value != 0.0)
                        })
                };
                clifford
                    .layer_norm_gamma_updates
                    .iter()
                    .any(|value| *value != 0.0)
                    || clifford
                        .layer_norm_beta_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || clifford
                        .determinant_weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || clifford
                        .determinant_bias_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || clifford
                        .layer_scale_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || clifford.depthwise.iter().any(convolution_has_updates)
                    || branch_has_updates(&clifford.local)
                    || clifford.global.as_ref().is_some_and(branch_has_updates)
                    || clifford.fourier.as_ref().is_some_and(|fourier| {
                        fourier.real_updates.iter().any(|value| *value != 0.0)
                            || fourier.imaginary_updates.iter().any(|value| *value != 0.0)
                    })
            }
            darknet_core::LayerTrainBuffers::DeformConv(deform) => {
                deform.main.bias_updates.iter().any(|value| *value != 0.0)
                    || deform.main.weight_updates.iter().any(|value| *value != 0.0)
                    || deform
                        .offset_weight_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || deform.offset_bias_updates.iter().any(|value| *value != 0.0)
                    || deform
                        .main
                        .batch_norm
                        .as_ref()
                        .is_some_and(|bn| bn.scale_updates.iter().any(|value| *value != 0.0))
                    || deform
                        .mask_weight_updates
                        .as_ref()
                        .is_some_and(|updates| updates.iter().any(|value| *value != 0.0))
                    || deform
                        .mask_bias_updates
                        .as_ref()
                        .is_some_and(|updates| updates.iter().any(|value| *value != 0.0))
            }
            darknet_core::LayerTrainBuffers::DptHead(dpt) => {
                dpt.weight_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::HyperbolicFoveation(hf) => {
                hf.hf_param_updates.iter().any(|value| *value != 0.0)
                    || hf.pred_w1_updates.iter().any(|value| *value != 0.0)
                    || hf.pred_b1_updates.iter().any(|value| *value != 0.0)
                    || hf.pred_w2_updates.iter().any(|value| *value != 0.0)
                    || hf.pred_b2_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::FoveationFilm(film) => {
                film.w_embed_updates.iter().any(|value| *value != 0.0)
                    || film.b_embed_updates.iter().any(|value| *value != 0.0)
                    || film.w_gamma_updates.iter().any(|value| *value != 0.0)
                    || film.b_gamma_updates.iter().any(|value| *value != 0.0)
                    || film.w_beta_updates.iter().any(|value| *value != 0.0)
                    || film.b_beta_updates.iter().any(|value| *value != 0.0)
            }
            darknet_core::LayerTrainBuffers::RecursiveBlock(recursive) => {
                recursive
                    .controller_weight_updates
                    .iter()
                    .any(|value| *value != 0.0)
                    || recursive
                        .controller_bias_updates
                        .iter()
                        .any(|value| *value != 0.0)
                    || recursive.body.iter().any(|child| match child {
                        darknet_core::LayerTrainBuffers::Convolutional(conv) => {
                            conv.bias_updates.iter().any(|value| *value != 0.0)
                                || conv.weight_updates.iter().any(|value| *value != 0.0)
                                || conv.batch_norm.as_ref().is_some_and(|bn| {
                                    bn.scale_updates.iter().any(|value| *value != 0.0)
                                })
                        }
                        darknet_core::LayerTrainBuffers::Connected(connected) => {
                            connected.bias_updates.iter().any(|value| *value != 0.0)
                                || connected.weight_updates.iter().any(|value| *value != 0.0)
                                || connected.batch_norm.as_ref().is_some_and(|bn| {
                                    bn.scale_updates.iter().any(|value| *value != 0.0)
                                })
                        }
                        _ => false,
                    })
            }
            darknet_core::LayerTrainBuffers::Gru(_)
            | darknet_core::LayerTrainBuffers::Rnn(_)
            | darknet_core::LayerTrainBuffers::Lstm(_)
            | darknet_core::LayerTrainBuffers::Crnn(_) => true,
            darknet_core::LayerTrainBuffers::None => false,
        })
        .count();

    println!("layers executed {}", forward.layer_outputs.len());
    println!("seeded layers {}", active_seed_layers);
    println!("yolo heads {}", forward.yolo_heads.len());
    println!("input gradient {}", backward.input_gradient.shape);
    println!("input gradient l1 {:.6}", input_grad_l1);
    println!("parameter-gradient layers {}", param_grad_layers);
    Ok(())
}

fn make_backward_seeds(
    network: &Network,
    forward: &darknet_core::ForwardResult,
    seed_mode: &str,
) -> Result<Vec<Option<Tensor>>, Box<dyn std::error::Error>> {
    let mut seeds = vec![None; network.layers.len()];
    match seed_mode {
        "auto" | "yolo" if !forward.yolo_heads.is_empty() => {
            for head in &forward.yolo_heads {
                seeds[head.layer_index] = Some(Tensor::filled(head.output.shape, 1.0));
            }
        }
        "auto" | "final" => {
            let final_output = forward
                .final_output()
                .ok_or("network produced no layer outputs")?;
            let final_index = network
                .layers
                .len()
                .checked_sub(1)
                .ok_or("network has no layers")?;
            seeds[final_index] = Some(Tensor::filled(final_output.shape, 1.0));
        }
        other => {
            return Err(
                format!("invalid seed mode: {other} (expected auto, yolo, or final)").into(),
            );
        }
    }
    Ok(seeds)
}

fn print_usage() {
    println!("darknet-rs <command> [args]");
    println!("commands:");
    println!(
        "  detector train <cfg|data> <data|cfg> [weights] [-clear] [-dont_show] [-seed <u64>] [-gpus 0] [-map] [-colour|-nocolour]"
    );
    println!(
        "    -map validates all valid= images; -map_every <N> overrides max(100, epoch) cadence"
    );
    println!("    -thresh <0..1> (default .005), -iou_thresh <0..1> (default .5), NMS .45");
    println!(
        "  detector map <cfg> <data> [weights] [-map] [-thresh <0..1>] [-iou_thresh <0..1>] [-nms <0..1>] [-letter_box]"
    );
    println!("    -thresh default .005, -iou_thresh default .5, -nms default .45; CPU evaluation");
    println!("    seed defaults to 0; unknown flags fail explicitly; -gpus 0 requires oxide");
    println!("  version");
    println!("  cfg-validate <cfg>");
    println!("  cfg-summary <cfg>");
    println!("  names <names>");
    println!("  weights-inspect <cfg> <weights>");
    println!("  forward-dummy <cfg> [weights]");
    println!("  forward-backward-dummy <cfg> [weights] [auto|yolo|final]");
    println!(
        "  bench-train <cfg> [steps] [warmup]   (synthetic fwd+bwd+SGD steps, no dataset; DARKNET_BENCH_BATCH, DARKNET_BACKEND)"
    );
    println!(
        "  detect-image <cfg> <weights> <names> <image> [thresh] [nms] [resize|letterbox|resize-exact]"
    );
    println!("  export-onnx <cfg> <weights> [out.onnx] [--no-fuse]");
    println!(
        "env: DARKNET_BACKEND=cpu|oxide, DARKNET_OXIDE_CONV, DARKNET_OXIDE_GEMM, DARKNET_OXIDE_AMP=1 (mixed-precision training), DARKNET_DETECT_TSV=1"
    );
    println!(
        "  detect TSV emits `det` AABB rows or `det_obb` rows with geometry kind and coordinates"
    );
}

fn print_network_summary(network: &Network) {
    println!("cfg {}", display_path(&network.source));
    println!(
        "input {} batch={} subdivisions={}",
        network.net.input_shape(),
        network.net.batch,
        network.net.subdivisions
    );
    println!(
        "layers={} weighted_layers={} parameters={}",
        network.layers.len(),
        network.weighted_layers().count(),
        network.total_parameter_count()
    );

    for layer in &network.layers {
        let params = layer
            .weights()
            .map(|weights| weights.parameter_count())
            .unwrap_or(0);
        println!(
            "#{:>3} {:<18} {:<14} -> {:<14} params={}",
            layer.index,
            layer.kind.name(),
            layer.input_shape,
            layer.output_shape,
            params
        );
    }
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn cfg_base_name(cfg: &str) -> String {
    Path::new(cfg)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("darknet")
        .to_owned()
}

fn shuffle_paths(paths: &mut [PathBuf], seed: u64) {
    if paths.len() < 2 {
        return;
    }
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for i in (1..paths.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let j = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) as usize) % (i + 1);
        paths.swap(i, j);
    }
}

/// Deterministic per-epoch ordering so a resumed run replays the exact sample
/// order: the permutation depends only on (`seed`, epoch), not on how many
/// epochs were shuffled before it.
fn epoch_order(paths: &[PathBuf], seed: u64, epoch: usize) -> Vec<PathBuf> {
    let mut order = paths.to_vec();
    shuffle_paths(
        &mut order,
        seed ^ (epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
    );
    order
}

/// `(epoch_order, cursor)` for the next iteration's first batch, mirroring the
/// end-of-iteration bookkeeping without mutating the caller's state.  This lets
/// the trainer request the following batch before the current one is processed.
///
/// Assumes `images_per_iteration <= train_images.len()`; the existing trainer
/// bookkeeping shares that assumption.
fn advanced_epoch_images(
    train_images: &[PathBuf],
    seed: u64,
    epoch: usize,
    epoch_images: &std::sync::Arc<Vec<PathBuf>>,
    image_cursor: usize,
    images_per_iteration: usize,
) -> (std::sync::Arc<Vec<PathBuf>>, usize) {
    let next = image_cursor + images_per_iteration;
    if next >= train_images.len() {
        (
            std::sync::Arc::new(epoch_order(train_images, seed, epoch + 1)),
            next % train_images.len(),
        )
    } else {
        (epoch_images.clone(), next)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DetectionOptions, ImageDetections, MapOptions, MapSchedule, ObbDetectionGeometry,
        TrainOptions, classify_detector_train_paths, decode_image_detections,
        evaluate_detector_map, format_obb_detection_line, parse_export_onnx_args,
        run_detector_train, run_detector_train_cpu, validate_obb_augmentation,
    };

    #[test]
    fn export_onnx_args_parse_defaults_and_flags() {
        let a = parse_export_onnx_args(vec!["m.cfg".into(), "m.weights".into()]).unwrap();
        assert_eq!(a.out, std::path::PathBuf::from("m.onnx"));
        assert!(a.fuse);
        let b = parse_export_onnx_args(vec![
            "m.cfg".into(),
            "m.weights".into(),
            "o.onnx".into(),
            "--no-fuse".into(),
        ])
        .unwrap();
        assert_eq!(b.out, std::path::PathBuf::from("o.onnx"));
        assert!(!b.fuse);
        assert!(parse_export_onnx_args(vec!["m.cfg".into()]).is_err());
        assert!(
            parse_export_onnx_args(vec!["a.cfg".into(), "b.weights".into(), "--bogus".into(),])
                .is_err()
        );
        assert!(
            parse_export_onnx_args(vec![
                "a.cfg".into(),
                "b.weights".into(),
                "c.onnx".into(),
                "d.onnx".into(),
            ])
            .is_err()
        );
    }

    fn options(args: &[&str]) -> Result<TrainOptions, Box<dyn std::error::Error>> {
        TrainOptions::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn obb_cli_still_rejects_mosaic() {
        use darknet_core::{ConfigFile, DetectorAugmentation};

        let cfg = ConfigFile::parse_str(
            "obb-mosaic-cli.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=3\nobb=1\nmosaic=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\njitter=0\n",
        )
        .expect("cfg parses");
        let augmentation = DetectorAugmentation::from_config(&cfg).expect("augmentation parses");
        let error = validate_obb_augmentation(&augmentation).expect_err("mosaic stays rejected");
        assert!(error.to_string().contains("`mosaic`"));
    }

    #[test]
    fn detect_image_decodes_and_formats_obb_geometry() {
        use darknet_core::{ConfigFile, ExecutableNetwork, Network, Tensor};

        let cfg = ConfigFile::parse_str(
            "obb-detect-cli.cfg",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
        )
        .expect("OBB cfg parses");
        let network = Network::from_config(cfg).expect("OBB network builds");
        let stride = network.net.width * network.net.height;
        let mut data = vec![0.0; network.net.channels * stride];
        data[4 * stride] = 0.25;
        data[6 * stride] = 2.0;
        data[7 * stride] = 2.0;
        let input = Tensor::from_data(network.net.input_shape(), data).expect("input tensor");
        let forward = ExecutableNetwork::new(network.clone())
            .forward(&input)
            .expect("forward pass");
        let decoded =
            decode_image_detections(&network, &forward, 4, 2, DetectionOptions::default())
                .expect("CLI OBB decoder");
        let ImageDetections::Oriented(detections) = decoded else {
            panic!("OBB cfg must use oriented decoding");
        };
        assert_eq!(detections.len(), 1);
        let line = format_obb_detection_line(&detections[0], "vehicle", true);
        assert!(line.starts_with("det_obb\tvehicle\t"), "{line}");
        assert!(
            line.contains("\tfront_point\t0.250000000\t0.000000000"),
            "{line}"
        );

        let mut quad = detections[0].clone();
        quad.geometry = ObbDetectionGeometry::CarpnQuad([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let quad_tsv = format_obb_detection_line(&quad, "vehicle", true);
        assert_eq!(quad_tsv.split('\t').count(), 16, "{quad_tsv}");
        assert!(quad_tsv.contains("\tcarpn_quad\t0.000000000\t0.100000001"));
        let human = format_obb_detection_line(&quad, "vehicle", false);
        assert!(human.contains("quad=[(0.00,0.10),(0.20,0.30)"), "{human}");
    }

    #[test]
    fn detect_image_routes_carpn_forward_and_formats_quadrilateral_output() {
        use darknet_core::{ConfigFile, ExecutableNetwork, LayerParams, Network, Tensor};

        let cfg = ConfigFile::parse_str(
            "carpn-detect-cli.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1
                batch=1
                obb=1
                [convolutional]
                filters=2
                size=1
                batch_normalize=0
                activation=linear
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                activation=linear
                [obb_yolo]
                classes=1
                num=1
                anchors=1,1
                new_coords=1
                [obb_refine]
                from=0
                proposal_from=2
                proposal_topk_train=1
                proposal_topk_infer=1
                proposal_capacity=1
                sample_grid=1
            "#,
        )
        .expect("CARPN cfg parses");
        let network = Network::from_config(cfg).expect("CARPN network builds");
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("feature params");
        };
        feature.weights.copy_from_slice(&[1.0, 0.0]);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("head params");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.9, 0.8]);
        let input = Tensor::from_data(network.net.input_shape(), vec![1.0]).expect("input");
        let forward = executable.forward(&input).expect("forward");
        let decoded = decode_image_detections(
            &network,
            &forward,
            1,
            1,
            DetectionOptions {
                confidence_threshold: 0.1,
                nms_threshold: 1.1,
                letterbox: false,
            },
        )
        .expect("CARPN image decoder");
        let ImageDetections::Oriented(detections) = decoded else {
            panic!("CARPN cfg must use oriented decoding");
        };
        assert_eq!(detections.len(), 1);
        assert!(matches!(
            detections[0].geometry,
            ObbDetectionGeometry::CarpnQuad(_)
        ));
        let tsv = format_obb_detection_line(&detections[0], "vehicle", true);
        assert!(tsv.starts_with("det_obb\tvehicle\t"), "{tsv}");
        assert!(tsv.contains("\tcarpn_quad\t"), "{tsv}");
        assert_eq!(tsv.split('\t').count(), 16, "{tsv}");
        let human = format_obb_detection_line(&detections[0], "vehicle", false);
        assert!(human.contains("quad=[("), "{human}");
    }

    #[test]
    fn detect_image_keeps_axis_aligned_yolo_dispatch() {
        use darknet_core::{ConfigFile, ExecutableNetwork, Network, Tensor};

        let cfg = ConfigFile::parse_str(
            "yolo-detect-cli.cfg",
            "[net]\nwidth=4\nheight=2\nchannels=6\n[yolo]\nclasses=1\nnum=1\nmask=0\nanchors=4,2\n",
        )
        .expect("YOLO cfg parses");
        let network = Network::from_config(cfg).expect("YOLO network builds");
        let stride = network.net.width * network.net.height;
        let mut data = vec![0.0; network.net.channels * stride];
        data[4 * stride] = 2.0;
        data[5 * stride] = 2.0;
        let input = Tensor::from_data(network.net.input_shape(), data).expect("input tensor");
        let forward = ExecutableNetwork::new(network.clone())
            .forward(&input)
            .expect("forward pass");
        let decoded =
            decode_image_detections(&network, &forward, 4, 2, DetectionOptions::default())
                .expect("CLI YOLO decoder");
        let ImageDetections::AxisAligned(detections) = decoded else {
            panic!("standard YOLO cfg must use axis-aligned decoding");
        };
        assert_eq!(detections.len(), 1);
        assert!(detections[0].confidence > 0.25);
    }

    #[test]
    fn training_flags_parse_values_without_consuming_weights() {
        let parsed = options(&[
            "-seed",
            "42",
            "-gpus",
            "0",
            "model.weights",
            "-clear",
            "-dont_show",
        ])
        .unwrap();
        assert_eq!(
            parsed,
            TrainOptions {
                weights: Some("model.weights".into()),
                seed: 42,
                clear: true,
                dont_show: true,
                gpu: true,
                map: false,
                map_every: None,
                map_thresh: 0.005,
                map_iou: 0.5,
                save_every: 1000,
                dynamic_loss_scale: false,
                colour: None,
            }
        );
        assert_eq!(options(&[]).unwrap().seed, 0);
    }

    #[test]
    fn map_flags_and_cadence_include_final_iteration() {
        let parsed = options(&[
            "model.weights",
            "-map",
            "-map_every",
            "12",
            "-thresh",
            "0.01",
            "-iou_thresh",
            "0.75",
        ])
        .unwrap();
        assert!(parsed.map);
        assert_eq!(parsed.map_every, Some(12));
        assert_eq!(parsed.map_thresh, 0.01);
        assert_eq!(parsed.map_iou, 0.75);
        let mut schedule = MapSchedule {
            interval: 100,
            burn_in: 200,
            last_iteration: 0,
        };
        assert!(!schedule.due(100, 1000));
        assert!(schedule.due(200, 1000));
        schedule.last_iteration = 200;
        assert!(!schedule.due(299, 1000));
        assert!(schedule.due(300, 1000));
        assert!(schedule.due(250, 250));
        assert!(!schedule.due(250, 0));
        assert!(options(&["-map"]).is_ok());
    }

    #[test]
    fn standalone_map_options_parse_and_validate() {
        let parsed = MapOptions::parse(
            [
                "weights.weights",
                "-map",
                "-thresh",
                "0.02",
                "-iou_thresh",
                "0.7",
                "-nms",
                "0.4",
                "-letter_box",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            parsed,
            MapOptions {
                weights: Some("weights.weights".into()),
                confidence_threshold: 0.02,
                iou_threshold: 0.7,
                nms_threshold: 0.4,
                letterbox: true,
            }
        );
        assert_eq!(
            MapOptions::parse(std::iter::empty()).unwrap(),
            MapOptions {
                weights: None,
                confidence_threshold: 0.005,
                iou_threshold: 0.5,
                nms_threshold: 0.45,
                letterbox: false,
            }
        );
        for args in [
            vec!["-thresh", "NaN"],
            vec!["-iou_thresh", "1.1"],
            vec!["-nms", "-0.1"],
            vec!["-thresh"],
            vec!["a.weights", "b.weights"],
            vec!["-gpus", "0"],
        ] {
            assert!(MapOptions::parse(args.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn clear_resets_seen_but_default_resumes() {
        assert_eq!(options(&[]).unwrap().initial_seen(512), 512);
        assert_eq!(options(&["-clear"]).unwrap().initial_seen(512), 0);
    }

    #[test]
    fn unsupported_training_flags_and_bad_values_fail() {
        for args in [
            vec!["-map", "-map_every", "0"],
            vec!["-map", "-thresh", "NaN"],
            vec!["-map", "-iou_thresh", "2"],
            vec!["-gpus", "0,1"],
            vec!["-gpus"],
            vec!["-seed"],
            vec!["-seed", "-1"],
            vec!["-thresh", "0.5"],
            vec!["a.weights", "b.weights"],
        ] {
            assert!(options(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn save_every_parses_and_rejects_zero() {
        assert_eq!(options(&[]).unwrap().save_every, 1000);
        let parsed = options(&["model.weights", "-save_every", "250"]).unwrap();
        assert_eq!(parsed.save_every, 250);
        assert_eq!(parsed.weights.as_deref(), Some("model.weights"));
        assert!(options(&["-save_every", "0"]).is_err());
        assert!(options(&["-save_every"]).is_err());
    }

    #[test]
    fn epoch_order_is_resumable_and_seed_dependent() {
        use super::epoch_order;
        let paths: Vec<std::path::PathBuf> = (0..8)
            .map(|i| std::path::PathBuf::from(format!("{i}.png")))
            .collect();
        assert_eq!(epoch_order(&paths, 42, 0), epoch_order(&paths, 42, 0));
        assert_ne!(epoch_order(&paths, 42, 0), epoch_order(&paths, 42, 1));
        assert_ne!(epoch_order(&paths, 42, 0), epoch_order(&paths, 7, 0));
        let mut sorted = epoch_order(&paths, 42, 3);
        sorted.sort();
        assert_eq!(sorted, paths);
    }

    #[test]
    fn advanced_epoch_images_matches_iteration_bookkeeping() {
        use super::{advanced_epoch_images, epoch_order};
        let paths: Vec<std::path::PathBuf> = (0..8)
            .map(|i| std::path::PathBuf::from(format!("{i}.png")))
            .collect();
        let epoch0 = std::sync::Arc::new(epoch_order(&paths, 42, 0));
        // Within an epoch the order is reused and the cursor advances.
        let (order, cursor) = advanced_epoch_images(&paths, 42, 0, &epoch0, 0, 4);
        assert!(std::sync::Arc::ptr_eq(&order, &epoch0));
        assert_eq!(cursor, 4);
        // Crossing the boundary wraps to the next epoch's permutation.
        let (order, cursor) = advanced_epoch_images(&paths, 42, 0, &epoch0, 4, 4);
        assert_eq!(cursor, 0);
        assert_eq!(*order, epoch_order(&paths, 42, 1));
    }

    #[test]
    fn detector_train_runs_cpu_obb_subdivisions_and_map_cadence() {
        let dir = std::env::temp_dir().join(format!("darknet-obb-cli-{}", std::process::id()));
        let backup = dir.join("backup");
        std::fs::create_dir_all(&backup).expect("test backup directory");
        let image = dir.join("sample.ppm");
        let labels = dir.join("sample.txt");
        let train_list = dir.join("train.txt");
        let valid_list = dir.join("valid.txt");
        let cfg = dir.join("obb-cli.cfg");
        let data = dir.join("obb-cli.data");
        std::fs::write(&image, b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00").expect("PPM image");
        std::fs::write(&labels, "0 0.5 0.5 0.25 0.5 0.7 0.55\n").expect("OBB label");
        std::fs::write(&train_list, format!("{}\n", image.display())).expect("train list");
        std::fs::write(&valid_list, format!("{}\n", image.display())).expect("valid list");
        std::fs::write(
            &data,
            format!(
                "classes=1\ntrain={}\nvalid={}\nbackup={}\n",
                train_list.display(),
                valid_list.display(),
                backup.display()
            ),
        )
        .expect("data cfg");
        std::fs::write(
            &cfg,
            "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\nsubdivisions=2\nmax_batches=3\nlearning_rate=.01\nmomentum=0\ndecay=0\nobb=1\nflip=0\n[convolutional]\nfilters=8\nsize=1\nstride=1\nactivation=linear\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\njitter=0\niou_loss=ciou_obb\n",
        )
        .expect("OBB cfg");

        let cfg_path = cfg.to_str().expect("cfg path is UTF-8");
        let data_path = data.to_str().expect("data path is UTF-8");
        run_detector_train_cpu(
            cfg_path,
            data_path,
            &options(&["-map", "-map_every", "2"]).expect("mAP train options"),
        )
        .expect("OBB CPU training with periodic/final mAP");

        assert!(backup.join("obb-cli_final.weights").exists());
        let loss_log =
            std::fs::read_to_string(backup.join("obb-cli_loss.csv")).expect("OBB loss log");
        assert!(
            loss_log.lines().any(|line| line.starts_with("3,6,")),
            "{loss_log}"
        );
        let metrics_log = std::fs::read_to_string(backup.join("obb-cli_obb_metrics.csv"))
            .expect("OBB metrics log");
        let metrics_header = metrics_log.lines().next().expect("OBB metrics header");
        assert!(metrics_header.contains("mean_angle_error_degrees"));
        assert_eq!(metrics_header.split(',').count(), 19);
        let metrics_row = metrics_log
            .lines()
            .find(|line| line.starts_with("3,6,"))
            .expect("one OBB metric row");
        let columns: Vec<_> = metrics_row.split(',').collect();
        assert_eq!(columns.len(), 19, "{metrics_row}");
        assert_eq!(
            columns[4], "2",
            "two subdivision box assignments: {metrics_row}"
        );
        assert_eq!(
            columns[8], "2",
            "two subdivision class assignments: {metrics_row}"
        );
        assert!(backup.join("obb-cli_best.weights").exists());
        let map_log = std::fs::read_to_string(backup.join("obb-cli_map.csv")).expect("mAP log");
        let map_rows: Vec<_> = map_log
            .lines()
            .filter(|line| line.starts_with("2,4,") || line.starts_with("3,6,"))
            .collect();
        assert_eq!(
            map_rows.len(),
            2,
            "expected interval and final mAP rows: {map_log}"
        );
        assert!(map_rows[0].starts_with("2,4,"), "{map_log}");
        assert!(map_rows[1].starts_with("3,6,"), "{map_log}");
    }

    #[test]
    fn detector_train_runs_cpu_obb_adam_with_exposure_for_two_iterations() {
        let dir = std::env::temp_dir().join(format!(
            "darknet-obb-adam-exposure-cli-{}",
            std::process::id()
        ));
        let backup = dir.join("backup");
        std::fs::create_dir_all(&backup).expect("test backup directory");
        let image = dir.join("sample.ppm");
        let labels = dir.join("sample.txt");
        let train_list = dir.join("train.txt");
        let cfg = dir.join("obb-adam-exposure-cli.cfg");
        let data = dir.join("obb-adam-exposure-cli.data");
        std::fs::write(&image, b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00").expect("PPM image");
        std::fs::write(&labels, "0 0.5 0.5 0.25 0.5 0.7 0.55\n").expect("OBB label");
        std::fs::write(&train_list, format!("{}\n", image.display())).expect("train list");
        std::fs::write(
            &data,
            format!(
                "classes=1\ntrain={}\nbackup={}\n",
                train_list.display(),
                backup.display()
            ),
        )
        .expect("data cfg");
        std::fs::write(
            &cfg,
            "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\nsubdivisions=2\nmax_batches=2\nlearning_rate=.01\nmomentum=0\ndecay=0\nadam=1\nobb=1\nflip=0\nexposure=1.5\n[convolutional]\nfilters=8\nsize=1\nstride=1\nactivation=linear\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\njitter=0\niou_loss=ciou_obb\n",
        )
        .expect("OBB Adam cfg");

        run_detector_train_cpu(
            cfg.to_str().expect("cfg path is UTF-8"),
            data.to_str().expect("data path is UTF-8"),
            &options(&["-seed", "7"]).expect("seeded train options"),
        )
        .expect("OBB CPU Adam training with exposure augmentation");

        assert!(backup.join("obb-adam-exposure-cli_final.weights").exists());
        let optimizer_state = backup.join("obb-adam-exposure-cli_optimizer.state");
        assert!(
            optimizer_state.metadata().is_ok(),
            "Adam state was not saved"
        );
        let loss_log = std::fs::read_to_string(backup.join("obb-adam-exposure-cli_loss.csv"))
            .expect("OBB loss log");
        assert!(
            loss_log.lines().any(|line| line.starts_with("1,2,")),
            "{loss_log}"
        );
        assert!(
            loss_log.lines().any(|line| line.starts_with("2,4,")),
            "{loss_log}"
        );
    }

    #[test]
    fn carpn_cli_detector_map_loads_cfg_weights_and_returns_polygon_report() {
        use darknet_core::{ExecutableNetwork, LayerParams, Network};

        let dir =
            std::env::temp_dir().join(format!("darknet-cli-carpn-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let cfg = dir.join("model.cfg");
        let data = dir.join("data.data");
        let image = dir.join("sample.ppm");
        let label = dir.join("sample.txt");
        let weights = dir.join("model.weights");
        std::fs::write(
            &cfg,
            r#"
                [net]
                width=2
                height=1
                channels=3
                batch=1
                obb=1
                [convolutional]
                filters=2
                size=1
                batch_normalize=0
                activation=linear
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                activation=linear
                [obb_yolo]
                classes=1
                num=1
                anchors=0.2,0.2
                new_coords=1
                max=1
                [obb_refine]
                from=0
                proposal_from=2
                proposal_topk_train=1
                proposal_topk_infer=1
                proposal_capacity=1
                sample_grid=1
                front_range=1
            "#,
        )
        .expect("cfg writes");
        std::fs::write(&image, b"P6\n2 1\n255\n\x80\x40\x20\x80\x40\x20").expect("image writes");
        std::fs::write(&label, "0 0.25 0.5 0.15811388 0.15811388 0.15 0.4\n")
            .expect("label writes");
        std::fs::write(dir.join("valid.txt"), "sample.ppm\n").expect("valid list writes");
        std::fs::write(&data, "classes=1\nvalid=valid.txt\n").expect("data writes");

        let network = Network::from_path(&cfg)
            .expect("network builds")
            .with_batch(1);
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("feature parameters");
        };
        feature.weights.fill(0.0);
        feature.biases.fill(0.0);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("proposal parameters");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.9, 0.8]);
        darknet_core::save_weights(&executable.network, &executable.params, &weights, 0)
            .expect("fixed CARPN weights save");

        let options = MapOptions {
            weights: Some(weights.to_string_lossy().into_owned()),
            confidence_threshold: 0.1,
            iou_threshold: 0.5,
            nms_threshold: 1.0,
            letterbox: false,
        };
        let (classes, image_count, names, report) = evaluate_detector_map(
            cfg.to_str().expect("cfg path"),
            data.to_str().expect("data path"),
            &options,
        )
        .expect("CLI detector map CARPN path");
        assert_eq!((classes, image_count), (1, 1));
        assert_eq!(names, vec!["class_0".to_owned()]);
        assert_eq!(report.average_precision, vec![1.0]);
        assert_eq!(report.detection_counts, vec![1]);
        assert_eq!(report.true_positives, vec![1]);
        std::fs::remove_dir_all(dir).expect("temporary directory cleanup");
    }

    #[test]
    fn yolo_3d_cli_rejects_train_map_and_detect_server_before_loading_assets() {
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = Fixture(
            std::env::temp_dir().join(format!("yolo3d-cli-{}-{nanos}", std::process::id())),
        );
        std::fs::create_dir(&dir.0).unwrap();
        let cfg = dir.0.join("model.cfg");
        let data = dir.0.join("data.data");
        std::fs::write(&cfg, "[net]\nwidth=3\nheight=5\nchannels=30\n[yolo_3d]\n").unwrap();
        // Deliberately malformed: rejection must happen before loading data,
        // weights, images, creating outputs or starting any training work.
        std::fs::write(&data, "this is not a data config").unwrap();
        let cfg = cfg.to_str().unwrap();
        let data = data.to_str().unwrap();
        let errors = [
            run_detector_train(cfg, data, &options(&[]).unwrap())
                .unwrap_err()
                .to_string(),
            super::evaluate_detector_map(
                cfg,
                data,
                &super::MapOptions::parse(Vec::<String>::new().into_iter()).unwrap(),
            )
            .unwrap_err()
            .to_string(),
            super::run_detect_server(cfg, "nonexistent.weights")
                .unwrap_err()
                .to_string(),
        ];
        for error in errors {
            assert!(
                error.contains("[yolo_3d]") && error.contains("not implemented"),
                "{error}"
            );
        }
    }

    #[test]
    fn detector_train_rejects_carpn_before_starting_training_or_map() {
        let dir = std::env::temp_dir().join(format!(
            "darknet-cli-carpn-train-reject-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let cfg = dir.join("model.cfg");
        let data = dir.join("data.data");
        std::fs::write(
            &cfg,
            r#"
                [net]
                width=1
                height=1
                channels=1
                obb=1
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                [obb_yolo]
                classes=1
                num=1
                anchors=1,1
                [obb_refine]
            "#,
        )
        .expect("CARPN cfg");
        std::fs::write(&data, "classes=1\n").expect("data cfg");
        let error = run_detector_train(
            cfg.to_str().expect("cfg path"),
            data.to_str().expect("data path"),
            &options(&["-map"]).expect("training options"),
        )
        .expect_err("CARPN train and train -map must fail before starting");
        assert!(error.to_string().contains("[obb_refine]"), "{error}");
        assert!(error.to_string().contains("Stage-2"), "{error}");
        std::fs::remove_dir_all(dir).expect("temporary directory cleanup");
    }

    #[test]
    fn detector_train_accepts_cfg_then_data() {
        let (cfg, data) = classify_detector_train_paths("model.cfg", "dataset.data")
            .expect("paths should classify");
        assert_eq!(cfg, "model.cfg");
        assert_eq!(data, "dataset.data");
    }

    #[test]
    fn detector_train_accepts_data_then_cfg() {
        let (cfg, data) = classify_detector_train_paths("dataset.data", "model.cfg")
            .expect("paths should classify");
        assert_eq!(cfg, "model.cfg");
        assert_eq!(data, "dataset.data");
    }

    #[test]
    fn detector_train_rejects_invalid_path_pair() {
        let err = classify_detector_train_paths("model.cfg", "weights.weights")
            .expect_err("pair should be rejected");
        assert!(
            err.to_string()
                .contains("expects one .cfg path and one .data path")
        );
    }

    #[test]
    fn colour_flags_parse() {
        assert_eq!(options(&["-colour"]).unwrap().colour, Some(true));
        assert_eq!(options(&["--colour"]).unwrap().colour, Some(true));
        assert_eq!(options(&["-nocolour"]).unwrap().colour, Some(false));
        assert_eq!(options(&["--no-colour"]).unwrap().colour, Some(false));
        // Auto-detect by default (terminal + no NO_COLOR).
        assert_eq!(options(&[]).unwrap().colour, None);
    }

    #[test]
    fn map_report_block_matches_cpp_layout() {
        use super::{colour, format_map_report, map_progress_fields};
        let _guard = colour::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        colour::configure(Some(false));
        let report = darknet_core::MapReport {
            average_precision: vec![1.0, 0.5],
            ground_truth_counts: vec![4, 2],
            detection_counts: vec![4, 3],
            mean_average_precision: 0.75,
            true_positives: vec![4, 1],
            false_positives: vec![0, 1],
            false_negatives: vec![0, 1],
            average_iou: vec![0.9, 0.8],
        };
        let text = format_map_report(&report, &["gear".to_owned(), "pin".to_owned()], 0.5);
        assert!(text.contains(
            "  Id Name                       AP      TP      FP      FN      GT       F1   AvgIoU"
        ));
        assert!(text.contains("   0 gear                 100.0000       4       0       0       4 100.0000  90.0000"), "{text}");
        assert!(
            text.contains("-> for conf_thresh=0.25, TP=5, FP=1, FN=1, average IoU=88%"),
            "{text}"
        );
        assert!(
            text.contains("-> IoU threshold=50%, used area-under-curve for each unique recall")
        );
        assert!(text.contains("-> mean average precision (mAP@0.50)=75.00%"));
        assert_eq!(
            map_progress_fields(None),
            ", last=none, best=none, mAP=disabled"
        );
    }

    #[test]
    fn training_line_without_colour_is_plain() {
        use super::{colour, format_training_progress_line};
        let _guard = colour::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        colour::configure(Some(false));
        let line = format_training_progress_line(
            7,
            0.4444,
            0.329,
            0.000026,
            383296,
            "7 seconds",
            64.424,
            "",
        );
        assert!(!line.contains('\x1b'), "plain line had escapes: {line:?}");
        assert_eq!(
            line,
            "7: loss=0.444, avg loss=0.329, rate=0.00002600, train=64.424 milliseconds, 383296 images, time remaining=7 seconds"
        );
    }

    #[test]
    fn training_line_colours_iteration_and_loss() {
        use super::{colour, format_training_progress_line};
        let _guard = colour::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        colour::configure(Some(true));
        let line = format_training_progress_line(7, 0.5, 25.0, 0.001, 10, "1 second", 1.0, "");
        // Iteration bright white, loss 0.5 bright cyan, avg loss 25.0 red.
        assert!(line.starts_with("\x1b[1;37m7\x1b[0m: "), "{line:?}");
        assert!(
            line.contains("loss=\x1b[1;36m0.500\x1b[0m, avg loss=\x1b[0;31m25.000\x1b[0m"),
            "{line:?}"
        );
        colour::configure(Some(false));
    }
}
