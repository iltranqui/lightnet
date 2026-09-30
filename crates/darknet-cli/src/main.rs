//! `darknet-rs`, the command-line front end for the Rust port.
//!
//! CLI code stays thin: it parses commands, loads models/data, and formats
//! Darknet-style summaries while `darknet-core` owns graph and training
//! semantics.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "cutile")]
use darknet_core::GpuExecutableNetwork;
use darknet_core::{
    DataConfig, DetectionOptions, ExecutableNetwork, Network, PlanarRgbImage, Tensor,
    YoloTrainingState, compute_learning_rate, compute_yolo_training_loss_with_state,
    decode_yolo_detections, inspect_weights_file, load_names_file, load_paths_file,
    load_training_batch_from_paths, save_weights,
};
#[cfg(feature = "cutile")]
use darknet_cuda_cutile::CutileBackend;

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
                    let mut weights = None;
                    let mut flags = Vec::new();
                    for arg in args {
                        if arg.starts_with('-') {
                            flags.push(arg);
                        } else if weights.is_none() {
                            weights = Some(arg);
                        } else {
                            return Err(format!("unexpected detector train argument: {arg}").into());
                        }
                    }
                    run_detector_train(&cfg, &data, weights.as_deref(), &flags)?;
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
        "detect-image" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let weights = required_arg(args.next(), "weights path")?;
            let names_path = required_arg(args.next(), "names path")?;
            let image_path = required_arg(args.next(), "image path")?;
            let threshold = optional_f32(args.next(), 0.25, "confidence threshold")?;
            let nms = optional_f32(args.next(), 0.45, "nms threshold")?;
            let preprocess = args.next().unwrap_or_else(|| "letterbox".to_owned());
            let letterbox = match preprocess.as_str() {
                "letterbox" => true,
                "resize" => false,
                other => {
                    return Err(format!(
                        "invalid preprocess mode: {other} (expected letterbox or resize)"
                    )
                    .into());
                }
            };

            let names = load_names_file(&names_path)?;
            let network = Network::from_path(&cfg)?.with_batch(1);
            let mut executable = ExecutableNetwork::new(network);
            let header = executable.load_weights_file(&weights)?;
            println!(
                "loaded weights {}.{}.{} seen={}",
                header.major, header.minor, header.revision, header.seen
            );

            let original = PlanarRgbImage::load(&image_path)?;
            let prepared = if letterbox {
                original.letterbox(executable.network.net.width, executable.network.net.height)?
            } else {
                original.resize(executable.network.net.width, executable.network.net.height)?
            };
            let result = executable.forward(&prepared.clone().into())?;
            let detections = decode_yolo_detections(
                &executable.network,
                &result.yolo_heads,
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
                if letterbox { "letterbox" } else { "resize" }
            );
            println!("detections {}", detections.len());
            for detection in detections {
                let class_name = names
                    .get(detection.class_id)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
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
        "help" | "--help" | "-h" => print_usage(),
        other => {
            return Err(format!("unknown command: {other}").into());
        }
    }

    Ok(())
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

fn run_detector_train(
    cfg: &str,
    data: &str,
    maybe_weights: Option<&str>,
    flags: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(cfg).exists() {
        return Err(format!("cfg path does not exist: {cfg}").into());
    }
    if !Path::new(data).exists() {
        return Err(format!("data path does not exist: {data}").into());
    }

    println!("detector train");
    println!("cfg {}", cfg);
    println!("data {}", data);
    if !flags.is_empty() {
        println!("flags {}", flags.join(" "));
    }

    run_detector_train_preferred_backend(cfg, data, maybe_weights)
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
        // loss, avg loss, rate, elapsed training time, images, and ETA.
        println!(
            "{batch}: loss={loss:.6}, avg loss={avg_loss:.6}, rate={learning_rate:.8}, train={:.3} milliseconds, {images_seen} images, time remaining={time_remaining}",
            iteration_duration.as_secs_f64() * 1000.0,
        );
    }
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

#[cfg(feature = "cutile")]
fn run_detector_train_preferred_backend(
    cfg: &str,
    data: &str,
    maybe_weights: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    match CutileBackend::new(0) {
        Ok(backend) => {
            println!("gpu backend cutile device=0");
            run_detector_train_gpu(cfg, data, maybe_weights, backend)
        }
        Err(err) => {
            println!("gpu backend unavailable ({err}), falling back to cpu");
            run_detector_train_cpu(cfg, data, maybe_weights)
        }
    }
}

#[cfg(not(feature = "cutile"))]
fn run_detector_train_preferred_backend(
    cfg: &str,
    data: &str,
    maybe_weights: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("gpu backend disabled at compile time, falling back to cpu");
    run_detector_train_cpu(cfg, data, maybe_weights)
}

fn run_detector_train_cpu(
    cfg: &str,
    data: &str,
    maybe_weights: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_config = DataConfig::from_path(data)?;
    let mut executable = ExecutableNetwork::from_path(cfg)?;
    let mut seen = 0u64;
    if let Some(weights) = maybe_weights {
        let header = executable.load_weights_file(weights)?;
        seen = header.seen;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    } else {
        println!("running with zero-initialized weights");
    }

    let mut train_images = load_paths_file(data_config.required_train_list()?)?;
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
    let images_per_iteration =
        executable.network.net.batch.max(1) * executable.network.net.subdivisions.max(1);
    let mut batch_num = (seen as usize)
        .checked_div(images_per_iteration)
        .unwrap_or(0);
    let mut yolo_state = YoloTrainingState::default();
    let mut progress = TrainingProgress::new(batch_num);
    let mut image_cursor = if train_images.is_empty() {
        0usize
    } else {
        (batch_num * images_per_iteration) % train_images.len()
    };

    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        let iteration_start = Instant::now();
        let lr = compute_learning_rate(&executable.network.net, batch_num);
        let mut iter_loss = 0.0f32;
        for subdivision in 0..executable.network.net.subdivisions.max(1) {
            let start_index = image_cursor + subdivision * executable.network.net.batch.max(1);
            let batch = load_training_batch_from_paths(
                &executable.network,
                &data_config,
                &train_images,
                start_index,
            )?;
            let forward = executable.forward_training(&batch.input)?;
            let yolo = compute_yolo_training_loss_with_state(
                &executable.network,
                &forward,
                &batch.labels,
                batch_num,
                Some(&mut yolo_state),
            )?;
            iter_loss += yolo.loss;
            executable.backward_accumulate_seeded(
                &batch.input,
                &forward,
                &yolo.layer_output_gradient_seeds,
            )?;
        }
        executable.update_weights(
            lr,
            executable.network.net.momentum,
            executable.network.net.decay,
            executable.network.net.batch.max(1),
        );

        batch_num += 1;
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            shuffle_paths(&mut train_images, batch_num as u64 + 1);
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
        );

        if batch_num % 1000 == 0 {
            let checkpoint = backup_dir.join(format!("{base_name}_{batch_num}.weights"));
            save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
            println!("saved {}", checkpoint.display());
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
    Ok(())
}

#[cfg(feature = "cutile")]
fn run_detector_train_gpu(
    cfg: &str,
    data: &str,
    maybe_weights: Option<&str>,
    backend: CutileBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_config = DataConfig::from_path(data)?;
    let mut executable = GpuExecutableNetwork::from_path(cfg, backend)?;
    let mut seen = 0u64;
    if let Some(weights) = maybe_weights {
        let header = executable.load_weights_file(weights)?;
        seen = header.seen;
        println!(
            "loaded weights {}.{}.{} seen={}",
            header.major, header.minor, header.revision, header.seen
        );
    } else {
        println!("running with zero-initialized weights");
    }
    executable.init_train_buffers_gpu()?;
    println!("allocated GPU params and train buffers");

    let mut train_images = load_paths_file(data_config.required_train_list()?)?;
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
    let images_per_iteration =
        executable.network.net.batch.max(1) * executable.network.net.subdivisions.max(1);
    let mut batch_num = if images_per_iteration == 0 {
        0usize
    } else {
        (seen as usize) / images_per_iteration
    };
    let mut yolo_state = YoloTrainingState::default();
    let mut progress = TrainingProgress::new(batch_num);
    let mut image_cursor = if train_images.is_empty() {
        0usize
    } else {
        (batch_num * images_per_iteration) % train_images.len()
    };

    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        let iteration_start = Instant::now();
        let lr = compute_learning_rate(&executable.network.net, batch_num);
        let mut iter_loss = 0.0f32;
        // NOTE: do NOT zero train buffers here. After update_weights_gpu, the buffers
        // hold momentum * previous_gradient (C++ weight_updates *= momentum). Zeroing
        // would destroy the momentum carry. init_train_buffers_gpu() above initialises
        // them to zero only once at startup, which is correct.
        for subdivision in 0..executable.network.net.subdivisions.max(1) {
            let start_index = image_cursor + subdivision * executable.network.net.batch.max(1);
            let batch = load_training_batch_from_paths(
                &executable.network,
                &data_config,
                &train_images,
                start_index,
            )?;
            let forward = executable.forward_training_with_state(&batch.input)?;
            let yolo = compute_yolo_training_loss_with_state(
                &executable.network,
                &forward.cpu_forward,
                &batch.labels,
                batch_num,
                Some(&mut yolo_state),
            )?;
            iter_loss += yolo.loss;
            executable
                .backward_accumulate_seeded_gpu(&forward, &yolo.layer_output_gradient_seeds)?;
        }
        executable.update_weights_gpu(
            lr,
            executable.network.net.momentum,
            executable.network.net.decay,
            executable.network.net.batch.max(1),
        )?;

        batch_num += 1;
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            shuffle_paths(&mut train_images, batch_num as u64 + 1);
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
        );

        if batch_num % 1000 == 0 {
            let checkpoint = backup_dir.join(format!("{base_name}_{batch_num}.weights"));
            save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
            println!("saved {}", checkpoint.display());
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
    println!("  detector train <cfg|data> <data|cfg> [weights] [flags...]");
    println!("  version");
    println!("  cfg-validate <cfg>");
    println!("  cfg-summary <cfg>");
    println!("  names <names>");
    println!("  weights-inspect <cfg> <weights>");
    println!("  forward-dummy <cfg> [weights]");
    println!("  forward-backward-dummy <cfg> [weights] [auto|yolo|final]");
    println!("  detect-image <cfg> <weights> <names> <image> [thresh] [nms] [letterbox|resize]");
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

#[cfg(test)]
mod tests {
    use super::classify_detector_train_paths;

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
}
