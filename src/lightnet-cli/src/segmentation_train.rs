//! `detector train` for semantic-segmentation networks on the oxide backend.
//!
//! The detector loop loads YOLO label files and runs augmentation that moves
//! boxes; a `[segmentation]` network has neither. This loop loads each image,
//! resizes it to the network input, loads the matching pixel mask
//! ([`lightnet_core::segmentation_data`]), and drives the native segmentation
//! loss ([`GpuExecutableNetwork::compute_segmentation_training_loss_gpu`]) into
//! the seeded device backward and the usual deferred optimizer step.
//!
//! No geometric augmentation is applied: a flip or crop would need the same
//! transform on the mask, which this loop does not do yet.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use lightnet_core::segmentation_data::{
    load_segmentation_truth, segmentation_head_layers, segmentation_mask_path,
};
use lightnet_core::{
    DataConfig, GpuExecutableNetwork, LayerKind, PlanarRgbImage, Tensor, compute_learning_rate,
    initialize_training_params, load_paths_file, save_weights,
};
use lightnet_cuda_oxide::OxideBackend;

use super::{
    MapSchedule, colour, TrainOptions, TrainingLossLog, TrainingProgress, cfg_base_name, epoch_order,
    map_progress_fields, print_training_configuration,
};

/// True when the network trains through the segmentation loss: it ends in a
/// `[segmentation]` head or a legacy `[logistic]` / `[softmax] spatial=1` head,
/// and has no detection head the detector loop would need.
pub(crate) fn is_segmentation_network(network: &lightnet_core::Network) -> bool {
    !segmentation_head_layers(network).is_empty()
        && !lightnet_core::train::network_uses_dense_heads(network)
        && !network.layers.iter().any(|layer| {
            matches!(
                layer.kind,
                LayerKind::Yolo(_) | LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_)
            )
        })
}

/// Load `paths` (cycling) into one input batch and one truth tensor per
/// segmentation head. Returns `(input, truth)`; `truth` is indexed by layer.
fn load_batch(
    network: &lightnet_core::Network,
    paths: &[PathBuf],
    start: usize,
) -> Result<(Tensor, Vec<Option<Tensor>>), Box<dyn std::error::Error>> {
    let batch = network.net.batch.max(1);
    let (width, height) = (network.net.width, network.net.height);
    let mut input = Tensor::zeros(network.net.input_shape());
    // `element_count` is per image; the tensor holds `batch` of them.
    let per_image = input.shape.element_count();

    let mut truth_data: Vec<(usize, Vec<f32>)> = segmentation_head_layers(network)
        .into_iter()
        .map(|index| (index, Vec::new()))
        .collect();

    for slot in 0..batch {
        let image_path = &paths[(start + slot) % paths.len()];
        let image = PlanarRgbImage::load(image_path)?.resize(width, height)?;
        if image.shape.channels != network.net.channels {
            return Err(format!(
                "image {} has {} channels, expected {}",
                image_path.display(),
                image.shape.channels,
                network.net.channels
            )
            .into());
        }
        let destination = input
            .data
            .get_mut(slot * per_image..(slot + 1) * per_image)
            .ok_or_else(|| {
                format!(
                    "batch slot {slot} is outside the input tensor ({} values, {per_image} per image, batch {batch}, net {width}x{height}x{})",
                    input.shape.element_count() * batch,
                    network.net.channels
                )
            })?;
        if image.data.len() != destination.len() {
            return Err(format!(
                "image {} resized to {}x{}x{} has {} values, but the network input slot holds {} (net {width}x{height}x{}, batch {batch})",
                image_path.display(),
                image.shape.width,
                image.shape.height,
                image.shape.channels,
                image.data.len(),
                destination.len(),
                network.net.channels
            )
            .into());
        }
        destination.copy_from_slice(&image.data);

        let mask_path = segmentation_mask_path(image_path);
        if !mask_path.exists() {
            return Err(format!(
                "missing segmentation mask for {} (looked for {})",
                image_path.display(),
                mask_path.display()
            )
            .into());
        }
        for (layer_index, data) in &mut truth_data {
            let shape = network.layers[*layer_index].output_shape;
            data.extend(load_segmentation_truth(
                &mask_path,
                shape.width,
                shape.height,
                shape.channels,
            )?);
        }
    }

    let mut truth: Vec<Option<Tensor>> = vec![None; network.layers.len()];
    for (layer_index, data) in truth_data {
        truth[layer_index] = Some(Tensor::from_data(
            network.layers[layer_index].output_shape,
            data,
        )?);
    }
    Ok((input, truth))
}

pub(crate) fn run(
    cfg: &str,
    data: &str,
    options: &TrainOptions,
    backend: OxideBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_config = DataConfig::from_path(data)?;
    let mut executable = GpuExecutableNetwork::from_path(cfg, backend)?;
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
    println!("segmentation training: native GPU loss, no geometric augmentation");

    let train_images = load_paths_file(data_config.required_train_list()?)?;
    println!(
        "loaded {} images from {}",
        train_images.len(),
        data_config.required_train_list()?.display()
    );
    if train_images.is_empty() {
        return Err("training list is empty".into());
    }
    print_training_configuration(
        train_images.len(),
        executable.network.net.batch,
        executable.network.net.subdivisions,
        executable.network.net.max_batches,
        executable.network.net.learning_rate,
        executable.network.net.momentum,
        executable.network.net.decay,
    );

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
    if let Some(scale) = options.loss_scale {
        executable.set_loss_scale(scale);
    }
    let mut scaler = lightnet_core::LossScaler::new(executable.loss_scale());
    executable.set_loss_scale(scaler.scale());

    let batch = executable.network.net.batch.max(1);
    let subdivisions = executable.network.net.subdivisions.max(1);
    let images_per_iteration = batch * subdivisions;
    let mut batch_num = (seen as usize) / images_per_iteration;
    let mut epoch = (seen as usize) / train_images.len();
    let mut epoch_images = epoch_order(&train_images, options.seed, epoch);
    let mut image_cursor = (seen as usize) % train_images.len();
    let mut progress = TrainingProgress::new(batch_num);
    let mut loss_log = TrainingLossLog::new(&backup_dir, &base_name)?;
    let mut validation = if options.map {
        Some(SegValidation::new(
            options,
            &executable.network,
            &data_config,
            train_images.len(),
            images_per_iteration,
            batch_num,
            &backup_dir,
            &base_name,
        )?)
    } else {
        None
    };

    while executable.network.net.max_batches == 0 || batch_num < executable.network.net.max_batches
    {
        let iteration_start = Instant::now();
        let lr = compute_learning_rate(&executable.network.net, batch_num);
        executable.set_train_iteration(batch_num);
        let mut iter_loss = 0.0f32;

        for subdivision in 0..subdivisions {
            let (input, truth) = load_batch(
                &executable.network,
                &epoch_images,
                image_cursor + subdivision * batch,
            )?;
            let forward = executable.forward_training_device(&input)?;
            let loss = executable.compute_segmentation_training_loss_gpu(
                &forward, &truth, batch_num,
            )?;
            iter_loss += loss.loss * batch as f32;
            executable
                .backward_accumulate_seeded_gpu(&forward, &loss.layer_output_gradient_seeds)?;
        }

        let scale_step = if options.dynamic_loss_scale {
            Some(scaler.update(!executable.gradients_non_finite()?))
        } else {
            None
        };
        if scale_step == Some(lightnet_core::ScaleStep::Skip) {
            executable.zero_train_buffers_gpu()?;
            executable.set_loss_scale(scaler.scale());
            println!(
                "{batch_num}: non-finite gradients; loss scale -> {}",
                scaler.scale()
            );
        } else {
            executable.update_weights_gpu_deferred(
                lr,
                executable.network.net.momentum,
                executable.network.net.decay,
                images_per_iteration,
            )?;
            if scale_step.is_some() {
                executable.set_loss_scale(scaler.scale());
            }
        }

        batch_num += 1;
        seen += images_per_iteration as u64;
        image_cursor += images_per_iteration;
        if image_cursor >= train_images.len() {
            epoch += 1;
            epoch_images = epoch_order(&train_images, options.seed, epoch);
            image_cursor %= train_images.len();
        }

        let loss = iter_loss / images_per_iteration as f32;
        progress.print(
            batch_num,
            loss,
            lr,
            seen,
            executable.network.net.max_batches,
            iteration_start.elapsed(),
            &validation
                .as_ref()
                .map_or_else(|| map_progress_fields(None), SegValidation::progress_fields),
        );
        loss_log.record(
            batch_num,
            seen,
            loss,
            progress.avg_loss.unwrap_or(loss),
            lr,
            iteration_start.elapsed().as_secs_f64() * 1000.0,
        )?;

        if let Some(validation) = &mut validation
            && validation
                .schedule
                .due(batch_num, executable.network.net.max_batches)
        {
            println!(
                "Validating current training weights on all {} images...",
                validation.images.len()
            );
            // Deferred updates leave the device parameters authoritative: sync, then
            // run a batch-1 inference copy.
            executable.sync_host_weights_from_device()?;
            let mut inference = GpuExecutableNetwork::new(
                executable.network.with_batch(1),
                OxideBackend::new()?,
            );
            inference.params = executable.params.clone();
            let report = validation.evaluate(&inference)?;
            if validation.record(&report, batch_num, seen)? {
                let checkpoint = backup_dir.join(format!("{base_name}_best.weights"));
                save_weights(&executable.network, &executable.params, &checkpoint, seen)?;
                println!("saved {}", checkpoint.display());
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

/// Argmax over the channel axis of a channel-major `[classes, spatial]` slab.
fn argmax_class(data: &[f32], classes: usize, spatial: usize, pixel: usize) -> usize {
    let mut best = 0;
    for class in 1..classes {
        if data[class * spatial + pixel] > data[best * spatial + pixel] {
            best = class;
        }
    }
    best
}

/// Per-class and mean intersection-over-union over a validation list.
struct SegReport {
    class_iou: Vec<Option<f64>>,
    /// Pixels per class that were predicted and labelled / predicted only /
    /// labelled only.
    true_positive: Vec<u64>,
    false_positive: Vec<u64>,
    false_negative: Vec<u64>,
    mean_iou: f64,
    pixel_accuracy: f64,
    images: usize,
    pixels: u64,
}

/// `-map` for segmentation: mean IoU on the `valid=` list every `-map_every`
/// iterations (default `max(100, images / batch)`), after burn-in, and at the
/// final iteration. Binary heads count two classes (background, foreground).
struct SegValidation {
    images: Vec<PathBuf>,
    class_names: Vec<String>,
    schedule: MapSchedule,
    best: Option<f64>,
    last: Option<f64>,
    log: fs::File,
}

impl SegValidation {
    fn new(
        options: &TrainOptions,
        network: &lightnet_core::Network,
        data: &DataConfig,
        train_images: usize,
        images_per_iteration: usize,
        iteration: usize,
        backup: &std::path::Path,
        name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let valid_list = data
            .valid
            .as_ref()
            .ok_or("-map needs a valid= list in the data file")?;
        let images = load_paths_file(valid_list)?;
        if images.is_empty() {
            return Err(format!("validation list {} is empty", valid_list.display()).into());
        }
        let head = *segmentation_head_layers(network)
            .first()
            .ok_or("-map: no segmentation head")?;
        let channels = network.layers[head].output_shape.channels;
        let count = if channels == 1 { 2 } else { channels };
        let class_names = match data.names.as_ref().map(lightnet_core::load_names_file) {
            Some(Ok(names)) if names.len() == count => names,
            _ if count == 2 && channels == 1 => vec!["background".into(), "foreground".into()],
            _ => (0..count).map(|i| format!("class_{i}")).collect(),
        };
        let interval = options
            .map_every
            .unwrap_or_else(|| (train_images / images_per_iteration.max(1)).max(100));
        println!(
            "mIoU: {} validation images, every {interval} iterations (after burn-in {}), and at final iteration",
            images.len(),
            network.net.burn_in
        );
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(backup.join(format!("{name}_miou.csv")))?;
        if log.metadata()?.len() == 0 {
            writeln!(log, "iteration,seen,mean_iou,pixel_accuracy,best_mean_iou")?;
        }
        Ok(Self {
            images,
            class_names,
            schedule: MapSchedule {
                interval,
                burn_in: network.net.burn_in,
                last_iteration: iteration,
            },
            best: None,
            last: None,
            log,
        })
    }

    fn evaluate(
        &self,
        inference: &GpuExecutableNetwork<OxideBackend>,
    ) -> Result<SegReport, Box<dyn std::error::Error>> {
        let network = &inference.network;
        let head = *segmentation_head_layers(network)
            .first()
            .ok_or("-map: no segmentation head")?;
        let shape = network.layers[head].output_shape;
        let classes = shape.channels;
        let count = if classes == 1 { 2 } else { classes };
        let spatial = shape.width * shape.height;
        let (mut inter, mut predicted, mut actual) =
            (vec![0u64; count], vec![0u64; count], vec![0u64; count]);
        let (mut correct, mut total) = (0u64, 0u64);
        let progress = ValidationBar::new(self.images.len());
        for (done, path) in self.images.iter().enumerate() {
            let image = PlanarRgbImage::load(path)?
                .resize(network.net.width, network.net.height)?;
            let input = Tensor::from_data(network.net.input_shape(), image.data)?;
            // Device forward; only the head output is copied back to the host.
            let state = inference.forward_inference_device(&input)?;
            let output = inference.download_layer_output(&state, head)?;
            let mask = segmentation_mask_path(path);
            if !mask.exists() {
                return Err(format!(
                    "missing segmentation mask for {} (looked for {})",
                    path.display(),
                    mask.display()
                )
                .into());
            }
            let truth = load_segmentation_truth(&mask, shape.width, shape.height, classes)?;
            for pixel in 0..spatial {
                let (p, a) = if classes == 1 {
                    (usize::from(output[pixel] > 0.5), usize::from(truth[pixel] > 0.5))
                } else {
                    (
                        argmax_class(&output, classes, spatial, pixel),
                        argmax_class(&truth, classes, spatial, pixel),
                    )
                };
                predicted[p] += 1;
                actual[a] += 1;
                total += 1;
                if p == a {
                    inter[p] += 1;
                    correct += 1;
                }
            }
            progress.draw(done + 1);
        }
        progress.finish();
        let class_iou: Vec<Option<f64>> = (0..count)
            .map(|c| {
                let union = predicted[c] + actual[c] - inter[c];
                (union > 0).then(|| inter[c] as f64 / union as f64)
            })
            .collect();
        let present: Vec<f64> = class_iou.iter().flatten().copied().collect();
        Ok(SegReport {
            mean_iou: present.iter().sum::<f64>() / present.len().max(1) as f64,
            pixel_accuracy: correct as f64 / total.max(1) as f64,
            class_iou,
            false_positive: (0..count).map(|c| predicted[c] - inter[c]).collect(),
            false_negative: (0..count).map(|c| actual[c] - inter[c]).collect(),
            true_positive: inter,
            images: self.images.len(),
            pixels: total,
        })
    }

    /// Print and log `report`; true when mean IoU is the best so far.
    fn record(
        &mut self,
        report: &SegReport,
        iteration: usize,
        seen: u64,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let improved = self.best.is_none_or(|best| report.mean_iou >= best);
        if improved {
            self.best = Some(report.mean_iou);
        }
        self.last = Some(report.mean_iou);
        print!("{}", self.format_report(report, iteration));
        if improved {
            println!("New best mIoU, saving weights!");
        }
        writeln!(
            self.log,
            "{iteration},{seen},{:.8},{:.8},{:.8}",
            report.mean_iou,
            report.pixel_accuracy,
            self.best.unwrap_or(0.0)
        )?;
        self.log.flush()?;
        self.schedule.last_iteration = iteration;
        Ok(improved)
    }

    /// The `last=..., best=..., map=<burn_in>` fields of the training line:
    /// validation starts once the cfg `burn_in` iteration is reached.
    fn progress_fields(&self) -> String {
        let show =
            |value: Option<f64>| colour::format_map_accuracy(value.map_or(-1.0, |v| v as f32));
        format!(
            ", last={}, best={}, map={}",
            show(self.last),
            show(self.best),
            self.schedule.burn_in
        )
    }

    /// Per-class table (IoU, Dice, pixel precision and recall, pixel counts) and a
    /// summary, in the layout of the detector's mAP report.
    fn format_report(&self, report: &SegReport, iteration: usize) -> String {
        use colour::EColour;
        let percent = |num: u64, den: u64| {
            if den > 0 { 100.0 * num as f32 / den as f32 } else { 0.0 }
        };
        let mut out = format!(
            "\n\n Segmentation validation at iteration {iteration}: {} images, {} pixels\n \
             IoU=intersection over union, Dice=2TP/(2TP+FP+FN),\n \
             Prec/Rec=pixel precision/recall, GT=ground-truth pixels, Pred=predicted pixels\n\n  \
             Id Name                      IoU     Dice     Prec      Rec         GT       Pred\n  \
             -- -------------------- -------- -------- -------- -------- ---------- ----------\n",
            report.images, report.pixels
        );
        for (class, name) in self.class_names.iter().enumerate() {
            let (tp, fp, fn_) = (
                report.true_positive[class],
                report.false_positive[class],
                report.false_negative[class],
            );
            let iou = report.class_iou[class].map_or(0.0, |v| 100.0 * v as f32);
            out += &format!(
                "  {} {} {} {} {} {} {} {}\n",
                colour::format_int(class, EColour::Normal, 2),
                colour::format_name(name, EColour::BrightWhite, 20),
                colour::format_ratio(iou, 8),
                colour::format_ratio(percent(2 * tp, 2 * tp + fp + fn_), 8),
                colour::format_ratio(percent(tp, tp + fp), 8),
                colour::format_ratio(percent(tp, tp + fn_), 8),
                colour::format_int((tp + fn_) as usize, EColour::Normal, 10),
                colour::format_int((tp + fp) as usize, EColour::Normal, 10),
            );
        }
        out += &format!(
            "\n-> pixel accuracy={}, classes scored={}/{}\n\
             -> mean IoU (mIoU)={}, best so far={}\n",
            colour::format_map_accuracy(report.pixel_accuracy as f32),
            report.class_iou.iter().flatten().count(),
            report.class_iou.len(),
            colour::format_map_accuracy(report.mean_iou as f32),
            colour::format_map_accuracy(self.best.unwrap_or(report.mean_iou) as f32),
        );
        out
    }
}

/// One-line progress bar for a validation pass, redrawn in place with `\r`.
/// Silent when stdout is not a terminal, so redirected logs stay clean.
struct ValidationBar {
    total: usize,
    started: Instant,
    live: bool,
}

impl ValidationBar {
    const WIDTH: usize = 30;

    fn new(total: usize) -> Self {
        use std::io::IsTerminal;
        Self {
            total,
            started: Instant::now(),
            live: std::io::stdout().is_terminal(),
        }
    }

    /// `[#########-----------]  42/100  42%  3.1s  eta 4.3s`
    fn draw(&self, done: usize) {
        if !self.live {
            return;
        }
        let fraction = done as f32 / self.total.max(1) as f32;
        let filled = ((fraction * Self::WIDTH as f32) as usize).min(Self::WIDTH);
        let elapsed = self.started.elapsed().as_secs_f32();
        let eta = if done > 0 { elapsed * (self.total - done) as f32 / done as f32 } else { 0.0 };
        print!(
            "\r  [{}{}] {done:>width$}/{}  {:>3.0}%  {elapsed:.1}s  eta {eta:.1}s ",
            "#".repeat(filled),
            "-".repeat(Self::WIDTH - filled),
            self.total,
            fraction * 100.0,
            width = self.total.to_string().len(),
        );
        let _ = std::io::stdout().flush();
    }

    /// Erase the bar so the report starts on a clean line.
    fn finish(&self) {
        if self.live {
            print!("\r{}\r", " ".repeat(Self::WIDTH + 48));
            let _ = std::io::stdout().flush();
        }
    }
}
