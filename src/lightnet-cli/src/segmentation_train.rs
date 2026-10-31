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
use std::path::PathBuf;
use std::time::Instant;

use lightnet_core::segmentation_data::{load_segmentation_truth, segmentation_mask_path};
use lightnet_core::{
    DataConfig, GpuExecutableNetwork, LayerKind, PlanarRgbImage, Tensor, compute_learning_rate,
    initialize_training_params, load_paths_file, save_weights,
};
use lightnet_cuda_oxide::OxideBackend;

use super::{
    TrainOptions, TrainingLossLog, TrainingProgress, cfg_base_name, epoch_order,
    map_progress_fields, print_training_configuration,
};

/// True when the network trains through the segmentation loss: it has a
/// `[segmentation]` head and no detection head the detector loop would need.
pub(crate) fn is_segmentation_network(network: &lightnet_core::Network) -> bool {
    network
        .layers
        .iter()
        .any(|layer| matches!(layer.kind, LayerKind::Segmentation(_)))
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
    let per_image = input.shape.element_count() / batch;

    let mut truth_data: Vec<(usize, Vec<f32>)> = network
        .layers
        .iter()
        .filter(|layer| matches!(layer.kind, LayerKind::Segmentation(_)))
        .map(|layer| (layer.index, Vec::new()))
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
        input.data[slot * per_image..(slot + 1) * per_image].copy_from_slice(&image.data);

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
            &map_progress_fields(None),
        );
        loss_log.record(
            batch_num,
            seen,
            loss,
            progress.avg_loss.unwrap_or(loss),
            lr,
            iteration_start.elapsed().as_secs_f64() * 1000.0,
        )?;

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
