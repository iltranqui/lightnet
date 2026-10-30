use super::*;

use crate::model::YoloIouLoss;
use crate::train::YoloLabel;

/// Loss, device gradient seeds and statistics of the native classic-YOLO loss.
pub struct GpuYoloLoss<T> {
    pub loss: f32,
    /// Seeds indexed by network layer; feed them to
    /// [`GpuExecutableNetwork::backward_accumulate_device_seeded_gpu`].
    pub device_seeds: Vec<Option<T>>,
    /// Mean IoU of the box assignments (summed over heads).
    pub avg_iou: f32,
    /// Number of box assignments (summed over heads).
    pub box_count: usize,
    pub total_bbox: usize,
    pub rewritten_bbox: usize,
}

const YOLO_GPU_MAX_ANCHORS: usize = 32;
const YOLO_GPU_MAX_BOXES: usize = 1024;

/// Native classic-YOLO CIoU training loss (`[yolo]` heads on a device backend).
///
/// The head output on the device is raw, so each head is activated here and the
/// loss reads the activated tensor; the seeds are deltas on the head output, just
/// like the host loss. Returns `Ok(None)` when the network or backend is outside
/// the supported subset (same subset as the reference `yolo_gpu=1`), so callers
/// keep using [`crate::compute_yolo_training_loss_with_state`].
impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn compute_yolo_training_loss_gpu(
        &self,
        forward: &GpuForwardState<B::Tensor>,
        labels: &[Vec<YoloLabel>],
    ) -> Result<Option<GpuYoloLoss<B::Tensor>>> {
        let net = &self.network.net;
        let batch_size = net.batch.max(1);
        if labels.len() != batch_size {
            return Err(Error::InvalidConfig(format!(
                "training label batch mismatch: expected {batch_size}, got {}",
                labels.len()
            )));
        }
        if net.obb
            || net.adversarial
            || net.equidistant_point != 0
            || net.badlabels_rejection_percentage != 0.0
            || net.num_sigmas_reject_badlabels != 0.0
        {
            return Ok(None);
        }
        let mut heads = Vec::new();
        for layer in &self.network.layers {
            match &layer.kind {
                LayerKind::Yolo(spec) => {
                    let supported = spec.iou_loss == YoloIouLoss::Ciou
                        && spec.iou_thresh_kind == YoloIouLoss::Iou
                        && !spec.focal_loss
                        && spec.label_smooth_eps == 0.0
                        && spec.class_multipliers.is_none()
                        && spec.class_map.is_none()
                        && spec.anchors.len() / 2 <= YOLO_GPU_MAX_ANCHORS
                        && spec.mask.iter().all(|&m| 2 * m + 1 < spec.anchors.len());
                    if !supported {
                        return Ok(None);
                    }
                    heads.push((layer, spec));
                }
                LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_) => return Ok(None),
                _ => {}
            }
        }
        if heads.is_empty() {
            return Ok(None);
        }
        let max_boxes = labels.iter().map(Vec::len).max().unwrap_or(0).max(1);
        if max_boxes > YOLO_GPU_MAX_BOXES {
            return Ok(None);
        }

        // Truth rows are (x, y, w, h, class); an all-zero row ends an image's list.
        const TRUTH_SIZE: usize = 5;
        let truths = max_boxes * TRUTH_SIZE;
        let mut truth_host = vec![0.0f32; batch_size * truths];
        for (image, image_labels) in labels.iter().enumerate() {
            for (row, label) in image_labels.iter().enumerate() {
                let base = image * truths + row * TRUTH_SIZE;
                truth_host[base..base + TRUTH_SIZE]
                    .copy_from_slice(&[label.x, label.y, label.w, label.h, label.class_id as f32]);
            }
        }
        let truth = self
            .backend
            .upload_f32(&[batch_size, truths, 1, 1], &truth_host)
            .map_err(|err| backend_error("upload YOLO truth", err))?;

        let config = ElementwiseLaunchConfig::default();
        let mut seeds: Vec<Option<B::Tensor>> = (0..self.network.layers.len()).map(|_| None).collect();
        let (mut loss, mut iou_sum, mut count_sum) = (0.0f32, 0.0f32, 0.0f32);
        let (mut total_bbox, mut rewritten_bbox) = (0.0f32, 0.0f32);
        for (layer, spec) in heads {
            let raw = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
                Error::InvalidConfig(format!("missing device output for YOLO head #{}", layer.index))
            })?;
            let shape = tensor_backend_shape(layer.output_shape);
            let activated = self
                .backend
                .yolo_activate_f32(raw, &shape, spec.classes, spec.scale_x_y, spec.new_coords, config)
                .map_err(|err| backend_error("GPU YOLO activation", err))?;
            let args = lightnet_cuda::YoloGpuLossArgs {
                classes: spec.classes,
                anchors: spec.mask.len(),
                total_anchors: spec.anchors.len() / 2,
                max_boxes,
                truth_size: TRUTH_SIZE,
                truths,
                net_width: net.width,
                net_height: net.height,
                ignore_thresh: spec.ignore_thresh,
                iou_normalizer: spec.iou_normalizer,
                obj_normalizer: spec.obj_normalizer,
                max_delta: spec.max_delta,
                new_coords: spec.new_coords,
                objectness_smooth: spec.objectness_smooth,
                iou_thresh: spec.iou_thresh,
                truth_thresh: spec.truth_thresh,
                mask: spec.mask.clone(),
                biases: spec.anchors.clone(),
            };
            let Some((delta, metrics)) = self
                .backend
                .yolo_loss_f32(&activated, &truth, &shape, &args, config)
                .map_err(|err| backend_error("GPU YOLO loss", err))?
            else {
                return Ok(None);
            };

            let sum = |field: fn(&lightnet_cuda::YoloImageMetrics) -> f32| metrics.iter().map(field).sum::<f32>();
            let count = sum(|m| m.count);
            let iou_loss = sum(|m| m.total_iou_loss);
            let classification = sum(|m| m.classification_delta_squared);
            loss += if count > 0.0 { spec.iou_normalizer * (iou_loss / count) } else { 0.0 }
                + spec.obj_normalizer * classification;
            iou_sum += sum(|m| m.total_iou);
            count_sum += count;
            total_bbox += sum(|m| m.total_bbox);
            rewritten_bbox += sum(|m| m.rewritten_bbox);

            // C++ `backward_yolo_layer_gpu`: delta_normalizer scales only the propagated delta.
            seeds[layer.index] = Some(if spec.delta_normalizer != 1.0 {
                let zero = self
                    .backend
                    .zeros_f32(&shape)
                    .map_err(|err| backend_error("allocate YOLO seed", err))?;
                self.backend
                    .saxpy_f32(spec.delta_normalizer, &delta, &zero, config)
                    .map_err(|err| backend_error("scale YOLO seed", err))?
            } else {
                delta
            });
        }
        Ok(Some(GpuYoloLoss {
            loss,
            device_seeds: seeds,
            avg_iou: if count_sum > 0.0 { iou_sum / count_sum } else { 0.0 },
            box_count: count_sum as usize,
            total_bbox: total_bbox as usize,
            rewritten_bbox: rewritten_bbox as usize,
        }))
    }
}
