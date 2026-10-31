use super::*;

use crate::model::ModernYoloKind;
use crate::training_types::YoloLabel;

/// Loss and device gradient seeds of the native `[yolox]` / `[ppyoloe]` / `[yolonas]` / `[centernet]` loss.
pub struct GpuDenseHeadLoss<T> {
    /// Mean of the per-head cost, as [`crate::training_types::DenseTrainingLoss::loss`].
    pub loss: f32,
    /// Seeds indexed by network layer; feed them to
    /// [`GpuExecutableNetwork::backward_accumulate_device_seeded_gpu`].
    pub device_seeds: Vec<Option<T>>,
    /// Number of positive cells (modern heads) or centre cells (CenterNet), summed over heads.
    pub positives: usize,
}

/// Native training loss of the dense box-table heads on a device backend.
///
/// The head outputs held in the forward state are already activated (the forward arm applies the
/// head activation), so the loss kernels read them directly and the seeds never leave the device.
/// Returns `Ok(None)` when the network mixes in a classic YOLO head or the backend has no native
/// loss, so callers keep using [`crate::compute_dense_training_loss_at`].
impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn compute_dense_head_training_loss_gpu(
        &self,
        forward: &GpuForwardState<B::Tensor>,
        labels: &[Vec<YoloLabel>],
    ) -> Result<Option<GpuDenseHeadLoss<B::Tensor>>> {
        if !crate::training_types::network_uses_dense_heads(&self.network)
            || self.network.layers.iter().any(|layer| {
                matches!(
                    layer.kind,
                    LayerKind::Yolo(_) | LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_)
                )
            })
        {
            return Ok(None);
        }
        let truth = crate::training_types::dense_head_truth(&self.network, labels)?;
        let config = ElementwiseLaunchConfig::default();
        let net = &self.network.net;
        let mut seeds: Vec<Option<B::Tensor>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        let (mut total, mut count, mut positives) = (0.0f32, 0usize, 0usize);
        for layer in &self.network.layers {
            let Some(truth) = truth[layer.index].as_ref() else {
                continue;
            };
            let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing device output for [{}] head #{}",
                    layer.kind.name(),
                    layer.index
                ))
            })?;
            let shape = tensor_backend_shape(layer.output_shape);
            let truth_gpu = self
                .backend
                .upload_f32(&tensor_backend_shape(truth.shape), &truth.data)
                .map_err(|err| backend_error("upload dense-head truth", err))?;
            let (delta, cost, head_positives) = match &layer.kind {
                LayerKind::Yolox(spec) | LayerKind::Ppyoloe(spec) | LayerKind::Yolonas(spec) => {
                    // The host loss uses `ModernYoloSettings::default()`; stay in sync with it.
                    let settings = crate::modern_yolo::ModernYoloSettings::default();
                    let args = lightnet_cuda::ModernYoloGpuLossArgs {
                        classes: spec.classes,
                        coords: spec.coords(),
                        kind: match spec.kind {
                            ModernYoloKind::Yolox => 0,
                            ModernYoloKind::Ppyoloe => 1,
                            ModernYoloKind::Yolonas => 2,
                        },
                        max_boxes: spec.max_boxes,
                        truth_size: settings.truth_size,
                        cls_normalizer: settings.cls_normalizer,
                        obj_normalizer: settings.obj_normalizer,
                        iou_normalizer: settings.iou_normalizer,
                        ignore_thresh: settings.ignore_thresh,
                        max_delta: settings.max_delta,
                        label_smooth_eps: settings.label_smooth_eps,
                        objectness_smooth: settings.objectness_smooth,
                        min_object_cells: spec.min_object_cells,
                        vfl_gamma: spec.vfl_gamma,
                    };
                    let Some((delta, metrics)) = self
                        .backend
                        .modern_yolo_loss_f32(output, &truth_gpu, &shape, &args, config)
                        .map_err(|err| backend_error("GPU modern-head loss", err))?
                    else {
                        return Ok(None);
                    };
                    if metrics.iter().any(|m| m.invalid_truth != 0.0) {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} ([{}]) has an invalid truth box or class id",
                            layer.index,
                            layer.kind.name()
                        )));
                    }
                    // C++ reports `||delta||^2` over the whole batch as the cost.
                    let cost = metrics.iter().map(|m| m.delta_squared).sum::<f32>();
                    let hits = metrics.iter().map(|m| m.positives).sum::<f32>() as usize;
                    (delta, cost, hits)
                }
                LayerKind::CenterNet(spec) => {
                    let args = lightnet_cuda::CenterNetGpuLossArgs {
                        classes: spec.classes,
                        max_boxes: spec.max_boxes,
                        truth_size: 6,
                        net_width: net.width,
                        net_height: net.height,
                        center_min_radius: spec.center_min_radius,
                        anisotropic_gaussian: spec.anisotropic_gaussian != 0,
                        small_ref_size: spec.small_ref_size,
                        small_boost: spec.small_boost,
                        scale_min_px: spec.scale_min_px,
                        scale_max_px: spec.scale_max_px,
                        gaussian_iou: spec.gaussian_iou,
                        focal_alpha: spec.focal_alpha,
                        focal_beta: spec.focal_beta,
                        hm_normalizer: spec.hm_normalizer,
                        wh_normalizer: spec.wh_normalizer,
                        off_normalizer: spec.off_normalizer,
                        max_delta: spec.max_delta,
                    };
                    let Some((delta, metrics)) = self
                        .backend
                        .centernet_loss_f32(output, &truth_gpu, &shape, &args, config)
                        .map_err(|err| backend_error("GPU CenterNet loss", err))?
                    else {
                        return Ok(None);
                    };
                    if metrics.iter().any(|m| m.invalid_truth != 0.0) {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} ([centernet]) has an invalid truth box or class id",
                            layer.index
                        )));
                    }
                    // C++ reports `(hm + wh + off) / batch`.
                    let sum = metrics
                        .iter()
                        .map(|m| m.hm_loss + m.wh_loss + m.off_loss)
                        .sum::<f32>();
                    let hits = metrics.iter().map(|m| m.positives).sum::<f32>() as usize;
                    (delta, sum / layer.output_shape.batch.max(1) as f32, hits)
                }
                _ => continue,
            };
            seeds[layer.index] = Some(delta);
            total += cost;
            count += 1;
            positives += head_positives;
        }
        if count == 0 {
            return Ok(None);
        }
        Ok(Some(GpuDenseHeadLoss {
            loss: total / count as f32,
            device_seeds: seeds,
            positives,
        }))
    }
}
