use super::*;

/// Device-side dense-head loss for `[jev_ce]` and `[segmentation]`.
///
/// Layers without truth are skipped; any other dense-truth head with truth is
/// rejected so the caller can use the host
/// [`crate::dense_loss::compute_dense_training_loss`] path instead.
impl<B> GpuExecutableNetwork<B>
where
    B: DenseLossBackend,
    B::Error: std::fmt::Display,
{
    pub fn compute_jev_ce_training_loss_gpu(
        &self,
        forward: &GpuForwardState<B::Tensor>,
        truth: &[Option<Tensor>],
    ) -> Result<crate::dense_loss::DenseTrainingLoss> {
        if truth.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "dense truth count mismatch: expected {}, got {}",
                self.network.layers.len(),
                truth.len()
            )));
        }
        let config = ElementwiseLaunchConfig::default();
        let mut seeds = vec![None; self.network.layers.len()];
        let (mut total, mut count) = (0.0f32, 0usize);
        for layer in &self.network.layers {
            let Some(target) = truth[layer.index].as_ref() else {
                continue;
            };
            let LayerKind::JevCe(spec) = &layer.kind else {
                return Err(Error::UnsupportedLayer {
                    layer_index: layer.index,
                    name: format!("{} (native dense loss)", layer.kind.name()),
                });
            };
            if layer.index == 0 {
                return Err(Error::InvalidConfig(
                    "jev_ce cannot be the first layer".to_owned(),
                ));
            }
            if target.shape != layer.output_shape {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([jev_ce]) truth shape {} != output shape {}",
                    layer.index, target.shape, layer.output_shape
                )));
            }
            let logits = forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing device logits for jev_ce at layer #{}",
                    layer.index
                ))
            })?;
            let truth_gpu = self
                .backend
                .upload_f32(&tensor_backend_shape(target.shape), &target.data)
                .map_err(|err| backend_error("upload jev_ce truth", err))?;
            let (cost, delta) = self
                .backend
                .jev_ce_loss_f32(
                    logits,
                    &truth_gpu,
                    &tensor_backend_shape(layer.input_shape),
                    spec.groups,
                    spec.candidates,
                    spec.temperature,
                    spec.normalize,
                    spec.ignore_empty,
                    spec.label_smoothing,
                    spec.scale,
                    spec.channel_major,
                    config,
                )
                .map_err(|err| backend_error("jev_ce loss", err))?;
            let data = self
                .backend
                .download_f32(&delta)
                .map_err(|err| backend_error("download jev_ce delta", err))?;
            seeds[layer.index] = Some(Tensor::from_data(layer.input_shape, data)?);
            total += cost;
            count += 1;
        }
        Ok(crate::dense_loss::DenseTrainingLoss {
            loss: if count == 0 {
                0.0
            } else {
                total / count as f32
            },
            layer_output_gradient_seeds: seeds,
        })
    }

    pub fn compute_segmentation_training_loss_gpu(
        &self,
        forward: &GpuForwardState<B::Tensor>,
        truth: &[Option<Tensor>],
        iteration: usize,
    ) -> Result<crate::dense_loss::DenseTrainingLoss> {
        if truth.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "dense truth count mismatch: expected {}, got {}",
                self.network.layers.len(),
                truth.len()
            )));
        }
        let config = ElementwiseLaunchConfig::default();
        let mut seeds = vec![None; self.network.layers.len()];
        let (mut total, mut count) = (0.0f32, 0usize);
        for layer in &self.network.layers {
            let Some(target) = truth[layer.index].as_ref() else {
                continue;
            };
            let head = match &layer.kind {
                LayerKind::Segmentation(spec) => SegmentationLossParams {
                    softmax: spec.loss == crate::model::SegmentationLoss::SoftmaxCe,
                    scale: spec.lambda_ce * spec.ce_normalizer * spec.cls_normalizer,
                    gate_strength: crate::segmentation::gate_strength(spec, iteration),
                    gate_t_correct: spec.gate_t_correct,
                    gate_max_compensation: spec.gate_max_compensation,
                    max_delta: spec.max_delta,
                    seed_index: layer.index,
                },
                // Legacy heads: a trailing `[logistic]` (binary) or `[softmax] spatial=1`
                // trained on pixel masks. Plain cross-entropy, no gate, no clip. The
                // delta is already with respect to the logits, so a softmax head seeds
                // the layer before it: the softmax backward would apply its Jacobian again.
                LayerKind::Logistic => SegmentationLossParams::legacy(false, layer.index),
                LayerKind::Softmax { spatial: true, .. } if layer.index > 0 => {
                    SegmentationLossParams::legacy(true, layer.index - 1)
                }
                _ => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{} (native dense loss)", layer.kind.name()),
                    });
                }
            };
            if target.shape != layer.output_shape {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([segmentation]) truth shape {} != output shape {}",
                    layer.index, target.shape, layer.output_shape
                )));
            }
            let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing device output for segmentation at layer #{}",
                    layer.index
                ))
            })?;
            let truth_gpu = self
                .backend
                .upload_f32(&tensor_backend_shape(target.shape), &target.data)
                .map_err(|err| backend_error("upload segmentation truth", err))?;
            let (cost, delta) = self
                .backend
                .segmentation_loss_f32(
                    output,
                    &truth_gpu,
                    &tensor_backend_shape(layer.output_shape),
                    layer.output_shape.channels,
                    layer.output_shape.width * layer.output_shape.height,
                    head.softmax,
                    head.scale,
                    head.gate_strength,
                    head.gate_t_correct,
                    head.gate_max_compensation,
                    head.max_delta,
                    head.max_delta < crate::segmentation::FLT_MAX,
                    config,
                )
                .map_err(|err| backend_error("segmentation loss", err))?;
            let data = self
                .backend
                .download_f32(&delta)
                .map_err(|err| backend_error("download segmentation delta", err))?;
            seeds[head.seed_index] = Some(Tensor::from_data(layer.output_shape, data)?);
            total += cost;
            count += 1;
        }
        Ok(crate::dense_loss::DenseTrainingLoss {
            loss: if count == 0 {
                0.0
            } else {
                total / count as f32
            },
            layer_output_gradient_seeds: seeds,
        })
    }
}

/// Per-head inputs of the native segmentation loss.
struct SegmentationLossParams {
    softmax: bool,
    scale: f32,
    gate_strength: f32,
    gate_t_correct: f32,
    gate_max_compensation: f32,
    max_delta: f32,
    /// Layer whose output receives the delta.
    seed_index: usize,
}

impl SegmentationLossParams {
    /// Cross-entropy without gate or clip, the defaults of `[segmentation]`.
    fn legacy(softmax: bool, seed_index: usize) -> Self {
        Self {
            softmax,
            scale: 1.0,
            gate_strength: 0.0,
            gate_t_correct: 0.9,
            gate_max_compensation: 10.0,
            max_delta: crate::segmentation::FLT_MAX,
            seed_index,
        }
    }
}
