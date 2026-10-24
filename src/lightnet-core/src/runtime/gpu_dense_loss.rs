use super::*;

/// Device-side dense-head loss for `[jev_ce]`.
///
/// Only the JevCe loss is native for now.  Layers without truth are skipped;
/// any other dense-truth head with truth is rejected so the caller can use the
/// host [`crate::dense_loss::compute_dense_training_loss`] path instead.
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
}
