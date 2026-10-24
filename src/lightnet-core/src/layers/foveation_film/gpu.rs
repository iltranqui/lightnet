//! GPU-runtime glue for `[foveation_film]`.
use crate::error::{Error, Result};
use crate::foveation_film::FoveationFilmForwardCache;
use crate::model::{FoveationFilmSpec, Layer};
use crate::runtime::{
    accumulate_vec,
    LayerTrainBuffers,
    GpuForwardState,
    FoveationFilmParams, GpuExecutableNetwork, HyperbolicFoveationForwardCache, backend_error,
    tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native FiLM modulation driven by the selected hyperbolic-foveation layer's host
    /// parameter cache. Returns the output and the embedding/gamma/beta host cache backward uses.
    pub(crate) fn forward_foveation_film_gpu(
        &self,
        layer: &Layer,
        spec: &FoveationFilmSpec,
        params: &FoveationFilmParams,
        hyperbolic_foveation_caches: &[Option<HyperbolicFoveationForwardCache>],
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, FoveationFilmForwardCache)> {
        let hf_cache = hyperbolic_foveation_caches
            .get(spec.hf_layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[foveation_film] selected HF layer #{} has no forward cache",
                    spec.hf_layer
                ))
            })?;
        let batch = layer.input_shape.batch;
        let channels = layer.input_shape.channels;
        let hf_params = self
            .backend
            .upload_f32(&[batch, spec.param_count], &hf_cache.params_per_batch)
            .map_err(|err| backend_error("upload FoveationFilm HF parameters", err))?;
        let w_embed = self
            .backend
            .upload_f32(&[spec.embed_dim, spec.param_count], &params.w_embed)
            .map_err(|err| {
                backend_error("upload FoveationFilm embedding weights", err)
            })?;
        let b_embed = self
            .backend
            .upload_f32(&[spec.embed_dim], &params.b_embed)
            .map_err(|err| backend_error("upload FoveationFilm embedding bias", err))?;
        let w_gamma = self
            .backend
            .upload_f32(&[channels, spec.embed_dim], &params.w_gamma)
            .map_err(|err| backend_error("upload FoveationFilm gamma weights", err))?;
        let b_gamma = self
            .backend
            .upload_f32(&[channels], &params.b_gamma)
            .map_err(|err| backend_error("upload FoveationFilm gamma bias", err))?;
        let w_beta = self
            .backend
            .upload_f32(&[channels, spec.embed_dim], &params.w_beta)
            .map_err(|err| backend_error("upload FoveationFilm beta weights", err))?;
        let b_beta = self
            .backend
            .upload_f32(&[channels], &params.b_beta)
            .map_err(|err| backend_error("upload FoveationFilm beta bias", err))?;
        let (output, embedding, gamma, beta) = self
            .backend
            .foveation_film_forward_f32(
                current_input,
                &hf_params,
                &w_embed,
                &b_embed,
                &w_gamma,
                &b_gamma,
                &w_beta,
                &b_beta,
                &tensor_backend_shape(layer.input_shape),
                spec.param_count,
                spec.embed_dim,
                launch_config,
            )
            .map_err(|err| backend_error("FoveationFilm native forward", err))?;
        let cache = FoveationFilmForwardCache {
            embedding: self.backend.download_f32(&embedding).map_err(|err| {
                backend_error("download FoveationFilm embedding cache", err)
            })?,
            gamma: self.backend.download_f32(&gamma).map_err(|err| {
                backend_error("download FoveationFilm gamma cache", err)
            })?,
            beta: self.backend.download_f32(&beta).map_err(|err| {
                backend_error("download FoveationFilm beta cache", err)
            })?,
        };
        Ok((output, cache))
    }

    /// Native FoveationFilm backward; the HF-layer gradient goes into the shared auxiliary slots.
    pub(crate) fn backward_foveation_film_gpu(
        &self,
        layer: &Layer,
        spec: &FoveationFilmSpec,
        params: &FoveationFilmParams,
        foveation_hf_auxiliary_gradients: &mut [Option<Vec<f32>>],
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let film_cache = forward
            .cpu_forward
            .foveation_film_caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::InvalidConfig(format!(
                "missing [foveation_film] forward cache for layer #{}",
                layer.index
            )))?;
        let hf_cache = forward
            .cpu_forward
            .hyperbolic_foveation_caches
            .get(spec.hf_layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::InvalidConfig(format!(
                "[foveation_film] selected HF layer #{} has no forward cache",
                spec.hf_layer
            )))?;
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for FoveationFilm layer #{}",
                    layer.index
                ))
            })?
        };
        let batch = layer.input_shape.batch;
        let channels = layer.input_shape.channels;
        let hf_params = self.backend
            .upload_f32(&[batch, spec.param_count], &hf_cache.params_per_batch)
            .map_err(|err| backend_error("upload FoveationFilm backward HF cache", err))?;
        let embedding = self.backend
            .upload_f32(&[batch, spec.embed_dim], &film_cache.embedding)
            .map_err(|err| backend_error("upload FoveationFilm embedding cache", err))?;
        let gamma = self.backend
            .upload_f32(&[batch, channels], &film_cache.gamma)
            .map_err(|err| backend_error("upload FoveationFilm gamma cache", err))?;
        let w_embed = self.backend
            .upload_f32(&[spec.embed_dim, spec.param_count], &params.w_embed)
            .map_err(|err| backend_error("upload FoveationFilm embedding weights", err))?;
        let w_gamma = self.backend
            .upload_f32(&[channels, spec.embed_dim], &params.w_gamma)
            .map_err(|err| backend_error("upload FoveationFilm gamma weights", err))?;
        let w_beta = self.backend
            .upload_f32(&[channels, spec.embed_dim], &params.w_beta)
            .map_err(|err| backend_error("upload FoveationFilm beta weights", err))?;
        let gradients = self.backend
            .foveation_film_backward_f32(
                current_input,
                &grad,
                &hf_params,
                &embedding,
                &gamma,
                &w_embed,
                &w_gamma,
                &w_beta,
                &tensor_backend_shape(layer.input_shape),
                spec.param_count,
                spec.embed_dim,
                config,
            )
            .map_err(|err| backend_error("FoveationFilm native backward", err))?;
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::FoveationFilm(train)) =
            buffers.get_mut(layer.index)
        else {
            return Err(Error::InvalidConfig(format!(
                "missing FoveationFilm train buffers for layer #{}",
                layer.index
            )));
        };
        accumulate_vec(&mut train.w_embed_updates, &self.backend.download_f32(&gradients.w_embed)
            .map_err(|err| backend_error("download FoveationFilm embedding gradient", err))?)?;
        accumulate_vec(&mut train.b_embed_updates, &self.backend.download_f32(&gradients.b_embed)
            .map_err(|err| backend_error("download FoveationFilm embedding bias gradient", err))?)?;
        accumulate_vec(&mut train.w_gamma_updates, &self.backend.download_f32(&gradients.w_gamma)
            .map_err(|err| backend_error("download FoveationFilm gamma gradient", err))?)?;
        accumulate_vec(&mut train.b_gamma_updates, &self.backend.download_f32(&gradients.b_gamma)
            .map_err(|err| backend_error("download FoveationFilm gamma bias gradient", err))?)?;
        accumulate_vec(&mut train.w_beta_updates, &self.backend.download_f32(&gradients.w_beta)
            .map_err(|err| backend_error("download FoveationFilm beta gradient", err))?)?;
        accumulate_vec(&mut train.b_beta_updates, &self.backend.download_f32(&gradients.b_beta)
            .map_err(|err| backend_error("download FoveationFilm beta bias gradient", err))?)?;
        drop(buffers);
        if spec.backprop_to_hf {
            let hf_gradient = self.backend.download_f32(&gradients.hf_params)
                .map_err(|err| backend_error("download FoveationFilm HF gradient", err))?;
            let slot = foveation_hf_auxiliary_gradients
                .get_mut(spec.hf_layer)
                .ok_or_else(|| Error::InvalidConfig(format!(
                    "FoveationFilm selected HF layer #{} is unavailable during backward",
                    spec.hf_layer
                )))?;
            if let Some(existing) = slot {
                if existing.len() != hf_gradient.len() {
                    return Err(Error::InvalidConfig(format!(
                        "FoveationFilm HF gradient layout mismatch at layer #{}",
                        spec.hf_layer
                    )));
                }
                for (sum, value) in existing.iter_mut().zip(hf_gradient) {
                    *sum += value;
                }
            } else {
                *slot = Some(hf_gradient);
            }
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            gradients.input,
            layer_output_gradients,
        )?;
        Ok(())
    }
}
