//! GPU-runtime glue for `[foveation_film]`.
use crate::error::{Error, Result};
use crate::foveation_film::FoveationFilmForwardCache;
use crate::model::{FoveationFilmSpec, Layer};
use crate::runtime::{
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
}
