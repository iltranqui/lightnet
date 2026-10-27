//! Foveation FiLM backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend, FoveationFilmGradients};

use crate::{launch, launch_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn foveation_film_forward_backend_f32(
        &self,
        input: &OxideTensorF32,
        hf_params: &OxideTensorF32,
        w_embed: &OxideTensorF32,
        b_embed: &OxideTensorF32,
        w_gamma: &OxideTensorF32,
        b_gamma: &OxideTensorF32,
        w_beta: &OxideTensorF32,
        b_beta: &OxideTensorF32,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_forward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let (batch, channels) = (shape[0], shape[1]);
        if param_count == 0 || embed_dim == 0 {
            return Err(OxideError::Shape(
                "foveation_film_forward_f32: param_count and embed_dim must be positive".to_owned(),
            ));
        }
        let embedding_pre = self.linear_f32(
            hf_params,
            w_embed,
            Some(b_embed),
            &[batch, embed_dim],
            batch,
            param_count,
            embed_dim,
            config,
        )?;
        let embedding = self.activation_f32(&embedding_pre, F32ActivationKind::Relu, config)?;
        let gamma = self.linear_f32(
            &embedding,
            w_gamma,
            Some(b_gamma),
            &[batch, channels],
            batch,
            embed_dim,
            channels,
            config,
        )?;
        let beta = self.linear_f32(
            &embedding,
            w_beta,
            Some(b_beta),
            &[batch, channels],
            batch,
            embed_dim,
            channels,
            config,
        )?;
        let output = self.foveation_film_affine_f32(input, &gamma, &beta, shape, config)?;
        Ok((output, embedding, gamma, beta))
    }

    pub(crate) fn foveation_film_affine_backend_f32(
        &self,
        input: &OxideTensorF32,
        gamma: &OxideTensorF32,
        beta: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input = self.f32v(input)?;
        let input = &*input;
        let gamma = self.f32v(gamma)?;
        let gamma = &*gamma;
        let beta = self.f32v(beta)?;
        let beta = &*beta;
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_affine_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: spatial extent overflows".to_owned())
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: channel extent overflows".to_owned())
        })?;
        let expected = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: tensor extent overflows".to_owned())
        })?;
        if input.len() != expected || gamma.len() != channel_count || beta.len() != channel_count {
            return Err(OxideError::Shape(
                "foveation_film_affine_f32: tensor extents do not match NCHW / [B,C] layouts"
                    .to_owned(),
            ));
        }
        let mut output = self.alloc_uninit(expected)?;
        // SAFETY: host validation proves the input and [B,C] operands cover
        // the full launch; each thread owns exactly one output element.
        unsafe {
            self.foveation_film.film_affine(
                &self.stream,
                launch::elementwise(expected, config),
                &input.buf,
                &gamma.buf,
                &beta.buf,
                &mut output,
                channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn foveation_film_backward_backend_f32(
        &self,
        input: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        hf_params: &OxideTensorF32,
        embedding: &OxideTensorF32,
        gamma: &OxideTensorF32,
        w_embed: &OxideTensorF32,
        w_gamma: &OxideTensorF32,
        w_beta: &OxideTensorF32,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<FoveationFilmGradients<OxideTensorF32>, OxideError> {
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_backward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        if param_count == 0 || embed_dim == 0 {
            return Err(OxideError::Shape(
                "foveation_film_backward_f32: param_count and embed_dim must be positive"
                    .to_owned(),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: spatial extent overflows".to_owned())
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: channel extent overflows".to_owned())
        })?;
        let input_count = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: input extent overflows".to_owned())
        })?;
        let embed_count = batch.checked_mul(embed_dim).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: embedding extent overflows".to_owned())
        })?;
        let hf_count = batch.checked_mul(param_count).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: HF extent overflows".to_owned())
        })?;
        let embed_weights_count = embed_dim.checked_mul(param_count).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_backward_f32: embedding weight extent overflows".to_owned(),
            )
        })?;
        let channel_weights_count = channels.checked_mul(embed_dim).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_backward_f32: channel weight extent overflows".to_owned(),
            )
        })?;
        let input = self.f32v(input)?;
        let input = &*input;
        let hf_params = self.f32v(hf_params)?;
        let hf_params = &*hf_params;
        let embedding = self.f32v(embedding)?;
        let embedding = &*embedding;
        let w_embed = self.f32v(w_embed)?;
        let w_embed = &*w_embed;
        let w_gamma = self.f32v(w_gamma)?;
        let w_gamma = &*w_gamma;
        let w_beta = self.f32v(w_beta)?;
        let w_beta = &*w_beta;
        if input.len() != input_count
            || hf_params.len() != hf_count
            || embedding.len() != embed_count
            || w_embed.len() != embed_weights_count
            || w_gamma.len() != channel_weights_count
            || w_beta.len() != channel_weights_count
        {
            return Err(OxideError::Shape(
                "foveation_film_backward_f32: operand extents do not match configured layouts"
                    .to_owned(),
            ));
        }
        let (dx, dgamma, dbeta) =
            self.foveation_film_affine_backward_f32(input, output_gradient, gamma, shape, config)?;
        let dgamma_f = self.f32v(&dgamma)?;
        let dgamma_f = &*dgamma_f;
        let dbeta_f = self.f32v(&dbeta)?;
        let dbeta_f = &*dbeta_f;
        let mut d_embedding = self.alloc_uninit(embed_count)?;
        // Every embedding-gradient element is written by one thread.
        unsafe {
            self.foveation_film.film_embedding_backward(
                &self.stream,
                launch::elementwise(embed_count, config),
                &embedding.buf,
                &dgamma_f.buf,
                &dbeta_f.buf,
                &w_gamma.buf,
                &w_beta.buf,
                &mut d_embedding,
                channels,
                embed_dim,
            )
        }
        .map_err(launch_err)?;

        let dw_embed_count = embed_weights_count;
        let dw_channel_count = channel_weights_count;
        let target_count = dw_embed_count
            .max(embed_dim)
            .max(dw_channel_count)
            .max(channels)
            .max(hf_count);
        let mut dw_embed = self.alloc_uninit(dw_embed_count)?;
        let mut db_embed = self.alloc_uninit(embed_dim)?;
        let mut dw_gamma = self.alloc_uninit(dw_channel_count)?;
        let mut db_gamma = self.alloc_uninit(channels)?;
        let mut dw_beta = self.alloc_uninit(dw_channel_count)?;
        let mut db_beta = self.alloc_uninit(channels)?;
        let mut dhf = self.alloc_uninit(hf_count)?;
        // Every output element is owned by a single thread; reductions run in
        // a fixed order over the batch or embedding axes.
        unsafe {
            self.foveation_film.film_projection_backward(
                &self.stream,
                launch::elementwise(target_count, config),
                &hf_params.buf,
                &embedding.buf,
                &dgamma_f.buf,
                &dbeta_f.buf,
                &d_embedding,
                &w_embed.buf,
                &mut dw_embed,
                &mut db_embed,
                &mut dw_gamma,
                &mut db_gamma,
                &mut dw_beta,
                &mut db_beta,
                &mut dhf,
                batch,
                channels,
                param_count,
                embed_dim,
            )
        }
        .map_err(launch_err)?;
        Ok(FoveationFilmGradients {
            input: dx,
            hf_params: OxideTensorF32::new(dhf, &[batch, param_count]),
            w_embed: OxideTensorF32::new(dw_embed, &[embed_dim, param_count]),
            b_embed: OxideTensorF32::new(db_embed, &[embed_dim]),
            w_gamma: OxideTensorF32::new(dw_gamma, &[channels, embed_dim]),
            b_gamma: OxideTensorF32::new(db_gamma, &[channels]),
            w_beta: OxideTensorF32::new(dw_beta, &[channels, embed_dim]),
            b_beta: OxideTensorF32::new(db_beta, &[channels]),
        })
    }

    pub(crate) fn foveation_film_affine_backward_backend_f32(
        &self,
        input: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        gamma: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let input = self.f32v(input)?;
        let input = &*input;
        let output_gradient = self.f32v(output_gradient)?;
        let output_gradient = &*output_gradient;
        let gamma = self.f32v(gamma)?;
        let gamma = &*gamma;
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_affine_backward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: spatial extent overflows".to_owned(),
            )
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: channel extent overflows".to_owned(),
            )
        })?;
        let expected = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: tensor extent overflows".to_owned(),
            )
        })?;
        if input.len() != expected
            || output_gradient.len() != expected
            || gamma.len() != channel_count
        {
            return Err(OxideError::Shape(
                "foveation_film_affine_backward_f32: tensor extents do not match NCHW / [B,C] layouts".to_owned(),
            ));
        }
        let mut dx = self.alloc_uninit(expected)?;
        let mut dgamma = self.alloc_uninit(channel_count)?;
        let mut dbeta = self.alloc_uninit(channel_count)?;
        // SAFETY: checked extents; dx is element-owned and parameter gradients
        // are reduced by one thread per sample/channel.
        unsafe {
            self.foveation_film.film_affine_dx(
                &self.stream,
                launch::elementwise(expected, config),
                &output_gradient.buf,
                &gamma.buf,
                &mut dx,
                channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.foveation_film.film_affine_param_grads(
                &self.stream,
                launch::elementwise(channel_count, config),
                &input.buf,
                &output_gradient.buf,
                &mut dgamma,
                &mut dbeta,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(dx, shape),
            OxideTensorF32::new(dgamma, &[batch, channels]),
            OxideTensorF32::new(dbeta, &[batch, channels]),
        ))
    }

}
