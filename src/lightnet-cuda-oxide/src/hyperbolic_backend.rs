//! Hyperbolic foveation backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn hyperbolic_foveation_predictor_backend_f32(
        &self,
        input: &OxideTensorF32,
        shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let source_f = self.f32v(input)?;
        let source = &*source_f;
        let op = "hyperbolic_foveation_predictor_f32";
        if shape.len() != 4
            || shape.contains(&0)
            || downsample == 0
            || hidden_count == 0
            || param_count < 4
        {
            return Err(shape_err(
                op,
                "invalid source shape or predictor dimensions".to_owned(),
            ));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let input_count = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "source size overflows".to_owned()))?;
        if source.len() != input_count {
            return Err(shape_err(
                op,
                format!("source has {} values, expected {input_count}", source.len()),
            ));
        }
        let pooled_height = (height / downsample).max(1);
        let pooled_width = (width / downsample).max(1);
        let input_dim = channels
            .checked_mul(pooled_height)
            .and_then(|n| n.checked_mul(pooled_width))
            .ok_or_else(|| shape_err(op, "pooled predictor size overflows".to_owned()))?;
        let expected_w1 = hidden_count
            .checked_mul(input_dim)
            .ok_or_else(|| shape_err(op, "first projection size overflows".to_owned()))?;
        let expected_w2 = param_count
            .checked_mul(hidden_count)
            .ok_or_else(|| shape_err(op, "second projection size overflows".to_owned()))?;
        let pooled_count = batch
            .checked_mul(input_dim)
            .ok_or_else(|| shape_err(op, "pooled batch size overflows".to_owned()))?;
        let output_count = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "predictor output size overflows".to_owned()))?;
        if w1.len() != expected_w1
            || b1.len() != hidden_count
            || w2.len() != expected_w2
            || b2.len() != param_count
        {
            return Err(shape_err(
                op,
                "predictor parameter lengths do not match dimensions".to_owned(),
            ));
        }
        let mut pooled = self.alloc_uninit(pooled_count)?;
        // SAFETY: output geometry is bounded by the validated positive source
        // dimensions and each output element owns its tile reduction.
        unsafe {
            self.resample.hyperbolic_predictor_pool(
                &self.stream,
                launch::elementwise(pooled_count, config),
                &source.buf,
                &mut pooled,
                channels,
                height,
                width,
                pooled_height,
                pooled_width,
                downsample,
            )
        }
        .map_err(launch_err)?;
        let pooled = OxideTensorF32::new(pooled, &[batch, input_dim]);
        let w1 = self.upload_f32(&[hidden_count, input_dim], w1)?;
        let b1 = self.upload_f32(&[hidden_count], b1)?;
        let preactivation = self.linear_f32(
            &pooled,
            &w1,
            Some(&b1),
            &[batch, hidden_count],
            batch,
            input_dim,
            hidden_count,
            config,
        )?;
        let hidden = self.activation_f32(&preactivation, F32ActivationKind::Relu, config)?;
        let w2 = self.upload_f32(&[param_count, hidden_count], w2)?;
        let b2 = self.upload_f32(&[param_count], b2)?;
        let raw = self.linear_f32(
            &hidden,
            &w2,
            Some(&b2),
            &[batch, param_count],
            batch,
            hidden_count,
            param_count,
            config,
        )?;
        let raw_f = self.f32v(&raw)?;
        let raw = &*raw_f;
        let mut constrained = self.alloc_uninit(output_count)?;
        // SAFETY: each output element owns one validated raw parameter value.
        unsafe {
            self.resample.hyperbolic_predictor_constrain(
                &self.stream,
                launch::elementwise(output_count, config),
                &raw.buf,
                &mut constrained,
                param_count,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(constrained, &[batch, param_count]))
    }

    pub(crate) fn hyperbolic_foveation_predictor_backward_backend_f32(
        &self,
        pooled: &OxideTensorF32,
        hidden_preactivation: &OxideTensorF32,
        hidden: &OxideTensorF32,
        raw_output: &OxideTensorF32,
        constrained_seed: &OxideTensorF32,
        w1: &OxideTensorF32,
        w2: &OxideTensorF32,
        input_shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let pooled_f = self.f32v(pooled)?;
        let pooled = &*pooled_f;
        let pre_f = self.f32v(hidden_preactivation)?;
        let pre = &*pre_f;
        let hidden_f = self.f32v(hidden)?;
        let hidden = &*hidden_f;
        let raw_f = self.f32v(raw_output)?;
        let raw = &*raw_f;
        let seed_f = self.f32v(constrained_seed)?;
        let seed = &*seed_f;
        let w1_f = self.f32v(w1)?;
        let w1 = &*w1_f;
        let w2_f = self.f32v(w2)?;
        let w2 = &*w2_f;
        let op = "hyperbolic_foveation_predictor_backward_f32";
        if input_shape.len() != 4
            || input_shape.contains(&0)
            || downsample == 0
            || hidden_count == 0
            || param_count < 4
        {
            return Err(shape_err(op, "expected positive NCHW shape, downsample, hidden count, and at least four parameters".to_owned()));
        }
        let (batch, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let pooled_height = (height / downsample).max(1);
        let pooled_width = (width / downsample).max(1);
        let input_dim = channels
            .checked_mul(pooled_height)
            .and_then(|n| n.checked_mul(pooled_width))
            .ok_or_else(|| shape_err(op, "pooled predictor size overflows".to_owned()))?;
        let input_count = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let hidden_count_batch = batch
            .checked_mul(hidden_count)
            .ok_or_else(|| shape_err(op, "hidden size overflows".to_owned()))?;
        let param_count_batch = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if pooled.len() != batch * input_dim
            || pre.len() != hidden_count_batch
            || hidden.len() != hidden_count_batch
            || raw.len() != param_count_batch
            || seed.len() != param_count_batch
            || w1.len() != hidden_count * input_dim
            || w2.len() != param_count * hidden_count
        {
            return Err(shape_err(
                op,
                "predictor cache/weight lengths do not match dimensions".to_owned(),
            ));
        }
        let source_len = input_count;
        let dw1_len = hidden_count
            .checked_mul(input_dim)
            .and_then(|n| n.checked_add(hidden_count))
            .ok_or_else(|| shape_err(op, "w1 gradient size overflows".to_owned()))?;
        let dw2_len = param_count
            .checked_mul(hidden_count)
            .and_then(|n| n.checked_add(param_count))
            .ok_or_else(|| shape_err(op, "w2 gradient size overflows".to_owned()))?;
        let mut source_grad = self.alloc_uninit(source_len)?;
        let mut w1_grad = self.alloc_uninit(dw1_len)?;
        let mut w2_grad = self.alloc_uninit(dw2_len)?;
        // SAFETY: all caches and weights have validated lengths; each output
        // scalar is written by exactly one thread in each kernel.
        unsafe {
            self.resample.hyperbolic_predictor_backward_source(
                &self.stream,
                launch::elementwise(source_len, config),
                &pre.buf,
                &raw.buf,
                &seed.buf,
                &w1.buf,
                &w2.buf,
                &mut source_grad,
                channels,
                height,
                width,
                downsample,
                pooled_height,
                pooled_width,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_predictor_backward_w1(
                &self.stream,
                launch::elementwise(dw1_len, config),
                &pooled.buf,
                &pre.buf,
                &raw.buf,
                &seed.buf,
                &w2.buf,
                &mut w1_grad,
                batch,
                input_dim,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_predictor_backward_w2(
                &self.stream,
                launch::elementwise(dw2_len, config),
                &hidden.buf,
                &raw.buf,
                &seed.buf,
                &mut w2_grad,
                batch,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(source_grad, input_shape),
            OxideTensorF32::new(w1_grad, &[dw1_len]),
            OxideTensorF32::new(w2_grad, &[dw2_len]),
        ))
    }

    pub(crate) fn hyperbolic_newton_grid_backend_f32(
        &self,
        params: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let op = "hyperbolic_newton_grid_f32";
        if batch == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch and spatial dimensions must be positive".to_owned(),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected parameter count 4 or 5, got {param_count}"),
            ));
        }
        let expected_params = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if params.len() != expected_params {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_params} parameters, got {}",
                    params.len()
                ),
            ));
        }
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        let mut grid = self.alloc_uninit(grid_len)?;
        // SAFETY: each output coordinate is owned by one thread and parameter
        // rows are validated to contain four/five values per batch.
        unsafe {
            self.resample.hyperbolic_newton_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &params.buf,
                &mut grid,
                height,
                width,
                param_count,
                alpha,
                p_exp,
                iterations,
                eta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(grid, &[batch, height, width, 2]))
    }

    pub(crate) fn hyperbolic_fixed_point_grid_backend_f32(
        &self,
        params: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let op = "hyperbolic_fixed_point_grid_f32";
        if batch == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch and spatial dimensions must be positive".to_owned(),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected parameter count 4 or 5, got {param_count}"),
            ));
        }
        let expected_params = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if params.len() != expected_params {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_params} parameters, got {}",
                    params.len()
                ),
            ));
        }
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        let mut grid = self.alloc_uninit(grid_len)?;
        // SAFETY: each thread owns one grid scalar, dimensions/parameter lengths
        // were validated, and every thread reads only its batch parameter row.
        unsafe {
            self.resample.hyperbolic_fixed_point_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &params.buf,
                &mut grid,
                height,
                width,
                param_count,
                alpha,
                p_exp,
                iterations,
                eta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(grid, &[batch, height, width, 2]))
    }

    pub(crate) fn hyperbolic_foveation_warp_backend_f32(
        &self,
        input: &OxideTensorF32,
        grid: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let grid_f = self.f32v(grid)?;
        let grid = &*grid_f;
        let op = "hyperbolic_foveation_warp_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if input.len() != input_len || grid.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "input/grid lengths are {}/{}, expected {input_len}/{grid_len}",
                    input.len(),
                    grid.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(input_len)?;
        // SAFETY: each output owns one element and only reads validated input
        // and the corresponding per-batch/pixel grid coordinates.
        unsafe {
            self.resample.hyperbolic_warp(
                &self.stream,
                launch::elementwise(input_len, config),
                &input.buf,
                &grid.buf,
                &mut output,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn hyperbolic_foveation_warp_backward_backend_f32(
        &self,
        input: &OxideTensorF32,
        grid: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let grid_f = self.f32v(grid)?;
        let grid = &*grid_f;
        let dy_f = self.f32v(output_gradient)?;
        let dy = &*dy_f;
        let op = "hyperbolic_foveation_warp_backward_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if input.len() != input_len || dy.len() != input_len || grid.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "input/gradient/grid lengths are {}/{}/{}, expected {input_len}/{input_len}/{grid_len}",
                    input.len(),
                    dy.len(),
                    grid.len()
                ),
            ));
        }
        let input_gradient = self.alloc(input_len)?;
        let mut grid_gradient = self.alloc_uninit(grid_len)?;
        // SAFETY: dimensions and lengths are validated; input-gradient atomic
        // scatters target a zeroed buffer, and grid-gradient scalars have unique owners.
        unsafe {
            self.resample.hyperbolic_warp_backward_input(
                &self.stream,
                launch::elementwise(input_len, config),
                &grid.buf,
                &dy.buf,
                input_gradient.cu_deviceptr() as *mut f32,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_warp_backward_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &input.buf,
                &grid.buf,
                &dy.buf,
                &mut grid_gradient,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_gradient, shape),
            OxideTensorF32::new(grid_gradient, &[batch, height, width, 2]),
        ))
    }

    pub(crate) fn hyperbolic_foveation_inverse_param_grad_backend_f32(
        &self,
        params: &OxideTensorF32,
        grid_gradient: &OxideTensorF32,
        shape: &[usize],
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        newton: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Option<OxideTensorF32>, OxideError> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let grid_f = self.f32v(grid_gradient)?;
        let grid_gradient = &*grid_f;
        let op = "hyperbolic_foveation_inverse_param_grad_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected 4 or 5 HF parameters, got {param_count}"),
            ));
        }
        let (batch, height, width) = (shape[0], shape[2], shape[3]);
        let params_len = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if params.len() != params_len || grid_gradient.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "parameter/grid lengths are {}/{}, expected {params_len}/{grid_len}",
                    params.len(),
                    grid_gradient.len()
                ),
            ));
        }
        let output_len = batch
            .checked_mul(5)
            .ok_or_else(|| shape_err(op, "output size overflows".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: all tensor lengths and dimensions are checked and each
        // thread owns one output batch/parameter lane.
        if newton {
            unsafe {
                self.resample.hyperbolic_newton_param_grad(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &params.buf,
                    &grid_gradient.buf,
                    &mut output,
                    batch,
                    height,
                    width,
                    param_count,
                    alpha,
                    p_exp,
                    iterations,
                    eta,
                )
            }
            .map_err(launch_err)?;
        } else {
            unsafe {
                self.resample.hyperbolic_fixed_point_param_grad(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &params.buf,
                    &grid_gradient.buf,
                    &mut output,
                    batch,
                    height,
                    width,
                    param_count,
                    alpha,
                    p_exp,
                    iterations,
                    eta,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(Some(OxideTensorF32::new(output, &[batch, 5])))
    }
}
