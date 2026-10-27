//! DSConv backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    /// Build one single-template DSConv morph branch's coordinate maps from
    /// tanh-transformed `[B,2*K,H,W]` offsets. `morph=2` composes two calls.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_coordinate_map_f32(
        &self,
        offsets: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_coordinate_map_f32";
        if batch == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 1
        {
            return Err(shape_err(
                op,
                "invalid batch/spatial/kernel/morph geometry".to_owned(),
            ));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        if offsets.shape() != [batch, offset_channels, height, width] {
            return Err(shape_err(
                op,
                "offset tensor shape does not match geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "coordinate map length overflows".to_owned()))?;
        let offsets = self.f32v(offsets)?;
        let mut y_map = self.alloc_uninit(output_len)?;
        let mut x_map = self.alloc_uninit(output_len)?;
        // SAFETY: operand shape and output lengths are validated above; each
        // thread owns one y/x coordinate pair and all offset taps stay in bounds.
        unsafe {
            self.dsconv.dsconv_coordinate_map(
                &self.stream,
                launch::elementwise(output_len, config),
                &offsets.buf,
                &mut y_map,
                &mut x_map,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                morph,
            )
        }
        .map_err(launch_err)?;
        let shape = [batch, height, width, kernel_size];
        Ok((
            OxideTensorF32::new(y_map, &shape),
            OxideTensorF32::new(x_map, &shape),
        ))
    }

    /// Border-clamped bilinear sampling for the DSConv's deformed tensor,
    /// returning `[B,C,H,W,K]` in the CPU reference's axis order.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_bilinear_sample_f32(
        &self,
        input: &OxideTensorF32,
        y_map: &OxideTensorF32,
        x_map: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_bilinear_sample_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || kernel_size == 0 {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        let coord_shape = [batch, height, width, kernel_size];
        if input.shape() != [batch, channels, height, width]
            || y_map.shape() != coord_shape
            || x_map.shape() != coord_shape
        {
            return Err(shape_err(
                op,
                "input and coordinate tensor shapes do not match geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "sampled tensor length overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let y_map = self.f32v(y_map)?;
        let x_map = self.f32v(x_map)?;
        let mut sampled = self.alloc_uninit(output_len)?;
        // SAFETY: exact source/map shapes and output extent were checked above.
        unsafe {
            self.dsconv.dsconv_bilinear_sample(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &y_map.buf,
                &x_map.buf,
                &mut sampled,
                batch,
                channels,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        let shape = [batch, channels, height, width, kernel_size];
        Ok(OxideTensorF32::new(sampled, &shape))
    }

    /// Adjoint of DSConv's border-clamped bilinear gather. Input gradients use
    /// float atomics; coordinate gradients are uniquely owned per sample.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_bilinear_backward_f32(
        &self,
        input: &OxideTensorF32,
        y_map: &OxideTensorF32,
        x_map: &OxideTensorF32,
        sampled_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_bilinear_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || kernel_size == 0 {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        let coord_shape = [batch, height, width, kernel_size];
        let sampled_shape = [batch, channels, height, width, kernel_size];
        if input.shape() != [batch, channels, height, width]
            || y_map.shape() != coord_shape
            || x_map.shape() != coord_shape
            || sampled_gradient.shape() != sampled_shape
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input gradient length overflows".to_owned()))?;
        let coord_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "coordinate gradient length overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let y_map = self.f32v(y_map)?;
        let x_map = self.f32v(x_map)?;
        let sampled_gradient = self.f32v(sampled_gradient)?;
        let mut input_gradient = self.alloc_zeroed(input_len)?;
        let mut y_gradient = self.alloc_uninit(coord_len)?;
        let mut x_gradient = self.alloc_uninit(coord_len)?;
        // SAFETY: all tensor shapes and gradient extents are checked; output
        // coordinate writers are unique and input collisions use float atomics.
        unsafe {
            self.dsconv.dsconv_bilinear_backward(
                &self.stream,
                launch::elementwise(coord_len, config),
                &input.buf,
                &y_map.buf,
                &x_map.buf,
                &sampled_gradient.buf,
                &mut input_gradient,
                &mut y_gradient,
                &mut x_gradient,
                batch,
                channels,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_gradient, &[batch, channels, height, width]),
            OxideTensorF32::new(y_gradient, &coord_shape),
            OxideTensorF32::new(x_gradient, &coord_shape),
        ))
    }

    /// Backpropagate coordinate-map adjoints through the center-anchored
    /// cumulative offsets and tanh, preserving the configured morph axes.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_offset_map_backward_f32(
        &self,
        offsets_tanh: &OxideTensorF32,
        y_coordinate_gradient: &OxideTensorF32,
        x_coordinate_gradient: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_offset_map_backward_f32";
        if batch == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 2
        {
            return Err(shape_err(op, "invalid DSConv offset geometry".to_owned()));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        let coord_shape = [batch, height, width, kernel_size];
        if offsets_tanh.shape() != [batch, offset_channels, height, width]
            || y_coordinate_gradient.shape() != coord_shape
            || x_coordinate_gradient.shape() != coord_shape
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let offset_len = batch
            .checked_mul(offset_channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "offset gradient length overflows".to_owned()))?;
        let offsets_tanh = self.f32v(offsets_tanh)?;
        let y_coordinate_gradient = self.f32v(y_coordinate_gradient)?;
        let x_coordinate_gradient = self.f32v(x_coordinate_gradient)?;
        let mut offset_gradient = self.alloc_uninit(offset_len)?;
        // SAFETY: input and output layouts are validated above; one thread owns
        // each raw-offset gradient and all cumulative taps are within K.
        unsafe {
            self.dsconv.dsconv_offset_map_backward(
                &self.stream,
                launch::elementwise(offset_len, config),
                &offsets_tanh.buf,
                &y_coordinate_gradient.buf,
                &x_coordinate_gradient.buf,
                &mut offset_gradient,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                morph,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            offset_gradient,
            &[batch, offset_channels, height, width],
        ))
    }

    /// Collapse DSConv's `[B,C,H,W,K]` sampled tensor with `[F,C,K]` weights.
    pub fn dsconv_collapse_f32(
        &self,
        sampled: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_collapse_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if sampled.shape() != [batch, channels, height, width, kernel_size]
            || weights.shape() != [filters, channels, kernel_size]
            || biases.shape() != [filters]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(filters)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
        let sampled = self.f32v(sampled)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: operand shapes and output length were checked above.
        unsafe {
            self.dsconv.dsconv_collapse(
                &self.stream,
                launch::elementwise(output_len, config),
                &sampled.buf,
                &weights.buf,
                &biases.buf,
                &mut output,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, filters, height, width],
        ))
    }

    /// Exact adjoints of the linear DSConv collapse for one output-channel
    /// partition (used separately for the two branches of morph=2).
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_collapse_backward_f32(
        &self,
        sampled: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        filter_start: usize,
        filter_count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_collapse_backward_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if filter_count == 0 || filter_start > filters || filter_count > filters - filter_start {
            return Err(shape_err(op, "invalid output-channel partition".to_owned()));
        }
        if sampled.shape() != [batch, channels, height, width, kernel_size]
            || weights.shape() != [filters, channels, kernel_size]
            || output_gradient.shape() != [batch, filters, height, width]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let sampled_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "sampled gradient length overflows".to_owned()))?;
        let weight_len = filters
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "weight gradient length overflows".to_owned()))?;
        let sampled = self.f32v(sampled)?;
        let weights = self.f32v(weights)?;
        let output_gradient = self.f32v(output_gradient)?;
        let mut sampled_gradient = self.alloc_uninit(sampled_len)?;
        let mut weight_gradient = self.alloc_uninit(weight_len)?;
        let mut bias_gradient = self.alloc_uninit(filters)?;
        // SAFETY: all operands were shape-checked; every output element has one writer.
        unsafe {
            self.dsconv.dsconv_collapse_backward_sampled(
                &self.stream,
                launch::elementwise(sampled_len, config),
                &weights.buf,
                &output_gradient.buf,
                &mut sampled_gradient,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: shape-checked parameter/output gradients; unique parameter writers.
        unsafe {
            self.dsconv.dsconv_collapse_backward_weights(
                &self.stream,
                launch::elementwise(weight_len, config),
                &sampled.buf,
                &output_gradient.buf,
                &mut weight_gradient,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: one thread owns each bias-gradient channel.
        unsafe {
            self.dsconv.dsconv_collapse_backward_biases(
                &self.stream,
                launch::elementwise(filters, config),
                &output_gradient.buf,
                &mut bias_gradient,
                batch,
                filters,
                height,
                width,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(
                sampled_gradient,
                &[batch, channels, height, width, kernel_size],
            ),
            OxideTensorF32::new(weight_gradient, &[filters, channels, kernel_size]),
            OxideTensorF32::new(bias_gradient, &[filters]),
        ))
    }

    /// Composed DSConv f32 forward for the bounded single-template contract.
    /// This returns the post-activation layer output but does not retain the
    /// forward intermediates needed by runtime backward.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_forward_native_f32(
        &self,
        input: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_forward_native_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 2
            || (morph == 2 && filters < 2)
        {
            return Err(shape_err(
                op,
                "invalid batch, layer, or morph geometry".to_owned(),
            ));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        if input.shape() != [batch, channels, height, width]
            || offset_weights.shape() != [offset_channels, channels, 3, 3]
            || offset_biases.shape() != [offset_channels]
            || weights.shape() != [filters, channels, kernel_size]
            || biases.shape() != [filters]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let offset_shape = [batch, offset_channels, height, width];
        let raw_offsets = self.conv2d_f32(
            input,
            offset_weights,
            &offset_shape,
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let biased_offsets = self.bias_add_f32(
            &raw_offsets,
            offset_biases,
            offset_channels,
            height * width,
            config,
        )?;
        let offsets_tanh = self.activation_f32(&biased_offsets, F32ActivationKind::Tanh, config)?;
        let (x_y_map, x_x_map) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            0,
            config,
        )?;
        let (y_y_map, y_x_map) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            1,
            config,
        )?;
        let sampled_x = self.dsconv_bilinear_sample_f32(
            input,
            &x_y_map,
            &x_x_map,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let sampled_y = self.dsconv_bilinear_sample_f32(
            input,
            &y_y_map,
            &y_x_map,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let output_len = batch
            .checked_mul(filters)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
        let sampled_x = self.f32v(&sampled_x)?;
        let sampled_y = self.f32v(&sampled_y)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut raw_output = self.alloc_uninit(output_len)?;
        // SAFETY: all input/parameter extents and output geometry were checked.
        unsafe {
            self.dsconv.dsconv_collapse_morph(
                &self.stream,
                launch::elementwise(output_len, config),
                &sampled_x.buf,
                &sampled_y.buf,
                &weights.buf,
                &biases.buf,
                &mut raw_output,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                morph,
            )
        }
        .map_err(launch_err)?;
        let raw_output = OxideTensorF32::new(raw_output, &[batch, filters, height, width]);
        self.activation_f32(&raw_output, activation, config)
    }

    /// Composed exact adjoint of the bounded single-template DSConv forward.
    /// Intermediate device values are recomputed; input-sampling scatter uses
    /// atomics, so only summation order is nondeterministic.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_backward_native_f32(
        &self,
        input: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "dsconv_backward_native_f32";
        if output_gradient.shape() != [batch, filters, height, width] {
            return Err(shape_err(
                op,
                "output-gradient shape does not match DSConv output".to_owned(),
            ));
        }
        let raw_output = self.dsconv_forward_native_f32(
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            batch,
            channels,
            filters,
            height,
            width,
            kernel_size,
            extend_scope,
            morph,
            F32ActivationKind::Linear,
            config,
        )?;
        let output_delta =
            self.activation_backward_f32(&raw_output, output_gradient, activation, config)?;

        // Rebuild predictor offsets and both branch sampling caches.
        let offset_channels = 2 * kernel_size;
        let offset_shape = [batch, offset_channels, height, width];
        let raw_offsets = self.conv2d_f32(
            input,
            offset_weights,
            &offset_shape,
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let biased_offsets = self.bias_add_f32(
            &raw_offsets,
            offset_biases,
            offset_channels,
            height * width,
            config,
        )?;
        let offsets_tanh = self.activation_f32(&biased_offsets, F32ActivationKind::Tanh, config)?;
        let (x_y, x_x) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            0,
            config,
        )?;
        let (y_y, y_x) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            1,
            config,
        )?;
        let sampled_x = self.dsconv_bilinear_sample_f32(
            input,
            &x_y,
            &x_x,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let sampled_y = self.dsconv_bilinear_sample_f32(
            input,
            &y_y,
            &y_x,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;

        let (x_start, x_count, y_start, y_count) = match morph {
            0 => (0, filters, 0, 0),
            1 => (0, 0, 0, filters),
            2 => (0, filters / 2, filters / 2, filters - filters / 2),
            _ => return Err(shape_err(op, "morph must be 0, 1, or 2".to_owned())),
        };
        let mut grad_weights: Option<OxideTensorF32> = None;
        let mut grad_biases: Option<OxideTensorF32> = None;
        let mut grad_input_sampling: Option<OxideTensorF32> = None;
        let mut grad_offset_map: Option<OxideTensorF32> = None;
        for (start, count, sampled, y_map, x_map, branch_morph) in [
            (x_start, x_count, &sampled_x, &x_y, &x_x, 0usize),
            (y_start, y_count, &sampled_y, &y_y, &y_x, 1usize),
        ] {
            if count == 0 {
                continue;
            }
            let (sampled_delta, weight_delta, bias_delta) = self.dsconv_collapse_backward_f32(
                sampled,
                weights,
                &output_delta,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                start,
                count,
                config,
            )?;
            grad_weights = Some(match grad_weights {
                Some(ref existing) => self.vector_add_f32(existing, &weight_delta, config)?,
                None => weight_delta,
            });
            grad_biases = Some(match grad_biases {
                Some(ref existing) => self.vector_add_f32(existing, &bias_delta, config)?,
                None => bias_delta,
            });
            let (input_delta, y_coordinate_delta, x_coordinate_delta) = self
                .dsconv_bilinear_backward_f32(
                    input,
                    y_map,
                    x_map,
                    &sampled_delta,
                    batch,
                    channels,
                    height,
                    width,
                    kernel_size,
                    config,
                )?;
            grad_input_sampling = Some(match grad_input_sampling {
                Some(ref existing) => self.vector_add_f32(existing, &input_delta, config)?,
                None => input_delta,
            });
            let offset_delta = self.dsconv_offset_map_backward_f32(
                &offsets_tanh,
                &y_coordinate_delta,
                &x_coordinate_delta,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                branch_morph,
                config,
            )?;
            grad_offset_map = Some(match grad_offset_map {
                Some(ref existing) => self.vector_add_f32(existing, &offset_delta, config)?,
                None => offset_delta,
            });
        }
        let grad_weights =
            grad_weights.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_biases =
            grad_biases.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_input_sampling = grad_input_sampling
            .ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_offset_map =
            grad_offset_map.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_offset_weights = self.conv2d_backward_weights_f32(
            input,
            &grad_offset_map,
            &[offset_channels, channels, 3, 3],
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let grad_offset_biases =
            self.bias_backward_f32(&grad_offset_map, offset_channels, height * width, config)?;
        let grad_input_predictor = self.conv2d_backward_input_f32(
            &grad_offset_map,
            offset_weights,
            &[batch, channels, height, width],
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let grad_input =
            self.vector_add_f32(&grad_input_sampling, &grad_input_predictor, config)?;
        Ok((
            grad_input,
            grad_weights,
            grad_biases,
            grad_offset_weights,
            grad_offset_biases,
        ))
    }

}
