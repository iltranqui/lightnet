//! GPU-runtime glue for `[wmhf]`.
//!
//! The native f32 arm composes the WMHF device primitives (channel split,
//! local-branch concatenation, Haar DWT/IWT, four-direction scan,
//! high-frequency energy and gate fusion) with the backend convolution,
//! batch-norm and route ops for the seven sub-convolutions (`pre`, the
//! 3x/5x/7x depthwise local branches, `local_mix`, `hf_gate`, `fuse`). The
//! forward keeps its intermediates in a [`DeviceCache`]; the backward is the
//! exact derivative of `forward_wmhf_with_cache` and keeps every activation
//! gradient on the device. Parameter gradients are downloaded once and staged
//! into the host train buffers, where the existing SGD/Adam arms update them
//! (host-staged updates); batch-statistics rolling means and variances are
//! applied to the host parameters after backward, as the CPU accumulation does.
//!
//! Activations without a device f32 kernel keep the host-reference forward
//! and force the CPU backward fallback ([`needs_host`]).
use crate::error::{Error, Result};
use crate::model::{Layer, TensorShape, WmhfSpec};
use crate::runtime::{
    BATCH_NORM_EPSILON, BATCH_NORM_ROLLING_KEEP_CPU, ConvolutionalParams,
    ConvolutionalTrainBuffers, GpuExecutableNetwork, GpuForwardState, LayerParams,
    LayerTrainBuffers, WmhfParams, WmhfRuntimeCache, backend_error, forward_wmhf_with_cache,
    tensor_backend_shape, validate_batchnorm_params,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

/// Whether `[wmhf]` keeps the host-reference path. The device arm covers every
/// parsed channel split, shortcut and BN setting, but only output activations
/// with a device f32 kernel. `requires_cpu_backward_fallback` uses the same
/// predicate, so the forward and the backward always choose the same path.
pub(crate) fn needs_host(spec: &WmhfSpec) -> bool {
    spec.activation.to_cuda_f32_kind().is_none()
}

/// One WMHF sub-convolution: stride 1, dilation 1 and `size / 2` padding, as
/// `wmhf_internal_convolution_spec`.
#[derive(Clone, Copy, Debug)]
struct SubConv {
    name: &'static str,
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    activation: F32ActivationKind,
}

impl SubConv {
    fn padding(&self) -> usize {
        self.size / 2
    }

    fn weight_shape(&self) -> [usize; 4] {
        [
            self.filters,
            self.input_channels / self.groups,
            self.size,
            self.size,
        ]
    }
}

/// The seven sub-convolutions in the CPU forward order.
struct Branches {
    pre: SubConv,
    local_dw3: SubConv,
    local_dw5: SubConv,
    local_dw7: SubConv,
    local_mix: SubConv,
    hf_gate: SubConv,
    fuse: SubConv,
}

/// Layer geometry shared by the forward, the backward and the device cache.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Geometry {
    batch: usize,
    input_channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    identity: usize,
    local: usize,
    global: usize,
    use_shortcut: bool,
}

impl Geometry {
    fn new(layer: &Layer, spec: &WmhfSpec, params: &WmhfParams) -> Result<Self> {
        let invalid = |what: String| {
            Error::InvalidConfig(format!("[wmhf] layer #{}: {what}", layer.index))
        };
        let input = layer.input_shape;
        if input.batch == 0 || input.channels == 0 || input.height == 0 || input.width == 0 {
            return Err(invalid("input dimensions must be positive".to_owned()));
        }
        if spec.local_channels == 0
            || spec.global_channels == 0
            || spec.identity_channels + spec.local_channels + spec.global_channels
                != spec.filters
        {
            return Err(invalid(format!(
                "invalid channel split {}+{}+{} for {} filters",
                spec.identity_channels, spec.local_channels, spec.global_channels, spec.filters
            )));
        }
        let expected_output = TensorShape {
            channels: spec.filters,
            ..input
        };
        if layer.output_shape != expected_output {
            return Err(invalid(format!(
                "output shape {} does not match {expected_output}",
                layer.output_shape
            )));
        }
        if params.scan_weights.len() != 4 * spec.global_channels {
            return Err(invalid(format!(
                "expected {} scan weights, got {}",
                4 * spec.global_channels,
                params.scan_weights.len()
            )));
        }
        Ok(Self {
            batch: input.batch,
            input_channels: input.channels,
            height: input.height,
            width: input.width,
            filters: spec.filters,
            identity: spec.identity_channels,
            local: spec.local_channels,
            global: spec.global_channels,
            // Same condition as the CPU forward; the spatial shape is preserved.
            use_shortcut: spec.shortcut && input.channels == spec.filters,
        })
    }

    fn plane(&self) -> usize {
        self.height * self.width
    }

    fn height2(&self) -> usize {
        self.height.div_ceil(2)
    }

    fn width2(&self) -> usize {
        self.width.div_ceil(2)
    }

    fn shape(&self, channels: usize) -> TensorShape {
        TensorShape {
            width: self.width,
            height: self.height,
            channels,
            batch: self.batch,
        }
    }

    fn branches(&self) -> Branches {
        let linear = F32ActivationKind::Linear;
        let conv = |name: &'static str,
                    input_channels: usize,
                    filters: usize,
                    size: usize,
                    groups: usize,
                    activation: F32ActivationKind| SubConv {
            name,
            input_channels,
            filters,
            size,
            groups,
            activation,
        };
        Branches {
            pre: conv("pre", self.input_channels, self.filters, 1, 1, linear),
            local_dw3: conv("local_dw3", self.local, self.local, 3, self.local, linear),
            local_dw5: conv("local_dw5", self.local, self.local, 5, self.local, linear),
            local_dw7: conv("local_dw7", self.local, self.local, 7, self.local, linear),
            local_mix: conv("local_mix", 3 * self.local, self.local, 1, 1, linear),
            hf_gate: conv(
                "hf_gate",
                self.global,
                self.filters,
                1,
                1,
                F32ActivationKind::Logistic,
            ),
            fuse: conv("fuse", self.filters, self.filters, 1, 1, linear),
        }
    }
}

/// Batch-norm operands of one sub-convolution's backward.
#[derive(Clone, Debug)]
enum BatchNormCache<T> {
    /// Batch statistics (training forward). `mean` and the biased `variance`
    /// also feed the rolling update after backward.
    Batch {
        scales: T,
        x_hat: T,
        mean: T,
        variance: T,
    },
    /// Rolling statistics (inference forward): the normalized input and the
    /// per-channel input slope `gamma / sqrt(rolling_variance + eps)`.
    Rolling { x_hat: T, slope: T },
}

/// Device intermediates of one sub-convolution.
#[derive(Clone, Debug)]
struct SubConvCache<T> {
    input: T,
    weights: T,
    batch_norm: Option<BatchNormCache<T>>,
    activation_input: T,
    output: T,
}

/// Device gradients of one sub-convolution.
struct SubConvGradients<T> {
    input: T,
    weights: T,
    biases: T,
    scales: Option<T>,
}

/// Device intermediates of one native `[wmhf]` forward: the operands of the
/// native backward. The layer pre-activation is in `gpu_layer_activation_inputs`.
#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: Geometry,
    training: bool,
    pre: SubConvCache<T>,
    local_dw3: SubConvCache<T>,
    local_dw5: SubConvCache<T>,
    local_dw7: SubConvCache<T>,
    local_mix: SubConvCache<T>,
    hf_gate: SubConvCache<T>,
    fuse: SubConvCache<T>,
    ll: T,
    lh: T,
    hl: T,
    hh: T,
    scan_weights: T,
}

/// Output of one `[wmhf]` forward: the activated tensor, the pre-activation
/// the backward needs, and whichever cache the chosen path produced.
pub(crate) struct WmhfGpuForward<T> {
    pub(crate) output: T,
    pub(crate) activation_input: T,
    pub(crate) device_cache: Option<DeviceCache<T>>,
    pub(crate) host_cache: Option<WmhfRuntimeCache>,
}

fn add_gradient(name: &str, dst: &mut [f32], src: Vec<f32>) -> Result<()> {
    if dst.len() != src.len() {
        return Err(Error::InvalidConfig(format!(
            "[wmhf] {name} gradient has {} values, expected {}",
            src.len(),
            dst.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += src;
    }
    Ok(())
}

/// `updates += gradient`, as `accumulate_convolutional_train_buffer`; the BN
/// mean and variance deltas have zero gradient.
fn stage_convolution(
    name: &str,
    train: &mut ConvolutionalTrainBuffers,
    weights: Vec<f32>,
    biases: Vec<f32>,
    scales: Option<Vec<f32>>,
) -> Result<()> {
    add_gradient(&format!("{name} weight"), &mut train.weight_updates, weights)?;
    add_gradient(&format!("{name} bias"), &mut train.bias_updates, biases)?;
    match (train.batch_norm.as_mut(), scales) {
        (Some(batch_norm), Some(scales)) => add_gradient(
            &format!("{name} batch-norm scale"),
            &mut batch_norm.scale_updates,
            scales,
        ),
        (None, None) => Ok(()),
        _ => Err(Error::InvalidConfig(format!(
            "[wmhf] {name} batch-norm train buffers do not match the forward"
        ))),
    }
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Full `[wmhf]` forward arm. `native` is the network-wide gate shared with
    /// the backward (no CPU backward fallback). With it and a device output
    /// activation the layer runs natively; otherwise it keeps the documented
    /// host-reference path and its host cache for the CPU backward fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_wmhf_gpu(
        &self,
        layer: &Layer,
        spec: &WmhfSpec,
        params: &WmhfParams,
        current_input: &B::Tensor,
        training: bool,
        native: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<WmhfGpuForward<B::Tensor>> {
        if native && !needs_host(spec) {
            self.forward_wmhf_device_gpu(layer, spec, params, current_input, training, config)
        } else {
            self.forward_wmhf_host_gpu(layer, spec, params, current_input, training)
        }
    }

    /// Host-reference path: download, run the CPU forward, upload the
    /// activated output and the pre-activation, keep the host cache.
    fn forward_wmhf_host_gpu(
        &self,
        layer: &Layer,
        spec: &WmhfSpec,
        params: &WmhfParams,
        current_input: &B::Tensor,
        training: bool,
    ) -> Result<WmhfGpuForward<B::Tensor>> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download [wmhf] input", err))?;
        let (output, activation_input, cache) =
            forward_wmhf_with_cache(spec, params, &host_input, layer.output_shape, training)?;
        let shape = tensor_backend_shape(layer.output_shape);
        let activation_input = self
            .backend
            .upload_f32(&shape, &activation_input.data)
            .map_err(|err| backend_error("upload [wmhf] activation cache", err))?;
        let output = self
            .backend
            .upload_f32(&shape, &output.data)
            .map_err(|err| backend_error("upload [wmhf] output", err))?;
        Ok(WmhfGpuForward {
            output,
            activation_input,
            device_cache: None,
            host_cache: Some(cache),
        })
    }

    /// Native forward, in the order of `forward_wmhf_with_cache`.
    fn forward_wmhf_device_gpu(
        &self,
        layer: &Layer,
        spec: &WmhfSpec,
        params: &WmhfParams,
        current_input: &B::Tensor,
        training: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<WmhfGpuForward<B::Tensor>> {
        let activation = spec.activation.to_cuda_f32_kind().ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[wmhf] unsupported GPU activation {:?}", spec.activation),
            }
        })?;
        let g = Geometry::new(layer, spec, params)?;
        let b = g.branches();
        let be = &self.backend;
        let err = |what: &str| {
            let what = format!("[wmhf] {what}");
            move |e: B::Error| backend_error(&what, e)
        };
        let (batch, height, width) = (g.batch, g.height, g.width);
        let projected_shape = g.shape(g.filters);
        let local_shape = g.shape(g.local);

        let pre = self.wmhf_sub_conv_forward(
            &b.pre,
            &params.pre,
            current_input,
            projected_shape,
            training,
            config,
        )?;
        let local_input = be
            .wmhf_split_channels_f32(
                &pre.output,
                batch,
                g.filters,
                height,
                width,
                g.identity,
                g.local,
                config,
            )
            .map_err(err("local channel split"))?;
        let global_input = be
            .wmhf_split_channels_f32(
                &pre.output,
                batch,
                g.filters,
                height,
                width,
                g.identity + g.local,
                g.global,
                config,
            )
            .map_err(err("global channel split"))?;

        let local_dw3 = self.wmhf_sub_conv_forward(
            &b.local_dw3,
            &params.local_dw3,
            &local_input,
            local_shape,
            training,
            config,
        )?;
        let local_dw5 = self.wmhf_sub_conv_forward(
            &b.local_dw5,
            &params.local_dw5,
            &local_input,
            local_shape,
            training,
            config,
        )?;
        let local_dw7 = self.wmhf_sub_conv_forward(
            &b.local_dw7,
            &params.local_dw7,
            &local_input,
            local_shape,
            training,
            config,
        )?;
        let local_cat = be
            .wmhf_concat_local_f32(
                &local_dw3.output,
                &local_dw5.output,
                &local_dw7.output,
                batch,
                g.local,
                height,
                width,
                config,
            )
            .map_err(err("local branch concatenation"))?;
        let local_mix = self.wmhf_sub_conv_forward(
            &b.local_mix,
            &params.local_mix,
            &local_cat,
            local_shape,
            training,
            config,
        )?;

        // Global branch: the scan replaces LL, the high bands pass through
        // the synthesis and also drive the high-frequency gate.
        let (ll, lh, hl, hh) = be
            .wmhf_haar_dwt_f32(&global_input, batch, g.global, height, width, config)
            .map_err(err("Haar DWT"))?;
        let scan_weights = be
            .upload_f32(&[4, g.global], &params.scan_weights)
            .map_err(err("scan weight upload"))?;
        let scanned = be
            .wmhf_scan4_forward_f32(
                &ll,
                &scan_weights,
                batch,
                g.global,
                g.height2(),
                g.width2(),
                config,
            )
            .map_err(err("scan"))?;
        let global_values = be
            .wmhf_haar_iwt_f32(&scanned, &lh, &hl, &hh, batch, g.global, height, width, config)
            .map_err(err("Haar IWT"))?;
        let energy = be
            .wmhf_high_frequency_forward_f32(&lh, &hl, &hh, batch, g.global, height, width, config)
            .map_err(err("high-frequency energy"))?;
        let hf_gate = self.wmhf_sub_conv_forward(
            &b.hf_gate,
            &params.hf_gate,
            &energy,
            projected_shape,
            training,
            config,
        )?;

        let fuse_input =
            self.wmhf_fuse_concat(&g, &pre.output, &local_mix.output, &global_values, config)?;
        let fuse = self.wmhf_sub_conv_forward(
            &b.fuse,
            &params.fuse,
            &fuse_input,
            projected_shape,
            training,
            config,
        )?;
        let activation_input = be
            .wmhf_apply_gate_f32(
                &fuse.output,
                &hf_gate.output,
                &pre.output,
                g.use_shortcut.then_some(current_input),
                spec.freq_scale,
                &tensor_backend_shape(layer.output_shape),
                config,
            )
            .map_err(err("gate fusion"))?;
        let output = if activation == F32ActivationKind::Linear {
            activation_input.clone()
        } else {
            be.activation_f32(&activation_input, activation, config)
                .map_err(err("activation"))?
        };
        Ok(WmhfGpuForward {
            output,
            activation_input,
            device_cache: Some(DeviceCache {
                geometry: g,
                training,
                pre,
                local_dw3,
                local_dw5,
                local_dw7,
                local_mix,
                hf_gate,
                fuse,
                ll,
                lh,
                hl,
                hh,
                scan_weights,
            }),
            host_cache: None,
        })
    }

    /// `[identity channels of projected | local | global]`, the CPU
    /// `concat_fuse`. The backward reuses it to insert the local and global
    /// input gradients next to the identity part of the fuse-input gradient:
    /// the three channel ranges are disjoint, so this is the sum of the
    /// `insert_channels` calls of the CPU backward.
    fn wmhf_fuse_concat(
        &self,
        g: &Geometry,
        projected: &B::Tensor,
        local: &B::Tensor,
        global: &B::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let mut inputs = Vec::with_capacity(3);
        let mut channels = Vec::with_capacity(3);
        let mut totals = Vec::with_capacity(3);
        if g.identity > 0 {
            inputs.push(projected);
            channels.push(g.identity);
            totals.push(g.filters);
        }
        inputs.push(local);
        channels.push(g.local);
        totals.push(g.local);
        inputs.push(global);
        channels.push(g.global);
        totals.push(g.global);
        let offsets = vec![0; inputs.len()];
        self.backend
            .route_concat_f32(
                &inputs,
                &offsets,
                &channels,
                &totals,
                &tensor_backend_shape(g.shape(g.filters)),
                g.plane(),
                config,
            )
            .map_err(|err| backend_error("[wmhf] fuse concatenation", err))
    }

    /// Convolution, BN (batch statistics when training, rolling statistics
    /// otherwise) or bias, then the sub-convolution activation, as
    /// `forward_wmhf_convolution`.
    fn wmhf_sub_conv_forward(
        &self,
        conv: &SubConv,
        params: &ConvolutionalParams,
        input: &B::Tensor,
        shape: TensorShape,
        training: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<SubConvCache<B::Tensor>> {
        let be = &self.backend;
        let err = |what: &str| {
            let what = format!("[wmhf] {} {what}", conv.name);
            move |e: B::Error| backend_error(&what, e)
        };
        let weight_shape = conv.weight_shape();
        let weight_count = weight_shape.iter().product::<usize>();
        if params.weights.len() != weight_count || params.biases.len() != conv.filters {
            return Err(Error::InvalidConfig(format!(
                "[wmhf] {} expects {weight_count} weights and {} biases, got {} and {}",
                conv.name,
                conv.filters,
                params.weights.len(),
                params.biases.len()
            )));
        }
        let filters = conv.filters;
        let out_shape = tensor_backend_shape(shape);
        let plane = shape.height * shape.width;
        let upload = |what: &str, values: &[f32], dims: &[usize]| {
            be.upload_f32(dims, values).map_err(err(what))
        };
        let weights = upload("weight upload", &params.weights, &weight_shape)?;
        let raw = be
            .conv2d_f32(
                input,
                &weights,
                &out_shape,
                conv.input_channels,
                shape.height,
                shape.width,
                filters,
                conv.size,
                1,
                1,
                1,
                conv.padding(),
                conv.groups,
                config,
            )
            .map_err(err("convolution"))?;
        let biases = upload("bias upload", &params.biases, &[filters])?;
        let (activation_input, batch_norm) = match &params.batch_norm {
            None => (
                be.bias_add_f32(&raw, &biases, filters, plane, config)
                    .map_err(err("bias"))?,
                None,
            ),
            Some(batch_norm) => {
                validate_batchnorm_params(batch_norm, filters)?;
                // The BN shift is the sub-convolution bias, as the CPU's
                // effective batch norm.
                let scales = upload("BN scale upload", &batch_norm.scales, &[filters])?;
                if training {
                    // The kernel's rolling EMA is discarded: the host update
                    // after backward applies the CPU rule to the batch stats.
                    let rolling_mean =
                        upload("BN rolling mean upload", &batch_norm.rolling_mean, &[filters])?;
                    let rolling_variance = upload(
                        "BN rolling variance upload",
                        &batch_norm.rolling_variance,
                        &[filters],
                    )?;
                    let result = be
                        .batchnorm_train_forward_f32(
                            &raw,
                            &scales,
                            &biases,
                            &rolling_mean,
                            &rolling_variance,
                            &out_shape,
                            BATCH_NORM_ROLLING_KEEP_CPU,
                            config,
                        )
                        .map_err(err("batch-norm training forward"))?;
                    (
                        result.output,
                        Some(BatchNormCache::Batch {
                            scales,
                            x_hat: result.x_hat,
                            mean: result.mean,
                            variance: result.variance,
                        }),
                    )
                } else {
                    let inv_std = batch_norm
                        .rolling_variance
                        .iter()
                        .map(|variance| 1.0 / (variance + BATCH_NORM_EPSILON).sqrt())
                        .collect::<Vec<_>>();
                    let shift = batch_norm
                        .rolling_mean
                        .iter()
                        .zip(&inv_std)
                        .map(|(mean, inv_std)| -mean * inv_std)
                        .collect::<Vec<_>>();
                    let slope = batch_norm
                        .scales
                        .iter()
                        .zip(&inv_std)
                        .map(|(scale, inv_std)| scale * inv_std)
                        .collect::<Vec<_>>();
                    let inv_std = upload("BN inverse std upload", &inv_std, &[filters])?;
                    let shift = upload("BN shift upload", &shift, &[filters])?;
                    let normalized = be
                        .channel_scale_f32(&raw, &inv_std, filters, plane, config)
                        .map_err(err("BN normalize"))?;
                    let x_hat = be
                        .bias_add_f32(&normalized, &shift, filters, plane, config)
                        .map_err(err("BN shift"))?;
                    let scaled = be
                        .channel_scale_f32(&x_hat, &scales, filters, plane, config)
                        .map_err(err("BN scale"))?;
                    let activation_input = be
                        .bias_add_f32(&scaled, &biases, filters, plane, config)
                        .map_err(err("BN bias"))?;
                    let slope = upload("BN slope upload", &slope, &[filters])?;
                    (
                        activation_input,
                        Some(BatchNormCache::Rolling { x_hat, slope }),
                    )
                }
            }
        };
        let output = if conv.activation == F32ActivationKind::Linear {
            activation_input.clone()
        } else {
            be.activation_f32(&activation_input, conv.activation, config)
                .map_err(err("activation"))?
        };
        Ok(SubConvCache {
            input: input.clone(),
            weights,
            batch_norm,
            activation_input,
            output,
        })
    }

    /// Exact adjoint of [`Self::wmhf_sub_conv_forward`], as
    /// `backward_wmhf_convolution`.
    fn wmhf_sub_conv_backward(
        &self,
        conv: &SubConv,
        cache: &SubConvCache<B::Tensor>,
        grad: &B::Tensor,
        shape: TensorShape,
        config: ElementwiseLaunchConfig,
    ) -> Result<SubConvGradients<B::Tensor>> {
        let be = &self.backend;
        let err = |what: &str| {
            let what = format!("[wmhf] {} backward {what}", conv.name);
            move |e: B::Error| backend_error(&what, e)
        };
        let filters = conv.filters;
        let out_shape = tensor_backend_shape(shape);
        let plane = shape.height * shape.width;
        let delta = if conv.activation == F32ActivationKind::Linear {
            grad.clone()
        } else {
            be.activation_backward_f32(&cache.activation_input, grad, conv.activation, config)
                .map_err(err("activation"))?
        };
        let (raw_grad, biases, scales) = match &cache.batch_norm {
            None => {
                let biases = be
                    .bias_backward_f32(&delta, filters, plane, config)
                    .map_err(err("bias"))?;
                (delta, biases, None)
            }
            Some(BatchNormCache::Batch {
                scales,
                x_hat,
                variance,
                ..
            }) => {
                let (raw_grad, scale_grad, bias_grad) = be
                    .batchnorm_train_backward_f32(&delta, x_hat, variance, scales, &out_shape, config)
                    .map_err(err("batch norm"))?;
                (raw_grad, bias_grad, Some(scale_grad))
            }
            Some(BatchNormCache::Rolling { x_hat, slope }) => {
                let biases = be
                    .bias_backward_f32(&delta, filters, plane, config)
                    .map_err(err("BN bias"))?;
                let weighted = be
                    .vector_mul_f32(&delta, x_hat, &out_shape, config)
                    .map_err(err("BN scale product"))?;
                let scale_grad = be
                    .bias_backward_f32(&weighted, filters, plane, config)
                    .map_err(err("BN scale"))?;
                let raw_grad = be
                    .channel_scale_f32(&delta, slope, filters, plane, config)
                    .map_err(err("BN input"))?;
                (raw_grad, biases, Some(scale_grad))
            }
        };
        let input_shape = [shape.batch, conv.input_channels, shape.height, shape.width];
        let input = be
            .conv2d_backward_input_f32(
                &raw_grad,
                &cache.weights,
                &input_shape,
                conv.input_channels,
                shape.height,
                shape.width,
                filters,
                conv.size,
                1,
                1,
                1,
                conv.padding(),
                conv.groups,
                config,
            )
            .map_err(err("input"))?;
        let weights = be
            .conv2d_backward_weights_f32(
                &cache.input,
                &raw_grad,
                &conv.weight_shape(),
                conv.input_channels,
                shape.height,
                shape.width,
                filters,
                conv.size,
                1,
                1,
                1,
                conv.padding(),
                conv.groups,
                config,
            )
            .map_err(err("weights"))?;
        Ok(SubConvGradients {
            input,
            weights,
            biases,
            scales,
        })
    }

    /// Native `[wmhf]` backward from the device cache of the forward, in the
    /// reverse order of the `[wmhf]` arm of the CPU backward: output
    /// activation, gate fusion, `fuse`, the fuse-input split, `hf_gate`, the
    /// local branch (`local_mix`, the 3x/5x/7x depthwise convolutions), the
    /// global branch (IWT adjoint plus the high-frequency adjoint, scan, DWT
    /// adjoint), `pre` and the shortcut. The input gradient stays on the
    /// device; parameter gradients are staged into the host train buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_wmhf_gpu(
        &self,
        layer: &Layer,
        spec: &WmhfSpec,
        params: &WmhfParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let activation = spec.activation.to_cuda_f32_kind().ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[wmhf] unsupported GPU activation {:?}", spec.activation),
            }
        })?;
        let g = Geometry::new(layer, spec, params)?;
        let caches = &forward.wmhf_device_caches;
        let cache = caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing [wmhf] device forward cache for layer #{}; run a native GPU forward first",
                    layer.index
                ))
            })?;
        if cache.geometry != g {
            return Err(Error::InvalidConfig(format!(
                "[wmhf] device cache of layer #{} was built for another geometry",
                layer.index
            )));
        }
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing activation cache for [wmhf] layer #{}",
                    layer.index
                ))
            })?;
        let b = g.branches();
        let be = &self.backend;
        let err = |what: &str| {
            let what = format!("[wmhf] backward {what}");
            move |e: B::Error| backend_error(&what, e)
        };
        let (batch, height, width) = (g.batch, g.height, g.width);
        let projected_shape = g.shape(g.filters);
        let local_shape = g.shape(g.local);

        let delta = if activation == F32ActivationKind::Linear {
            grad
        } else {
            be.activation_backward_f32(activation_input, &grad, activation, config)
                .map_err(err("activation"))?
        };
        let (d_fuse_output, d_gate_output, d_pre_gate, d_shortcut) = be
            .wmhf_apply_gate_backward_f32(
                &delta,
                &cache.hf_gate.output,
                &cache.pre.output,
                spec.freq_scale,
                g.use_shortcut,
                &tensor_backend_shape(layer.output_shape),
                config,
            )
            .map_err(err("gate fusion"))?;

        // `fuse.input` is the fuse-input gradient: identity | local | global.
        let fuse = self.wmhf_sub_conv_backward(
            &b.fuse,
            &cache.fuse,
            &d_fuse_output,
            projected_shape,
            config,
        )?;
        let d_local_mix_output = be
            .wmhf_split_channels_f32(
                &fuse.input,
                batch,
                g.filters,
                height,
                width,
                g.identity,
                g.local,
                config,
            )
            .map_err(err("local fuse-gradient split"))?;
        let d_global_values = be
            .wmhf_split_channels_f32(
                &fuse.input,
                batch,
                g.filters,
                height,
                width,
                g.identity + g.local,
                g.global,
                config,
            )
            .map_err(err("global fuse-gradient split"))?;
        // `hf_gate.input` is the high-frequency energy gradient.
        let hf_gate = self.wmhf_sub_conv_backward(
            &b.hf_gate,
            &cache.hf_gate,
            &d_gate_output,
            projected_shape,
            config,
        )?;

        let local_mix = self.wmhf_sub_conv_backward(
            &b.local_mix,
            &cache.local_mix,
            &d_local_mix_output,
            local_shape,
            config,
        )?;
        let split_local = |begin: usize, what: &str| {
            be.wmhf_split_channels_f32(
                &local_mix.input,
                batch,
                3 * g.local,
                height,
                width,
                begin,
                g.local,
                config,
            )
            .map_err(err(what))
        };
        let local_dw3 = self.wmhf_sub_conv_backward(
            &b.local_dw3,
            &cache.local_dw3,
            &split_local(0, "3x branch gradient split")?,
            local_shape,
            config,
        )?;
        let local_dw5 = self.wmhf_sub_conv_backward(
            &b.local_dw5,
            &cache.local_dw5,
            &split_local(g.local, "5x branch gradient split")?,
            local_shape,
            config,
        )?;
        let local_dw7 = self.wmhf_sub_conv_backward(
            &b.local_dw7,
            &cache.local_dw7,
            &split_local(2 * g.local, "7x branch gradient split")?,
            local_shape,
            config,
        )?;
        let d_local_input = be
            .vector_add_f32(&local_dw3.input, &local_dw5.input, config)
            .and_then(|sum| be.vector_add_f32(&sum, &local_dw7.input, config))
            .map_err(err("local branch gradient sum"))?;

        let (d_ll, d_lh_iwt, d_hl_iwt, d_hh_iwt) = be
            .wmhf_haar_iwt_backward_f32(&d_global_values, batch, g.global, height, width, config)
            .map_err(err("Haar IWT"))?;
        let (d_lh_energy, d_hl_energy, d_hh_energy) = be
            .wmhf_high_frequency_backward_f32(
                &hf_gate.input,
                &cache.lh,
                &cache.hl,
                &cache.hh,
                batch,
                g.global,
                height,
                width,
                config,
            )
            .map_err(err("high-frequency energy"))?;
        let d_lh = be
            .vector_add_f32(&d_lh_iwt, &d_lh_energy, config)
            .map_err(err("LH gradient sum"))?;
        let d_hl = be
            .vector_add_f32(&d_hl_iwt, &d_hl_energy, config)
            .map_err(err("HL gradient sum"))?;
        let d_hh = be
            .vector_add_f32(&d_hh_iwt, &d_hh_energy, config)
            .map_err(err("HH gradient sum"))?;
        let (d_ll, d_scan_weights) = be
            .wmhf_scan4_backward_f32(
                &cache.ll,
                &d_ll,
                &cache.scan_weights,
                batch,
                g.global,
                g.height2(),
                g.width2(),
                config,
            )
            .map_err(err("scan"))?;
        let d_global_input = be
            .wmhf_haar_dwt_backward_f32(&d_ll, &d_lh, &d_hl, &d_hh, batch, g.global, height, width, config)
            .map_err(err("Haar DWT"))?;

        let d_pre_branches =
            self.wmhf_fuse_concat(&g, &fuse.input, &d_local_input, &d_global_input, config)?;
        let d_pre = be
            .vector_add_f32(&d_pre_gate, &d_pre_branches, config)
            .map_err(err("pre gradient sum"))?;
        let pre = self.wmhf_sub_conv_backward(&b.pre, &cache.pre, &d_pre, projected_shape, config)?;

        // Parameter staging: one download per tensor at the optimizer boundary.
        let host = |what: String, tensor: &B::Tensor| {
            be.download_f32(tensor)
                .map_err(|e| backend_error(&format!("download [wmhf] {what} gradient"), e))
        };
        let mut staged = Vec::with_capacity(7);
        for (conv, grads) in [
            (&b.pre, &pre),
            (&b.local_dw3, &local_dw3),
            (&b.local_dw5, &local_dw5),
            (&b.local_dw7, &local_dw7),
            (&b.local_mix, &local_mix),
            (&b.hf_gate, &hf_gate),
            (&b.fuse, &fuse),
        ] {
            let scales = grads
                .scales
                .as_ref()
                .map(|scales| host(format!("{} BN scale", conv.name), scales))
                .transpose()?;
            staged.push((
                conv.name,
                host(format!("{} weight", conv.name), &grads.weights)?,
                host(format!("{} bias", conv.name), &grads.biases)?,
                scales,
            ));
        }
        let d_scan_weights = host("scan weight".to_owned(), &d_scan_weights)?;
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::Wmhf(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [wmhf] train buffers for layer #{}",
                    layer.index
                )));
            };
            let targets = [
                &mut train.pre,
                &mut train.local_dw3,
                &mut train.local_dw5,
                &mut train.local_dw7,
                &mut train.local_mix,
                &mut train.hf_gate,
                &mut train.fuse,
            ];
            for (target, (name, weights, biases, scales)) in targets.into_iter().zip(staged) {
                stage_convolution(name, target, weights, biases, scales)?;
            }
            add_gradient("scan weight", &mut train.scan_weight_updates, d_scan_weights)?;
        }

        let d_input = if g.use_shortcut {
            be.vector_add_f32(&d_shortcut, &pre.input, config)
                .map_err(err("shortcut gradient sum"))?
        } else {
            pre.input
        };
        self.accumulate_previous_gpu_gradient(layer.index, d_input, layer_output_gradients)
    }

    /// Applies the batch-statistics rolling update of each native `[wmhf]`
    /// training forward to the host BN parameters of `pre`, `local_mix` and
    /// `fuse`, with the CPU rule of the accumulation (`keep = 0.9`, biased
    /// batch variance). Called once after the native backward.
    pub(crate) fn update_wmhf_rolling_stats_gpu(&mut self,
        forward: &GpuForwardState<B::Tensor>) -> Result<()> {
        let caches = &forward.wmhf_device_caches;
        let update = 1.0 - BATCH_NORM_ROLLING_KEEP_CPU;
        for (index, cache) in caches.iter().enumerate() {
            let Some(cache) = cache.as_ref().filter(|cache| cache.training) else {
                continue;
            };
            let Some(LayerParams::Wmhf(params)) = self.params.get_mut(index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [wmhf] params for BN stats at layer #{index}"
                )));
            };
            for (name, conv, sub) in [
                ("pre", &mut params.pre, &cache.pre),
                ("local_mix", &mut params.local_mix, &cache.local_mix),
                ("fuse", &mut params.fuse, &cache.fuse),
            ] {
                let Some(BatchNormCache::Batch { mean, variance, .. }) = &sub.batch_norm else {
                    continue;
                };
                let Some(batch_norm) = conv.batch_norm.as_mut() else {
                    return Err(Error::InvalidConfig(format!(
                        "missing [wmhf] {name} BN parameters for layer #{index}"
                    )));
                };
                let mean = self
                    .backend
                    .download_f32(mean)
                    .map_err(|err| backend_error("download [wmhf] BN batch mean", err))?;
                let variance = self
                    .backend
                    .download_f32(variance)
                    .map_err(|err| backend_error("download [wmhf] BN batch variance", err))?;
                if mean.len() != batch_norm.rolling_mean.len()
                    || variance.len() != batch_norm.rolling_variance.len()
                {
                    return Err(Error::InvalidConfig(format!(
                        "[wmhf] {name} BN statistics size mismatch at layer #{index}"
                    )));
                }
                for (dst, src) in batch_norm.rolling_mean.iter_mut().zip(mean) {
                    *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + update * src;
                }
                for (dst, src) in batch_norm.rolling_variance.iter_mut().zip(variance) {
                    *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + update * src;
                }
            }
        }
        Ok(())
    }
}
