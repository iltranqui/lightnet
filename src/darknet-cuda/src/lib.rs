//! Backend-neutral CUDA tensor contracts used by the Rust runtime.
//!
//! This crate does not require a CUDA driver for CPU builds.  It defines the
//! operation traits and tensor metadata needed by the core executor; the
//! optional cuTile bridge supplies a concrete device implementation.

pub mod host_ops;
pub use host_ops::{AdamParams, NanInfMode, dropout_unit};

pub const CUTILE_RS_REFERENCE: &str = "https://github.com/NVlabs/cutile-rs";
pub const DEFAULT_TILE_1D: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ActivationKind {
    Linear,
    /// Darknet loggy: `2 * logistic(x) - 1`.
    Loggy,
    /// Hard tanh, clamped to `[-1, 1]` with zero gradient at the endpoints.
    Hardtan,
    /// Relie: leaky ReLU with Darknet's 0.01 negative slope.
    Relie,
    /// Leaky hard tanh with unit slope over [0, 1] and 0.001 tails.
    Lhtan,
    /// Piecewise linear saturating sigmoid with 0.01 tails.
    Plse,
    /// Stair-step function with alternating flat and unit-slope intervals.
    Stair,
    Logistic,
    Relu,
    Leaky,
    Mish,
    Swish,
    Tanh,
    /// `clamp(x, 0, 6)`; gradient 1 strictly inside (0, 6) (C++ `relu6_*`).
    Relu6,
    /// `x*(x>0) + 0.1x`; gradient `(x>0) + 0.1` (C++ `ramp_*`).
    Ramp,
    /// tanh-approximation GELU `0.5x(1+tanh(0.797885x+0.035677x^3))`; exact derivative (C++ `gelu_*`).
    Gelu,
    /// Exponential linear unit: `x` for nonnegative x, otherwise `exp(x)-1`.
    Elu,
    /// Scaled exponential linear unit using Darknet's fixed scale and alpha.
    Selu,
    /// Hard Mish: `x` above zero, quadratic transition on `(-2, 0]`, else zero.
    HardMish,
}

/// Normalization applied to raw weighted-shortcut coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ShortcutWeightNormalization {
    None,
    Relu,
    Softmax,
}

/// Channel-wise normalization used by the convolution-only `NORM_CHAN*` activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ChannelNormalization {
    Relu,
    Softmax,
    SoftmaxMaxval,
}

/// Outputs of [`F32TensorOpsBackend::batchnorm_train_forward_f32`].
#[derive(Clone, Debug)]
pub struct BatchNormTrainForward<T> {
    /// `x_hat * scale + bias`, shaped like the input.
    pub output: T,
    /// Normalised input, needed by the backward pass.
    pub x_hat: T,
    /// Per-channel batch mean.
    pub mean: T,
    /// Per-channel biased batch variance.
    pub variance: T,
    pub rolling_mean: T,
    pub rolling_variance: T,
}

/// One tensor of a multi-tensor SGD step (see
/// [`F32TensorOpsBackend::sanitized_sgd_update_multi_f32`]).
#[derive(Clone, Debug)]
pub struct SgdUpdateItem<'a, T> {
    pub weights: &'a T,
    pub updates: &'a T,
    /// A not yet accumulated gradient: the step uses `updates + grad`
    /// (exactly `saxpy_f32(1.0, grad, updates)` first).
    pub grad: Option<&'a T>,
    pub shape: &'a [usize],
    pub fix_weights: bool,
    /// Weight decay (0 for biases / BN scales).
    pub decay: f32,
}

/// One tensor of a multi-tensor Adam step (see
/// [`F32TensorOpsBackend::sanitized_adam_update_multi_f32`]).
#[derive(Clone, Debug)]
pub struct AdamUpdateItem<'a, T> {
    pub weights: &'a T,
    pub updates: &'a T,
    pub grad: Option<&'a T>,
    pub m: &'a T,
    pub v: &'a T,
    pub shape: &'a [usize],
    pub fix_weights: bool,
    pub params: AdamParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    BF16,
    F32,
    I32,
    U8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
    RowMajor,
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelLanguage {
    CudaC,
    Ptx,
    CutileDsl,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTensorDesc {
    pub name: String,
    pub shape: [usize; 4],
    pub scalar_type: ScalarType,
    pub layout: TensorLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarArg {
    I32(i32),
    U32(u32),
    F32(f32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionStream {
    pub id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelModule {
    pub name: String,
    pub language: KernelLanguage,
    pub source: String,
    pub entry_points: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelLaunch {
    pub entry_point: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
    pub tensors: Vec<DeviceTensorDesc>,
    pub scalars: Vec<ScalarArg>,
    pub stream: ExecutionStream,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledKernel {
    pub module_name: String,
    pub entry_point: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementwiseLaunchConfig {
    pub tile_len: usize,
}

impl Default for ElementwiseLaunchConfig {
    fn default() -> Self {
        Self {
            tile_len: DEFAULT_TILE_1D,
        }
    }
}

pub trait CudaBackend {
    type Error;

    fn backend_name(&self) -> &'static str;
    fn compile(&self, module: &KernelModule) -> Result<Vec<CompiledKernel>, Self::Error>;
    fn launch(&self, kernel: &CompiledKernel, launch: &KernelLaunch) -> Result<(), Self::Error>;
}

fn moe_router_gates_host(
    input: &[f32],
    router_weights: &[f32],
    router_biases: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    experts: usize,
    top_k: usize,
    temperature: f32,
) -> Vec<f32> {
    assert!(
        experts > 0 && spatial > 0,
        "MoE router requires experts and spatial elements"
    );
    assert_eq!(input.len(), batch * channels * spatial);
    assert_eq!(router_weights.len(), experts * channels);
    assert_eq!(router_biases.len(), experts);
    let inverse_spatial = 1.0 / spatial as f32;
    let temperature = temperature.max(1e-6);
    let mut gates = vec![0.0; batch * experts];
    let mut logits = vec![0.0; experts];
    for b in 0..batch {
        for e in 0..experts {
            let mut value = router_biases[e];
            for c in 0..channels {
                let base = (b * channels + c) * spatial;
                let mut sum = 0.0f32;
                for s in 0..spatial {
                    sum += input[base + s];
                }
                let pooled = sum * inverse_spatial;
                value += router_weights[e * channels + c] * pooled;
            }
            logits[e] = value / temperature;
        }

        let mut masked = logits.clone();
        if top_k > 0 && top_k < experts {
            let mut ranked = logits.clone();
            ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let kth = ranked[top_k - 1];
            for value in &mut masked {
                if *value < kth {
                    *value = -f32::MAX;
                }
            }
        }
        let maximum = masked.iter().copied().fold(-f32::MAX, f32::max);
        let mut sum = 0.0f32;
        for e in 0..experts {
            let value = if masked[e] <= -f32::MAX * 0.5 {
                0.0
            } else {
                (masked[e] - maximum).exp()
            };
            gates[b * experts + e] = value;
            sum += value;
        }
        if sum <= 0.0 || !sum.is_finite() {
            gates[b * experts..(b + 1) * experts].fill(1.0 / experts as f32);
        } else {
            for gate in &mut gates[b * experts..(b + 1) * experts] {
                *gate /= sum;
            }
        }
    }
    gates
}

// `KernelLaunch` describes generic launches, but a real runtime also needs typed
// device buffers. This trait is the first concrete execution seam for Rust-side
// tensor ops while larger Darknet kernels are still being ported.
//
// The operation signatures intentionally keep the complete Darknet kernel
// geometry at the call site. Grouping those values into opaque descriptors
// would make the backend API less explicit and would hide layout mistakes.
#[allow(clippy::too_many_arguments)]
pub trait F32TensorOpsBackend {
    type Error;
    type Tensor: Clone;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error>;
    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;
    /// True if any element of `x` is NaN or infinite.  The default downloads
    /// and scans on the host; backends may override with a device reduction.
    fn any_non_finite_f32(&self, x: &Self::Tensor) -> Result<bool, Self::Error> {
        Ok(self.download_f32(x)?.iter().any(|value| !value.is_finite()))
    }
    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// `activation(bias_add(x, bias))` (inference epilogue).  Backends may
    /// fuse it into one launch; the result must equal the composition below.
    fn bias_activation_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let biased = self.bias_add_f32(x, bias, channels, channel_stride, config)?;
        self.activation_f32(&biased, activation, config)
    }
    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Normalize NCHW activations across channels independently at each
    /// `(batch, spatial)` position. `spatial` is `height * width`.
    fn normalize_channels_f32(
        &self,
        input: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(input)?;
        let output = host_ops::normalize_channels(&input, batch, channels, spatial, normalization);
        assert_eq!(output_shape.iter().product::<usize>(), output.len());
        self.upload_f32(output_shape, &output)
    }
    /// Exact derivative of [`Self::normalize_channels_f32`], except when
    /// `cpp_grad` requests the legacy C++ CUDA diagonal shortcut.
    fn normalize_channels_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        output: &Self::Tensor,
        delta: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let activation_input = self.download_f32(activation_input)?;
        let output = self.download_f32(output)?;
        let delta = self.download_f32(delta)?;
        let gradient = host_ops::normalize_channels_backward(
            &activation_input,
            &output,
            &delta,
            batch,
            channels,
            spatial,
            normalization,
            cpp_grad,
        );
        assert_eq!(output_shape.iter().product::<usize>(), gradient.len());
        self.upload_f32(output_shape, &gradient)
    }
    /// EML convolution's two-branch combine:
    /// `scale * (exp(clamp(x, -c, c)) - ln(softplus(y) + eps)) + residual`.
    /// `residual` is optional; when supplied, its flattened extent must match
    /// both branch tensors. Backends should preserve `output_shape` on the result.
    fn eml_convolutional_f32(
        &self,
        x_branch: &Self::Tensor,
        self_branch: &Self::Tensor,
        residual: Option<&Self::Tensor>,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let x = self.download_f32(x_branch)?;
        let y = self.download_f32(self_branch)?;
        let residual = residual
            .map(|tensor| self.download_f32(tensor))
            .transpose()?;
        let output =
            host_ops::eml_convolutional_forward(&x, &y, residual.as_deref(), clamp, eps, scale);
        assert_eq!(output_shape.iter().product::<usize>(), output.len());
        self.upload_f32(output_shape, &output)
    }
    /// Exact branch adjoints of [`Self::eml_convolutional_f32`]. The identity
    /// residual gradient is the incoming `output_grad` and needs no separate
    /// kernel; the caller accumulates it on its residual source.
    fn eml_convolutional_backward_f32(
        &self,
        x_branch: &Self::Tensor,
        self_branch: &Self::Tensor,
        output_grad: &Self::Tensor,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let x = self.download_f32(x_branch)?;
        let y = self.download_f32(self_branch)?;
        let delta = self.download_f32(output_grad)?;
        let (x_grad, self_grad) =
            host_ops::eml_convolutional_backward(&x, &y, &delta, clamp, eps, scale);
        assert_eq!(output_shape.iter().product::<usize>(), x_grad.len());
        Ok((
            self.upload_f32(output_shape, &x_grad)?,
            self.upload_f32(output_shape, &self_grad)?,
        ))
    }
    /// LSTM gate derivatives for one sequence step. Returns forget/input/
    /// candidate/output gate gradients and the cell gradient carried to the
    /// preceding step. The default is a host reference; device backends should
    /// override it to keep BPTT gate math resident on the accelerator.
    #[allow(clippy::too_many_arguments)]
    fn lstm_gate_backward_f32(
        &self,
        hidden_gradient: &Self::Tensor,
        cell: &Self::Tensor,
        previous_cell: &Self::Tensor,
        forget: &Self::Tensor,
        input_gate: &Self::Tensor,
        candidate: &Self::Tensor,
        output_gate: &Self::Tensor,
        next_cell_gradient: &Self::Tensor,
        elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
        ),
        Self::Error,
    > {
        let _ = config;
        let hidden_gradient = self.download_f32(hidden_gradient)?;
        let cell = self.download_f32(cell)?;
        let previous_cell = self.download_f32(previous_cell)?;
        let forget = self.download_f32(forget)?;
        let input_gate = self.download_f32(input_gate)?;
        let candidate = self.download_f32(candidate)?;
        let output_gate = self.download_f32(output_gate)?;
        let next_cell_gradient = self.download_f32(next_cell_gradient)?;
        for values in [
            &hidden_gradient,
            &cell,
            &previous_cell,
            &forget,
            &input_gate,
            &candidate,
            &output_gate,
            &next_cell_gradient,
        ] {
            assert_eq!(values.len(), elements, "LSTM gate backward tensor size");
        }
        let mut forget_gradient = vec![0.0; elements];
        let mut input_gate_gradient = vec![0.0; elements];
        let mut candidate_gradient = vec![0.0; elements];
        let mut output_gate_gradient = vec![0.0; elements];
        let mut cell_gradient = vec![0.0; elements];
        for index in 0..elements {
            let tanh_cell = cell[index].tanh();
            output_gate_gradient[index] = hidden_gradient[index] * tanh_cell;
            cell_gradient[index] =
                hidden_gradient[index] * output_gate[index] * (1.0 - tanh_cell * tanh_cell)
                    + next_cell_gradient[index];
            forget_gradient[index] = cell_gradient[index] * previous_cell[index];
            input_gate_gradient[index] = cell_gradient[index] * candidate[index];
            candidate_gradient[index] = cell_gradient[index] * input_gate[index];
            forget_gradient[index] *= forget[index] * (1.0 - forget[index]);
            input_gate_gradient[index] *= input_gate[index] * (1.0 - input_gate[index]);
            candidate_gradient[index] *= 1.0 - candidate[index] * candidate[index];
            output_gate_gradient[index] *= output_gate[index] * (1.0 - output_gate[index]);
            cell_gradient[index] *= forget[index];
        }
        Ok((
            self.upload_f32(&[elements], &forget_gradient)?,
            self.upload_f32(&[elements], &input_gate_gradient)?,
            self.upload_f32(&[elements], &candidate_gradient)?,
            self.upload_f32(&[elements], &output_gate_gradient)?,
            self.upload_f32(&[elements], &cell_gradient)?,
        ))
    }
    /// Extract one contiguous recurrent timestep from a sequence-major tensor.
    /// The input is split into `steps` equal-sized chunks; shapes describe the
    /// contiguous storage layout.
    fn sequence_slice_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        step_index: usize,
        steps: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(steps > 0, "sequence slice requires at least one step");
        assert!(step_index < steps, "sequence step index out of range");
        let input_elements = input_shape.iter().product::<usize>();
        assert_eq!(
            input_elements % steps,
            0,
            "sequence input size must divide evenly into steps"
        );
        let step_elements = input_elements / steps;
        assert_eq!(
            output_shape.iter().product::<usize>(),
            step_elements,
            "sequence slice output size"
        );
        let values = self.download_f32(input)?;
        assert_eq!(values.len(), input_elements, "sequence input tensor size");
        let start = step_index * step_elements;
        self.upload_f32(output_shape, &values[start..start + step_elements])
    }
    /// Concatenate same-sized recurrent timestep outputs along the sequence
    /// axis. `output_shape` carries the caller's tensor layout; each input has
    /// `step_elements` contiguous values.
    fn sequence_concat_f32(
        &self,
        steps: &[&Self::Tensor],
        output_shape: &[usize],
        step_elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            !steps.is_empty(),
            "sequence concat requires at least one step"
        );
        assert_eq!(
            output_shape.iter().product::<usize>(),
            steps.len() * step_elements,
            "sequence concat output size"
        );
        let mut output = Vec::with_capacity(steps.len() * step_elements);
        for step in steps {
            let values = self.download_f32(step)?;
            assert_eq!(values.len(), step_elements, "sequence step tensor size");
            output.extend_from_slice(&values);
        }
        self.upload_f32(output_shape, &output)
    }
    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Sum equally-shaped inputs with per-feature/per-channel weights. `weights`
    /// is `[input][channels]`; each input is scaled with `channel_stride` values
    /// per weight before the results are added. Caller validates dimensions and
    /// supplies already-normalized weights.
    fn shortcut_weighted_sum_f32(
        &self,
        inputs: &[&Self::Tensor],
        weights: &[f32],
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut accumulated = None;
        for (index, input) in inputs.iter().enumerate() {
            let scale = self.upload_f32(
                &[channels],
                &weights[index * channels..(index + 1) * channels],
            )?;
            let weighted =
                self.channel_scale_f32(input, &scale, channels, channel_stride, config)?;
            accumulated = Some(match accumulated {
                Some(previous) => self.shortcut_add_f32(&previous, &weighted, config)?,
                None => weighted,
            });
        }
        Ok(accumulated.expect("weighted shortcut requires at least the layer input"))
    }
    /// Reduce `delta * source` into the weighted-shortcut coefficient slots.
    /// The source can be shorter than the output per sample; elements outside
    /// that prefix contribute zero, matching shortcut crop/zero-pad semantics.
    /// `layer_step` is 1 for per-feature weights or channels for per-channel.
    /// Backends should return a `[layer_step]` tensor and override this method
    /// to avoid downloading the full activation tensors.
    fn shortcut_weight_gradient_f32(
        &self,
        delta: &Self::Tensor,
        source: &Self::Tensor,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(batch > 0 && layer_step > 0 && per_sample > 0);
        assert_eq!(per_sample % layer_step, 0);
        let delta = self.download_f32(delta)?;
        let source = self.download_f32(source)?;
        assert_eq!(delta.len(), batch * per_sample);
        assert_eq!(source.len(), batch * source_per_sample);
        let span = per_sample / layer_step;
        let mut gradient = vec![0.0f32; layer_step];
        for b in 0..batch {
            for e in 0..per_sample.min(source_per_sample) {
                gradient[e / span] += delta[b * per_sample + e] * source[b * source_per_sample + e];
            }
        }
        self.upload_f32(&[layer_step], &gradient)
    }
    /// Add one source's `delta * source` reduction into its `[input][slot]`
    /// row. This allows device backends to compose all source reductions
    /// without downloading per-source vectors. The source prefix behavior is
    /// identical to [`Self::shortcut_weight_gradient_f32`].
    fn shortcut_weight_gradient_accumulate_f32(
        &self,
        accumulated: &Self::Tensor,
        delta: &Self::Tensor,
        source: &Self::Tensor,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(batch > 0 && per_sample > 0 && layer_step > 0);
        assert_eq!(per_sample % layer_step, 0);
        assert!(source_index < num_inputs);
        let mut products = self.download_f32(accumulated)?;
        let delta = self.download_f32(delta)?;
        let source = self.download_f32(source)?;
        assert_eq!(products.len(), num_inputs * layer_step);
        assert_eq!(delta.len(), batch * per_sample);
        assert_eq!(source.len(), batch * source_per_sample);
        let span = per_sample / layer_step;
        let start = source_index * layer_step;
        for slot in 0..layer_step {
            let begin = slot * span;
            let end = (begin + span).min(per_sample).min(source_per_sample);
            let mut sum = 0.0f32;
            for b in 0..batch {
                for e in begin..end {
                    sum += delta[b * per_sample + e] * source[b * source_per_sample + e];
                }
            }
            products[start + slot] += sum;
        }
        self.upload_f32(&[num_inputs, layer_step], &products)
    }
    /// Apply the exact Rust weighted-shortcut normalization derivative, or
    /// the legacy raw-`z` gradient when `cpp_grad` is true.
    fn shortcut_weight_normalized_gradient_f32(
        &self,
        raw_weights: &Self::Tensor,
        products: &Self::Tensor,
        num_inputs: usize,
        layer_step: usize,
        normalization: F32ShortcutWeightNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(num_inputs > 0 && layer_step > 0);
        let raw_weights = self.download_f32(raw_weights)?;
        let products = self.download_f32(products)?;
        assert_eq!(raw_weights.len(), num_inputs * layer_step);
        assert_eq!(products.len(), raw_weights.len());
        let gradient = host_ops::shortcut_weight_normalized_gradient(
            &raw_weights,
            &products,
            layer_step,
            normalization,
            cpp_grad,
        );
        self.upload_f32(&[num_inputs, layer_step], &gradient)
    }
    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;
    fn conv2d_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn conv2d_backward_input_f32(
        &self,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        input_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn conv2d_backward_weights_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        weight_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn maxpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn maxpool_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn local_avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn local_avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Compute `[moe_convolutional]`'s per-sample router gates. The Oxide backend
    /// keeps pooling, linear projection, top-k masking, and softmax on device;
    /// other backends use this exact host fallback.
    fn moe_router_gates_f32(
        &self,
        input: &Self::Tensor,
        router_weights: &Self::Tensor,
        router_biases: &Self::Tensor,
        batch: usize,
        channels: usize,
        spatial: usize,
        experts: usize,
        top_k: usize,
        temperature: f32,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let gates = moe_router_gates_host(
            &self.download_f32(input)?,
            &self.download_f32(router_weights)?,
            &self.download_f32(router_biases)?,
            batch,
            channels,
            spatial,
            experts,
            top_k,
            temperature,
        );
        self.upload_f32(&[batch, experts], &gates)
    }
    /// Global-average pool an NCHW input to the router descriptor `[batch, channels]`.
    fn moe_router_pool_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        spatial: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        assert!(
            spatial > 0,
            "MoE router pooling requires nonzero spatial size"
        );
        let input = self.download_f32(input)?;
        assert_eq!(input.len(), batch * channels * spatial);
        let mut pooled = vec![0.0; batch * channels];
        for b in 0..batch {
            for c in 0..channels {
                let start = (b * channels + c) * spatial;
                pooled[b * channels + c] =
                    input[start..start + spatial].iter().sum::<f32>() * (1.0 / spatial as f32);
            }
        }
        self.upload_f32(&[batch, channels], &pooled)
    }
    /// Backward of global-average pooling: broadcast each descriptor gradient
    /// over the spatial plane and divide by its element count.
    fn moe_router_pool_backward_f32(
        &self,
        pooled_gradient: &Self::Tensor,
        input_shape: &[usize],
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        assert!(input_shape.len() >= 3, "MoE router input must be NCHW");
        let batch = input_shape[0];
        let channels = input_shape[1];
        let spatial: usize = input_shape[2..].iter().product();
        assert!(
            spatial > 0,
            "MoE router pooling requires nonzero spatial size"
        );
        let pooled_gradient = self.download_f32(pooled_gradient)?;
        assert_eq!(pooled_gradient.len(), batch * channels);
        let inverse_spatial = 1.0 / spatial as f32;
        let mut input_gradient = vec![0.0; batch * channels * spatial];
        for i in 0..input_gradient.len() {
            input_gradient[i] = pooled_gradient[i / spatial] * inverse_spatial;
        }
        self.upload_f32(input_shape, &input_gradient)
    }
    /// Per-sample dot products used for `dL/dgate[e]`: sum of output gradient
    /// times one expert's output over each sample's contiguous output slice.
    fn moe_gate_dot_f32(
        &self,
        output_gradient: &Self::Tensor,
        expert_output: &Self::Tensor,
        batch: usize,
        per_image: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_gradient = self.download_f32(output_gradient)?;
        let expert_output = self.download_f32(expert_output)?;
        assert_eq!(output_gradient.len(), batch * per_image);
        assert_eq!(expert_output.len(), batch * per_image);
        let mut dots = vec![0.0; batch];
        for b in 0..batch {
            let start = b * per_image;
            dots[b] = output_gradient[start..start + per_image]
                .iter()
                .zip(&expert_output[start..start + per_image])
                .map(|(grad, output)| grad * output)
                .sum();
        }
        self.upload_f32(&[batch], &dots)
    }
    /// Extract one expert's gate for every batch item from `[batch, experts]`.
    fn moe_gate_column_f32(
        &self,
        gates: &Self::Tensor,
        batch: usize,
        experts: usize,
        expert: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        assert!(batch > 0 && experts > 0 && expert < experts);
        let gates = self.download_f32(gates)?;
        assert_eq!(gates.len(), batch * experts);
        let column = (0..batch)
            .map(|b| gates[b * experts + expert])
            .collect::<Vec<_>>();
        self.upload_f32(&[batch], &column)
    }
    /// Compute MoE `dlogit` from expert outputs and load-balancing loss. Oxide
    /// keeps gate-dot reductions and the softmax Jacobian on-device; this default
    /// method provides the host-reference implementation for other backends.
    fn moe_router_backward_f32(
        &self,
        gates: &Self::Tensor,
        output_gradient: &Self::Tensor,
        expert_outputs: &[Self::Tensor],
        batch: usize,
        experts: usize,
        per_image: usize,
        load_balance: f32,
        temperature: f32,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        assert!(batch > 0 && experts > 0 && per_image > 0);
        let gates = self.download_f32(gates)?;
        let output_gradient = self.download_f32(output_gradient)?;
        assert_eq!(gates.len(), batch * experts);
        assert_eq!(output_gradient.len(), batch * per_image);
        assert_eq!(expert_outputs.len(), experts);

        let mut raw = vec![0.0f32; batch * experts];
        for (e, expert_output) in expert_outputs.iter().enumerate() {
            let expert_output = self.download_f32(expert_output)?;
            assert_eq!(expert_output.len(), batch * per_image);
            for b in 0..batch {
                let base = b * per_image;
                raw[b * experts + e] = output_gradient[base..base + per_image]
                    .iter()
                    .zip(&expert_output[base..base + per_image])
                    .map(|(grad, output)| grad * output)
                    .sum();
            }
        }
        for e in 0..experts {
            let mean_gate = (0..batch).map(|b| gates[b * experts + e]).sum::<f32>() / batch as f32;
            let balance_gradient = if load_balance <= 0.0 {
                0.0
            } else {
                load_balance * 2.0 * experts as f32 * mean_gate / batch as f32
            };
            for b in 0..batch {
                raw[b * experts + e] += balance_gradient;
            }
        }
        let temperature = if temperature > 1e-6 {
            temperature
        } else {
            1e-6
        };
        let mut dlogit = vec![0.0f32; batch * experts];
        for b in 0..batch {
            let base = b * experts;
            let weighted = (0..experts)
                .map(|e| gates[base + e] * raw[base + e])
                .sum::<f32>();
            for e in 0..experts {
                dlogit[base + e] = gates[base + e] * (raw[base + e] - weighted) / temperature;
            }
        }
        self.upload_f32(&[batch, experts], &dlogit)
    }
    /// Bilinearly sample an NCHW tensor using one source coordinate pair per
    /// batch/output pixel. `input_scales` and `input_offsets` contain one
    /// affine level adjustment per batch and are applied after interpolation.
    fn crop_bilinear_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        input_offsets: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            input_offsets,
            config,
        );
        panic!("crop_bilinear_f32 is not implemented by this CUDA backend")
    }
    /// Scatter the bilinear crop gradient back into the NCHW input. Out of
    /// bounds taps are ignored, matching Darknet's zero-padded sampler.
    fn crop_bilinear_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            config,
        );
        panic!("crop_bilinear_backward_f32 is not implemented by this CUDA backend")
    }
    fn upsample_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn upsample_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Elementwise product with an explicit output shape. The default is a
    /// host fallback; device backends should override it for native execution.
    fn vector_mul_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let x = self.download_f32(x)?;
        let y = self.download_f32(y)?;
        assert_eq!(x.len(), y.len(), "vector_mul_f32: input lengths must match");
        assert_eq!(
            shape.iter().product::<usize>(),
            x.len(),
            "vector_mul_f32: output shape must match the input length"
        );
        let product = x
            .iter()
            .zip(y)
            .map(|(lhs, rhs)| lhs * rhs)
            .collect::<Vec<_>>();
        self.upload_f32(shape, &product)
    }
    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error>;

    /// Darknet shortcut resize for sources whose shape differs from the layer
    /// output: per batch sample the flat `x.len() / batch` elements are
    /// cropped or zero-padded to `output_shape`'s per-sample length (the C++
    /// `shortcut_multilayer_*` adds `src_i < min(src_outputs, outputs)` flat).
    /// Forward: `shortcut_add_f32(input, shortcut_fit_f32(source, ..))`;
    /// backward: the source gradient is `shortcut_fit_f32(layer_grad, ..)`.
    fn shortcut_fit_f32(
        &self,
        x: &Self::Tensor,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let out_len: usize = output_shape.iter().product();
        let fitted = host_ops::shortcut_fit(&host, batch, out_len / batch.max(1));
        self.upload_f32(output_shape, &fitted)
    }
    /// The batched-GEMM wrapper every transformer projection needs:
    /// `out[r, n] = sum_k x[r, k] * w[n, k] + b[n]` over a flat
    /// `rows x in_features` activation, with the **torch `nn.Linear` weight
    /// layout** (`w` is `[out_features, in_features]`, row-major).
    ///
    /// `rows` is folded into the GEMM's `m` dimension, so one call covers every
    /// token of a sequence.  `bias` may be `None` for the bias-free
    /// projections.  `shape` is the output tensor's shape, normally
    /// `[rows, out_features]`.
    ///
    /// Backends override this to route into their own GEMM; the default is the
    /// host reference, so `FakeBackend` and any other backend work unchanged.
    fn linear_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        in_features: usize,
        out_features: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            rows > 0 && in_features > 0 && out_features > 0,
            "linear_f32: rows, in_features and out_features must be > 0"
        );
        let xv = self.download_f32(x)?;
        assert_eq!(
            rows * in_features,
            xv.len(),
            "linear_f32: activation must be rows x in_features"
        );
        let w = self.download_f32(weights)?;
        let b = match bias {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(
            shape,
            &host_ops::linear(&xv, rows, in_features, out_features, &w, b.as_deref()),
        )
    }

    /// Fused residual + bias epilogue for a transformer block: over a flat
    /// `rows x cols` row-major slab, `out[i] = residual[i] + y[i] + bias[i % cols]`
    /// with `bias` per **column** (the last dimension, so the row - the token -
    /// does not shift the offset).  `bias` may be `None`.
    ///
    /// `shape` is the output tensor's shape.  One pass instead of
    /// `bias_add` + `vector_add`, which would materialise an extra full-size
    /// temporary per residual connection.
    fn residual_bias_f32(
        &self,
        residual: &Self::Tensor,
        y: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            rows > 0 && cols > 0,
            "residual_bias_f32: rows and cols must be > 0"
        );
        let r = self.download_f32(residual)?;
        let yv = self.download_f32(y)?;
        let b = match bias {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(
            shape,
            &host_ops::residual_bias(&r, &yv, b.as_deref(), rows, cols),
        )
    }
    /// Row-wise LayerNorm over a flat `rows x cols` buffer:
    /// `y = gamma * (x - mean) * inv_std + beta`, population variance.
    ///
    /// `gamma` and `beta` may be `None` for the bias-free variant that
    /// ModernBERT's `attn_norm` / `mlp_norm` and every `[laya_*]` LayerNorm use.
    /// `shape` is the output tensor's shape.  Shared building block for the
    /// transformer family (see `tasks/rust/layers.md` Phase 3A prerequisite).
    fn layernorm_forward_f32(
        &self,
        x: &Self::Tensor,
        gamma: Option<&Self::Tensor>,
        beta: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let g = match gamma {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        let b = match beta {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(
            shape,
            &host_ops::layernorm(&host, rows, cols, g.as_deref(), b.as_deref(), eps),
        )
    }
    /// Row-wise LayerNorm backward: the input gradient only, with the reduction
    /// statistics recomputed from `x` (`dx = g - mean(g) - xhat * mean(g*xhat)`).
    ///
    /// `gamma` may be `None`.  Parameter gradients are not produced here: every
    /// LayerNorm in the ported families is either frozen or trained by a path
    /// that folds them in separately.
    fn layernorm_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        x: &Self::Tensor,
        gamma: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let dy = self.download_f32(output_grad)?;
        let host = self.download_f32(x)?;
        let g = match gamma {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(
            shape,
            &host_ops::layernorm_backward(&dy, &host, rows, cols, g.as_deref(), eps),
        )
    }
    /// Online-softmax attention for fused QKV `[batch, tokens, 3*channels]`.
    /// Returns context `[batch, tokens, channels]` and natural-log LSE
    /// `[batch, heads, tokens]`; the reference implementation uses linear
    /// workspace, not a sequence-length-squared score/probability matrix.
    fn flash_attention_forward_f32(
        &self,
        qkv: &Self::Tensor,
        output_shape: &[usize],
        lse_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        assert!(
            batch > 0 && tokens > 0 && channels > 0 && heads > 0 && channels % heads == 0,
            "flash_attention_forward_f32: invalid dimensions"
        );
        let rows = batch * tokens;
        let qkv_host = self.download_f32(qkv)?;
        assert_eq!(
            qkv_host.len(),
            rows * 3 * channels,
            "flash attention QKV shape"
        );
        assert_eq!(
            output_shape,
            [batch, tokens, channels],
            "flash attention output shape"
        );
        assert_eq!(
            lse_shape,
            [batch, heads, tokens],
            "flash attention LSE shape"
        );
        let (output, lse) =
            host_ops::flash_attention_forward(&qkv_host, batch, tokens, channels, heads);
        Ok((
            self.upload_f32(output_shape, &output)?,
            self.upload_f32(lse_shape, &lse)?,
        ))
    }
    /// Recompute-based derivative of standard online attention. Inputs are
    /// fused QKV `[B,T,3C]`, context/output gradient `[B,T,C]`, and LSE
    /// `[B,H,T]`; output is fused dQKV `[B,T,3C]`.
    fn flash_attention_backward_f32(
        &self,
        qkv: &Self::Tensor,
        context: &Self::Tensor,
        output_grad: &Self::Tensor,
        lse: &Self::Tensor,
        qkv_grad_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        assert!(
            batch > 0 && tokens > 0 && channels > 0 && heads > 0 && channels % heads == 0,
            "flash_attention_backward_f32: invalid dimensions"
        );
        let rows = batch * tokens;
        let elements = rows * channels;
        assert_eq!(
            qkv_grad_shape,
            [batch, tokens, 3 * channels],
            "flash attention dQKV shape"
        );
        let qkv_host = self.download_f32(qkv)?;
        let context_host = self.download_f32(context)?;
        let grad_host = self.download_f32(output_grad)?;
        let lse_host = self.download_f32(lse)?;
        assert_eq!(qkv_host.len(), 3 * elements, "flash attention QKV length");
        assert_eq!(
            context_host.len(),
            elements,
            "flash attention context length"
        );
        assert_eq!(grad_host.len(), elements, "flash attention gradient length");
        assert_eq!(
            lse_host.len(),
            batch * heads * tokens,
            "flash attention LSE length"
        );
        let dqkv = host_ops::flash_attention_backward(
            &qkv_host,
            &context_host,
            &grad_host,
            &lse_host,
            batch,
            tokens,
            channels,
            heads,
        );
        self.upload_f32(qkv_grad_shape, &dqkv)
    }
    /// Softmax over every contiguous `group_size` run of `x` (max-subtracted);
    /// `shape` is the tensor shape of `x` and the output.
    fn softmax_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::softmax(&host, group_size))
    }
    /// `dx = y * (dy - sum(y * dy))` per softmax group (the full Jacobian,
    /// matching the Rust CPU runtime; Darknet C++ passes `dy` through because
    /// its cost layer already folds the softmax derivative in).
    fn softmax_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let y = self.download_f32(output)?;
        let dy = self.download_f32(output_grad)?;
        self.upload_f32(shape, &host_ops::softmax_backward(&y, &dy, group_size))
    }
    /// Darknet reorg on NCHW `[batch, c, h, w]` shapes.  Non-reverse folds
    /// `stride x stride` spatial blocks into channels (`c * stride^2`, `h /
    /// stride`); `reverse` is the inverse layout.  Channel `k` of the small
    /// tensor reads big channel `k % c_big` at block offset `k / c_big`.
    fn reorg_f32(
        &self,
        x: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let d = host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .unwrap_or_else(|err| panic!("{err}"));
        let host = self.download_f32(x)?;
        let out = if reverse {
            host_ops::reorg_to_big(&host, &d)
        } else {
            host_ops::reorg_to_small(&host, &d)
        };
        self.upload_f32(output_shape, &out)
    }
    /// Exact adjoint of [`Self::reorg_f32`]: returns the gradient of the layer
    /// input (`input_shape`) from the output gradient.
    fn reorg_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let d = host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .unwrap_or_else(|err| panic!("{err}"));
        let host = self.download_f32(output_grad)?;
        let dx = if reverse {
            host_ops::reorg_to_small(&host, &d)
        } else {
            host_ops::reorg_to_big(&host, &d)
        };
        self.upload_f32(input_shape, &dx)
    }
    /// `[channel_shuffle]` on NCHW `shape` (see [`host_ops::channel_permute`]): forward
    /// (`inverse = false`) uses rows = groups, cols = c / groups; `inverse = true` is the exact
    /// adjoint (rows = c / groups, cols = groups) used by the backward pass.
    fn channel_shuffle_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4 && groups > 0 && shape[1] % groups == 0,
            "channel_shuffle_f32: bad shape {shape:?} or groups {groups}"
        );
        let (rows, cols) = if inverse {
            (shape[1] / groups, groups)
        } else {
            (groups, shape[1] / groups)
        };
        let host = self.download_f32(x)?;
        let out = host_ops::channel_permute(&host, shape[0], rows, cols, shape[2] * shape[3]);
        self.upload_f32(shape, &out)
    }
    /// Gather hidden-state columns using LAYA's per-sample marker metadata.
    /// Oxide overrides this generic host reference with a native device kernel.
    fn marker_gather_f32(
        &self,
        features: &Self::Tensor,
        markers: &Self::Tensor,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let layout = host_ops::marker_gather::Layout::new(feature_shape, marker_shape, options)
            .unwrap_or_else(|err| panic!("{err}"));
        let features = self.download_f32(features)?;
        let markers = self.download_f32(markers)?;
        self.upload_f32(&layout.output_shape(), &layout.forward(&features, &markers))
    }

    /// Exact feature adjoint; marker indices/masks have no derivative.
    fn marker_gather_backward_f32(
        &self,
        markers: &Self::Tensor,
        output_grad: &Self::Tensor,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let layout = host_ops::marker_gather::Layout::new(feature_shape, marker_shape, options)
            .unwrap_or_else(|err| panic!("{err}"));
        let markers = self.download_f32(markers)?;
        let dy = self.download_f32(output_grad)?;
        self.upload_f32(feature_shape, &layout.backward(&markers, &dy))
    }

    /// Inverted dropout: element `i` is zeroed when `dropout_unit(seed, i) <
    /// probability`, otherwise scaled by `1 / (1 - probability)`.  Returns
    /// `(output, mask)` where `mask` holds the per-element multiplier.
    fn dropout_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        probability: f32,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let (out, mask) = host_ops::dropout(&host, probability, seed);
        Ok((
            self.upload_f32(shape, &out)?,
            self.upload_f32(shape, &mask)?,
        ))
    }
    /// `dx = dy * mask`.
    fn dropout_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        mask: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let dy = self.download_f32(output_grad)?;
        let m = self.download_f32(mask)?;
        let dx: Vec<f32> = dy.iter().zip(&m).map(|(a, b)| a * b).collect();
        self.upload_f32(shape, &dx)
    }
    /// DropBlock forward (see [`host_ops::dropblock`]): `shape` is NCHW,
    /// `block_prob` the per-cell selection probability and `block_size` the
    /// square side (`1..=min(w, h)`).  Returns `(output, mask)` with the
    /// per-element multiplier, so the backward is [`Self::dropout_backward_f32`].
    fn dropblock_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        block_prob: f32,
        block_size: usize,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let (out, mask) = host_ops::dropblock(&host, shape, block_prob, block_size, seed);
        Ok((
            self.upload_f32(shape, &out)?,
            self.upload_f32(shape, &mask)?,
        ))
    }
    /// Darknet `constrain_ongpu`: out-of-place `min(limit, max(-limit, x))`.
    fn constrain_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        limit: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::constrain(&host, limit))
    }
    /// Darknet `gradient_centralization_gpu` on weight updates laid out as
    /// `filters` contiguous runs of `f_size` values (no-op unless
    /// `f_size % 32 == 0`, like C++).
    fn gradient_centralization_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        f_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::gradient_centralization(&host, f_size))
    }
    /// Darknet scale_channels pre-activation: `out[b,c,y,x] = scale * source`
    /// with `scale` shaped `[batch, h*w]` (`scale_wh`) or `[batch, c]`.
    /// `shape` is the NCHW shape of `source` / the output.
    fn scale_channels_f32(
        &self,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(shape.len() == 4, "scale_channels_f32: shape must be NCHW");
        let s = self.download_f32(scale)?;
        let src = self.download_f32(source)?;
        let out = host_ops::scale_channels(&s, &src, shape[1], shape[2] * shape[3], scale_wh);
        self.upload_f32(shape, &out)
    }
    /// `(d_scale, d_source)` for upstream gradient `delta` (already through
    /// the activation derivative).
    fn scale_channels_backward_f32(
        &self,
        delta: &Self::Tensor,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4,
            "scale_channels_backward_f32: shape must be NCHW"
        );
        let g = self.download_f32(delta)?;
        let s = self.download_f32(scale)?;
        let src = self.download_f32(source)?;
        let (d_scale, d_source) = host_ops::scale_channels_backward(
            &g,
            &s,
            &src,
            shape[1],
            shape[2] * shape[3],
            scale_wh,
        );
        let scale_shape = if scale_wh {
            [shape[0], 1, shape[2], shape[3]]
        } else {
            [shape[0], shape[1], 1, 1]
        };
        Ok((
            self.upload_f32(&scale_shape, &d_scale)?,
            self.upload_f32(shape, &d_source)?,
        ))
    }
    /// Connected layer pre-activation `out[b, o] = x[b, :] . w[o, :] + bias[o]`
    /// (weights `[outputs][inputs]`, Darknet layout), output `[batch, outputs]`.
    fn connected_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let xh = self.download_f32(x)?;
        let wh = self.download_f32(weights)?;
        let bh = self.download_f32(biases)?;
        let out = host_ops::connected(&xh, &wh, &bh, batch, inputs, outputs);
        self.upload_f32(&[batch, outputs], &out)
    }
    /// `(dx, dw, db)` of [`Self::connected_f32`] for `delta` `[batch, outputs]`
    /// (already through the activation derivative).
    fn connected_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        delta: &Self::Tensor,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let xh = self.download_f32(x)?;
        let wh = self.download_f32(weights)?;
        let dh = self.download_f32(delta)?;
        let (dx, dw, db) = host_ops::connected_backward(&xh, &wh, &dh, batch, inputs, outputs);
        Ok((
            self.upload_f32(&[batch, inputs], &dx)?,
            self.upload_f32(&[outputs, inputs], &dw)?,
            self.upload_f32(&[outputs], &db)?,
        ))
    }
    /// Batch-norm training forward on NCHW `shape`: batch mean / variance over
    /// `(batch, h, w)` per channel, normalise, scale + bias, and the rolling
    /// EMA `new = keep * old + (1 - keep) * batch` (unbiased variance for the
    /// rolling estimate).  Reductions are deterministic.
    fn batchnorm_train_forward_f32(
        &self,
        x: &Self::Tensor,
        scales: &Self::Tensor,
        biases: &Self::Tensor,
        rolling_mean: &Self::Tensor,
        rolling_variance: &Self::Tensor,
        shape: &[usize],
        keep: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<BatchNormTrainForward<Self::Tensor>, Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4,
            "batchnorm_train_forward_f32: shape must be NCHW"
        );
        let r = host_ops::batchnorm_train_forward(
            &self.download_f32(x)?,
            &self.download_f32(scales)?,
            &self.download_f32(biases)?,
            &self.download_f32(rolling_mean)?,
            &self.download_f32(rolling_variance)?,
            shape[1],
            shape[2] * shape[3],
            keep,
        );
        let c = [shape[1]];
        Ok(BatchNormTrainForward {
            output: self.upload_f32(shape, &r.output)?,
            x_hat: self.upload_f32(shape, &r.x_hat)?,
            mean: self.upload_f32(&c, &r.mean)?,
            variance: self.upload_f32(&c, &r.variance)?,
            rolling_mean: self.upload_f32(&c, &r.rolling_mean)?,
            rolling_variance: self.upload_f32(&c, &r.rolling_variance)?,
        })
    }
    /// `(dx, dscale, dbias)` of [`Self::batchnorm_train_forward_f32`] for
    /// upstream gradient `output_grad` (already through the activation
    /// derivative); `variance` is the biased batch variance it returned.
    fn batchnorm_train_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        x_hat: &Self::Tensor,
        variance: &Self::Tensor,
        scales: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4,
            "batchnorm_train_backward_f32: shape must be NCHW"
        );
        let (dx, dscale, dbias) = host_ops::batchnorm_train_backward(
            &self.download_f32(output_grad)?,
            &self.download_f32(x_hat)?,
            &self.download_f32(variance)?,
            &self.download_f32(scales)?,
            shape[1],
            shape[2] * shape[3],
        );
        Ok((
            self.upload_f32(shape, &dx)?,
            self.upload_f32(&[shape[1]], &dscale)?,
            self.upload_f32(&[shape[1]], &dbias)?,
        ))
    }
    /// Darknet `reset_nan_and_inf` / `fix_nan_and_inf` (see [`NanInfMode`]),
    /// out of place.
    fn sanitize_nan_inf_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        mode: NanInfMode,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::sanitize_nan_inf(&host, mode))
    }
    /// Darknet `adam_update_gpu` (see [`AdamParams`]): out of place, returns
    /// `(weights', m', v')`; the caller zeroes the accumulated `updates`.
    fn adam_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        m: &Self::Tensor,
        v: &Self::Tensor,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let (w, m1, v1) = host_ops::adam_update(
            &self.download_f32(weights)?,
            &self.download_f32(updates)?,
            &self.download_f32(m)?,
            &self.download_f32(v)?,
            params,
        );
        let shape = [w.len()];
        Ok((
            self.upload_f32(&shape, &w)?,
            self.upload_f32(&shape, &m1)?,
            self.upload_f32(&shape, &v1)?,
        ))
    }

    /// Fused `reset_nan_and_inf(updates)` + (`fix_weights`) `fix_nan_and_inf(weights)`
    /// + [`Self::sgd_update_f32`] (decay already `0` for biases / BN scales).
    /// Backends override it to do the three steps in one launch; the result is
    /// identical to the composition below.
    fn sanitized_sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        shape: &[usize],
        fix_weights: bool,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let u = self.sanitize_nan_inf_f32(updates, shape, NanInfMode::Reset, config)?;
        let w = if fix_weights {
            self.sanitize_nan_inf_f32(weights, shape, NanInfMode::Fix, config)?
        } else {
            weights.clone()
        };
        self.sgd_update_f32(&w, &u, lr, momentum, decay, batch, config)
    }

    /// Fused sanitize + [`Self::adam_update_f32`]; returns
    /// `(weights', m', v', zeroed updates)` (Adam clears the accumulated updates).
    fn sanitized_adam_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        m: &Self::Tensor,
        v: &Self::Tensor,
        shape: &[usize],
        fix_weights: bool,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let u = self.sanitize_nan_inf_f32(updates, shape, NanInfMode::Reset, config)?;
        let w = if fix_weights {
            self.sanitize_nan_inf_f32(weights, shape, NanInfMode::Fix, config)?
        } else {
            weights.clone()
        };
        let (w, m, v) = self.adam_update_f32(&w, &u, m, v, params, config)?;
        Ok((w, m, v, self.zeros_f32(shape)?))
    }

    /// [`Self::sanitized_sgd_update_f32`] over several tensors (one layer's
    /// weights, biases and BN scales), in the same order as `items`; results
    /// are bit-identical to per-tensor calls.  Backends may do it in one launch.
    fn sanitized_sgd_update_multi_f32(
        &self,
        items: &[SgdUpdateItem<'_, Self::Tensor>],
        lr: f32,
        momentum: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor)>, Self::Error> {
        items
            .iter()
            .map(|it| {
                let acc;
                let updates = match it.grad {
                    Some(g) => {
                        acc = self.saxpy_f32(1.0, g, it.updates, config)?;
                        &acc
                    }
                    None => it.updates,
                };
                self.sanitized_sgd_update_f32(
                    it.weights,
                    updates,
                    it.shape,
                    it.fix_weights,
                    lr,
                    momentum,
                    it.decay,
                    batch,
                    config,
                )
            })
            .collect()
    }

    /// [`Self::sanitized_adam_update_f32`] over several tensors; returns
    /// `(weights', m', v', zeroed updates)` per item.
    fn sanitized_adam_update_multi_f32(
        &self,
        items: &[AdamUpdateItem<'_, Self::Tensor>],
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor)>, Self::Error> {
        items
            .iter()
            .map(|it| {
                let acc;
                let updates = match it.grad {
                    Some(g) => {
                        acc = self.saxpy_f32(1.0, g, it.updates, config)?;
                        &acc
                    }
                    None => it.updates,
                };
                self.sanitized_adam_update_f32(
                    it.weights,
                    updates,
                    it.m,
                    it.v,
                    it.shape,
                    it.fix_weights,
                    &it.params,
                    config,
                )
            })
            .collect()
    }

    /// Per-channel inference fold of batch norm: `scale = gamma / sqrt(var + eps)`
    /// and `bias' = bias - mean * scale`.  Returns `(scale, bias')`.
    fn batchnorm_fold_f32(
        &self,
        gamma: &Self::Tensor,
        mean: &Self::Tensor,
        variance: &Self::Tensor,
        bias: &Self::Tensor,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let (g, m, v, b) = (
            self.download_f32(gamma)?,
            self.download_f32(mean)?,
            self.download_f32(variance)?,
            self.download_f32(bias)?,
        );
        let scale: Vec<f32> = g
            .iter()
            .zip(&v)
            .map(|(g, v)| g / (v + eps).sqrt())
            .collect();
        let folded: Vec<f32> = b
            .iter()
            .zip(&m)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        let shape = [scale.len()];
        Ok((
            self.upload_f32(&shape, &scale)?,
            self.upload_f32(&shape, &folded)?,
        ))
    }
}

/// Opt-in half-precision (binary16) INFERENCE surface on top of the f32 ops.
///
/// Weights and activations live on the device as f16 (`TensorF16`), GEMM
/// accumulation, bias and activation are f32, results are rounded to nearest
/// even on store.  Only the ops an inference graph needs exist; there is no
/// backward and no f16 training (training stays f32).  Weights are prepared
/// from the backend's f32 tensors, so the runtime keeps one f32 source of
/// truth (folded BN) and caches the converted f16 copy.
#[allow(clippy::too_many_arguments)]
pub trait F16TensorOpsBackend: F32TensorOpsBackend {
    /// `'static` so the runtime can cache it type-erased per layer.
    type TensorF16: Clone + 'static;

    /// Uploads host f32 data and rounds it to f16 on the device.
    fn upload_f16(&self, shape: &[usize], host: &[f32]) -> Result<Self::TensorF16, Self::Error>;
    /// Downloads an f16 tensor widened to f32.
    fn download_f16(&self, tensor: &Self::TensorF16) -> Result<Vec<f32>, Self::Error>;
    /// Converts `[filters, per_filter]` f32 conv weights to the backend's f16
    /// GEMM operand layout (backend-defined padding, opaque to the caller).
    fn conv_weights_f16(
        &self,
        weights: &Self::Tensor,
        filters: usize,
        per_filter: usize,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(conv(x, weights) + bias))`; `bias` is f32 `[filters]`.
    fn conv2d_bias_act_f16(
        &self,
        x: &Self::TensorF16,
        weights: &Self::TensorF16,
        bias: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn maxpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn upsample_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn route_concat_f16(
        &self,
        inputs: &[&Self::TensorF16],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Mean over the in-image taps of each window.
    fn local_avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Global average pool to `[batch, channels, 1, 1]`.
    fn avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Flat per-sample crop / zero-pad of a mismatched shortcut source.
    fn shortcut_fit_f16(
        &self,
        x: &Self::TensorF16,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(a + b))`, same shapes (shortcut layer).
    fn shortcut_add_act_f16(
        &self,
        a: &Self::TensorF16,
        b: &Self::TensorF16,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(x * scale[c] + bias[c]))` per channel (`scale = None`:
    /// bias + activation only); `scale` / `bias` are f32 `[channels]`.
    fn channel_affine_act_f16(
        &self,
        x: &Self::TensorF16,
        scale: Option<&Self::Tensor>,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CutilePlanningNote {
    pub target_sm: String,
    pub tile_shape: [usize; 2],
    pub async_stages: usize,
    pub notes: String,
}

#[derive(Clone, Debug, Default)]
pub struct StubCudaBackend;

impl CudaBackend for StubCudaBackend {
    type Error = String;

    fn backend_name(&self) -> &'static str {
        "stub"
    }

    fn compile(&self, module: &KernelModule) -> Result<Vec<CompiledKernel>, Self::Error> {
        Ok(module
            .entry_points
            .iter()
            .map(|entry_point| CompiledKernel {
                module_name: module.name.clone(),
                entry_point: entry_point.clone(),
            })
            .collect())
    }

    fn launch(&self, _kernel: &CompiledKernel, _launch: &KernelLaunch) -> Result<(), Self::Error> {
        Err("no CUDA backend is wired into this workspace yet".to_owned())
    }
}

pub fn default_cutile_plan(kernel_name: impl Into<String>) -> CutilePlanningNote {
    CutilePlanningNote {
        target_sm: "sm_80+".to_owned(),
        tile_shape: [128, 128],
        async_stages: 2,
        notes: format!(
            "plan for {name}: map tensor tiles and host-side async launches through a future cutile-rs backend",
            name = kernel_name.into()
        ),
    }
}
