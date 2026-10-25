//! Backend-neutral CUDA tensor contracts used by the Rust runtime.
//!
//! This crate does not require a CUDA driver for CPU builds.  It defines the
//! operation traits and tensor metadata needed by the core executor; the
//! optional cuTile bridge supplies a concrete device implementation.

// Same pinned-nightly exact-erf reference as lightnet-core's LAYA math.
#![feature(float_erf)]

pub mod host_ops;
pub mod laya_act;
pub use host_ops::bilinear_gather::BilinearGatherGeometry;
pub use host_ops::{AdamParams, NanInfMode, dropout_unit};
pub use laya_act::{LayaActBackwardInputs, LayaActHeadGrad};

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

/// Gradients from the full FoveationFilm projection backward operation.
#[derive(Clone, Debug)]
pub struct FoveationFilmGradients<T> {
    pub input: T,
    pub hf_params: T,
    pub w_embed: T,
    pub b_embed: T,
    pub w_gamma: T,
    pub b_gamma: T,
    pub w_beta: T,
    pub b_beta: T,
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
    /// Exact-erf GELU, separate from F32ActivationKind::Gelu (tanh).
    /// Generic reference only; Oxide overrides with native libdevice math.
    fn gelu_erf_f32(
        &self,
        input: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let len = host_ops::gelu_erf::checked_len(shape).unwrap_or_else(|err| panic!("{err}"));
        let x = self.download_f32(input)?;
        assert_eq!(x.len(), len);
        let values: Vec<_> = x.into_iter().map(host_ops::gelu_erf::value).collect();
        self.upload_f32(shape, &values)
    }

    /// Seeded exact derivative of gelu_erf_f32 with respect to raw input.
    fn gelu_erf_backward_f32(
        &self,
        input: &Self::Tensor,
        seeds: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let len = host_ops::gelu_erf::checked_len(shape).unwrap_or_else(|err| panic!("{err}"));
        let x = self.download_f32(input)?;
        let seeds = self.download_f32(seeds)?;
        assert_eq!(x.len(), len);
        assert_eq!(seeds.len(), len);
        let values: Vec<_> = x
            .into_iter()
            .zip(seeds)
            .map(|(x, s)| s * host_ops::gelu_erf::slope(x))
            .collect();
        self.upload_f32(shape, &values)
    }

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
    /// Deformable bilinear gather used by the deformable layer family
    /// (`[deform_conv]`, `[dcnv4]`, `[dsconv]`, `[graph_conv]`).
    ///
    /// `input` is `[B, C, H, W]`; `sample_y` and `sample_x` are `[B, S, OH,
    /// OW]` positions in input pixels; the result is `[B, S, C, OH, OW]`
    /// (im2col-style columns). The coordinate contract — including the
    /// far-outside early-out and zero padding — is
    /// [`host_ops::bilinear_gather`].
    ///
    /// Wired primitives only: no layer runtime calls this yet, and a backend
    /// that does not override it round-trips through the host reference.
    fn deform_bilinear_gather_f32(
        &self,
        input: &Self::Tensor,
        sample_y: &Self::Tensor,
        sample_x: &Self::Tensor,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        geometry
            .checked_products()
            .unwrap_or_else(|err| panic!("{err}"));
        let host_input = self.download_f32(input)?;
        let host_y = self.download_f32(sample_y)?;
        let host_x = self.download_f32(sample_x)?;
        let values = host_ops::bilinear_gather::forward(&host_input, &host_y, &host_x, &geometry)
            .unwrap_or_else(|err| panic!("{err}"));
        let shape = [
            geometry.batch,
            geometry.samples,
            geometry.channels,
            geometry.output_height,
            geometry.output_width,
        ];
        self.upload_f32(&shape, &values)
    }

    /// Exact adjoint of [`Self::deform_bilinear_gather_f32`]: returns
    /// `(d_input, d_sample_y, d_sample_x)` with the coordinate gradients
    /// summed over channels.
    fn deform_bilinear_gather_backward_f32(
        &self,
        input: &Self::Tensor,
        sample_y: &Self::Tensor,
        sample_x: &Self::Tensor,
        output_grad: &Self::Tensor,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        geometry
            .checked_products()
            .unwrap_or_else(|err| panic!("{err}"));
        let host_input = self.download_f32(input)?;
        let host_y = self.download_f32(sample_y)?;
        let host_x = self.download_f32(sample_x)?;
        let host_grad = self.download_f32(output_grad)?;
        let (d_input, d_y, d_x) = host_ops::bilinear_gather::backward(
            &host_input,
            &host_y,
            &host_x,
            &host_grad,
            &geometry,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        let input_shape = [
            geometry.batch,
            geometry.channels,
            geometry.input_height,
            geometry.input_width,
        ];
        let sample_shape = [
            geometry.batch,
            geometry.samples,
            geometry.output_height,
            geometry.output_width,
        ];
        Ok((
            self.upload_f32(&input_shape, &d_input)?,
            self.upload_f32(&sample_shape, &d_y)?,
            self.upload_f32(&sample_shape, &d_x)?,
        ))
    }

    /// Raw DCNv4 NCHW output and dense NHWC offset cache. Generic backends use
    /// the host reference; native backends must override both directions.
    fn dcnv4_forward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        desc: host_ops::dcnv4::Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let (out, offsets) = host_ops::dcnv4::composition_forward(
            &self.download_f32(input)?,
            &self.download_f32(weights)?,
            &self.download_f32(biases)?,
            desc,
        )
        .expect("invalid DCNv4 composition operands");
        let g = desc.geometry;
        Ok((
            self.upload_f32(
                &[g.batch, g.channels, g.output_height, g.output_width],
                &out,
            )?,
            self.upload_f32(&[g.batch, g.spatial(), g.offset_dim()], &offsets)?,
        ))
    }

    /// Positive, batch-summed input/offset-weight/offset-bias gradients.
    fn dcnv4_backward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        offsets: &Self::Tensor,
        seed: &Self::Tensor,
        desc: host_ops::dcnv4::Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let grad = host_ops::dcnv4::composition_backward(
            &self.download_f32(input)?,
            &self.download_f32(weights)?,
            &self.download_f32(offsets)?,
            &self.download_f32(seed)?,
            desc,
        )
        .expect("invalid DCNv4 composition backward operands");
        let g = desc.geometry;
        Ok((
            self.upload_f32(
                &[g.batch, g.channels, g.input_height, g.input_width],
                &grad.input,
            )?,
            self.upload_f32(
                &[g.offset_dim(), g.channels, g.size, g.size],
                &grad.offset_weights,
            )?,
            self.upload_f32(&[g.offset_dim()], &grad.offset_biases)?,
        ))
    }

    /// Source `sanitize`: finite -> clamp to +/-10000, non-finite -> 0.
    fn deform_sanitize_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let values = self.download_f32(x)?;
        let out: Vec<f32> = values
            .into_iter()
            .map(host_ops::deform_conv::sanitize)
            .collect();
        self.upload_f32(shape, &out)
    }

    /// Per-batch `[deform_conv]` offset statistics: returns
    /// `(mean, inv_std, normalize)` of length `batch`.
    fn deform_offset_stats(
        &self,
        raw: &Self::Tensor,
        per_batch: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let values = self.download_f32(raw)?;
        let stats = host_ops::deform_conv::offset_stats(&values, per_batch, max_offset)
            .unwrap_or_else(|err| panic!("{err}"));
        let batch = stats.mean.len();
        Ok((
            self.upload_f32(&[batch], &stats.mean)?,
            self.upload_f32(&[batch], &stats.inv_std)?,
            self.upload_f32(&[batch], &stats.normalize)?,
        ))
    }

    /// Offset envelope: `sanitize -> optional normalize -> sanitize -> lhtan`.
    #[allow(clippy::too_many_arguments)]
    fn deform_offset_envelope(
        &self,
        raw: &Self::Tensor,
        mean: &Self::Tensor,
        inv_std: &Self::Tensor,
        normalize: &Self::Tensor,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let values = self.download_f32(raw)?;
        let mean = self.download_f32(mean)?;
        let inv_std = self.download_f32(inv_std)?;
        let normalize = self.download_f32(normalize)?;
        let out = host_ops::deform_conv::offset_envelope(
            &values, &mean, &inv_std, &normalize, planes, spatial, max_offset,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(shape, &out)
    }

    /// `stable_sigmoid(sanitize(logit))`, elementwise.
    fn deform_mask_sigmoid(
        &self,
        raw: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let values = self.download_f32(raw)?;
        let out = host_ops::deform_conv::mask_sigmoid(&values);
        self.upload_f32(shape, &out)
    }

    /// Plain convolution used by deform_conv offset/mask predictors, with the
    /// source-specific output extent that includes the dilated kernel extent.
    #[allow(clippy::too_many_arguments)]
    fn deform_predictor_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        in_h: usize,
        in_w: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(channels > 0 && in_h > 0 && in_w > 0 && filters > 0 && size > 0);
        assert!(stride_x > 0 && stride_y > 0 && dilation > 0);
        let x = self.download_f32(input)?;
        let w = self.download_f32(weights)?;
        let b = self.download_f32(biases)?;
        let per_batch = channels * in_h * in_w;
        assert!(per_batch > 0 && x.len() % per_batch == 0);
        let batch = x.len() / per_batch;
        assert_eq!(w.len(), filters * channels * size * size);
        assert_eq!(b.len(), filters);
        assert_eq!(output_shape, [batch, filters, out_h, out_w]);
        let mut out = vec![0.0; batch * filters * out_h * out_w];
        for n in 0..batch {
            for f in 0..filters {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut value = b[f];
                        for c in 0..channels {
                            for ky in 0..size {
                                for kx in 0..size {
                                    let y = (oy * stride_y) as isize
                                        - (padding * dilation) as isize
                                        + (ky * dilation) as isize;
                                    let x_idx = (ox * stride_x) as isize
                                        - (padding * dilation) as isize
                                        + (kx * dilation) as isize;
                                    if (0..in_h as isize).contains(&y)
                                        && (0..in_w as isize).contains(&x_idx)
                                    {
                                        let xi = ((n * channels + c) * in_h + y as usize) * in_w
                                            + x_idx as usize;
                                        let wi = ((f * channels + c) * size + ky) * size + kx;
                                        value += x[xi] * w[wi];
                                    }
                                }
                            }
                        }
                        out[((n * filters + f) * out_h + oy) * out_w + ox] = value;
                    }
                }
            }
        }
        self.upload_f32(output_shape, &out)
    }

    /// Native single-template DSConv f32 forward, including activation. `None`
    /// means this backend does not provide the optional native layer path.
    #[allow(clippy::too_many_arguments)]
    fn dsconv_forward_native_f32(
        &self,
        input: &Self::Tensor,
        offset_weights: &Self::Tensor,
        offset_biases: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        geometry: host_ops::DsConvGeometry,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Option<Result<Self::Tensor, Self::Error>> {
        let _ = (
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            geometry,
            activation,
            config,
        );
        None
    }

    /// Native DSConv f32 adjoint. Returns `(d_input, d_collapse_weights,
    /// d_collapse_biases, d_offset_weights, d_offset_biases)`; `None` means the
    /// backend does not provide native DSConv backward.
    #[allow(clippy::too_many_arguments)]
    fn dsconv_backward_native_f32(
        &self,
        input: &Self::Tensor,
        offset_weights: &Self::Tensor,
        offset_biases: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        output_gradient: &Self::Tensor,
        geometry: host_ops::DsConvGeometry,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Option<
        Result<
            (
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
            ),
            Self::Error,
        >,
    > {
        let _ = (
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            output_gradient,
            geometry,
            activation,
            config,
        );
        None
    }

    /// Deformable accumulation: `sum_c sum_k weight * sample(base + offset) *
    /// mask`, then `sanitize`. `masks` is ignored when `use_mask` is false.
    #[allow(clippy::too_many_arguments)]
    fn deform_conv_accumulate_f32(
        &self,
        input: &Self::Tensor,
        offsets: &Self::Tensor,
        masks: &Self::Tensor,
        weights: &Self::Tensor,
        geometry: host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(input)?;
        let offsets = self.download_f32(offsets)?;
        let weights = self.download_f32(weights)?;
        let masks = if use_mask {
            Some(self.download_f32(masks)?)
        } else {
            None
        };
        let out = host_ops::deform_conv::accumulate(
            &input,
            &offsets,
            masks.as_deref(),
            &weights,
            &geometry,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(shape, &out)
    }

    /// Adjoint of [`Self::deform_conv_accumulate_f32`].
    fn deform_conv_accumulate_backward_f32(
        &self,
        input: &Self::Tensor,
        offsets: &Self::Tensor,
        masks: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        geometry: host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let input = self.download_f32(input)?;
        let offsets = self.download_f32(offsets)?;
        let weights = self.download_f32(weights)?;
        let masks = if use_mask {
            Some(self.download_f32(masks)?)
        } else {
            None
        };
        let grad = self.download_f32(output_grad)?;
        let out = host_ops::deform_conv::accumulate_backward(
            &input,
            &offsets,
            masks.as_deref(),
            &weights,
            &grad,
            &geometry,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        let kernels = geometry.kernels();
        let plane = geometry.plane();
        let mask_len = geometry.batch * kernels * plane;
        Ok((
            self.upload_f32(
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
                &out.d_input,
            )?,
            self.upload_f32(
                &[
                    geometry.batch,
                    2 * kernels,
                    geometry.output_height,
                    geometry.output_width,
                ],
                &out.d_offsets,
            )?,
            self.upload_f32(&[if use_mask { mask_len } else { 0 }], &out.d_masks)?,
            self.upload_f32(&[out.d_weights.len()], &out.d_weights)?,
        ))
    }

    /// Pre-normalization envelope adjoint (lhtan + sanitize slopes).
    #[allow(clippy::too_many_arguments)]
    fn deform_envelope_pre_backward_f32(
        &self,
        raw: &Self::Tensor,
        mean: &Self::Tensor,
        inv_std: &Self::Tensor,
        normalize: &Self::Tensor,
        output_grad: &Self::Tensor,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let out = host_ops::deform_conv::offset_envelope_pre_backward(
            &self.download_f32(raw)?,
            &self.download_f32(mean)?,
            &self.download_f32(inv_std)?,
            &self.download_f32(normalize)?,
            &self.download_f32(output_grad)?,
            planes,
            spatial,
            max_offset,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(shape, &out)
    }

    /// Mask sigmoid adjoint.
    fn deform_mask_sigmoid_backward_f32(
        &self,
        logits: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let out = host_ops::deform_conv::mask_sigmoid_backward(
            &self.download_f32(logits)?,
            &self.download_f32(output_grad)?,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(shape, &out)
    }

    /// Predictor + column-gather adjoint: `(d_input, d_weights, d_biases)`.
    #[allow(clippy::too_many_arguments)]
    fn deform_predictor_backward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        filters: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let input_host = self.download_f32(input)?;
        let kernels = size * size;
        let column_count = channels * kernels;
        let columns = host_ops::deform_conv::predictor_columns_xy(
            &input_host,
            batch,
            channels,
            in_h,
            in_w,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            out_h,
            out_w,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        let out = host_ops::deform_conv::predictor_backward_xy(
            &input_host,
            &columns,
            &self.download_f32(weights)?,
            &self.download_f32(output_grad)?,
            batch,
            channels,
            filters,
            in_h,
            in_w,
            out_h,
            out_w,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        Ok((
            self.upload_f32(&[batch, channels, in_h, in_w], &out.d_input)?,
            self.upload_f32(&[filters * column_count], &out.d_weights)?,
            self.upload_f32(&[filters], &out.d_biases)?,
        ))
    }

    /// Scale a summed input gradient by the raw input's `sanitize` slope.
    fn deform_scale_input_gradient_f32(
        &self,
        raw: &Self::Tensor,
        grad: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let raw = self.download_f32(raw)?;
        let grad = self.download_f32(grad)?;
        let out: Vec<f32> = raw
            .iter()
            .zip(grad)
            .map(|(&r, g)| {
                let slope = if r.is_finite() && r.abs() < 10_000.0 {
                    1.0
                } else {
                    0.0
                };
                g * slope
            })
            .collect();
        self.upload_f32(shape, &out)
    }

    /// Bias-free transposed convolution in Darknet `[input, filter, ky, kx]`
    /// weight order, with NCHW input/output tensors.
    #[allow(clippy::too_many_arguments)]
    fn deconvolutional_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape.len(), 4, "deconvolution output must be NCHW");
        assert_eq!(output_shape[1], filters, "deconvolution filter count");
        let batch = output_shape[0];
        let output = host_ops::deconvolutional::forward(
            &self.download_f32(x)?,
            &self.download_f32(weights)?,
            batch,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride,
            output_shape[2],
            output_shape[3],
        );
        self.upload_f32(output_shape, &output)
    }
    /// `(dx, dw, db)` of [`Self::deconvolutional_f32`], where `output_grad`
    /// already includes activation backward.
    #[allow(clippy::too_many_arguments)]
    fn deconvolutional_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert_eq!(input_shape.len(), 4, "deconvolution input must be NCHW");
        assert_eq!(output_shape.len(), 4, "deconvolution output must be NCHW");
        let (batch, channels, input_height, input_width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        assert_eq!(output_shape[0], batch, "deconvolution output batch");
        assert_eq!(output_shape[1], filters, "deconvolution output filters");
        let (dx, dw, db) = host_ops::deconvolutional::backward(
            &self.download_f32(x)?,
            &self.download_f32(weights)?,
            &self.download_f32(output_grad)?,
            batch,
            channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride,
            output_shape[2],
            output_shape[3],
        );
        Ok((
            self.upload_f32(input_shape, &dx)?,
            self.upload_f32(&[channels, filters, kernel_size, kernel_size], &dw)?,
            self.upload_f32(&[filters], &db)?,
        ))
    }
    /// Uniform valid-neighbor aggregation for graph-convolution edge mode 0.
    fn graph_conv_aggregate_f32(
        &self,
        input: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(input)?;
        let output = host_ops::graph_conv::aggregate_forward(&input, geometry)
            .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(
            &[
                geometry.batch,
                geometry.channels,
                geometry.output_height,
                geometry.output_width,
            ],
            &output,
        )
    }
    /// Gather GraphConv's center/reference feature for optional self projection.
    fn graph_conv_reference_f32(
        &self,
        input: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let reference =
            host_ops::graph_conv::center_reference_forward(&self.download_f32(input)?, geometry)
                .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(
            &[
                geometry.batch,
                geometry.channels,
                geometry.output_height,
                geometry.output_width,
            ],
            &reference,
        )
    }
    /// Exact input adjoint of [`Self::graph_conv_reference_f32`].
    fn graph_conv_reference_backward_f32(
        &self,
        output_gradient: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let output_gradient = self.download_f32(output_gradient)?;
        let input_gradient =
            host_ops::graph_conv::center_reference_backward(&output_gradient, geometry)
                .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(
            &[
                geometry.batch,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
            ],
            &input_gradient,
        )
    }
    /// Learned-edge softmax aggregation for graph-convolution edge mode 1.
    fn graph_conv_learned_aggregate_f32(
        &self,
        input: &Self::Tensor,
        edge_kernel: &Self::Tensor,
        edge_biases: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let output = host_ops::graph_conv::learned_aggregate_forward(
            &self.download_f32(input)?,
            &self.download_f32(edge_kernel)?,
            &self.download_f32(edge_biases)?,
            geometry,
            groups,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(
            &[
                geometry.batch,
                geometry.channels,
                geometry.output_height,
                geometry.output_width,
            ],
            &output,
        )
    }
    /// Exact adjoint of [`Self::graph_conv_learned_aggregate_f32`].
    fn graph_conv_learned_aggregate_backward_f32(
        &self,
        input: &Self::Tensor,
        edge_kernel: &Self::Tensor,
        edge_biases: &Self::Tensor,
        output_gradient: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let gradients = host_ops::graph_conv::learned_aggregate_backward(
            &self.download_f32(input)?,
            &self.download_f32(output_gradient)?,
            &self.download_f32(edge_kernel)?,
            &self.download_f32(edge_biases)?,
            geometry,
            groups,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        Ok((
            self.upload_f32(
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
                &gradients.input,
            )?,
            self.upload_f32(&[gradients.edge_kernel.len()], &gradients.edge_kernel)?,
            self.upload_f32(&[gradients.edge_biases.len()], &gradients.edge_biases)?,
        ))
    }
    /// Exact input adjoint of [`Self::graph_conv_aggregate_f32`].
    fn graph_conv_aggregate_backward_f32(
        &self,
        output_gradient: &Self::Tensor,
        geometry: host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let output_gradient = self.download_f32(output_gradient)?;
        let input_gradient = host_ops::graph_conv::aggregate_backward(&output_gradient, geometry)
            .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(
            &[
                geometry.batch,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
            ],
            &input_gradient,
        )
    }
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
    /// Locally connected f32 projection. Weights use `[location, filter, channel, ky, kx]`.
    #[allow(clippy::too_many_arguments)]
    fn local_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(input_shape.len(), 4, "local input must be NCHW");
        assert_eq!(output_shape.len(), 4, "local output must be NCHW");
        let values = host_ops::local::forward(
            &self.download_f32(x)?,
            &self.download_f32(weights)?,
            &self.download_f32(biases)?,
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            filters,
            output_shape[2],
            output_shape[3],
            kernel_size,
            stride_x,
            stride_y,
            padding,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(output_shape, &values)
    }
    /// Local input, location-specific weight, and bias gradients for an already
    /// activation-adjusted output gradient.
    #[allow(clippy::too_many_arguments)]
    fn local_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert_eq!(input_shape.len(), 4, "local input must be NCHW");
        assert_eq!(output_shape.len(), 4, "local output must be NCHW");
        let (dx, dw, db) = host_ops::local::backward(
            &self.download_f32(x)?,
            &self.download_f32(weights)?,
            &self.download_f32(output_grad)?,
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            filters,
            output_shape[2],
            output_shape[3],
            kernel_size,
            stride_x,
            stride_y,
            padding,
        )
        .unwrap_or_else(|err| panic!("{err}"));
        Ok((
            self.upload_f32(input_shape, &dx)?,
            self.upload_f32(
                &[
                    output_shape[2] * output_shape[3],
                    filters,
                    input_shape[1],
                    kernel_size,
                    kernel_size,
                ],
                &dw,
            )?,
            self.upload_f32(&[output_shape[2] * output_shape[3], filters], &db)?,
        ))
    }
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
    /// Depth pooling: output channel `g` is the max of input channels
    /// `g, g + output_channels, ...` at each spatial position.
    fn maxpool_depth_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.download_f32(x)?;
        let batch = output_shape[0];
        let output_channels = output_shape[1];
        let height = output_shape[2];
        let width = output_shape[3];
        let spatial = height * width;
        let mut output = vec![0.0; batch * output_channels * spatial];
        for (index, value) in output.iter_mut().enumerate() {
            let group = index / spatial % output_channels;
            let b = index / (output_channels * spatial);
            let position = index % spatial;
            let first = (b * input_channels + group) * spatial + position;
            let mut best = input[first];
            let mut channel = group + output_channels;
            while channel < input_channels {
                let candidate = input[(b * input_channels + channel) * spatial + position];
                if candidate > best {
                    best = candidate;
                }
                channel += output_channels;
            }
            *value = best;
        }
        self.upload_f32(output_shape, &output)
    }
    /// Adjoint of [`Self::maxpool_depth_f32`], with first-channel tie breaking.
    fn maxpool_depth_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_channels: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.download_f32(x)?;
        let output_grad = self.download_f32(output_grad)?;
        let batch = input_shape[0];
        let input_channels = input_shape[1];
        let spatial = input_shape[2] * input_shape[3];
        let mut dx = vec![0.0; batch * input_channels * spatial];
        for b in 0..batch {
            for channel in 0..input_channels {
                let group = channel % output_channels;
                for position in 0..spatial {
                    let first = (b * input_channels + group) * spatial + position;
                    let mut winner = group;
                    let mut best = input[first];
                    let mut candidate_channel = group + output_channels;
                    while candidate_channel < input_channels {
                        let candidate =
                            input[(b * input_channels + candidate_channel) * spatial + position];
                        if candidate > best {
                            best = candidate;
                            winner = candidate_channel;
                        }
                        candidate_channel += output_channels;
                    }
                    if channel == winner {
                        dx[(b * input_channels + channel) * spatial + position] =
                            output_grad[(b * output_channels + group) * spatial + position];
                    }
                }
            }
        }
        self.upload_f32(input_shape, &dx)
    }
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
    /// Run the HF pooled-source MLP and constrain its per-sample parameters.
    /// The Oxide backend overrides this composite with device pooling, linear
    /// projections, ReLU, and output constraints.
    fn hyperbolic_foveation_predictor_f32(
        &self,
        input: &Self::Tensor,
        shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(shape.len(), 4, "HF predictor expects NCHW source shape");
        assert!(downsample > 0 && hidden_count > 0 && param_count >= 4);
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let source = self.download_f32(input)?;
        assert_eq!(source.len(), batch * channels * height * width);
        let pooled_height = (height / downsample).max(1);
        let pooled_width = (width / downsample).max(1);
        let input_dim = channels * pooled_height * pooled_width;
        let mut pooled = vec![0.0; batch * input_dim];
        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..pooled_height {
                    for ox in 0..pooled_width {
                        let mut sum = 0.0;
                        let mut count = 0usize;
                        for ky in 0..downsample {
                            let y = oy * downsample + ky;
                            for kx in 0..downsample {
                                let x = ox * downsample + kx;
                                if y < height && x < width {
                                    sum += source[((b * channels + c) * height + y) * width + x];
                                    count += 1;
                                }
                            }
                        }
                        pooled[b * input_dim + (c * pooled_height + oy) * pooled_width + ox] =
                            sum / count as f32;
                    }
                }
            }
        }
        let preactivation = host_ops::linear(&pooled, batch, input_dim, hidden_count, w1, Some(b1));
        let hidden: Vec<f32> = preactivation
            .into_iter()
            .map(|value| value.max(0.0))
            .collect();
        let raw = host_ops::linear(&hidden, batch, hidden_count, param_count, w2, Some(b2));
        let constrained: Vec<f32> = raw
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let parameter = i % param_count;
                if parameter < 2 {
                    0.9 * value.tanh()
                } else if parameter + 1 == param_count {
                    if value >= 0.0 {
                        1.0 / (1.0 + (-value).exp())
                    } else {
                        let exp = value.exp();
                        exp / (1.0 + exp)
                    }
                } else {
                    let softplus = if value > 20.0 {
                        value
                    } else {
                        value.exp().ln_1p()
                    };
                    (2.0 * softplus + 0.1).clamp(0.1, 2.0)
                }
            })
            .collect();
        self.upload_f32(&[batch, param_count], &constrained)
    }

    /// Backpropagate through the cached HF pooled-source MLP. Outputs are the
    /// source adjoint, then summed `w1`, `b1`, `w2`, and `b2` gradients.
    #[allow(clippy::too_many_arguments)]
    fn hyperbolic_foveation_predictor_backward_f32(
        &self,
        pooled: &Self::Tensor,
        hidden_preactivation: &Self::Tensor,
        hidden: &Self::Tensor,
        raw_output: &Self::Tensor,
        constrained_seed: &Self::Tensor,
        w1: &Self::Tensor,
        w2: &Self::Tensor,
        input_shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert_eq!(
            input_shape.len(),
            4,
            "HF predictor backward expects NCHW shape"
        );
        let [batch, channels, height, width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        assert!(downsample > 0 && hidden_count > 0 && param_count >= 4);
        let pooled_h = (height / downsample).max(1);
        let pooled_w = (width / downsample).max(1);
        let input_dim = channels * pooled_h * pooled_w;
        let pooled = self.download_f32(pooled)?;
        let pre = self.download_f32(hidden_preactivation)?;
        let hidden = self.download_f32(hidden)?;
        let raw = self.download_f32(raw_output)?;
        let seed = self.download_f32(constrained_seed)?;
        let w1 = self.download_f32(w1)?;
        let w2 = self.download_f32(w2)?;
        assert_eq!(pooled.len(), batch * input_dim);
        assert_eq!(pre.len(), batch * hidden_count);
        assert_eq!(hidden.len(), pre.len());
        assert_eq!(raw.len(), batch * param_count);
        assert_eq!(seed.len(), raw.len());
        assert_eq!(w1.len(), hidden_count * input_dim);
        assert_eq!(w2.len(), param_count * hidden_count);
        let mut dsource = vec![0.0; batch * channels * height * width];
        let mut dw1 = vec![0.0; w1.len()];
        let mut db1 = vec![0.0; hidden_count];
        let mut dw2 = vec![0.0; w2.len()];
        let mut db2 = vec![0.0; param_count];
        for n in 0..batch {
            let mut draw = vec![0.0; param_count];
            for p in 0..param_count {
                let r = raw[n * param_count + p];
                let derivative = if p < 2 {
                    let t = r.tanh();
                    0.9 * (1.0 - t * t)
                } else if p + 1 == param_count {
                    let s = if r >= 0.0 {
                        1.0 / (1.0 + (-r).exp())
                    } else {
                        let e = r.exp();
                        e / (1.0 + e)
                    };
                    s * (1.0 - s)
                } else {
                    let sp = if r > 20.0 { r } else { r.exp().ln_1p() };
                    let value = 2.0 * sp + 0.1;
                    if value <= 0.1 || value >= 2.0 {
                        0.0
                    } else {
                        let s = if r >= 0.0 {
                            1.0 / (1.0 + (-r).exp())
                        } else {
                            let e = r.exp();
                            e / (1.0 + e)
                        };
                        2.0 * s
                    }
                };
                draw[p] = seed[n * param_count + p] * derivative;
                db2[p] += draw[p];
                for h in 0..hidden_count {
                    dw2[p * hidden_count + h] += draw[p] * hidden[n * hidden_count + h];
                }
            }
            let mut dhidden = vec![0.0; hidden_count];
            for h in 0..hidden_count {
                for p in 0..param_count {
                    dhidden[h] += draw[p] * w2[p * hidden_count + h];
                }
                if pre[n * hidden_count + h] <= 0.0 {
                    dhidden[h] = 0.0;
                }
                db1[h] += dhidden[h];
                for i in 0..input_dim {
                    dw1[h * input_dim + i] += dhidden[h] * pooled[n * input_dim + i];
                }
            }
            let mut dpooled = vec![0.0; input_dim];
            for i in 0..input_dim {
                for h in 0..hidden_count {
                    dpooled[i] += dhidden[h] * w1[h * input_dim + i];
                }
            }
            for c in 0..channels {
                for py in 0..pooled_h {
                    for px in 0..pooled_w {
                        let y0 = py * downsample;
                        let x0 = px * downsample;
                        let count = (height - y0).min(downsample) * (width - x0).min(downsample);
                        let value = dpooled[(c * pooled_h + py) * pooled_w + px] / count as f32;
                        for y in y0..(y0 + downsample).min(height) {
                            for x in x0..(x0 + downsample).min(width) {
                                dsource[((n * channels + c) * height + y) * width + x] += value;
                            }
                        }
                    }
                }
            }
        }
        let mut w1_and_b1 = dw1;
        w1_and_b1.extend(db1);
        let mut w2_and_b2 = dw2;
        w2_and_b2.extend(db2);
        Ok((
            self.upload_f32(input_shape, &dsource)?,
            self.upload_f32(&[w1_and_b1.len()], &w1_and_b1)?,
            self.upload_f32(&[w2_and_b2.len()], &w2_and_b2)?,
        ))
    }

    /// Generate the isotropic or anisotropic Newton HF sampling grid. The default backend
    /// computes the reference grid on the host; Oxide overrides with a kernel.
    fn hyperbolic_newton_grid_f32(
        &self,
        params: &Self::Tensor,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let params = self.download_f32(params)?;
        assert!(param_count == 4 || param_count == 5);
        assert_eq!(params.len(), batch * param_count);
        let mut grid = vec![0.0; batch * height * width * 2];
        for b in 0..batch {
            let cx = params[b * param_count].clamp(-0.9, 0.9);
            let cy = params[b * param_count + 1].clamp(-0.9, 0.9);
            let rx = params[b * param_count + 2].clamp(0.1, 2.0);
            let radius = if param_count == 5 {
                (rx.powf(8.0) + params[b * param_count + 3].clamp(0.1, 2.0).powf(8.0))
                    .powf(0.125)
                    .clamp(0.1, 2.0)
            } else {
                rx.max(1e-4)
            };
            let strength = params[b * param_count + param_count - 1].clamp(0.0, 1.0);
            for y in 0..height {
                let qy = if height > 1 {
                    2.0 * y as f32 / (height - 1) as f32 - 1.0
                } else {
                    0.0
                };
                for x in 0..width {
                    let qx = if width > 1 {
                        2.0 * x as f32 / (width - 1) as f32 - 1.0
                    } else {
                        0.0
                    };
                    let mut px = qx.clamp(-1.0, 1.0);
                    let mut py = qy.clamp(-1.0, 1.0);
                    for _ in 0..iterations {
                        let dx = px - cx;
                        let dy = py - cy;
                        let r = (dx * dx + dy * dy).sqrt();
                        let (u, du_dr) = if r > 1.0e-6 {
                            let ar = alpha.max(1.0e-4) * r;
                            let tanh = ar.tanh();
                            (
                                tanh / r,
                                (alpha.max(1.0e-4) * (1.0 - tanh * tanh) * r - tanh) / (r * r),
                            )
                        } else {
                            (alpha.max(1.0e-4), 0.0)
                        };
                        let (weight, dw_dr) = if r < radius {
                            let base = (1.0 - r / radius).max(0.0);
                            (
                                base.powf(p_exp.max(1.0e-4)),
                                if r > 1.0e-6 {
                                    -(p_exp.max(1.0e-4) / radius)
                                        * base.powf((p_exp.max(1.0e-4) - 1.0).max(0.0))
                                } else {
                                    0.0
                                },
                            )
                        } else {
                            (0.0, 0.0)
                        };
                        let gain = 1.0 + strength * weight * (u - 1.0);
                        let dg = strength * (dw_dr * (u - 1.0) + weight * du_dr);
                        let mut jac = [[gain, 0.0], [0.0, gain]];
                        if r > 1.0e-6 {
                            let scale = dg / r;
                            jac[0][0] += scale * dx * dx;
                            jac[0][1] += scale * dx * dy;
                            jac[1][0] += scale * dy * dx;
                            jac[1][1] += scale * dy * dy;
                        }
                        let rx = qx - (cx + gain * dx);
                        let ry = qy - (cy + gain * dy);
                        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                        if det.abs() < 1.0e-8 {
                            px = (px + eta * rx).clamp(-1.0, 1.0);
                            py = (py + eta * ry).clamp(-1.0, 1.0);
                        } else {
                            let sx = (jac[1][1] * rx - jac[0][1] * ry) / det;
                            let sy = (-jac[1][0] * rx + jac[0][0] * ry) / det;
                            px = (px + eta * sx).clamp(-1.0, 1.0);
                            py = (py + eta * sy).clamp(-1.0, 1.0);
                        }
                    }
                    let i = ((b * height + y) * width + x) * 2;
                    grid[i] = (px + 1.0) * 0.5 * width.saturating_sub(1) as f32;
                    grid[i + 1] = (py + 1.0) * 0.5 * height.saturating_sub(1) as f32;
                }
            }
        }
        self.upload_f32(&[batch, height, width, 2], &grid)
    }

    /// Generate an isotropic or anisotropic fixed-point HF sampling grid.
    fn hyperbolic_fixed_point_grid_f32(
        &self,
        params: &Self::Tensor,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let params = self.download_f32(params)?;
        assert!(param_count == 4 || param_count == 5);
        assert_eq!(
            params.len(),
            batch * param_count,
            "HF fixed-point parameter length"
        );
        let mut grid = vec![0.0; batch * height * width * 2];
        for b in 0..batch {
            let cx = params[b * param_count].clamp(-0.9, 0.9);
            let cy = params[b * param_count + 1].clamp(-0.9, 0.9);
            let rx = params[b * param_count + 2].clamp(0.1, 2.0);
            let radius = if param_count == 5 {
                (rx.powf(8.0) + params[b * param_count + 3].clamp(0.1, 2.0).powf(8.0))
                    .powf(0.125)
                    .clamp(0.1, 2.0)
            } else {
                rx
            };
            let strength = params[b * param_count + param_count - 1].clamp(0.0, 1.0);
            for y in 0..height {
                let qy = if height > 1 {
                    2.0 * y as f32 / (height - 1) as f32 - 1.0
                } else {
                    0.0
                };
                for x in 0..width {
                    let qx = if width > 1 {
                        2.0 * x as f32 / (width - 1) as f32 - 1.0
                    } else {
                        0.0
                    };
                    let mut px = qx.clamp(-1.0, 1.0);
                    let mut py = qy.clamp(-1.0, 1.0);
                    for _ in 0..iterations {
                        let dx = px - cx;
                        let dy = py - cy;
                        let r = (dx * dx + dy * dy).sqrt();
                        let u = if r > 1.0e-6 {
                            (alpha.max(1.0e-4) * r).tanh() / r
                        } else {
                            alpha.max(1.0e-4)
                        };
                        let weight = if r < radius {
                            (1.0 - r / radius).max(0.0).powf(p_exp.max(1.0e-4))
                        } else {
                            0.0
                        };
                        let gain = 1.0 + strength * weight * (u - 1.0);
                        px = (px + eta * (qx - (cx + gain * dx))).clamp(-1.0, 1.0);
                        py = (py + eta * (qy - (cy + gain * dy))).clamp(-1.0, 1.0);
                    }
                    let index = ((b * height + y) * width + x) * 2;
                    grid[index] = (px + 1.0) * 0.5 * width.saturating_sub(1) as f32;
                    grid[index + 1] = (py + 1.0) * 0.5 * height.saturating_sub(1) as f32;
                }
            }
        }
        self.upload_f32(&[batch, height, width, 2], &grid)
    }

    /// Bilinearly sample NCHW input through an HF sampling grid in pixel
    /// coordinates; out-of-bounds taps contribute zero.
    fn hyperbolic_foveation_warp_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(shape.len(), 4, "HF warp expects NCHW shape");
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let input = self.download_f32(input)?;
        let grid = self.download_f32(grid)?;
        let spatial = height * width;
        assert_eq!(
            input.len(),
            batch * channels * spatial,
            "HF warp input length"
        );
        assert_eq!(grid.len(), batch * spatial * 2, "HF warp grid length");
        let mut output = vec![0.0; input.len()];
        for b in 0..batch {
            for c in 0..channels {
                for p in 0..spatial {
                    let grid_index = (b * spatial + p) * 2;
                    let sx = grid[grid_index];
                    let sy = grid[grid_index + 1];
                    let x0 = sx.floor() as isize;
                    let y0 = sy.floor() as isize;
                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;
                    let base = (b * channels + c) * spatial;
                    let mut value = 0.0;
                    for tap in 0..4 {
                        let tx = x0 + (tap & 1) as isize;
                        let ty = y0 + (tap >> 1) as isize;
                        if tx >= 0 && tx < width as isize && ty >= 0 && ty < height as isize {
                            let wx = if tap & 1 == 1 { dx } else { 1.0 - dx };
                            let wy = if tap >> 1 == 1 { dy } else { 1.0 - dy };
                            value += wy * wx * input[base + ty as usize * width + tx as usize];
                        }
                    }
                    output[base + p] = value;
                }
            }
        }
        self.upload_f32(shape, &output)
    }

    /// Compute input and shared-grid adjoints for the NCHW HF bilinear warp.
    fn hyperbolic_foveation_warp_backward_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        output_gradient: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert_eq!(shape.len(), 4, "HF warp backward expects NCHW shape");
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let input = self.download_f32(input)?;
        let grid = self.download_f32(grid)?;
        let output_gradient = self.download_f32(output_gradient)?;
        let spatial = height * width;
        assert_eq!(
            input.len(),
            batch * channels * spatial,
            "HF warp input length"
        );
        assert_eq!(
            output_gradient.len(),
            input.len(),
            "HF warp output-gradient length"
        );
        assert_eq!(grid.len(), batch * spatial * 2, "HF warp grid length");
        let mut input_gradient = vec![0.0; input.len()];
        let mut grid_gradient = vec![0.0; grid.len()];
        for b in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let pixel = y * width + x;
                    let gi = (b * spatial + pixel) * 2;
                    let sx = grid[gi];
                    let sy = grid[gi + 1];
                    let x0 = sx.floor() as isize;
                    let y0 = sy.floor() as isize;
                    let fx = sx - x0 as f32;
                    let fy = sy - y0 as f32;
                    for c in 0..channels {
                        let base = (b * channels + c) * spatial;
                        let seed = output_gradient[base + pixel];
                        let value = |row: isize, col: isize| -> f32 {
                            if row >= 0 && row < height as isize && col >= 0 && col < width as isize
                            {
                                input[base + row as usize * width + col as usize]
                            } else {
                                0.0
                            }
                        };
                        let v00 = value(y0, x0);
                        let v01 = value(y0, x0 + 1);
                        let v10 = value(y0 + 1, x0);
                        let v11 = value(y0 + 1, x0 + 1);
                        grid_gradient[gi] += seed * ((1.0 - fy) * (v01 - v00) + fy * (v11 - v10));
                        grid_gradient[gi + 1] +=
                            seed * ((1.0 - fx) * (v10 - v00) + fx * (v11 - v01));
                        for (row, col, weight) in [
                            (y0, x0, (1.0 - fy) * (1.0 - fx)),
                            (y0, x0 + 1, (1.0 - fy) * fx),
                            (y0 + 1, x0, fy * (1.0 - fx)),
                            (y0 + 1, x0 + 1, fy * fx),
                        ] {
                            if row >= 0 && row < height as isize && col >= 0 && col < width as isize
                            {
                                input_gradient[base + row as usize * width + col as usize] +=
                                    seed * weight;
                            }
                        }
                    }
                }
            }
        }
        Ok((
            self.upload_f32(shape, &input_gradient)?,
            self.upload_f32(&[batch, height, width, 2], &grid_gradient)?,
        ))
    }

    /// Optionally compute native fixed-point HF inverse parameter gradients
    /// from the accumulated grid adjoint. The output is `[batch, 5]` in
    /// semantic `[cx, cy, rx, ry, strength]` order; unsupported backends return
    /// `None` so callers can retain the host reference path.
    #[allow(clippy::too_many_arguments)]
    fn hyperbolic_foveation_inverse_param_grad_f32(
        &self,
        _params: &Self::Tensor,
        _grid_gradient: &Self::Tensor,
        _shape: &[usize],
        _param_count: usize,
        _alpha: f32,
        _p_exp: f32,
        _iterations: usize,
        _eta: f32,
        _newton: bool,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Option<Self::Tensor>, Self::Error> {
        Ok(None)
    }

    /// Bilinearly sample an NCHW tensor using one source coordinate pair per
    /// batch/output pixel. `input_scales` contains one affine scale per batch,
    /// and `input_offsets` one affine offset per batch/channel. Saturation and
    /// exposure are per-batch HSV adjustments, applied to RGB input taps during
    /// training when `apply_hsv` is true. Out-of-bounds taps remain zero.
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
        input_saturations: &Self::Tensor,
        input_exposures: &Self::Tensor,
        apply_hsv: bool,
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
            input_saturations,
            input_exposures,
            apply_hsv,
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
    /// Native WMHF Haar analysis transform and its four source-ordered sub-bands.
    fn wmhf_haar_dwt_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = (input, batch, channels, height, width, config);
        panic!("wmhf_haar_dwt_f32 is not implemented by this CUDA backend")
    }
    /// Exact input adjoint of WMHF Haar DWT.
    fn wmhf_haar_dwt_backward_f32(
        &self,
        ll_grad: &Self::Tensor,
        lh_grad: &Self::Tensor,
        hl_grad: &Self::Tensor,
        hh_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            ll_grad, lh_grad, hl_grad, hh_grad, batch, channels, height, width, config,
        );
        panic!("wmhf_haar_dwt_backward_f32 is not implemented by this CUDA backend")
    }
    /// Native WMHF Haar synthesis and its exact input-gradient adjoint.
    fn wmhf_haar_iwt_f32(
        &self,
        ll: &Self::Tensor,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (ll, lh, hl, hh, batch, channels, height, width, config);
        panic!("wmhf_haar_iwt_f32 is not implemented by this CUDA backend")
    }
    fn wmhf_haar_iwt_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = (output_grad, batch, channels, height, width, config);
        panic!("wmhf_haar_iwt_backward_f32 is not implemented by this CUDA backend")
    }
    /// Four-direction WMHF EMA scan forward and exact input/weight adjoint.
    fn wmhf_scan4_forward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (input, weights, batch, channels, height, width, config);
        panic!("wmhf_scan4_forward_f32 is not implemented by this CUDA backend")
    }
    fn wmhf_scan4_backward_f32(
        &self,
        input: &Self::Tensor,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = (
            input,
            output_grad,
            weights,
            batch,
            channels,
            height,
            width,
            config,
        );
        panic!("wmhf_scan4_backward_f32 is not implemented by this CUDA backend")
    }
    /// High-frequency WMHF magnitude sum and nearest-tile upsampling.
    fn wmhf_high_frequency_forward_f32(
        &self,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (lh, hl, hh, batch, channels, height, width, config);
        panic!("wmhf_high_frequency_forward_f32 is not implemented by this CUDA backend")
    }
    fn wmhf_high_frequency_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = (
            output_grad,
            lh,
            hl,
            hh,
            batch,
            channels,
            height,
            width,
            config,
        );
        panic!("wmhf_high_frequency_backward_f32 is not implemented by this CUDA backend")
    }
    /// Gather a contiguous NCHW channel range for a WMHF branch.
    fn wmhf_split_channels_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        begin: usize,
        count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (input, batch, channels, height, width, begin, count, config);
        panic!("wmhf_split_channels_f32 is not implemented by this CUDA backend")
    }
    /// ViT patch gather: NCHW input to `[batch * tokens, patch * patch * channels]`
    /// with `(dy, dx, channel)` patch order. Returns `(out_h, out_w, patches)`.
    fn vit_patchify_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(usize, usize, Self::Tensor), Self::Error> {
        let _ = config;
        let host = self.download_f32(input)?;
        let (out_h, out_w, patches) = host_ops::vit_patchify(
            &host,
            batch,
            channels,
            height,
            width,
            patch_size,
            patch_stride,
            pad,
        );
        let rows = batch * out_h * out_w;
        let tensor = self.upload_f32(&[rows, patch_size * patch_size * channels], &patches)?;
        Ok((out_h, out_w, tensor))
    }
    /// Add a `[tokens, hidden]` positional table to every sample slab.
    fn vit_add_pos_embed_f32(
        &self,
        tokens: &Self::Tensor,
        pos: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(tokens)?;
        let table = self.download_f32(pos)?;
        let out = host_ops::vit_add_pos_embed(&host, &table);
        self.upload_f32(&[out.len()], &out)
    }
    /// ViT feature guard: clamp finite values to `[-clip, clip]`, map the rest to 0.
    fn vit_clamp_feature_f32(
        &self,
        input: &Self::Tensor,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(input)?;
        let out: Vec<f32> = host
            .iter()
            .map(|&x| host_ops::vit_clamp_feature(x, clip))
            .collect();
        self.upload_f32(&[out.len()], &out)
    }
    /// ViT mHC residual `clamp(a*clamp(skip) + b*clamp(branch))`.
    fn vit_mhc_residual_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let s = self.download_f32(skip)?;
        let b = self.download_f32(branch)?;
        assert_eq!(s.len(), b.len(), "vit_mhc_residual_f32: length mismatch");
        let out: Vec<f32> = s
            .iter()
            .zip(&b)
            .map(|(&s, &b)| {
                host_ops::vit_clamp_feature(
                    skip_scale * host_ops::vit_clamp_feature(s, clip)
                        + branch_scale * host_ops::vit_clamp_feature(b, clip),
                    clip,
                )
            })
            .collect();
        self.upload_f32(&[out.len()], &out)
    }
    /// `[batch * tokens, hidden]` token rows to NCHW `[batch, hidden, out_h, out_w]`.
    fn vit_tokens_to_nchw_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(input)?;
        let tokens = out_h * out_w;
        assert_eq!(host.len(), batch * tokens * hidden, "vit_tokens_to_nchw_f32: length");
        let mut out = vec![0.0; host.len()];
        for b in 0..batch {
            for t in 0..tokens {
                for c in 0..hidden {
                    out[(b * hidden + c) * tokens + t] = host[(b * tokens + t) * hidden + c];
                }
            }
        }
        self.upload_f32(&[batch, hidden, out_h, out_w], &out)
    }
    /// NCHW `[batch, hidden, out_h, out_w]` to `[batch * tokens, hidden]` token
    /// rows: the inverse (and adjoint) of [`Self::vit_tokens_to_nchw_f32`].
    fn vit_nchw_to_tokens_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(input)?;
        let tokens = out_h * out_w;
        assert_eq!(host.len(), batch * tokens * hidden, "vit_nchw_to_tokens_f32: length");
        let mut out = vec![0.0; host.len()];
        for b in 0..batch {
            for t in 0..tokens {
                for c in 0..hidden {
                    out[(b * tokens + t) * hidden + c] = host[(b * hidden + c) * tokens + t];
                }
            }
        }
        self.upload_f32(&[batch * tokens, hidden], &out)
    }
    /// Adjoint of [`Self::vit_patchify_f32`]: `[batch * tokens, patch_dim]` patch
    /// gradients accumulated into an NCHW `[batch, channels, height, width]`
    /// input gradient (overlapping patches add).
    fn vit_patch_scatter_f32(
        &self,
        patch_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(patch_grad)?;
        let tokens = out_h * out_w;
        let patch_dim = patch_size * patch_size * channels;
        assert_eq!(host.len(), batch * tokens * patch_dim, "vit_patch_scatter_f32: length");
        let mut out = vec![0.0; batch * channels * height * width];
        for b in 0..batch {
            for ty in 0..out_h {
                for tx in 0..out_w {
                    let t = ty * out_w + tx;
                    for dy in 0..patch_size {
                        let y = ty * patch_stride + dy;
                        if y < pad || y - pad >= height {
                            continue;
                        }
                        for dx in 0..patch_size {
                            let x = tx * patch_stride + dx;
                            if x < pad || x - pad >= width {
                                continue;
                            }
                            for c in 0..channels {
                                out[((b * channels + c) * height + y - pad) * width + x - pad] +=
                                    host[(b * tokens + t) * patch_dim + (dy * patch_size + dx) * channels + c];
                            }
                        }
                    }
                }
            }
        }
        self.upload_f32(&[batch, channels, height, width], &out)
    }
    /// Adjoint of [`Self::vit_clamp_feature_f32`] at its raw input: `grad` where
    /// `raw` is finite and strictly inside `(-clip, clip)`, else zero.
    fn vit_clamp_backward_f32(
        &self,
        raw: &Self::Tensor,
        grad: &Self::Tensor,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let x = self.download_f32(raw)?;
        let g = self.download_f32(grad)?;
        assert_eq!(x.len(), g.len(), "vit_clamp_backward_f32: length mismatch");
        let out: Vec<f32> = x
            .iter()
            .zip(&g)
            .map(|(&x, &g)| if x.is_finite() && x.abs() < clip { g } else { 0.0 })
            .collect();
        self.upload_f32(&[out.len()], &out)
    }
    /// Adjoint of [`Self::vit_mhc_residual_f32`]. Returns `(d_skip, d_branch,
    /// sums)` with `sums = [grad_post_skip, grad_post_branch, grad_mix]`, the
    /// three reductions the six mHC parameter gradients of one site need.
    /// `mix`, `post_skip` and `post_branch` are the site's `p`, `2*sigmoid`
    /// scales; elements whose mix saturated or whose seed is non-finite add
    /// nothing, as in the Rust CPU reference.
    fn vit_mhc_residual_backward_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        grad: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let s = self.download_f32(skip)?;
        let b = self.download_f32(branch)?;
        let g = self.download_f32(grad)?;
        assert!(
            s.len() == b.len() && s.len() == g.len(),
            "vit_mhc_residual_backward_f32: length mismatch"
        );
        let mut ds = vec![0.0; s.len()];
        let mut db = vec![0.0; s.len()];
        let mut sums = [0.0f32; 3];
        for i in 0..s.len() {
            let sc = host_ops::vit_clamp_feature(s[i], clip);
            let bc = host_ops::vit_clamp_feature(b[i], clip);
            let mixed = skip_scale * sc + branch_scale * bc;
            if !mixed.is_finite() || mixed.abs() >= clip || !g[i].is_finite() {
                continue;
            }
            if s[i].is_finite() && s[i].abs() < clip {
                ds[i] = g[i] * skip_scale;
            }
            if b[i].is_finite() && b[i].abs() < clip {
                db[i] = g[i] * branch_scale;
            }
            sums[0] += g[i] * (mix * sc + (1.0 - mix) * bc);
            sums[1] += g[i] * ((1.0 - mix) * sc + mix * bc);
            sums[2] += g[i] * (post_skip - post_branch) * (sc - bc);
        }
        Ok((
            self.upload_f32(&[ds.len()], &ds)?,
            self.upload_f32(&[db.len()], &db)?,
            self.upload_f32(&[3], &sums)?,
        ))
    }
    /// Sum a `[batch, table]` token gradient over the batch (the learned
    /// positional-table gradient).
    fn vit_pos_embed_backward_f32(
        &self,
        grad: &Self::Tensor,
        batch: usize,
        table: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(grad)?;
        assert_eq!(host.len(), batch * table, "vit_pos_embed_backward_f32: length");
        let mut out = vec![0.0; table];
        for b in 0..batch {
            for (dst, src) in out.iter_mut().zip(&host[b * table..(b + 1) * table]) {
                *dst += src;
            }
        }
        self.upload_f32(&[table], &out)
    }
    /// Concatenate WMHF 3x/5x/7x local branch tensors in channel-major order.
    fn wmhf_concat_local_f32(
        &self,
        branch3: &Self::Tensor,
        branch5: &Self::Tensor,
        branch7: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            branch3, branch5, branch7, batch, channels, height, width, config,
        );
        panic!("wmhf_concat_local_f32 is not implemented by this CUDA backend")
    }
    /// WMHF gated residual fusion and its exact adjoint.
    fn wmhf_apply_gate_f32(
        &self,
        fuse: &Self::Tensor,
        gate: &Self::Tensor,
        projected: &Self::Tensor,
        shortcut: Option<&Self::Tensor>,
        freq_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (fuse, gate, projected, shortcut, freq_scale, config);
        panic!("wmhf_apply_gate_f32 is not implemented by this CUDA backend")
    }
    fn wmhf_apply_gate_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        gate: &Self::Tensor,
        projected: &Self::Tensor,
        freq_scale: f32,
        use_shortcut: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = (
            output_grad,
            gate,
            projected,
            freq_scale,
            use_shortcut,
            config,
        );
        panic!("wmhf_apply_gate_backward_f32 is not implemented by this CUDA backend")
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

    /// Newton–Schulz orthogonalization of one row-major Muon matrix.
    /// Backends without a native matrix path use the CPU reference.
    fn muon_orthogonalize_f32(
        &self,
        matrix: &Self::Tensor,
        shape: &[usize],
        rows: usize,
        cols: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.download_f32(matrix)?;
        let output = host_ops::muon::orthogonalize(&input, rows, cols)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(shape.iter().product::<usize>(), output.len());
        self.upload_f32(shape, &output)
    }

    /// Compute Darknet Muon's next momentum and Nesterov lookahead tensors.
    fn muon_momentum_lookahead_f32(
        &self,
        updates: &Self::Tensor,
        momentum: &Self::Tensor,
        shape: &[usize],
        batch: usize,
        loss_scale: f32,
        coefficient: f32,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let updates = self.download_f32(updates)?;
        let momentum = self.download_f32(momentum)?;
        let (next_momentum, lookahead) =
            host_ops::muon::momentum_lookahead(&updates, &momentum, batch, loss_scale, coefficient)
                .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(shape.iter().product::<usize>(), next_momentum.len());
        Ok((
            self.upload_f32(shape, &next_momentum)?,
            self.upload_f32(shape, &lookahead)?,
        ))
    }

    /// Apply the orthogonalized Muon direction plus decoupled AdamW decay.
    fn muon_apply_update_f32(
        &self,
        weights: &Self::Tensor,
        direction: &Self::Tensor,
        shape: &[usize],
        learning_rate: f32,
        decay: f32,
        direction_scale: f32,
    ) -> Result<Self::Tensor, Self::Error> {
        let output = host_ops::muon::apply_update(
            &self.download_f32(weights)?,
            &self.download_f32(direction)?,
            learning_rate,
            decay,
            direction_scale,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(shape.iter().product::<usize>(), output.len());
        self.upload_f32(shape, &output)
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
    /// LayaHead attention: fused QKV with a key-padding mask (`mask > 0.5`).
    /// Unlike the general flash-attention helper, this preserves unbounded
    /// source scores and maps an all-masked row to a zero context vector.
    fn laya_head_attention_f32(
        &self,
        qkv: &Self::Tensor,
        mask: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(batch > 0 && tokens > 0 && channels > 0 && heads > 0 && channels % heads == 0);
        assert!(
            channels / heads <= 128,
            "LayaHead attention head dim exceeds 128"
        );
        assert_eq!(output_shape, [batch * tokens, channels]);
        let qkv = self.download_f32(qkv)?;
        let mask = self.download_f32(mask)?;
        assert_eq!(qkv.len(), batch * tokens * 3 * channels);
        assert_eq!(mask.len(), batch * tokens);
        let head_dim = channels / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; batch * tokens * channels];
        let mut scores = vec![0.0f32; tokens];
        for b in 0..batch {
            for h in 0..heads {
                let head_start = h * head_dim;
                for query in 0..tokens {
                    let mut max_score = f32::NEG_INFINITY;
                    let mut any_valid = false;
                    for key in 0..tokens {
                        if mask[b * tokens + key] <= 0.5 {
                            scores[key] = f32::NEG_INFINITY;
                            continue;
                        }
                        any_valid = true;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            let qi = (b * tokens + query) * 3 * channels + head_start + d;
                            let ki = (b * tokens + key) * 3 * channels + channels + head_start + d;
                            dot += qkv[qi] * qkv[ki];
                        }
                        let score = dot * scale;
                        scores[key] = score;
                        max_score = max_score.max(score);
                    }
                    if !any_valid {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for score in &mut scores {
                        if *score != f32::NEG_INFINITY {
                            *score = (*score - max_score).exp();
                            sum += *score;
                        } else {
                            *score = 0.0;
                        }
                    }
                    let inv_sum = 1.0 / sum;
                    for d in 0..head_dim {
                        let mut value = 0.0f32;
                        for key in 0..tokens {
                            let vi =
                                (b * tokens + key) * 3 * channels + 2 * channels + head_start + d;
                            value += scores[key] * inv_sum * qkv[vi];
                        }
                        output[(b * tokens + query) * channels + head_start + d] = value;
                    }
                }
            }
        }
        self.upload_f32(output_shape, &output)
    }

    /// Convert channel-major LayaHead input `[B,C,1,T]` to token-major hidden
    /// states `[B,T,H]`, add the selected type embedding, and extract `[B,T]` mask.
    fn laya_head_prepare_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        type_embedding: &[f32],
        hidden: usize,
        qtype: i32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(input_shape.len() == 4 && input_shape[2] == 1);
        assert!((0..=2).contains(&qtype) || qtype == -1);
        let (batch, channels, tokens) = (input_shape[0], input_shape[1], input_shape[3]);
        assert!(batch > 0 && tokens > 0 && hidden > 0);
        assert_eq!(channels, hidden + if qtype < 0 { 2 } else { 1 });
        assert_eq!(type_embedding.len(), 3 * hidden);
        let source = self.download_f32(input)?;
        assert_eq!(source.len(), batch * channels * tokens);
        let (mut tokens_out, mut mask) = (
            vec![0.0; batch * tokens * hidden],
            vec![0.0; batch * tokens],
        );
        for b in 0..batch {
            let selected = if qtype >= 0 {
                qtype as usize
            } else {
                ((source[(b * channels + hidden + 1) * tokens].round() as i32).clamp(0, 2)) as usize
            };
            for t in 0..tokens {
                mask[b * tokens + t] = source[(b * channels + hidden) * tokens + t];
                for h in 0..hidden {
                    tokens_out[(b * tokens + t) * hidden + h] = source
                        [(b * channels + h) * tokens + t]
                        + type_embedding[selected * hidden + h];
                }
            }
        }
        Ok((
            self.upload_f32(&[batch * tokens, hidden], &tokens_out)?,
            self.upload_f32(&[batch, tokens], &mask)?,
        ))
    }

    /// Pack token-major `[B,T,H]` LayaHead states into channel-major
    /// `[B,H,1,T]` layer output.
    fn laya_head_pack_f32(
        &self,
        tokens: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape, [batch, hidden, 1, sequence]);
        let tokens = self.download_f32(tokens)?;
        assert_eq!(tokens.len(), batch * sequence * hidden);
        let mut output = vec![0.0; tokens.len()];
        for b in 0..batch {
            for t in 0..sequence {
                for h in 0..hidden {
                    output[(b * hidden + h) * sequence + t] =
                        tokens[(b * sequence + t) * hidden + h];
                }
            }
        }
        self.upload_f32(output_shape, &output)
    }

    /// Convert scorer input `[B,H,1,K]` into option rows `[B*K,H]`.
    fn laya_scorer_unpack_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        hidden: usize,
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(input_shape.len(), 4);
        assert_eq!(input_shape[1..], [hidden, 1, options]);
        let batch = input_shape[0];
        let source = self.download_f32(input)?;
        assert_eq!(source.len(), batch * hidden * options);
        let mut rows = vec![0.0; batch * options * hidden];
        for b in 0..batch {
            for k in 0..options {
                for h in 0..hidden {
                    rows[(b * options + k) * hidden + h] = source[(b * hidden + h) * options + k];
                }
            }
        }
        self.upload_f32(&[batch * options, hidden], &rows)
    }

    /// Rewrap packed option logits `[B*K]` as the layer's `[B,1,1,K]` output.
    fn laya_scorer_pack_f32(
        &self,
        logits: &Self::Tensor,
        batch: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape, [batch, 1, 1, options]);
        let logits = self.download_f32(logits)?;
        assert_eq!(logits.len(), batch * options);
        self.upload_f32(output_shape, &logits)
    }

    /// Reinterpret output-gradient storage `[B,1,1,K]` as connected rows `[B*K,1]`.
    fn laya_scorer_seed_rows_f32(
        &self,
        seed: &Self::Tensor,
        batch: usize,
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let values = self.download_f32(seed)?;
        assert_eq!(values.len(), batch * options);
        self.upload_f32(&[batch * options, 1], &values)
    }

    /// Scatter option-row gradients `[B*K,H]` back to input layout `[B,H,1,K]`.
    fn laya_scorer_pack_input_grad_f32(
        &self,
        rows: &Self::Tensor,
        batch: usize,
        hidden: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape, [batch, hidden, 1, options]);
        let rows = self.download_f32(rows)?;
        assert_eq!(rows.len(), batch * options * hidden);
        let mut output = vec![0.0; rows.len()];
        for b in 0..batch {
            for k in 0..options {
                for h in 0..hidden {
                    output[(b * hidden + h) * options + k] = rows[(b * options + k) * hidden + h];
                }
            }
        }
        self.upload_f32(output_shape, &output)
    }

    /// Embed token IDs from `[B,2,1,T]` and extract the attention mask.
    fn laya_encoder_embed_f32(
        &self,
        input: &Self::Tensor,
        embeddings: &Self::Tensor,
        input_shape: &[usize],
        hidden: usize,
        vocab: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(input_shape.len() == 4 && input_shape[1] == 2 && input_shape[2] == 1);
        let (batch, sequence) = (input_shape[0], input_shape[3]);
        let source = self.download_f32(input)?;
        let embeddings = self.download_f32(embeddings)?;
        assert_eq!(source.len(), batch * 2 * sequence);
        assert_eq!(embeddings.len(), vocab * hidden);
        let mut tokens = vec![0.0; batch * sequence * hidden];
        let mut mask = vec![0.0; batch * sequence];
        for b in 0..batch {
            for t in 0..sequence {
                let mut id = source[(b * 2) * sequence + t].round() as i64;
                id = id.clamp(0, vocab as i64 - 1);
                mask[b * sequence + t] = source[(b * 2 + 1) * sequence + t];
                tokens[(b * sequence + t) * hidden..(b * sequence + t + 1) * hidden]
                    .copy_from_slice(&embeddings[id as usize * hidden..(id as usize + 1) * hidden]);
            }
        }
        Ok((
            self.upload_f32(&[batch * sequence, hidden], &tokens)?,
            self.upload_f32(&[batch, sequence], &mask)?,
        ))
    }

    /// Rotary, masked (optionally sliding-window) online-softmax attention.
    fn laya_encoder_attention_f32(
        &self,
        qkv: &Self::Tensor,
        mask: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        heads: usize,
        sliding: bool,
        window_radius: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(batch > 0 && sequence > 0 && hidden > 0 && heads > 0 && hidden % heads == 0);
        assert_eq!(output_shape, [batch * sequence, hidden]);
        let hd = hidden / heads;
        assert_eq!(hd % 2, 0);
        let qkv = self.download_f32(qkv)?;
        let mask = self.download_f32(mask)?;
        let cos = self.download_f32(cos)?;
        let sin = self.download_f32(sin)?;
        assert_eq!(qkv.len(), batch * sequence * 3 * hidden);
        assert_eq!(mask.len(), batch * sequence);
        assert_eq!(cos.len(), sequence * hd / 2);
        assert_eq!(sin.len(), cos.len());
        let mut output = vec![0.0; batch * sequence * hidden];
        let scale = 1.0 / (hd as f32).sqrt();
        let half = hd / 2;
        for b in 0..batch {
            for head in 0..heads {
                for query in 0..sequence {
                    let mut scores = vec![f32::NEG_INFINITY; sequence];
                    let mut maximum = f32::NEG_INFINITY;
                    for key in 0..sequence {
                        if mask[b * sequence + key] <= 0.5
                            || (sliding && query.abs_diff(key) > window_radius)
                        {
                            continue;
                        }
                        let qrow = (b * sequence + query) * 3 * hidden + head * hd;
                        let krow = (b * sequence + key) * 3 * hidden + hidden + head * hd;
                        let mut dot = 0.0;
                        for d in 0..hd {
                            let pair = d % half;
                            let c = cos[query * half + pair];
                            let s = sin[query * half + pair];
                            let qd = if d < half {
                                qkv[qrow + d] * c - qkv[qrow + d + half] * s
                            } else {
                                qkv[qrow + d] * c + qkv[qrow + d - half] * s
                            };
                            let kc = cos[key * half + pair];
                            let ks = sin[key * half + pair];
                            let kd = if d < half {
                                qkv[krow + d] * kc - qkv[krow + d + half] * ks
                            } else {
                                qkv[krow + d] * kc + qkv[krow + d - half] * ks
                            };
                            dot += qd * kd;
                        }
                        scores[key] = dot * scale;
                        maximum = maximum.max(scores[key]);
                    }
                    if !maximum.is_finite() {
                        continue;
                    }
                    let mut sum = 0.0;
                    for score in &mut scores {
                        if score.is_finite() {
                            *score = (*score - maximum).exp();
                            sum += *score;
                        } else {
                            *score = 0.0;
                        }
                    }
                    for d in 0..hd {
                        let mut value = 0.0;
                        for key in 0..sequence {
                            let v =
                                qkv[(b * sequence + key) * 3 * hidden + 2 * hidden + head * hd + d];
                            value += scores[key] * v;
                        }
                        output[(b * sequence + query) * hidden + head * hd + d] = value / sum;
                    }
                }
            }
        }
        self.upload_f32(output_shape, &output)
    }

    /// Exact-erf GeGLU, activating the first half and multiplying by the raw second.
    fn laya_encoder_geglu_f32(
        &self,
        input: &Self::Tensor,
        rows: usize,
        intermediate: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape, [rows, intermediate]);
        let input = self.download_f32(input)?;
        assert_eq!(input.len(), rows * 2 * intermediate);
        let mut output = vec![0.0; rows * intermediate];
        for r in 0..rows {
            for i in 0..intermediate {
                let a = input[r * 2 * intermediate + i];
                let gate = input[r * 2 * intermediate + intermediate + i];
                output[r * intermediate + i] = host_ops::gelu_erf::value(a) * gate;
            }
        }
        self.upload_f32(output_shape, &output)
    }

    /// Token-major `[B*T,H]` to the layer's NCHW `[B,H,1,T]` output.
    fn laya_encoder_pack_f32(
        &self,
        tokens: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert_eq!(output_shape, [batch, hidden, 1, sequence]);
        let tokens = self.download_f32(tokens)?;
        assert_eq!(tokens.len(), batch * sequence * hidden);
        let mut output = vec![0.0; tokens.len()];
        for b in 0..batch {
            for t in 0..sequence {
                for h in 0..hidden {
                    output[(b * hidden + h) * sequence + t] =
                        tokens[(b * sequence + t) * hidden + h];
                }
            }
        }
        self.upload_f32(output_shape, &output)
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
    /// CenterNet forward: apply sigmoid to the first `classes` NCHW channels,
    /// preserve regression channels, then zero non-finite values. The default
    /// is a host reference; native backends should keep this pass on-device.
    fn centernet_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        // The runtime validates `[batch, channels, height, width]` and the
        // class extent before dispatch, as it does for the CPU reference.
        let input = self.download_f32(x)?;
        let spatial = shape[2] * shape[3];
        let mut output = input;
        for b in 0..shape[0] {
            let base = b * shape[1] * spatial;
            for i in 0..shape[1] * spatial {
                let value = if i < classes * spatial {
                    1.0 / (1.0 + (-output[base + i]).exp())
                } else {
                    output[base + i]
                };
                output[base + i] = if value.is_finite() { value } else { 0.0 };
            }
        }
        self.upload_f32(shape, &output)
    }
    /// Apply the shared YOLOX/PP-YOLOE/YOLO-NAS output activation in NCHW
    /// layout: sanitize all entries, then sigmoid the trailing objectness and
    /// class planes with logits clamped to [-30, 30].
    fn modern_yolo_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(x)?;
        let channels = shape[1];
        let spatial = shape[2] * shape[3];
        let class_begin = channels.saturating_sub(classes + 1);
        let mut output = input;
        for b in 0..shape[0] {
            let base = b * channels * spatial;
            for c in 0..channels {
                for p in 0..spatial {
                    let index = base + c * spatial + p;
                    let mut value = output[index];
                    if !value.is_finite() {
                        value = 0.0;
                    }
                    if c >= class_begin {
                        value = 1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp());
                    }
                    output[index] = value;
                }
            }
        }
        self.upload_f32(shape, &output)
    }
    /// Segmentation activation: binary sigmoid or per-pixel NCHW softmax.
    fn segmentation_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(x)?;
        let (batch, channels, spatial) = (shape[0], shape[1], shape[2] * shape[3]);
        let mut output = vec![0.0; input.len()];
        let temp = temperature.max(1.0e-6);
        for b in 0..batch {
            let base = b * channels * spatial;
            for p in 0..spatial {
                if classes == 1 {
                    let v = input[base + p];
                    let v = if v.is_finite() { v } else { 0.0 };
                    output[base + p] = 1.0 / (1.0 + (-v).exp());
                } else {
                    let mut max = f32::NEG_INFINITY;
                    for c in 0..classes {
                        let value = input[base + c * spatial + p] / temp;
                        if value > max {
                            max = value;
                        }
                    }
                    let mut sum = 0.0;
                    for c in 0..classes {
                        let index = base + c * spatial + p;
                        let value = (input[index] / temp - max).exp();
                        output[index] = value;
                        sum += value;
                    }
                    for c in 0..classes {
                        output[base + c * spatial + p] /= sum;
                    }
                }
            }
        }
        self.upload_f32(shape, &output)
    }
    /// Temperature-scaled grouped softmax for `[jev_ce]`, preserving either
    /// group-major or channel-major storage within each batch sample.
    fn jev_ce_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(x)?;
        let mut output = vec![0.0; input.len()];
        let temp = temperature.max(1.0e-6);
        for b in 0..shape[0] {
            let base = b * groups * candidates;
            for g in 0..groups {
                let index = |i: usize| {
                    if channel_major {
                        base + i * groups + g
                    } else {
                        base + g * candidates + i
                    }
                };
                let mut max = f32::NEG_INFINITY;
                for i in 0..candidates {
                    let value = input[index(i)];
                    if value > max {
                        max = value;
                    }
                }
                let mut sum = 0.0f32;
                for i in 0..candidates {
                    let value = ((input[index(i)] - max) / temp).exp();
                    output[index(i)] = value;
                    sum += value;
                }
                for i in 0..candidates {
                    output[index(i)] /= sum;
                }
            }
        }
        self.upload_f32(shape, &output)
    }
    /// LayaLoss inference forward: stable row softmax over each sample's K
    /// option logits. Training losses and masks remain outside this operation.
    fn laya_loss_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        max_options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let input = self.download_f32(x)?;
        let batch = shape[0];
        let mut output = vec![0.0; input.len()];
        for b in 0..batch {
            let base = b * max_options;
            let mut maximum = -1.0e300f64;
            for i in 0..max_options {
                maximum = maximum.max(input[base + i] as f64);
            }
            let mut exponentials = vec![0.0f64; max_options];
            let mut sum = 0.0f64;
            for i in 0..max_options {
                exponentials[i] = (input[base + i] as f64 - maximum).exp();
                sum += exponentials[i];
            }
            for i in 0..max_options {
                output[base + i] = (exponentials[i] / sum) as f32;
            }
        }
        self.upload_f32(shape, &output)
    }
    /// Region forward: transpose channel-major detector entries to spatial-
    /// major `[batch, spatial, anchor, entry]`, then activate objectness and
    /// optionally class scores. The default is a host reference.
    fn region_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(
            shape,
            &host_ops::region_forward(&host, shape, anchors, entries, classes, softmax_classes),
        )
    }
    /// Exact Rust-forward adjoint for Region, returning channel-major NCHW
    /// input gradients from the spatial-major activated output and its seed.
    fn region_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let y = self.download_f32(output)?;
        let dy = self.download_f32(output_grad)?;
        self.upload_f32(
            shape,
            &host_ops::region_backward(&y, &dy, shape, anchors, entries, classes, softmax_classes),
        )
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
    /// Standalone positive act-head gradients and CLS adjoint, not runtime
    /// backward/update integration. Oxide checks all resident operand storage
    /// before composing native dense/exact-GELU/route operations.
    fn laya_act_head_backward_f32(
        &self,
        inputs: &LayaActBackwardInputs<'_, Self::Tensor>,
        layout: &host_ops::laya_act::MlpLayout,
        config: ElementwiseLaunchConfig,
    ) -> Result<LayaActHeadGrad<Self::Tensor>, Self::Error> {
        laya_act::backward_composed(self, inputs, layout, config)
    }

    /// Add an act-head CLS adjoint into token zero of a full NCHW head
    /// gradient. Generic backends use a host reference; device backends should
    /// override this to avoid downloading the sequence gradient.
    fn laya_act_cls_scatter_f32(
        &self,
        dcls: &Self::Tensor,
        head_grad: &Self::Tensor,
        batch: usize,
        hidden: usize,
        tokens: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let dcls = self.download_f32(dcls)?;
        let head_grad = self.download_f32(head_grad)?;
        let scattered =
            host_ops::laya_act::scatter_cls_adjoint(&dcls, &head_grad, batch, hidden, tokens)
                .unwrap_or_else(|err| panic!("{err}"));
        self.upload_f32(&[batch, hidden, 1, tokens], &scattered)
    }

    /// Build CLS plus four detached LAYA act-head features in [B,H+4,1,1].
    /// This is not the MLP/backward and does not alter parameter update semantics.
    /// Generic backends use this numeric host reference; Oxide overrides it.
    fn laya_act_features_f32(
        &self,
        head: &Self::Tensor,
        scorer: &Self::Tensor,
        markers: &Self::Tensor,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let layout = host_ops::laya_act::Layout::new(head_shape, scorer_shape, marker_shape)
            .unwrap_or_else(|err| panic!("{err}"));
        let head = self.download_f32(head)?;
        let scorer = self.download_f32(scorer)?;
        let markers = self.download_f32(markers)?;
        self.upload_f32(
            &layout.output_shape(),
            &layout.forward(&head, &scorer, &markers),
        )
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
    /// Apply per-sample FiLM channel affine to an NCHW tensor.
    /// `gamma` and `beta` are `[batch, channels]`; output is
    /// `input * (1 + gamma) + beta`, broadcast over spatial positions.
    fn foveation_film_affine_f32(
        &self,
        input: &Self::Tensor,
        gamma: &Self::Tensor,
        beta: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let x = self.download_f32(input)?;
        let gamma = self.download_f32(gamma)?;
        let beta = self.download_f32(beta)?;
        self.upload_f32(
            shape,
            &host_ops::foveation_film::affine(&x, &gamma, &beta, shape),
        )
    }
    /// Full FiLM projection, returning `(output, embedding, gamma, beta)`.
    /// HF inputs are `[B,P]`, and the output is NCHW; all projection weights
    /// use row-major `[out, in]` layout.
    #[allow(clippy::too_many_arguments)]
    fn foveation_film_forward_f32(
        &self,
        input: &Self::Tensor,
        hf_params: &Self::Tensor,
        w_embed: &Self::Tensor,
        b_embed: &Self::Tensor,
        w_gamma: &Self::Tensor,
        b_gamma: &Self::Tensor,
        w_beta: &Self::Tensor,
        b_beta: &Self::Tensor,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4,
            "foveation_film_forward_f32: shape must be NCHW"
        );
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let x = self.download_f32(input)?;
        let hf = self.download_f32(hf_params)?;
        let we = self.download_f32(w_embed)?;
        let be = self.download_f32(b_embed)?;
        let wg = self.download_f32(w_gamma)?;
        let bg = self.download_f32(b_gamma)?;
        let wb = self.download_f32(w_beta)?;
        let bb = self.download_f32(b_beta)?;
        let (output, embedding, gamma, beta) = host_ops::foveation_film::forward(
            &x,
            &hf,
            batch,
            channels,
            height,
            width,
            param_count,
            embed_dim,
            &we,
            &be,
            &wg,
            &bg,
            &wb,
            &bb,
        );
        Ok((
            self.upload_f32(shape, &output)?,
            self.upload_f32(&[batch, embed_dim], &embedding)?,
            self.upload_f32(&[batch, channels], &gamma)?,
            self.upload_f32(&[batch, channels], &beta)?,
        ))
    }
    /// Full FiLM projection adjoint, including the optional HF-source vector.
    #[allow(clippy::too_many_arguments)]
    fn foveation_film_backward_f32(
        &self,
        input: &Self::Tensor,
        output_gradient: &Self::Tensor,
        hf_params: &Self::Tensor,
        embedding: &Self::Tensor,
        gamma: &Self::Tensor,
        w_embed: &Self::Tensor,
        w_gamma: &Self::Tensor,
        w_beta: &Self::Tensor,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<FoveationFilmGradients<Self::Tensor>, Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4,
            "foveation_film_backward_f32: shape must be NCHW"
        );
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let x = self.download_f32(input)?;
        let dy = self.download_f32(output_gradient)?;
        let hf = self.download_f32(hf_params)?;
        let embed = self.download_f32(embedding)?;
        let g = self.download_f32(gamma)?;
        let we = self.download_f32(w_embed)?;
        let wg = self.download_f32(w_gamma)?;
        let wb = self.download_f32(w_beta)?;
        let grads = host_ops::foveation_film::backward(
            &x,
            &dy,
            &hf,
            &embed,
            &g,
            batch,
            channels,
            height,
            width,
            param_count,
            embed_dim,
            &we,
            &wg,
            &wb,
        );
        Ok(FoveationFilmGradients {
            input: self.upload_f32(shape, &grads.input)?,
            hf_params: self.upload_f32(&[batch, param_count], &grads.hf_params)?,
            w_embed: self.upload_f32(&[embed_dim, param_count], &grads.w_embed)?,
            b_embed: self.upload_f32(&[embed_dim], &grads.b_embed)?,
            w_gamma: self.upload_f32(&[channels, embed_dim], &grads.w_gamma)?,
            b_gamma: self.upload_f32(&[channels], &grads.b_gamma)?,
            w_beta: self.upload_f32(&[channels, embed_dim], &grads.w_beta)?,
            b_beta: self.upload_f32(&[channels], &grads.b_beta)?,
        })
    }
    /// Exact adjoint of [`Self::foveation_film_affine_f32`], returning
    /// `(d_input, d_gamma, d_beta)` with the latter two `[batch, channels]`.
    fn foveation_film_affine_backward_f32(
        &self,
        input: &Self::Tensor,
        output_gradient: &Self::Tensor,
        gamma: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let x = self.download_f32(input)?;
        let gradient = self.download_f32(output_gradient)?;
        let gamma = self.download_f32(gamma)?;
        let (dx, dgamma, dbeta) =
            host_ops::foveation_film::affine_backward(&x, &gradient, &gamma, shape);
        let batch_channels = [shape[0], shape[1]];
        Ok((
            self.upload_f32(shape, &dx)?,
            self.upload_f32(&batch_channels, &dgamma)?,
            self.upload_f32(&batch_channels, &dbeta)?,
        ))
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
/// Optional device loss kernels.  Kept separate from the core tensor-op
/// traits so backends that only run inference do not have to implement them.
pub trait DenseLossBackend: F32TensorOpsBackend {
    /// Soft-target grouped cross-entropy for `[jev_ce]`: returns
    /// `(mean_ce, scaled_delta)` over the same logits layout as the forward.
    #[allow(clippy::too_many_arguments)]
    fn jev_ce_loss_f32(
        &self,
        logits: &Self::Tensor,
        truth: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        normalize: bool,
        ignore_empty: bool,
        label_smoothing: f32,
        scale: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, Self::Tensor), Self::Error>;

    /// Per-pixel segmentation cross-entropy with the correct-pixel gate,
    /// compensation and clip.  Returns `(mean_loss, scaled_delta)`.
    #[allow(clippy::too_many_arguments)]
    fn segmentation_loss_f32(
        &self,
        output: &Self::Tensor,
        truth: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        spatial: usize,
        softmax_loss: bool,
        scale: f32,
        gate_strength: f32,
        gate_t_correct: f32,
        gate_max_compensation: f32,
        max_delta: f32,
        clipping: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, Self::Tensor), Self::Error>;
}

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
    /// `f16(activation(deconv(x, weights) + bias))`; `bias` is f32 `[filters]`.
    #[allow(clippy::too_many_arguments)]
    fn deconvolutional_f16(
        &self,
        x: &Self::TensorF16,
        weights: &Self::TensorF16,
        bias: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(conv(x, weights) + bias))`; `bias` is f32 `[filters]`.
    fn normalize_channels_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
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
    fn connected_f16(
        &self,
        x: &Self::TensorF16,
        weights: &[f32],
        biases: &[f32],
        input_shape: &[usize],
        output_shape: &[usize],
        outputs: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Exact-erf GELU over a half-storage tensor, returning half-storage values.
    fn gelu_erf_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        let host = self.download_f16(x)?;
        let output: Vec<_> = host.into_iter().map(host_ops::gelu_erf::value).collect();
        self.upload_f16(shape, &output)
    }
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
    /// Depth pooling over strided input-channel groups at each spatial slot.
    fn maxpool_depth_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        input_channels: usize,
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
    /// LAYA marker gather on half-storage tensors. The generic default is a
    /// numeric host reference; Oxide overrides it with bit-preserving device copies.
    fn marker_gather_f16(
        &self,
        features: &Self::TensorF16,
        markers: &Self::TensorF16,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        let layout = host_ops::marker_gather::Layout::new(feature_shape, marker_shape, options)
            .unwrap_or_else(|err| panic!("{err}"));
        let features = self.download_f16(features)?;
        let markers = self.download_f16(markers)?;
        self.upload_f16(&layout.output_shape(), &layout.forward(&features, &markers))
    }

    /// Detached LAYA act-head summary features from compact half-storage sources.
    fn laya_act_features_f16(
        &self,
        head: &Self::TensorF16,
        scorer: &Self::TensorF16,
        marker: &Self::TensorF16,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        let layout = host_ops::laya_act::Layout::new(head_shape, scorer_shape, marker_shape)
            .unwrap_or_else(|err| panic!("{err}"));
        let head = self.download_f16(head)?;
        let scorer = self.download_f16(scorer)?;
        let marker = self.download_f16(marker)?;
        self.upload_f16(
            &layout.output_shape(),
            &layout.forward(&head, &scorer, &marker),
        )
    }

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
    /// SAM inference: `f16(act(input * source))` for same-shaped inputs.
    fn sam_f16(
        &self,
        input: &Self::TensorF16,
        source: &Self::TensorF16,
        output_shape: &[usize],
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Scale source features by per-channel or per-spatial input values, then activate.
    fn scale_channels_f16(
        &self,
        input: &Self::TensorF16,
        source: &Self::TensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        scale_wh: bool,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Channel shuffle permutation over NCHW channel planes.
    fn channel_shuffle_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Softmax over `groups` equal contiguous groups per sample, stored as f16.
    fn softmax_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        let host = self.download_f16(x)?;
        let per_batch = shape[1..].iter().product::<usize>();
        assert!(groups > 0 && per_batch.is_multiple_of(groups));
        self.upload_f16(shape, &host_ops::softmax(&host, per_batch / groups))
    }
    /// Softmax over candidates in NCHW channel-major storage, with one group
    /// per spatial location. The generic default is a numeric host reference.
    fn softmax_channel_major_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        candidates: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[1], candidates);
        assert_eq!(shape[2] * shape[3], groups);
        let input = self.download_f16(x)?;
        let mut output = input.clone();
        for batch in 0..shape[0] {
            for group in 0..groups {
                let mut max_value = f32::NEG_INFINITY;
                for candidate in 0..candidates {
                    let index = batch * candidates * groups + candidate * groups + group;
                    max_value = max_value.max(input[index]);
                }
                let sum = (0..candidates)
                    .map(|candidate| {
                        let index = batch * candidates * groups + candidate * groups + group;
                        (input[index] - max_value).exp()
                    })
                    .sum::<f32>();
                for candidate in 0..candidates {
                    let index = batch * candidates * groups + candidate * groups + group;
                    output[index] = (input[index] - max_value).exp() / sum;
                }
            }
        }
        self.upload_f16(shape, &output)
    }
    /// Reorg permutation between spatially large and channel-expanded NCHW tensors.
    fn reorg_f16(
        &self,
        x: &Self::TensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Region inference forward in half storage; no f16 backward/training path.
    fn region_forward_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        let _ = config;
        let host = self.download_f16(x)?;
        self.upload_f16(
            shape,
            &host_ops::region_forward(&host, shape, anchors, entries, classes, softmax_classes),
        )
    }
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
