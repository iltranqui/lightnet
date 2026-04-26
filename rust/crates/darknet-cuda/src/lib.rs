pub const CUTILE_RS_REFERENCE: &str = "https://github.com/NVlabs/cutile-rs";
pub const DEFAULT_TILE_1D: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ActivationKind {
    Linear,
    Logistic,
    Relu,
    Leaky,
    Mish,
    Swish,
    Tanh,
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

// `KernelLaunch` describes generic launches, but a real runtime also needs typed
// device buffers. This trait is the first concrete execution seam for Rust-side
// tensor ops while larger Darknet kernels are still being ported.
pub trait F32TensorOpsBackend {
    type Error;
    type Tensor: Clone;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error>;
    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;
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
    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
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
