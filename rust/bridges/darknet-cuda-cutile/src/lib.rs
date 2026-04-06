use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub use darknet_cuda::{CUTILE_RS_REFERENCE, DEFAULT_TILE_1D, default_cutile_plan};
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use cutile::api;
use cutile::cuda_async::device_operation::DeviceOperation;
use cutile::cuda_core::{CudaContext, CudaStream, DriverError};
use cutile::tensor::{IntoPartition, Tensor};
use cutile::tile_kernel::TileKernel;

const POINTER_TILE_1D: usize = 128;

#[derive(Debug)]
pub enum CutileBackendError {
    Cutile(cutile::error::Error),
    Device(cutile::cuda_async::error::DeviceError),
    Driver(DriverError),
    Shape(String),
}

impl Display for CutileBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cutile(err) => write!(f, "{err}"),
            Self::Device(err) => write!(f, "{err}"),
            Self::Driver(err) => write!(f, "{err}"),
            Self::Shape(message) => write!(f, "{message}"),
        }
    }
}

impl StdError for CutileBackendError {}

impl From<cutile::error::Error> for CutileBackendError {
    fn from(value: cutile::error::Error) -> Self {
        Self::Cutile(value)
    }
}

impl From<cutile::cuda_async::error::DeviceError> for CutileBackendError {
    fn from(value: cutile::cuda_async::error::DeviceError) -> Self {
        Self::Device(value)
    }
}

impl From<DriverError> for CutileBackendError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

#[derive(Clone, Debug)]
pub struct CutileTensorF32 {
    inner: Arc<Tensor<f32>>,
}

impl CutileTensorF32 {
    pub fn shape(&self) -> &[i32] {
        &self.inner.shape
    }

    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape().iter().map(|dim| *dim as usize).collect()
    }

    pub fn arc(&self) -> Arc<Tensor<f32>> {
        Arc::clone(&self.inner)
    }
}

#[derive(Clone, Debug)]
pub struct CutileBackend {
    device: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CutileBackend {
    pub fn new(device: usize) -> Result<Self, CutileBackendError> {
        let context = CudaContext::new(device)?;
        let stream = context.new_stream()?;
        Ok(Self {
            device,
            context,
            stream,
        })
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn hello_world(&self, grid: (u32, u32, u32)) -> Result<(), CutileBackendError> {
        cutile_darknet_kernels::hello_world_kernel()
            .grid(grid)
            .sync_on(&self.stream)?;
        Ok(())
    }

    pub fn upload_f32(
        &self,
        shape: &[usize],
        host: &[f32],
    ) -> Result<CutileTensorF32, CutileBackendError> {
        ensure_num_elements(shape, host.len())?;

        let host = Arc::new(host.to_vec());
        let tensor = api::copy_host_vec_to_device(&host).sync_on(&self.stream)?;
        let tensor = tensor.reshape_dyn(shape);
        Ok(CutileTensorF32 {
            inner: Arc::new(tensor),
        })
    }

    pub fn zeros_f32(&self, shape: &[usize]) -> Result<CutileTensorF32, CutileBackendError> {
        let tensor = api::zeros([shape.iter().product::<usize>()]).sync_on(&self.stream)?;
        let tensor = tensor.reshape_dyn(shape);
        Ok(CutileTensorF32 {
            inner: Arc::new(tensor),
        })
    }

    pub fn download_f32(&self, tensor: &CutileTensorF32) -> Result<Vec<f32>, CutileBackendError> {
        Ok(api::copy_device_to_host_vec(&tensor.inner).sync_on(&self.stream)?)
    }

    pub fn bias_add_f32(
        &self,
        x: &CutileTensorF32,
        bias: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let (rows, cols) =
            validate_bias_add_layout(x.len(), bias.len(), channels, channel_stride)?;
        let tile_len = tile_len.min(cols).max(1);
        let x_view = x.arc().try_view_dyn(&[rows, cols])?;
        let bias_view = bias.arc().try_view_dyn(&[channels])?;
        let z = api::zeros([rows, cols])
            .sync_on(&self.stream)?
            .partition([1, tile_len as i32]);
        let (z, _, _, _) = cutile_darknet_kernels::bias_add_2d(
            z,
            x_view,
            bias_view,
            channels as i32,
        )
            .grid((rows as u32, grid_1d(cols, tile_len), 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape_dyn(&x.shape_usize())),
        })
    }

    pub fn activation_f32(
        &self,
        x: &CutileTensorF32,
        activation: F32ActivationKind,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let len = x.len();
        let tile_len = tile_len.min(len).max(1);
        let z = api::zeros([len])
            .sync_on(&self.stream)?
            .partition([tile_len as i32]);
        let (z, _) = match activation {
            F32ActivationKind::Linear => {
                let (z, _) = cutile_darknet_kernels::activation_linear(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Logistic => {
                let (z, _) = cutile_darknet_kernels::activation_logistic(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Relu => {
                let (z, _) = cutile_darknet_kernels::activation_relu(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Leaky => {
                let (z, _) = cutile_darknet_kernels::activation_leaky(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Mish => {
                let (z, _) = cutile_darknet_kernels::activation_mish(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Swish => {
                let (z, _) = cutile_darknet_kernels::activation_swish(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Tanh => {
                let (z, _) = cutile_darknet_kernels::activation_tanh(z, x.arc())
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
        };
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape_dyn(&x.shape_usize())),
        })
    }

    pub fn shortcut_add_f32(
        &self,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        self.vector_add_f32(x, y, tile_len)
    }

    pub fn route_concat_f32(
        &self,
        inputs: &[&CutileTensorF32],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let input_lens = inputs.iter().map(|input| input.len()).collect::<Vec<_>>();
        let layout = validate_route_concat_layout(
            &input_lens,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = api::zeros([layout.output_len]).sync_on(&self.stream)?;
        let out_ptr = out.device_pointer();
        let mut channel_offset = 0usize;

        for (((input, source_channel_offset), src_channels), source_total_channels) in inputs
            .iter()
            .zip(input_channel_offsets.iter().copied())
            .zip(input_channels.iter().copied())
            .zip(input_total_channels.iter().copied())
        {
            let copy_len = layout.batch * src_channels * channel_stride;
            unsafe {
                cutile_darknet_kernels::route_copy_flat(
                    out_ptr,
                    input.arc().device_pointer(),
                    copy_len as i32,
                    source_total_channels as i32,
                    source_channel_offset as i32,
                    src_channels as i32,
                    layout.total_channels as i32,
                    channel_offset as i32,
                    channel_stride as i32,
                )
            }
            .grid((grid_1d(copy_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
            channel_offset += src_channels;
        }

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape_dyn(output_shape)),
        })
    }

    pub fn conv2d_f32(
        &self,
        x: &CutileTensorF32,
        weights: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_layout(
            x.len(),
            weights.len(),
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            groups,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = api::zeros([layout.output_len]).sync_on(&self.stream)?;
        unsafe {
            cutile_darknet_kernels::conv2d_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                weights.arc().device_pointer(),
                layout.output_len as i32,
                input_channels as i32,
                input_height as i32,
                input_width as i32,
                filters as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                kernel_size as i32,
                stride_x as i32,
                stride_y as i32,
                dilation as i32,
                padding as i32,
                groups.max(1) as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape_dyn(output_shape)),
        })
    }

    pub fn upsample_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_upsample_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            reverse,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = api::zeros([layout.output_len]).sync_on(&self.stream)?;

        if reverse {
            unsafe {
                cutile_darknet_kernels::upsample_reverse_flat(
                    out.device_pointer(),
                    x.arc().device_pointer(),
                    x.len() as i32,
                    input_height as i32,
                    input_width as i32,
                    layout.output_height as i32,
                    layout.output_width as i32,
                    stride as i32,
                    scale,
                )
            }
            .grid((grid_1d(x.len(), POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
        } else {
            unsafe {
                cutile_darknet_kernels::upsample_nearest_flat(
                    out.device_pointer(),
                    x.arc().device_pointer(),
                    layout.output_len as i32,
                    input_height as i32,
                    input_width as i32,
                    layout.output_height as i32,
                    layout.output_width as i32,
                    stride as i32,
                    scale,
                )
            }
            .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
        }

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape_dyn(output_shape)),
        })
    }

    pub fn vector_add_f32(
        &self,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if x.shape() != y.shape() {
            return Err(CutileBackendError::Shape(format!(
                "vector_add_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = tile_len.min(len).max(1);
        let z = api::zeros([len])
            .sync_on(&self.stream)?
            .partition([tile_len as i32]);
        let (z, _, _) = cutile_darknet_kernels::vector_add(z, x.arc(), y.arc())
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape_dyn(
                    &x.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                ),
            ),
        })
    }

    pub fn saxpy_f32(
        &self,
        alpha: f32,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if x.shape() != y.shape() {
            return Err(CutileBackendError::Shape(format!(
                "saxpy_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = tile_len.min(len).max(1);
        let z = api::zeros([len])
            .sync_on(&self.stream)?
            .partition([tile_len as i32]);
        let (_alpha, z, _, _) = cutile_darknet_kernels::saxpy(alpha, z, x.arc(), y.arc())
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape_dyn(
                    &x.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                ),
            ),
        })
    }
}

impl F32TensorOpsBackend for CutileBackend {
    type Error = CutileBackendError;
    type Tensor = CutileTensorF32;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upload_f32(self, shape, host)
    }

    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::zeros_f32(self, shape)
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        CutileBackend::download_f32(self, tensor)
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::bias_add_f32(self, x, bias, channels, channel_stride, config.tile_len)
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::activation_f32(self, x, activation, config.tile_len)
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::shortcut_add_f32(self, x, y, config.tile_len)
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::route_concat_f32(
            self,
            inputs,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
            config.tile_len,
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_f32(
            self,
            x,
            weights,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config.tile_len,
        )
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upsample_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            scale,
            reverse,
            config.tile_len,
        )
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::vector_add_f32(self, x, y, config.tile_len)
    }

    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::saxpy_f32(self, alpha, x, y, config.tile_len)
    }
}

fn ensure_num_elements(shape: &[usize], len: usize) -> Result<(), CutileBackendError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| CutileBackendError::Shape("shape overflow".to_owned()))?;

    if expected != len {
        return Err(CutileBackendError::Shape(format!(
            "shape {:?} expects {expected} elements, got {len}",
            shape
        )));
    }

    Ok(())
}

fn validate_bias_add_layout(
    x_len: usize,
    bias_len: usize,
    channels: usize,
    channel_stride: usize,
) -> Result<(usize, usize), CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "bias_add_f32 requires channels > 0".to_owned(),
        ));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "bias_add_f32 requires channel_stride > 0".to_owned(),
        ));
    }
    if bias_len != channels {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 bias length mismatch: expected {channels}, got {bias_len}"
        )));
    }
    if x_len % channel_stride != 0 {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 input length {x_len} is not divisible by channel_stride {channel_stride}"
        )));
    }

    let rows = x_len / channel_stride;
    if rows % channels != 0 {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 rows {rows} are not divisible by channels {channels}"
        )));
    }

    Ok((rows, channel_stride))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RouteConcatLayout {
    batch: usize,
    total_channels: usize,
    output_len: usize,
}

fn validate_route_concat_layout(
    input_lens: &[usize],
    input_channel_offsets: &[usize],
    input_channels: &[usize],
    input_total_channels: &[usize],
    output_shape: &[usize],
    channel_stride: usize,
) -> Result<RouteConcatLayout, CutileBackendError> {
    if input_lens.is_empty() {
        return Err(CutileBackendError::Shape(
            "route_concat_f32 requires at least one input".to_owned(),
        ));
    }
    if input_lens.len() != input_channel_offsets.len()
        || input_lens.len() != input_channels.len()
        || input_lens.len() != input_total_channels.len()
    {
        return Err(CutileBackendError::Shape(format!(
            "route_concat_f32 metadata length mismatch: inputs={}, channel_offsets={}, channel_counts={}, total_channels={}",
            input_lens.len(),
            input_channel_offsets.len(),
            input_channels.len(),
            input_total_channels.len(),
        )));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "route_concat_f32 requires channel_stride > 0".to_owned(),
        ));
    }

    let output_len = shape_num_elements(output_shape)?;
    let mut batch = None;
    let mut total_channels = 0usize;
    for (((input_len, channel_offset), channels), total_input_channels) in input_lens
        .iter()
        .copied()
        .zip(input_channel_offsets.iter().copied())
        .zip(input_channels.iter().copied())
        .zip(input_total_channels.iter().copied())
    {
        if channels == 0 {
            return Err(CutileBackendError::Shape(
                "route_concat_f32 requires input_channels > 0".to_owned(),
            ));
        }
        if total_input_channels == 0 {
            return Err(CutileBackendError::Shape(
                "route_concat_f32 requires input_total_channels > 0".to_owned(),
            ));
        }
        if channel_offset + channels > total_input_channels {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 channel slice [{}..{}) exceeds input total channels {}",
                channel_offset,
                channel_offset + channels,
                total_input_channels
            )));
        }
        if input_len % channel_stride != 0 {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 input length {input_len} is not divisible by channel_stride {channel_stride}"
            )));
        }
        let rows = input_len / channel_stride;
        if rows % total_input_channels != 0 {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 rows {rows} are not divisible by total input channels {total_input_channels}"
            )));
        }
        let input_batch = rows / total_input_channels;
        if let Some(expected) = batch {
            if expected != input_batch {
                return Err(CutileBackendError::Shape(format!(
                    "route_concat_f32 batch mismatch: expected {expected}, got {input_batch}"
                )));
            }
        } else {
            batch = Some(input_batch);
        }
        total_channels += channels;
    }

    let expected_output_len = batch.unwrap_or(0) * total_channels * channel_stride;
    if output_len != expected_output_len {
        return Err(CutileBackendError::Shape(format!(
            "route_concat_f32 output shape {:?} expects {output_len} elements, but inputs require {expected_output_len}",
            output_shape
        )));
    }

    Ok(RouteConcatLayout {
        batch: batch.unwrap_or(0),
        total_channels,
        output_len,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Conv2dLayout {
    channels_per_group: usize,
    filters_per_group: usize,
    output_height: usize,
    output_width: usize,
    output_len: usize,
}

fn validate_conv2d_layout(
    input_len: usize,
    weight_len: usize,
    output_shape: &[usize],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    groups: usize,
) -> Result<Conv2dLayout, CutileBackendError> {
    if input_channels == 0 || input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires non-zero input dimensions".to_owned(),
        ));
    }
    if filters == 0 || kernel_size == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires non-zero filters and kernel size".to_owned(),
        ));
    }
    if groups == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires groups > 0".to_owned(),
        ));
    }
    if input_channels % groups != 0 {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input channels {input_channels} are not divisible by groups {groups}"
        )));
    }
    if filters % groups != 0 {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 filters {filters} are not divisible by groups {groups}"
        )));
    }

    let input_plane = input_height.checked_mul(input_width).ok_or_else(|| {
        CutileBackendError::Shape("conv2d_f32 input plane overflow".to_owned())
    })?;
    let input_batch_stride = input_channels.checked_mul(input_plane).ok_or_else(|| {
        CutileBackendError::Shape("conv2d_f32 input batch stride overflow".to_owned())
    })?;
    if input_len % input_batch_stride != 0 {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input length {input_len} is not divisible by input batch stride {input_batch_stride}"
        )));
    }

    let channels_per_group = input_channels / groups;
    let filters_per_group = filters / groups;
    let weights_per_filter = channels_per_group
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(kernel_size))
        .ok_or_else(|| CutileBackendError::Shape("conv2d_f32 weights-per-filter overflow".to_owned()))?;
    let expected_weight_len = filters
        .checked_mul(weights_per_filter)
        .ok_or_else(|| CutileBackendError::Shape("conv2d_f32 weight length overflow".to_owned()))?;
    if weight_len != expected_weight_len {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 weight length mismatch: expected {expected_weight_len}, got {weight_len}"
        )));
    }

    if output_shape.len() < 4 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 output shape must be [batch, channels, height, width]".to_owned(),
        ));
    }
    let output_len = shape_num_elements(output_shape)?;
    Ok(Conv2dLayout {
        channels_per_group,
        filters_per_group,
        output_height: output_shape[output_shape.len() - 2],
        output_width: output_shape[output_shape.len() - 1],
        output_len,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpsampleLayout {
    output_height: usize,
    output_width: usize,
    output_len: usize,
}

fn validate_upsample_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    stride: usize,
    reverse: bool,
) -> Result<UpsampleLayout, CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires channels > 0".to_owned(),
        ));
    }
    if input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires non-zero input spatial dimensions".to_owned(),
        ));
    }
    if stride == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires stride > 0".to_owned(),
        ));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 input plane overflow".to_owned()))?;
    if input_len % input_plane != 0 {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 input length {input_len} is not divisible by input plane {input_plane}"
        )));
    }
    let batch_channels = input_len / input_plane;
    if batch_channels % channels != 0 {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 batch*channels {batch_channels} is not divisible by channels {channels}"
        )));
    }

    let output_len = shape_num_elements(output_shape)?;
    let output_plane = output_len
        .checked_div(batch_channels)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 invalid output plane".to_owned()))?;
    if output_plane == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 output plane cannot be zero".to_owned(),
        ));
    }

    let (output_height, output_width) = if output_shape.len() >= 2 {
        (
            output_shape[output_shape.len() - 2],
            output_shape[output_shape.len() - 1],
        )
    } else {
        return Err(CutileBackendError::Shape(
            "upsample_f32 output shape must include at least two spatial dimensions".to_owned(),
        ));
    };
    if output_height
        .checked_mul(output_width)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 output plane overflow".to_owned()))?
        != output_plane
    {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 output shape {:?} does not match batch/channels-derived plane {output_plane}",
            output_shape
        )));
    }

    let expected_output_plane = if reverse {
        input_height.div_ceil(stride) * input_width.div_ceil(stride)
    } else {
        input_height
            .checked_mul(stride)
            .and_then(|height| input_width.checked_mul(stride).map(|width| height * width))
            .ok_or_else(|| CutileBackendError::Shape("upsample_f32 expanded plane overflow".to_owned()))?
    };
    if output_plane != expected_output_plane {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 output plane mismatch: expected {expected_output_plane}, got {output_plane}"
        )));
    }

    Ok(UpsampleLayout {
        output_height,
        output_width,
        output_len,
    })
}

fn shape_num_elements(shape: &[usize]) -> Result<usize, CutileBackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| CutileBackendError::Shape("shape overflow".to_owned()))
}

fn grid_1d(len: usize, tile_len: usize) -> u32 {
    len.div_ceil(tile_len) as u32
}

#[cutile::module]
mod cutile_darknet_kernels {
    use cutile::core::*;

    unsafe fn get_tensor_1d<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        unsafe { make_tensor_view(ptr_tile, shape, strides, new_token_unordered()) }
    }

    #[cutile::entry(print_ir = true)]
    fn hello_world_kernel() {
        let _pids: (i32, i32, i32) = get_tile_block_id();
        let _npids: (i32, i32, i32) = get_num_tile_blocks();
        cuda_tile_print!(
            "darknet-cuda tile <{}, {}, {}> of <{}, {}, {}>\n",
            _pids.0,
            _pids.1,
            _pids.2,
            _npids.0,
            _npids.1,
            _npids.2
        );
    }

    #[cutile::entry(print_ir = true)]
    fn vector_add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let tile_y = load_tile_like_1d(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true)]
    fn saxpy<const S: [i32; 1]>(
        alpha: f32,
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_alpha = alpha.broadcast(z.shape());
        let tile_x = load_tile_like_1d(x, z);
        let tile_y = load_tile_like_1d(y, z);
        z.store(tile_alpha * tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true)]
    fn bias_add_2d<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [-1] }>,
        channels: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_idx = pid.0;
        let col_idx = pid.1;
        let channel_idx = row_idx % channels;
        let tile_x = x.load_tile(const_shape!(S), [row_idx, col_idx]);
        let bias_scalar = bias.load_tile(const_shape![1], [channel_idx]);
        let bias_tile = bias_scalar.reshape(const_shape![1, 1]).broadcast(z.shape());
        z.store(tile_x + bias_tile);
    }

    #[cutile::entry(print_ir = true)]
    fn activation_linear<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        z.store(tile_x);
    }

    #[cutile::entry(print_ir = true)]
    fn activation_logistic<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(one / (one + exp_neg_x));
    }

    #[cutile::entry(print_ir = true)]
    fn activation_relu<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(tile_x, zero));
    }

    #[cutile::entry(print_ir = true)]
    fn activation_leaky<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let slope: Tile<f32, S> = constant(0.1f32, z.shape());
        z.store(select(gt_tile(tile_x, zero), tile_x, slope * tile_x));
    }

    #[cutile::entry(print_ir = true)]
    fn activation_mish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let exp_x: Tile<f32, S> = exp(tile_x);
        let softplus: Tile<f32, S> = log(one + exp_x);
        let softplus_tanh: Tile<f32, S> = tanh(softplus);
        z.store(tile_x * softplus_tanh);
    }

    #[cutile::entry(print_ir = true)]
    fn activation_swish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(tile_x / (one + exp_neg_x));
    }

    #[cutile::entry(print_ir = true)]
    fn activation_tanh<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let tanh_x: Tile<f32, S> = tanh(tile_x);
        z.store(tanh_x);
    }

    #[cutile::entry()]
    unsafe fn route_copy_flat(
        dst_ptr: *mut f32,
        src_ptr: *mut f32,
        copy_len: i32,
        source_total_channels: i32,
        source_channel_offset: i32,
        source_channels: i32,
        output_total_channels: i32,
        output_channel_offset: i32,
        channel_stride: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let src_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(src_ptr);
        let src_base: PointerTile<*mut f32, { [1] }> = src_root.reshape(const_shape![1]);
        let src_ptrs: PointerTile<*mut f32, { [128] }> = src_base.broadcast(const_shape![128]);
        let dst_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(dst_ptr);
        let dst_base: PointerTile<*mut f32, { [1] }> = dst_root.reshape(const_shape![1]);
        let dst_ptrs: PointerTile<*mut f32, { [128] }> = dst_base.broadcast(const_shape![128]);

        let copy_len_tile: Tile<i32, { [128] }> = broadcast_scalar(copy_len, const_shape![128]);
        let source_group_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_channels * channel_stride, const_shape![128]);
        let source_batch_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_total_channels * channel_stride, const_shape![128]);
        let source_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(source_channel_offset * channel_stride, const_shape![128]);
        let output_batch_stride: Tile<i32, { [128] }> =
            broadcast_scalar(output_total_channels * channel_stride, const_shape![128]);
        let output_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_channel_offset * channel_stride, const_shape![128]);
        let channel_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channel_stride, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..copy_len).step_by(step as usize) {
            let local_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(local_idx, copy_len_tile);
            let batch_idx = local_idx / source_group_stride;
            let within_batch = local_idx % source_group_stride;
            let channel_idx = within_batch / channel_stride_tile;
            let elem_idx = within_batch % channel_stride_tile;

            let src_offsets =
                batch_idx * source_batch_stride + source_channel_offset_tile + channel_idx * channel_stride_tile + elem_idx;
            let src: PointerTile<*mut f32, { [128] }> = src_ptrs.offset_tile(src_offsets);
            let src_values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                "weak",
                "tl_blk",
                Some(mask),
                Some(0.0f32),
                None,
                None,
            );
            let src_tile: Tile<f32, { [128] }> = src_values.0;

            let dst_offsets =
                batch_idx * output_batch_stride + output_channel_offset_tile + channel_idx * channel_stride_tile + elem_idx;
            let dst: PointerTile<*mut f32, { [128] }> = dst_ptrs.offset_tile(dst_offsets);
            let _store_token: Token = store_ptr_tko(
                dst,
                src_tile,
                "weak",
                "tl_blk",
                Some(mask),
                None,
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn upsample_nearest_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        stride: i32,
        scale: f32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let stride_tile: Tile<i32, { [128] }> = broadcast_scalar(stride, const_shape![128]);
        let scale_tile: Tile<f32, { [128] }> = broadcast_scalar(scale, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let input_y = output_y / stride_tile;
            let input_x = output_x / stride_tile;
            let input_offsets =
                batch_channel * broadcast_scalar(input_height * input_width, const_shape![128])
                    + input_y * input_width_tile
                    + input_x;
            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(input_offsets);
            let values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                "weak",
                "tl_blk",
                Some(mask),
                Some(0.0f32),
                None,
                None,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                scaled_values,
                "weak",
                "tl_blk",
                Some(mask),
                None,
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn upsample_reverse_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        input_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        stride: i32,
        scale: f32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let input_len_tile: Tile<i32, { [128] }> = broadcast_scalar(input_len, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> = broadcast_scalar(input_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let stride_tile: Tile<i32, { [128] }> = broadcast_scalar(stride, const_shape![128]);
        let scale_tile: Tile<f32, { [128] }> = broadcast_scalar(scale, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..input_len).step_by(step as usize) {
            let input_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(input_idx, input_len_tile);
            let batch_channel = input_idx / input_plane_tile;
            let within_plane = input_idx % input_plane_tile;
            let input_y = within_plane / input_width_tile;
            let input_x = within_plane % input_width_tile;
            let output_y = input_y / stride_tile;
            let output_x = input_x / stride_tile;
            let output_offsets =
                batch_channel * output_plane_tile + output_y * output_width_tile + output_x;
            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(input_idx);
            let values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                "weak",
                "tl_blk",
                Some(mask),
                Some(0.0f32),
                None,
                None,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_offsets);
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst,
                scaled_values,
                "addf",
                "relaxed",
                "device",
                Some(mask),
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_flat(
        out_ptr: *mut f32,
        input_ptr: *mut f32,
        weight_ptr: *mut f32,
        output_len: i32,
        input_channels: i32,
        input_height: i32,
        input_width: i32,
        filters: i32,
        output_height: i32,
        output_width: i32,
        kernel_size: i32,
        stride_x: i32,
        stride_y: i32,
        dilation: i32,
        padding: i32,
        _groups: i32,
        channels_per_group: i32,
        filters_per_group: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> = weight_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let filter_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters * output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let input_width_limit_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);

            let batch_idx = output_idx / filter_plane_tile;
            let batch_rem = output_idx % filter_plane_tile;
            let filter_idx = batch_rem / output_plane_tile;
            let output_rem = batch_rem % output_plane_tile;
            let output_y = output_rem / output_width_tile;
            let output_x = output_rem % output_width_tile;
            let group_idx = filter_idx / filters_per_group_tile;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for ic in 0i32..channels_per_group {
                let ic_tile: Tile<i32, { [128] }> = broadcast_scalar(ic, const_shape![128]);
                let input_channel =
                    group_idx * channels_per_group_tile + ic_tile;
                for ky in 0i32..kernel_size {
                    let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                    let input_y =
                        output_y * broadcast_scalar(stride_y, const_shape![128])
                            + ky_tile * broadcast_scalar(dilation, const_shape![128])
                            - broadcast_scalar(padding, const_shape![128]);
                    for kx in 0i32..kernel_size {
                        let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                        let input_x =
                            output_x * broadcast_scalar(stride_x, const_shape![128])
                                + kx_tile * broadcast_scalar(dilation, const_shape![128])
                                - broadcast_scalar(padding, const_shape![128]);
                        let input_valid = output_mask
                            & ge_tile(input_y, zero_i32)
                            & lt_tile(input_y, input_height_tile)
                            & ge_tile(input_x, zero_i32)
                            & lt_tile(input_x, input_width_limit_tile);
                        let input_offsets = batch_idx
                            * broadcast_scalar(input_channels * input_height * input_width, const_shape![128])
                            + input_channel * input_plane_tile
                            + input_y * input_width_tile
                            + input_x;
                        let safe_input_offsets = select(input_valid, input_offsets, zero_i32);
                        let src: PointerTile<*mut f32, { [128] }> =
                            input_ptrs.offset_tile(safe_input_offsets);
                        let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            src,
                            "weak",
                            "tl_blk",
                            Some(input_valid),
                            Some(0.0f32),
                            None,
                            None,
                        );

                        let weight_offsets =
                            ((filter_idx * channels_per_group_tile + ic_tile)
                                * broadcast_scalar(kernel_size, const_shape![128])
                                + ky_tile)
                                * broadcast_scalar(kernel_size, const_shape![128])
                                + kx_tile;
                        let weight_ptrs_offset: PointerTile<*mut f32, { [128] }> =
                            weight_ptrs.offset_tile(weight_offsets);
                        let weight_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            weight_ptrs_offset,
                            "weak",
                            "tl_blk",
                            Some(output_mask),
                            Some(0.0f32),
                            None,
                            None,
                        );
                        let product: Tile<f32, { [128] }> = input_tile.0 * weight_tile.0;
                        acc = acc + product;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                "weak",
                "tl_blk",
                Some(output_mask),
                None,
                None,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use darknet_cuda::F32ActivationKind;

    use super::{
        validate_conv2d_layout, DEFAULT_TILE_1D, CutileBackendError, grid_1d,
        validate_bias_add_layout, validate_route_concat_layout, validate_upsample_layout,
    };

    #[test]
    fn computes_grid_1d() {
        assert_eq!(grid_1d(1024, DEFAULT_TILE_1D), 4);
        assert_eq!(grid_1d(1025, DEFAULT_TILE_1D), 5);
    }

    #[test]
    fn validates_bias_add_layout_for_batched_channels() {
        assert_eq!(validate_bias_add_layout(24, 3, 3, 4).expect("valid"), (6, 4));
    }

    #[test]
    fn rejects_bias_length_mismatch() {
        let err = validate_bias_add_layout(12, 2, 3, 4).expect_err("expected mismatch");
        assert!(matches!(err, CutileBackendError::Shape(message) if message.contains("bias length mismatch")));
    }

    #[test]
    fn activation_kinds_are_stable_for_bridge_dispatch() {
        let supported = [
            F32ActivationKind::Linear,
            F32ActivationKind::Logistic,
            F32ActivationKind::Relu,
            F32ActivationKind::Leaky,
            F32ActivationKind::Mish,
            F32ActivationKind::Swish,
            F32ActivationKind::Tanh,
        ];
        assert_eq!(supported.len(), 7);
    }

    #[test]
    fn validates_route_concat_layout() {
        let layout = validate_route_concat_layout(&[12, 4], &[1, 0], &[2, 1], &[3, 1], &[1, 3, 2, 2], 4)
            .expect("valid route layout");
        assert_eq!(layout.batch, 1);
        assert_eq!(layout.total_channels, 3);
        assert_eq!(layout.output_len, 12);
    }

    #[test]
    fn validates_upsample_layout() {
        let layout =
            validate_upsample_layout(4, &[1, 1, 4, 4], 1, 2, 2, 2, false).expect("valid");
        assert_eq!(layout.output_height, 4);
        assert_eq!(layout.output_width, 4);
        assert_eq!(layout.output_len, 16);
    }

    #[test]
    fn validates_conv2d_layout() {
        let layout = validate_conv2d_layout(8, 1, &[4, 1, 1, 2], 1, 1, 2, 1, 1, 1)
            .expect("valid conv layout");
        assert_eq!(layout.channels_per_group, 1);
        assert_eq!(layout.filters_per_group, 1);
        assert_eq!(layout.output_height, 1);
        assert_eq!(layout.output_width, 2);
        assert_eq!(layout.output_len, 8);
    }
}
