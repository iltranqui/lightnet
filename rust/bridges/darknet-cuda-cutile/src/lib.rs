//! Optional `cutile-rs` device backend for the Rust Darknet executor.
//!
//! The bridge owns CUDA context/stream creation and translates the generic
//! `darknet-cuda` operations into cuTile kernels.  It remains outside the main
//! workspace because CUDA headers, libclang, and a compatible driver are not
//! required for CPU-only development.

#![allow(clippy::too_many_arguments)]

use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub use darknet_cuda::{CUTILE_RS_REFERENCE, DEFAULT_TILE_1D, default_cutile_plan};
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use cutile::api;
use cutile::cuda_core::{Device, DriverError, Stream};
use cutile::prelude::*;

// Build-time generated distributed-slice entries.  An empty file is valid on
// CPU-only machines; runtime JIT behavior remains unchanged unless the AOT
// requirement flag is enabled in the local cuTile fork.
#[doc(hidden)]
mod aot_registry {
    include!(concat!(env!("OUT_DIR"), "/cutile_aot_registry.rs"));
}

/// Parse the comma-separated CUDA SM list used by the optional AOT build.
///
/// The default keeps the bridge usable on the repository's Ada development
/// GPU while allowing a release build to embed several architectures.  The
/// parser is intentionally shared with host-only tests; build-time compiler
/// availability is handled by `build.rs`.
pub fn parse_aot_target_sms(raw: Option<&str>) -> Result<Vec<String>, String> {
    let value = raw.unwrap_or("sm_89");
    let mut targets = Vec::new();
    for target in value.split(',').map(str::trim) {
        if target.is_empty() {
            return Err("DARKNET_CUTILE_AOT_SMS contains an empty target".to_owned());
        }
        let Some(digits) = target.strip_prefix("sm_") else {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        }
        if !targets.iter().any(|existing| existing == target) {
            targets.push(target.to_owned());
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod aot_registry_tests {
    use super::parse_aot_target_sms;

    #[test]
    fn parses_default_and_deduplicates_targets() {
        assert_eq!(parse_aot_target_sms(None).unwrap(), ["sm_89"]);
        assert_eq!(
            parse_aot_target_sms(Some(" sm_89,sm_90,sm_89 ")).unwrap(),
            ["sm_89", "sm_90"]
        );
    }

    #[test]
    fn rejects_malformed_targets() {
        assert!(parse_aot_target_sms(Some("sm_89,,sm_90")).is_err());
        assert!(parse_aot_target_sms(Some("89")).is_err());
        assert!(parse_aot_target_sms(Some("sm_")).is_err());
    }
}

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
        self.inner.shape()
    }

    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    context: Arc<Device>,
    stream: Arc<Stream>,
}

impl CutileBackend {
    pub fn new(device: usize) -> Result<Self, CutileBackendError> {
        let context = Device::new(device)?;
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

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<Device> {
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
        let tensor = tensor.reshape(shape)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(tensor),
        })
    }

    pub fn zeros_f32(&self, shape: &[usize]) -> Result<CutileTensorF32, CutileBackendError> {
        let tensor = self.zeroed_tensor_f32(shape.iter().product::<usize>())?;
        let tensor = tensor.reshape(shape)?;
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
        let (rows, cols) = validate_bias_add_layout(x.len(), bias.len(), channels, channel_stride)?;
        let tile_len = normalize_partition_tile_len(cols, tile_len);
        let x_view = x.inner.view(&[rows, cols])?;
        let bias_view = bias.inner.view(&[channels])?;
        let z = self
            .zeroed_tensor_f32(rows * cols)?
            .reshape(&[rows, cols])?
            .partition([1, tile_len]);
        let (z, _, _, _) =
            cutile_darknet_kernels::bias_add_2d(z, &x_view, &bias_view, channels as i32)
                .grid((rows as u32, grid_1d(cols, tile_len), 1))
                .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn channel_scale_f32(
        &self,
        x: &CutileTensorF32,
        scale: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let (rows, cols) =
            validate_bias_add_layout(x.len(), scale.len(), channels, channel_stride)?;
        let tile_len = normalize_partition_tile_len(cols, tile_len);
        let x_view = x.inner.view(&[rows, cols])?;
        let scale_view = scale.inner.view(&[channels])?;
        let z = self
            .zeroed_tensor_f32(rows * cols)?
            .reshape(&[rows, cols])?
            .partition([1, tile_len]);
        let (z, _, _, _) =
            cutile_darknet_kernels::channel_scale_2d(z, &x_view, &scale_view, channels as i32)
                .grid((rows as u32, grid_1d(cols, tile_len), 1))
                .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn bias_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let _ = tile_len;
        validate_bias_backward_layout(output_grad.len(), channels, channel_stride)?;
        let out = self.zeroed_tensor_f32(channels)?;
        unsafe {
            cutile_darknet_kernels::bias_backward_flat(
                out.device_pointer(),
                output_grad.arc().device_pointer(),
                output_grad.len() as i32,
                channels as i32,
                channel_stride as i32,
            )
        }
        .grid((grid_1d(output_grad.len(), POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&[channels])?),
        })
    }

    pub fn activation_f32(
        &self,
        x: &CutileTensorF32,
        activation: F32ActivationKind,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _) = match activation {
            F32ActivationKind::Linear => {
                let (z, _) = cutile_darknet_kernels::activation_linear(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Logistic => {
                let (z, _) = cutile_darknet_kernels::activation_logistic(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Relu => {
                let (z, _) = cutile_darknet_kernels::activation_relu(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Leaky => {
                let (z, _) = cutile_darknet_kernels::activation_leaky(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Mish => {
                let (z, _) = cutile_darknet_kernels::activation_mish(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Swish => {
                let (z, _) = cutile_darknet_kernels::activation_swish(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Tanh => {
                let (z, _) = cutile_darknet_kernels::activation_tanh(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
        };
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn activation_backward_f32(
        &self,
        activation_input: &CutileTensorF32,
        delta: &CutileTensorF32,
        activation: F32ActivationKind,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        ensure_same_shape(
            "activation_backward_f32",
            activation_input.shape(),
            delta.shape(),
        )?;

        let len = activation_input.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let activation_input_flat = activation_input.inner.view(&[activation_input.len()])?;
        let delta_flat = delta.inner.view(&[delta.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _, _) = match activation {
            F32ActivationKind::Linear => cutile_darknet_kernels::activation_backward_linear(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Logistic => cutile_darknet_kernels::activation_backward_logistic(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Relu => cutile_darknet_kernels::activation_backward_relu(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Leaky => cutile_darknet_kernels::activation_backward_leaky(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Mish => cutile_darknet_kernels::activation_backward_mish(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Swish => cutile_darknet_kernels::activation_backward_swish(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Tanh => cutile_darknet_kernels::activation_backward_tanh(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
        };
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&activation_input.shape_usize())?),
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
        let out = self.zeroed_tensor_f32(layout.output_len)?;
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
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn route_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<Vec<CutileTensorF32>, CutileBackendError> {
        let input_lens = input_shapes
            .iter()
            .map(|shape| shape_num_elements(shape))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = validate_route_concat_layout(
            &input_lens,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            &output_grad.shape_usize(),
            channel_stride,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);

        let mut output_channel_offset = 0usize;
        let mut input_grads = Vec::with_capacity(input_shapes.len());
        for (((input_shape, source_channel_offset), src_channels), source_total_channels) in
            input_shapes
                .iter()
                .zip(input_channel_offsets.iter().copied())
                .zip(input_channels.iter().copied())
                .zip(input_total_channels.iter().copied())
        {
            let input_len = shape_num_elements(input_shape)?;
            let input_grad = self.zeroed_tensor_f32(input_len)?;
            let copy_len = layout.batch * src_channels * channel_stride;
            unsafe {
                cutile_darknet_kernels::route_scatter_flat(
                    input_grad.device_pointer(),
                    output_grad.arc().device_pointer(),
                    copy_len as i32,
                    layout.total_channels as i32,
                    output_channel_offset as i32,
                    src_channels as i32,
                    source_total_channels as i32,
                    source_channel_offset as i32,
                    channel_stride as i32,
                )
            }
            .grid((grid_1d(copy_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;

            input_grads.push(CutileTensorF32 {
                inner: Arc::new(input_grad.reshape(input_shape)?),
            });
            output_channel_offset += src_channels;
        }

        Ok(input_grads)
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
        let out = self.zeroed_tensor_f32(layout.output_len)?;
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
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn conv2d_backward_input_f32(
        &self,
        output_grad: &CutileTensorF32,
        weights: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_backward_input_layout(
            output_grad.len(),
            weights.len(),
            input_shape,
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
        )?;
        let _tile_len = tile_len.min(layout.input_len).max(1);
        let input_grad = self.zeroed_tensor_f32(layout.input_len)?;
        unsafe {
            cutile_darknet_kernels::conv2d_backward_input_flat(
                input_grad.device_pointer(),
                output_grad.arc().device_pointer(),
                weights.arc().device_pointer(),
                layout.input_len as i32,
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
                groups as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.input_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(input_grad.reshape(input_shape)?),
        })
    }

    pub fn conv2d_backward_weights_f32(
        &self,
        x: &CutileTensorF32,
        output_grad: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_backward_weights_layout(
            x.len(),
            output_grad.len(),
            weight_shape,
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
        )?;
        let _tile_len = tile_len.min(layout.weight_len).max(1);
        let weight_grad = self.zeroed_tensor_f32(layout.weight_len)?;
        unsafe {
            cutile_darknet_kernels::conv2d_backward_weights_flat(
                weight_grad.device_pointer(),
                x.arc().device_pointer(),
                output_grad.arc().device_pointer(),
                layout.weight_len as i32,
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
                groups as i32,
                layout.batch as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.weight_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(weight_grad.reshape(weight_shape)?),
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
        let out = self.zeroed_tensor_f32(layout.output_len)?;

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
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn maxpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = self.zeroed_tensor_f32(layout.output_len)?;

        unsafe {
            cutile_darknet_kernels::maxpool_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                layout.output_len as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                size as i32,
                stride_x as i32,
                stride_y as i32,
                (padding / 2) as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn maxpool_backward_f32(
        &self,
        x: &CutileTensorF32,
        output_grad: &CutileTensorF32,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            &output_grad.shape_usize(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let input_grad = self.zeroed_tensor_f32(x.len())?;

        unsafe {
            cutile_darknet_kernels::maxpool_backward_flat(
                input_grad.device_pointer(),
                x.arc().device_pointer(),
                output_grad.arc().device_pointer(),
                layout.output_len as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                size as i32,
                stride_x as i32,
                stride_y as i32,
                (padding / 2) as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(input_grad.reshape(&x.shape_usize())?),
        })
    }

    pub fn local_avgpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let input = self.download_f32(x)?;
        let mut out = vec![0.0; layout.output_len];
        let pad_offset = (padding / 2) as isize;
        let batch = output_shape[0];

        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..layout.output_height {
                    for ox in 0..layout.output_width {
                        let mut acc = 0.0;
                        let mut count = 0usize;
                        for ky in 0..size {
                            let in_y = oy as isize * stride_y as isize + ky as isize - pad_offset;
                            if !(0..input_height as isize).contains(&in_y) {
                                continue;
                            }
                            for kx in 0..size {
                                let in_x =
                                    ox as isize * stride_x as isize + kx as isize - pad_offset;
                                if !(0..input_width as isize).contains(&in_x) {
                                    continue;
                                }
                                let src = ((((b * channels) + c) * input_height) + in_y as usize)
                                    * input_width
                                    + in_x as usize;
                                acc += input[src];
                                count += 1;
                            }
                        }
                        if count > 0 {
                            let dst = ((((b * channels) + c) * layout.output_height) + oy)
                                * layout.output_width
                                + ox;
                            out[dst] = acc / count as f32;
                        }
                    }
                }
            }
        }

        self.upload_f32(output_shape, &out)
    }

    pub fn local_avgpool_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if input_shape.len() != 4
            || input_shape[1] != channels
            || input_shape[2] != input_height
            || input_shape[3] != input_width
        {
            return Err(CutileBackendError::Shape(format!(
                "local_avgpool_backward_f32 input shape mismatch: expected [batch, {channels}, {input_height}, {input_width}], got {input_shape:?}"
            )));
        }
        let output_shape = output_grad.shape_usize();
        let layout = validate_maxpool_layout(
            input_shape.iter().product(),
            &output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let output_grad = self.download_f32(output_grad)?;
        let mut input_grad = vec![0.0; input_shape.iter().product()];
        let pad_offset = (padding / 2) as isize;
        let batch = input_shape[0];

        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..layout.output_height {
                    for ox in 0..layout.output_width {
                        let mut valid = Vec::new();
                        for ky in 0..size {
                            let in_y = oy as isize * stride_y as isize + ky as isize - pad_offset;
                            if !(0..input_height as isize).contains(&in_y) {
                                continue;
                            }
                            for kx in 0..size {
                                let in_x =
                                    ox as isize * stride_x as isize + kx as isize - pad_offset;
                                if !(0..input_width as isize).contains(&in_x) {
                                    continue;
                                }
                                valid.push((in_y as usize, in_x as usize));
                            }
                        }
                        if !valid.is_empty() {
                            let src = ((((b * channels) + c) * layout.output_height) + oy)
                                * layout.output_width
                                + ox;
                            let share = output_grad[src] / valid.len() as f32;
                            for (y, x) in valid {
                                let dst =
                                    ((((b * channels) + c) * input_height) + y) * input_width + x;
                                input_grad[dst] += share;
                            }
                        }
                    }
                }
            }
        }

        self.upload_f32(input_shape, &input_grad)
    }

    pub fn avgpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if output_shape.len() != 4
            || output_shape[1] != channels
            || output_shape[2] != 1
            || output_shape[3] != 1
        {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 output shape must be [batch, {channels}, 1, 1], got {output_shape:?}"
            )));
        }
        let batch_stride = channels
            .checked_mul(input_height)
            .and_then(|value| value.checked_mul(input_width))
            .ok_or_else(|| {
                CutileBackendError::Shape("avgpool_f32 input stride overflow".to_owned())
            })?;
        if batch_stride == 0 || !x.len().is_multiple_of(batch_stride) {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 input length {} is incompatible with channels={channels}, height={input_height}, width={input_width}",
                x.len()
            )));
        }
        let batch = x.len() / batch_stride;
        if output_shape[0] != batch {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 output batch mismatch: expected {batch}, got {}",
                output_shape[0]
            )));
        }

        let input = self.download_f32(x)?;
        let area = (input_height * input_width) as f32;
        let mut out = vec![0.0; batch * channels];
        for b in 0..batch {
            for c in 0..channels {
                let mut acc = 0.0;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let src = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        acc += input[src];
                    }
                }
                out[b * channels + c] = acc / area;
            }
        }

        self.upload_f32(output_shape, &out)
    }

    pub fn avgpool_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if input_shape.len() != 4
            || input_shape[1] != channels
            || input_shape[2] != input_height
            || input_shape[3] != input_width
        {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_backward_f32 input shape mismatch: expected [batch, {channels}, {input_height}, {input_width}], got {input_shape:?}"
            )));
        }
        let batch = input_shape[0];
        if output_grad.len() != batch * channels {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_backward_f32 output gradient length mismatch: expected {}, got {}",
                batch * channels,
                output_grad.len()
            )));
        }

        let output_grad = self.download_f32(output_grad)?;
        let area = (input_height * input_width) as f32;
        let mut input_grad = vec![0.0; input_shape.iter().product()];
        for b in 0..batch {
            for c in 0..channels {
                let grad = output_grad[b * channels + c] / area;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let dst = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        input_grad[dst] = grad;
                    }
                }
            }
        }

        self.upload_f32(input_shape, &input_grad)
    }

    pub fn crop_bilinear_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &CutileTensorF32,
        source_y: &CutileTensorF32,
        input_scales: &CutileTensorF32,
        input_offsets: &CutileTensorF32,
        _tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_crop_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            source_x.len(),
            source_y.len(),
            input_scales.len(),
            input_offsets.len(),
        )?;
        let out = self.zeroed_tensor_f32(layout.output_len)?;
        unsafe {
            cutile_darknet_kernels::crop_bilinear_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                source_x.arc().device_pointer(),
                source_y.arc().device_pointer(),
                input_scales.arc().device_pointer(),
                input_offsets.arc().device_pointer(),
                layout.output_len as i32,
                channels as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn crop_bilinear_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &CutileTensorF32,
        source_y: &CutileTensorF32,
        input_scales: &CutileTensorF32,
        _tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let output_shape = output_grad.shape_usize();
        let expected_input = output_shape[0]
            .checked_mul(channels)
            .and_then(|len| len.checked_mul(input_height))
            .and_then(|len| len.checked_mul(input_width))
            .ok_or_else(|| CutileBackendError::Shape("crop input shape overflow".to_owned()))?;
        let layout = validate_crop_layout(
            expected_input,
            &output_shape,
            channels,
            input_height,
            input_width,
            source_x.len(),
            source_y.len(),
            input_scales.len(),
            input_scales.len(),
        )?;
        if output_grad.len() != layout.output_len {
            return Err(CutileBackendError::Shape(format!(
                "crop output-gradient length mismatch: expected {}, got {}",
                layout.output_len,
                output_grad.len()
            )));
        }
        let input_len = shape_num_elements(input_shape)?;
        let expected_input = layout.batch * channels * input_height * input_width;
        if input_len != expected_input {
            return Err(CutileBackendError::Shape(format!(
                "crop_bilinear_backward_f32 input shape {:?} expects {input_len} elements, expected {expected_input}",
                input_shape
            )));
        }
        let out = self.zeroed_tensor_f32(input_len)?;
        unsafe {
            cutile_darknet_kernels::crop_bilinear_backward_flat(
                out.device_pointer(),
                output_grad.arc().device_pointer(),
                source_x.arc().device_pointer(),
                source_y.arc().device_pointer(),
                input_scales.arc().device_pointer(),
                layout.output_len as i32,
                channels as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(input_shape)?),
        })
    }

    pub fn vector_add_f32(
        &self,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if x.len() != y.len() {
            return Err(CutileBackendError::Shape(format!(
                "vector_add_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let y_flat = y.inner.view(&[y.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _, _) = cutile_darknet_kernels::vector_add(z, &x_flat, &y_flat)
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape(
                    &y.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                )?,
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
        if x.len() != y.len() {
            return Err(CutileBackendError::Shape(format!(
                "saxpy_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let y_flat = y.inner.view(&[y.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (_alpha, z, _, _) = cutile_darknet_kernels::saxpy(alpha, z, &x_flat, &y_flat)
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape(
                    &y.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                )?,
            ),
        })
    }

    pub fn sgd_update_f32(
        &self,
        weights: &CutileTensorF32,
        updates: &CutileTensorF32,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        _tile_len: usize,
    ) -> Result<(CutileTensorF32, CutileTensorF32), CutileBackendError> {
        if weights.len() != updates.len() {
            return Err(CutileBackendError::Shape(format!(
                "sgd_update_f32 shape mismatch: {:?} vs {:?}",
                weights.shape(),
                updates.shape()
            )));
        }

        let batch = batch.max(1) as f32;
        unsafe {
            cutile_darknet_kernels::sgd_update_flat(
                weights.inner.device_pointer(),
                updates.inner.device_pointer(),
                weights.len() as i32,
                lr / batch,
                momentum,
                decay * batch,
            )
        }
        .grid((grid_1d(weights.len(), POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok((weights.clone(), updates.clone()))
    }

    pub fn upsample_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        self.upsample_f32(
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            true,
            tile_len,
        )
    }

    fn zeroed_tensor_f32(&self, len: usize) -> Result<Tensor<f32>, CutileBackendError> {
        // `api::zeros` launches cuTile's generic `creation::full` kernel.  Keep
        // buffer initialization independent of that JIT-only helper so native
        // flat kernels remain usable under the strict AOT policy.
        let host_zeros = Arc::new(vec![0.0f32; len]);
        Ok(api::copy_host_vec_to_device(&host_zeros).sync_on(&self.stream)?)
    }
}

fn normalize_partition_tile_len(limit: usize, requested: usize) -> usize {
    let capped = requested.min(limit).max(1);
    if capped.is_power_of_two() {
        capped
    } else {
        capped.next_power_of_two() / 2
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

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::channel_scale_f32(self, x, scale, channels, channel_stride, config.tile_len)
    }

    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::bias_backward_f32(
            self,
            output_grad,
            channels,
            channel_stride,
            config.tile_len,
        )
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::activation_f32(self, x, activation, config.tile_len)
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::activation_backward_f32(
            self,
            activation_input,
            delta,
            activation,
            config.tile_len,
        )
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

    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        CutileBackend::route_backward_f32(
            self,
            output_grad,
            input_shapes,
            input_channel_offsets,
            input_channels,
            input_total_channels,
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_backward_input_f32(
            self,
            output_grad,
            weights,
            input_shape,
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_backward_weights_f32(
            self,
            x,
            output_grad,
            weight_shape,
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upsample_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            config.tile_len,
        )
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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::maxpool_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config.tile_len,
        )
    }

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
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::maxpool_backward_f32(
            self,
            x,
            output_grad,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config.tile_len,
        )
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
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::local_avgpool_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
        )
    }

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
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::local_avgpool_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
        )
    }

    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::avgpool_f32(self, x, output_shape, channels, input_height, input_width)
    }

    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::avgpool_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
        )
    }

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
        CutileBackend::crop_bilinear_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            input_offsets,
            config.tile_len,
        )
    }

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
        CutileBackend::crop_bilinear_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
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

    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        CutileBackend::sgd_update_f32(
            self,
            weights,
            updates,
            lr,
            momentum,
            decay,
            batch,
            config.tile_len,
        )
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

fn ensure_same_shape(op: &str, lhs: &[i32], rhs: &[i32]) -> Result<(), CutileBackendError> {
    if lhs != rhs {
        return Err(CutileBackendError::Shape(format!(
            "{op} shape mismatch: {lhs:?} vs {rhs:?}"
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
    if !x_len.is_multiple_of(channel_stride) {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 input length {x_len} is not divisible by channel_stride {channel_stride}"
        )));
    }

    let rows = x_len / channel_stride;
    if !rows.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 rows {rows} are not divisible by channels {channels}"
        )));
    }

    Ok((rows, channel_stride))
}

fn validate_bias_backward_layout(
    output_grad_len: usize,
    channels: usize,
    channel_stride: usize,
) -> Result<(usize, usize), CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "bias_backward_f32 requires channels > 0".to_owned(),
        ));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "bias_backward_f32 requires channel_stride > 0".to_owned(),
        ));
    }
    if !output_grad_len.is_multiple_of(channel_stride) {
        return Err(CutileBackendError::Shape(format!(
            "bias_backward_f32 output gradient length {output_grad_len} is not divisible by channel_stride {channel_stride}"
        )));
    }

    let rows = output_grad_len / channel_stride;
    if !rows.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "bias_backward_f32 rows {rows} are not divisible by channels {channels}"
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
        if !input_len.is_multiple_of(channel_stride) {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 input length {input_len} is not divisible by channel_stride {channel_stride}"
            )));
        }
        let rows = input_len / channel_stride;
        if !rows.is_multiple_of(total_input_channels) {
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
    batch: usize,
    channels_per_group: usize,
    filters_per_group: usize,
    input_len: usize,
    weight_len: usize,
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
    if !input_channels.is_multiple_of(groups) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input channels {input_channels} are not divisible by groups {groups}"
        )));
    }
    if !filters.is_multiple_of(groups) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 filters {filters} are not divisible by groups {groups}"
        )));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("conv2d_f32 input plane overflow".to_owned()))?;
    let input_batch_stride = input_channels.checked_mul(input_plane).ok_or_else(|| {
        CutileBackendError::Shape("conv2d_f32 input batch stride overflow".to_owned())
    })?;
    if !input_len.is_multiple_of(input_batch_stride) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input length {input_len} is not divisible by input batch stride {input_batch_stride}"
        )));
    }
    let batch = input_len / input_batch_stride;

    let channels_per_group = input_channels / groups;
    let filters_per_group = filters / groups;
    let weights_per_filter = channels_per_group
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(kernel_size))
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d_f32 weights-per-filter overflow".to_owned())
        })?;
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
        batch,
        channels_per_group,
        filters_per_group,
        input_len,
        weight_len,
        output_height: output_shape[output_shape.len() - 2],
        output_width: output_shape[output_shape.len() - 1],
        output_len,
    })
}

fn conv_output_dim(
    input: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> Result<usize, CutileBackendError> {
    if stride == 0 || kernel_size == 0 || dilation == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d backward requires non-zero kernel, stride, and dilation".to_owned(),
        ));
    }
    let kernel_extent = dilation
        .checked_mul(kernel_size.saturating_sub(1))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward kernel extent overflow".to_owned())
        })?;
    let padded = input
        .checked_add(padding.checked_mul(2).ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward padding overflow".to_owned())
        })?)
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward padded input overflow".to_owned())
        })?;
    if padded < kernel_extent {
        return Err(CutileBackendError::Shape(format!(
            "conv2d backward kernel extent {kernel_extent} exceeds padded input {padded}"
        )));
    }
    Ok((padded - kernel_extent) / stride + 1)
}

fn validate_conv2d_backward_input_layout(
    output_grad_len: usize,
    weight_len: usize,
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
) -> Result<Conv2dLayout, CutileBackendError> {
    let input_len = shape_num_elements(input_shape)?;
    let output_height = conv_output_dim(input_height, kernel_size, stride_y, dilation, padding)?;
    let output_width = conv_output_dim(input_width, kernel_size, stride_x, dilation, padding)?;
    let batch = input_len / input_channels / input_height / input_width;
    validate_conv2d_layout(
        input_len,
        weight_len,
        &[batch, filters, output_height, output_width],
        input_channels,
        input_height,
        input_width,
        filters,
        kernel_size,
        groups,
    )
    .and_then(|layout| {
        if output_grad_len != layout.output_len {
            Err(CutileBackendError::Shape(format!(
                "conv2d_backward_input_f32 output gradient length mismatch: expected {}, got {output_grad_len}",
                layout.output_len
            )))
        } else {
            Ok(layout)
        }
    })
}

fn validate_conv2d_backward_weights_layout(
    input_len: usize,
    output_grad_len: usize,
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
) -> Result<Conv2dLayout, CutileBackendError> {
    let weight_len = shape_num_elements(weight_shape)?;
    let output_height = conv_output_dim(input_height, kernel_size, stride_y, dilation, padding)?;
    let output_width = conv_output_dim(input_width, kernel_size, stride_x, dilation, padding)?;
    let batch = input_len / input_channels / input_height / input_width;
    validate_conv2d_layout(
        input_len,
        weight_len,
        &[batch, filters, output_height, output_width],
        input_channels,
        input_height,
        input_width,
        filters,
        kernel_size,
        groups,
    )
    .and_then(|layout| {
        if output_grad_len != layout.output_len {
            Err(CutileBackendError::Shape(format!(
                "conv2d_backward_weights_f32 output gradient length mismatch: expected {}, got {output_grad_len}",
                layout.output_len
            )))
        } else {
            Ok(layout)
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpsampleLayout {
    output_height: usize,
    output_width: usize,
    output_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaxPoolLayout {
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
    if !input_len.is_multiple_of(input_plane) {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 input length {input_len} is not divisible by input plane {input_plane}"
        )));
    }
    let batch_channels = input_len / input_plane;
    if !batch_channels.is_multiple_of(channels) {
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
            .ok_or_else(|| {
                CutileBackendError::Shape("upsample_f32 expanded plane overflow".to_owned())
            })?
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

fn validate_maxpool_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
) -> Result<MaxPoolLayout, CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires channels > 0".to_owned(),
        ));
    }
    if input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires non-zero input spatial dimensions".to_owned(),
        ));
    }
    if size == 0 || stride_x == 0 || stride_y == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires non-zero kernel size and strides".to_owned(),
        ));
    }
    if output_shape.len() != 4 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 output shape must be [batch, channels, height, width]".to_owned(),
        ));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("maxpool_f32 input plane overflow".to_owned()))?;
    if !input_len.is_multiple_of(input_plane) {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 input length {input_len} is not divisible by input plane {input_plane}"
        )));
    }
    let batch_channels = input_len / input_plane;
    if !batch_channels.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 batch*channels {batch_channels} is not divisible by channels {channels}"
        )));
    }
    let batch = batch_channels / channels;
    if output_shape[0] != batch {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output batch mismatch: expected {batch}, got {}",
            output_shape[0]
        )));
    }
    if output_shape[1] != channels {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output channels mismatch: expected {channels}, got {}",
            output_shape[1]
        )));
    }

    let output_len = shape_num_elements(output_shape)?;
    let output_height = output_shape[2];
    let output_width = output_shape[3];
    let expected_output_len = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or_else(|| {
            CutileBackendError::Shape("maxpool_f32 output length overflow".to_owned())
        })?;
    if output_len != expected_output_len {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output shape {:?} expects {output_len} elements, got {expected_output_len}",
            output_shape
        )));
    }

    Ok(MaxPoolLayout {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CropLayout {
    batch: usize,
    output_height: usize,
    output_width: usize,
    output_len: usize,
}

fn validate_crop_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    source_x_len: usize,
    source_y_len: usize,
    scale_len: usize,
    offset_len: usize,
) -> Result<CropLayout, CutileBackendError> {
    if output_shape.len() != 4 || channels == 0 || input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "crop_bilinear_f32 requires [batch, channels, height, width] and non-zero dimensions"
                .to_owned(),
        ));
    }
    if output_shape[1] != channels {
        return Err(CutileBackendError::Shape(format!(
            "crop output channels mismatch: expected {channels}, got {}",
            output_shape[1]
        )));
    }
    let batch = output_shape[0];
    let output_height = output_shape[2];
    let output_width = output_shape[3];
    let output_len = shape_num_elements(output_shape)?;
    let expected_input = batch
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(input_height))
        .and_then(|n| n.checked_mul(input_width))
        .ok_or_else(|| CutileBackendError::Shape("crop input shape overflow".to_owned()))?;
    if input_len != expected_input {
        return Err(CutileBackendError::Shape(format!(
            "crop input length mismatch: expected {expected_input}, got {input_len}"
        )));
    }
    let expected_coords = batch * output_height * output_width;
    if source_x_len != expected_coords || source_y_len != expected_coords {
        return Err(CutileBackendError::Shape(format!(
            "crop coordinate length mismatch: expected {expected_coords}, got source_x={source_x_len}, source_y={source_y_len}"
        )));
    }
    if scale_len != batch || offset_len != batch {
        return Err(CutileBackendError::Shape(format!(
            "crop adjustment length mismatch: expected {batch}, got scale={scale_len}, offset={offset_len}"
        )));
    }
    Ok(CropLayout {
        batch,
        output_height,
        output_width,
        output_len,
    })
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

    #[cutile::entry()]
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

    #[cutile::entry()]
    fn vector_add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry()]
    fn saxpy<const S: [i32; 1]>(
        alpha: f32,
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_alpha: Tile<f32, S> = alpha.broadcast(z.shape());
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_alpha * tile_x + tile_y);
    }

    #[cutile::entry()]
    fn bias_add_2d<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [-1] }>,
        channels: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_idx = pid.0;
        let channel_idx = row_idx % channels;
        let tile_x: Tile<f32, S> = load_tile_like(x, z);
        let bias_scalar = bias.load_tile(const_shape![1], [channel_idx]);
        let bias_tile: Tile<f32, S> = bias_scalar.reshape(const_shape![1, 1]).broadcast(z.shape());
        z.store(tile_x + bias_tile);
    }

    #[cutile::entry()]
    fn channel_scale_2d<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        scale: &Tensor<f32, { [-1] }>,
        channels: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_idx = pid.0;
        let channel_idx = row_idx % channels;
        let tile_x: Tile<f32, S> = load_tile_like(x, z);
        let scale_scalar = scale.load_tile(const_shape![1], [channel_idx]);
        let scale_tile: Tile<f32, S> = scale_scalar
            .reshape(const_shape![1, 1])
            .broadcast(z.shape());
        z.store(tile_x * scale_tile);
    }

    #[cutile::entry()]
    unsafe fn bias_backward_flat(
        bias_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        len: i32,
        channels: i32,
        channel_stride: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let grad_base: PointerTile<*mut f32, { [1] }> = grad_root.reshape(const_shape![1]);
        let grad_ptrs: PointerTile<*mut f32, { [128] }> = grad_base.broadcast(const_shape![128]);
        let bias_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(bias_grad_ptr);
        let bias_base: PointerTile<*mut f32, { [1] }> = bias_root.reshape(const_shape![1]);
        let bias_ptrs: PointerTile<*mut f32, { [128] }> = bias_base.broadcast(const_shape![128]);

        let len_tile: Tile<i32, { [128] }> = broadcast_scalar(len, const_shape![128]);
        let channel_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channel_stride, const_shape![128]);
        let channels_tile: Tile<i32, { [128] }> = broadcast_scalar(channels, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..len).step_by(step as usize) {
            let idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(idx, len_tile);
            let channel_idx = (idx / channel_stride_tile) % channels_tile;
            let grad_src: PointerTile<*mut f32, { [128] }> = grad_ptrs.offset_tile(idx);
            let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let grad_values: Tile<f32, { [128] }> = grad_tile.0;
            let bias_dst: PointerTile<*mut f32, { [128] }> = bias_ptrs.offset_tile(channel_idx);
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                bias_dst,
                grad_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(mask),
                None,
            );
        }
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn crop_bilinear_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        source_x_ptr: *mut f32,
        source_y_ptr: *mut f32,
        scale_ptr: *mut f32,
        offset_ptr: *mut f32,
        output_len: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let sx_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_x_ptr);
        let sx_base: PointerTile<*mut f32, { [1] }> = sx_root.reshape(const_shape![1]);
        let sx_ptrs: PointerTile<*mut f32, { [128] }> = sx_base.broadcast(const_shape![128]);
        let sy_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_y_ptr);
        let sy_base: PointerTile<*mut f32, { [1] }> = sy_root.reshape(const_shape![1]);
        let sy_ptrs: PointerTile<*mut f32, { [128] }> = sy_base.broadcast(const_shape![128]);
        let scale_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(scale_ptr);
        let scale_base: PointerTile<*mut f32, { [1] }> = scale_root.reshape(const_shape![1]);
        let scale_ptrs: PointerTile<*mut f32, { [128] }> = scale_base.broadcast(const_shape![128]);
        let offset_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(offset_ptr);
        let offset_base: PointerTile<*mut f32, { [1] }> = offset_root.reshape(const_shape![1]);
        let offset_ptrs: PointerTile<*mut f32, { [128] }> =
            offset_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let plane = output_height * output_width;
        let output_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(plane, const_shape![128]);
        let channel_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels * plane, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let input_width_limit: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_idx: Tile<i32, { [128] }> = output_idx / channel_plane_tile;
            let in_batch: Tile<i32, { [128] }> = output_idx % channel_plane_tile;
            let channel_idx: Tile<i32, { [128] }> = in_batch / output_plane_tile;
            let pixel_idx: Tile<i32, { [128] }> = in_batch % output_plane_tile;
            let coord_idx: Tile<i32, { [128] }> = batch_idx * output_plane_tile + pixel_idx;
            let source_x_src: PointerTile<*mut f32, { [128] }> = sx_ptrs.offset_tile(coord_idx);
            let source_x_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_x_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_y_src: PointerTile<*mut f32, { [128] }> = sy_ptrs.offset_tile(coord_idx);
            let source_y_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_y_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_x: Tile<f32, { [128] }> = source_x_load.0;
            let source_y: Tile<f32, { [128] }> = source_y_load.0;
            let ix: Tile<i32, { [128] }> = convert_tile(floor(source_x));
            let iy: Tile<i32, { [128] }> = convert_tile(floor(source_y));
            let dx: Tile<f32, { [128] }> = source_x - convert_tile(ix);
            let dy: Tile<f32, { [128] }> = source_y - convert_tile(iy);
            let one = broadcast_scalar(1.0f32, const_shape![128]);
            let w00 = (one - dy) * (one - dx);
            let w01 = dy * (one - dx);
            let w10 = (one - dy) * dx;
            let w11 = dy * dx;
            let scale_src: PointerTile<*mut f32, { [128] }> = scale_ptrs.offset_tile(batch_idx);
            let batch_scales: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(1.0f32),
                None,
                Latency::<0>,
            );
            let offset_src: PointerTile<*mut f32, { [128] }> = offset_ptrs.offset_tile(batch_idx);
            let batch_offsets: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                offset_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scale: Tile<f32, { [128] }> = batch_scales.0;
            let adjustment: Tile<f32, { [128] }> = batch_offsets.0;
            let input_base = batch_idx
                * broadcast_scalar(channels * input_height * input_width, const_shape![128])
                + channel_idx * input_plane_tile;
            let mut value: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            let valid00 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let offsets00 = input_base + iy * input_width_tile + ix;
            let safe_offsets00: Tile<i32, { [128] }> = select(valid00, offsets00, zero_i32);
            let sample00_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets00);
            let sample00: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample00_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid00),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample00.0 * w00;
            let iy1 = iy + broadcast_scalar(1i32, const_shape![128]);
            let valid01 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets01 = input_base + iy1 * input_width_tile + ix;
            let safe_offsets01: Tile<i32, { [128] }> = select(valid01, offsets01, zero_i32);
            let sample01_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets01);
            let sample01: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample01_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid01),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample01.0 * w01;
            let ix1 = ix + broadcast_scalar(1i32, const_shape![128]);
            let valid10 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let offsets10 = input_base + iy * input_width_tile + ix1;
            let safe_offsets10: Tile<i32, { [128] }> = select(valid10, offsets10, zero_i32);
            let sample10_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets10);
            let sample10: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample10_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid10),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample10.0 * w10;
            let valid11 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets11 = input_base + iy1 * input_width_tile + ix1;
            let safe_offsets11: Tile<i32, { [128] }> = select(valid11, offsets11, zero_i32);
            let sample11_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets11);
            let sample11: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample11_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid11),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample11.0 * w11;
            let dst = out_ptrs.offset_tile(output_idx);
            let _store = store_ptr_tko(
                dst,
                value * scale + adjustment,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn crop_bilinear_backward_flat(
        input_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        source_x_ptr: *mut f32,
        source_y_ptr: *mut f32,
        scale_ptr: *mut f32,
        output_len: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();
        let grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let grad_base: PointerTile<*mut f32, { [1] }> = grad_root.reshape(const_shape![1]);
        let grad_ptrs: PointerTile<*mut f32, { [128] }> = grad_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let sx_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_x_ptr);
        let sx_base: PointerTile<*mut f32, { [1] }> = sx_root.reshape(const_shape![1]);
        let sx_ptrs: PointerTile<*mut f32, { [128] }> = sx_base.broadcast(const_shape![128]);
        let sy_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_y_ptr);
        let sy_base: PointerTile<*mut f32, { [1] }> = sy_root.reshape(const_shape![1]);
        let sy_ptrs: PointerTile<*mut f32, { [128] }> = sy_base.broadcast(const_shape![128]);
        let scale_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(scale_ptr);
        let scale_base: PointerTile<*mut f32, { [1] }> = scale_root.reshape(const_shape![1]);
        let scale_ptrs: PointerTile<*mut f32, { [128] }> = scale_base.broadcast(const_shape![128]);
        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let plane = output_height * output_width;
        let output_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(plane, const_shape![128]);
        let channel_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels * plane, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let input_width_limit: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask = lt_tile(output_idx, output_len_tile);
            let batch_idx: Tile<i32, { [128] }> = output_idx / channel_plane_tile;
            let in_batch: Tile<i32, { [128] }> = output_idx % channel_plane_tile;
            let channel_idx: Tile<i32, { [128] }> = in_batch / output_plane_tile;
            let pixel_idx: Tile<i32, { [128] }> = in_batch % output_plane_tile;
            let coord_idx: Tile<i32, { [128] }> = batch_idx * output_plane_tile + pixel_idx;
            let source_x_ptr: PointerTile<*mut f32, { [128] }> = sx_ptrs.offset_tile(coord_idx);
            let source_x_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_x_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_x: Tile<f32, { [128] }> = source_x_load.0;
            let source_y_ptr: PointerTile<*mut f32, { [128] }> = sy_ptrs.offset_tile(coord_idx);
            let source_y_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_y_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_y: Tile<f32, { [128] }> = source_y_load.0;
            let ix: Tile<i32, { [128] }> = convert_tile(floor(source_x));
            let iy: Tile<i32, { [128] }> = convert_tile(floor(source_y));
            let dx = source_x - convert_tile(ix);
            let dy = source_y - convert_tile(iy);
            let one = broadcast_scalar(1.0f32, const_shape![128]);
            let w00 = (one - dy) * (one - dx);
            let w01 = dy * (one - dx);
            let w10 = (one - dy) * dx;
            let w11 = dy * dx;
            let grad_ptr: PointerTile<*mut f32, { [128] }> = grad_ptrs.offset_tile(output_idx);
            let grad_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scale_ptr: PointerTile<*mut f32, { [128] }> = scale_ptrs.offset_tile(batch_idx);
            let scale_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(1.0f32),
                None,
                Latency::<0>,
            );
            let grad: Tile<f32, { [128] }> = grad_load.0 * scale_load.0;
            let input_base = batch_idx
                * broadcast_scalar(channels * input_height * input_width, const_shape![128])
                + channel_idx * input_plane_tile;
            let ix1 = ix + broadcast_scalar(1i32, const_shape![128]);
            let iy1 = iy + broadcast_scalar(1i32, const_shape![128]);
            let valid00 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let valid01 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let valid10 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let valid11 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let off00 = input_base + iy * input_width_tile + ix;
            let off01 = input_base + iy1 * input_width_tile + ix;
            let off10 = input_base + iy * input_width_tile + ix1;
            let off11 = input_base + iy1 * input_width_tile + ix1;
            let dst00: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid00, off00, zero_i32));
            let _a00: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst00,
                grad * w00,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid00),
                None,
            );
            let dst01: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid01, off01, zero_i32));
            let _a01: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst01,
                grad * w01,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid01),
                None,
            );
            let dst10: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid10, off10, zero_i32));
            let _a10: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst10,
                grad * w10,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid10),
                None,
            );
            let dst11: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid11, off11, zero_i32));
            let _a11: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst11,
                grad * w11,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid11),
                None,
            );
        }
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn sgd_update_flat(
        weights_ptr: *mut f32,
        updates_ptr: *mut f32,
        len: i32,
        lr_over_batch: f32,
        momentum: f32,
        decay_times_batch: f32,
    ) {
        let pid = get_tile_block_id();
        let base_offset = pid.0 * 128i32;
        let idx: Tile<i32, { [128] }> =
            iota(const_shape![128]) + broadcast_scalar(base_offset, const_shape![128]);
        let mask: Tile<bool, { [128] }> = lt_tile(idx, broadcast_scalar(len, const_shape![128]));

        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weights_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);
        let weight_ptrs_offset: PointerTile<*mut f32, { [128] }> = weight_ptrs.offset_tile(idx);

        let update_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(updates_ptr);
        let update_base: PointerTile<*mut f32, { [1] }> = update_root.reshape(const_shape![1]);
        let update_ptrs: PointerTile<*mut f32, { [128] }> =
            update_base.broadcast(const_shape![128]);
        let update_ptrs_offset: PointerTile<*mut f32, { [128] }> = update_ptrs.offset_tile(idx);

        let loaded_weights: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            weight_ptrs_offset,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        let loaded_updates: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            update_ptrs_offset,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );

        let weights_tile = loaded_weights.0;
        let updates_tile = loaded_updates.0;
        let decayed_updates =
            updates_tile - broadcast_scalar(decay_times_batch, const_shape![128]) * weights_tile;
        let new_weights =
            weights_tile + broadcast_scalar(lr_over_batch, const_shape![128]) * decayed_updates;
        let new_updates = broadcast_scalar(momentum, const_shape![128]) * decayed_updates;

        let _store_weights = store_ptr_tko(
            weight_ptrs_offset,
            new_weights,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
        let _store_updates = store_ptr_tko(
            update_ptrs_offset,
            new_updates,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    fn activation_linear<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        z.store(tile_x);
    }

    #[cutile::entry()]
    fn activation_logistic<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(one / (one + exp_neg_x));
    }

    #[cutile::entry()]
    fn activation_relu<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(tile_x, zero));
    }

    #[cutile::entry()]
    fn activation_leaky<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let slope: Tile<f32, S> = constant(0.1f32, z.shape());
        z.store(select(gt_tile(tile_x, zero), tile_x, slope * tile_x));
    }

    #[cutile::entry()]
    fn activation_mish<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let exp_x: Tile<f32, S> = exp(tile_x);
        let softplus: Tile<f32, S> = log(one + exp_x);
        let softplus_tanh: Tile<f32, S> = tanh(softplus);
        z.store(tile_x * softplus_tanh);
    }

    #[cutile::entry()]
    fn activation_swish<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(tile_x / (one + exp_neg_x));
    }

    #[cutile::entry()]
    fn activation_tanh<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let tanh_x: Tile<f32, S> = tanh(tile_x);
        z.store(tanh_x);
    }

    #[cutile::entry()]
    fn activation_backward_linear<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        _x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_delta = load_tile_like(delta, z);
        z.store(tile_delta);
    }

    #[cutile::entry()]
    fn activation_backward_logistic<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        let sigmoid: Tile<f32, S> = one / (one + exp_neg_x);
        z.store(tile_delta * sigmoid * (one - sigmoid));
    }

    #[cutile::entry()]
    fn activation_backward_relu<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let grad = select(gt_tile(tile_x, zero), one, zero);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_leaky<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let slope: Tile<f32, S> = constant(0.1f32, z.shape());
        let grad = select(gt_tile(tile_x, zero), one, slope);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_mish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let threshold: Tile<f32, S> = constant(20.0f32, z.shape());
        let neg_threshold: Tile<f32, S> = constant(-20.0f32, z.shape());
        let exp_x: Tile<f32, S> = exp(tile_x);
        let log_term: Tile<f32, S> = log(one + exp_x);
        let softplus: Tile<f32, S> = select(
            gt_tile(tile_x, threshold),
            tile_x,
            select(lt_tile(tile_x, neg_threshold), exp_x, log_term),
        );
        let neg_softplus: Tile<f32, S> = negf(softplus);
        let exp_neg_softplus: Tile<f32, S> = exp(neg_softplus);
        let grad_sp: Tile<f32, S> = one - exp_neg_softplus;
        let tsp: Tile<f32, S> = tanh(softplus);
        let grad_tsp: Tile<f32, S> = (one - tsp * tsp) * grad_sp;
        let grad: Tile<f32, S> = tile_x * grad_tsp + tsp;
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_swish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        let sigmoid: Tile<f32, S> = one / (one + exp_neg_x);
        let grad: Tile<f32, S> = sigmoid + tile_x * sigmoid * (one - sigmoid);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_tanh<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let tanh_x: Tile<f32, S> = tanh(tile_x);
        z.store(tile_delta * (one - tanh_x * tanh_x));
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

            let src_offsets = batch_idx * source_batch_stride
                + source_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let src: PointerTile<*mut f32, { [128] }> = src_ptrs.offset_tile(src_offsets);
            let src_values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let src_tile: Tile<f32, { [128] }> = src_values.0;

            let dst_offsets = batch_idx * output_batch_stride
                + output_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let dst: PointerTile<*mut f32, { [128] }> = dst_ptrs.offset_tile(dst_offsets);
            let _store_token: Token = store_ptr_tko(
                dst,
                src_tile,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn route_scatter_flat(
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
        let source_selected_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_channels * channel_stride, const_shape![128]);
        let source_group_stride: Tile<i32, { [128] }> =
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
            let batch_idx = local_idx / source_selected_stride;
            let within_batch = local_idx % source_selected_stride;
            let channel_idx = within_batch / channel_stride_tile;
            let elem_idx = within_batch % channel_stride_tile;

            let src_offsets = batch_idx * source_group_stride
                + source_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let src: PointerTile<*mut f32, { [128] }> = src_ptrs.offset_tile(src_offsets);
            let src_values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let src_tile: Tile<f32, { [128] }> = src_values.0;

            let dst_offsets = batch_idx * output_batch_stride
                + output_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let dst: PointerTile<*mut f32, { [128] }> = dst_ptrs.offset_tile(dst_offsets);
            let _store_token: Token = store_ptr_tko(
                dst,
                src_tile,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
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

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
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
            let input_offsets = batch_channel
                * broadcast_scalar(input_height * input_width, const_shape![128])
                + input_y * input_width_tile
                + input_x;
            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(input_offsets);
            let values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                scaled_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
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
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
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
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_offsets);
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst,
                scaled_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(mask),
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn maxpool_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        size: i32,
        stride_x: i32,
        stride_y: i32,
        pad_offset: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let mut acc: Tile<f32, { [128] }> = constant(-1.0e30f32, const_shape![128]);

            for ky in 0i32..size {
                let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                let input_y = output_y * broadcast_scalar(stride_y, const_shape![128]) + ky_tile
                    - broadcast_scalar(pad_offset, const_shape![128]);
                for kx in 0i32..size {
                    let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                    let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                        + kx_tile
                        - broadcast_scalar(pad_offset, const_shape![128]);
                    let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let input_y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, input_height_tile);
                    let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                    let input_x_lt_width: Tile<bool, { [128] }> =
                        lt_tile(input_x, input_width_tile);
                    let valid: Tile<bool, { [128] }> = output_mask
                        & input_y_ge_zero
                        & input_y_lt_height
                        & input_x_ge_zero
                        & input_x_lt_width;
                    let input_offsets =
                        batch_channel * input_plane_tile + input_y * input_width_tile + input_x;
                    let safe_input_offsets = select(valid, input_offsets, zero_i32);
                    let src: PointerTile<*mut f32, { [128] }> =
                        in_ptrs.offset_tile(safe_input_offsets);
                    let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                        src,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(valid),
                        Some(-1.0e30f32),
                        None,
                        Latency::<0>,
                    );
                    let values: Tile<f32, { [128] }> = input_tile.0;
                    let better: Tile<bool, { [128] }> = gt_tile(values, acc);
                    acc = select(better, values, acc);
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn maxpool_backward_flat(
        input_grad_ptr: *mut f32,
        input_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        size: i32,
        stride_x: i32,
        stride_y: i32,
        pad_offset: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);
        let input_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_grad_base: PointerTile<*mut f32, { [1] }> =
            input_grad_root.reshape(const_shape![1]);
        let input_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            input_grad_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let false_tile: Tile<bool, { [128] }> = broadcast_scalar(false, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let mut best_value: Tile<f32, { [128] }> = constant(-1.0e30f32, const_shape![128]);
            let mut best_offset: Tile<i32, { [128] }> = zero_i32;
            let mut best_found: Tile<bool, { [128] }> = false_tile;

            for ky in 0i32..size {
                let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                let input_y = output_y * broadcast_scalar(stride_y, const_shape![128]) + ky_tile
                    - broadcast_scalar(pad_offset, const_shape![128]);
                for kx in 0i32..size {
                    let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                    let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                        + kx_tile
                        - broadcast_scalar(pad_offset, const_shape![128]);
                    let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let input_y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, input_height_tile);
                    let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                    let input_x_lt_width: Tile<bool, { [128] }> =
                        lt_tile(input_x, input_width_tile);
                    let valid: Tile<bool, { [128] }> = output_mask
                        & input_y_ge_zero
                        & input_y_lt_height
                        & input_x_ge_zero
                        & input_x_lt_width;
                    let input_offsets =
                        batch_channel * input_plane_tile + input_y * input_width_tile + input_x;
                    let safe_input_offsets = select(valid, input_offsets, zero_i32);
                    let src: PointerTile<*mut f32, { [128] }> =
                        input_ptrs.offset_tile(safe_input_offsets);
                    let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                        src,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(valid),
                        Some(-1.0e30f32),
                        None,
                        Latency::<0>,
                    );
                    let values: Tile<f32, { [128] }> = input_tile.0;
                    let better: Tile<bool, { [128] }> = valid & gt_tile(values, best_value);
                    best_value = select(better, values, best_value);
                    best_offset = select(better, input_offsets, best_offset);
                    best_found = select(better, valid, best_found);
                }
            }

            let grad_src: PointerTile<*mut f32, { [128] }> =
                output_grad_ptrs.offset_tile(output_idx);
            let output_grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let grad_values: Tile<f32, { [128] }> = output_grad_tile.0;
            let grad_dst: PointerTile<*mut f32, { [128] }> =
                input_grad_ptrs.offset_tile(best_offset);
            let write_mask = output_mask & best_found;
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                grad_dst,
                grad_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(write_mask),
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
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
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
                let input_channel = group_idx * channels_per_group_tile + ic_tile;
                for ky in 0i32..kernel_size {
                    let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                    let input_y = output_y * broadcast_scalar(stride_y, const_shape![128])
                        + ky_tile * broadcast_scalar(dilation, const_shape![128])
                        - broadcast_scalar(padding, const_shape![128]);
                    for kx in 0i32..kernel_size {
                        let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                        let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                            + kx_tile * broadcast_scalar(dilation, const_shape![128])
                            - broadcast_scalar(padding, const_shape![128]);
                        let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                        let input_y_lt_height: Tile<bool, { [128] }> =
                            lt_tile(input_y, input_height_tile);
                        let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                        let input_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(input_x, input_width_limit_tile);
                        let input_valid: Tile<bool, { [128] }> = output_mask
                            & input_y_ge_zero
                            & input_y_lt_height
                            & input_x_ge_zero
                            & input_x_lt_width;
                        let input_offsets = batch_idx
                            * broadcast_scalar(
                                input_channels * input_height * input_width,
                                const_shape![128],
                            )
                            + input_channel * input_plane_tile
                            + input_y * input_width_tile
                            + input_x;
                        let safe_input_offsets = select(input_valid, input_offsets, zero_i32);
                        let src: PointerTile<*mut f32, { [128] }> =
                            input_ptrs.offset_tile(safe_input_offsets);
                        let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(input_valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let weight_offsets = ((filter_idx * channels_per_group_tile + ic_tile)
                            * broadcast_scalar(kernel_size, const_shape![128])
                            + ky_tile)
                            * broadcast_scalar(kernel_size, const_shape![128])
                            + kx_tile;
                        let weight_ptrs_offset: PointerTile<*mut f32, { [128] }> =
                            weight_ptrs.offset_tile(weight_offsets);
                        let weight_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            weight_ptrs_offset,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(output_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
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
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_backward_input_flat(
        input_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        weight_ptr: *mut f32,
        input_len: i32,
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

        let input_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_grad_base: PointerTile<*mut f32, { [1] }> =
            input_grad_root.reshape(const_shape![1]);
        let input_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            input_grad_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);
        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);

        let input_len_tile: Tile<i32, { [128] }> = broadcast_scalar(input_len, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_channel_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(
            input_channels * input_height * input_width,
            const_shape![128],
        );
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let kernel_size_tile: Tile<i32, { [128] }> =
            broadcast_scalar(kernel_size, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..input_len).step_by(step as usize) {
            let input_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let input_mask: Tile<bool, { [128] }> = lt_tile(input_idx, input_len_tile);

            let batch_idx = input_idx / input_channel_plane_tile;
            let batch_rem = input_idx % input_channel_plane_tile;
            let input_channel = batch_rem / input_plane_tile;
            let within_plane = batch_rem % input_plane_tile;
            let input_y = within_plane / input_width_tile;
            let input_x = within_plane % input_width_tile;
            let group_idx = input_channel / channels_per_group_tile;
            let input_channel_in_group = input_channel % channels_per_group_tile;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for filter_in_group in 0i32..filters_per_group {
                let filter_in_group_tile: Tile<i32, { [128] }> =
                    broadcast_scalar(filter_in_group, const_shape![128]);
                let filter_idx = group_idx * filters_per_group_tile + filter_in_group_tile;
                for ky in 0i32..kernel_size {
                    let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                    let oy_num = input_y + broadcast_scalar(padding, const_shape![128])
                        - ky_tile * broadcast_scalar(dilation, const_shape![128]);
                    let oy_ge_zero: Tile<bool, { [128] }> = ge_tile(oy_num, zero_i32);
                    let oy_on_stride: Tile<bool, { [128] }> = eq_tile(
                        oy_num % broadcast_scalar(stride_y, const_shape![128]),
                        zero_i32,
                    );
                    let oy_valid: Tile<bool, { [128] }> = oy_ge_zero & oy_on_stride;
                    let output_y = oy_num / broadcast_scalar(stride_y, const_shape![128]);
                    for kx in 0i32..kernel_size {
                        let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                        let ox_num = input_x + broadcast_scalar(padding, const_shape![128])
                            - kx_tile * broadcast_scalar(dilation, const_shape![128]);
                        let ox_ge_zero: Tile<bool, { [128] }> = ge_tile(ox_num, zero_i32);
                        let ox_on_stride: Tile<bool, { [128] }> = eq_tile(
                            ox_num % broadcast_scalar(stride_x, const_shape![128]),
                            zero_i32,
                        );
                        let ox_valid: Tile<bool, { [128] }> = ox_ge_zero & ox_on_stride;
                        let output_x = ox_num / broadcast_scalar(stride_x, const_shape![128]);
                        let output_y_ge_zero: Tile<bool, { [128] }> = ge_tile(output_y, zero_i32);
                        let output_y_lt_height: Tile<bool, { [128] }> =
                            lt_tile(output_y, broadcast_scalar(output_height, const_shape![128]));
                        let output_x_ge_zero: Tile<bool, { [128] }> = ge_tile(output_x, zero_i32);
                        let output_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(output_x, broadcast_scalar(output_width, const_shape![128]));
                        let valid: Tile<bool, { [128] }> = input_mask
                            & oy_valid
                            & ox_valid
                            & output_y_ge_zero
                            & output_y_lt_height
                            & output_x_ge_zero
                            & output_x_lt_width;

                        let grad_offsets = batch_idx
                            * broadcast_scalar(
                                filters * output_height * output_width,
                                const_shape![128],
                            )
                            + filter_idx * output_plane_tile
                            + output_y * output_width_tile
                            + output_x;
                        let safe_grad_offsets = select(valid, grad_offsets, zero_i32);
                        let grad_src: PointerTile<*mut f32, { [128] }> =
                            output_grad_ptrs.offset_tile(safe_grad_offsets);
                        let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            grad_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let weight_offsets = ((filter_idx * channels_per_group_tile
                            + input_channel_in_group)
                            * kernel_size_tile
                            + ky_tile)
                            * kernel_size_tile
                            + kx_tile;
                        let weight_src: PointerTile<*mut f32, { [128] }> =
                            weight_ptrs.offset_tile(weight_offsets);
                        let weight_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            weight_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(input_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );
                        acc = acc + grad_tile.0 * weight_tile.0;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = input_grad_ptrs.offset_tile(input_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(input_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_backward_weights_flat(
        weight_grad_ptr: *mut f32,
        input_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        weight_len: i32,
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
        batch: i32,
        channels_per_group: i32,
        filters_per_group: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let weight_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_grad_ptr);
        let weight_grad_base: PointerTile<*mut f32, { [1] }> =
            weight_grad_root.reshape(const_shape![1]);
        let weight_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_grad_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);

        let weight_len_tile: Tile<i32, { [128] }> = broadcast_scalar(weight_len, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let kernel_size_tile: Tile<i32, { [128] }> =
            broadcast_scalar(kernel_size, const_shape![128]);
        let input_batch_stride_tile: Tile<i32, { [128] }> = broadcast_scalar(
            input_channels * input_height * input_width,
            const_shape![128],
        );
        let output_batch_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters * output_height * output_width, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..weight_len).step_by(step as usize) {
            let weight_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let weight_mask: Tile<bool, { [128] }> = lt_tile(weight_idx, weight_len_tile);

            let filter_idx =
                weight_idx / (channels_per_group_tile * kernel_size_tile * kernel_size_tile);
            let filter_rem =
                weight_idx % (channels_per_group_tile * kernel_size_tile * kernel_size_tile);
            let input_channel_in_group = filter_rem / (kernel_size_tile * kernel_size_tile);
            let kernel_rem = filter_rem % (kernel_size_tile * kernel_size_tile);
            let ky = kernel_rem / kernel_size_tile;
            let kx = kernel_rem % kernel_size_tile;
            let group_idx = filter_idx / filters_per_group_tile;
            let input_channel = group_idx * channels_per_group_tile + input_channel_in_group;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for batch_idx_scalar in 0i32..batch {
                let batch_idx: Tile<i32, { [128] }> =
                    broadcast_scalar(batch_idx_scalar, const_shape![128]);
                for output_y_scalar in 0i32..output_height {
                    let output_y: Tile<i32, { [128] }> =
                        broadcast_scalar(output_y_scalar, const_shape![128]);
                    let input_y = output_y * broadcast_scalar(stride_y, const_shape![128])
                        + ky * broadcast_scalar(dilation, const_shape![128])
                        - broadcast_scalar(padding, const_shape![128]);
                    let y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, broadcast_scalar(input_height, const_shape![128]));
                    let y_valid: Tile<bool, { [128] }> = y_ge_zero & y_lt_height;
                    for output_x_scalar in 0i32..output_width {
                        let output_x: Tile<i32, { [128] }> =
                            broadcast_scalar(output_x_scalar, const_shape![128]);
                        let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                            + kx * broadcast_scalar(dilation, const_shape![128])
                            - broadcast_scalar(padding, const_shape![128]);
                        let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                        let input_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(input_x, broadcast_scalar(input_width, const_shape![128]));
                        let valid: Tile<bool, { [128] }> =
                            weight_mask & y_valid & input_x_ge_zero & input_x_lt_width;

                        let input_offsets = batch_idx * input_batch_stride_tile
                            + input_channel * input_plane_tile
                            + input_y * input_width_tile
                            + input_x;
                        let safe_input_offsets = select(valid, input_offsets, zero_i32);
                        let input_src: PointerTile<*mut f32, { [128] }> =
                            input_ptrs.offset_tile(safe_input_offsets);
                        let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            input_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let grad_offsets = batch_idx * output_batch_stride_tile
                            + filter_idx * output_plane_tile
                            + output_y * output_width_tile
                            + output_x;
                        let output_grad_src: PointerTile<*mut f32, { [128] }> =
                            output_grad_ptrs.offset_tile(grad_offsets);
                        let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            output_grad_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(weight_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );
                        acc = acc + input_tile.0 * grad_tile.0;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = weight_grad_ptrs.offset_tile(weight_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(weight_mask),
                None,
                Latency::<0>,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use darknet_cuda::F32ActivationKind;

    use super::{
        CutileBackend, CutileBackendError, DEFAULT_TILE_1D, grid_1d, validate_bias_add_layout,
        validate_conv2d_layout, validate_maxpool_layout, validate_route_concat_layout,
        validate_upsample_layout,
    };

    struct ConvCase {
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        filters: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        input: Vec<f32>,
        weights: Vec<f32>,
        output_gradient: Vec<f32>,
    }

    /// Independent analytical NCHW convolution oracle used by the device
    /// differential test.  It applies the chain rule directly to the same
    /// cross-correlation convention as Darknet; it never perturbs parameters.
    fn analytical_convolution_reference(
        case: &ConvCase,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
        let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
        let channels_per_group = case.channels / case.groups;
        let filters_per_group = case.filters / case.groups;
        let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
            (((batch * case.channels + channel) * case.height) + y) * case.width + x
        };
        let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
            (((batch * case.filters + filter) * output_height) + y) * output_width + x
        };
        let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
            (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
        };
        let mut output = vec![0.0; case.batch * case.filters * output_height * output_width];
        let mut input_gradient = vec![0.0; case.input.len()];
        let mut weight_gradient = vec![0.0; case.weights.len()];
        let mut bias_gradient = vec![0.0; case.filters];

        for batch in 0..case.batch {
            for (filter, bias) in bias_gradient.iter_mut().enumerate() {
                let group = filter / filters_per_group;
                for y in 0..output_height {
                    for x in 0..output_width {
                        let mut value = 0.0;
                        for channel_in_group in 0..channels_per_group {
                            let channel = group * channels_per_group + channel_in_group;
                            for ky in 0..case.kernel {
                                for kx in 0..case.kernel {
                                    let src_y = y * case.stride + ky;
                                    let src_x = x * case.stride + kx;
                                    if src_y < case.padding
                                        || src_x < case.padding
                                        || src_y >= case.height + case.padding
                                        || src_x >= case.width + case.padding
                                    {
                                        continue;
                                    }
                                    let src_y = src_y - case.padding;
                                    let src_x = src_x - case.padding;
                                    value += case.input[input_index(batch, channel, src_y, src_x)]
                                        * case.weights
                                            [weight_index(filter, channel_in_group, ky, kx)];
                                }
                            }
                        }
                        let output_offset = output_index(batch, filter, y, x);
                        output[output_offset] = value;
                        let grad = case.output_gradient[output_offset];
                        *bias += grad;
                        for channel_in_group in 0..channels_per_group {
                            let channel = group * channels_per_group + channel_in_group;
                            for ky in 0..case.kernel {
                                for kx in 0..case.kernel {
                                    let src_y = y * case.stride + ky;
                                    let src_x = x * case.stride + kx;
                                    if src_y < case.padding
                                        || src_x < case.padding
                                        || src_y >= case.height + case.padding
                                        || src_x >= case.width + case.padding
                                    {
                                        continue;
                                    }
                                    let src_y = src_y - case.padding;
                                    let src_x = src_x - case.padding;
                                    let input_offset = input_index(batch, channel, src_y, src_x);
                                    let weight_offset =
                                        weight_index(filter, channel_in_group, ky, kx);
                                    weight_gradient[weight_offset] +=
                                        case.input[input_offset] * grad;
                                    input_gradient[input_offset] +=
                                        case.weights[weight_offset] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        (output, input_gradient, weight_gradient, bias_gradient)
    }

    fn assert_device_close(name: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len(), "{name} length mismatch");
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 3.0e-4 + 3.0e-4 * expected.abs();
            assert!(
                (expected - actual).abs() <= tolerance,
                "{name}[{index}] differs: expected {expected:.8e}, actual {actual:.8e}, tolerance {tolerance:.8e}"
            );
        }
    }

    #[test]
    fn computes_grid_1d() {
        assert_eq!(grid_1d(1024, DEFAULT_TILE_1D), 4);
        assert_eq!(grid_1d(1025, DEFAULT_TILE_1D), 5);
    }

    #[test]
    fn validates_bias_add_layout_for_batched_channels() {
        assert_eq!(
            validate_bias_add_layout(24, 3, 3, 4).expect("valid"),
            (6, 4)
        );
    }

    #[test]
    fn rejects_bias_length_mismatch() {
        let err = validate_bias_add_layout(12, 2, 3, 4).expect_err("expected mismatch");
        assert!(
            matches!(err, CutileBackendError::Shape(message) if message.contains("bias length mismatch"))
        );
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
        let layout =
            validate_route_concat_layout(&[12, 4], &[1, 0], &[2, 1], &[3, 1], &[1, 3, 2, 2], 4)
                .expect("valid route layout");
        assert_eq!(layout.batch, 1);
        assert_eq!(layout.total_channels, 3);
        assert_eq!(layout.output_len, 12);
    }

    #[test]
    fn validates_upsample_layout() {
        let layout = validate_upsample_layout(4, &[1, 1, 4, 4], 1, 2, 2, 2, false).expect("valid");
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

    #[test]
    fn validates_maxpool_layout() {
        let layout = validate_maxpool_layout(16, &[1, 1, 2, 2], 1, 4, 4, 2, 2, 2).expect("valid");
        assert_eq!(layout.output_height, 2);
        assert_eq!(layout.output_width, 2);
        assert_eq!(layout.output_len, 4);
    }

    #[test]
    #[ignore = "requires a CUDA device; set DARKNET_CUTILE_GPU_TEST=1 and run ignored tests"]
    fn device_convolution_backward_and_sgd_match_reference() {
        assert!(
            env::var_os("DARKNET_CUTILE_GPU_TEST").is_some(),
            "set DARKNET_CUTILE_GPU_TEST=1 to run the device test"
        );
        let case = ConvCase {
            batch: 2,
            channels: 2,
            height: 3,
            width: 3,
            filters: 2,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
            input: (0..36).map(|index| (index as f32 - 13.0) / 11.0).collect(),
            weights: (0..36).map(|index| 0.02 * (index as f32 - 17.0)).collect(),
            output_gradient: (0..36).map(|index| 0.03 * (index as f32 - 8.0)).collect(),
        };
        let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
        let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
        let (
            expected_output,
            expected_input_gradient,
            expected_weight_gradient,
            expected_bias_gradient,
        ) = analytical_convolution_reference(&case);
        let backend = CutileBackend::new(0).expect("CUDA device should initialize");
        let input = backend
            .upload_f32(
                &[case.batch, case.channels, case.height, case.width],
                &case.input,
            )
            .expect("input upload");
        let weights = backend
            .upload_f32(
                &[
                    case.filters,
                    case.channels / case.groups,
                    case.kernel,
                    case.kernel,
                ],
                &case.weights,
            )
            .expect("weight upload");
        let output_gradient = backend
            .upload_f32(
                &[case.batch, case.filters, output_height, output_width],
                &case.output_gradient,
            )
            .expect("output-gradient upload");
        let output = backend
            .conv2d_f32(
                &input,
                &weights,
                &[case.batch, case.filters, output_height, output_width],
                case.channels,
                case.height,
                case.width,
                case.filters,
                case.kernel,
                case.stride,
                case.stride,
                1,
                case.padding,
                case.groups,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile convolution forward");
        let input_gradient = backend
            .conv2d_backward_input_f32(
                &output_gradient,
                &weights,
                &[case.batch, case.channels, case.height, case.width],
                case.channels,
                case.height,
                case.width,
                case.filters,
                case.kernel,
                case.stride,
                case.stride,
                1,
                case.padding,
                case.groups,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile input gradient");
        let weight_gradient = backend
            .conv2d_backward_weights_f32(
                &input,
                &output_gradient,
                &[
                    case.filters,
                    case.channels / case.groups,
                    case.kernel,
                    case.kernel,
                ],
                case.channels,
                case.height,
                case.width,
                case.filters,
                case.kernel,
                case.stride,
                case.stride,
                1,
                case.padding,
                case.groups,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile weight gradient");
        let bias_gradient = backend
            .bias_backward_f32(
                &output_gradient,
                case.filters,
                output_height * output_width,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile bias gradient");

        assert_device_close(
            "cuTile convolution output",
            &expected_output,
            &backend.download_f32(&output).expect("output download"),
        );
        assert_device_close(
            "cuTile input gradient",
            &expected_input_gradient,
            &backend
                .download_f32(&input_gradient)
                .expect("input-gradient download"),
        );
        assert_device_close(
            "cuTile weight gradient",
            &expected_weight_gradient,
            &backend
                .download_f32(&weight_gradient)
                .expect("weight-gradient download"),
        );
        assert_device_close(
            "cuTile bias gradient",
            &expected_bias_gradient,
            &backend
                .download_f32(&bias_gradient)
                .expect("bias-gradient download"),
        );

        // Feed the negative mathematical gradient into cuTile's Darknet SGD
        // kernel.  With zero momentum/decay the update must be w - lr*g/batch.
        let descent_weight_update = backend
            .upload_f32(
                &[case.weights.len()],
                &expected_weight_gradient
                    .iter()
                    .map(|gradient| -*gradient)
                    .collect::<Vec<_>>(),
            )
            .expect("weight update upload");
        let descent_bias_update = backend
            .upload_f32(
                &[case.filters],
                &expected_bias_gradient
                    .iter()
                    .map(|gradient| -*gradient)
                    .collect::<Vec<_>>(),
            )
            .expect("bias update upload");
        let (updated_weights, _) = backend
            .sgd_update_f32(
                &weights,
                &descent_weight_update,
                1.0e-3,
                0.0,
                0.0,
                case.batch,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile weight SGD update");
        let biases = backend
            .upload_f32(&[case.filters], &[0.11, -0.19])
            .expect("bias upload");
        let (updated_biases, _) = backend
            .sgd_update_f32(
                &biases,
                &descent_bias_update,
                1.0e-3,
                0.0,
                0.0,
                case.batch,
                DEFAULT_TILE_1D,
            )
            .expect("cuTile bias SGD update");
        let expected_updated_weights = case
            .weights
            .iter()
            .zip(&expected_weight_gradient)
            .map(|(weight, gradient)| weight - 1.0e-3 * gradient / case.batch as f32)
            .collect::<Vec<_>>();
        let expected_updated_biases = [0.11f32, -0.19f32]
            .iter()
            .zip(&expected_bias_gradient)
            .map(|(bias, gradient)| bias - 1.0e-3 * gradient / case.batch as f32)
            .collect::<Vec<_>>();
        assert_device_close(
            "cuTile convolution SGD weights",
            &expected_updated_weights,
            &backend
                .download_f32(&updated_weights)
                .expect("updated-weight download"),
        );
        assert_device_close(
            "cuTile convolution SGD biases",
            &expected_updated_biases,
            &backend
                .download_f32(&updated_biases)
                .expect("updated-bias download"),
        );
    }
}
