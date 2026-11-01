//! GPU executor: runs a [`Model`] on the AMD GPU with the hip-tile YOLO kernels.
//!
//! Weights are uploaded once; every layer output has its own preallocated device buffer.
//! Route group slices are zero-copy views, concatenation uses `launch_concat`; `[yolo]` layers
//! alias their input. All launches go on one stream, with one sync at the end of a forward.

use crate::error::{Error, Result};
use crate::model::{LayerKind, Model};
use crate::cfg::Activation;
use hip_rt::{Device, DeviceBuffer, Module, Stream};
use hip_tile::{Tensor, TensorMut};
use kernels::yolo::concat::launch_concat;
use kernels::yolo::conv::{ConvParams, launch_conv};
use kernels::yolo::maxpool::{PoolParams, launch_maxpool};
use kernels::yolo::upsample::launch_upsample;

/// The kernels crate compiled for gfx1031.
pub(crate) static KERNELS_ELF: &[u8] = include_bytes!(env!("HIP_TILE_KERNELS_ELF"));

/// Blocking host-to-device copy into a buffer owned elsewhere by shared reference.
pub(crate) fn upload(dst: &DeviceBuffer<f32>, src: &[f32]) -> hip_rt::Result<()> {
    assert_eq!(dst.len(), src.len());
    // SAFETY: `dst` is a live device allocation of `src.len()` floats; the host never reads it directly.
    let rc = unsafe {
        hip_sys::hipMemcpy(
            dst.as_ptr().cast_mut().cast(),
            src.as_ptr().cast(),
            std::mem::size_of_val(src),
            hip_sys::hipMemcpyKind::HostToDevice,
        )
    };
    if rc == 0 { Ok(()) } else { Err(hip_rt::HipError::other(format!("hipMemcpy failed: {rc:?}"))) }
}

pub(crate) fn hip<T>(r: hip_rt::Result<T>) -> Result<T> {
    r.map_err(|e| Error::Unsupported(format!("hip: {e}")))
}

struct ConvBufs {
    w: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
}

/// A [`Model`] resident on the GPU.
pub struct GpuModel<'m> {
    model: &'m Model,
    _device: Device,
    module: Module,
    stream: Stream,
    input: DeviceBuffer<f32>,
    /// Per layer; `None` for `[yolo]` (aliases the previous layer's output).
    outs: Vec<Option<DeviceBuffer<f32>>>,
    convs: Vec<Option<ConvBufs>>,
    yolo: Vec<usize>,
}

impl<'m> GpuModel<'m> {
    pub fn new(model: &'m Model) -> Result<Self> {
        let device = hip(Device::init())?;
        let module = hip(Module::from_elf(KERNELS_ELF))?;
        let stream = hip(Stream::new())?;
        let input = hip(DeviceBuffer::zeroed(model.input_shape().len()))?;
        let mut outs = Vec::new();
        let mut convs = Vec::new();
        for (i, l) in model.layers.iter().enumerate() {
            match &l.kind {
                LayerKind::Conv(c) => {
                    if !matches!(c.cfg.activation, Activation::Linear | Activation::Leaky) {
                        return Err(Error::Unsupported(format!("layer {i}: GPU conv activation {:?}", c.cfg.activation)));
                    }
                    if c.weights.len() != c.n_weights() {
                        return Err(Error::Model(format!("layer {i}: weights not loaded")));
                    }
                    convs.push(Some(ConvBufs {
                        w: hip(DeviceBuffer::from_slice(&c.weights))?,
                        b: hip(DeviceBuffer::from_slice(&c.bias))?,
                    }));
                }
                _ => convs.push(None),
            }
            outs.push(match l.kind {
                LayerKind::Yolo(_) => None,
                _ => Some(hip(DeviceBuffer::zeroed(l.out_shape.len()))?),
            });
        }
        Ok(GpuModel { model, _device: device, module, stream, input, outs, convs, yolo: model.yolo_layers() })
    }

    /// Buffer of layer `i`'s output (a yolo layer resolves to its input).
    fn out_buf(&self, i: usize) -> &DeviceBuffer<f32> {
        match &self.outs[i] {
            Some(b) => b,
            None => self.out_buf(i - 1),
        }
    }

    fn out_ptr(&self, i: usize) -> *const f32 {
        self.out_buf(i).as_ptr()
    }

    /// Queue every layer (async) after uploading `chw`; does not synchronize.
    fn enqueue(&self, chw: &[f32]) -> Result<()> {
        let m = self.model;
        if chw.len() != self.input.len() {
            return Err(Error::Model(format!("input has {} floats, expected {}", chw.len(), self.input.len())));
        }
        // Buffers are device memory; the host never aliases them, so a shared `&self` may hand out pointers.
        hip(upload(&self.input, chw))?;
        let (md, st) = (&self.module, Some(&self.stream));
        for (i, l) in m.layers.iter().enumerate() {
            let prev: *const f32 = if i == 0 { self.input.as_ptr() } else { self.out_ptr(i - 1) };
            let (ins, outs) = (l.in_shape, l.out_shape);
            let (ohw, ihw) = ((outs.h * outs.w) as u32, (ins.h * ins.w) as u32);
            let Some(ob) = &self.outs[i] else { continue };
            let y = TensorMut::new(ob.as_ptr() as *mut f32, outs.c as u32, ohw, ohw);
            match &l.kind {
                LayerKind::Conv(c) => {
                    let cb = self.convs[i].as_ref().expect("conv buffers");
                    let act = if c.cfg.activation == Activation::Leaky { ConvParams::LEAKY } else { ConvParams::LINEAR };
                    let p = ConvParams::new(
                        ins.c as u32, ins.h as u32, ins.w as u32, c.cfg.size as u32,
                        c.cfg.stride as u32, c.cfg.padding as u32, act,
                    );
                    let x = Tensor::new(prev, ins.c as u32, ihw, ihw);
                    let w = cb.w.tensor(c.cfg.filters as u32, c.k() as u32);
                    let bias = Tensor::vector(cb.b.as_ptr(), c.cfg.filters as u32);
                    hip(launch_conv(md, st, x, w, bias, y, &p))?;
                }
                LayerKind::Maxpool(mp) => {
                    let p = PoolParams::new(ins.h as u32, ins.w as u32, mp.size as u32, mp.stride as u32, Some(mp.padding as u32));
                    hip(launch_maxpool(md, st, Tensor::new(prev, ins.c as u32, ihw, ihw), y, &p))?;
                }
                LayerKind::Upsample { stride } => {
                    hip(launch_upsample(md, st, Tensor::new(prev, ins.c as u32, ihw, ihw), y, ins.w as u32, *stride as u32))?;
                }
                LayerKind::Route { sources, groups, group_id } => {
                    let mut ch0 = 0usize;
                    for &s in sources {
                        let c = m.layers[s].out_shape.c / groups;
                        let src = unsafe { self.out_ptr(s).add(group_id * c * ohw as usize) };
                        let dst = TensorMut::new(unsafe { ob.as_ptr().cast_mut().add(ch0 * ohw as usize) }, c as u32, ohw, ohw);
                        hip(launch_concat(md, st, Tensor::new(src, c as u32, ohw, ohw), dst))?;
                        ch0 += c;
                    }
                }
                LayerKind::Yolo(_) => unreachable!(),
            }
        }
        Ok(())
    }

    /// Run the net and return the `[yolo]` layers' inputs (in `model.yolo_layers()` order).
    pub fn forward(&self, chw: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.enqueue(chw)?;
        hip(self.stream.synchronize())?;
        self.yolo
            .iter()
            .map(|&i| hip(self.out_buf(i).copy_to_vec()))
            .collect()
    }

    /// Run the net and download every layer's output (for parity tests).
    pub fn forward_all(&self, chw: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.enqueue(chw)?;
        hip(self.stream.synchronize())?;
        (0..self.model.layers.len())
            .map(|i| hip(self.out_buf(i).copy_to_vec()))
            .collect()
    }

    /// Queue + sync only, no download (for timing the device work).
    pub fn run(&self, chw: &[f32]) -> Result<()> {
        self.enqueue(chw)?;
        hip(self.stream.synchronize())
    }
}
