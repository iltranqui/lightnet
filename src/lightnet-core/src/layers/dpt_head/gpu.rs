//! Native DPT neck/head composition. The small reverse tape records only
//! this layer's conv, resize and residual graph; tensor math stays on device.
//! Parameter gradients stage into the existing source-ordered optimizer arena.
use super::cpu::{DptHeadGeometry, DptHeadWeightLayout, dpt_head_weight_layout};
use crate::error::{Error, Result};
use crate::model::{DptHeadSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, LayerTrainBuffers, backend_error, tensor_backend_shape};
use lightnet_cuda::{BilinearGatherGeometry, ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

#[derive(Clone, Debug)]
enum Op<T> {
    Input,
    Tap { input: usize, offset: usize },
    Conv { input: usize, weight: usize, bias: Option<usize>, kernel: usize, stride: usize, pad: usize, transpose: bool },
    Relu(usize),
    Add(usize, usize),
    Resize { input: usize, y: T, x: T, geometry: BilinearGatherGeometry },
}

#[derive(Clone, Debug)]
struct Node<T> { value: T, shape: [usize; 4], op: Op<T> }

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: DptHeadGeometry,
    layout: DptHeadWeightLayout,
    weights: Vec<Option<T>>,
    nodes: Vec<Node<T>>,
}

struct Builder<'a, B: F32TensorOpsBackend> {
    be: &'a B,
    config: ElementwiseLaunchConfig,
    cache: DeviceCache<B::Tensor>,
}

impl<B: F32TensorOpsBackend> Builder<'_, B> where B::Error: std::fmt::Display {
    fn push(&mut self, value: B::Tensor, shape: [usize; 4], op: Op<B::Tensor>) -> usize {
        let i = self.cache.nodes.len(); self.cache.nodes.push(Node { value, shape, op }); i
    }
    fn tap(&mut self, offset: usize, channels: usize) -> Result<usize> {
        let n = &self.cache.nodes[0];
        let [b, c, h, w] = n.shape;
        let shape = [b, channels, h, w];
        let value = self.be.route_concat_f32(&[&n.value], &[offset], &[channels], &[c], &shape,
            h*w, self.config).map_err(|e| backend_error("DPT tap gather", e))?;
        Ok(self.push(value, shape, Op::Tap { input: 0, offset }))
    }
    fn conv(&mut self, input: usize, weight: usize, bias: Option<usize>, filters: usize,
        kernel: usize, stride: usize, pad: usize, transpose: bool) -> Result<usize> {
        let n = &self.cache.nodes[input];
        let [b, c, h, w] = n.shape;
        let (oh, ow) = if transpose { (h*stride, w*stride) }
            else { ((h+2*pad-kernel)/stride+1, (w+2*pad-kernel)/stride+1) };
        let shape = [b, filters, oh, ow];
        let weights = self.cache.weights[weight].as_ref().ok_or_else(|| Error::InvalidConfig("DPT missing convolution weights".into()))?;
        let err = |e| backend_error("GPU DPT convolution", e);
        let raw = if transpose {
            self.be.deconvolutional_f32(&n.value, weights, &shape, c, h, w,
                filters, kernel, stride, self.config).map_err(err)?
        } else {
            self.be.conv2d_f32(&n.value, weights, &shape, c, h, w, filters, kernel,
                stride, stride, 1, pad, 1, self.config).map_err(err)?
        };
        let value = if let Some(bias) = bias {
            self.be.bias_add_f32(&raw, self.cache.weights[bias].as_ref().ok_or_else(||
                Error::InvalidConfig("DPT missing convolution bias".into()))?, filters, oh*ow, self.config).map_err(err)?
        } else { raw };
        Ok(self.push(value, shape, Op::Conv { input, weight, bias, kernel, stride, pad, transpose }))
    }
    fn relu(&mut self, input: usize) -> Result<usize> {
        let n = &self.cache.nodes[input];
        let value = self.be.activation_f32(&n.value, F32ActivationKind::Relu, self.config)
            .map_err(|e| backend_error("GPU DPT ReLU", e))?;
        Ok(self.push(value, n.shape, Op::Relu(input)))
    }
    fn add(&mut self, a: usize, b: usize) -> Result<usize> {
        let (a_node, b_node) = (&self.cache.nodes[a], &self.cache.nodes[b]);
        if a_node.shape != b_node.shape { return Err(Error::InvalidConfig("DPT fusion residual geometry mismatch".into())); }
        let value = self.be.residual_bias_f32(&a_node.value, &b_node.value, None,
            &a_node.shape, a_node.shape[0], a_node.shape[1..].iter().product(), self.config)
            .map_err(|e| backend_error("GPU DPT residual", e))?;
        Ok(self.push(value, a_node.shape, Op::Add(a, b)))
    }
    fn resconf(&mut self, input: usize, w1: usize, w2: usize) -> Result<usize> {
        let c = self.cache.nodes[input].shape[1];
        let r = self.relu(input)?;
        let r = self.conv(r, w1, Some(w1+1), c, 3, 1, 1, false)?;
        let r = self.relu(r)?;
        let r = self.conv(r, w2, Some(w2+1), c, 3, 1, 1, false)?;
        self.add(input, r)
    }
    fn resize(&mut self, input: usize, oh: usize, ow: usize) -> Result<usize> {
        let n = &self.cache.nodes[input];
        let [b, c, h, w] = n.shape;
        let geometry = BilinearGatherGeometry::new(b, c, h, w, 1, oh, ow).map_err(Error::InvalidConfig)?;
        let mut ys = Vec::with_capacity(b*oh*ow);
        let mut xs = Vec::with_capacity(b*oh*ow);
        // Align corners exactly as dpt_head::bilinear_coordinates. These
        // coordinates depend only on shape and have no trainable derivative.
        for _ in 0..b { for y in 0..oh { for x in 0..ow {
            ys.push(if oh > 1 { y as f32 * (h-1) as f32 / (oh-1) as f32 } else { 0.0 });
            xs.push(if ow > 1 { x as f32 * (w-1) as f32 / (ow-1) as f32 } else { 0.0 });
        }}}
        let err = |e| backend_error("GPU DPT bilinear resize", e);
        let y = self.be.upload_f32(&[b, 1, oh, ow], &ys).map_err(err)?;
        let x = self.be.upload_f32(&[b, 1, oh, ow], &xs).map_err(err)?;
        let value = self.be.deform_bilinear_gather_f32(&n.value, &y, &x, geometry, self.config).map_err(err)?;
        // The singleton sample axis has identical storage to NCHW. Nodes
        // carry explicit logical shapes, used by every subsequent operation.
        Ok(self.push(value, [b, c, oh, ow], Op::Resize { input, y, x, geometry }))
    }
}

impl<B: F32TensorOpsBackend> GpuExecutableNetwork<B> where B::Error: std::fmt::Display {
    pub(crate) fn forward_dpt_head_gpu(&self, layer: &Layer, spec: &DptHeadSpec,
        params: &super::DptHeadWeights, input: &B::Tensor, config: ElementwiseLaunchConfig,
        training: bool) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        let geometry = spec.geometry(layer.input_shape);
        let layout = dpt_head_weight_layout(geometry)?;
        if params.values.len() != layout.total {
            return Err(Error::InvalidConfig("DPT GPU parameter length mismatch".into()));
        }
        let mut weights = Vec::with_capacity(66);
        for range in &layout.slots {
            weights.push(if range.len == 0 { None } else {
                Some(self.backend.upload_f32(&[range.len], &params.values[range.start..range.start+range.len])
                    .map_err(|e| backend_error("upload DPT weights", e))?)
            });
        }
        let mut builder = Builder { be: &self.backend, config, cache: DeviceCache {
            geometry, layout, weights, nodes: vec![Node { value: input.clone(),
                shape: tensor_backend_shape(layer.input_shape), op: Op::Input }] } };
        let g = geometry;
        let mut neck = [0usize; 4];
        for tap in 0..4 {
            let x = builder.tap(tap * g.embed_dim, g.embed_dim)?;
            let x = builder.conv(x, 2*tap, Some(2*tap+1), g.neck[tap], 1, 1, 0, false)?;
            let x = match tap {
                0 | 1 => { let k = if tap == 0 { 4 } else { 2 };
                    builder.conv(x, 8+2*tap, Some(9+2*tap), g.neck[tap], k, k, 0, true)? }
                2 => x,
                _ => builder.conv(x, 14, Some(15), g.neck[tap], 3, 2, 1, false)?,
            };
            neck[tap] = builder.conv(x, 16+tap, None, g.fusion, 3, 1, 1, false)?;
        }
        let mut current = neck[3];
        for index in 0..4 {
            if index > 0 {
                let residual = builder.resconf(neck[3-index], 28+2*index, 36+2*index)?;
                current = builder.add(current, residual)?;
            }
            current = builder.resconf(current, 44+2*index, 52+2*index)?;
            let [_, _, h, w] = builder.cache.nodes[current].shape;
            let (oh, ow) = if index < 3 {
                let s = builder.cache.nodes[neck[2-index]].shape; (s[2], s[3])
            } else { (h*2, w*2) };
            current = builder.resize(current, oh, ow)?;
            current = builder.conv(current, 20+2*index, Some(21+2*index), g.fusion, 1, 1, 0, false)?;
        }
        current = builder.conv(current, 60, Some(61), g.fusion/2, 3, 1, 1, false)?;
        current = builder.resize(current, g.output_height()?, g.output_width()?)?;
        current = builder.conv(current, 62, Some(63), g.head_hidden, 3, 1, 1, false)?;
        current = builder.relu(current)?;
        current = builder.conv(current, 64, Some(65), 1, 1, 1, 0, false)?;
        current = builder.relu(current)?;
        if builder.cache.nodes[current].shape != tensor_backend_shape(layer.output_shape) {
            return Err(Error::InvalidConfig("DPT GPU output geometry mismatch".into()));
        }
        let output = builder.cache.nodes[current].value.clone();
        Ok((output, training.then_some(builder.cache)))
    }

    pub(crate) fn backward_dpt_head_gpu(&self, layer: &Layer, spec: &DptHeadSpec,
        grad: B::Tensor, config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>]) -> Result<()> {
        if crate::runtime::GradMode::from_env() == crate::runtime::GradMode::Cpp { return Ok(()); }
        let caches = self.dpt_head_device_caches.borrow();
        let cache = caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig("missing DPT GPU training cache".into()))?;
        if cache.geometry != spec.geometry(layer.input_shape) {
            return Err(Error::InvalidConfig("DPT device cache geometry mismatch".into()));
        }
        let be = &self.backend;
        let err = |e| backend_error("GPU DPT backward", e);
        let mut gradients = vec![None; cache.nodes.len()];
        *gradients.last_mut().ok_or_else(|| Error::InvalidConfig("empty DPT device tape".into()))? = Some(grad);
        let mut dparams = vec![0.0; cache.layout.total];
        let mut parameter_grad = |slot: usize, tensor: &B::Tensor| -> Result<()> {
            let values = be.download_f32(tensor).map_err(err)?;
            let range = cache.layout.get(slot);
            if values.len() != range.len { return Err(Error::InvalidConfig("DPT gradient slot length mismatch".into())); }
            for (dst, src) in dparams[range.start..range.start+range.len].iter_mut().zip(values) { *dst += src; }
            Ok(())
        };
        for index in (1..cache.nodes.len()).rev() {
            let Some(dy) = gradients[index].take() else { continue; };
            let node = &cache.nodes[index];
            let mut pending = Vec::with_capacity(2);
            match &node.op {
                Op::Input => {}
                Op::Add(a, b) => { pending.push((*a, dy.clone())); pending.push((*b, dy)); }
                Op::Relu(input) => {
                    let dx = be.activation_backward_f32(&cache.nodes[*input].value, &dy,
                        F32ActivationKind::Relu, config).map_err(err)?;
                    pending.push((*input, dx));
                }
                Op::Tap { input, offset } => {
                    let source = &cache.nodes[*input];
                    let result = be.route_backward_f32(&dy, &[source.shape.to_vec()], &[*offset],
                        &[node.shape[1]], &[source.shape[1]], source.shape[2]*source.shape[3], config).map_err(err)?;
                    let dx = result.into_iter().next().ok_or_else(|| Error::InvalidConfig("DPT tap adjoint arity".into()))?;
                    pending.push((*input, dx));
                }
                Op::Resize { input, y, x, geometry } => {
                    let (dx, _, _) = be.deform_bilinear_gather_backward_f32(&cache.nodes[*input].value,
                        y, x, &dy, *geometry, config).map_err(err)?;
                    pending.push((*input, dx));
                }
                Op::Conv { input, weight, bias, kernel, stride, pad, transpose } => {
                    let source = &cache.nodes[*input];
                    let [_, c, h, w] = source.shape;
                    let f = node.shape[1];
                    let weights = cache.weights[*weight].as_ref().ok_or_else(|| Error::InvalidConfig("missing DPT backward weights".into()))?;
                    let (dx, dw, db) = if *transpose {
                        be.deconvolutional_backward_f32(&source.value, weights, &dy,
                            &source.shape, &node.shape, f, *kernel, *stride, config).map_err(err)?
                    } else {
                        let dx = be.conv2d_backward_input_f32(&dy, weights, &source.shape,
                            c, h, w, f, *kernel, *stride, *stride, 1, *pad, 1, config).map_err(err)?;
                        let dw = be.conv2d_backward_weights_f32(&source.value, &dy, &[f,c,*kernel,*kernel],
                            c, h, w, f, *kernel, *stride, *stride, 1, *pad, 1, config).map_err(err)?;
                        let db = be.bias_backward_f32(&dy, f, node.shape[2]*node.shape[3], config).map_err(err)?;
                        (dx, dw, db)
                    };
                    parameter_grad(*weight, &dw)?;
                    if let Some(bias) = bias { parameter_grad(*bias, &db)?; }
                    pending.push((*input, dx));
                }
            }
            for (input, dx) in pending {
                gradients[input] = Some(match gradients[input].take() {
                    Some(existing) => be.residual_bias_f32(&existing, &dx, None, &cache.nodes[input].shape,
                        cache.nodes[input].shape[0], cache.nodes[input].shape[1..].iter().product(), config).map_err(err)?,
                    None => dx,
                });
            }
        }
        drop(parameter_grad);
        let dx = gradients[0].take().ok_or_else(|| Error::InvalidConfig("DPT missing input adjoint".into()))?;
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::DptHead(train)) = buffers.get_mut(layer.index) else {
            return Err(Error::InvalidConfig("missing DPT train buffers".into()));
        };
        if train.weight_updates.len() != dparams.len() {
            return Err(Error::InvalidConfig("DPT train buffer length mismatch".into()));
        }
        train.has_gradient |= dparams.iter().any(|&x| x != 0.0);
        for (dst, gradient) in train.weight_updates.iter_mut().zip(dparams) { *dst -= gradient; }
        drop(buffers); drop(caches);
        self.accumulate_previous_gpu_gradient(layer.index, dx, layer_output_gradients)
    }
}
