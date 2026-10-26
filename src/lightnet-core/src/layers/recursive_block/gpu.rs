//! Native static shared-weight recurrence with convolution/connected bodies.
//! Controller, lattice, BN and other body variants keep the explicit host
//! reference. Backward sums shared parameter adjoints over every recurrence.
use crate::error::{Error, Result};
use crate::model::{Layer, LayerKind, RecursiveBlockSpec};
use crate::runtime::{GpuExecutableNetwork, LayerParams, LayerTrainBuffers, RecursiveBlockParams,
    backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

pub(crate) fn supported(spec: &RecursiveBlockSpec) -> bool {
    spec.ouroboros == 0 && spec.lattice_mix == 0.0 && !spec.body.is_empty()
        && spec.body.iter().all(|layer| match &layer.kind {
            LayerKind::Convolutional(s) => !s.batch_normalize && s.share_index.is_none()
                && s.activation.to_cuda_f32_kind().is_some(),
            LayerKind::Connected(s) => !s.batch_normalize && s.activation.to_cuda_f32_kind().is_some(),
            _ => false,
        })
}

#[derive(Clone, Debug)]
struct BodyCache<T> { input: T, pre: T, weights: T }

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> { spec: RecursiveBlockSpec, passes: Vec<Vec<BodyCache<T>>> }

impl<B: F32TensorOpsBackend> GpuExecutableNetwork<B> where B::Error: std::fmt::Display {
    pub(crate) fn forward_recursive_block_gpu(&self, layer: &Layer, spec: &RecursiveBlockSpec,
        params: &RecursiveBlockParams, input: &B::Tensor, config: ElementwiseLaunchConfig,
        training: bool) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        if !supported(spec) || spec.loops == 0 || params.body.len()!=spec.body.len()
            || layer.input_shape!=layer.output_shape {
            return Err(Error::InvalidConfig("recursive native body/recurrence geometry mismatch".into()));
        }
        let be=&self.backend;
        let err=|e|backend_error("GPU recursive forward",e);
        let scale=|x:&B::Tensor,value:f32| {
            let scalar=be.upload_f32(&[1],&[value]).map_err(err)?;
            be.channel_scale_f32(x,&scalar,1,1,config).map_err(err)
        };
        let shape=tensor_backend_shape(layer.output_shape);
        let add=|x:&B::Tensor,y:&B::Tensor|be.residual_bias_f32(x,y,None,&shape,
            shape[0],shape[1..].iter().product(),config).map_err(err);
        let mut hidden=input.clone();
        let mut passes=Vec::with_capacity(if training {spec.loops}else{0});
        for _ in 0..spec.loops {
            let mut body=hidden.clone();
            let mut pass=Vec::with_capacity(if training {spec.body.len()}else{0});
            for (body_layer,parameters) in spec.body.iter().zip(&params.body) {
                let (raw,weights,bias,activation)=match (&body_layer.kind,parameters) {
                    (LayerKind::Convolutional(s),LayerParams::Convolutional(p)) if p.batch_norm.is_none()=>{
                        let c=body_layer.input_shape.channels;
                        let expected=s.filters*(c/s.groups.max(1))*s.size*s.size;
                        if p.weights.len()!=expected || p.biases.len()!=s.filters {
                            return Err(Error::InvalidConfig("recursive conv parameter length mismatch".into()));
                        }
                        let weights=be.upload_f32(&[s.filters,c/s.groups.max(1),s.size,s.size],&p.weights).map_err(err)?;
                        let bias=be.upload_f32(&[s.filters],&p.biases).map_err(err)?;
                        let raw=be.conv2d_f32(&body,&weights,&tensor_backend_shape(body_layer.output_shape),
                            c,body_layer.input_shape.height,body_layer.input_shape.width,s.filters,s.size,
                            s.stride_x,s.stride_y,s.dilation,s.padding,s.groups.max(1),config).map_err(err)?;
                        (raw,weights,Some(bias),s.activation.to_cuda_f32_kind())
                    }
                    (LayerKind::Connected(s),LayerParams::Connected(p)) if p.batch_norm.is_none()=>{
                        let ins=body_layer.input_shape.element_count();
                        if p.weights.len()!=s.output*ins || p.biases.len()!=s.output {
                            return Err(Error::InvalidConfig("recursive connected parameter length mismatch".into()));
                        }
                        let weights=be.upload_f32(&[s.output,ins],&p.weights).map_err(err)?;
                        let bias=be.upload_f32(&[s.output],&p.biases).map_err(err)?;
                        let raw=be.connected_f32(&body,&weights,&bias,body_layer.input_shape.batch,
                            ins,s.output,config).map_err(err)?;
                        (raw,weights,None,s.activation.to_cuda_f32_kind())
                    }
                    _=>return Err(Error::InvalidConfig("recursive native body parameter kind mismatch".into())),
                };
                let pre=if let Some(bias)=bias {be.bias_add_f32(&raw,&bias,body_layer.output_shape.channels,
                    body_layer.output_shape.width*body_layer.output_shape.height,config).map_err(err)?}else{raw};
                let activation=activation.ok_or_else(||Error::InvalidConfig("recursive native activation missing".into()))?;
                let output=be.activation_f32(&pre,activation,config).map_err(err)?;
                if training {pass.push(BodyCache{input:body,pre,weights});}
                body=output;
            }
            hidden=add(&scale(&body,spec.body_scale)?,&scale(&hidden,spec.residual_scale)?)?;
            if spec.injection {hidden=add(&hidden,&scale(input,spec.injection_scale)?)?;}
            if training {passes.push(pass);}
        }
        Ok((hidden,training.then(||DeviceCache{spec:spec.clone(),passes})))
    }

    pub(crate) fn backward_recursive_block_gpu(&self,layer:&Layer,spec:&RecursiveBlockSpec,
        forward:&crate::runtime::GpuForwardState<B::Tensor>,
        grad:B::Tensor,config:ElementwiseLaunchConfig,layer_output_gradients:&mut[Option<B::Tensor>])->Result<()> {
        let caches=&forward.recursive_block_device_caches;
        let cache=caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig("missing recursive native training cache".into()))?;
        if cache.spec!=*spec || cache.passes.len()!=spec.loops {
            return Err(Error::InvalidConfig("recursive native cache geometry mismatch".into()));
        }
        let be=&self.backend;
        let err=|e|backend_error("GPU recursive backward",e);
        let scale=|x:&B::Tensor,value:f32| {
            let scalar=be.upload_f32(&[1],&[value]).map_err(err)?;
            be.channel_scale_f32(x,&scalar,1,1,config).map_err(err)
        };
        let shape=tensor_backend_shape(layer.input_shape);
        let add=|x:&B::Tensor,y:&B::Tensor|be.residual_bias_f32(x,y,None,&shape,
            shape[0],shape[1..].iter().product(),config).map_err(err);
        let mut input_grad=be.zeros_f32(&shape).map_err(err)?;
        let mut hidden_grad=grad;
        let mut updates:Vec<Option<(Vec<f32>,Vec<f32>)>>=vec![None;spec.body.len()];
        for pass in cache.passes.iter().rev() {
            if pass.len()!=spec.body.len(){return Err(Error::InvalidConfig("recursive native pass length mismatch".into()));}
            if spec.injection {input_grad=add(&input_grad,&scale(&hidden_grad,spec.injection_scale)?)?;}
            let direct=scale(&hidden_grad,spec.residual_scale)?;
            let mut dy=scale(&hidden_grad,spec.body_scale)?;
            for (i,(body_layer,body_cache)) in spec.body.iter().zip(pass).enumerate().rev() {
                let activation=match &body_layer.kind {
                    LayerKind::Convolutional(s)=>s.activation.to_cuda_f32_kind(),
                    LayerKind::Connected(s)=>s.activation.to_cuda_f32_kind(),
                    _=>None,
                }.ok_or_else(||Error::InvalidConfig("recursive body activation mismatch".into()))?;
                let delta=be.activation_backward_f32(&body_cache.pre,&dy,activation,config).map_err(err)?;
                let (dx,dw,db)=match &body_layer.kind {
                    LayerKind::Convolutional(s)=>{
                        let (c,h,w)=(body_layer.input_shape.channels,body_layer.input_shape.height,body_layer.input_shape.width);
                        let dx=be.conv2d_backward_input_f32(&delta,&body_cache.weights,&tensor_backend_shape(body_layer.input_shape),
                            c,h,w,s.filters,s.size,s.stride_x,s.stride_y,s.dilation,s.padding,s.groups.max(1),config).map_err(err)?;
                        let dw=be.conv2d_backward_weights_f32(&body_cache.input,&delta,&[s.filters,c/s.groups.max(1),s.size,s.size],
                            c,h,w,s.filters,s.size,s.stride_x,s.stride_y,s.dilation,s.padding,s.groups.max(1),config).map_err(err)?;
                        let db=be.bias_backward_f32(&delta,s.filters,body_layer.output_shape.height*body_layer.output_shape.width,config).map_err(err)?;
                        (dx,dw,db)
                    }
                    LayerKind::Connected(s)=>be.connected_backward_f32(&body_cache.input,&body_cache.weights,&delta,
                        body_layer.input_shape.batch,body_layer.input_shape.element_count(),s.output,config).map_err(err)?,
                    _=>return Err(Error::InvalidConfig("recursive native body kind mismatch".into())),
                };
                let (dw,db)=(be.download_f32(&dw).map_err(err)?,be.download_f32(&db).map_err(err)?);
                if let Some((weights,biases))=&mut updates[i] {
                    if weights.len()!=dw.len() || biases.len()!=db.len(){return Err(Error::InvalidConfig("recursive shared gradient length mismatch".into()));}
                    for (dst,src) in weights.iter_mut().zip(dw){*dst+=src;}
                    for (dst,src) in biases.iter_mut().zip(db){*dst+=src;}
                }else{updates[i]=Some((dw,db));}
                dy=dx;
            }
            hidden_grad=add(&direct,&dy)?;
        }
        input_grad=add(&input_grad,&hidden_grad)?;
        let mut buffers=self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::RecursiveBlock(train))=buffers.get_mut(layer.index)else{
            return Err(Error::InvalidConfig("missing recursive train buffers".into()));
        };
        if train.body.len()!=updates.len(){return Err(Error::InvalidConfig("recursive body buffer count mismatch".into()));}
        // Validate every destination before updating any shared accumulator.
        for (buffer,update) in train.body.iter().zip(&updates) {
            let Some((dw,db))=update else{return Err(Error::InvalidConfig("missing recursive body gradient".into()));};
            let (w,b)=match buffer {
                LayerTrainBuffers::Convolutional(t)=>(t.weight_updates.len(),t.bias_updates.len()),
                LayerTrainBuffers::Connected(t)=>(t.weight_updates.len(),t.bias_updates.len()),
                _=>return Err(Error::InvalidConfig("recursive body train kind mismatch".into())),
            };
            if w!=dw.len() || b!=db.len(){return Err(Error::InvalidConfig("recursive train gradient length mismatch".into()));}
        }
        for (buffer,update) in train.body.iter_mut().zip(updates) {
            let (dw,db)=update.ok_or_else(||Error::InvalidConfig("missing recursive gradient".into()))?;
            let (w,b)=match buffer {
                LayerTrainBuffers::Convolutional(t)=>(&mut t.weight_updates,&mut t.bias_updates),
                LayerTrainBuffers::Connected(t)=>(&mut t.weight_updates,&mut t.bias_updates),
                _=>unreachable!("validated recursive train buffer"),
            };
            for (dst,src) in w.iter_mut().zip(dw){*dst-=src;}
            for (dst,src) in b.iter_mut().zip(db){*dst-=src;}
        }
        drop(buffers);
        self.accumulate_previous_gpu_gradient(layer.index,input_grad,layer_output_gradients)
    }
}
