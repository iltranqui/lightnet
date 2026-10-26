//! Native non-BN GRU training, resident optimizers and legacy inference.
use crate::error::{Error,Result};
use crate::model::{GruSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, GpuForwardState, GruParams, ResidentParameterRange,
    ResidentMomentSlot, backend_error, tensor_backend_shape, forward_gru_layer_device};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use super::cpu::{Step,projections,serialized,validate};

#[derive(Clone,Debug)]
pub(crate) struct DeviceCache<T> {
    params:GruParams, weights:Vec<T>, steps:Vec<Step<T>>, generation:Option<u64>,
    ranges:Vec<std::ops::Range<usize>>,
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_gru_training_gpu(&self,layer:&Layer,spec:&GruSpec,params:&GruParams,
        input:&B::Tensor,config:ElementwiseLaunchConfig)->Result<(B::Tensor,DeviceCache<B::Tensor>)> {
        let batch=validate(spec,params,layer.input_shape,layer.output_shape)?;
        let (ins,outs)=(layer.input_shape.element_count(),spec.output);
        let be=&self.backend;let err=|e|backend_error("native GRU forward",e);
        let mut ranges=Vec::with_capacity(12);let mut layout=Vec::with_capacity(12);let mut cursor=0;
        for(i,p)in projections(params).into_iter().enumerate(){
            for (len,decay,slot) in [(p.biases.len(),false,ResidentMomentSlot::ChildBiases(i)),
                (p.weights.len(),true,ResidentMomentSlot::ChildWeights(i))] {
                let range=cursor..cursor+len;cursor+=len;ranges.push(range.clone());
                layout.push(ResidentParameterRange{range,decay,coupled_decay:true,moment_slot:slot});
            }
        }
        let host=serialized(params);
        let (weights,generation)=if self.resident_parameter_path_enabled(){
            let(p,g)=self.resident_parameter_slices(layer.index,&host,layout,&ranges,config)?;(p,Some(g))
        }else{
            let p=ranges.iter().map(|r|be.upload_f32(&[r.len()],&host[r.clone()]).map_err(err)).collect::<Result<Vec<_>>>()?;
            (p,None)
        };
        let shape=[batch,outs];let mut previous=be.zeros_f32(&shape).map_err(err)?;
        let mut tape=Vec::with_capacity(spec.steps);let mut outputs=Vec::with_capacity(spec.steps);
        for t in 0..spec.steps {
            let x=be.sequence_slice_f32(input,&tensor_backend_shape(layer.input_shape),t,spec.steps,&[batch,ins],config).map_err(err)?;
            let lin=|x:&B::Tensor,i:usize|be.connected_f32(x,&weights[2*i+1],&weights[2*i],batch,if i%2==0{ins}else{outs},outs,config).map_err(err);
            let gate=|a:B::Tensor,b:B::Tensor|->Result<B::Tensor>{
                let pre=be.vector_add_f32(&a,&b,config).map_err(err)?;
                be.activation_f32(&pre,F32ActivationKind::Logistic,config).map_err(err)
            };
            let z=gate(lin(&x,0)?,lin(&previous,1)?)?;
            let r=gate(lin(&x,2)?,lin(&previous,3)?)?;
            let reset=be.vector_mul_f32(&previous,&r,&shape,config).map_err(err)?;
            let candidate=gate(lin(&x,4)?,lin(&reset,5)?)?;
            let difference=be.saxpy_f32(-1.0,&candidate,&previous,config).map_err(err)?;
            let gated=be.vector_mul_f32(&z,&difference,&shape,config).map_err(err)?;
            let state=be.vector_add_f32(&candidate,&gated,config).map_err(err)?;
            tape.push(Step{x,previous,z,r,candidate,reset});previous=state.clone();outputs.push(state);
        }
        let output=be.sequence_concat_f32(&outputs.iter().collect::<Vec<_>>(),&tensor_backend_shape(layer.output_shape),batch*outs,config).map_err(err)?;
        Ok((output,DeviceCache{params:params.clone(),weights,steps:tape,generation,ranges}))
    }

    pub(crate) fn backward_gru_gpu(&self,layer:&Layer,spec:&GruSpec,params:&GruParams,
        forward:&GpuForwardState<B::Tensor>,grad:B::Tensor,config:ElementwiseLaunchConfig,
        layer_gradients:&mut[Option<B::Tensor>])->Result<()> {
        let batch=validate(spec,params,layer.input_shape,layer.output_shape)?;
        let cache=forward.gpu_gru_caches.get(layer.index).and_then(Option::as_ref)
            .ok_or_else(||Error::InvalidConfig("missing native GRU training cache".into()))?;
        if cache.params!=*params || cache.steps.len()!=spec.steps {
            return Err(Error::InvalidConfig("GRU parameters/cache changed; rerun training forward".into()));
        }
        let (ins,outs)=(layer.input_shape.element_count(),spec.output);let shape=[batch,outs];
        let be=&self.backend;let err=|e|backend_error("native GRU backward",e);
        let ones=be.upload_f32(&shape,&vec![1.0;batch*outs]).map_err(err)?;
        let add=|a:&B::Tensor,b:&B::Tensor|be.vector_add_f32(a,b,config).map_err(err);
        let mul=|a:&B::Tensor,b:&B::Tensor|be.vector_mul_f32(a,b,&shape,config).map_err(err);
        let complement=|a:&B::Tensor|be.saxpy_f32(-1.0,a,&ones,config).map_err(err);
        let gate_grad=|dy:&B::Tensor,g:&B::Tensor|->Result<B::Tensor>{mul(&mul(dy,g)?,&complement(g)?)};
        let mut gradients=cache.ranges.iter().map(|r|be.zeros_f32(&[r.len()]).map_err(err)).collect::<Result<Vec<_>>>()?;
        let mut carry=be.zeros_f32(&shape).map_err(err)?;let mut dxs=vec![None;spec.steps];
        for t in (0..spec.steps).rev(){
            let step=&cache.steps[t];
            let upstream=be.sequence_slice_f32(&grad,&tensor_backend_shape(layer.output_shape),t,spec.steps,&shape,config).map_err(err)?;
            let dh=add(&upstream,&carry)?;
            let difference=be.saxpy_f32(-1.0,&step.candidate,&step.previous,config).map_err(err)?;
            let dz=gate_grad(&mul(&dh,&difference)?,&step.z)?;
            let dn=gate_grad(&mul(&dh,&complement(&step.z)?)?,&step.candidate)?;
            let mut lin=|x:&B::Tensor,dy:&B::Tensor,i:usize|->Result<B::Tensor>{
                let(dx,dw,db)=be.connected_backward_f32(x,&cache.weights[2*i+1],dy,batch,if i%2==0{ins}else{outs},outs,config).map_err(err)?;
                gradients[2*i]=add(&gradients[2*i],&db)?;
                gradients[2*i+1]=add(&gradients[2*i+1],&dw)?;Ok(dx)
            };
            let dreset=lin(&step.reset,&dn,5)?;
            let dr=gate_grad(&mul(&dreset,&step.previous)?,&step.r)?;
            let dxh=lin(&step.x,&dn,4)?;let dxr=lin(&step.x,&dr,2)?;let dxz=lin(&step.x,&dz,0)?;
            let dhr=lin(&step.previous,&dr,3)?;let dhz=lin(&step.previous,&dz,1)?;
            carry=add(&add(&mul(&dh,&step.z)?,&mul(&dreset,&step.r)?)?,&add(&dhr,&dhz)?)?;
            dxs[t]=Some(add(&add(&dxh,&dxr)?,&dxz)?);
        }
        let generation=cache.generation.ok_or_else(||Error::InvalidConfig("native GRU backward requires resident parameters".into()))?;
        self.accumulate_resident_parameter_gradients(layer.index,generation,
            cache.ranges.iter().cloned().zip(gradients).collect(),config)?;
        let inputs=dxs.iter().map(|x|x.as_ref().expect("GRU timestep input adjoint")).collect::<Vec<_>>();
        let dx=be.sequence_concat_f32(&inputs,&tensor_backend_shape(layer.input_shape),batch*ins,config).map_err(err)?;
        self.accumulate_previous_gpu_gradient(layer.index,dx,layer_gradients)
    }

    /// Resident non-BN inference, or legacy rolling-stat BN inference.
    pub(crate) fn forward_gru_gpu(
        &self,
        layer: &Layer,
        spec: &GruSpec,
        params: &GruParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        if !spec.batch_normalize {
            return self.forward_gru_training_gpu(layer,spec,params,current_input,launch_config)
                .map(|(output,_)|output);
        }
        forward_gru_layer_device(
            &self.backend,
            current_input,
            layer.input_shape,
            spec,
            params,
            layer.output_shape,
            layer.index,
            launch_config,
        )
    }
}
