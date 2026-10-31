//! Exact adjoint of the existing sigmoid-candidate GRU.
//! Sequence batch is time-major: [step, sequence_batch, feature].
use crate::error::{Error,Result};
use crate::model::{GruSpec,TensorShape};
use crate::params::{ConnectedParams,GruParams};
use crate::model::Tensor;

#[derive(Clone,Debug)]
#[doc(hidden)]
pub struct Step<T> {
    pub x:T, pub previous:T, pub z:T, pub r:T, pub candidate:T, pub reset:T,
}

#[doc(hidden)]
pub fn projections(p:&GruParams)->[&ConnectedParams;6] {
    [&p.input_z,&p.state_z,&p.input_r,&p.state_r,&p.input_h,&p.state_h]
}
#[doc(hidden)]
pub fn projections_mut(p:&mut GruParams)->[&mut ConnectedParams;6] {
    [&mut p.input_z,&mut p.state_z,&mut p.input_r,&mut p.state_r,&mut p.input_h,&mut p.state_h]
}
#[doc(hidden)]
pub fn serialized(p:&GruParams)->Vec<f32> {
    projections(p).into_iter().flat_map(|p|p.biases.iter().chain(&p.weights).copied()).collect()
}
#[doc(hidden)]
pub fn set_serialized(p:&mut GruParams,values:&[f32])->Result<()> {
    let total=projections(p).iter().map(|p|p.biases.len()+p.weights.len()).sum::<usize>();
    if total!=values.len(){return Err(Error::InvalidConfig("resident GRU parameter length mismatch".into()));}
    let mut cursor=0;
    for p in projections_mut(p){
        let end=cursor+p.biases.len();p.biases.copy_from_slice(&values[cursor..end]);cursor=end;
        let end=cursor+p.weights.len();p.weights.copy_from_slice(&values[cursor..end]);cursor=end;
    }
    Ok(())
}
#[doc(hidden)]
pub fn validate(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
    if s.steps==0 || input.batch==0 || !input.batch.is_multiple_of(s.steps) || s.output==0
        || output!=TensorShape::new(1,1,s.output,input.batch) {
        return Err(Error::InvalidConfig("GRU sequence/output geometry mismatch".into()));
    }
    if s.batch_normalize || projections(p).iter().any(|p|p.batch_norm.is_some()) {
        return Err(Error::InvalidConfig("GRU BN training backward is not implemented".into()));
    }
    validate_cpu(s,p,input,output)
}

#[doc(hidden)]
pub fn validate_cpu(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
    if s.steps==0 || input.batch==0 || !input.batch.is_multiple_of(s.steps) || s.output==0
        || output!=TensorShape::new(1,1,s.output,input.batch) {
        return Err(Error::InvalidConfig("GRU sequence/output geometry mismatch".into()));
    }
    for(i,p)in projections(p).iter().enumerate(){
        let ins=if i%2==0{input.element_count()}else{s.output};
        if p.batch_norm.is_some()!=s.batch_normalize || p.batch_norm.as_ref().is_some_and(|bn|
            bn.scales.len()!=s.output || bn.biases.len()!=s.output || bn.rolling_mean.len()!=s.output || bn.rolling_variance.len()!=s.output) {
            return Err(Error::InvalidConfig("GRU batch norm configuration mismatch".into()));
        }
        if p.weights.len()!=ins*s.output || p.biases.len()!=s.output {
            return Err(Error::InvalidConfig("GRU projection geometry mismatch".into()));
        }
    }
    Ok(input.batch/s.steps)
}

#[doc(hidden)]
pub fn tape(s:&GruSpec,p:&GruParams,input:&Tensor,output:TensorShape)->Result<Vec<Step<Vec<f32>>>> {
    let batch=validate_cpu(s,p,input.shape,output)?;
    let (ins,outs)=(input.shape.element_count(),s.output);
    let len=batch*outs;
    let pp=projections(p);
    let lin=|x:&[f32],i:usize| {
        let width=if i.is_multiple_of(2){ins}else{outs};
        let mut y=lightnet_cuda::host_ops::connected(x,&pp[i].weights,&vec![0.0;outs],batch,width,outs);
        for (j,v) in y.iter_mut().enumerate() {
            let c=j%outs;
            if let Some(bn)=&pp[i].batch_norm { *v=(*v-bn.rolling_mean[c])/(bn.rolling_variance[c]+crate::params::BATCH_NORM_EPSILON).sqrt()*bn.scales[c]; }
            *v+=pp[i].biases[c];
        }
        y
    };
    let gate=|a:Vec<f32>,b:Vec<f32>|a.iter().zip(b).map(|(a,b)|1.0/(1.0+(-(a+b)).exp())).collect::<Vec<_>>();
    let mut previous=vec![0.0;len]; let mut tape=Vec::with_capacity(s.steps);
    for t in 0..s.steps {
        let x=input.data[t*batch*ins..(t+1)*batch*ins].to_vec();
        let z=gate(lin(&x,0),lin(&previous,1));
        let r=gate(lin(&x,2),lin(&previous,3));
        let reset=previous.iter().zip(&r).map(|(h,r)|h*r).collect::<Vec<_>>();
        let candidate=gate(lin(&x,4),lin(&reset,5));
        let state=z.iter().zip(&previous).zip(&candidate).map(|((z,h),n)|z*h+(1.0-z)*n).collect();
        tape.push(Step{x,previous,z,r,candidate,reset}); previous=state;
    }
    Ok(tape)
}

#[doc(hidden)]
pub fn backward(s:&GruSpec,p:&GruParams,input:&Tensor,grad:&Tensor)->Result<(Tensor,GruParams)> {
    let batch=validate_cpu(s,p,input.shape,grad.shape)?;
    let (ins,outs)=(input.shape.element_count(),s.output);
    let len=batch*outs;
    let pp=projections(p);
    let tape=tape(s,p,input,grad.shape)?;
    let mut gradients=p.clone();
    for p in projections_mut(&mut gradients){p.weights.fill(0.0);p.biases.fill(0.0); if let Some(bn)=&mut p.batch_norm { bn.biases.fill(0.0); bn.scales.fill(0.0); bn.rolling_mean.fill(0.0); bn.rolling_variance.fill(0.0); }}
    let mut dx=vec![0.0;input.len()]; let mut carry=vec![0.0;len];
    for t in (0..s.steps).rev(){
        let step=&tape[t];
        let dh=(0..len).map(|i|grad.data[t*len+i]+carry[i]).collect::<Vec<_>>();
        let dz=(0..len).map(|i|dh[i]*(step.previous[i]-step.candidate[i])*step.z[i]*(1.0-step.z[i])).collect::<Vec<_>>();
        let dn=(0..len).map(|i|dh[i]*(1.0-step.z[i])*step.candidate[i]*(1.0-step.candidate[i])).collect::<Vec<_>>();
        let mut project=|x:&[f32],dy:&[f32],i:usize| {
            let width=if i.is_multiple_of(2){ins}else{outs};
            let p=projections_mut(&mut gradients).into_iter().nth(i).expect("GRU projection");
            let mut raw_dy=dy.to_vec();
            if let (Some(bn),Some(gbn))=(&pp[i].batch_norm,&mut p.batch_norm) {
                let raw=lightnet_cuda::host_ops::connected(x,&pp[i].weights,&vec![0.0;outs],batch,width,outs);
                for j in 0..dy.len() {
                    let c=j%outs; let inv=1.0/(bn.rolling_variance[c]+crate::params::BATCH_NORM_EPSILON).sqrt();
                    gbn.scales[c]+=dy[j]*(raw[j]-bn.rolling_mean[c])*inv;
                    raw_dy[j]*=bn.scales[c]*inv;
                }
            }
            let (dx,dw,_)=lightnet_cuda::host_ops::connected_backward(x,&pp[i].weights,&raw_dy,batch,width,outs);
            let mut db=vec![0.0;outs]; for (j,v) in dy.iter().enumerate() { db[j%outs]+=v; }
            for(a,b)in p.weights.iter_mut().zip(dw){*a+=b;}
            for(a,b)in p.biases.iter_mut().zip(db){*a+=b;}
            dx
        };
        let dreset=project(&step.reset,&dn,5);
        let dr=(0..len).map(|i|dreset[i]*step.previous[i]*step.r[i]*(1.0-step.r[i])).collect::<Vec<_>>();
        let dxh=project(&step.x,&dn,4);
        let dxr=project(&step.x,&dr,2); let dxz=project(&step.x,&dz,0);
        let dhr=project(&step.previous,&dr,3); let dhz=project(&step.previous,&dz,1);
        for i in 0..len{carry[i]=dh[i]*step.z[i]+dreset[i]*step.r[i]+dhr[i]+dhz[i];}
        for i in 0..batch*ins{dx[t*batch*ins+i]=dxh[i]+dxr[i]+dxz[i];}
    }
    Ok((Tensor::from_data(input.shape,dx)?,gradients))
}

