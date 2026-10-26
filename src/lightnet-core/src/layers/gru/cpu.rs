//! Exact adjoint of the existing sigmoid-candidate GRU, without BN.
//! Sequence batch is time-major: [step, sequence_batch, feature].
use crate::error::{Error,Result};
use crate::model::{GruSpec,TensorShape};
use crate::runtime::{ConnectedParams,GruParams,Tensor};

#[derive(Clone,Debug)]
pub(crate) struct Step<T> {
    pub x:T, pub previous:T, pub z:T, pub r:T, pub candidate:T, pub reset:T,
}

pub(crate) fn projections(p:&GruParams)->[&ConnectedParams;6] {
    [&p.input_z,&p.state_z,&p.input_r,&p.state_r,&p.input_h,&p.state_h]
}
pub(crate) fn projections_mut(p:&mut GruParams)->[&mut ConnectedParams;6] {
    [&mut p.input_z,&mut p.state_z,&mut p.input_r,&mut p.state_r,&mut p.input_h,&mut p.state_h]
}
pub(crate) fn serialized(p:&GruParams)->Vec<f32> {
    projections(p).into_iter().flat_map(|p|p.biases.iter().chain(&p.weights).copied()).collect()
}
pub(crate) fn set_serialized(p:&mut GruParams,values:&[f32])->Result<()> {
    let total=projections(p).iter().map(|p|p.biases.len()+p.weights.len()).sum::<usize>();
    if total!=values.len(){return Err(Error::InvalidConfig("resident GRU parameter length mismatch".into()));}
    let mut cursor=0;
    for p in projections_mut(p){
        let end=cursor+p.biases.len();p.biases.copy_from_slice(&values[cursor..end]);cursor=end;
        let end=cursor+p.weights.len();p.weights.copy_from_slice(&values[cursor..end]);cursor=end;
    }
    Ok(())
}
pub(crate) fn validate(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
    if s.steps==0 || input.batch==0 || input.batch%s.steps!=0 || s.output==0
        || output!=TensorShape::new(1,1,s.output,input.batch) {
        return Err(Error::InvalidConfig("GRU sequence/output geometry mismatch".into()));
    }
    if s.batch_normalize || projections(p).iter().any(|p|p.batch_norm.is_some()) {
        return Err(Error::InvalidConfig("GRU BN training backward is not implemented".into()));
    }
    for(i,p)in projections(p).iter().enumerate(){
        let ins=if i%2==0{input.element_count()}else{s.output};
        if p.weights.len()!=ins*s.output || p.biases.len()!=s.output {
            return Err(Error::InvalidConfig("GRU projection geometry mismatch".into()));
        }
    }
    Ok(input.batch/s.steps)
}

pub(crate) fn backward(s:&GruSpec,p:&GruParams,input:&Tensor,grad:&Tensor)->Result<(Tensor,GruParams)> {
    let batch=validate(s,p,input.shape,grad.shape)?;
    let (ins,outs)=(input.shape.element_count(),s.output);
    let len=batch*outs;
    let pp=projections(p);
    let lin=|x:&[f32],i:usize|lightnet_cuda::host_ops::connected(x,&pp[i].weights,&pp[i].biases,
        batch,if i%2==0{ins}else{outs},outs);
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
    let mut gradients=p.clone();
    for p in projections_mut(&mut gradients){p.weights.fill(0.0);p.biases.fill(0.0);}
    let mut dx=vec![0.0;input.len()]; let mut carry=vec![0.0;len];
    for t in (0..s.steps).rev(){
        let step=&tape[t];
        let dh=(0..len).map(|i|grad.data[t*len+i]+carry[i]).collect::<Vec<_>>();
        let dz=(0..len).map(|i|dh[i]*(step.previous[i]-step.candidate[i])*step.z[i]*(1.0-step.z[i])).collect::<Vec<_>>();
        let dn=(0..len).map(|i|dh[i]*(1.0-step.z[i])*step.candidate[i]*(1.0-step.candidate[i])).collect::<Vec<_>>();
        let mut project=|x:&[f32],dy:&[f32],i:usize| {
            let (dx,dw,db)=lightnet_cuda::host_ops::connected_backward(x,&pp[i].weights,dy,batch,if i%2==0{ins}else{outs},outs);
            let p=projections_mut(&mut gradients).into_iter().nth(i).expect("GRU projection");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigFile,ExecutableNetwork,LayerParams,Network};

    #[test]
    fn gru_exact_adjoint_all_projection_blocks_and_time_major_input() {
        let net=Network::from_config(ConfigFile::parse_str("gru-gradient.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=2\nbatch=2\ntime_steps=3\n[gru]\noutput=3\n").unwrap()).unwrap();
        let mut cpu=ExecutableNetwork::new(net);
        let LayerParams::Gru(p)=&mut cpu.params[0]else{panic!("GRU params")};
        for (slot,p)in projections_mut(p).into_iter().enumerate(){
            for(i,x)in p.biases.iter_mut().chain(&mut p.weights).enumerate(){*x=(i as f32*0.23+slot as f32).sin()*0.2;}
        }
        let shape=cpu.network.net.input_shape();
        let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).cos()*0.3).collect()).unwrap();
        let outshape=cpu.network.layers[0].output_shape;
        let seed=Tensor::from_data(outshape,(0..outshape.batch*outshape.element_count()).map(|i|(i as f32*0.31).sin()*0.2).collect()).unwrap();
        let forward=cpu.forward_training(&input).unwrap();
        let result=cpu.backward_seeded(&input,&forward,&[Some(seed.clone())]).unwrap();
        let objective=|net:&ExecutableNetwork,x:&Tensor|net.forward(x).unwrap().final_output().unwrap().data.iter()
            .zip(&seed.data).map(|(x,g)|x*g).sum::<f32>();
        let eps=0.001;
        for i in 0..input.len(){
            let mut plus=input.clone();plus.data[i]+=eps;let mut minus=input.clone();minus.data[i]-=eps;
            let numeric=(objective(&cpu,&plus)-objective(&cpu,&minus))/(2.0*eps);
            assert!((numeric-result.input_gradient.data[i]).abs()<3e-4,"GRU input {i}");
        }
        let LayerParams::Gru(gradient)=&result.param_gradients[0]else{panic!("GRU gradients")};
        let expected=serialized(gradient);
        let original=match &cpu.params[0]{LayerParams::Gru(p)=>serialized(p),_=>unreachable!()};
        for i in 0..original.len(){
            let mut blob=original.clone();blob[i]+=eps;
            let LayerParams::Gru(p)=&mut cpu.params[0]else{unreachable!()};set_serialized(p,&blob).unwrap();
            let plus=objective(&cpu,&input);blob[i]-=2.0*eps;
            let LayerParams::Gru(p)=&mut cpu.params[0]else{unreachable!()};set_serialized(p,&blob).unwrap();
            let minus=objective(&cpu,&input);
            assert!(((plus-minus)/(2.0*eps)-expected[i]).abs()<3e-4,"GRU serialized parameter {i}");
        }
    }
}
