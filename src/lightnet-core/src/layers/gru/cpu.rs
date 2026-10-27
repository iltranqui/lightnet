//! Exact adjoint of the existing sigmoid-candidate GRU.
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
    validate_cpu(s,p,input,output)
}

pub(crate) fn validate_cpu(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
    if s.steps==0 || input.batch==0 || input.batch%s.steps!=0 || s.output==0
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

pub(crate) fn tape(s:&GruSpec,p:&GruParams,input:&Tensor,output:TensorShape)->Result<Vec<Step<Vec<f32>>>> {
    let batch=validate_cpu(s,p,input.shape,output)?;
    let (ins,outs)=(input.shape.element_count(),s.output);
    let len=batch*outs;
    let pp=projections(p);
    let lin=|x:&[f32],i:usize| {
        let width=if i%2==0{ins}else{outs};
        let mut y=lightnet_cuda::host_ops::connected(x,&pp[i].weights,&vec![0.0;outs],batch,width,outs);
        for (j,v) in y.iter_mut().enumerate() {
            let c=j%outs;
            if let Some(bn)=&pp[i].batch_norm { *v=(*v-bn.rolling_mean[c])/(bn.rolling_variance[c]+crate::runtime::BATCH_NORM_EPSILON).sqrt()*bn.scales[c]; }
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

pub(crate) fn backward(s:&GruSpec,p:&GruParams,input:&Tensor,grad:&Tensor)->Result<(Tensor,GruParams)> {
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
            let width=if i%2==0{ins}else{outs};
            let p=projections_mut(&mut gradients).into_iter().nth(i).expect("GRU projection");
            let mut raw_dy=dy.to_vec();
            if let (Some(bn),Some(gbn))=(&pp[i].batch_norm,&mut p.batch_norm) {
                let raw=lightnet_cuda::host_ops::connected(x,&pp[i].weights,&vec![0.0;outs],batch,width,outs);
                for j in 0..dy.len() {
                    let c=j%outs; let inv=1.0/(bn.rolling_variance[c]+crate::runtime::BATCH_NORM_EPSILON).sqrt();
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
    #[test]
    fn gru_bn_adjoint_each_timestep_and_three_step_bptt() {
        for steps in [1, 3] {
            let cfg=format!("[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\ntime_steps={steps}\n[gru]\noutput=2\nbatch_normalize=1\n");
            let net=Network::from_config(ConfigFile::parse_str("gru-bn.cfg",&cfg).unwrap()).unwrap();
            let mut cpu=ExecutableNetwork::new(net);
            let LayerParams::Gru(p)=&mut cpu.params[0] else { unreachable!() };
            for (slot,p) in projections_mut(p).into_iter().enumerate() {
                for (i,v) in p.biases.iter_mut().chain(&mut p.weights).enumerate() { *v=((slot*7+i) as f32*0.31).sin()*0.3; }
                let bn=p.batch_norm.as_mut().unwrap();
                for i in 0..2 { bn.scales[i]=0.7+i as f32*0.3; bn.rolling_mean[i]=0.1*(slot as f32-2.0); bn.rolling_variance[i]=0.6+i as f32*0.4; }
            }
            let shape=cpu.network.net.input_shape();
            let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.4).cos()*0.4).collect()).unwrap();
            let outshape=cpu.network.layers[0].output_shape;
            for target in 0..steps {
                let mut seed=Tensor::zeros(outshape);
                for i in target*2*2..(target+1)*2*2 { seed.data[i]=(i as f32*0.3+0.2).sin(); }
                let f=cpu.forward_training(&input).unwrap();
                let result=cpu.backward_seeded(&input,&f,&[Some(seed.clone())]).unwrap();
                let objective=|net:&ExecutableNetwork,x:&Tensor| net.forward(x).unwrap().final_output().unwrap().data.iter().zip(&seed.data).map(|(a,b)|a*b).sum::<f32>();
                let eps=0.001;
                for i in 0..input.len() {
                    let mut a=input.clone(); let mut b=input.clone(); a.data[i]+=eps; b.data[i]-=eps;
                    let n=(objective(&cpu,&a)-objective(&cpu,&b))/(2.0*eps);
                    assert!((n-result.input_gradient.data[i]).abs()<5e-4,"steps {steps} target {target} input {i}: {n} vs {}",result.input_gradient.data[i]);
                }
                let LayerParams::Gru(g)=&result.param_gradients[0] else { unreachable!() };
                for slot in 0..6 { for group in 0..3 {
                    let expected=match group {0=>projections(g)[slot].weights.clone(),1=>projections(g)[slot].biases.clone(),_=>projections(g)[slot].batch_norm.as_ref().unwrap().scales.clone()};
                    for (i,want) in expected.iter().enumerate() {
                        let change=|cpu:&mut ExecutableNetwork,delta:f32| {
                            let LayerParams::Gru(p)=&mut cpu.params[0] else { unreachable!() };
                            let p=projections_mut(p).into_iter().nth(slot).unwrap();
                            match group {0=>p.weights[i]+=delta,1=>p.biases[i]+=delta,_=>p.batch_norm.as_mut().unwrap().scales[i]+=delta}
                        };
                        change(&mut cpu,eps); let a=objective(&cpu,&input); change(&mut cpu,-2.0*eps); let b=objective(&cpu,&input); change(&mut cpu,eps);
                        let n=(a-b)/(2.0*eps);
                        assert!((n-want).abs()<5e-4,"steps {steps} target {target} slot {slot} group {group} index {i}: {n} vs {want}");
                    }
                }}
            }
        }
    }

    #[test]
    fn gru_bn_rejects_inconsistent_projection_and_preserves_gpu_rejection() {
        let net=Network::from_config(ConfigFile::parse_str("gru-bn-invalid.cfg", "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\n[gru]\noutput=2\nbatch_normalize=1\n").unwrap()).unwrap();
        let mut cpu=ExecutableNetwork::new(net);
        let crate::LayerKind::Gru(spec)=&cpu.network.layers[0].kind else { unreachable!() };
        let shape=cpu.network.layers[0].input_shape; let out=cpu.network.layers[0].output_shape;
        let LayerParams::Gru(p)=&mut cpu.params[0] else { unreachable!() };
        assert!(validate(spec,p,shape,out).is_err());
        p.state_h.batch_norm=None;
        assert!(validate_cpu(spec,p,shape,out).is_err());
        p.state_h.batch_norm=p.input_h.batch_norm.clone();
        p.state_h.batch_norm.as_mut().unwrap().scales.clear();
        assert!(validate_cpu(spec,p,shape,out).is_err());
    }

    #[test]
    fn gru_bn_accumulation_updates_stats_after_backward_and_optimizers() {
        for optimizer in 0..3 {
            let mode=match optimizer { 1=>"adam=1\n", 2=>"muon=1\n", _=>"" };
            let cfg=format!("[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\ntime_steps=3\n{mode}[gru]\noutput=2\nbatch_normalize=1\n");
            let net=Network::from_config(ConfigFile::parse_str("gru-bn-update.cfg",&cfg).unwrap()).unwrap();
            let mut cpu=ExecutableNetwork::new(net);
            let LayerParams::Gru(p)=&mut cpu.params[0] else { unreachable!() };
            for (slot,p) in projections_mut(p).into_iter().enumerate() {
                for (i,v) in p.weights.iter_mut().enumerate() { *v=0.1+0.02*(slot+i) as f32; }
                let bn=p.batch_norm.as_mut().unwrap(); bn.scales.fill(0.8); bn.rolling_mean.fill(0.2); bn.rolling_variance.fill(1.0);
            }
            let shape=cpu.network.net.input_shape();
            let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|0.03*i as f32+0.1).collect()).unwrap();
            let seed=Tensor::from_data(cpu.network.layers[0].output_shape,vec![0.5;shape.batch*2]).unwrap();
            let forward=cpu.forward_training(&input).unwrap();
            let before=cpu.backward_seeded(&input,&forward,&[Some(seed.clone())]).unwrap();
            let got=cpu.backward_accumulate_seeded(&input,&forward,&[Some(seed)]).unwrap();
            assert_eq!(before.input_gradient,got.input_gradient);
            let LayerParams::Gru(g)=&got.param_gradients[0] else { unreachable!() };
            let LayerParams::Gru(p)=&cpu.params[0] else { unreachable!() };
            assert_ne!(p.input_h.batch_norm.as_ref().unwrap().rolling_mean[0],0.2);
            let old=p.input_h.batch_norm.as_ref().unwrap().scales[0];
            assert!(g.input_h.batch_norm.as_ref().unwrap().scales[0].abs()>1e-7);
            match optimizer { 0=>cpu.update_weights(0.01,0.0,0.0,1), 1=>cpu.update_weights_adam(0.01,1).unwrap(), _=>cpu.update_weights_muon(0.01,1).unwrap() }
            let LayerParams::Gru(p)=&cpu.params[0] else { unreachable!() };
            assert_ne!(p.input_h.batch_norm.as_ref().unwrap().scales[0],old,"optimizer {optimizer}");
        }
    }

}
