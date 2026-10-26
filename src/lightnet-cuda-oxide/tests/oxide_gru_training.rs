//! Unrun non-BN GRU full-sequence source fixtures. Rust CPU reference only.
use lightnet_core::{ConfigFile,ExecutableNetwork,GpuExecutableNetwork,LayerParams,
    LayerTrainBuffers,Network,Tensor};
use lightnet_cuda::F32TensorOpsBackend;
use lightnet_cuda_oxide::OxideBackend;

fn params(p:&LayerParams)->Vec<f32>{
    let LayerParams::Gru(p)=p else{panic!("GRU params")};
    [&p.input_z,&p.state_z,&p.input_r,&p.state_r,&p.input_h,&p.state_h].into_iter()
        .flat_map(|p|p.biases.iter().chain(&p.weights).copied()).collect()
}
fn updates(t:&LayerTrainBuffers)->Vec<f32>{
    let LayerTrainBuffers::Gru(t)=t else{panic!("GRU buffers")};
    [&t.input_z,&t.state_z,&t.input_r,&t.state_r,&t.input_h,&t.state_h].into_iter()
        .flat_map(|t|t.bias_updates.iter().chain(&t.weight_updates).copied()).collect()
}
fn close(label:&str,a:&[f32],b:&[f32],tol:f32){
    assert_eq!(a.len(),b.len(),"{label}");let scale=b.iter().fold(1.0_f32,|m,x|m.max(x.abs()));
    for(i,(&a,&b))in a.iter().zip(b).enumerate(){assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<=tol*scale,"{label}[{i}]: {a} vs {b}");}
}
fn fixture(optimizer:&str,amp:bool)->(ExecutableNetwork,GpuExecutableNetwork<OxideBackend>,Tensor){
    let cfg=format!("[net]\nwidth=2\nheight=1\nchannels=2\nbatch=2\ntime_steps=3\nloss_scale=8\ndecay=0.003\neps=0.001\n{optimizer}\n[gru]\noutput=3\n");
    let net=Network::from_config(ConfigFile::parse_str("gru-native.cfg",&cfg).unwrap()).unwrap();
    let mut cpu=ExecutableNetwork::new(net.clone());
    let LayerParams::Gru(p)=&mut cpu.params[0]else{panic!("GRU params")};
    for(slot,p)in [&mut p.input_z,&mut p.state_z,&mut p.input_r,&mut p.state_r,&mut p.input_h,&mut p.state_h].into_iter().enumerate(){
        for(i,x)in p.biases.iter_mut().chain(&mut p.weights).enumerate(){*x=(i as f32*0.23+slot as f32).sin()*0.2;}
    }
    let mut gpu=GpuExecutableNetwork::new(net,OxideBackend::new().unwrap().with_amp(amp));
    gpu.params=cpu.params.clone();gpu.init_train_buffers_gpu().unwrap();
    let shape=cpu.network.net.input_shape();
    let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).cos()*0.3).collect()).unwrap();
    (cpu,gpu,input)
}

#[test]
fn gru_three_step_native_bptt_resident_sgd_adam_muon_and_amp(){
    for amp in [false,true]{
        for optimizer in ["adam=0\n","adam=1\n","muon=1\ndecoupled_decay=1\n"]{
            let(mut cpu,mut gpu,input)=fixture(optimizer,amp);let original=gpu.params.clone();
            let tol=if amp{0.01}else{0.001};
            for iteration in 0..3 {
                let host=cpu.forward_training(&input).unwrap();let device=gpu.forward_training_device(&input).unwrap();
                let out=host.final_output().unwrap();
                close("GRU output",&gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(),&out.data,tol);
                let seed=Tensor::from_data(out.shape,vec![0.1;out.len()]).unwrap();
                let scaled=Tensor::from_data(out.shape,vec![0.8;out.len()]).unwrap();
                // Half/compact inference between training forward and BPTT must
                // read resident masters and preserve the forward-owned tape.
                gpu.forward_compact_f16(&input).unwrap();
                for _ in 0..2{
                    let want=cpu.backward_accumulate_seeded(&input,&host,&[Some(scaled.clone())]).unwrap();
                    let got=gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device,&[Some(seed.clone())]).unwrap().unwrap();
                    close("GRU input BPTT",&got.data,&want.input_gradient.data,tol);
                }
                if iteration==0{assert!(updates(&gpu.host_fallback_train_buffers()[0]).iter().all(|&x|x==0.0));}
                gpu.synchronize_resident_parameter_gradients().unwrap();
                close("GRU parameter BPTT",&updates(&gpu.host_fallback_train_buffers()[0]),&updates(&cpu.train_buffers[0]),tol);
                if cpu.network.net.muon{cpu.update_weights_muon(0.002,6).unwrap();}
                else if cpu.network.net.adam{cpu.update_weights_adam(0.002,6).unwrap();}
                else{cpu.update_weights(0.002/8.0,0.7,0.003*8.0,6);}
                gpu.update_weights_gpu_deferred(0.002,0.7,0.003,6).unwrap();
                assert_eq!(gpu.params,original);
                let want=cpu.forward(&input).unwrap().final_output().unwrap().clone();
                let got=gpu.forward_compact(&input).unwrap().final_output.unwrap();
                close("GRU deferred forward",&got.data,&want.data,tol);
            }
            let path=std::env::temp_dir().join(format!("gru-resume-{}-{amp}-{}.opt",cpu.network.net.muon,std::process::id()));
            gpu.save_optimizer_state(&path).unwrap();gpu.sync_host_weights_from_device().unwrap();
            close("GRU synchronized masters",&params(&gpu.params[0]),&params(&cpu.params[0]),tol);
            if cpu.network.net.adam||cpu.network.net.muon{
                let mut resumed=GpuExecutableNetwork::new(gpu.network.clone(),OxideBackend::new().unwrap().with_amp(amp));
                resumed.params=gpu.params.clone();resumed.init_train_buffers_gpu().unwrap();resumed.load_optimizer_state(&path).unwrap();
                let host=cpu.forward_training(&input).unwrap();let state=resumed.forward_training_device(&input).unwrap();
                let shape=host.final_output().unwrap().shape;let count=shape.batch*shape.element_count();
                cpu.backward_accumulate_seeded(&input,&host,&[Some(Tensor::from_data(shape,vec![0.8;count]).unwrap())]).unwrap();
                resumed.backward_accumulate_seeded_gpu(&state,&[Some(Tensor::from_data(shape,vec![0.1;count]).unwrap())]).unwrap();
                if cpu.network.net.muon{cpu.update_weights_muon(0.002,6).unwrap();}else{cpu.update_weights_adam(0.002,6).unwrap();}
                resumed.update_weights_gpu(0.002,0.7,0.003,6).unwrap();
                close("GRU checkpoint resume",&params(&resumed.params[0]),&params(&cpu.params[0]),tol);
            }
            std::fs::remove_file(path).unwrap();
        }
    }
}
