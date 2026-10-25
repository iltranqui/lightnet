//! Unrun native static recurrence parity fixtures. Controller/lattice/BN
//! body variants remain explicit CPU fallbacks. No compiled-C++ golden.
use lightnet_core::{ConfigFile,ExecutableNetwork,GpuExecutableNetwork,LayerParams,LayerTrainBuffers,Network,Tensor};
use lightnet_cuda::F32TensorOpsBackend;
use lightnet_cuda_oxide::OxideBackend;

fn close(name:&str,got:&[f32],expected:&[f32]) {
    assert_eq!(got.len(),expected.len(),"{name}");
    let scale=expected.iter().fold(1.0_f32,|m,x|m.max(x.abs()));
    for(i,(&a,&b))in got.iter().zip(expected).enumerate(){
        assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<=8e-4*scale,"{name}[{i}]: {a} vs {b}");
    }
}
fn initialize(p:&mut LayerParams) {
    let (weights,biases)=match p {
        LayerParams::Convolutional(p)=>(&mut p.weights,&mut p.biases),
        LayerParams::Connected(p)=>(&mut p.weights,&mut p.biases),
        _=>panic!("body params"),
    };
    for(i,x)in weights.iter_mut().enumerate(){*x=(i as f32*0.31+0.2).sin()*0.2;}
    for(i,x)in biases.iter_mut().enumerate(){*x=(i as f32*0.19+0.1).cos()*0.1;}
}
fn params(p:&LayerParams)->Vec<f32>{
    match p{
        LayerParams::RecursiveBlock(p)=>{
            let mut out=[p.controller_weights.as_slice(),p.controller_biases.as_slice()].concat();
            for body in &p.body{out.extend(params(body));}out
        }
        LayerParams::Convolutional(p)=>[p.weights.as_slice(),p.biases.as_slice()].concat(),
        LayerParams::Connected(p)=>[p.weights.as_slice(),p.biases.as_slice()].concat(),
        _=>panic!("body params"),
    }
}
fn updates(t:&LayerTrainBuffers)->Vec<f32>{
    match t{
        LayerTrainBuffers::RecursiveBlock(t)=>{
            let mut out=[t.controller_weight_updates.as_slice(),t.controller_bias_updates.as_slice()].concat();
            for body in &t.body{out.extend(updates(body));}out
        }
        LayerTrainBuffers::Convolutional(t)=>[t.weight_updates.as_slice(),t.bias_updates.as_slice()].concat(),
        LayerTrainBuffers::Connected(t)=>[t.weight_updates.as_slice(),t.bias_updates.as_slice()].concat(),
        _=>panic!("body buffers"),
    }
}

#[test]
fn static_recursive_body_shares_gradients_across_loops_and_accumulated_backwards(){
    for(connected,injection,optimizer)in[(false,0,""),(false,1,"adam=1\n"),(true,1,"muon=1\ndecoupled_decay=1\n")]{
        let (w,h,c,body)=if connected{(1,1,3,"[connected]\noutput=3\nactivation=tanh\n[connected]\noutput=3\nactivation=linear\n")}
            else{(3,2,2,"[convolutional]\nfilters=2\nsize=3\npad=1\nactivation=leaky\n[convolutional]\nfilters=2\nsize=1\nactivation=tanh\n")};
        let text=format!("[net]\nbatch=2\nwidth={w}\nheight={h}\nchannels={c}\n{optimizer}\n[recursive_block]\nbody=2\nloops=3\ninjection={injection}\nbody_scale=0.4\nresidual_scale=0.3\ninjection_scale=0.2\n{body}");
        let network=Network::from_config(ConfigFile::parse_str("native-recursive.cfg",&text).unwrap()).unwrap();
        let mut cpu=ExecutableNetwork::new(network.clone());
        let LayerParams::RecursiveBlock(p)=&mut cpu.params[0]else{panic!("recursive params")};
        for body in &mut p.body{initialize(body);}
        let mut gpu=GpuExecutableNetwork::new(network,OxideBackend::new().unwrap());
        gpu.params=cpu.params.clone();gpu.init_train_buffers_gpu().unwrap();
        let shape=cpu.network.net.input_shape();
        let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).cos()*0.3).collect()).unwrap();
        for _ in 0..2{
            cpu.zero_train_buffers();gpu.zero_train_buffers_gpu().unwrap();
            let host=cpu.forward_training(&input).unwrap();
            let state=gpu.forward_training_device(&input).unwrap();
            assert!(state.cpu_forward.layer_outputs[0].data.is_empty());
            let expected=host.final_output().unwrap();
            close("output",&gpu.backend.download_f32(&state.gpu_layer_outputs[0]).unwrap(),&expected.data);
            close("compact",&gpu.forward_compact(&input).unwrap().final_output.unwrap().data,&expected.data);
            let seed=Tensor::from_data(expected.shape,(0..expected.len()).map(|i|(i as f32*0.11).sin()*0.2).collect()).unwrap();
            let seeds=[Some(seed)];
            for _ in 0..2{
                let expected=cpu.backward_accumulate_seeded(&input,&host,&seeds).unwrap();
                let got=gpu.backward_accumulate_seeded_gpu_with_input_gradient(&state,&seeds).unwrap().unwrap();
                close("input adjoint",&got.data,&expected.input_gradient.data);
            }
            close("shared parameter accumulation",&updates(&gpu.host_fallback_train_buffers()[0]),&updates(&cpu.train_buffers[0]));
            cpu.update_weights(0.003,0.9,0.0005,2);gpu.update_weights_gpu(0.003,0.9,0.0005,2).unwrap();
            close("updates",&params(&gpu.params[0]),&params(&cpu.params[0]));
        }
    }
}
