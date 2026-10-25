//! Unrun source fixtures for device-resident gradients, SGD/Adam parameters,
//! deferred synchronization and DNOPT resume. Rust CPU reference; no C++ golden.
use lightnet_core::{ConfigFile,ExecutableNetwork,GpuExecutableNetwork,LayerParams,
    LayerTrainBuffers,Network,Tensor};
use lightnet_cuda::F32TensorOpsBackend;
use lightnet_cuda_oxide::OxideBackend;

fn close(label:&str,a:&[f32],b:&[f32]){
    assert_eq!(a.len(),b.len(),"{label}");
    let scale=b.iter().fold(1.0_f32,|m,x|m.max(x.abs()));
    for(i,(&a,&b))in a.iter().zip(b).enumerate(){
        assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<=1e-3*scale,"{label}[{i}]: {a} vs {b}");
    }
}
fn fixture(optimizer:&str,layer_options:&str)->(ExecutableNetwork,GpuExecutableNetwork<OxideBackend>,Tensor){
    let cfg=format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=3\nloss_scale=8\ndecay=0.003\n{optimizer}\n[mambavision]\nfilters=4\nstate=2\nconv_size=3\ndt_rank=2\nffn_ratio=2\nactivation=gelu\ncausal=1\n{layer_options}\n");
    let net=Network::from_config(ConfigFile::parse_str("resident-mamba.cfg",&cfg).unwrap()).unwrap();
    let mut cpu=ExecutableNetwork::new(net.clone());
    let LayerParams::MambaVision(p)=&mut cpu.params[0]else{panic!("Mamba params")};
    for(i,x)in p.values.iter_mut().enumerate(){*x=(i as f32*0.13+0.2).sin()*0.2;}
    let mut gpu=GpuExecutableNetwork::new(net,OxideBackend::new().unwrap());
    gpu.params=cpu.params.clone();gpu.init_train_buffers_gpu().unwrap();
    let shape=cpu.network.net.input_shape();
    let input=Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).cos()*0.2).collect()).unwrap();
    (cpu,gpu,input)
}
fn values(p:&LayerParams)->&[f32]{let LayerParams::MambaVision(p)=p else{panic!("Mamba params")};&p.values}
fn updates(t:&LayerTrainBuffers)->&[f32]{let LayerTrainBuffers::MambaVision(t)=t else{panic!("Mamba buffers")};&t.updates}

fn backward(cpu:&mut ExecutableNetwork,gpu:&mut GpuExecutableNetwork<OxideBackend>,input:&Tensor){
    let host=cpu.forward_training(input).unwrap();
    let device=gpu.forward_training_device(input).unwrap();
    let out=host.final_output().unwrap();
    close("resident forward",&gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(),&out.data);
    let seed=Tensor::from_data(out.shape,(0..out.len()).map(|i|(i as f32*0.19).sin()*0.1).collect()).unwrap();
    let seeds=[Some(seed)];
    for _ in 0..2{
        let host_seeds=seeds.iter().map(|s|s.as_ref().map(|s|Tensor::from_data(s.shape,
            s.data.iter().map(|x|x*gpu.loss_scale()).collect()).unwrap())).collect::<Vec<_>>();
        cpu.backward_accumulate_seeded(input,&host,&host_seeds).unwrap();
        gpu.backward_accumulate_seeded_gpu(&device,&seeds).unwrap();
    }
    assert!(!gpu.gradients_non_finite().unwrap());
}

fn cpu_step(cpu:&mut ExecutableNetwork){
    if cpu.network.net.adam {cpu.update_weights_adam(0.002,2).unwrap();}
    else {let scale=cpu.network.net.loss_scale;cpu.update_weights(0.002/scale,0.7,0.003*scale,2);}
}

#[test]
fn deferred_native_sgd_adam_and_adamw_reuse_weights_gradients_and_moments(){
    for optimizer in ["adam=0\n","adam=1\ndecoupled_decay=0\nb1=0.8\nb2=0.95\neps=0.001\n",
        "adam=1\ndecoupled_decay=1\nb1=0.8\nb2=0.95\neps=0.001\n"]{
        let(mut cpu,mut gpu,input)=fixture(optimizer,"");
        let original=gpu.params.clone();
        for iteration in 0..3{
            // No gradient reset here: SGD's momentum accumulator must survive
            // optimizer steps and subsequent native forwards/backwards.
            backward(&mut cpu,&mut gpu,&input);
            if iteration==0{
                assert!(updates(&gpu.host_fallback_train_buffers()[0]).iter().all(|&x|x==0.0),
                    "native backward must not silently download parameter gradients");
            }
            gpu.synchronize_resident_parameter_gradients().unwrap();
            close("accumulated direction",updates(&gpu.host_fallback_train_buffers()[0]),updates(&cpu.train_buffers[0]));
            cpu_step(&mut cpu);
            gpu.update_weights_gpu_deferred(0.002,0.7,0.003,2).unwrap();
            assert_eq!(gpu.params,original,"deferred native steps leave public host mirrors unchanged");
            let want=cpu.forward(&input).unwrap().final_output().unwrap().clone();
            let got=gpu.forward_compact(&input).unwrap().final_output.unwrap();
            close("next forward reads resident updated weights",&got.data,&want.data);
        }
        gpu.sync_host_weights_from_device().unwrap();
        close("synchronized parameters",values(&gpu.params[0]),values(&cpu.params[0]));
        close("synchronized momentum",updates(&gpu.host_fallback_train_buffers()[0]),updates(&cpu.train_buffers[0]));
    }
}

#[test]
fn deferred_adam_sidecar_materializes_current_device_moments_and_resumes(){
    let(mut cpu,mut gpu,input)=fixture("adam=1\ndecoupled_decay=1\nb1=0.8\nb2=0.95\neps=0.001\n","");
    for _ in 0..2{
        backward(&mut cpu,&mut gpu,&input);
        cpu_step(&mut cpu);
        gpu.update_weights_gpu_deferred(0.002,0.7,0.003,2).unwrap();
    }
    let path=std::env::temp_dir().join(format!("resident-mamba-adam-{}.opt",std::process::id()));
    gpu.save_optimizer_state(&path).unwrap(); // Before host weight/moment sync.
    gpu.sync_host_weights_from_device().unwrap();
    let mut resumed=GpuExecutableNetwork::new(gpu.network.clone(),OxideBackend::new().unwrap());
    resumed.params=gpu.params.clone();resumed.init_train_buffers_gpu().unwrap();
    resumed.load_optimizer_state(&path).unwrap();
    assert_eq!(resumed.adam_step(),2);
    backward(&mut cpu,&mut resumed,&input);
    cpu_step(&mut cpu);
    resumed.update_weights_gpu(0.002,0.7,0.003,2).unwrap();
    close("resumed Adam weights",values(&resumed.params[0]),values(&cpu.params[0]));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn public_host_edits_cannot_overwrite_unsynchronized_device_steps(){
    let(mut cpu,mut gpu,input)=fixture("adam=0\n","");
    let before=gpu.params.clone();
    backward(&mut cpu,&mut gpu,&input);
    gpu.update_weights_gpu_deferred(0.002,0.7,0.003,2).unwrap();
    let LayerParams::MambaVision(p)=&mut gpu.params[0]else{panic!("Mamba params")};
    p.values[0]+=1.0;
    let error=gpu.forward_compact(&input).unwrap_err().to_string();
    assert!(error.contains("synchronize"),"{error}");
    gpu.params=before;gpu.sync_host_weights_from_device().unwrap();
    let old=gpu.params.clone();
    let LayerParams::MambaVision(p)=&mut gpu.params[0]else{panic!("Mamba params")};
    p.values[0]+=0.1;
    gpu.forward_compact(&input).unwrap(); // A current host mirror may be edited.
    assert_ne!(gpu.params,old);
}

#[test]
fn resident_update_gates_retain_pending_directions_until_released(){
    for (net_options,layer_options,release) in [
        ("adam=1\n","dont_update=1\n",Some(1)),
        ("adam=0\nburn_in=4\n","burnin_update=1\n",Some(4)),
        ("adam=1\n","train_only_bn=1\n",None),
    ]{
        let(mut cpu,mut gpu,input)=fixture(net_options,layer_options);
        let original=cpu.params.clone();
        backward(&mut cpu,&mut gpu,&input);
        cpu_step(&mut cpu);gpu.update_weights_gpu_deferred(0.002,0.7,0.003,2).unwrap();
        gpu.sync_host_weights_from_device().unwrap();
        assert_eq!(gpu.params,original,"gated native update must not change parameters");
        close("gated pending gradients",updates(&gpu.host_fallback_train_buffers()[0]),updates(&cpu.train_buffers[0]));
        if let Some(iteration)=release{
            cpu.set_train_iteration(iteration);gpu.set_train_iteration(iteration);
            cpu_step(&mut cpu);gpu.update_weights_gpu(0.002,0.7,0.003,2).unwrap();
            close("released gated update",values(&gpu.params[0]),values(&cpu.params[0]));
            assert_ne!(gpu.params,original);
        }
    }
}

#[test]
fn resident_nonfinite_gradients_reset_without_changing_parameters(){
    let(_,mut gpu,input)=fixture("adam=1\n","");
    let original=gpu.params.clone();
    let state=gpu.forward_training_device(&input).unwrap();
    let shape=gpu.network.layers[0].output_shape;
    let seeds=[Some(Tensor::from_data(shape,vec![f32::NAN;shape.batch*shape.element_count()]).unwrap())];
    gpu.backward_accumulate_seeded_gpu(&state,&seeds).unwrap();
    assert!(gpu.gradients_non_finite().unwrap());
    gpu.zero_train_buffers_gpu().unwrap();
    assert!(!gpu.gradients_non_finite().unwrap());
    assert_eq!(gpu.params,original);
}

#[test]
fn resident_backward_rejects_a_forward_from_before_the_update(){
    let(_,mut gpu,input)=fixture("adam=0\n","");
    let state=gpu.forward_training_device(&input).unwrap();
    let shape=gpu.network.layers[0].output_shape;
    let seeds=[Some(Tensor::from_data(shape,vec![0.1;shape.batch*shape.element_count()]).unwrap())];
    gpu.backward_accumulate_seeded_gpu(&state,&seeds).unwrap();
    gpu.update_weights_gpu(0.002,0.7,0.003,2).unwrap();
    let error=gpu.backward_accumulate_seeded_gpu(&state,&seeds).unwrap_err().to_string();
    assert!(error.contains("snapshot"),"{error}");
}

#[test]
fn native_nonfinite_reduction_checks_partial_chunk_tail(){
    let be=OxideBackend::new().unwrap();
    for n in [1,255,256,257,1025]{
        let mut data=vec![0.25;n];
        let input=be.upload_f32(&[n],&data).unwrap();
        assert!(!be.any_non_finite_f32(&input).unwrap());
        for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY]{
            data[n-1]=bad;
            let input=be.upload_f32(&[n],&data).unwrap();
            assert!(be.any_non_finite_f32(&input).unwrap());
        }
    }
}

fn kimi_fixture(optimizer: &str) -> (ExecutableNetwork, GpuExecutableNetwork<OxideBackend>, Tensor) {
    let cfg = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=3\nloss_scale=8\ndecay=0.03\n{optimizer}\n[kimi_kda]\nfilters=3\nrank=2\nexperts=3\ntop_k=2\nhidden=4\ntemperature=0.7\nload_balance=0.03\nactivation=gelu\n");
    let net = Network::from_config(ConfigFile::parse_str("resident-kimi.cfg", &cfg).unwrap()).unwrap();
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::KimiKda(p) = &mut cpu.params[0] else { panic!("Kimi params") };
    for (i, x) in p.biases.iter_mut().chain(&mut p.weights).enumerate() {
        *x = (i as f32 * 0.13 + 0.2).sin() * 0.2;
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    let shape = cpu.network.net.input_shape();
    let input = Tensor::from_data(shape, (0..shape.batch * shape.element_count())
        .map(|i| (i as f32 * 0.17).cos() * 0.2).collect()).unwrap();
    (cpu, gpu, input)
}

fn kimi_values(p: &LayerParams) -> Vec<f32> {
    let LayerParams::KimiKda(p) = p else { panic!("Kimi params") };
    p.to_blob().unwrap()
}

fn kimi_updates(t: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::KimiKda(t) = t else { panic!("Kimi buffers") };
    [t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()
}

#[test]
fn kimi_resident_deferred_updates_preserve_decay_groups_and_momentum() {
    for optimizer in ["adam=0\n", "adam=1\ndecoupled_decay=0\neps=0.001\n",
        "adam=1\ndecoupled_decay=1\neps=0.001\n"] {
        let (mut cpu, mut gpu, input) = kimi_fixture(optimizer);
        let original = gpu.params.clone();
        for iteration in 0..3 {
            backward(&mut cpu, &mut gpu, &input);
            if iteration == 0 {
                assert!(kimi_updates(&gpu.host_fallback_train_buffers()[0]).iter().all(|&x| x == 0.0));
            }
            gpu.synchronize_resident_parameter_gradients().unwrap();
            close("Kimi resident gradients", &kimi_updates(&gpu.host_fallback_train_buffers()[0]),
                &kimi_updates(&cpu.train_buffers[0]));
            if cpu.network.net.adam { cpu.update_weights_adam(0.002, 2).unwrap(); }
            else { cpu.update_weights(0.002 / 8.0, 0.7, 0.03 * 8.0, 2); }
            gpu.update_weights_gpu_deferred(0.002, 0.7, 0.03, 2).unwrap();
            assert_eq!(gpu.params, original);
            let want = cpu.forward(&input).unwrap().final_output().unwrap().clone();
            let got = gpu.forward_compact(&input).unwrap().final_output.unwrap();
            close("Kimi resident inference", &got.data, &want.data);
        }
        gpu.sync_host_weights_from_device().unwrap();
        close("Kimi resident weights", &kimi_values(&gpu.params[0]), &kimi_values(&cpu.params[0]));
        close("Kimi resident momentum", &kimi_updates(&gpu.host_fallback_train_buffers()[0]),
            &kimi_updates(&cpu.train_buffers[0]));
    }
}

#[test]
fn kimi_resident_adam_sidecar_uses_root_bias_and_weight_slots() {
    for decay_mode in [0, 1] {
        let (mut cpu, mut gpu, input) = kimi_fixture(&format!("adam=1\ndecoupled_decay={decay_mode}\neps=0.001\n"));
        backward(&mut cpu, &mut gpu, &input);
        cpu.update_weights_adam(0.002, 2).unwrap();
        gpu.update_weights_gpu_deferred(0.002, 0.7, 0.03, 2).unwrap();
        let path = std::env::temp_dir().join(format!("resident-kimi-{decay_mode}-{}.opt", std::process::id()));
        gpu.save_optimizer_state(&path).unwrap();
        gpu.sync_host_weights_from_device().unwrap();
        let mut resumed = GpuExecutableNetwork::new(gpu.network.clone(), OxideBackend::new().unwrap());
        resumed.params = gpu.params.clone();
        resumed.init_train_buffers_gpu().unwrap();
        resumed.load_optimizer_state(&path).unwrap();
        backward(&mut cpu, &mut resumed, &input);
        cpu.update_weights_adam(0.002, 2).unwrap();
        resumed.update_weights_gpu(0.002, 0.7, 0.03, 2).unwrap();
        close("Kimi resumed moments", &kimi_values(&resumed.params[0]), &kimi_values(&cpu.params[0]));
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn kimi_resident_gates_reset_stale_cache_and_public_edit_boundaries() {
    let (mut cpu, mut gpu, input) = kimi_fixture("adam=1\n");
    cpu.network.layers[0].train.dont_update = 1;
    gpu.network.layers[0].train.dont_update = 1;
    let original = gpu.params.clone();
    backward(&mut cpu, &mut gpu, &input);
    cpu.update_weights_adam(0.002, 2).unwrap();
    gpu.update_weights_gpu_deferred(0.002, 0.7, 0.03, 2).unwrap();
    gpu.synchronize_resident_parameter_gradients().unwrap();
    assert_eq!(gpu.params, original);
    close("Kimi gated gradients", &kimi_updates(&gpu.host_fallback_train_buffers()[0]),
        &kimi_updates(&cpu.train_buffers[0]));
    cpu.set_train_iteration(1);
    gpu.set_train_iteration(1);
    let old_forward = gpu.forward_training_device(&input).unwrap();
    cpu.update_weights_adam(0.002, 2).unwrap();
    gpu.update_weights_gpu_deferred(0.002, 0.7, 0.03, 2).unwrap();
    let shape = gpu.network.layers[0].output_shape;
    let seeds = [Some(Tensor::from_data(shape, vec![0.1; shape.batch * shape.element_count()]).unwrap())];
    let error = gpu.backward_accumulate_seeded_gpu(&old_forward, &seeds).unwrap_err().to_string();
    assert!(error.contains("snapshot"), "{error}");
    let LayerParams::KimiKda(p) = &mut gpu.params[0] else { panic!("Kimi params") };
    p.biases[0] += 0.1;
    let error = gpu.forward_compact_f16(&input).unwrap_err().to_string();
    assert!(error.contains("synchronize"), "{error}");
    gpu.params = original;
    gpu.sync_host_weights_from_device().unwrap();
    close("Kimi gate release", &kimi_values(&gpu.params[0]), &kimi_values(&cpu.params[0]));
    let state = gpu.forward_training_device(&input).unwrap();
    gpu.backward_accumulate_seeded_gpu(&state, &seeds).unwrap();
    gpu.zero_train_buffers_gpu().unwrap();
    gpu.synchronize_resident_parameter_gradients().unwrap();
    assert!(kimi_updates(&gpu.host_fallback_train_buffers()[0]).iter().all(|&x| x == 0.0));
    assert!(!gpu.gradients_non_finite().unwrap());
}

#[test]
fn half_storage_mixer_inference_keeps_training_cache_and_resident_updates() {
    for kimi in [false, true] {
        let (mut cpu, mut gpu, input) = if kimi { kimi_fixture("adam=1\n") }
            else { fixture("adam=1\n", "") };
        let forward = gpu.forward_training_device(&input).unwrap();
        // Compare with f32 inference fed the same half-rounded input. This
        // isolates the input/output storage boundary from the mixer arithmetic.
        use lightnet_cuda::F16TensorOpsBackend;
        let half = gpu.backend.upload_f16(&[input.len()], &input.data).unwrap();
        let rounded = Tensor::from_data(input.shape, gpu.backend.download_f16(&half).unwrap()).unwrap();
        let expected = cpu.forward(&rounded).unwrap().final_output().unwrap().clone();
        let got = gpu.forward_compact_f16(&input).unwrap().final_output.unwrap();
        close("half-storage f32 island", &got.data, &expected.data);
        let shape = gpu.network.layers[0].output_shape;
        let seeds = [Some(Tensor::from_data(shape, vec![0.1; shape.batch * shape.element_count()]).unwrap())];
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds).unwrap();
        gpu.update_weights_gpu_deferred(0.002, 0.7, 0.003, 2).unwrap();
        let current = gpu.forward_compact_f16(&input).unwrap().final_output.unwrap();
        gpu.sync_host_weights_from_device().unwrap();
        cpu.params = gpu.params.clone();
        let expected = cpu.forward(&rounded).unwrap().final_output().unwrap().clone();
        close("half-storage reads updated resident master", &current.data, &expected.data);
    }
}
