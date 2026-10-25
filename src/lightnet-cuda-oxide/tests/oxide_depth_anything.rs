//! Unrun frozen encoder parity fixtures: Rust reference, no C++ golden.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

fn close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name}");
    let scale = expected.iter().fold(1.0_f32, |m,x|m.max(x.abs()));
    for (i,(&a,&b)) in got.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= tol*scale,
            "{name}[{i}]: {a} vs {b}");
    }
}

#[test]
fn native_encoder_repeated_taps_patch_layout_and_input_derivative_match_cpu() {
    for (patch, imagenet, clamp) in [(1,0,0.0),(2,1,0.15)] {
        let width = 3*patch;
        let height = 2*patch;
        let text = format!("[net]\nbatch=2\nwidth={width}\nheight={height}\nchannels=3\n[depth_anything]\npatch_size={patch}\ndim=4\ndepth=2\nheads=2\nmlp_dim=6\npos_grid=2\ntaps=1,0,1\napply_imagenet_norm={imagenet}\nclamp={clamp}\nfreeze=1\n");
        let network = Network::from_config(ConfigFile::parse_str("native-depth.cfg",&text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::DepthAnything(p) = &mut cpu.params[0] else {panic!("depth params")};
        for (i,x) in p.as_mut_slice().iter_mut().enumerate() { *x = (i as f32*0.29+0.2).sin()*0.15; }
        let frozen = cpu.params.clone();
        let mut gpu = GpuExecutableNetwork::new(network,OxideBackend::new().unwrap());
        gpu.params = frozen.clone();
        gpu.init_train_buffers_gpu().unwrap();
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).cos()*0.3).collect()).unwrap();
        let host = cpu.forward_training(&input).unwrap();
        let state = gpu.forward_training_device(&input).unwrap();
        assert!(state.cpu_forward.layer_outputs[0].data.is_empty());
        let expected = host.final_output().unwrap();
        close("output",&gpu.backend.download_f32(&state.gpu_layer_outputs[0]).unwrap(),&expected.data,4e-4);
        let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
        close("compact forward",&compact.data,&expected.data,4e-4);
        let seed = Tensor::from_data(expected.shape,(0..expected.len()).map(|i|(i as f32*0.13).sin()*0.1).collect()).unwrap();
        let seeds = [Some(seed)];
        let expected = cpu.backward_accumulate_seeded(&input,&host,&seeds).unwrap();
        let got = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&state,&seeds).unwrap().unwrap();
        close("input derivative",&got.data,&expected.input_gradient.data,1e-3);
        gpu.update_weights_gpu(0.01,0.9,0.0005,2).unwrap();
        assert_eq!(gpu.params,frozen,"encoder parameters stay frozen");
    }
}

#[test]
fn depth_clamp_preserves_nan_and_cpu_inclusive_boundary_adjoint() {
    let be = OxideBackend::new().unwrap();
    let raw = [f32::NEG_INFINITY,-1.0,-0.5,0.5,1.0,f32::INFINITY,f32::NAN];
    let x = be.upload_f32(&[7],&raw).unwrap();
    let seeds = be.upload_f32(&[7],&[1.0;7]).unwrap();
    let out = be.depth_tap_clamp_f32(&x,None,&[7],0.5,ElementwiseLaunchConfig::default()).unwrap();
    let out = be.download_f32(&out).unwrap();
    assert_eq!(&out[..6],&[-0.5,-0.5,-0.5,0.5,0.5,0.5]);
    assert!(out[6].is_nan());
    let grad = be.depth_tap_clamp_f32(&x,Some(&seeds),&[7],0.5,ElementwiseLaunchConfig::default()).unwrap();
    assert_eq!(be.download_f32(&grad).unwrap(),[0.0,0.0,1.0,1.0,0.0,0.0,1.0]);
}
