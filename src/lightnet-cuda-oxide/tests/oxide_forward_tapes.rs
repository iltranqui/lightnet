//! Tape-lifetime regression: compare interleaved execution to isolated GPU
//! execution. Numeric CPU parity remains the responsibility of layer suites.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams,
    Network, Tensor, initialize_training_params};
use lightnet_cuda_oxide::OxideBackend;

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let scale = expected.iter().fold(1.0_f32, |a, x| a.max(x.abs()));
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a - b).abs() < 1.0e-5 * scale,
            "{label}[{index}]: {a} versus {b}");
    }
}

fn rolling_stats(params: &LayerParams) -> Vec<f32> {
    let convs = match params {
        LayerParams::Wmhf(p) => vec![&p.pre, &p.local_mix, &p.fuse],
        LayerParams::Clifford(p) => p.depthwise.iter().collect(),
        _ => Vec::new(),
    };
    convs.into_iter().filter_map(|p| p.batch_norm.as_ref())
        .flat_map(|bn| bn.rolling_mean.iter().chain(&bn.rolling_variance).copied()).collect()
}

#[test]
fn native_layer_tapes_survive_later_training_and_full_inference() {
    for (name, width, height, channels, body) in [
        ("deform", 3, 2, 2, "[deform_conv]\nfilters=2\nsize=1\ngroups=1\nmask=1\nactivation=leaky\n"),
        ("vit", 4, 4, 2, "[vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=8\nflash_attention=1\n"),
        ("swin", 3, 2, 4, "[swin]\nfilters=4\nsize=2\nheads=2\nshift=1\nffn_ratio=2\nactivation=gelu\n"),
        ("wmhf", 3, 2, 4, "[wmhf]\nfilters=4\nbatch_normalize=1\nshortcut=1\nactivation=leaky\n"),
        ("clifford", 3, 2, 3, "[clifford]\nfilters=3\ngffn_mode=local\ncli_mode=full\nctx_mode=abs\ncli_higher_mode=off\ndwconv_size=3\n"),
        ("dpt", 2, 2, 4, "[dpt_head]\nembed_dim=1\nfusion=2\nhead_hidden=2\npatch_size=1\ntaps=4\nneck=1,1,1,1\nreassemble=4,2,1,0\n"),
        ("depth", 4, 2, 3, "[depth_anything]\npatch_size=2\ndim=4\ndepth=2\nheads=2\nmlp_dim=6\npos_grid=2\ntaps=1,0,1\napply_imagenet_norm=1\nclamp=0\nfreeze=1\n"),
        ("recursive", 3, 2, 2, "[recursive_block]\nbody=2\nloops=3\ninjection=1\nbody_scale=0.4\nresidual_scale=0.3\ninjection_scale=0.2\n[convolutional]\nfilters=2\nsize=3\npad=1\nactivation=leaky\n[convolutional]\nfilters=2\nsize=1\nactivation=tanh\n"),
    ] {
        let text = format!("[net]\nbatch=2\nwidth={width}\nheight={height}\nchannels={channels}\n{body}");
        let net = Network::from_config(ConfigFile::parse_str(name, &text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(net.clone());
        initialize_training_params(&net, &mut cpu.params, 137);
        match &mut cpu.params[0] {
            LayerParams::DptHead(p) => p.values.fill(0.02),
            LayerParams::DepthAnything(p) => for (i, x) in p.as_mut_slice().iter_mut().enumerate() {
                *x = (i as f32 * 0.29 + 0.2).sin() * 0.15;
            },
            LayerParams::Clifford(p) => p.layer_scale.fill(0.6),
            _ => {}
        }
        let mut isolated = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().unwrap());
        isolated.params = cpu.params.clone();
        isolated.init_train_buffers_gpu().unwrap();
        let mut interleaved = GpuExecutableNetwork::new(net, OxideBackend::new().unwrap());
        interleaved.params = cpu.params;
        interleaved.init_train_buffers_gpu().unwrap();
        let shape = isolated.network.net.input_shape();
        let input_a = Tensor::from_data(shape, (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.17).cos() * 0.3).collect()).unwrap();
        let input_b = Tensor::from_data(shape, input_a.data.iter().enumerate()
            .map(|(i, x)| -2.7 * x + (i as f32 * 0.37).sin() * 0.4).collect()).unwrap();
        let reference = isolated.forward_training_device(&input_a).unwrap();
        let earlier = interleaved.forward_training_device(&input_a).unwrap();
        let _later = interleaved.forward_training_device(&input_b).unwrap();
        interleaved.forward_with_state(&input_b).unwrap();
        assert!(earlier.cpu_forward.layer_outputs[0].data.is_empty(), "{name}: requires native backward");
        let out_shape = isolated.network.layers.last().unwrap().output_shape;
        let seed = Tensor::from_data(out_shape, (0..out_shape.batch * out_shape.element_count())
            .map(|i| (i as f32 * 0.19).sin() * 0.2).collect()).unwrap();
        for _ in 0..2 {
            let expected = isolated.backward_accumulate_seeded_gpu_with_input_gradient(&reference,
                &[Some(seed.clone())]).unwrap().expect("isolated input adjoint");
            let actual = interleaved.backward_accumulate_seeded_gpu_with_input_gradient(&earlier,
                &[Some(seed.clone())]).unwrap().expect("interleaved input adjoint");
            assert!(expected.data.iter().any(|x| *x != 0.0), "{name}: nontrivial derivative");
            close(name, &actual.data, &expected.data);
            close(&format!("{name} rolling statistics"), &rolling_stats(&interleaved.params[0]),
                &rolling_stats(&isolated.params[0]));
        }
    }
}
