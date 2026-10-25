//! Unrun compact/full native Tucker source fixture against the Rust CPU.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams,
    LayerTrainBuffers, Network, Tensor};
use lightnet_cuda::{F16TensorOpsBackend, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

#[test]
fn compact_tucker_padded_windows_dense_and_sparse_match_cpu() {
    for msa in [0,1] {
        let text = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=4\n[tucker_attention]\nfilters=4\nsize=2\nheads=2\nrank_q=2\nrank_k=3\nrank_v=2\nrank_o=3\nactivation=gelu\nmsa={msa}\nmsa_block_size=2\nmsa_topk=1\n");
        let network = Network::from_config(ConfigFile::parse_str("compact-tucker.cfg",&text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::TuckerAttention(p) = &mut cpu.params[0] else { panic!("Tucker params") };
        for (i,w) in p.weights.iter_mut().enumerate() { *w=(i as f32*0.19).sin()*0.15; }
        for (i,b) in p.biases.iter_mut().enumerate() { *b=(i as f32*0.23).cos()*0.1; }
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).sin()).collect()).unwrap();
        let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
        let mut gpu = GpuExecutableNetwork::new(network,OxideBackend::new().unwrap());
        gpu.params=cpu.params.clone();
        let full=gpu.forward(&input).unwrap().final_output().unwrap().clone();
        let compact=gpu.forward_compact(&input).unwrap().final_output.unwrap();
        let scale=expected.data.iter().fold(1.0_f32,|a,x|a.max(x.abs()));
        for result in [full,compact] {
            assert_eq!(result.shape,expected.shape);
            for (i,(&a,&b)) in result.data.iter().zip(&expected.data).enumerate() {
                assert!((a-b).abs()<=3e-4*scale,"msa={msa} [{i}]: {a} vs {b}");
            }
        }
    }
}

fn close(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label}");
    let scale = expected.iter().fold(1.0_f32, |m, x| m.max(x.abs()));
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= 1e-3 * scale,
            "{label}[{i}]: {a} vs {b}");
    }
}

#[test]
fn tucker_resident_factor_permutations_backward_and_deferred_optimizers() {
    for optimizer in ["adam=0\n", "adam=1\ndecoupled_decay=0\n", "adam=1\ndecoupled_decay=1\n"] {
        for msa in [0, 1] {
            let text = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=4\nloss_scale=8\ndecay=0.03\neps=0.001\n{optimizer}\n[tucker_attention]\nfilters=4\nsize=2\nheads=2\nrank_q=2\nrank_k=3\nrank_v=2\nrank_o=3\nactivation=gelu\nmsa={msa}\nmsa_block_size=2\nmsa_topk=1\n");
            let net = Network::from_config(ConfigFile::parse_str("resident-tucker.cfg", &text).unwrap()).unwrap();
            let mut cpu = ExecutableNetwork::new(net.clone());
            let LayerParams::TuckerAttention(p) = &mut cpu.params[0] else { panic!("Tucker params") };
            for (i, x) in p.biases.iter_mut().chain(&mut p.weights).enumerate() {
                *x = (i as f32 * 0.19).sin() * 0.15;
            }
            let shape = cpu.network.net.input_shape();
            let input = Tensor::from_data(shape, (0..shape.batch * shape.element_count())
                .map(|i| (i as f32 * 0.17).sin() * 0.2).collect()).unwrap();
            let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().unwrap());
            gpu.params = cpu.params.clone();
            gpu.init_train_buffers_gpu().unwrap();
            let original = gpu.params.clone();
            for iteration in 0..3 {
                let host = cpu.forward_training(&input).unwrap();
                let device = gpu.forward_training_device(&input).unwrap();
                let out = host.final_output().unwrap();
                close("Tucker resident forward", &gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(), &out.data);
                let half = gpu.forward_compact_f16(&input).unwrap().final_output.unwrap();
                close("Tucker half-storage inference", &half.data, &out.data);
                let seed = Tensor::from_data(out.shape, vec![0.1; out.len()]).unwrap();
                let scaled = Tensor::from_data(out.shape, vec![0.8; out.len()]).unwrap();
                for _ in 0..2 {
                    let expected = cpu.backward_accumulate_seeded(&input, &host, &[Some(scaled.clone())]).unwrap();
                    let got = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device, &[Some(seed.clone())]).unwrap().unwrap();
                    close("Tucker input adjoint", &got.data, &expected.input_gradient.data);
                }
                if iteration == 0 {
                    let buffers = gpu.host_fallback_train_buffers();
                    let LayerTrainBuffers::TuckerAttention(t) = &buffers[0] else { panic!("Tucker train") };
                    assert!(t.bias_updates.iter().chain(&t.weight_updates).all(|&x| x == 0.0));
                }
                gpu.synchronize_resident_parameter_gradients().unwrap();
                {
                    let buffers = gpu.host_fallback_train_buffers();
                    let LayerTrainBuffers::TuckerAttention(t) = &buffers[0] else { panic!("Tucker train") };
                    let LayerTrainBuffers::TuckerAttention(want) = &cpu.train_buffers[0] else { panic!("Tucker train") };
                    close("Tucker weight gradients", &t.weight_updates, &want.weight_updates);
                    close("Tucker bias gradients", &t.bias_updates, &want.bias_updates);
                }
                if cpu.network.net.adam { cpu.update_weights_adam(0.002, 2).unwrap(); }
                else { cpu.update_weights(0.002 / 8.0, 0.7, 0.03 * 8.0, 2); }
                gpu.update_weights_gpu_deferred(0.002, 0.7, 0.03, 2).unwrap();
                assert_eq!(gpu.params, original);
                let want = cpu.forward(&input).unwrap().final_output().unwrap().clone();
                let got = gpu.forward_compact(&input).unwrap().final_output.unwrap();
                close("Tucker updated resident inference", &got.data, &want.data);
            }
            gpu.sync_host_weights_from_device().unwrap();
            let LayerParams::TuckerAttention(got) = &gpu.params[0] else { panic!("Tucker params") };
            let LayerParams::TuckerAttention(want) = &cpu.params[0] else { panic!("Tucker params") };
            close("Tucker resident weights", &got.weights, &want.weights);
            close("Tucker resident biases", &got.biases, &want.biases);
        }
    }
}

#[test]
fn device_half_cast_trait_preserves_shape_and_nonfinite_values() {
    let be = OxideBackend::new().unwrap();
    let values = [0.125, -0.25, 70000.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let input = be.upload_f32(&[2, 3], &values).unwrap();
    let half = be.narrow_f32(&input).unwrap();
    assert_eq!(half.shape(), &[2, 3]);
    let wide = be.widen_f16(&half).unwrap();
    assert_eq!(wide.shape(), &[2, 3]);
    let got = be.download_f32(&wide).unwrap();
    assert_eq!(&got[..2], &values[..2]);
    assert_eq!(got[2], f32::INFINITY);
    assert!(got[3].is_nan());
    assert_eq!(got[4], f32::INFINITY);
    assert_eq!(got[5], f32::NEG_INFINITY);
}
