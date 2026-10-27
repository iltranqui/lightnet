//! DCNv4 channel normalization after optional rolling-stat BN, real GPU runtime.
//! CPU agreement is not compiled-C++ parity. Only supplied-seed unit step tests.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor};
use lightnet_cuda_oxide::{GemmMode, OxideBackend};

const NORMS: [&str;3]=["normalize_channels","normalize_channels_softmax","normalize_channels_softmax_maxval"];
fn backend()->OxideBackend { OxideBackend::new().unwrap().with_gemm_mode(GemmMode::Fp32) }
fn net(norm:&str,bn:bool,adam:bool,fd:bool,conv:bool)->Network {
    let (dims,body)=if fd {
        ("width=3\nheight=2\nchannels=4\nbatch=2", "size=1\npadding=0\ngroups=2\nd_stride=2\nsoftmax=0\noffset_scale=1.0")
    } else {
        ("width=5\nheight=3\nchannels=2\nbatch=2", "size=3\npadding=2\ndilation=2\ngroups=2\nd_stride=2\nsoftmax=1\nremove_center=1\noffset_scale=0.75")
    };
    Network::from_config(ConfigFile::parse_str("norm.cfg",&format!(
        "[net]\n{dims}\ntry_fix_nan=0\nadam={}\ndecay=0.01\n{}\n[dcnv4]\n{body}\nactivation={norm}\nbatch_normalize={}\n",
        u8::from(adam),if conv {"[convolutional]\nfilters=4\nsize=1\nactivation=linear"} else {""},u8::from(bn)
    )).unwrap()).unwrap()
}
fn fill(params:&mut [LayerParams],fd:bool) {
    for p in params {
        match p {
            LayerParams::Convolutional(c)=>{
                if fd { c.weights.fill(0.0); for i in 0..4 { c.weights[i*4+i]=1.0; } }
                else { c.weights.copy_from_slice(&[0.7,0.1,-0.2,0.8,0.6,-0.3,0.1,0.5]); }
                c.biases.fill(0.0);
            }
            LayerParams::Dcnv4(p)=>{
                for (i,w) in p.offset_weights.iter_mut().enumerate() { *w=((i%5) as f32-2.0)*0.0005; }
                p.offset_biases.fill(0.47);
                if fd { p.offset_biases[..6].copy_from_slice(&[0.47,0.53,0.8,0.51,0.49,0.9]); }
                p.biases.copy_from_slice(&[0.6,-1.4,2.0,-1.2]);
                if let Some(b)=&mut p.batch_norm {
                    p.biases.fill(7.0); b.biases.fill(9.0);
                    b.scales.copy_from_slice(&[0.7,1.1,0.9,1.3]);
                    b.rolling_mean.copy_from_slice(&[-0.6,1.4,-2.0,1.2]);
                    b.rolling_variance.copy_from_slice(&[0.9,1.3,0.7,1.1]);
                }
            }
            _=>{}
        }
    }
}
fn tensor(shape:lightnet_core::TensorShape,seed:bool)->Tensor {
    Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i| if seed {
        ((i*7%11) as f32-5.0)*0.13
    } else { (i%7) as f32*0.05+0.1 }).collect()).unwrap()
}
fn near(label:&str,a:&[f32],b:&[f32],tol:f32) {
    assert_eq!(a.len(),b.len());
    for (i,(x,y)) in a.iter().zip(b).enumerate() { assert!((x-y).abs()<tol*(1.0+x.abs()),"{label}[{i}]: {x} vs {y}"); }
}

#[test]
fn normalized_dcnv4_cpu_gpu_two_step_sgd_adam_bn_matrix() {
    let mut be=backend();
    for norm in NORMS { for bn in [false,true] { for adam in [false,true] {
        let network=net(norm,bn,adam,false,true);
        let mut cpu=ExecutableNetwork::new(network.clone()); fill(&mut cpu.params,false);
        let mut gpu=GpuExecutableNetwork::new(network,be); gpu.params=cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        for step in 0..2 {
            let mut input=tensor(cpu.network.net.input_shape(),false);
            for x in &mut input.data { *x*=1.0+step as f32*0.1; }
            let cf=cpu.forward_training(&input).unwrap(); let gf=gpu.forward_training_with_state(&input).unwrap();
            for (c,g) in cf.layer_outputs.iter().zip(&gf.cpu_forward.layer_outputs) { near(norm,&c.data,&g.data,3e-4); }
            let seed=tensor(cpu.network.layers[1].output_shape,true);
            assert!(cpu.backward(&input,&gf.cpu_forward,&seed).unwrap_err().to_string().contains("missing [dcnv4] forward cache"));
            let seeds=[None,Some(seed)];
            cpu.backward_accumulate_seeded(&input,&cf,&seeds).unwrap(); gpu.backward_accumulate_seeded_gpu(&gf,&seeds).unwrap();
            let bc=cpu.params.clone(); let bg=gpu.params.clone();
            if adam { cpu.update_weights_adam(0.005,2).unwrap(); } else { cpu.update_weights(0.005,0.3,0.01,2); }
            gpu.update_weights_gpu(0.005,0.3,0.01,2).unwrap();
            let compare=|name:&str,c:&[f32],g:&[f32],oldc:&[f32],oldg:&[f32]| {
                near(name,c,g,2e-5);
                assert_ne!(g,oldg,"{norm} bn={bn} adam={adam} {name} did not move");
                for i in 0..c.len() { let dc=c[i]-oldc[i];let dg=g[i]-oldg[i];
                    assert!((dc-dg).abs()<1e-6+2e-3*dc.abs(),"{norm} {name} step={step} delta[{i}]: {dc} vs {dg}"); }
            };
            let (LayerParams::Convolutional(c),LayerParams::Convolutional(g),LayerParams::Convolutional(oc),LayerParams::Convolutional(og))=
                (&cpu.params[0],&gpu.params[0],&bc[0],&bg[0]) else {unreachable!()};
            compare("upstream weights",&c.weights,&g.weights,&oc.weights,&og.weights);
            compare("upstream biases",&c.biases,&g.biases,&oc.biases,&og.biases);
            let (LayerParams::Dcnv4(c),LayerParams::Dcnv4(g),LayerParams::Dcnv4(oc),LayerParams::Dcnv4(og))=
                (&cpu.params[1],&gpu.params[1],&bc[1],&bg[1]) else {unreachable!()};
            compare("offset weights",&c.offset_weights,&g.offset_weights,&oc.offset_weights,&og.offset_weights);
            compare("offset biases",&c.offset_biases,&g.offset_biases,&oc.offset_biases,&og.offset_biases);
            assert_eq!(g.dummy_weights,og.dummy_weights); assert_eq!(c.dummy_weights,g.dummy_weights);
            if let (Some(c),Some(g),Some(oc),Some(og))=(&c.batch_norm,&g.batch_norm,&oc.batch_norm,&og.batch_norm) {
                compare("gamma",&c.scales,&g.scales,&oc.scales,&og.scales);
                near("rolling mean",&c.rolling_mean,&g.rolling_mean,3e-5); near("rolling variance",&c.rolling_variance,&g.rolling_variance,3e-5);
                assert_eq!(c.biases,oc.biases); assert_eq!(g.biases,og.biases);
            } else { compare("output bias",&c.biases,&g.biases,&oc.biases,&og.biases); }
        }
        be=gpu.backend;
    }}}
}

#[test]
fn normalized_dcnv4_gpu_forward_finite_differences_with_fixed_stats() {
    let mut be=backend();
    for norm in NORMS { for bn in [false,true] {
        let network=net(norm,bn,false,true,true);
        let mut gpu=GpuExecutableNetwork::new(network,be);fill(&mut gpu.params,true);gpu.init_train_buffers_gpu().unwrap();
        let old=gpu.params.clone(); let input=tensor(gpu.network.net.input_shape(),false);
        let f=gpu.forward_training_with_state(&input).unwrap(); let middle=f.cpu_forward.layer_outputs[0].clone();
        let seed=tensor(gpu.network.layers[1].output_shape,true);
        gpu.backward_accumulate_seeded_gpu(&f,&[None,Some(seed.clone())]).unwrap(); gpu.update_weights_gpu(1.0,0.0,0.0,2).unwrap();
        let LayerParams::Convolutional(c)=&gpu.params[0] else {unreachable!()}; let input_grad=(c.weights[0]-1.0)*2.0;
        let (LayerParams::Dcnv4(p),LayerParams::Dcnv4(o))=(&gpu.params[1],&old[1]) else {unreachable!()};
        let dw:Vec<_>=p.offset_weights.iter().zip(&o.offset_weights).map(|(x,y)| (x-y)*200.0).collect();
        let db:Vec<_>=p.offset_biases.iter().zip(&o.offset_biases).map(|(x,y)| (x-y)*200.0).collect();
        let ds:Vec<_>=if let (Some(x),Some(y))=(&p.batch_norm,&o.batch_norm) { x.scales.iter().zip(&y.scales).map(|(x,y)| (x-y)*2.0).collect() }
            else { p.biases.iter().zip(&o.biases).map(|(x,y)| (x-y)*2.0).collect() };
        let network=net(norm,bn,false,true,false);
        let mut fd=GpuExecutableNetwork::new(network,gpu.backend);
        let objective=|fd:&mut GpuExecutableNetwork<OxideBackend>,input:&Tensor,params:&LayerParams|->f32 {
            fd.params[0]=params.clone();let before=fd.params.clone();
            let f=fd.forward_training_with_state(input).unwrap();assert_eq!(fd.params,before);
            f.cpu_forward.final_output().unwrap().data.iter().zip(&seed.data).map(|(v,s)| v*s).sum()
        };
        let eps=0.002;
        // Half-cell offsets, small weights; fixed positive/negative ReLU signs
        // and a separated channel-2 maximum exclude all derivative seams.
        let mut hi=middle.clone();let mut lo=middle.clone();
        for b in 0..2 {for q in 0..6 {let i=b*24+q;hi.data[i]+=eps*input.data[i];lo.data[i]-=eps*input.data[i];}}
        let fdv=(objective(&mut fd,&hi,&old[1])-objective(&mut fd,&lo,&old[1]))/(2.0*eps);
        assert!((input_grad-fdv).abs()<3e-3,"{norm} bn={bn} input {input_grad} vs {fdv}");
        for (slot,indices) in [(0,vec![0,6,18,24]),(1,vec![0,2,3,5]),(2,vec![0,1,2,3])] {
            for i in indices {
                let mut hi=old[1].clone();let mut lo=old[1].clone();
                let (LayerParams::Dcnv4(h),LayerParams::Dcnv4(l))=(&mut hi,&mut lo) else {unreachable!()};
                let actual=match slot {
                    0=>{h.offset_weights[i]+=eps;l.offset_weights[i]-=eps;dw[i]},
                    1=>{h.offset_biases[i]+=eps;l.offset_biases[i]-=eps;db[i]},
                    _=>{if let (Some(h),Some(l))=(&mut h.batch_norm,&mut l.batch_norm) {h.scales[i]+=eps;l.scales[i]-=eps;}
                        else {h.biases[i]+=eps;l.biases[i]-=eps;}ds[i]},
                };
                let fdv=(objective(&mut fd,&middle,&hi)-objective(&mut fd,&middle,&lo))/(2.0*eps);
                assert!((actual-fdv).abs()<3e-3,"{norm} bn={bn} slot={slot} i={i}: {actual} vs {fdv}");
            }
        }
        be=fd.backend;
    }}
}

#[test]
fn normalized_dcnv4_mixed_network_keeps_cpu_cache() {
    let mut be=backend();
    for norm in NORMS {
        // Parse a genuine mixed graph; no activation/shape metadata patching.
        let cfg=format!("[net]\nwidth=3\nheight=2\nchannels=4\nbatch=2\ntry_fix_nan=0\n[dcnv4]\nsize=1\npadding=0\ngroups=2\nd_stride=2\nsoftmax=0\nbatch_normalize=1\nactivation={norm}\n[deform_conv]\nfilters=4\nsize=1\nbatch_normalize=1\nactivation=linear\n");
        let network=Network::from_config(ConfigFile::parse_str("mixed.cfg",&cfg).unwrap()).unwrap();
        let mut cpu=ExecutableNetwork::new(network.clone());fill(&mut cpu.params,true);
        let mut gpu=GpuExecutableNetwork::new(network,be);gpu.params=cpu.params.clone();
        let input=tensor(cpu.network.net.input_shape(),false);let cf=cpu.forward_training(&input).unwrap();let gf=gpu.forward_training_with_state(&input).unwrap();
        let seed=tensor(cpu.network.layers[0].output_shape,true);let seeds=[Some(seed),None];
        cpu.backward_seeded(&input,&gf.cpu_forward,&seeds).unwrap();
        cpu.backward_accumulate_seeded(&input,&cf,&seeds).unwrap();gpu.backward_accumulate_seeded_gpu(&gf,&seeds).unwrap();
        be=gpu.backend;
    }
}
