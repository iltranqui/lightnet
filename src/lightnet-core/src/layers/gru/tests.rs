    use lightnet_runtime::layers::gru::cpu::*;
    use crate::model::Tensor;
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

