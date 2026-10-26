//! Unrun dense AMP source fixtures. Half operands in forward/dgrad/wgrad,
//! f32 master parameters, moments and parameter gradients. No C++ golden.
use lightnet_cuda::{AdamParams, ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::{GemmMode, HalfFormat, OxideBackend};
use lightnet_cuda_oxide::half::{f32_to_f16_bits,f16_bits_to_f32,f32_to_bf16_bits,bf16_bits_to_f32};

fn round(x:f32,fmt:HalfFormat)->f32 {
    match fmt { HalfFormat::F16=>f16_bits_to_f32(f32_to_f16_bits(x)),
        HalfFormat::Bf16=>bf16_bits_to_f32(f32_to_bf16_bits(x)) }
}
fn close(label:&str,got:&[f32],want:&[f32],tol:f32) {
    assert_eq!(got.len(),want.len(),"{label}");
    let scale=want.iter().fold(1.0_f32,|m,x|m.max(x.abs()));
    for(i,(&a,&b))in got.iter().zip(want).enumerate(){
        assert!(a.is_finite()&&b.is_finite()&&(a-b).abs()<=tol*scale,"{label}[{i}]: {a} vs {b}");
    }
}

#[test]
fn dense_amp_forward_dgrad_wgrad_and_bias_match_rounded_operands() {
    for fmt in [HalfFormat::F16,HalfFormat::Bf16] {
        let be=OxideBackend::new().unwrap().with_amp_format(Some(fmt));
        // Ragged M/N/K and a reduction spanning more than one tensor-core tile.
        for (batch,ins,outs) in [(3,5,7),(17,19,13),(1,3,2)] {
            let values=|n:usize,phase:f32|(0..n).map(|i|(i as f32*0.17+phase).sin()*0.2).collect::<Vec<_>>();
            let (x,w,b,dy)=(values(batch*ins,0.1),values(outs*ins,0.3),values(outs,0.5),values(batch*outs,0.7));
            let up=|shape:&[usize],v:&[f32]|be.upload_f32(shape,v).unwrap();
            let (xt,wt,bt,dyt)=(up(&[batch,ins],&x),up(&[outs,ins],&w),up(&[outs],&b),up(&[batch,outs],&dy));
            let output=be.connected_f32(&xt,&wt,&bt,batch,ins,outs,ElementwiseLaunchConfig::default()).unwrap();
            assert!(output.is_half());
            let (dx,dw,db)=be.connected_backward_f32(&xt,&wt,&dyt,batch,ins,outs,ElementwiseLaunchConfig::default()).unwrap();
            assert!(dx.is_half()); assert!(!dw.is_half()); assert!(!db.is_half());
            let (xr,wr,dyr)=(x.iter().map(|&v|round(v,fmt)).collect::<Vec<_>>(),
                w.iter().map(|&v|round(v,fmt)).collect::<Vec<_>>(),dy.iter().map(|&v|round(v,fmt)).collect::<Vec<_>>());
            let mut want=lightnet_cuda::host_ops::connected(&xr,&wr,&b,batch,ins,outs);
            for v in &mut want{*v=round(*v,fmt);}
            let(mut want_dx,want_dw,want_db)=lightnet_cuda::host_ops::connected_backward(&xr,&wr,&dyr,batch,ins,outs);
            for v in &mut want_dx{*v=round(*v,fmt);}
            let tol=if fmt==HalfFormat::F16{0.001}else{0.008};
            close("dense AMP forward",&be.download_f32(&output).unwrap(),&want,tol);
            close("dense AMP dgrad",&be.download_f32(&dx).unwrap(),&want_dx,tol);
            close("dense AMP f32 wgrad",&be.download_f32(&dw).unwrap(),&want_dw,2e-5);
            close("dense AMP f32 bias",&be.download_f32(&db).unwrap(),&want_db,2e-5);
            assert!(!wt.is_half(),"master weights stay f32");
        }
    }
}

#[test]
fn dense_amp_updated_master_is_used_and_adam_state_stays_f32() {
    let be=OxideBackend::new().unwrap().with_amp(true);
    let cfg=ElementwiseLaunchConfig::default();
    let x=be.upload_f32(&[2,3],&[0.25,-0.5,0.125,0.75,0.25,-0.25]).unwrap();
    let mut w=be.upload_f32(&[2,3],&[0.1,0.2,-0.3,0.4,0.2,-0.1]).unwrap();
    let b=be.zeros_f32(&[2]).unwrap();
    let dy=be.upload_f32(&[2,2],&[0.5,-0.25,0.125,0.25]).unwrap();
    let mut m=be.zeros_f32(&[2,3]).unwrap(); let mut v=m.clone();
    let mut previous=be.download_f32(&be.connected_f32(&x,&w,&b,2,3,2,cfg).unwrap()).unwrap();
    for step in 1..=3 {
        let (_,dw,_)=be.connected_backward_f32(&x,&w,&dy,2,3,2,cfg).unwrap();
        let zero=be.zeros_f32(&[2,3]).unwrap();
        let descent=be.saxpy_f32(-1.0,&dw,&zero,cfg).unwrap();
        let adam=AdamParams{lr:0.01,beta1:0.9,beta2:0.99,eps:1e-5,decay:0.01,batch:2,step,decoupled_decay:true};
        let pair=be.adam_update_f32(&w,&descent,&m,&v,&adam,cfg).unwrap();
        (w,m,v)=pair;
        assert!(!w.is_half()&&!m.is_half()&&!v.is_half());
        let got=be.download_f32(&be.connected_f32(&x,&w,&b,2,3,2,cfg).unwrap()).unwrap();
        assert_ne!(got,previous,"new master weight must refresh half operands"); previous=got;
    }
    let huge=be.upload_f32(&[2,2],&[1e10;4]).unwrap();
    let (_,dw,_)=be.connected_backward_f32(&x,&w,&huge,2,3,2,cfg).unwrap();
    assert!(be.any_non_finite_f32(&dw).unwrap(),"overflow remains visible to loss scaling");
}

#[test]
fn dense_amp_off_retains_f32_outputs_and_gradients() {
    let be=OxideBackend::new().unwrap().with_amp(false).with_gemm_mode(GemmMode::Fp32);
    let x=be.upload_f32(&[1,2],&[0.1,0.2]).unwrap();
    let w=be.upload_f32(&[1,2],&[0.3,0.4]).unwrap(); let b=be.zeros_f32(&[1]).unwrap();
    let cfg=ElementwiseLaunchConfig::default();
    let y=be.connected_f32(&x,&w,&b,1,2,1,cfg).unwrap();
    let(dx,dw,db)=be.connected_backward_f32(&x,&w,&y,1,2,1,cfg).unwrap();
    assert!(!y.is_half()&&!dx.is_half()&&!dw.is_half()&&!db.is_half());
}
