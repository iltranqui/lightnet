//! Batched forward launches must equal the single-image launches run image by image
//! (bitwise: same kernels, same arithmetic, only the base pointers differ).

use hip_rt::DeviceBuffer;
use hip_tile::{Tensor, TensorMut};
use kernels::yolo::batched::*;
use kernels::yolo::concat::launch_concat;
use kernels::yolo::conv::{ConvParams, launch_conv};
use kernels::yolo::maxpool::{PoolParams, launch_maxpool};
use kernels::yolo::upsample::launch_upsample;
use runner::random_vec;

const BATCH: usize = 3;

fn signed(len: usize, seed: u64) -> Vec<f32> {
    random_vec(len, seed).into_iter().map(|v| v * 2.0 - 1.0).collect()
}

fn module() -> hip_rt::Module {
    runner::load().unwrap().1
}

#[test]
fn conv_batched_matches_per_image() {
    let md = module();
    // Linear with a zero bias is the training forward's "pre-BN sum"; leaky covers the fused path.
    for act in [ConvParams::LINEAR, ConvParams::LEAKY] {
        let p = ConvParams::with_cout(5, 9, 11, 7, 3, 1, 1, act);
        let (hw, ohw) = (11 * 7, (p.oh * p.ow) as usize);
        let x = signed(BATCH * 5 * hw, 1);
        let w = signed(9 * 5 * 9, 2);
        let bias = if act == ConvParams::LINEAR { vec![0.0; 9] } else { signed(9, 3) };
        let (dx, dw, db) = (
            DeviceBuffer::from_slice(&x).unwrap(),
            DeviceBuffer::from_slice(&w).unwrap(),
            DeviceBuffer::from_slice(&bias).unwrap(),
        );
        let mut got = DeviceBuffer::from_slice(&vec![-7.0f32; BATCH * 9 * ohw]).unwrap();
        let (xv, wv) = (dx.tensor(5, hw as u32), dw.tensor(9, 45));
        let bv = Tensor::vector(db.as_ptr(), 9);
        launch_conv_batched(&md, None, BATCH as u32, xv, wv, bv, got.tensor_mut(9, ohw as u32), &p).unwrap();
        let mut want = Vec::new();
        for b in 0..BATCH {
            let xi = DeviceBuffer::from_slice(&x[b * 5 * hw..(b + 1) * 5 * hw]).unwrap();
            let mut yi = DeviceBuffer::zeroed(9 * ohw).unwrap();
            launch_conv(&md, None, xi.tensor(5, hw as u32), wv, bv, yi.tensor_mut(9, ohw as u32), &p).unwrap();
            want.extend(yi.copy_to_vec().unwrap());
        }
        assert_eq!(got.copy_to_vec().unwrap(), want, "act {act}");
    }
}

#[test]
fn maxpool_batched_matches_per_image() {
    let md = module();
    let p = PoolParams::new(13, 9, 2, 2, None);
    let (hw, ohw) = (13 * 9, (p.oh * p.ow) as usize);
    let x = signed(BATCH * 6 * hw, 4);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut got = DeviceBuffer::from_slice(&vec![-7.0f32; BATCH * 6 * ohw]).unwrap();
    launch_maxpool_batched(&md, None, BATCH as u32, dx.tensor(6, hw as u32), got.tensor_mut(6, ohw as u32), &p).unwrap();
    let mut want = Vec::new();
    for b in 0..BATCH {
        let xi = DeviceBuffer::from_slice(&x[b * 6 * hw..(b + 1) * 6 * hw]).unwrap();
        let mut yi = DeviceBuffer::zeroed(6 * ohw).unwrap();
        launch_maxpool(&md, None, xi.tensor(6, hw as u32), yi.tensor_mut(6, ohw as u32), &p).unwrap();
        want.extend(yi.copy_to_vec().unwrap());
    }
    assert_eq!(got.copy_to_vec().unwrap(), want);
}

#[test]
fn upsample_batched_matches_per_image() {
    let md = module();
    let (c, h, w, s) = (7usize, 5usize, 4usize, 2usize);
    let (hw, ohw) = (h * w, h * s * w * s);
    let x = signed(BATCH * c * hw, 5);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut got = DeviceBuffer::from_slice(&vec![-7.0f32; BATCH * c * ohw]).unwrap();
    launch_upsample_batched(&md, None, BATCH as u32, dx.tensor(c as u32, hw as u32), got.tensor_mut(c as u32, ohw as u32), w as u32, s as u32).unwrap();
    let mut want = Vec::new();
    for b in 0..BATCH {
        let xi = DeviceBuffer::from_slice(&x[b * c * hw..(b + 1) * c * hw]).unwrap();
        let mut yi = DeviceBuffer::zeroed(c * ohw).unwrap();
        launch_upsample(&md, None, xi.tensor(c as u32, hw as u32), yi.tensor_mut(c as u32, ohw as u32), w as u32, s as u32).unwrap();
        want.extend(yi.copy_to_vec().unwrap());
    }
    assert_eq!(got.copy_to_vec().unwrap(), want);
}

#[test]
fn concat_batched_matches_per_image() {
    let md = module();
    let hw = 35 * 3 + 1;
    let (ca, cb) = (13usize, 21usize);
    let total = ca + cb;
    let (a, b) = (signed(BATCH * ca * hw, 6), signed(BATCH * cb * hw, 7));
    let (da, db) = (DeviceBuffer::from_slice(&a).unwrap(), DeviceBuffer::from_slice(&b).unwrap());
    let mut got = DeviceBuffer::from_slice(&vec![-7.0f32; BATCH * total * hw]).unwrap();
    let base = got.as_mut_ptr();
    let view = |ch0: usize, rows: usize| {
        TensorMut::new(unsafe { base.add(ch0 * hw) }, rows as u32, hw as u32, hw as u32)
    };
    let stride = total * hw;
    launch_concat_batched(&md, None, BATCH as u32, da.tensor(ca as u32, hw as u32), ca * hw, view(0, ca), stride).unwrap();
    launch_concat_batched(&md, None, BATCH as u32, db.tensor(cb as u32, hw as u32), cb * hw, view(ca, cb), stride).unwrap();
    let mut want = Vec::new();
    for i in 0..BATCH {
        want.extend_from_slice(&a[i * ca * hw..(i + 1) * ca * hw]);
        want.extend_from_slice(&b[i * cb * hw..(i + 1) * cb * hw]);
    }
    assert_eq!(got.copy_to_vec().unwrap(), want);
    let _ = launch_concat; // single-image path is covered in yolo_ops.rs
}

/// `dst += src` per image for a group slice of a larger gradient (the route-backward shape).
#[test]
fn route_bwd_batched_group_slice() {
    use kernels::backward::batched::launch_route_bwd_batched;
    let md = module();
    let (c_total, c, hw) = (128usize, 64usize, 20 * 28);
    let src = signed(BATCH * c * hw, 8);
    let base = signed(BATCH * c_total * hw, 9);
    let dsrc = DeviceBuffer::from_slice(&src).unwrap();
    let mut ddst = DeviceBuffer::from_slice(&base).unwrap();
    let dptr = ddst.as_mut_ptr();
    launch_route_bwd_batched(
        &md, None, BATCH as u32,
        dsrc.tensor(c as u32, hw as u32), c * hw,
        TensorMut::new(unsafe { dptr.add(c * hw) }, c as u32, hw as u32, hw as u32), c_total * hw,
    )
    .unwrap();
    let mut want = base.clone();
    for b in 0..BATCH {
        for ch in 0..c {
            for j in 0..hw {
                want[(b * c_total + c + ch) * hw + j] += src[(b * c + ch) * hw + j];
            }
        }
    }
    assert_eq!(ddst.copy_to_vec().unwrap(), want);
}
