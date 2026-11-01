//! GPU yolo ops against straightforward CPU references (f64 accumulation).

use hip_rt::{DeviceBuffer, Module};
use hip_tile::{Tensor, TensorMut};
use kernels::yolo::concat::launch_concat;
use kernels::yolo::conv::{Config, ConvParams, launch_conv, launch_conv_with};
use kernels::yolo::maxpool::{PoolParams, launch_maxpool};
use kernels::yolo::upsample::launch_upsample;
use runner::random_vec;

/// Values in [-1, 1) so leaky and cancellation both matter.
fn signed(len: usize, seed: u64) -> Vec<f32> {
    random_vec(len, seed)
        .into_iter()
        .map(|v| v * 2.0 - 1.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn conv_cpu(x: &[f32], w: &[f32], bias: &[f32], cout: usize, p: &ConvParams) -> Vec<f64> {
    let (cin, h, wd, k) = (p.cin as usize, p.h as usize, p.w as usize, p.k as usize);
    let (s, pad, oh, ow) = (
        p.stride as usize,
        p.pad as isize,
        p.oh as usize,
        p.ow as usize,
    );
    let mut out = vec![0f64; cout * oh * ow];
    for co in 0..cout {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = f64::from(bias[co]);
                for ci in 0..cin {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = (oy * s + ky) as isize - pad;
                            let ix = (ox * s + kx) as isize - pad;
                            if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < wd {
                                acc += f64::from(w[((co * cin + ci) * k + ky) * k + kx])
                                    * f64::from(x[ci * h * wd + iy as usize * wd + ix as usize]);
                            }
                        }
                    }
                }
                if p.act == ConvParams::LEAKY && acc < 0.0 {
                    acc *= 0.1;
                }
                out[co * oh * ow + oy * ow + ox] = acc;
            }
        }
    }
    out
}

/// Max relative error with a floor relative to the tensor scale (outputs cross zero).
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / w.abs().max(scale * 1e-2))
        .fold(0.0, f64::max)
}

struct Case {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    act: u32,
}

/// Run `cfg` (or the auto choice) and compare. `slice` = (total channels, first channel)
/// of a bigger output buffer to write into (the rest must stay at the poison value).
fn check_conv(c: &Case, cfg: Option<Config>, slice: Option<(u32, u32)>) {
    let (_dev, module) = runner::load().unwrap();
    let pad = c.k / 2;
    let p = ConvParams::new(c.cin, c.h, c.w, c.k, c.stride, pad, c.act);
    let (hw, ohw) = ((c.h * c.w) as usize, (p.oh * p.ow) as usize);
    let kdim = (c.cin * c.k * c.k) as usize;
    let x = signed(c.cin as usize * hw, 1);
    let w: Vec<f32> = signed(c.cout as usize * kdim, 2)
        .iter()
        .map(|v| v / (kdim as f32).sqrt())
        .collect();
    let bias = signed(c.cout as usize, 3);
    let want = conv_cpu(&x, &w, &bias, c.cout as usize, &p);

    let (total, ch0) = slice.unwrap_or((c.cout, 0));
    let (dx, dw, db) = (
        DeviceBuffer::from_slice(&x).unwrap(),
        DeviceBuffer::from_slice(&w).unwrap(),
        DeviceBuffer::from_slice(&bias).unwrap(),
    );
    const POISON: f32 = -777.0;
    let mut dy = DeviceBuffer::from_slice(&vec![POISON; total as usize * ohw]).unwrap();
    // Output view at row `ch0` of a `[total x ohw]` buffer.
    let y = TensorMut::new(
        unsafe { dy.as_mut_ptr().add(ch0 as usize * ohw) },
        c.cout,
        ohw as u32,
        ohw as u32,
    );
    let (tx, tw, tb) = (
        dx.tensor(c.cin, hw as u32),
        dw.tensor(c.cout, kdim as u32),
        Tensor::vector(db.as_ptr(), c.cout),
    );
    match cfg {
        Some(cfg) => launch_conv_with(cfg, &module, None, tx, tw, tb, y, &p).unwrap(),
        None => launch_conv(&module, None, tx, tw, tb, y, &p).unwrap(),
    }
    let got = dy.copy_to_vec().unwrap();
    let lo = ch0 as usize * ohw;
    let hi = lo + c.cout as usize * ohw;
    let err = rel_err(&got[lo..hi], &want);
    assert!(
        err <= 1e-4,
        "conv {}->{} {}x{} k{} s{} {:?}: rel err {err:e}",
        c.cin,
        c.cout,
        c.h,
        c.w,
        c.k,
        c.stride,
        cfg
    );
    assert!(
        got[..lo].iter().chain(&got[hi..]).all(|&v| v == POISON),
        "conv wrote outside its slice"
    );
}

macro_rules! conv_test {
    ($name:ident, $cin:expr, $cout:expr, $h:expr, $w:expr, $k:expr, $s:expr, $act:expr) => {
        #[test]
        fn $name() {
            let c = Case {
                cin: $cin,
                cout: $cout,
                h: $h,
                w: $w,
                k: $k,
                stride: $s,
                act: $act,
            };
            check_conv(&c, None, None);
        }
    };
}

conv_test!(conv_3_32_k3s2_224x160, 3, 32, 160, 224, 3, 2, 1);
conv_test!(conv_32_64_k3s2_112x80, 32, 64, 80, 112, 3, 2, 1);
conv_test!(conv_64_64_k3_56x40, 64, 64, 40, 56, 3, 1, 1);
conv_test!(conv_128_64_k1_28x20, 128, 64, 20, 28, 1, 1, 1);
conv_test!(conv_256_256_k3_14x10, 256, 256, 10, 14, 3, 1, 1);
conv_test!(conv_512_30_k1_7x5_linear, 512, 30, 5, 7, 1, 1, 0);
conv_test!(conv_ragged_5x7_cout30, 19, 30, 5, 7, 3, 1, 1);
conv_test!(conv_ragged_k3s2_odd, 7, 41, 9, 13, 3, 2, 1);

#[test]
fn conv_all_configs() {
    let c = Case {
        cin: 37,
        cout: 70,
        h: 13,
        w: 17,
        k: 3,
        stride: 1,
        act: 1,
    };
    for cfg in Config::ALL {
        check_conv(&c, Some(cfg), None);
    }
}

#[test]
fn conv_into_channel_slice() {
    let c = Case {
        cin: 32,
        cout: 32,
        h: 10,
        w: 14,
        k: 3,
        stride: 1,
        act: 1,
    };
    check_conv(&c, None, Some((96, 32)));
    check_conv(&c, Some(Config::C64x64), Some((70, 5)));
}

fn load() -> Module {
    runner::load().unwrap().1
}

fn maxpool_cpu(x: &[f32], c: usize, p: &PoolParams) -> Vec<f32> {
    let (h, w, oh, ow) = (p.h as usize, p.w as usize, p.oh as usize, p.ow as usize);
    let off = (p.pad / 2) as isize;
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..p.size as isize {
                    for kx in 0..p.size as isize {
                        let iy = (oy as isize * p.stride as isize) - off + ky;
                        let ix = (ox as isize * p.stride as isize) - off + kx;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            m = m.max(x[ch * h * w + iy as usize * w + ix as usize]);
                        }
                    }
                }
                out[ch * oh * ow + oy * ow + ox] = m;
            }
        }
    }
    out
}

fn check_maxpool(c: u32, h: u32, w: u32, size: u32, stride: u32, pad: Option<u32>) {
    let module = load();
    let p = PoolParams::new(h, w, size, stride, pad);
    let x = signed((c * h * w) as usize, 7);
    let want = maxpool_cpu(&x, c as usize, &p);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::from_slice(&vec![-777.0f32; want.len()]).unwrap();
    launch_maxpool(
        &module,
        None,
        dx.tensor(c, h * w),
        dy.tensor_mut(c, p.oh * p.ow),
        &p,
    )
    .unwrap();
    assert_eq!(
        dy.copy_to_vec().unwrap(),
        want,
        "maxpool {c}x{h}x{w} s{size}/{stride}"
    );
}

#[test]
fn maxpool_2x2_56x40x64() {
    check_maxpool(64, 40, 56, 2, 2, None);
}

#[test]
fn maxpool_odd_sizes() {
    check_maxpool(5, 7, 9, 2, 2, None);
    check_maxpool(13, 5, 7, 2, 2, None);
    check_maxpool(3, 11, 6, 3, 2, None);
    check_maxpool(3, 8, 8, 3, 1, Some(2));
    check_maxpool(4, 8, 8, 2, 1, Some(0));
}

#[test]
fn upsample_7x5x128() {
    let module = load();
    let (c, h, w, s) = (128u32, 5u32, 7u32, 2u32);
    let x = signed((c * h * w) as usize, 9);
    let (oh, ow) = ((h * s) as usize, (w * s) as usize);
    let mut want = vec![0f32; c as usize * oh * ow];
    for ch in 0..c as usize {
        for y in 0..oh {
            for xx in 0..ow {
                want[ch * oh * ow + y * ow + xx] =
                    x[ch * (h * w) as usize + (y / s as usize) * w as usize + xx / s as usize];
            }
        }
    }
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::from_slice(&vec![-777.0f32; want.len()]).unwrap();
    launch_upsample(
        &module,
        None,
        dx.tensor(c, h * w),
        dy.tensor_mut(c, (oh * ow) as u32),
        w,
        s,
    )
    .unwrap();
    assert_eq!(dy.copy_to_vec().unwrap(), want);
}

#[test]
fn concat_into_slices() {
    let module = load();
    let hw = 35 * 3 + 1; // ragged
    let (ca, cb, cg) = (13usize, 21usize, 10usize);
    let (a, b) = (signed(ca * hw, 11), signed(cb * hw, 12));
    let total = ca + cb + cg;
    let (da, db) = (
        DeviceBuffer::from_slice(&a).unwrap(),
        DeviceBuffer::from_slice(&b).unwrap(),
    );
    let mut dy = DeviceBuffer::from_slice(&vec![-777.0f32; total * hw]).unwrap();
    let base = dy.as_mut_ptr();
    let view = |ch0: usize, rows: usize| {
        TensorMut::new(
            unsafe { base.add(ch0 * hw) },
            rows as u32,
            hw as u32,
            hw as u32,
        )
    };
    launch_concat(&module, None, da.tensor(ca as u32, hw as u32), view(0, ca)).unwrap();
    launch_concat(&module, None, db.tensor(cb as u32, hw as u32), view(ca, cb)).unwrap();
    // Route group slice: second half of `b` (zero-copy view) into the last slice.
    let half = Tensor::new(
        unsafe { db.as_ptr().add(11 * hw) },
        10,
        hw as u32,
        hw as u32,
    );
    launch_concat(&module, None, half, view(ca + cb, cg)).unwrap();
    let got = dy.copy_to_vec().unwrap();
    let mut want = [a, b.clone()].concat();
    want.extend_from_slice(&b[11 * hw..]);
    assert_eq!(got, want);
}
