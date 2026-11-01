//! GPU training-backward ops against straightforward CPU references (f64 accumulation where the
//! math is exact, the Darknet formula where the op is a port of one).

use hip_rt::DeviceBuffer;
use hip_tile::{Tensor, TensorMut};
use kernels::backward::bn_bwd::{BnBwdParams, launch_bn_bwd};
use kernels::backward::ConvBwdParams;
use kernels::backward::conv_bwd_dw::launch_conv_bwd_dw;
use kernels::backward::conv_bwd_dx::launch_conv_bwd_dx;
use kernels::backward::bn_fwd::{BnFwdParams, launch_bn_fwd};
use kernels::backward::bn_stats::{BnStatsParams, launch_bn_stats};
use kernels::backward::maxpool_bwd::launch_maxpool_bwd;
use kernels::backward::route_bwd::launch_route_bwd;
use kernels::backward::upsample_bwd::launch_upsample_bwd;
use kernels::backward::util::{SgdParams, launch_acc, launch_axpy_acc, launch_running_update, launch_sgd_update, launch_zero};
use kernels::yolo::conv::ConvParams;
use kernels::yolo::maxpool::PoolParams;

fn signed(len: usize, seed: u64) -> Vec<f32> {
    runner::random_vec(len, seed).into_iter().map(|v| v * 2.0 - 1.0).collect()
}

/// Same measure for f64 outputs (the convolution-gradient checks subtract the poison first).
fn rel_err_64(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    let mut worst = 0f64;
    let mut worst_i = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(scale * 1e-2);
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    if worst > 1e-6 {
        eprintln!("  [rel_err_64] worst {worst:e} at {worst_i}: got {} want {} (scale {scale:e})", got[worst_i], want[worst_i]);
    }
    worst
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    got.iter().zip(want).map(|(g, w)| (f64::from(*g) - w).abs() / w.abs().max(scale * 1e-2)).fold(0.0, f64::max)
}

#[test]
fn util_ops_match_the_formulas() {
    let (_dev, module) = runner::load().unwrap();
    let n = 1000usize;
    let x = signed(n, 1);
    let y = signed(n, 2);
    let bx = DeviceBuffer::from_slice(&x).unwrap();
    let mut by = DeviceBuffer::from_slice(&y).unwrap();

    // zero
    launch_zero(&module, None, by.tensor_mut(1, n as u32)).unwrap();
    assert!(by.copy_to_vec().unwrap().iter().all(|&v| v == 0.0));

    // axpy_acc / acc
    let mut y2 = signed(n, 3);
    let mut by2 = DeviceBuffer::from_slice(&y2).unwrap();
    launch_axpy_acc(&module, None, 0.5, bx.tensor(1, n as u32), by2.tensor_mut(1, n as u32)).unwrap();
    let got = by2.copy_to_vec().unwrap();
    let want: Vec<f64> = x.iter().zip(&y2).map(|(a, b)| 0.5 * f64::from(*a) + f64::from(*b)).collect();
    assert!(rel_err(&got, &want) < 1e-6, "axpy_acc");
    launch_acc(&module, None, bx.tensor(1, n as u32), by2.tensor_mut(1, n as u32)).unwrap();
    y2 = got.clone();
    let got2 = by2.copy_to_vec().unwrap();
    let want2: Vec<f64> = x.iter().zip(&y2).map(|(a, b)| f64::from(*a) + f64::from(*b)).collect();
    assert!(rel_err(&got2, &want2) < 1e-6, "acc");

    // sgd_update with decay (weights) and without (bias/scales)
    let w = signed(n, 4);
    let u = signed(n, 5);
    let mut bw = DeviceBuffer::from_slice(&w).unwrap();
    let mut bu = DeviceBuffer::from_slice(&u).unwrap();
    let p = SgdParams { learning_rate: 0.01, momentum: 0.9, decay: 0.0005, batch: 4.0 };
    launch_sgd_update(&module, None, bw.tensor_mut(1, n as u32), bu.tensor_mut(1, n as u32), &p).unwrap();
    let gw = bw.copy_to_vec().unwrap();
    let gu = bu.copy_to_vec().unwrap();
    let mut worst_w = 0f64;
    let mut worst_u = 0f64;
    for i in 0..n {
        let expect_u = f64::from(u[i]) - 0.0005 * 4.0 * f64::from(w[i]);
        let expect_w = f64::from(w[i]) + (0.01 / 4.0) * expect_u;
        worst_w = worst_w.max((f64::from(gw[i]) - expect_w).abs() / expect_w.abs().max(1e-6));
        worst_u = worst_u.max((f64::from(gu[i]) - expect_u * 0.9).abs() / (expect_u * 0.9).abs().max(1e-6));
    }
    assert!(worst_w < 1e-5 && worst_u < 1e-5, "sgd_update: {worst_w:e} {worst_u:e}");

    // running_update
    let rolling = signed(16, 6);
    let stat = signed(16, 7);
    let mut br = DeviceBuffer::from_slice(&rolling).unwrap();
    let bs = DeviceBuffer::from_slice(&stat).unwrap();
    launch_running_update(&module, None, br.tensor_mut(1, 16), bs.tensor(1, 16), 0.9).unwrap();
    let got = br.copy_to_vec().unwrap();
    let want: Vec<f64> = rolling.iter().zip(&stat).map(|(r, s)| 0.9 * f64::from(*r) + 0.1 * f64::from(*s)).collect();
    assert!(rel_err(&got, &want) < 1e-6, "running_update");
}

fn maxpool_bwd_cpu(x: &[f32], dy: &[f32], c: usize, p: &PoolParams) -> Vec<f64> {
    let (h, w) = (p.h as usize, p.w as usize);
    let (oh, ow) = (p.oh as usize, p.ow as usize);
    let off = -((p.pad / 2) as isize);
    let mut dx = vec![0f64; c * h * w];
    for ch in 0..c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f32::NEG_INFINITY;
                let mut at = (usize::MAX, usize::MAX);
                for ky in 0..p.size as isize {
                    for kx in 0..p.size as isize {
                        let iy = off + (oy * p.stride as usize + ky as usize) as isize;
                        let jx = off + (ox * p.stride as usize + kx as usize) as isize;
                        if iy >= 0 && jx >= 0 && (iy as usize) < h && (jx as usize) < w {
                            let i = (iy as usize) * w + jx as usize;
                            let v = x[ch * h * w + i];
                            if v > best {
                                best = v;
                                at = (iy as usize, jx as usize);
                            }
                        }
                    }
                }
                if at.0 != usize::MAX {
                    dx[ch * h * w + at.0 * w + at.1] += f64::from(dy[ch * oh * ow + oy * ow + ox]);
                }
            }
        }
    }
    dx
}

fn check_maxpool_bwd(c: u32, h: u32, w: u32, size: u32, stride: u32, pad: Option<u32>) {
    let (_dev, module) = runner::load().unwrap();
    let p = PoolParams::new(h, w, size, stride, pad);
    let n_in = (c * h * w) as usize;
    let n_out = (c * p.oh * p.ow) as usize;
    let x = signed(n_in, 11);
    let dy = signed(n_out, 12);
    let want = maxpool_bwd_cpu(&x, &dy, c as usize, &p);

    let bx = DeviceBuffer::from_slice(&x).unwrap();
    let bdy = DeviceBuffer::from_slice(&dy).unwrap();
    // Poison the destination to prove the op accumulates rather than overwrites.
    const POISON: f32 = 0.25;
    let mut bdx = DeviceBuffer::from_slice(&vec![POISON; n_in]).unwrap();
    launch_maxpool_bwd(&module, None, bx.tensor(c, h * w), bdy.tensor(c, p.oh * p.ow), bdx.tensor_mut(c, h * w), &p).unwrap();
    let got = bdx.copy_to_vec().unwrap();
    let want_plus: Vec<f64> = want.iter().map(|v| v + f64::from(POISON)).collect();
    let err = rel_err(&got, &want_plus);
    assert!(err < 1e-5, "maxpool_bwd {c}x{h}x{w} size {size} stride {stride}: {err:e}");
}

#[test]
fn maxpool_bwd_matches_cpu() {
    check_maxpool_bwd(3, 10, 12, 2, 2, None);
    check_maxpool_bwd(2, 9, 7, 2, 2, None);
    // Overlapping windows: stride < size, the case a scatter would get wrong.
    check_maxpool_bwd(2, 8, 8, 3, 2, Some(2));
    check_maxpool_bwd(1, 5, 5, 3, 1, Some(2));
}

#[test]
fn upsample_bwd_matches_cpu() {
    let (_dev, module) = runner::load().unwrap();
    let (c, h, w, s) = (3u32, 4u32, 5u32, 2u32);
    let n_in = (c * h * w) as usize;
    let n_out = (c * (h * s) * (w * s)) as usize;
    let dy = signed(n_out, 21);
    let mut want = vec![0f64; n_in];
    for ch in 0..c as usize {
        for iy in 0..h as usize {
            for ix in 0..w as usize {
                let mut acc = 0f64;
                for dyi in 0..s as usize {
                    for dxi in 0..s as usize {
                        let oy = iy * s as usize + dyi;
                        let ox = ix * s as usize + dxi;
                        acc += f64::from(dy[ch * (h * s * w * s) as usize + oy * (w * s) as usize + ox]);
                    }
                }
                want[ch * (h * w) as usize + iy * w as usize + ix] = acc;
            }
        }
    }
    let bdy = DeviceBuffer::from_slice(&dy).unwrap();
    let mut bdx = DeviceBuffer::from_slice(&vec![0.0f32; n_in]).unwrap();
    launch_upsample_bwd(&module, None, bdy.tensor(c, (h * s) * (w * s)), bdx.tensor_mut(c, h * w), w, s).unwrap();
    let got = bdx.copy_to_vec().unwrap();
    let err = rel_err(&got, &want);
    assert!(err < 1e-6, "upsample_bwd: {err:e}");
}

#[test]
fn route_bwd_accumulates_into_the_source_block() {
    let (_dev, module) = runner::load().unwrap();
    let (csrc, cg, hw) = (6u32, 3u32, 20u32);
    let src_block = signed((cg * hw) as usize, 31);
    let mut dst = signed((csrc * hw) as usize, 32);
    let before = dst.clone();
    let bsrc = DeviceBuffer::from_slice(&src_block).unwrap();
    let mut bdst = DeviceBuffer::from_slice(&dst).unwrap();
    // group 1 of 2 -> rows [3, 6) of the destination.
    let dst_view = TensorMut::new(unsafe { bdst.as_mut_ptr().add((cg * hw) as usize) }, cg, hw, hw);
    launch_route_bwd(&module, None, bsrc.tensor(cg, hw), dst_view).unwrap();
    dst = bdst.copy_to_vec().unwrap();
    let mut worst = 0f64;
    for i in 0..dst.len() {
        let row = i / hw as usize;
        let expect = f64::from(before[i])
            + if row >= cg as usize && row < 2 * cg as usize {
                f64::from(src_block[i - cg as usize * hw as usize])
            } else {
                0.0
            };
        worst = worst.max((f64::from(dst[i]) - expect).abs());
    }
    assert!(worst < 1e-6, "route_bwd: {worst:e}");
}

fn bn_cpu(x: &[f32], batch: usize, c: usize, spatial: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0f64; c];
    let mut var = vec![0f64; c];
    let n = (batch * spatial) as f64;
    for ch in 0..c {
        let mut sum = 0f64;
        for b in 0..batch {
            for j in 0..spatial {
                sum += f64::from(x[(b * c + ch) * spatial + j]);
            }
        }
        let m = sum / n;
        let mut ss = 0f64;
        for b in 0..batch {
            for j in 0..spatial {
                let d = f64::from(x[(b * c + ch) * spatial + j]) - m;
                ss += d * d;
            }
        }
        mean[ch] = m;
        var[ch] = if n > 1.0 { ss / (n - 1.0) } else { 0.0 };
    }
    (mean, var)
}

fn check_bn_stats(batch: u32, c: u32, spatial: u32) {
    let (_dev, module) = runner::load().unwrap();
    let x = signed((batch * c * spatial) as usize, 41);
    let (want_m, want_v) = bn_cpu(&x, batch as usize, c as usize, spatial as usize);
    let bx = DeviceBuffer::from_slice(&x).unwrap();
    let mut bm = DeviceBuffer::from_slice(&vec![0.0f32; c as usize]).unwrap();
    let mut bv = DeviceBuffer::from_slice(&vec![0.0f32; c as usize]).unwrap();
    // View channel 0 of the batch as [c x spatial] with row stride c*spatial.
    let xv = Tensor::new(bx.as_ptr(), c, spatial, c * spatial);
    launch_bn_stats(&module, None, xv, bm.tensor_mut(1, c), bv.tensor_mut(1, c), &BnStatsParams { batch, spatial }).unwrap();
    let m = bm.copy_to_vec().unwrap();
    let v = bv.copy_to_vec().unwrap();
    let e_m = rel_err(&m, &want_m);
    assert!(e_m < 1e-5, "bn_stats mean {batch}/{c}/{spatial}: got {m:?} want {want_m:?} (err {e_m:e})");
    let e_v = rel_err(&v, &want_v);
    assert!(e_v < 1e-4, "bn_stats var {batch}/{c}/{spatial}: got {v:?} want {want_v:?} (err {e_v:e})");
}

#[test]
fn bn_stats_matches_cpu() {
    check_bn_stats(3, 4, 40);
    check_bn_stats(1, 2, 7); // batch of one: variance must be 0
    check_bn_stats(9, 3, 300); // more rows than TILE_M, ragged tail
}

#[test]
fn bn_bwd_matches_the_darknet_formulas() {
    let (_dev, module) = runner::load().unwrap();
    let (batch, c, spatial) = (3usize, 2usize, 24usize);
    let x = signed(batch * c * spatial, 51);
    let dy0 = signed(batch * c * spatial, 52);
    let scales: Vec<f32> = vec![1.5, 0.75];
    let (mean, var) = bn_cpu(&x, batch, c, spatial);
    // xhat exactly as the forward computes it (epsilon inside the sqrt).
    let mut xhat = vec![0f32; x.len()];
    for ch in 0..c {
        let inv = 1.0 / (var[ch] + 1e-5).sqrt();
        for b in 0..batch {
            for j in 0..spatial {
                let i = (b * c + ch) * spatial + j;
                xhat[i] = ((f64::from(x[i]) - mean[ch]) * inv) as f32;
            }
        }
    }
    // CPU dx per the ported formula.
    let n = (batch * spatial) as f64;
    let mut want = vec![0f64; x.len()];
    let mut want_dscale = vec![0f64; c];
    for ch in 0..c {
        let g = f64::from(scales[ch]);
        let mut sdx = 0f64;
        let mut sd = 0f64;
        let mut sdyx = 0f64;
        for b in 0..batch {
            for j in 0..spatial {
                let i = (b * c + ch) * spatial + j;
                sdx += f64::from(xhat[i]) * f64::from(dy0[i]);
                sd += g * f64::from(dy0[i]);
                sdyx += g * f64::from(dy0[i]) * (f64::from(x[i]) - mean[ch]);
            }
        }
        want_dscale[ch] = sdx;
        let mean_delta = -sd / (var[ch] + 1e-5).sqrt();
        let variance_delta = -0.5 * sdyx / (var[ch] + 1e-5).powf(1.5);
        let inv_std = 1.0 / (var[ch].sqrt() + 1e-5);
        for b in 0..batch {
            for j in 0..spatial {
                let i = (b * c + ch) * spatial + j;
                want[i] = g * f64::from(dy0[i]) * inv_std + variance_delta * 2.0 * (f64::from(x[i]) - mean[ch]) / n
                    + mean_delta / n;
            }
        }
    }

    let bx = DeviceBuffer::from_slice(&x).unwrap();
    let bxh = DeviceBuffer::from_slice(&xhat).unwrap();
    let bdy = DeviceBuffer::from_slice(&dy0).unwrap();
    let bm = DeviceBuffer::from_slice(&mean.iter().map(|v| *v as f32).collect::<Vec<_>>()).unwrap();
    let bv = DeviceBuffer::from_slice(&var.iter().map(|v| *v as f32).collect::<Vec<_>>()).unwrap();
    let bg = DeviceBuffer::from_slice(&scales).unwrap();
    let mut bds = DeviceBuffer::from_slice(&vec![0.0f32; c]).unwrap();

    let view = |b: &DeviceBuffer<f32>| Tensor::new(b.as_ptr(), c as u32, spatial as u32, (c * spatial) as u32);
    launch_bn_bwd(
        &module,
        None,
        view(&bx),
        view(&bxh),
        TensorMut::new(bdy.as_ptr().cast_mut(), c as u32, spatial as u32, (c * spatial) as u32),
        bm.tensor(1, c as u32),
        bv.tensor(1, c as u32),
        bg.tensor(1, c as u32),
        bds.tensor_mut(1, c as u32),
        &BnBwdParams { batch: batch as u32, spatial: spatial as u32 },
    )
    .unwrap();

    // Compare channel-major views to the CPU layout.
    let got = bdy.copy_to_vec().unwrap();
    let mut worst = 0f64;
    for b in 0..batch {
        for ch in 0..c {
            for j in 0..spatial {
                let i = (b * c + ch) * spatial + j;
                worst = worst.max((f64::from(got[i]) - want[i]).abs() / want[i].abs().max(1e-3));
            }
        }
    }
    assert!(worst < 2e-4, "bn_bwd dx worst rel {worst:e}");
    let dscale = bds.copy_to_vec().unwrap();
    let err = rel_err(&dscale, &want_dscale);
    assert!(err < 1e-4, "bn_bwd dscale: {err:e}");
}

/// `(cin, cout, h, w, k, stride, pad, batch)` for the convolution-gradient tests.
struct ConvCase {
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
    batch: usize,
}

impl ConvCase {
    fn params(&self) -> ConvParams {
        ConvParams::with_cout(
            self.cin as u32,
            self.cout as u32,
            self.h as u32,
            self.w as u32,
            self.k as u32,
            self.stride as u32,
            self.pad as u32,
            ConvParams::LINEAR,
        )
    }
    fn oh(&self) -> usize {
        (self.h + 2 * self.pad - self.k) / self.stride + 1
    }
    fn ow(&self) -> usize {
        (self.w + 2 * self.pad - self.k) / self.stride + 1
    }
}

fn conv_dw_cpu(c: &ConvCase, x: &[f32], delta: &[f32]) -> Vec<f64> {
    let (oh, ow) = (c.oh(), c.ow());
    let mut dw = vec![0f64; c.cout * c.cin * c.k * c.k];
    for b in 0..c.batch {
        for co in 0..c.cout {
            for ci in 0..c.cin {
                for ky in 0..c.k {
                    for kx in 0..c.k {
                        let mut acc = 0f64;
                        for oy in 0..oh {
                            let iy = (oy * c.stride + ky) as isize - c.pad as isize;
                            if iy < 0 || iy as usize >= c.h {
                                continue;
                            }
                            for ox in 0..ow {
                                let ix = (ox * c.stride + kx) as isize - c.pad as isize;
                                if ix < 0 || ix as usize >= c.w {
                                    continue;
                                }
                                let d = delta[(b * c.cout + co) * oh * ow + oy * ow + ox];
                                let xv = x[(b * c.cin + ci) * c.h * c.w + iy as usize * c.w + ix as usize];
                                acc += f64::from(d) * f64::from(xv);
                            }
                        }
                        dw[((co * c.cin + ci) * c.k + ky) * c.k + kx] += acc;
                    }
                }
            }
        }
    }
    dw
}

fn conv_dx_cpu(c: &ConvCase, w: &[f32], delta: &[f32]) -> Vec<f64> {
    let (oh, ow) = (c.oh(), c.ow());
    let mut dx = vec![0f64; c.batch * c.cin * c.h * c.w];
    for b in 0..c.batch {
        for ci in 0..c.cin {
            for iy in 0..c.h {
                for ix in 0..c.w {
                    let mut acc = 0f64;
                    for co in 0..c.cout {
                        for ky in 0..c.k {
                            let ny = iy as isize + c.pad as isize - ky as isize;
                            if ny < 0 {
                                continue;
                            }
                            for kx in 0..c.k {
                                let nx = ix as isize + c.pad as isize - kx as isize;
                                if nx < 0 {
                                    continue;
                                }
                                if !(ny as usize).is_multiple_of(c.stride) || !(nx as usize).is_multiple_of(c.stride) {
                                    continue;
                                }
                                let (oy, ox) = (ny as usize / c.stride, nx as usize / c.stride);
                                if oy >= oh || ox >= ow {
                                    continue;
                                }
                                let wv = w[((co * c.cin + ci) * c.k + ky) * c.k + kx];
                                let d = delta[(b * c.cout + co) * oh * ow + oy * ow + ox];
                                acc += f64::from(wv) * f64::from(d);
                            }
                        }
                    }
                    dx[(b * c.cin + ci) * c.h * c.w + iy * c.w + ix] += acc;
                }
            }
        }
    }
    dx
}

fn check_conv_bwd(c: ConvCase) {
    let (_dev, module) = runner::load().unwrap();
    let (hw, ohw) = (c.h * c.w, c.oh() * c.ow());
    let kdim = c.cin * c.k * c.k;
    let x = signed(c.batch * c.cin * hw, 61);
    let delta = signed(c.batch * c.cout * ohw, 62);
    let w = signed(c.cout * kdim, 63);
    let want_dw = conv_dw_cpu(&c, &x, &delta);
    let want_dx = conv_dx_cpu(&c, &w, &delta);

    // The kernels see ONE image's shape; the batch is reached by stride inside the kernel, so the
    // views below are `[Cin x HW]` / `[Cout x OHW]` with the images laid out back to back.
    let bx = DeviceBuffer::from_slice(&x).unwrap();
    let bd = DeviceBuffer::from_slice(&delta).unwrap();
    let bw = DeviceBuffer::from_slice(&w).unwrap();
    let p = ConvBwdParams::new(c.params(), c.batch as u32);

    // One image's shape, contiguous images: `ld` is the per-image plane count, so image `b` sits
    // at `b * rows * ld` — exactly what the kernels stride over.
    const POISON: f32 = -0.5;
    let x_view = Tensor::new(bx.as_ptr(), c.cin as u32, hw as u32, hw as u32);
    let d_view = Tensor::new(bd.as_ptr(), c.cout as u32, ohw as u32, ohw as u32);

    // dW into a poisoned buffer, to prove the kernel accumulates.
    let mut bdw2 = DeviceBuffer::from_slice(&vec![POISON; c.cout * kdim]).unwrap();
    launch_conv_bwd_dw(&module, None, d_view, x_view, bdw2.tensor_mut(c.cout as u32, kdim as u32), &p).unwrap();
    let got: Vec<f64> = bdw2.copy_to_vec().unwrap().iter().map(|v| f64::from(*v) - f64::from(POISON)).collect();
    let err = rel_err_64(&got, &want_dw);
    assert!(err < 2e-4, "conv_bwd_dw {:?}: {err:e}", (c.cin, c.cout, c.h, c.w, c.k));

    // dX, also poisoned.
    let mut bdx = DeviceBuffer::from_slice(&vec![POISON; c.batch * c.cin * hw]).unwrap();
    let w_view = Tensor::new(bw.as_ptr(), c.cout as u32, kdim as u32, kdim as u32);
    launch_conv_bwd_dx(&module, None, d_view, w_view, bdx.tensor_mut(c.cin as u32, hw as u32), &p).unwrap();
    let got: Vec<f64> = bdx.copy_to_vec().unwrap().iter().map(|v| f64::from(*v) - f64::from(POISON)).collect();
    let err = rel_err_64(&got, &want_dx);
    assert!(err < 2e-4, "conv_bwd_dx {:?}: {err:e}", (c.cin, c.cout, c.h, c.w, c.k));
}

/// Both convolution gradients against an f64 CPU reference, batched, with the destination poisoned
/// to prove the kernels accumulate (the poison is subtracted before the comparison — comparing the
/// summed values would cancel to ~0 for entries where the gradient happens to equal `-POISON`).
#[test]
fn conv_bwd_matches_cpu() {
    // Single-image cases, all exact.
    check_conv_bwd(ConvCase { cin: 2, cout: 3, h: 6, w: 5, k: 3, stride: 1, pad: 1, batch: 1 });
    check_conv_bwd(ConvCase { cin: 1, cout: 1, h: 4, w: 4, k: 3, stride: 1, pad: 1, batch: 1 });
    check_conv_bwd(ConvCase { cin: 3, cout: 4, h: 8, w: 8, k: 3, stride: 2, pad: 0, batch: 1 });
    check_conv_bwd(ConvCase { cin: 5, cout: 6, h: 4, w: 4, k: 1, stride: 1, pad: 0, batch: 1 });
    // Wider than one tile: forces the k-loop and the co-loop to iterate.
    check_conv_bwd(ConvCase { cin: 4, cout: 20, h: 10, w: 9, k: 3, stride: 1, pad: 1, batch: 1 });
}

/// Batched cases (`batch > 1`). `conv_bwd_dx` once folded every image into image 0's `dx` slice;
/// each image must land at `dx + b·Cin·ld`. The dX buffer here holds the whole batch.
#[test]
fn conv_bwd_matches_cpu_batched() {
    check_conv_bwd(ConvCase { cin: 2, cout: 3, h: 6, w: 5, k: 3, stride: 1, pad: 1, batch: 2 });
    check_conv_bwd(ConvCase { cin: 3, cout: 4, h: 8, w: 8, k: 3, stride: 2, pad: 0, batch: 2 });
    check_conv_bwd(ConvCase { cin: 5, cout: 6, h: 4, w: 4, k: 1, stride: 1, pad: 0, batch: 2 });
    check_conv_bwd(ConvCase { cin: 4, cout: 20, h: 10, w: 9, k: 3, stride: 1, pad: 1, batch: 2 });
}

/// `bn_stats` → `bn_fwd` against the f64 formulas: `xhat = (x − m)/sqrt(var + 1e-5)`,
/// `y = act(xhat·scale + bias)`, over a batch, linear and leaky.
#[test]
fn bn_fwd_matches_cpu() {
    let (_dev, module) = runner::load().unwrap();
    let (batch, c, spatial) = (3usize, 4usize, 37usize);
    let len = batch * c * spatial;
    // A non-zero channel mean makes the one-pass variance honest.
    let x: Vec<f32> = signed(len, 71).iter().enumerate().map(|(i, v)| v + 2.0 * ((i / spatial) % c) as f32).collect();
    let scale = vec![1.5f32, -0.75, 0.5, 2.0];
    let bias = vec![0.25f32, -0.5, 1.0, 0.0];
    let (mean, var) = bn_cpu(&x, batch, c, spatial);
    for act in [ConvParams::LINEAR, ConvParams::LEAKY] {
        let (bx, bs, bb) = (
            DeviceBuffer::from_slice(&x).unwrap(),
            DeviceBuffer::from_slice(&scale).unwrap(),
            DeviceBuffer::from_slice(&bias).unwrap(),
        );
        let mut bm = DeviceBuffer::zeroed(c).unwrap();
        let mut bv = DeviceBuffer::zeroed(c).unwrap();
        let mut bxh = DeviceBuffer::from_slice(&vec![-9.0f32; len]).unwrap();
        let mut by = DeviceBuffer::from_slice(&vec![-9.0f32; len]).unwrap();
        let xv = Tensor::new(bx.as_ptr(), c as u32, spatial as u32, (c * spatial) as u32);
        launch_bn_stats(&module, None, xv, bm.tensor_mut(1, c as u32), bv.tensor_mut(1, c as u32), &BnStatsParams { batch: batch as u32, spatial: spatial as u32 }).unwrap();
        let flat = Tensor::vector(bx.as_ptr(), len as u32);
        let p = BnFwdParams { channels: c as u32, spatial: spatial as u32, act };
        launch_bn_fwd(
            &module, None, flat,
            Tensor::vector(bm.as_ptr(), c as u32), Tensor::vector(bv.as_ptr(), c as u32),
            Tensor::vector(bs.as_ptr(), c as u32), Tensor::vector(bb.as_ptr(), c as u32),
            TensorMut::vector(bxh.as_mut_ptr(), len as u32), TensorMut::vector(by.as_mut_ptr(), len as u32), &p,
        ).unwrap();
        let (mut want_xh, mut want_y) = (vec![0f64; len], vec![0f64; len]);
        for b in 0..batch {
            for ch in 0..c {
                let inv = 1.0 / (var[ch] + 1e-5).sqrt();
                for j in 0..spatial {
                    let i = (b * c + ch) * spatial + j;
                    let xh = (f64::from(x[i]) - mean[ch]) * inv;
                    let mut y = xh * f64::from(scale[ch]) + f64::from(bias[ch]);
                    if act == ConvParams::LEAKY && y < 0.0 {
                        y *= 0.1;
                    }
                    want_xh[i] = xh;
                    want_y[i] = y;
                }
            }
        }
        let e_xh = rel_err_64(&bxh.copy_to_vec().unwrap().iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), &want_xh);
        let e_y = rel_err_64(&by.copy_to_vec().unwrap().iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), &want_y);
        assert!(e_xh < 1e-4, "bn_fwd xhat act {act}: {e_xh:e}");
        assert!(e_y < 1e-4, "bn_fwd y act {act}: {e_y:e}");
    }
}
