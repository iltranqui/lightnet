//! GPU (cuda-oxide) vs straightforward CPU references, per implemented op.
//! Sizes are deliberately not multiples of the block size.

use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::{OxideBackend, OxideError};

const TOL: f32 = 1e-4;

struct Rng(u64);

impl Rng {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        lo + (hi - lo) * (r as f32 / (1u64 << 24) as f32)
    }

    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(lo, hi)).collect()
    }
}

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL + TOL * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

const KINDS: [F32ActivationKind; 7] = [
    F32ActivationKind::Linear,
    F32ActivationKind::Logistic,
    F32ActivationKind::Relu,
    F32ActivationKind::Leaky,
    F32ActivationKind::Mish,
    F32ActivationKind::Swish,
    F32ActivationKind::Tanh,
];

fn cpu_activate(kind: F32ActivationKind, x: f32) -> f32 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Logistic => 1.0 / (1.0 + (-x).exp()),
        F32ActivationKind::Relu => x.max(0.0),
        F32ActivationKind::Leaky => {
            if x > 0.0 {
                x
            } else {
                0.1 * x
            }
        }
        F32ActivationKind::Mish => x * (1.0 + x.exp()).ln().tanh(),
        F32ActivationKind::Swish => x / (1.0 + (-x).exp()),
        F32ActivationKind::Tanh => x.tanh(),
    }
}

fn cpu_gradient(kind: F32ActivationKind, x: f32) -> f32 {
    match kind {
        F32ActivationKind::Linear => 1.0,
        F32ActivationKind::Logistic => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        F32ActivationKind::Relu => (x > 0.0) as i32 as f32,
        F32ActivationKind::Leaky => {
            if x > 0.0 {
                1.0
            } else {
                0.1
            }
        }
        F32ActivationKind::Mish => {
            let softplus = if x > 20.0 {
                x
            } else if x < -20.0 {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tsp = softplus.tanh();
            x * (1.0 - tsp * tsp) * grad_sp + tsp
        }
        F32ActivationKind::Swish => {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }
        F32ActivationKind::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
    }
}

#[test]
fn upload_zeros_download_roundtrip() {
    let b = backend();
    let host = Rng(1).vec(1001, -3.0, 3.0);
    let t = b.upload_f32(&[7, 11, 13], &host).unwrap();
    assert_eq!(t.shape(), &[7, 11, 13]);
    assert_eq!(b.download_f32(&t).unwrap(), host);
    let z = b.zeros_f32(&[5, 3]).unwrap();
    assert_eq!(b.download_f32(&z).unwrap(), vec![0.0; 15]);
    // Shape/length mismatch is an error, not a panic.
    assert!(matches!(
        b.upload_f32(&[2, 2], &host),
        Err(OxideError::Shape(_))
    ));
}

#[test]
fn activations_forward_and_backward() {
    let b = backend();
    let mut rng = Rng(2);
    let n = 1237;
    // Include values beyond the Mish +-20 thresholds.
    let mut x = rng.vec(n, -6.0, 6.0);
    x[0] = 25.0;
    x[1] = -25.0;
    x[2] = 0.0;
    let delta = rng.vec(n, -2.0, 2.0);
    let xt = b.upload_f32(&[n], &x).unwrap();
    let dt = b.upload_f32(&[n], &delta).unwrap();
    for kind in KINDS {
        let fwd = b.activation_f32(&xt, kind, cfg()).unwrap();
        let want: Vec<f32> = x.iter().map(|&v| cpu_activate(kind, v)).collect();
        assert_close(&format!("{kind:?} fwd"), &b.download_f32(&fwd).unwrap(), &want);

        let bwd = b.activation_backward_f32(&xt, &dt, kind, cfg()).unwrap();
        let want: Vec<f32> = x
            .iter()
            .zip(&delta)
            .map(|(&v, &d)| d * cpu_gradient(kind, v))
            .collect();
        assert_close(&format!("{kind:?} bwd"), &b.download_f32(&bwd).unwrap(), &want);
    }
}

#[test]
fn bias_add_and_channel_scale() {
    let b = backend();
    let mut rng = Rng(3);
    // [batch=3, channels=5, stride=7*11] -> 1155 elements per sample.
    let (batch, channels, stride) = (3usize, 5usize, 77usize);
    let x = rng.vec(batch * channels * stride, -2.0, 2.0);
    let v = rng.vec(channels, -2.0, 2.0);
    let xt = b.upload_f32(&[batch, channels, stride], &x).unwrap();
    let vt = b.upload_f32(&[channels], &v).unwrap();

    let got = b.bias_add_f32(&xt, &vt, channels, stride, cfg()).unwrap();
    let want: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &e)| e + v[(i / stride) % channels])
        .collect();
    assert_close("bias_add", &b.download_f32(&got).unwrap(), &want);

    let got = b.channel_scale_f32(&xt, &vt, channels, stride, cfg()).unwrap();
    let want: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &e)| e * v[(i / stride) % channels])
        .collect();
    assert_close("channel_scale", &b.download_f32(&got).unwrap(), &want);

    // Bad layout is an error.
    assert!(b.bias_add_f32(&xt, &vt, channels + 1, stride, cfg()).is_err());
}

#[test]
fn bias_backward_reduction() {
    let b = backend();
    let mut rng = Rng(4);
    // Stride larger than 256 to exercise the strided inner loop.
    for &(batch, channels, stride) in &[(2usize, 6usize, 301usize), (1, 3, 5), (4, 9, 1)] {
        let g = rng.vec(batch * channels * stride, -1.0, 1.0);
        let gt = b.upload_f32(&[batch, channels, stride], &g).unwrap();
        let got = b.bias_backward_f32(&gt, channels, stride, cfg()).unwrap();
        let mut want = vec![0.0f32; channels];
        for (i, &e) in g.iter().enumerate() {
            want[(i / stride) % channels] += e;
        }
        assert_close(
            &format!("bias_backward {batch}x{channels}x{stride}"),
            &b.download_f32(&got).unwrap(),
            &want,
        );
    }
}

#[test]
fn vector_add_shortcut_saxpy() {
    let b = backend();
    let mut rng = Rng(5);
    let n = 2049;
    let x = rng.vec(n, -4.0, 4.0);
    let y = rng.vec(n, -4.0, 4.0);
    let xt = b.upload_f32(&[n], &x).unwrap();
    let yt = b.upload_f32(&[n], &y).unwrap();

    let want: Vec<f32> = x.iter().zip(&y).map(|(a, c)| a + c).collect();
    let got = b.vector_add_f32(&xt, &yt, cfg()).unwrap();
    assert_close("vector_add", &b.download_f32(&got).unwrap(), &want);
    let got = b.shortcut_add_f32(&xt, &yt, cfg()).unwrap();
    assert_close("shortcut_add", &b.download_f32(&got).unwrap(), &want);

    let want: Vec<f32> = x.iter().zip(&y).map(|(a, c)| -0.37 * a + c).collect();
    let got = b.saxpy_f32(-0.37, &xt, &yt, cfg()).unwrap();
    assert_close("saxpy", &b.download_f32(&got).unwrap(), &want);

    let short = b.upload_f32(&[n - 1], &x[..n - 1]).unwrap();
    assert!(b.vector_add_f32(&short, &yt, cfg()).is_err());
}

#[test]
fn sgd_update_matches_darknet_rule() {
    let b = backend();
    let mut rng = Rng(6);
    let n = 1999;
    let w = rng.vec(n, -1.0, 1.0);
    let u = rng.vec(n, -1.0, 1.0);
    let (lr, momentum, decay, batch) = (0.01f32, 0.9f32, 5e-4f32, 8usize);
    let wt = b.upload_f32(&[n], &w).unwrap();
    let ut = b.upload_f32(&[n], &u).unwrap();
    let (nw, nu) = b
        .sgd_update_f32(&wt, &ut, lr, momentum, decay, batch, cfg())
        .unwrap();

    // Reference: darknet-core `sgd_update`.
    let bf = batch as f32;
    let (mut want_w, mut want_u) = (w.clone(), u.clone());
    for (wi, ui) in want_w.iter_mut().zip(want_u.iter_mut()) {
        *ui += -decay * bf * *wi;
        *wi += (lr / bf) * *ui;
        *ui *= momentum;
    }
    assert_close("sgd weights", &b.download_f32(&nw).unwrap(), &want_w);
    assert_close("sgd updates", &b.download_f32(&nu).unwrap(), &want_u);
    // Inputs are left untouched (out-of-place).
    assert_eq!(b.download_f32(&wt).unwrap(), w);
}

#[test]
fn unimplemented_ops_return_error() {
    let b = backend();
    let t = b.zeros_f32(&[4]).unwrap();
    let err = b
        .upsample_f32(&t, &[4], 1, 2, 2, 1, 1.0, false, cfg())
        .err()
        .expect("upsample should be unimplemented");
    assert!(matches!(err, OxideError::NotImplemented("upsample_f32")));
    assert!(err.to_string().contains("not yet implemented in oxide backend"));
    let err = b
        .crop_bilinear_f32(&t, &[4], 1, 2, 2, &t, &t, &t, &t, cfg())
        .err()
        .expect("crop should be unimplemented");
    assert!(matches!(err, OxideError::NotImplemented(_)));
}
