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

const KINDS: [F32ActivationKind; 19] = [
    F32ActivationKind::Linear,
    F32ActivationKind::Loggy,
    F32ActivationKind::Hardtan,
    F32ActivationKind::Relie,
    F32ActivationKind::Lhtan,
    F32ActivationKind::Plse,
    F32ActivationKind::Stair,
    F32ActivationKind::Logistic,
    F32ActivationKind::Relu,
    F32ActivationKind::Leaky,
    F32ActivationKind::Mish,
    F32ActivationKind::Swish,
    F32ActivationKind::Tanh,
    F32ActivationKind::Relu6,
    F32ActivationKind::Ramp,
    F32ActivationKind::Gelu,
    F32ActivationKind::Elu,
    F32ActivationKind::Selu,
    F32ActivationKind::HardMish,
];

fn cpu_activate(kind: F32ActivationKind, x: f32) -> f32 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        F32ActivationKind::Hardtan => x.clamp(-1.0, 1.0),
        F32ActivationKind::Relie => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        F32ActivationKind::Lhtan => {
            if x < 0.0 {
                0.001 * x
            } else if x > 1.0 {
                0.001 * (x - 1.0) + 1.0
            } else {
                x
            }
        }
        F32ActivationKind::Plse => {
            if x < -4.0 {
                0.01 * (x + 4.0)
            } else if x > 4.0 {
                0.01 * (x - 4.0) + 1.0
            } else {
                0.125 * x + 0.5
            }
        }
        F32ActivationKind::Stair => {
            let n = x.floor() as i32;
            let half_floor = (x / 2.0).floor();
            if n % 2 == 0 {
                half_floor
            } else {
                (x - n as f32) + half_floor
            }
        }
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
        F32ActivationKind::Relu6 => x.clamp(0.0, 6.0),
        F32ActivationKind::Ramp => x * (x > 0.0) as i32 as f32 + 0.1 * x,
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507 * x
            } else {
                1.0507 * 1.6732 * (x.exp() - 1.0)
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                x
            } else if x > -2.0 {
                x * x / 2.0 + x
            } else {
                0.0
            }
        }
    }
}

fn cpu_gradient(kind: F32ActivationKind, x: f32) -> f32 {
    match kind {
        F32ActivationKind::Linear => 1.0,
        F32ActivationKind::Loggy => {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            2.0 * sigmoid * (1.0 - sigmoid)
        }
        F32ActivationKind::Hardtan => (x > -1.0 && x < 1.0) as i32 as f32,
        F32ActivationKind::Relie => {
            if x > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        F32ActivationKind::Lhtan => {
            if x > 0.0 && x < 1.0 {
                1.0
            } else {
                0.001
            }
        }
        F32ActivationKind::Plse => {
            if x.abs() > 4.0 {
                0.01
            } else {
                0.125
            }
        }
        F32ActivationKind::Stair => {
            if (x.floor() as i32) % 2 != 0 {
                1.0
            } else {
                0.0
            }
        }
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
        F32ActivationKind::Relu6 => (x > 0.0 && x < 6.0) as i32 as f32,
        F32ActivationKind::Ramp => (x > 0.0) as i32 as f32 + 0.1,
        F32ActivationKind::Gelu => {
            let t = (0.797885 * x + 0.035677 * x * x * x).tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (0.797885 + 3.0 * 0.035677 * x * x)
        }
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507
            } else {
                1.0507 * 1.6732 * x.exp()
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                1.0
            } else if x > -2.0 {
                x + 1.0
            } else {
                0.0
            }
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
    let mut x = rng.vec(n, -8.0, 8.0);
    x[0] = 25.0;
    x[1] = -25.0;
    x[2] = 0.0;
    x[3] = 6.0; // ReLU6 upper kink: gradient 0 at exactly 6 (C++ `x < 6`)
    x[4] = 5.999;
    x[5] = 6.001;
    x[6] = -2.0; // HardMish lower transition boundary.
    x[7] = -1.0; // Hardtan lower clamp boundary.
    x[8] = 1.0; // Hardtan upper clamp boundary and Lhtan upper endpoint.
    x[9] = 0.0; // Lhtan lower endpoint.
    x[10] = 4.0; // Plse upper transition boundary.
    x[11] = -4.0; // Plse lower transition boundary.
    x[12] = 3.0; // Stair odd interval.
    x[13] = -3.0; // Stair negative odd interval.
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
fn weighted_shortcut_composition_matches_host_reference() {
    let b = backend();
    let batch = 2;
    let channels = 3;
    let channel_stride = 4;
    let per_sample = channels * channel_stride;
    let host_inputs: Vec<Vec<f32>> = (0..3)
        .map(|input| (0..batch * per_sample).map(|i| (i as f32 * 0.17 + input as f32).sin()).collect())
        .collect();
    let inputs: Vec<_> = host_inputs.iter().map(|data| b.upload_f32(&[batch, channels, 1, channel_stride], data).unwrap()).collect();
    let refs: Vec<_> = inputs.iter().collect();
    let weights: Vec<f32> = (0..3 * channels).map(|i| 0.2 + i as f32 * 0.03).collect();
    let mut expected = vec![0.0; batch * per_sample];
    for (input_index, host) in host_inputs.iter().enumerate() {
        for batch_index in 0..batch {
            for channel in 0..channels {
                for spatial in 0..channel_stride {
                    let index = (batch_index * channels + channel) * channel_stride + spatial;
                    expected[index] += host[index] * weights[input_index * channels + channel];
                }
            }
        }
    }
    let got = b.shortcut_weighted_sum_f32(&refs, &weights, channels, channel_stride, cfg()).unwrap();
    assert_close("weighted shortcut", &b.download_f32(&got).unwrap(), &expected);
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

/// Every trait op now has a kernel; malformed shapes must surface as
/// `OxideError::Shape`, never a panic or an out-of-bounds launch.
#[test]
fn invalid_shapes_return_error() {
    let b = backend();
    let t = b.zeros_f32(&[4]).unwrap();
    let err = b
        .upsample_f32(&t, &[1, 1, 3, 3], 1, 2, 2, 2, 1.0, false, cfg())
        .err()
        .expect("upsample output shape is wrong");
    assert!(matches!(err, OxideError::Shape(_)), "{err}");
    let err = b
        .crop_bilinear_f32(&t, &[4], 1, 2, 2, &t, &t, &t, &t, &t, &t, false, cfg())
        .err()
        .expect("crop output shape is not NCHW");
    assert!(matches!(err, OxideError::Shape(_)), "{err}");
    let err = b
        .maxpool_f32(&t, &[1, 1, 2, 2], 1, 2, 2, 0, 1, 1, 0, cfg())
        .err()
        .expect("zero-sized pooling window");
    assert!(matches!(err, OxideError::Shape(_)), "{err}");
    let err = b
        .route_concat_f32(&[&t], &[0], &[3], &[2], &[1, 3, 2, 1], 2, cfg())
        .err()
        .expect("route slice exceeds input channels");
    assert!(matches!(err, OxideError::Shape(_)), "{err}");
}

// ---- Finite-difference gradient checks (f64 CPU forward) ----

fn act64(kind: F32ActivationKind, x: f64) -> f64 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        F32ActivationKind::Hardtan => x.clamp(-1.0, 1.0),
        F32ActivationKind::Relie => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        F32ActivationKind::Lhtan => {
            if x < 0.0 {
                0.001 * x
            } else if x > 1.0 {
                0.001 * (x - 1.0) + 1.0
            } else {
                x
            }
        }
        F32ActivationKind::Plse => {
            if x < -4.0 {
                0.01 * (x + 4.0)
            } else if x > 4.0 {
                0.01 * (x - 4.0) + 1.0
            } else {
                0.125 * x + 0.5
            }
        }
        F32ActivationKind::Stair => {
            let n = x.floor() as i32;
            let half_floor = (x / 2.0).floor();
            if n % 2 == 0 {
                half_floor
            } else {
                (x - n as f64) + half_floor
            }
        }
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
        F32ActivationKind::Relu6 => x.clamp(0.0, 6.0),
        F32ActivationKind::Ramp => x * (x > 0.0) as i32 as f64 + 0.1 * x,
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507 * x
            } else {
                1.0507 * 1.6732 * (x.exp() - 1.0)
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                x
            } else if x > -2.0 {
                x * x / 2.0 + x
            } else {
                0.0
            }
        }
    }
}

#[test]
fn activation_backward_matches_finite_difference() {
    let b = backend();
    let mut rng = Rng(21);
    let n = 513;
    // Stay away from the ReLU / Leaky / Lhtan / PLSE / ReLU6 kinks and Stair
    // discontinuities at 0, ±4, 1, 6, and every integer.
    let x: Vec<f32> = rng
        .vec(n, -4.0, 8.0)
        .into_iter()
        .map(|v| {
            if v.abs() < 0.05
                || (v - 1.0).abs() < 0.05
                || (v.abs() - 4.0).abs() < 0.05
                || (v - 6.0).abs() < 0.05
                || (v - v.round()).abs() < 0.05
            {
                v + 0.2
            } else {
                v
            }
        })
        .collect();
    let delta = rng.vec(n, -2.0, 2.0);
    let xt = b.upload_f32(&[n], &x).unwrap();
    let dt = b.upload_f32(&[n], &delta).unwrap();
    let eps = 1e-5f64;
    for kind in KINDS {
        let got = b
            .download_f32(&b.activation_backward_f32(&xt, &dt, kind, cfg()).unwrap())
            .unwrap();
        for i in 0..n {
            let xi = x[i] as f64;
            let fd = (act64(kind, xi + eps) - act64(kind, xi - eps)) / (2.0 * eps) * delta[i] as f64;
            let err = (got[i] as f64 - fd).abs();
            assert!(
                err <= 1e-3 + 1e-3 * fd.abs(),
                "{kind:?} dgrad[{i}] x={xi}: got {}, fd {fd}, err {err}",
                got[i]
            );
        }
    }
}

/// darknet-core's CPU gradient (`activation_gradient_from_output`) derives
/// Logistic / Relu / Leaky / Tanh / Lhtan from the activation OUTPUT, the GPU trait
/// takes the activation INPUT (the runtime passes the biased pre-activation
/// tensor).  The two are the same function for these kinds: check it.
#[test]
fn activation_backward_input_form_equals_core_output_form() {
    let b = backend();
    let mut rng = Rng(22);
    let n = 777;
    let mut x = rng.vec(n, -5.0, 5.0);
    x[0] = 0.0;
    x[1] = -1.0;
    x[2] = 1.0;
    let delta = rng.vec(n, -1.0, 1.0);
    let xt = b.upload_f32(&[n], &x).unwrap();
    let dt = b.upload_f32(&[n], &delta).unwrap();
    for kind in [
        F32ActivationKind::Linear,
        F32ActivationKind::Loggy,
        F32ActivationKind::Hardtan,
        F32ActivationKind::Relie,
        F32ActivationKind::Lhtan,
        F32ActivationKind::Plse,
        F32ActivationKind::Logistic,
        F32ActivationKind::Relu,
        F32ActivationKind::Leaky,
        F32ActivationKind::Tanh,
        F32ActivationKind::Relu6,
        F32ActivationKind::Ramp,
    ] {
        let got = b
            .download_f32(&b.activation_backward_f32(&xt, &dt, kind, cfg()).unwrap())
            .unwrap();
        let want: Vec<f32> = x
            .iter()
            .zip(&delta)
            .map(|(&xi, &d)| {
                let y = cpu_activate(kind, xi);
                // Formulas copied from darknet-core `activation_gradient_from_output`.
                let g = match kind {
                    F32ActivationKind::Linear => 1.0,
                    F32ActivationKind::Loggy => 0.5 * (1.0 - y * y),
                    F32ActivationKind::Hardtan => (y > -1.0 && y < 1.0) as i32 as f32,
                    F32ActivationKind::Relie => {
                        if y > 0.0 {
                            1.0
                        } else {
                            0.01
                        }
                    }
                    F32ActivationKind::Lhtan => {
                        if y > 0.0 && y < 1.0 {
                            1.0
                        } else {
                            0.001
                        }
                    }
                    F32ActivationKind::Plse => {
                        if y < 0.0 || y > 1.0 {
                            0.01
                        } else {
                            0.125
                        }
                    }
                    F32ActivationKind::Logistic => y * (1.0 - y),
                    F32ActivationKind::Relu => (y > 0.0) as i32 as f32,
                    F32ActivationKind::Leaky => {
                        if y > 0.0 {
                            1.0
                        } else {
                            0.1
                        }
                    }
                    F32ActivationKind::Tanh => 1.0 - y * y,
                    F32ActivationKind::Relu6 => (y > 0.0 && y < 6.0) as i32 as f32,
                    F32ActivationKind::Ramp => (y > 0.0) as i32 as f32 + 0.1,
                    _ => unreachable!(),
                };
                g * d
            })
            .collect();
        assert_close(&format!("{kind:?} input-vs-output form"), &got, &want);
    }
}

#[test]
fn bias_ops_match_finite_difference() {
    let b = backend();
    let mut rng = Rng(23);
    let (batch, channels, stride) = (3usize, 4usize, 19usize);
    let n = batch * channels * stride;
    let x = rng.vec(n, -1.0, 1.0);
    let bias = rng.vec(channels, -1.0, 1.0);
    let r = rng.vec(n, -1.0, 1.0); // upstream gradient
    let xt = b.upload_f32(&[n], &x).unwrap();
    let rt = b.upload_f32(&[n], &r).unwrap();

    // L(bias) = sum_i r_i * (x_i + bias[c(i)])
    let loss = |bv: &[f64]| -> f64 {
        (0..n)
            .map(|i| r[i] as f64 * (x[i] as f64 + bv[(i / stride) % channels]))
            .sum()
    };
    let bias64: Vec<f64> = bias.iter().map(|&v| v as f64).collect();
    let eps = 1e-3f64;
    let fd: Vec<f64> = (0..channels)
        .map(|c| {
            let (mut p, mut m) = (bias64.clone(), bias64.clone());
            p[c] += eps;
            m[c] -= eps;
            (loss(&p) - loss(&m)) / (2.0 * eps)
        })
        .collect();
    let got = b
        .download_f32(&b.bias_backward_f32(&rt, channels, stride, cfg()).unwrap())
        .unwrap();
    for c in 0..channels {
        assert!(
            (got[c] as f64 - fd[c]).abs() <= 1e-3 + 1e-3 * fd[c].abs(),
            "bias_backward[{c}]: got {}, fd {}",
            got[c],
            fd[c]
        );
    }

    // d bias_add / d x is the identity: the upstream gradient passes through.
    let bt = b.upload_f32(&[channels], &bias).unwrap();
    let y = b.download_f32(&b.bias_add_f32(&xt, &bt, channels, stride, cfg()).unwrap()).unwrap();
    let fd_x: Vec<f64> = (0..n)
        .step_by(37)
        .map(|i| {
            let f = |d: f64| (x[i] as f64 + d + bias[(i / stride) % channels] as f64) * r[i] as f64;
            (f(eps) - f(-eps)) / (2.0 * eps)
        })
        .collect();
    for (j, i) in (0..n).step_by(37).enumerate() {
        assert!((fd_x[j] - r[i] as f64).abs() <= 1e-3, "bias_add dx[{i}]");
        assert!((y[i] - (x[i] + bias[(i / stride) % channels])).abs() <= TOL);
    }
}
