//! GPU parity for the shared transformer primitives: row-wise LayerNorm and
//! its input gradient, plus the row-softmax backward.
//!
//! These are the Phase 3A prerequisite building blocks (see
//! `tasks/rust/layers.md`): every ported transformer layer - `vit`, `swin`,
//! `tucker_attention`, `mambavision`, `kimi_kda`, `minimax_sparse_attention` and
//! the LAYA encoder/head - normalises a `rows x cols` slab and needs its input
//! gradient, so the op is exercised against the host reference directly rather
//! than through one particular layer.

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, host_ops};
use darknet_cuda_oxide::OxideBackend;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        r as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * w.abs().max(1.0),
            "{name}[{i}]: got {g}, want {w}, tolerance {tolerance}"
        );
    }
}

fn backend() -> OxideBackend {
    OxideBackend::new().expect("backend")
}

#[test]
fn layernorm_forward_matches_the_host_for_many_shapes() {
    let be = backend();
    let mut rng = Rng(0x1a2b_3c4d);
    for &(rows, cols) in &[(1usize, 1usize), (1, 7), (3, 4), (8, 16), (17, 33), (64, 1024)] {
        let x = rng.vec(rows * cols);
        let gamma = rng.vec(cols);
        let beta = rng.vec(cols);
        let xt = be.upload_f32(&[rows, cols], &x).unwrap();
        let gt = be.upload_f32(&[cols], &gamma).unwrap();
        let bt = be.upload_f32(&[cols], &beta).unwrap();
        let out = be
            .layernorm_forward_f32(
                &xt,
                Some(&gt),
                Some(&bt),
                &[rows, cols],
                rows,
                cols,
                1e-5,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::layernorm(&x, rows, cols, Some(&gamma), Some(&beta), 1e-5);
        close(&format!("layernorm forward {rows}x{cols}"), &got, &want, 1e-5);
    }
}

#[test]
fn layernorm_forward_supports_the_bias_free_variant() {
    let be = backend();
    let mut rng = Rng(0x7777_1111);
    let (rows, cols) = (5usize, 12usize);
    let x = rng.vec(rows * cols);
    let gamma = rng.vec(cols);
    let xt = be.upload_f32(&[rows, cols], &x).unwrap();
    let gt = be.upload_f32(&[cols], &gamma).unwrap();
    let out = be
        .layernorm_forward_f32(&xt, Some(&gt), None, &[rows, cols], rows, cols, 1e-5, ElementwiseLaunchConfig::default())
        .unwrap();
    let got = be.download_f32(&out).unwrap();
    let want = host_ops::layernorm(&x, rows, cols, Some(&gamma), None, 1e-5);
    close("layernorm forward bias-free", &got, &want, 1e-5);
    // No gamma at all is the identity scale.
    let out = be
        .layernorm_forward_f32(&xt, None, None, &[rows, cols], rows, cols, 1e-5, ElementwiseLaunchConfig::default())
        .unwrap();
    let got = be.download_f32(&out).unwrap();
    let want = host_ops::layernorm(&x, rows, cols, None, None, 1e-5);
    close("layernorm forward plain", &got, &want, 1e-5);
}

#[test]
fn layernorm_backward_matches_the_host() {
    let be = backend();
    let mut rng = Rng(0x5f5f_0f0f);
    for &(rows, cols) in &[(1usize, 1usize), (2, 5), (9, 8), (32, 64), (7, 1024)] {
        let x = rng.vec(rows * cols);
        let dy = rng.vec(rows * cols);
        let gamma = rng.vec(cols);
        let xt = be.upload_f32(&[rows, cols], &x).unwrap();
        let dyt = be.upload_f32(&[rows, cols], &dy).unwrap();
        let gt = be.upload_f32(&[cols], &gamma).unwrap();
        let out = be
            .layernorm_backward_f32(&dyt, &xt, Some(&gt), &[rows, cols], rows, cols, 1e-5, ElementwiseLaunchConfig::default())
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::layernorm_backward(&dy, &x, rows, cols, Some(&gamma), 1e-5);
        close(&format!("layernorm backward {rows}x{cols}"), &got, &want, 1e-4);
    }
}

#[test]
fn layernorm_backward_without_gamma() {
    let be = backend();
    let mut rng = Rng(0x2468_1357);
    let (rows, cols) = (4usize, 9usize);
    let x = rng.vec(rows * cols);
    let dy = rng.vec(rows * cols);
    let xt = be.upload_f32(&[rows, cols], &x).unwrap();
    let dyt = be.upload_f32(&[rows, cols], &dy).unwrap();
    let out = be
        .layernorm_backward_f32(&dyt, &xt, None, &[rows, cols], rows, cols, 1e-5, ElementwiseLaunchConfig::default())
        .unwrap();
    let got = be.download_f32(&out).unwrap();
    let want = host_ops::layernorm_backward(&dy, &x, rows, cols, None, 1e-5);
    close("layernorm backward no gamma", &got, &want, 1e-4);
}

#[test]
fn layernorm_rejects_a_row_count_that_does_not_match() {
    let be = backend();
    let x = be.upload_f32(&[4, 4], &vec![1.0f32; 16]).unwrap();
    for (rows, cols) in [(0usize, 4usize), (4, 0), (3, 4), (5, 5)] {
        let err = match be.layernorm_forward_f32(
            &x, None, None, &[4, 4], rows, cols, 1e-5, ElementwiseLaunchConfig::default(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("rows={rows} cols={cols} should be rejected"),
        };
        assert!(format!("{err}").contains("rows*cols"), "{err}");
    }
    // A row longer than the kernel's limit is rejected rather than read past.
    let wide = 1025usize;
    let big = be.upload_f32(&[1, wide], &vec![1.0f32; wide]).unwrap();
    let err = match be.layernorm_forward_f32(
        &big, None, None, &[1, wide], 1, wide, 1e-5, ElementwiseLaunchConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("cols={wide} should be rejected"),
    };
    assert!(format!("{err}").contains("exceeds the kernel limit"), "{err}");
}

#[test]
fn the_row_softmax_backward_op_matches_the_host() {
    // The softmax backward already existed; this pins it through the backend so
    // the transformer primitives are covered in one place.
    let be = backend();
    let mut rng = Rng(0x0bad_c0de);
    for &(groups, group) in &[(1usize, 8usize), (4, 16), (3, 5)] {
        let logits = rng.vec(groups * group);
        let dy = rng.vec(groups * group);
        let logits_t = be.upload_f32(&[groups, group], &logits).unwrap();
        let dy_t = be.upload_f32(&[groups, group], &dy).unwrap();
        let y = be
            .softmax_f32(&logits_t, &[groups, group], group, ElementwiseLaunchConfig::default())
            .unwrap();
        let y_host = host_ops::softmax(&logits, group);
        close("softmax forward", &be.download_f32(&y).unwrap(), &y_host, 1e-5);
        let dx = be
            .softmax_backward_f32(&y, &dy_t, &[groups, group], group, ElementwiseLaunchConfig::default())
            .unwrap();
        let got = be.download_f32(&dx).unwrap();
        let want = host_ops::softmax_backward(&y_host, &dy, group);
        close("softmax backward", &got, &want, 1e-5);
    }
}

#[test]
fn the_residual_bias_epilogue_matches_the_host() {
    let be = backend();
    let mut rng = Rng(0x2b2b_1d1d);
    for &(rows, cols) in &[(1usize, 1usize), (3, 4), (8, 16), (17, 33), (4, 1024)] {
        let residual = rng.vec(rows * cols);
        let y = rng.vec(rows * cols);
        let bias = rng.vec(cols);
        let rt = be.upload_f32(&[rows, cols], &residual).unwrap();
        let yt = be.upload_f32(&[rows, cols], &y).unwrap();
        let bt = be.upload_f32(&[cols], &bias).unwrap();
        let out = be
            .residual_bias_f32(&rt, &yt, Some(&bt), &[rows, cols], rows, cols, ElementwiseLaunchConfig::default())
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::residual_bias(&residual, &y, Some(&bias), rows, cols);
        close(&format!("residual_bias {rows}x{cols}"), &got, &want, 1e-6);
    }
}

#[test]
fn the_residual_bias_epilogue_without_a_bias_is_a_plain_add() {
    let be = backend();
    let mut rng = Rng(0x9e9e_0101);
    let (rows, cols) = (6usize, 7usize);
    let residual = rng.vec(rows * cols);
    let y = rng.vec(rows * cols);
    let rt = be.upload_f32(&[rows, cols], &residual).unwrap();
    let yt = be.upload_f32(&[rows, cols], &y).unwrap();
    let out = be
        .residual_bias_f32(&rt, &yt, None, &[rows, cols], rows, cols, ElementwiseLaunchConfig::default())
        .unwrap();
    let got = be.download_f32(&out).unwrap();
    let want: Vec<f32> = residual.iter().zip(&y).map(|(a, b)| a + b).collect();
    close("residual_bias no bias", &got, &want, 1e-6);
}

#[test]
fn the_residual_bias_epilogue_rejects_a_mismatched_row_count() {
    let be = backend();
    let rt = be.upload_f32(&[4, 4], &vec![1.0f32; 16]).unwrap();
    for (rows, cols) in [(0usize, 4usize), (4, 0), (3, 4), (9, 9)] {
        let err = match be.residual_bias_f32(
            &rt, &rt, None, &[4, 4], rows, cols, ElementwiseLaunchConfig::default(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("rows={rows} cols={cols} should be rejected"),
        };
        assert!(format!("{err}").contains("rows*cols"), "{err}");
    }
}
