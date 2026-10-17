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

/// Elementwise comparison with a relative tolerance.
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
    for &(rows, cols) in &[
        (1usize, 1usize),
        (1, 7),
        (3, 4),
        (8, 16),
        (17, 33),
        (64, 1024),
    ] {
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
        close(
            &format!("layernorm forward {rows}x{cols}"),
            &got,
            &want,
            1e-5,
        );
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
        .layernorm_forward_f32(
            &xt,
            Some(&gt),
            None,
            &[rows, cols],
            rows,
            cols,
            1e-5,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let got = be.download_f32(&out).unwrap();
    let want = host_ops::layernorm(&x, rows, cols, Some(&gamma), None, 1e-5);
    close("layernorm forward bias-free", &got, &want, 1e-5);
    // No gamma at all is the identity scale.
    let out = be
        .layernorm_forward_f32(
            &xt,
            None,
            None,
            &[rows, cols],
            rows,
            cols,
            1e-5,
            ElementwiseLaunchConfig::default(),
        )
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
            .layernorm_backward_f32(
                &dyt,
                &xt,
                Some(&gt),
                &[rows, cols],
                rows,
                cols,
                1e-5,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::layernorm_backward(&dy, &x, rows, cols, Some(&gamma), 1e-5);
        close(
            &format!("layernorm backward {rows}x{cols}"),
            &got,
            &want,
            1e-4,
        );
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
        .layernorm_backward_f32(
            &dyt,
            &xt,
            None,
            &[rows, cols],
            rows,
            cols,
            1e-5,
            ElementwiseLaunchConfig::default(),
        )
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
            &x,
            None,
            None,
            &[4, 4],
            rows,
            cols,
            1e-5,
            ElementwiseLaunchConfig::default(),
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
        &big,
        None,
        None,
        &[1, wide],
        1,
        wide,
        1e-5,
        ElementwiseLaunchConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("cols={wide} should be rejected"),
    };
    assert!(
        format!("{err}").contains("exceeds the kernel limit"),
        "{err}"
    );
}

#[test]
fn flash_attention_forward_matches_the_online_host_reference() {
    let be = backend();
    let mut rng = Rng(0x6a09_e667);
    for &(batch, tokens, channels, heads) in &[
        (1usize, 1usize, 4usize, 2usize),
        (2, 3, 4, 2),
        (1, 5, 8, 1),
        (1, 3, 128, 1),
    ] {
        let qkv = rng.vec(batch * tokens * 3 * channels);
        let qkv_tensor = be.upload_f32(&[batch, tokens, 3 * channels], &qkv).unwrap();
        let (output, lse) = be
            .flash_attention_forward_f32(
                &qkv_tensor,
                &[batch, tokens, channels],
                &[batch, heads, tokens],
                batch,
                tokens,
                channels,
                heads,
            )
            .unwrap();
        let (want_output, want_lse) =
            host_ops::flash_attention_forward(&qkv, batch, tokens, channels, heads);
        close(
            &format!("flash output {batch}x{tokens}x{channels}/{heads}"),
            &be.download_f32(&output).unwrap(),
            &want_output,
            2e-5,
        );
        close(
            &format!("flash LSE {batch}x{heads}x{tokens}"),
            &be.download_f32(&lse).unwrap(),
            &want_lse,
            2e-5,
        );
    }

    // Q/K sanitization and magnitude/score clamps are part of the op contract.
    let qkv = [
        200.0,
        -300.0,
        1000.0,
        -1000.0,
        1.0,
        2.0,
        f32::INFINITY,
        f32::NAN,
        -200.0,
        300.0,
        3.0,
        4.0,
    ];
    let input = be.upload_f32(&[1, 2, 6], &qkv).unwrap();
    let (output, lse) = be
        .flash_attention_forward_f32(&input, &[1, 2, 2], &[1, 1, 2], 1, 2, 2, 1)
        .unwrap();
    let (want_output, want_lse) = host_ops::flash_attention_forward(&qkv, 1, 2, 2, 1);
    close(
        "flash clamped output",
        &be.download_f32(&output).unwrap(),
        &want_output,
        2e-5,
    );
    close(
        "flash clamped LSE",
        &be.download_f32(&lse).unwrap(),
        &want_lse,
        2e-5,
    );
}

#[test]
fn flash_attention_backward_matches_host_and_finite_differences() {
    let be = backend();
    let mut rng = Rng(0x510e_527f);
    let (batch, tokens, channels, heads) = (1usize, 3usize, 4usize, 2usize);
    let qkv = rng.vec(batch * tokens * 3 * channels);
    let (context, lse) = host_ops::flash_attention_forward(&qkv, batch, tokens, channels, heads);
    let output_grad = rng.vec(batch * tokens * channels);
    let qkv_tensor = be.upload_f32(&[batch, tokens, 3 * channels], &qkv).unwrap();
    let context_tensor = be.upload_f32(&[batch, tokens, channels], &context).unwrap();
    let grad_tensor = be
        .upload_f32(&[batch, tokens, channels], &output_grad)
        .unwrap();
    let lse_tensor = be.upload_f32(&[batch, heads, tokens], &lse).unwrap();
    let dqkv = be
        .flash_attention_backward_f32(
            &qkv_tensor,
            &context_tensor,
            &grad_tensor,
            &lse_tensor,
            &[batch, tokens, 3 * channels],
            batch,
            tokens,
            channels,
            heads,
        )
        .unwrap();
    assert!(
        be.flash_attention_backward_f32(
            &qkv_tensor,
            &context_tensor,
            &grad_tensor,
            &lse_tensor,
            &[batch, tokens, 3 * channels - 1],
            batch,
            tokens,
            channels,
            heads,
        )
        .is_err()
    );
    let got = be.download_f32(&dqkv).unwrap();
    let want = host_ops::flash_attention_backward(
        &qkv,
        &context,
        &output_grad,
        &lse,
        batch,
        tokens,
        channels,
        heads,
    );
    close("flash backward", &got, &want, 3e-5);

    let objective = |values: &[f32]| {
        let (out, _) = host_ops::flash_attention_forward(values, batch, tokens, channels, heads);
        out.iter()
            .zip(&output_grad)
            .map(|(a, b)| a * b)
            .sum::<f32>()
    };
    for index in [0usize, 4, 8, 13, 18, 32] {
        let eps = 1e-3;
        let mut plus = qkv.clone();
        let mut minus = qkv.clone();
        plus[index] += eps;
        minus[index] -= eps;
        let numeric = (objective(&plus) - objective(&minus)) / (2.0 * eps);
        let slack = 2e-3 * want[index].abs().max(0.1) + 2e-4;
        assert!(
            (numeric - want[index]).abs() <= slack,
            "dQKV[{index}] analytic={} numeric={numeric} slack={slack}",
            want[index]
        );
    }

    // The raw-Q clamp masks the derivative while the clipped value still
    // participates in the score; K/V gradients remain independently defined.
    let clamped_qkv = [
        200.0, -200.0, 20.0, -30.0, 1.0, 2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.0,
    ];
    let (context, lse) = host_ops::flash_attention_forward(&clamped_qkv, 1, 2, 2, 1);
    let grad = [0.3, -0.2, 0.4, 0.5];
    let input = be.upload_f32(&[1, 2, 6], &clamped_qkv).unwrap();
    let context_t = be.upload_f32(&[1, 2, 2], &context).unwrap();
    let grad_t = be.upload_f32(&[1, 2, 2], &grad).unwrap();
    let lse_t = be.upload_f32(&[1, 1, 2], &lse).unwrap();
    let got = be
        .flash_attention_backward_f32(&input, &context_t, &grad_t, &lse_t, &[1, 2, 6], 1, 2, 2, 1)
        .unwrap();
    let want = host_ops::flash_attention_backward(&clamped_qkv, &context, &grad, &lse, 1, 2, 2, 1);
    close(
        "flash backward clamp masks",
        &be.download_f32(&got).unwrap(),
        &want,
        3e-5,
    );
    assert_eq!(be.download_f32(&got).unwrap()[..2], [0.0, 0.0]);
}

#[test]
fn flash_attention_rejects_invalid_shapes_and_unsupported_head_dim() {
    let be = backend();
    let qkv = be.upload_f32(&[1, 1, 6], &[0.0; 6]).unwrap();
    let err = match be.flash_attention_forward_f32(&qkv, &[1, 1, 1], &[1, 1, 1], 1, 1, 2, 1) {
        Err(err) => err,
        Ok(_) => panic!("mismatched output shape should be rejected"),
    };
    assert!(format!("{err}").contains("output shape"), "{err}");

    let channels = 129;
    let input = be
        .upload_f32(&[1, 1, 3 * channels], &vec![0.0; 3 * channels])
        .unwrap();
    let err = match be.flash_attention_forward_f32(
        &input,
        &[1, 1, channels],
        &[1, 1, 1],
        1,
        1,
        channels,
        1,
    ) {
        Err(err) => err,
        Ok(_) => panic!("head dimension above the kernel limit should be rejected"),
    };
    assert!(format!("{err}").contains("kernel limit"), "{err}");

    let context = be
        .upload_f32(&[1, 1, channels], &vec![0.0; channels])
        .unwrap();
    let grad = be
        .upload_f32(&[1, 1, channels], &vec![0.0; channels])
        .unwrap();
    let lse = be.upload_f32(&[1, 1, 1], &[0.0]).unwrap();
    let err = match be.flash_attention_backward_f32(
        &input,
        &context,
        &grad,
        &lse,
        &[1, 1, 3 * channels],
        1,
        1,
        channels,
        1,
    ) {
        Err(err) => err,
        Ok(_) => panic!("backward head dimension above the kernel limit should be rejected"),
    };
    assert!(format!("{err}").contains("kernel limit"), "{err}");
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
            .softmax_f32(
                &logits_t,
                &[groups, group],
                group,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let y_host = host_ops::softmax(&logits, group);
        close(
            "softmax forward",
            &be.download_f32(&y).unwrap(),
            &y_host,
            1e-5,
        );
        let dx = be
            .softmax_backward_f32(
                &y,
                &dy_t,
                &[groups, group],
                group,
                ElementwiseLaunchConfig::default(),
            )
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
            .residual_bias_f32(
                &rt,
                &yt,
                Some(&bt),
                &[rows, cols],
                rows,
                cols,
                ElementwiseLaunchConfig::default(),
            )
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
        .residual_bias_f32(
            &rt,
            &yt,
            None,
            &[rows, cols],
            rows,
            cols,
            ElementwiseLaunchConfig::default(),
        )
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
            &rt,
            &rt,
            None,
            &[4, 4],
            rows,
            cols,
            ElementwiseLaunchConfig::default(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("rows={rows} cols={cols} should be rejected"),
        };
        assert!(format!("{err}").contains("rows*cols"), "{err}");
    }
}

#[test]
fn the_batched_gemm_wrapper_matches_the_host() {
    // This is the piece every transformer projection needs: `rows` is folded
    // into the GEMM's `m`, so one call covers every token of a sequence.
    let be = backend();
    let mut rng = Rng(0x3c3c_3c3c);
    for &(rows, k, n) in &[
        (1usize, 1usize, 1usize),
        (3, 4, 5),
        (8, 16, 16),
        (7, 33, 12),
        (2, 1024, 64),
    ] {
        let x = rng.vec(rows * k);
        let w = rng.vec(n * k); // [out_features, in_features], torch nn.Linear
        let bias = rng.vec(n);
        let xt = be.upload_f32(&[rows, k], &x).unwrap();
        let wt = be.upload_f32(&[n, k], &w).unwrap();
        let bt = be.upload_f32(&[n], &bias).unwrap();
        let out = be
            .linear_f32(
                &xt,
                &wt,
                Some(&bt),
                &[rows, n],
                rows,
                k,
                n,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::linear(&x, rows, k, n, &w, Some(&bias));
        // The GEMM accumulates a k-long dot product in its own order, so the
        // error scales with the row norm of the terms rather than with |out|.
        // Bound each output by the sum of |x_k * w_nk|, as the existing GEMM
        // tests do, so a long k does not trip a fixed tolerance.
        for r in 0..rows {
            for j in 0..n {
                let norm: f32 = (0..k)
                    .map(|t| x[r * k + t].abs() * w[j * k + t].abs())
                    .sum::<f32>()
                    + bias[j].abs();
                let err = (got[r * n + j] - want[r * n + j]).abs();
                assert!(
                    err <= 1e-5 * norm.max(1e-3),
                    "linear {rows}x{k}x{n}[{r},{j}]: got {}, want {}, row norm {norm}, err {err}",
                    got[r * n + j],
                    want[r * n + j]
                );
            }
        }
        // No bias is the bias-free projection variant.
        let out = be
            .linear_f32(
                &xt,
                &wt,
                None,
                &[rows, n],
                rows,
                k,
                n,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = host_ops::linear(&x, rows, k, n, &w, None);
        close(&format!("linear no bias {rows}x{k}x{n}"), &got, &want, 1e-4);
    }
}

#[test]
fn the_batched_gemm_wrapper_rejects_mismatched_shapes() {
    let be = backend();
    let xt = be.upload_f32(&[4, 3], &vec![1.0f32; 12]).unwrap();
    let wt = be.upload_f32(&[5, 3], &vec![1.0f32; 15]).unwrap();
    // rows * out_features must match the output shape.
    let err = match be.linear_f32(
        &xt,
        &wt,
        None,
        &[4, 4],
        4,
        3,
        5,
        ElementwiseLaunchConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("a mismatched output shape should be rejected"),
    };
    assert!(
        format!("{err}").contains("must hold rows*out_features"),
        "{err}"
    );
    // A zero extent is rejected rather than launching an empty grid.
    let err = match be.linear_f32(
        &xt,
        &wt,
        None,
        &[0, 0],
        0,
        3,
        5,
        ElementwiseLaunchConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("rows = 0 should be rejected"),
    };
    assert!(format!("{err}").contains("must be > 0"), "{err}");
}
