//! Native detached LAYA act-head feature construction, not the full MLP.
//! Hand-derived from source, no compiled-C++ golden.
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn reference(
    head: &[f32],
    logits: &[f32],
    marker: &[f32],
    b: usize,
    h: usize,
    k: usize,
    cb: usize,
    th: usize,
    tm: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; b * (h + 4)];
    for n in 0..b {
        darknet_core::laya_act_head::features_for_sample(
            head,
            logits,
            marker,
            n,
            h,
            k,
            cb,
            th,
            tm,
            &mut out[n * (h + 4)..(n + 1) * (h + 4)],
        );
    }
    out
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            if w.is_nan() {
                g.is_nan()
            } else {
                (g - w).abs() < 1e-5 * (1.0 + w.abs())
            },
            "{i}: {g} vs {w}"
        );
    }
}

#[test]
fn laya_act_features_native_matches_source_hand_math_and_cpu_batches() {
    let backend = OxideBackend::new().unwrap();
    let (b, h, k, cb, th, tm) = (3, 3, 4, 10, 5, 3);
    let hs = [b, h, 1, th];
    let ss = [b, 1, 1, k];
    let ms = [b, cb, 1, tm];
    let mut head = vec![f32::NAN; b * h * th];
    for n in 0..b {
        for c in 0..h {
            head[(n * h + c) * th] = (100 * n + 10 * c) as f32;
        }
    }
    let logits = [
        3.0_f32.ln(),
        0.0,
        99.0,
        99.0,
        -99.0,
        0.0,
        2.0,
        9.0,
        -10001.0,
        99.0,
        99.0,
        99.0,
    ];
    let masks = [
        [1.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.5001, 0.0, 0.0, 0.0],
    ];
    let mut marker = vec![f32::NAN; b * cb * tm];
    for n in 0..b {
        for j in 0..k {
            marker[(n * cb + 2 * j + 1) * tm] = masks[n][j];
        }
    }
    let want = reference(&head, &logits, &marker, b, h, k, cb, th, tm);
    let layout = darknet_cuda::host_ops::laya_act::Layout::new(&hs, &ss, &ms).unwrap();
    close(&layout.forward(&head, &logits, &marker), &want);
    // Row zero: two active options => (.75,.25,0,0).
    close(
        &want[..7],
        &[
            0.0,
            10.0,
            20.0,
            0.75,
            0.5,
            (-0.75 * 0.75_f32.ln() - 0.25 * 0.25_f32.ln()) / 2.0_f32.ln(),
            2.0 / 255.0,
        ],
    );
    // All masked is uniform over ALL K finite -1e4 slots, normalized by log(2).
    close(
        &want[7..14],
        &[100.0, 110.0, 120.0, 0.25, 0.0, 2.0, 2.0 / 255.0],
    );
    // Masked logits are replaced, not removed: they can outrank a valid <-1e4.
    let p = 1.0 / (3.0 + (-1.0_f32).exp());
    assert!((want[17] - p).abs() < 1e-6);
    assert_eq!(want[18], 0.0); // tied masked top-two
    let hg = backend.upload_f32(&hs, &head).unwrap();
    let sg = backend.upload_f32(&ss, &logits).unwrap();
    let mg = backend.upload_f32(&ms, &marker).unwrap();
    for tile_len in [32, 128, 256] {
        let out = backend
            .laya_act_features_f32(
                &hg,
                &sg,
                &mg,
                &hs,
                &ss,
                &ms,
                ElementwiseLaunchConfig { tile_len },
            )
            .unwrap();
        assert_eq!(out.shape(), [3, 7, 1, 1]);
        close(&backend.download_f32(&out).unwrap(), &want);
    }
}

#[test]
fn laya_act_features_native_preserves_rust_nonfinite_policy_and_large_k() {
    let backend = OxideBackend::new().unwrap();
    for k in [4, 257] {
        let (b, h, th, tm) = (3, 2, 1, 2);
        let cb = 2 * k;
        let hs = [b, h, 1, th];
        let ss = [b, 1, 1, k];
        let ms = [b, cb, 1, tm];
        let head = vec![1.0; b * h * th];
        let mut logits = vec![0.0; b * k];
        if k == 4 {
            logits[0] = f32::NAN;
            logits[k] = f32::INFINITY;
            logits[2 * k] = f32::NEG_INFINITY;
        }
        let marker = vec![1.0; b * cb * tm];
        let want = reference(&head, &logits, &marker, b, h, k, cb, th, tm);
        let layout = darknet_cuda::host_ops::laya_act::Layout::new(&hs, &ss, &ms).unwrap();
        close(&layout.forward(&head, &logits, &marker), &want);
        if k == 257 {
            assert!(want[h + 3] > 1.0);
        } // count feature is NOT clamped
        let hg = backend.upload_f32(&hs, &head).unwrap();
        let sg = backend.upload_f32(&ss, &logits).unwrap();
        let mg = backend.upload_f32(&ms, &marker).unwrap();
        let out = backend
            .laya_act_features_f32(
                &hg,
                &sg,
                &mg,
                &hs,
                &ss,
                &ms,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        close(&backend.download_f32(&out).unwrap(), &want);
    }
}

#[test]
fn laya_act_features_native_rejects_bad_layout_and_truncated_storage() {
    let backend = OxideBackend::new().unwrap();
    let config = ElementwiseLaunchConfig::default();
    let hs = [2, 3, 1, 5];
    let ss = [2, 1, 1, 4];
    let ms = [2, 10, 1, 3];
    let h = backend.upload_f32(&hs, &vec![0.0; 30]).unwrap();
    let s = backend.upload_f32(&ss, &vec![0.0; 8]).unwrap();
    let m = backend.upload_f32(&ms, &vec![0.0; 60]).unwrap();
    for (hsh, ssh, msh) in [
        (vec![2, 3, 2, 5], ss.to_vec(), ms.to_vec()),
        (hs.to_vec(), vec![2, 2, 1, 4], ms.to_vec()),
        (hs.to_vec(), vec![2, 1, 1, 1], ms.to_vec()),
        (hs.to_vec(), ss.to_vec(), vec![2, 7, 1, 3]),
        (hs.to_vec(), ss.to_vec(), vec![1, 10, 1, 3]),
        (vec![0, 3, 1, 5], ss.to_vec(), ms.to_vec()),
        (
            vec![usize::MAX, 3, 1, 5],
            vec![usize::MAX, 1, 1, 4],
            vec![usize::MAX, 10, 1, 3],
        ),
        (vec![2, 3, 5], ss.to_vec(), ms.to_vec()),
    ] {
        assert!(
            backend
                .laya_act_features_f32(&h, &s, &m, &hsh, &ssh, &msh, config)
                .is_err()
        );
    }
    let short = backend.upload_f32(&[1], &[0.0]).unwrap();
    assert!(
        backend
            .laya_act_features_f32(&short, &s, &m, &hs, &ss, &ms, config)
            .is_err()
    );
    assert!(
        backend
            .laya_act_features_f32(&h, &short, &m, &hs, &ss, &ms, config)
            .is_err()
    );
    assert!(
        backend
            .laya_act_features_f32(&h, &s, &short, &hs, &ss, &ms, config)
            .is_err()
    );
}
