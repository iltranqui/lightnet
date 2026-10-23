use lightnet_lm::attention::{causal_gqa, causal_gqa_backward};
use lightnet_lm::ops::{
    embedding, embedding_backward, rms_norm, rms_norm_backward, rope_in_place, sparse_ce, swiglu,
    swiglu_backward,
};
use lightnet_lm::{DecoderConfig, TokenBatch};

fn cfg() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 64,
        context: 8,
        hidden: 8,
        intermediate: 16,
        layers: 1,
        q_heads: 2,
        kv_heads: 1,
        rope_theta: 10_000.0,
        rms_eps: 1e-5,
        dropout: 0.0,
        tie_embeddings: true,
    }
}

fn values(count: usize, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index as f32 * 0.37 + offset).sin())
        .collect()
}

fn check_grad<F>(name: &str, point: &[f32], analytic: &[f32], mut loss_at: F)
where
    F: FnMut(&[f32]) -> f32,
{
    let step = 1e-3f32;
    for index in 0..point.len() {
        let mut plus = point.to_vec();
        let mut minus = point.to_vec();
        plus[index] += step;
        minus[index] -= step;
        let numeric = (loss_at(&plus) - loss_at(&minus)) / (2.0 * step);
        let analytic = analytic[index];
        let tolerance = 2e-3 + 2e-2 * numeric.abs();
        assert!(
            (analytic - numeric).abs() <= tolerance,
            "{name}[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

#[test]
fn lm_ops_rms_norm_matches_finite_differences() {
    let rows = 2;
    let x = vec![0.5, -1.0, 2.0, 0.25, -0.75, 0.1, 1.5, -2.0];
    let gamma = vec![1.0, 0.8, 1.2, 0.5];
    let weights = values(x.len(), 0.3);
    let eps = 1e-5;

    let (dx, dgamma) = rms_norm_backward(&x, &gamma, &weights, rows, eps).unwrap();
    check_grad("rms_norm dx", &x, &dx, |point| {
        let out = rms_norm(point, &gamma, rows, eps).unwrap();
        dot(&out, &weights)
    });
    check_grad("rms_norm dgamma", &gamma, &dgamma, |point| {
        let out = rms_norm(&x, point, rows, eps).unwrap();
        dot(&out, &weights)
    });

    let analytic = rms_norm(&x, &gamma, rows, eps).unwrap();
    assert_eq!(analytic.len(), x.len());
}

#[test]
fn lm_ops_embedding_accumulates_duplicate_ids() {
    let vocab = 8;
    let hidden = 3;
    let table = values(vocab * hidden, 0.1);
    let ids = [5u32, 5, 7];
    let out = embedding(&table, &ids, hidden).unwrap();
    assert_eq!(&out[0..3], &table[15..18]);
    assert_eq!(&out[3..6], &table[15..18]);
    assert_eq!(&out[6..9], &table[21..24]);

    let dy = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let dtable = embedding_backward(&ids, &dy, vocab, hidden).unwrap();
    for index in 0..vocab * hidden {
        let expected = if (15..18).contains(&index) {
            dy[index - 15] + dy[index - 15 + 3]
        } else if (21..24).contains(&index) {
            dy[index - 21 + 6]
        } else {
            0.0
        };
        assert!((dtable[index] - expected).abs() < 1e-6);
    }

    assert!(embedding(&table, &[8], hidden).is_err());
    assert!(embedding(&table, &[0], 5).is_err());
    assert!(embedding_backward(&ids, &dy[..8], vocab, hidden).is_err());
}

#[test]
fn lm_ops_rope_inverse_restores_input_and_preserves_norms() {
    let rows = 2;
    let heads = 2;
    let head_dim = 4;
    let positions = [0u32, 5];
    let original = values(rows * heads * head_dim, 0.9);
    let mut x = original.clone();
    rope_in_place(&mut x, &positions, heads, head_dim, 10_000.0, false).unwrap();
    assert!(
        (x[0] - original[0]).abs() < 1e-6,
        "position zero must not rotate"
    );
    let later = heads * head_dim;
    assert!(
        (x[later] - original[later]).abs() > 1e-4,
        "position five must rotate"
    );

    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            let before = dot(&original[base..base + 2], &original[base..base + 2])
                + dot(&original[base + 2..base + 4], &original[base + 2..base + 4]);
            let after = dot(&x[base..base + 2], &x[base..base + 2])
                + dot(&x[base + 2..base + 4], &x[base + 2..base + 4]);
            assert!((before - after).abs() < 1e-5);
        }
    }

    rope_in_place(&mut x, &positions, heads, head_dim, 10_000.0, true).unwrap();
    for (left, right) in x.iter().zip(&original) {
        assert!((left - right).abs() < 2e-5);
    }

    assert!(rope_in_place(&mut x, &positions, heads, 3, 10_000.0, false).is_err());
    assert!(rope_in_place(&mut x, &positions, heads, head_dim, 0.0, false).is_err());
    assert!(rope_in_place(&mut x, &[0], heads, head_dim, 10_000.0, false).is_err());
}

#[test]
fn lm_ops_causal_gqa_masks_future_and_other_segments() {
    let cfg = cfg();
    let dims = cfg.q_heads * cfg.head_dim();
    let kv_dims = cfg.kv_heads * cfg.head_dim();
    let batch = TokenBatch {
        batch: 1,
        time: 4,
        input_ids: vec![0, 1, 2, 3],
        targets: vec![-100; 4],
        positions: vec![0, 1, 2, 3],
        segments: vec![0, 0, 1, 1],
        valid: vec![true; 4],
    };
    let q = values(4 * dims, 0.2);
    let k = values(4 * kv_dims, 0.4);
    let v = values(4 * kv_dims, 0.6);
    let baseline = causal_gqa(&q, &k, &v, &batch, &cfg).unwrap();

    let mut changed_q = q.clone();
    let mut changed_k = k.clone();
    let mut changed_v = v.clone();
    for row in [1usize, 2, 3] {
        for channel in 0..dims {
            changed_q[row * dims + channel] += 10.0;
        }
    }
    for row in [1usize, 2, 3] {
        for channel in 0..kv_dims {
            changed_k[row * kv_dims + channel] -= 7.0;
            changed_v[row * kv_dims + channel] += 5.0;
        }
    }
    let changed = causal_gqa(&changed_q, &changed_k, &changed_v, &batch, &cfg).unwrap();
    for index in 0..dims {
        assert!((baseline[index] - changed[index]).abs() < 1e-6);
    }

    for head in 0..cfg.q_heads {
        let base = head * cfg.head_dim();
        for channel in 0..cfg.head_dim() {
            assert!((baseline[base + channel] - v[channel]).abs() < 1e-6);
        }
    }

    let mut later = v.clone();
    for row in [2usize, 3] {
        for channel in 0..kv_dims {
            later[row * kv_dims + channel] += 3.0;
        }
    }
    let with_later = causal_gqa(&q, &k, &later, &batch, &cfg).unwrap();
    for index in 0..2 * dims {
        assert!((baseline[index] - with_later[index]).abs() < 1e-6);
    }
}

#[test]
fn lm_ops_causal_gqa_matches_finite_differences() {
    let cfg = cfg();
    let dims = cfg.q_heads * cfg.head_dim();
    let kv_dims = cfg.kv_heads * cfg.head_dim();
    let batch = TokenBatch {
        batch: 1,
        time: 3,
        input_ids: vec![0, 1, 2],
        targets: vec![-100; 3],
        positions: vec![0, 1, 2],
        segments: vec![0, 0, 0],
        valid: vec![true; 3],
    };
    let q = values(3 * dims, 0.2);
    let k = values(3 * kv_dims, 0.5);
    let v = values(3 * kv_dims, 0.8);
    let weights = values(3 * dims, 1.1);

    let (dq, dk, dv) = causal_gqa_backward(&q, &k, &v, &weights, &batch, &cfg).unwrap();
    check_grad("gqa dq", &q, &dq, |point| {
        let out = causal_gqa(point, &k, &v, &batch, &cfg).unwrap();
        dot(&out, &weights)
    });
    check_grad("gqa dk", &k, &dk, |point| {
        let out = causal_gqa(&q, point, &v, &batch, &cfg).unwrap();
        dot(&out, &weights)
    });
    check_grad("gqa dv", &v, &dv, |point| {
        let out = causal_gqa(&q, &k, point, &batch, &cfg).unwrap();
        dot(&out, &weights)
    });
}

#[test]
fn lm_ops_swiglu_matches_finite_differences() {
    let gate = values(6, 0.25);
    let up = values(6, 0.75);
    let dy = values(6, 1.3);
    let (dgate, dup) = swiglu_backward(&gate, &up, &dy).unwrap();
    check_grad("swiglu dgate", &gate, &dgate, |point| {
        dot(&swiglu(point, &up).unwrap(), &dy)
    });
    check_grad("swiglu dup", &up, &dup, |point| {
        dot(&swiglu(&gate, point).unwrap(), &dy)
    });

    let out = swiglu(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
    assert!((out[0]).abs() < 1e-6);
    assert!((out[1] - 3.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    assert!(swiglu(&[1.0], &[1.0, 2.0]).is_err());
}

#[test]
fn lm_ops_sparse_ce_returns_descent_direction() {
    let (summary, seeds) = sparse_ce(&[0.0, 0.0], &[1], 2).unwrap();
    assert_eq!(summary.active_tokens, 1);
    assert!((summary.sum - std::f64::consts::LN_2).abs() < 1e-6);
    assert!((seeds[0] + 0.5).abs() < 1e-6);
    assert!((seeds[1] - 0.5).abs() < 1e-6);

    let logits = values(6, 0.4);
    let targets = [1i32, -100];
    let (summary, seeds) = sparse_ce(&logits, &targets, 3).unwrap();
    assert_eq!(summary.active_tokens, 1);
    assert_eq!(&seeds[3..], &[0.0, 0.0, 0.0]);

    for index in 0..3 {
        let mut plus = logits.clone();
        let mut minus = logits.clone();
        plus[index] += 1e-3;
        minus[index] -= 1e-3;
        let numeric = (sparse_ce(&plus, &targets, 3).unwrap().0.sum
            - sparse_ce(&minus, &targets, 3).unwrap().0.sum)
            / 2e-3;
        assert!(
            (seeds[index] + numeric as f32).abs() < 2e-3,
            "seed {} vs gradient {numeric}",
            seeds[index]
        );
    }

    let (summary, seeds) = sparse_ce(&[0.0, 0.0], &[-100], 2).unwrap();
    assert_eq!(summary, lightnet_lm::LossSummary::default());
    assert_eq!(seeds, vec![0.0; 2]);

    let (summary, seeds) = sparse_ce(&[1000.0, -1000.0], &[1], 2).unwrap();
    assert!((summary.sum - 2000.0).abs() < 1e-3);
    assert!(seeds.iter().all(|value| value.is_finite()));

    assert!(sparse_ce(&[0.0, f32::NAN], &[0], 2).is_err());
    assert!(sparse_ce(&[0.0, f32::INFINITY], &[0], 2).is_err());
    assert!(sparse_ce(&[0.0, 0.0], &[-100], 2).is_ok());
    assert!(sparse_ce(&[0.0, 0.0], &[2], 2).is_err());
    assert!(sparse_ce(&[0.0, 0.0], &[0, 1], 2).is_err());
}

#[test]
fn lm_ops_reject_bad_shapes() {
    let cfg = cfg();
    let batch = TokenBatch {
        batch: 1,
        time: 2,
        input_ids: vec![0, 1],
        targets: vec![-100; 2],
        positions: vec![0, 1],
        segments: vec![0, 0],
        valid: vec![true; 2],
    };
    let q = vec![0.0; 2 * cfg.q_heads * cfg.head_dim()];
    let k = vec![0.0; 2 * cfg.kv_heads * cfg.head_dim()];
    let v = vec![0.0; 2 * cfg.kv_heads * cfg.head_dim()];

    assert!(causal_gqa(&q[..q.len() - 1], &k, &v, &batch, &cfg).is_err());
    assert!(causal_gqa(&q, &k, &v[..v.len() - 1], &batch, &cfg).is_err());
    assert!(rms_norm(&[0.0; 4], &[1.0; 3], 1, 1e-5).is_err());
    assert!(rms_norm(&[0.0; 4], &[1.0; 4], 1, 0.0).is_err());
    assert!(rms_norm(&[], &[1.0; 4], 0, 1e-5).is_err());
    assert!(embedding(&[], &[0], 4).is_err());
    assert!(swiglu_backward(&[0.0], &[0.0], &[0.0, 0.0]).is_err());
}
