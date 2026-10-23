//! Native masked attention primitive used by the LayaHead forward.
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

fn reference(
    qkv: &[f32],
    mask: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
) -> Vec<f32> {
    let head_dim = channels / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; batch * tokens * channels];
    for b in 0..batch {
        for h in 0..heads {
            for query in 0..tokens {
                let mut scores = Vec::new();
                for key in 0..tokens {
                    if mask[b * tokens + key] > 0.5 {
                        let mut dot = 0.0;
                        for d in 0..head_dim {
                            let qi = (b * tokens + query) * 3 * channels + h * head_dim + d;
                            let ki =
                                (b * tokens + key) * 3 * channels + channels + h * head_dim + d;
                            dot += qkv[qi] * qkv[ki];
                        }
                        scores.push((key, dot * scale));
                    }
                }
                if scores.is_empty() {
                    continue;
                }
                let max = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<_> = scores
                    .iter()
                    .map(|(_, score)| (*score - max).exp())
                    .collect();
                let inv_sum = 1.0 / exps.iter().sum::<f32>();
                for d in 0..head_dim {
                    let mut value = 0.0;
                    for ((key, _), exp) in scores.iter().zip(&exps) {
                        let vi =
                            (b * tokens + *key) * 3 * channels + 2 * channels + h * head_dim + d;
                        value += exp * inv_sum * qkv[vi];
                    }
                    output[(b * tokens + query) * channels + h * head_dim + d] = value;
                }
            }
        }
    }
    output
}

#[test]
fn laya_head_masked_attention_matches_reference_including_empty_mask_row() {
    let (batch, tokens, channels, heads) = (2, 4, 6, 2);
    let qkv: Vec<_> = (0..batch * tokens * 3 * channels)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.031)
        .collect();
    let mask = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let expected = reference(&qkv, &mask, batch, tokens, channels, heads);
    let backend = OxideBackend::new().expect("oxide backend");
    let qkv_dev = backend
        .upload_f32(&[batch * tokens, 3 * channels], &qkv)
        .unwrap();
    let mask_dev = backend.upload_f32(&[batch, tokens], &mask).unwrap();
    let output = backend
        .laya_head_attention_f32(
            &qkv_dev,
            &mask_dev,
            &[batch * tokens, channels],
            batch,
            tokens,
            channels,
            heads,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let actual = backend.download_f32(&output).unwrap();
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
            "{index}: {got} vs {want}"
        );
    }
    let empty_row = &actual[channels * tokens..2 * channels * tokens];
    assert!(empty_row.iter().all(|value| *value == 0.0));
}

#[test]
fn laya_head_masked_attention_rejects_bad_extents() {
    let backend = OxideBackend::new().unwrap();
    let qkv = backend.upload_f32(&[2, 6], &[0.0; 12]).unwrap();
    let short_mask = backend.upload_f32(&[1, 1], &[1.0]).unwrap();
    assert!(
        backend
            .laya_head_attention_f32(
                &qkv,
                &short_mask,
                &[2, 2],
                1,
                2,
                2,
                1,
                ElementwiseLaunchConfig::default(),
            )
            .is_err()
    );
}
