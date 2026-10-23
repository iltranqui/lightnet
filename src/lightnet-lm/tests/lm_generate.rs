use lightnet_lm::{
    ByteBpe, CpuDecoder, DecoderConfig, KvCache, NormalizationSpec, RngState, SamplingOptions,
    TokenBatch, generate, sample_token,
};

fn cfg(kv_heads: usize) -> DecoderConfig {
    DecoderConfig {
        vocab_size: 260,
        context: 8,
        hidden: 8,
        intermediate: 12,
        layers: 1,
        q_heads: 2,
        kv_heads,
        rope_theta: 10_000.0,
        rms_eps: 1.0e-5,
        dropout: 0.0,
        tie_embeddings: true,
    }
}

fn full_logits(decoder: &CpuDecoder, ids: &[u32]) -> Vec<f32> {
    let time = ids.len();
    let batch = TokenBatch {
        batch: 1,
        time,
        input_ids: ids.to_vec(),
        targets: vec![-100; time],
        positions: (0..time as u32).collect(),
        segments: vec![0; time],
        valid: vec![true; time],
    };
    let forward = decoder.forward(&batch, None).unwrap();
    forward.logits[(time - 1) * decoder.config().vocab_size..].to_vec()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-5 + 1.0e-5 * e.abs();
        assert!(
            (a - e).abs() <= tolerance,
            "logit {index}: {a} vs {e} (tolerance {tolerance})"
        );
    }
}

#[test]
fn lm_generate_cached_matches_prefix() {
    for kv_heads in [2, 1] {
        let decoder = CpuDecoder::new(cfg(kv_heads), 42).unwrap();
        let mut cache = KvCache::new(decoder.config(), 1).unwrap();
        let prompt = [4u32, 5, 6];
        let prefill = decoder.prefill(&prompt, &mut cache).unwrap();
        close(&prefill, &full_logits(&decoder, &prompt));
        assert_eq!(cache.len(), prompt.len());

        let mut prefix = prompt.to_vec();
        for token in [7u32, 8, 9] {
            let produced = decoder.decode(&[token], &mut cache).unwrap();
            prefix.push(token);
            close(&produced, &full_logits(&decoder, &prefix));
        }
    }
}

#[test]
fn lm_generate_nonzero_rope_offset() {
    let decoder = CpuDecoder::new(cfg(1), 77).unwrap();
    let mut cache = KvCache::new(decoder.config(), 1).unwrap();
    let mut prefix = vec![10u32];
    decoder.prefill(&prefix, &mut cache).unwrap();
    for token in [11u32, 12, 13, 14] {
        let produced = decoder.decode(&[token], &mut cache).unwrap();
        prefix.push(token);
        close(&produced, &full_logits(&decoder, &prefix));
    }
}

#[test]
fn lm_generate_cache_validation() {
    let mut decoder = CpuDecoder::new(cfg(1), 5).unwrap();
    let mut cache = KvCache::new(decoder.config(), 1).unwrap();
    let full = [4u32, 5, 6, 7, 8, 9, 10, 11];
    decoder.prefill(&full, &mut cache).unwrap();
    let before = cache.len();
    assert!(decoder.decode(&[12], &mut cache).is_err());
    assert_eq!(cache.len(), before);

    let mut small = KvCache::new(decoder.config(), 1).unwrap();
    decoder.prefill(&[4, 5], &mut small).unwrap();
    assert!(decoder.decode(&[260], &mut small).is_err());
    assert_eq!(small.len(), 2);
    assert!(decoder.decode(&[4, 5], &mut small).is_err());
    assert_eq!(small.len(), 2);

    let mut probe = small.clone();
    let before_probe = decoder.decode(&[7], &mut probe).unwrap();
    assert!(decoder.prefill(&[8], &mut small).is_err());
    assert_eq!(small.len(), 2);
    let mut probe = small.clone();
    let after_probe = decoder.decode(&[7], &mut probe).unwrap();
    close(&before_probe, &after_probe);

    let mut parameters = decoder.parameters().clone();
    parameters.tensors[0].values[0] += 0.125;
    decoder.set_parameters(parameters).unwrap();
    assert!(decoder.decode(&[7], &mut small).is_err());
    assert_eq!(small.len(), 2);

    assert!(KvCache::new(decoder.config(), 2).is_err());
}

#[test]
fn lm_generate_greedy_ties() {
    let mut rng = RngState::new(0);
    let mut logits = vec![0.0f32; 8];
    logits[2] = -10.0;
    let options = SamplingOptions {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    };
    assert_eq!(sample_token(&logits, &options, &mut rng).unwrap(), 4);

    let mut logits = vec![0.0f32; 8];
    logits[2] = -10.0;
    logits[4] = -1.0;
    assert_eq!(sample_token(&logits, &options, &mut rng).unwrap(), 5);

    logits[0] = 100.0;
    logits[1] = 100.0;
    logits[3] = 100.0;
    assert_eq!(sample_token(&logits, &options, &mut rng).unwrap(), 5);
}

#[test]
fn lm_generate_seed_replay() {
    let logits = (0..16).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
    let options = SamplingOptions {
        temperature: 0.8,
        top_k: 6,
        top_p: 0.9,
        seed: 123,
    };
    let mut left = RngState::new(options.seed);
    let mut right = RngState::new(options.seed);
    for _ in 0..32 {
        let a = sample_token(&logits, &options, &mut left).unwrap();
        let b = sample_token(&logits, &options, &mut right).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn lm_generate_filters() {
    let mut logits = vec![-100.0f32; 7];
    logits[4] = 0.5f32.ln();
    logits[5] = 0.3f32.ln();
    logits[6] = 0.2f32.ln();
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.6,
        seed: 9,
    };
    let mut rng = RngState::new(9);
    for _ in 0..200 {
        let token = sample_token(&logits, &options, &mut rng).unwrap();
        assert!(token == 4 || token == 5, "sampled excluded token {token}");
    }

    let top_one = SamplingOptions {
        top_k: 1,
        ..options
    };
    let mut rng = RngState::new(9);
    for _ in 0..32 {
        assert_eq!(sample_token(&logits, &top_one, &mut rng).unwrap(), 4);
    }
}

#[test]
fn lm_generate_invalid_distribution() {
    let mut rng = RngState::new(0);
    let options = SamplingOptions::default();
    assert!(sample_token(&[0.0, 0.0, 0.0, f32::NAN, 1.0], &options, &mut rng).is_err());
    assert!(sample_token(&[0.0, 0.0, 0.0, 1.0, f32::INFINITY], &options, &mut rng).is_err());
    assert!(sample_token(&[0.0, 0.0], &options, &mut rng).is_err());
    assert!(
        sample_token(
            &[0.0, 0.0, 0.0, 1.0],
            &SamplingOptions {
                temperature: -1.0,
                ..options
            },
            &mut rng
        )
        .is_err()
    );
    assert!(
        sample_token(
            &[0.0, 0.0, 0.0, 1.0],
            &SamplingOptions {
                top_p: 0.0,
                ..options
            },
            &mut rng
        )
        .is_err()
    );
    assert!(
        sample_token(
            &[0.0, 0.0, 0.0, 1.0],
            &SamplingOptions {
                top_p: 1.5,
                ..options
            },
            &mut rng
        )
        .is_err()
    );
}

#[test]
fn lm_generate_eos_and_empty_prompt() {
    let mut rng = RngState::new(0);
    let options = SamplingOptions {
        temperature: 0.0,
        ..SamplingOptions::default()
    };
    let mut logits = vec![0.0f32; 8];
    logits[2] = 5.0;
    assert_eq!(sample_token(&logits, &options, &mut rng).unwrap(), 2);

    let config = cfg(1);
    let decoder = CpuDecoder::new(config.clone(), 3).unwrap();
    let tokenizer = ByteBpe::train(&[], 260, NormalizationSpec::default()).unwrap();
    let generated = generate(&decoder, &tokenizer, "", &options, 0).unwrap();
    assert!(generated.is_empty());

    let mut cache = KvCache::new(&config, 1).unwrap();
    let logits = decoder.prefill(&[1], &mut cache).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(logits.len(), config.vocab_size);
}
