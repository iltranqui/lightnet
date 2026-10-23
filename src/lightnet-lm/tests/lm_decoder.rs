use std::collections::BTreeSet;

use lightnet_lm::{CpuDecoder, DecoderConfig, RngState, TokenBatch};

fn tiny_cfg() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 13,
        context: 6,
        hidden: 8,
        intermediate: 12,
        layers: 2,
        q_heads: 2,
        kv_heads: 1,
        rope_theta: 10_000.0,
        rms_eps: 1e-5,
        dropout: 0.0,
        tie_embeddings: true,
    }
}

fn batch() -> TokenBatch {
    TokenBatch {
        batch: 1,
        time: 3,
        input_ids: vec![1, 5, 9],
        targets: vec![5, 9, 2],
        positions: vec![0, 1, 2],
        segments: vec![0, 0, 0],
        valid: vec![true; 3],
    }
}

fn loss_at(
    decoder: &CpuDecoder,
    batch: &TokenBatch,
    tensor: usize,
    index: usize,
    delta: f32,
) -> f32 {
    let mut probe = decoder.clone();
    let mut params = probe.parameters().clone();
    params.tensors[tensor].values[index] += delta;
    probe.set_parameters(params).unwrap();
    let forward = probe.forward(batch, None).unwrap();
    probe.loss(&forward, batch).unwrap().0.sum as f32
}

#[test]
fn lm_decoder_full_parameter_gradient() {
    let decoder = CpuDecoder::new(tiny_cfg(), 7).unwrap();
    let batch = batch();
    let forward = decoder.forward(&batch, None).unwrap();
    let (summary, seeds) = decoder.loss(&forward, &batch).unwrap();
    assert_eq!(summary.active_tokens, 3);
    let grads = decoder.backward(&forward, &seeds).unwrap();
    assert_eq!(grads.values.len(), decoder.parameters().tensors.len());

    let step = 1e-3f32;
    for (tensor, parameter) in decoder.parameters().tensors.iter().enumerate() {
        let length = parameter.values.len();
        let mut indices = vec![0usize, length / 2, length - 1];
        indices.sort_unstable();
        indices.dedup();
        for index in indices {
            let plus = loss_at(&decoder, &batch, tensor, index, step);
            let minus = loss_at(&decoder, &batch, tensor, index, -step);
            let numeric = (plus - minus) / (2.0 * step);
            let analytic = grads.values[tensor][index];
            let tolerance = 2e-3 + 2e-2 * numeric.abs();
            assert!(
                (analytic + numeric).abs() <= tolerance,
                "{}[{index}]: descent {analytic} vs -gradient {numeric}",
                parameter.name
            );
        }
    }
}

#[test]
fn lm_decoder_tied_gradient() {
    let cfg = tiny_cfg();
    let decoder = CpuDecoder::new(cfg.clone(), 11).unwrap();
    let batch = batch();
    let forward = decoder.forward(&batch, None).unwrap();
    let (_, seeds) = decoder.loss(&forward, &batch).unwrap();
    let grads = decoder.backward(&forward, &seeds).unwrap();
    let (head, gather) = decoder.embedding_gradient_terms(&forward, &seeds).unwrap();

    let rows = batch.rows();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden;
    let final_norm = forward.tape.final_norm();

    let mut expected_head = vec![0.0f32; vocab * hidden];
    for row in 0..rows {
        let seed_row = &seeds[row * vocab..(row + 1) * vocab];
        for (class, &seed) in seed_row.iter().enumerate() {
            for channel in 0..hidden {
                expected_head[class * hidden + channel] +=
                    seed * final_norm[row * hidden + channel];
            }
        }
    }

    for index in 0..vocab * hidden {
        assert!((head[index] - expected_head[index]).abs() < 1e-6);
        assert!(
            (grads.values[0][index] - (head[index] + gather[index])).abs() < 1e-6,
            "tied embedding entry {index} is not head + gather"
        );
    }
    for class in 0..vocab {
        if batch.input_ids.contains(&(class as u32)) {
            continue;
        }
        for channel in 0..hidden {
            assert_eq!(gather[class * hidden + channel], 0.0);
        }
    }
}

#[test]
fn lm_decoder_parameter_registry() {
    let cfg = tiny_cfg();
    let decoder = CpuDecoder::new(cfg.clone(), 3).unwrap();
    assert_eq!(decoder.parameters().tensors.len(), 1 + 9 * cfg.layers + 1);

    let mut names = BTreeSet::new();
    for (index, parameter) in decoder.parameters().tensors.iter().enumerate() {
        assert_eq!(parameter.id as usize, index);
        assert!(
            names.insert(parameter.name.clone()),
            "duplicate {}",
            parameter.name
        );
        assert_eq!(
            parameter.values.len(),
            parameter.shape.iter().product::<usize>()
        );
        assert!(parameter.values.iter().all(|value| value.is_finite()));
    }
    assert_eq!(decoder.parameters().tensors[0].name, "embedding");
    assert_eq!(
        decoder.parameters().tensors[0].shape,
        vec![cfg.vocab_size, cfg.hidden]
    );
    assert_eq!(
        decoder.parameters().tensors.last().unwrap().name,
        "final_norm"
    );
    assert_eq!(
        decoder
            .parameters()
            .tensors
            .iter()
            .filter(|parameter| parameter.name == "embedding")
            .count(),
        1
    );
    for parameter in &decoder.parameters().tensors {
        let is_norm = parameter.name.ends_with("_norm");
        assert_eq!(parameter.decay, !is_norm, "{}", parameter.name);
    }

    let again = CpuDecoder::new(cfg.clone(), 3).unwrap();
    assert_eq!(decoder.parameters(), again.parameters());
    let different = CpuDecoder::new(cfg, 4).unwrap();
    assert_ne!(decoder.parameters(), different.parameters());

    let forward = decoder.forward(&batch(), None).unwrap();
    let (_, seeds) = decoder.loss(&forward, &batch()).unwrap();
    let mut bad = decoder.parameters().clone();
    bad.tensors[0].name = "not_embedding".to_owned();
    let mut probe = decoder.clone();
    assert!(probe.set_parameters(bad).is_err());
    assert_eq!(probe.revision(), decoder.revision());

    let mut bad = decoder.parameters().clone();
    bad.tensors[1].values[0] = f32::NAN;
    let mut probe = decoder.clone();
    assert!(probe.set_parameters(bad).is_err());

    let mut replaced = decoder.clone();
    replaced
        .set_parameters(decoder.parameters().clone())
        .unwrap();
    assert!(replaced.backward(&forward, &seeds).is_err());
}

#[test]
fn lm_decoder_prefix_causality() {
    let decoder = CpuDecoder::new(tiny_cfg(), 5).unwrap();
    let mut original = batch();
    original.time = 4;
    original.input_ids = vec![1, 5, 9, 4];
    original.targets = vec![5, 9, 4, 2];
    original.positions = vec![0, 1, 2, 3];
    original.segments = vec![0, 0, 0, 0];
    original.valid = vec![true; 4];

    let baseline = decoder.forward(&original, None).unwrap();
    let mut changed = original.clone();
    changed.input_ids[2] = 12;
    changed.input_ids[3] = 7;
    let updated = decoder.forward(&changed, None).unwrap();

    let row_width = 13 * 2;
    assert_eq!(&baseline.logits[..row_width], &updated.logits[..row_width]);
}

#[test]
fn lm_decoder_padding_isolation() {
    let decoder = CpuDecoder::new(tiny_cfg(), 6).unwrap();
    let short = batch();
    let mut padded = short.clone();
    padded.time = 5;
    padded.input_ids = vec![1, 5, 9, 0, 0];
    padded.targets = vec![5, 9, 2, -100, -100];
    padded.positions = vec![0, 1, 2, 0, 1];
    padded.segments = vec![0, 0, 0, 1, 1];
    padded.valid = vec![true, true, true, false, false];

    let padded_forward = decoder.forward(&padded, None).unwrap();
    let short_forward = decoder.forward(&short, None).unwrap();
    let width = short.rows() * short.batch * 13;
    assert_eq!(&padded_forward.logits[..width], &short_forward.logits);
}

#[test]
fn lm_decoder_dropout_replay() {
    let mut cfg = tiny_cfg();
    cfg.dropout = 0.25;
    let decoder = CpuDecoder::new(cfg, 13).unwrap();
    let batch = batch();

    let mut first_rng = RngState::new(99);
    let mut second_rng = RngState::new(99);
    let first = decoder.forward(&batch, Some(&mut first_rng)).unwrap();
    let second = decoder.forward(&batch, Some(&mut second_rng)).unwrap();
    assert_eq!(first.logits, second.logits);
    assert_eq!(first_rng, second_rng);

    let (_, first_seeds) = decoder.loss(&first, &batch).unwrap();
    let (_, second_seeds) = decoder.loss(&second, &batch).unwrap();
    let first_grads = decoder.backward(&first, &first_seeds).unwrap();
    let second_grads = decoder.backward(&second, &second_seeds).unwrap();
    assert_eq!(first_grads, second_grads);

    let mut untouched = RngState::new(1234);
    let before = untouched;
    decoder.forward(&batch, None).unwrap();
    assert_eq!(untouched, before);
    decoder.forward(&batch, Some(&mut untouched)).unwrap();
    assert_ne!(untouched, before);
    assert_ne!(
        decoder
            .forward(&batch, Some(&mut RngState::new(1)))
            .unwrap()
            .logits,
        first.logits
    );
}

#[test]
fn lm_decoder_seed_step_lowers_loss() {
    let mut decoder = CpuDecoder::new(tiny_cfg(), 17).unwrap();
    let batch = batch();
    let forward = decoder.forward(&batch, None).unwrap();
    let (before, seeds) = decoder.loss(&forward, &batch).unwrap();
    let grads = decoder.backward(&forward, &seeds).unwrap();
    assert!(
        grads
            .values
            .iter()
            .all(|values| values.iter().all(|value| value.is_finite()))
    );

    let mut params = decoder.parameters().clone();
    let lr = 1e-3f32;
    for (parameter, gradient) in params.tensors.iter_mut().zip(&grads.values) {
        for (value, &update) in parameter.values.iter_mut().zip(gradient) {
            *value += lr * update;
        }
    }
    decoder.set_parameters(params).unwrap();

    let forward = decoder.forward(&batch, None).unwrap();
    let (after, _) = decoder.loss(&forward, &batch).unwrap();
    assert!(
        after.sum < before.sum,
        "seed step did not lower loss: before {}, after {}",
        before.sum,
        after.sum
    );
}
