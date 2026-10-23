use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use lightnet_cuda::host_ops::{AdamParams, adam_update};
use lightnet_lm::{
    ByteBpe, CHECKPOINT_SCHEMA_VERSION, CorpusDocument, CpuEngine, DecoderConfig, DecoderGrads,
    LmEngine, NormalizationSpec, TokenBatch, TokenLoader, TokenSchedule, TrainOptions,
    TrainingSnapshot, UpdateOptions, clip_global_norm, load_checkpoint, read_manifest,
    save_checkpoint, write_token_shards,
};
use sha2::{Digest, Sha256};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lightnet_lm_train_{}_{}_{}",
            std::process::id(),
            tag,
            stamp
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn cfg() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 260,
        context: 4,
        hidden: 8,
        intermediate: 12,
        layers: 1,
        q_heads: 2,
        kv_heads: 1,
        rope_theta: 10_000.0,
        rms_eps: 1.0e-5,
        dropout: 0.0,
        tie_embeddings: true,
    }
}

fn options(lr: f32) -> UpdateOptions {
    UpdateOptions {
        lr,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1.0e-8,
        weight_decay: 0.1,
        clip_norm: 1000.0,
        active_tokens: 0,
        step: 0,
    }
}

fn batch(ids: &[u32], targets: &[i32], time: usize, segments: &[u32]) -> TokenBatch {
    TokenBatch {
        batch: ids.len() / time,
        time,
        input_ids: ids.to_vec(),
        targets: targets.to_vec(),
        positions: (0..time as u32).cycle().take(ids.len()).collect(),
        segments: segments.to_vec(),
        valid: targets.iter().map(|target| *target != -100).collect(),
    }
}

fn one_row(id: u32, target: i32, segment: u32) -> TokenBatch {
    batch(&[id], &[target], 1, &[segment])
}

fn tokenizer() -> ByteBpe {
    ByteBpe::train(&[], 260, NormalizationSpec::default()).unwrap()
}

fn document(root: &Path, name: &str, text: &str, work: &str) -> CorpusDocument {
    let text_dir = root.join("text");
    fs::create_dir_all(&text_dir).unwrap();
    let path = text_dir.join(name);
    fs::write(&path, text).unwrap();
    CorpusDocument {
        id: name.to_owned(),
        work_id: work.to_owned(),
        split: "validation".to_owned(),
        source: "unit-test".to_owned(),
        license: "CC0-1.0".to_owned(),
        sha256: sha256_hex(text.as_bytes()),
        text_path: path,
    }
}

fn shard_dir(root: &Path, text: &str) -> PathBuf {
    let docs = vec![document(root, "doc.txt", text, "work")];
    let out = root.join("shards");
    write_token_shards(&docs, &tokenizer(), &out, 1000).unwrap();
    out.join("manifest.lmtxt")
}

fn snapshot(
    engine: &CpuEngine,
    loader: &TokenLoader,
    pending_tokens: u64,
    pending_microbatches: u32,
) -> TrainingSnapshot {
    let schedule = TokenSchedule {
        warmup_tokens: 2,
        total_tokens: 100,
        peak_lr: 1.0e-3,
        min_lr: 1.0e-4,
    };
    TrainingSnapshot {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        tokenizer_hash: loader.manifest().tokenizer_hash.clone(),
        manifest_hash: loader.manifest().manifest_hash.clone(),
        engine: engine.export_state().unwrap(),
        loader: loader.state(),
        train_options: TrainOptions {
            batch: 1,
            context: 2,
            accum: 1,
            seed: 7,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1.0e-8,
            weight_decay: 0.1,
            clip_norm: 1.0,
            schedule,
        },
        pending_active_tokens: pending_tokens,
        pending_microbatches,
        successful_updates: engine.step(),
        updated_tokens: engine.updated_tokens(),
        schedule,
    }
}

#[test]
fn lm_train_token_schedule() {
    let schedule = TokenSchedule {
        warmup_tokens: 10,
        total_tokens: 30,
        peak_lr: 1.0,
        min_lr: 0.1,
    };
    assert_eq!(schedule.lr(0).unwrap(), 0.0);
    assert!((schedule.lr(10).unwrap() - 1.0).abs() < 1.0e-6);
    assert!((schedule.lr(20).unwrap() - 0.55).abs() < 1.0e-6);
    assert!((schedule.lr(30).unwrap() - 0.1).abs() < 1.0e-6);
    assert!((schedule.lr(300).unwrap() - 0.1).abs() < 1.0e-6);

    let zero_warmup = TokenSchedule {
        warmup_tokens: 0,
        total_tokens: 10,
        peak_lr: 1.0,
        min_lr: 0.1,
    };
    assert!((zero_warmup.lr(0).unwrap() - 1.0).abs() < 1.0e-6);

    let invalid = TokenSchedule {
        warmup_tokens: 30,
        total_tokens: 30,
        peak_lr: 1.0,
        min_lr: 0.1,
    };
    assert!(invalid.lr(0).is_err());
}

#[test]
fn lm_train_global_clip_unique_params() {
    let mut grads = DecoderGrads {
        values: vec![vec![3.0, 4.0]],
    };
    let (norm, factor) = clip_global_norm(&mut grads, 100.0).unwrap();
    assert!((norm - 5.0).abs() < 1.0e-6);
    assert_eq!(factor, 1.0);
    assert_eq!(grads.values[0], vec![3.0, 4.0]);

    let (norm, factor) = clip_global_norm(&mut grads, 1.0).unwrap();
    assert!((norm - 5.0).abs() < 1.0e-6);
    assert!((factor - 0.2).abs() < 1.0e-6);
    assert!((grads.values[0][0] - 0.6).abs() < 1.0e-6);
    assert!((grads.values[0][1] - 0.8).abs() < 1.0e-6);

    let mut bad = DecoderGrads {
        values: vec![vec![f32::NAN]],
    };
    assert!(clip_global_norm(&mut bad, 1.0).is_err());
    assert!(clip_global_norm(&mut grads, 0.0).is_err());
}

#[test]
fn lm_train_adamw_groups() {
    let mut engine = CpuEngine::new(cfg(), 11).unwrap();
    let mut state = engine.export_state().unwrap();
    for (index, values) in state.pending.values.iter_mut().enumerate() {
        for value in values {
            *value = if index == 0 { 0.5 } else { 0.25 };
        }
    }
    state.pending_tokens = 2;
    engine.import_state(&state).unwrap();
    let before = engine.parameters().clone();
    let report = engine.apply_update(&options(0.1)).unwrap();
    assert!(report.applied);
    assert_eq!(engine.step(), 1);
    assert_eq!(engine.updated_tokens(), 2);

    for (index, parameter) in engine.parameters().tensors.iter().enumerate() {
        let update = state.pending.values[index]
            .iter()
            .map(|value| *value / 2.0)
            .collect::<Vec<_>>();
        let params = AdamParams {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1.0e-8,
            decay: if parameter.decay { 0.1 } else { 0.0 },
            batch: 1,
            step: 1,
            decoupled_decay: true,
        };
        let (expected, _, _) = adam_update(
            &before.tensors[index].values,
            &update,
            &vec![0.0; update.len()],
            &vec![0.0; update.len()],
            &params,
        );
        for (got, want) in parameter.values.iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-6, "tensor {index}");
        }
    }
    assert_eq!(engine.parameters().tensors.len(), 1 + 9 + 1);
}

#[test]
fn lm_train_token_normalization() {
    let mut split = CpuEngine::new(cfg(), 13).unwrap();
    split.loss_sum_backward(&one_row(5, 7, 0)).unwrap();
    split.loss_sum_backward(&one_row(6, 8, 0)).unwrap();
    let split_pending = split.export_state().unwrap().pending;

    let mut joined = CpuEngine::new(cfg(), 13).unwrap();
    joined
        .loss_sum_backward(&batch(&[5, 6], &[7, 8], 1, &[0, 1]))
        .unwrap();
    let joined_pending = joined.export_state().unwrap().pending;

    for (left, right) in split_pending.values.iter().zip(&joined_pending.values) {
        for (a, b) in left.iter().zip(right) {
            assert!((a - b).abs() < 1.0e-6);
        }
    }
    let report_split = split.apply_update(&options(0.01)).unwrap();
    let report_joined = joined.apply_update(&options(0.01)).unwrap();
    assert!(report_split.applied && report_joined.applied);
    for (left, right) in split
        .parameters()
        .tensors
        .iter()
        .zip(&joined.parameters().tensors)
    {
        for (a, b) in left.values.iter().zip(&right.values) {
            assert!((a - b).abs() < 1.0e-6);
        }
    }
}

#[test]
fn lm_train_all_ignored_no_step() {
    let mut engine = CpuEngine::new(cfg(), 17).unwrap();
    let before = engine.parameters().clone();
    let summary = engine
        .loss_sum_backward(&batch(&[5, 6], &[-100, -100], 1, &[0, 1]))
        .unwrap();
    assert_eq!(summary.active_tokens, 0);
    let report = engine.apply_update(&options(0.1)).unwrap();
    assert!(!report.applied);
    assert_eq!(engine.step(), 0);
    assert_eq!(engine.updated_tokens(), 0);
    assert_eq!(engine.parameters(), &before);
}

#[test]
fn lm_train_rejects_nonfinite_atomically() {
    let mut engine = CpuEngine::new(cfg(), 19).unwrap();
    let mut state = engine.export_state().unwrap();
    state.pending.values[0][0] = f32::NAN;
    state.pending_tokens = 1;
    engine.import_state(&state).unwrap();
    let before = engine.export_state().unwrap();
    let err = engine.apply_update(&options(0.1)).unwrap_err();
    assert!(err.to_string().contains("non-finite"));
    let after = engine.export_state().unwrap();
    assert_eq!(before.parameters, after.parameters);
    assert_eq!(before.first_moment, after.first_moment);
    assert_eq!(before.second_moment, after.second_moment);
    assert_eq!(before.step, after.step);
}

#[test]
fn lm_train_step_overflow() {
    let mut engine = CpuEngine::new(cfg(), 23).unwrap();
    let mut state = engine.export_state().unwrap();
    state.pending.values[0][0] = 1.0;
    state.pending_tokens = 1;
    state.step = u32::MAX;
    engine.import_state(&state).unwrap();
    let before = engine.parameters().clone();
    assert!(engine.apply_update(&options(0.1)).is_err());
    assert_eq!(engine.parameters(), &before);
    assert_eq!(engine.step(), u32::MAX);
}

#[test]
fn lm_train_resume_mid_accumulation() {
    let dir = TempDir::new("resume");
    let manifest = shard_dir(&dir.path, "abcdefgh");
    let mut loader = TokenLoader::open(&manifest, 1, 2, 7).unwrap();
    let first = loader.next_batch().unwrap().expect("first batch");

    let mut original = CpuEngine::new(cfg(), 7).unwrap();
    original.loss_sum_backward(&first).unwrap();
    let saved = snapshot(&original, &loader, original.pending_tokens(), 1);
    let path = dir.path.join("resume.lmckpt");
    save_checkpoint(&path, &saved).unwrap();
    let loaded = load_checkpoint(&path).unwrap();
    assert_eq!(loaded, saved);

    let mut restored = CpuEngine::new(cfg(), 7).unwrap();
    restored.import_state(&loaded.engine).unwrap();
    assert_eq!(
        restored.export_state().unwrap(),
        original.export_state().unwrap()
    );

    let mut restored_loader = TokenLoader::open(&manifest, 1, 2, 7).unwrap();
    restored_loader.restore(&loaded.loader).unwrap();
    assert_eq!(
        loader.next_batch().unwrap(),
        restored_loader.next_batch().unwrap()
    );

    let second = one_row(9, 11, 0);
    original.loss_sum_backward(&second).unwrap();
    restored.loss_sum_backward(&second).unwrap();
    let a = original.apply_update(&options(0.01)).unwrap();
    let b = restored.apply_update(&options(0.01)).unwrap();
    assert!(a.applied && b.applied);
    assert_eq!(original.parameters(), restored.parameters());
    assert_eq!(
        original.export_state().unwrap().first_moment,
        restored.export_state().unwrap().first_moment
    );
}

#[test]
fn lm_train_checkpoint_validation() {
    let dir = TempDir::new("checkpoint");
    let manifest = shard_dir(&dir.path, "abcdefgh");
    let mut loader = TokenLoader::open(&manifest, 1, 2, 7).unwrap();
    let batch = loader.next_batch().unwrap().expect("batch");
    let mut engine = CpuEngine::new(cfg(), 7).unwrap();
    engine.loss_sum_backward(&batch).unwrap();
    let saved = snapshot(&engine, &loader, engine.pending_tokens(), 1);
    let path = dir.path.join("valid.lmckpt");
    save_checkpoint(&path, &saved).unwrap();

    let bytes = fs::read(&path).unwrap();
    let mut corrupted = bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    let bad_path = dir.path.join("corrupt.lmckpt");
    fs::write(&bad_path, &corrupted).unwrap();
    assert!(load_checkpoint(&bad_path).is_err());

    let truncated = dir.path.join("truncated.lmckpt");
    fs::write(&truncated, &bytes[..bytes.len() - 4]).unwrap();
    assert!(load_checkpoint(&truncated).is_err());

    let mut magic = bytes.clone();
    magic[0] ^= 0x01;
    let magic_path = dir.path.join("magic.lmckpt");
    fs::write(&magic_path, &magic).unwrap();
    assert!(load_checkpoint(&magic_path).is_err());

    let mut mismatched = saved.clone();
    mismatched.manifest_hash = "00".repeat(32);
    assert!(save_checkpoint(&dir.path.join("mismatch.lmckpt"), &mismatched).is_err());

    let mut poisoned = saved;
    poisoned.engine.pending.values[0][0] = f32::INFINITY;
    assert!(save_checkpoint(&dir.path.join("poison.lmckpt"), &poisoned).is_err());
}

#[test]
fn lm_train_manifest_hashes_match_loader() {
    let dir = TempDir::new("hashes");
    let manifest_path = shard_dir(&dir.path, "abc");
    let manifest = read_manifest(&manifest_path).unwrap();
    assert_eq!(manifest.manifest_hash.len(), 64);
    assert_eq!(manifest.tokenizer_hash.len(), 64);
    let loader = TokenLoader::open(&manifest_path, 1, 2, 1).unwrap();
    assert_eq!(loader.state().manifest_hash, manifest.manifest_hash);
}
