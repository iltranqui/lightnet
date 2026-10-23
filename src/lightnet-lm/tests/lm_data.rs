use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use lightnet_lm::{
    BOS_ID, ByteBpe, CorpusDocument, EOS_ID, NormalizationSpec, PAD_ID, TokenLoader, read_manifest,
    write_token_shards,
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
            "lightnet_lm_data_{}_{}_{}",
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

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn tokenizer() -> ByteBpe {
    ByteBpe::train(&[], 260, NormalizationSpec::default()).unwrap()
}

fn document(root: &Path, name: &str, text: &str, split: &str, work: &str) -> CorpusDocument {
    let text_dir = root.join("text");
    fs::create_dir_all(&text_dir).unwrap();
    let path = text_dir.join(name);
    fs::write(&path, text).unwrap();
    CorpusDocument {
        id: name.to_owned(),
        work_id: work.to_owned(),
        split: split.to_owned(),
        source: "unit-test".to_owned(),
        license: "CC0-1.0".to_owned(),
        sha256: sha256_hex(text.as_bytes()),
        text_path: path,
    }
}

fn setup(tag: &str, text: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new(tag);
    let docs = vec![document(&dir.path, "doc.txt", text, "validation", "work")];
    let out = dir.path.join("shards");
    write_token_shards(&docs, &tokenizer(), &out, 1000).unwrap();
    (dir, out)
}

#[test]
fn lm_data_shift_and_pack() {
    let dir = TempDir::new("pack");
    let tokenizer = tokenizer();
    let docs = vec![
        document(&dir.path, "ab.txt", "ab", "validation", "w1"),
        document(&dir.path, "cd.txt", "cd", "validation", "w2"),
    ];
    let out = dir.path.join("shards");
    let manifest = write_token_shards(&docs, &tokenizer, &out, 1000).unwrap();

    assert_eq!(manifest.format_version, 1);
    assert_eq!(manifest.split, "validation");
    assert_eq!(manifest.tokenizer_hash, tokenizer.artifact_hash());
    assert_eq!(manifest.documents.len(), 2);
    assert_eq!(manifest.shards.len(), 1);
    assert_eq!(manifest.shards[0].token_count, 8);
    assert_eq!(manifest.documents[0].source, "unit-test");
    assert_eq!(manifest.documents[0].license, "CC0-1.0");
    assert_eq!(manifest.documents[0].len, 4);
    assert_eq!(manifest.documents[1].start, 4);
    assert_eq!(
        read_manifest(&out.join("manifest.lmtxt")).unwrap(),
        manifest
    );

    let mut loader = TokenLoader::open(&out.join("manifest.lmtxt"), 1, 6, 42).unwrap();
    let batch = loader.next_batch().unwrap().unwrap();
    assert_eq!(batch.batch, 1);
    assert_eq!(batch.time, 6);
    assert_eq!(batch.input_ids, vec![BOS_ID, 101, 102, BOS_ID, 103, 104]);
    assert_eq!(
        batch.targets,
        vec![101, 102, EOS_ID as i32, 103, 104, EOS_ID as i32]
    );
    assert_eq!(batch.positions, vec![0, 1, 2, 0, 1, 2]);
    assert_eq!(batch.segments, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(batch.valid, vec![true; 6]);

    let mut padded = TokenLoader::open(&out.join("manifest.lmtxt"), 2, 4, 42).unwrap();
    let batch = padded.next_batch().unwrap().unwrap();
    assert_eq!(
        batch.input_ids,
        vec![BOS_ID, 101, 102, PAD_ID, BOS_ID, 103, 104, PAD_ID]
    );
    assert_eq!(
        batch.targets,
        vec![101, 102, EOS_ID as i32, -100, 103, 104, EOS_ID as i32, -100]
    );
    assert_eq!(batch.positions, vec![0, 1, 2, 0, 0, 1, 2, 0]);
    assert_eq!(batch.segments, vec![0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        batch.valid,
        vec![true, true, true, false, true, true, true, false]
    );
}

#[test]
fn lm_data_long_document_coverage() {
    let (dir, out) = setup("long", "abcdefghijklmnopqrst");
    let mut loader = TokenLoader::open(&out.join("manifest.lmtxt"), 1, 4, 9).unwrap();
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for _ in 0..6 {
        let batch = loader.next_batch().unwrap().unwrap();
        for (&target, &valid) in batch.targets.iter().zip(&batch.valid) {
            if valid {
                *counts.entry(target).or_default() += 1;
            }
        }
        for (&input, &valid) in batch.input_ids.iter().zip(&batch.valid) {
            if valid {
                assert_ne!(input, EOS_ID, "EOS must never be an input token");
            }
        }
        for (&target, &valid) in batch.targets.iter().zip(&batch.valid) {
            if valid {
                assert_ne!(target, BOS_ID as i32, "BOS must never be a target");
            }
        }
    }
    for token in 101..=120 {
        assert_eq!(counts.get(&token), Some(&1), "token {token}");
    }
    assert_eq!(counts.get(&(EOS_ID as i32)), Some(&1));
    assert_eq!(counts.get(&(BOS_ID as i32)), None);
    assert_eq!(counts.values().sum::<usize>(), 21);
    assert!(dir.path.exists());
}

#[test]
fn lm_data_resume_cursor() {
    let dir = TempDir::new("resume");
    let docs = vec![
        document(&dir.path, "a.txt", "a", "validation", "w1"),
        document(&dir.path, "b.txt", "bcdefg", "validation", "w2"),
        document(&dir.path, "c.txt", "hij", "validation", "w3"),
    ];
    let out = dir.path.join("shards");
    write_token_shards(&docs, &tokenizer(), &out, 1000).unwrap();
    let manifest_path = out.join("manifest.lmtxt");

    let mut loader = TokenLoader::open(&manifest_path, 2, 5, 7).unwrap();
    loader.next_batch().unwrap().unwrap();
    let saved = loader.state();
    assert!(
        saved.document.is_some(),
        "expected a mid-document checkpoint"
    );
    assert_eq!(
        saved.manifest_hash,
        read_manifest(&manifest_path).unwrap().manifest_hash
    );

    let mut restored = TokenLoader::open(&manifest_path, 2, 5, 7).unwrap();
    restored.restore(&saved).unwrap();
    assert_eq!(restored.state(), saved);

    for _ in 0..4 {
        let left = loader.next_batch().unwrap();
        let right = restored.next_batch().unwrap();
        assert_eq!(left, right);
        assert_eq!(loader.state(), restored.state());
    }

    let before = restored.state();
    let mut bad = before.clone();
    bad.manifest_hash = "0".repeat(64);
    assert!(restored.restore(&bad).is_err());
    assert_eq!(restored.state(), before);
}

#[test]
fn lm_data_rejects_corruption() {
    let (dir, out) = setup("corrupt", "abcd");
    let manifest = read_manifest(&out.join("manifest.lmtxt")).unwrap();
    let shard = out.join(&manifest.shards[0].file);
    let original = fs::read(&shard).unwrap();

    let mut bytes = original.clone();
    bytes[0] ^= 0xff;
    fs::write(&shard, &bytes).unwrap();
    assert!(TokenLoader::open(&out.join("manifest.lmtxt"), 1, 8, 1).is_err());

    fs::write(&shard, &original).unwrap();
    let file = fs::OpenOptions::new().write(true).open(&shard).unwrap();
    file.set_len(original.len() as u64 - 4).unwrap();
    assert!(TokenLoader::open(&out.join("manifest.lmtxt"), 1, 8, 1).is_err());

    fs::write(&shard, &original).unwrap();
    let tokenizer_path = out.join("tokenizer.lmbt");
    let tokenizer_bytes = fs::read(&tokenizer_path).unwrap();
    let other = ByteBpe::train(&[b"other".to_vec()], 300, NormalizationSpec::default()).unwrap();
    other.save(&tokenizer_path).unwrap();
    assert!(TokenLoader::open(&out.join("manifest.lmtxt"), 1, 8, 1).is_err());
    fs::write(&tokenizer_path, tokenizer_bytes).unwrap();

    fs::write(&shard, &original).unwrap();
    let doc = &manifest.documents[0];
    let token_index = doc.start + doc.len - 1;
    patch_shard_token(&out, &manifest.shards[0].file, token_index, u32::MAX);
    let mut loader = TokenLoader::open(&out.join("manifest.lmtxt"), 1, 8, 1).unwrap();
    assert!(loader.next_batch().is_err());
    assert!(dir.path.exists());
}

fn patch_shard_token(out: &Path, file: &str, token_index: u64, value: u32) {
    let manifest = read_manifest(&out.join("manifest.lmtxt")).unwrap();
    let path = out.join(file);
    let mut bytes = fs::read(&path).unwrap();
    let token_count = manifest.shards[0].token_count as usize;
    let data_offset = bytes.len() - token_count * 4;
    let offset = data_offset + token_index as usize * 4;
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    rewrite_manifest_hash(out, file, &sha256_hex(&bytes));
}

fn rewrite_manifest_hash(out: &Path, file: &str, new_hash: &str) {
    let manifest_path = out.join("manifest.lmtxt");
    let text = fs::read_to_string(&manifest_path).unwrap();
    let mut lines: Vec<String> = text
        .lines()
        .map(|line| line.trim_end_matches('\r').to_owned())
        .filter(|line| !line.is_empty())
        .collect();
    lines.retain(|line| !line.starts_with("manifest_hash "));
    let encoded = hex_encode(file.as_bytes());
    for line in &mut lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 4 && parts[0] == "shard" && parts[1] == encoded.as_str() {
            *line = format!("shard {} {} {}", parts[1], new_hash, parts[3]);
        }
    }
    let mut content = lines.join("\n");
    content.push('\n');
    let manifest_hash = sha256_hex(content.as_bytes());
    content.push_str(&format!("manifest_hash {manifest_hash}\n"));
    fs::write(manifest_path, content).unwrap();
}

#[test]
fn lm_data_split_leakage() {
    let dir = TempDir::new("leak");
    let tokenizer = tokenizer();

    let docs = vec![
        document(&dir.path, "a.txt", "alpha", "train", "same-work"),
        document(&dir.path, "b.txt", "beta", "validation", "same-work"),
    ];
    assert!(write_token_shards(&docs, &tokenizer, &dir.path.join("shards"), 100).is_err());

    let docs = vec![
        document(&dir.path, "c.txt", "é", "train", "w1"),
        document(&dir.path, "d.txt", "e\u{301}", "validation", "w2"),
    ];
    assert!(write_token_shards(&docs, &tokenizer, &dir.path.join("shards2"), 100).is_err());

    let docs = vec![
        document(&dir.path, "e.txt", "same", "train", "shared"),
        document(&dir.path, "f.txt", "other", "train", "shared"),
    ];
    assert!(write_token_shards(&docs, &tokenizer, &dir.path.join("shards3"), 100).is_ok());

    let docs = vec![
        document(&dir.path, "g.txt", "one", "train", "w3"),
        document(&dir.path, "h.txt", "two", "validation", "w4"),
    ];
    assert!(write_token_shards(&docs, &tokenizer, &dir.path.join("shards4"), 100).is_err());
}

#[test]
fn lm_data_empty_corpus() {
    let dir = TempDir::new("empty");
    let tokenizer = tokenizer();
    assert!(write_token_shards(&[], &tokenizer, &dir.path.join("none"), 100).is_err());

    let docs = vec![document(&dir.path, "empty.txt", "", "validation", "w1")];
    let out = dir.path.join("shards");
    let manifest = write_token_shards(&docs, &tokenizer, &out, 100).unwrap();
    assert_eq!(manifest.documents[0].len, 2);

    let mut loader = TokenLoader::open(&out.join("manifest.lmtxt"), 1, 4, 1).unwrap();
    let batch = loader.next_batch().unwrap().unwrap();
    assert_eq!(batch.input_ids, vec![BOS_ID, PAD_ID, PAD_ID, PAD_ID]);
    assert_eq!(batch.targets, vec![EOS_ID as i32, -100, -100, -100]);
    assert_eq!(batch.valid, vec![true, false, false, false]);
}
