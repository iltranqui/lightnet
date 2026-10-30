//! Fit a bounded train-only BPE sample and prepare native pretraining shards.
//! Enabled by the `lightnet-examples/llm` feature; see tools/latin_pretraining.md.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use lightnet_lm::{ByteBpe, CorpusDocument, NormalizationSpec, TokenLoader, write_token_shards};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn read_documents(path: &Path) -> Result<Vec<CorpusDocument>> {
    let mut result = Vec::new();
    let base = path.parent().ok_or("missing parent directory")?;
    let contents = fs::read_to_string(path)?;
    let mut lines = contents.lines();
    if lines.next() != Some("id\twork_id\tsplit\tsource\tlicense\tsha256\ttext_path") {
        return Err(format!("{}: unexpected document header", path.display()).into());
    }
    for line in lines {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 7 {
            return Err(format!("{}: expected seven fields", path.display()).into());
        }
        if f.iter().any(|field| field.is_empty()) {
            return Err(format!("{}: empty document field", path.display()).into());
        }
        result.push(CorpusDocument {
            id: f[0].into(), work_id: f[1].into(), split: f[2].into(),
            source: f[3].into(), license: f[4].into(), sha256: f[5].into(),
            text_path: base.join(f[6]),
        });
    }
    Ok(result)
}

fn verified_text(doc: &CorpusDocument) -> Result<String> {
    let text = fs::read_to_string(&doc.text_path)?;
    let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
    if hash != doc.sha256 {
        return Err(format!("{}: SHA-256 mismatch", doc.id).into());
    }
    Ok(text)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: llm_prepare_corpus <prepared-directory>".into());
    }
    let root = Path::new(&args[1]);
    if fs::read_to_string(root.join("PREPARATION_COMPLETE"))? != "latin-preparation-v1\n" {
        return Err("extraction is incomplete or has an unknown version".into());
    }
    let docs = read_documents(&root.join("documents.tsv"))?;
    let sample_docs = read_documents(&root.join("tokenizer-sample.tsv"))?;
    let mut works = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    let mut ids = BTreeSet::new();
    let mut authors = BTreeMap::new();
    let mut train = BTreeMap::new();
    for doc in &docs {
        if !ids.insert(&doc.id) {
            return Err(format!("duplicate document {}", doc.id).into());
        }
        for (map, key) in [(&mut works, &doc.work_id), (&mut hashes, &doc.sha256)] {
            if let Some(previous) = map.insert(key, &doc.split) {
                if previous != &doc.split {
                    return Err(format!("cross-split work/text: {key}").into());
                }
            }
        }
        if !doc.split.starts_with("diagnostic_") {
            let author = doc.work_id.rsplit(':').next().ok_or("bad CTS work")?
                .split('.').next().ok_or("bad CTS author")?;
            if let Some(previous) = authors.insert(author, &doc.split) {
                if previous != &doc.split {
                    return Err(format!("cross-split author: {author}").into());
                }
            }
        }
        if doc.split == "train" {
            train.insert(doc.id.as_str(), doc);
        }
    }
    if sample_docs.is_empty() || train.is_empty() {
        return Err("empty training corpus or tokenizer sample".into());
    }
    let mut sample = Vec::new();
    for doc in &sample_docs {
        if doc.split != "train" {
            return Err("heldout document in tokenizer sample".into());
        }
        let parent_id = doc.id.strip_prefix("sample:").ok_or("missing sample parent")?;
        let parent = train.get(parent_id).ok_or("sample parent not in training corpus")?;
        let text = verified_text(doc)?;
        if parent.work_id != doc.work_id || !verified_text(parent)?.starts_with(&text) {
            return Err("sample is not a prefix of its declared training work".into());
        }
        sample.push(text.into_bytes());
    }
    let sample_bytes: usize = sample.iter().map(Vec::len).sum();
    if sample_bytes > 512 * 1024 || sample.iter().any(|s| s.len() > 4096) {
        return Err("tokenizer sample exceeds documented bounds".into());
    }
    let artifact = root.join("tokenizer-8192.lmbt");
    let started = Instant::now();
    let tokenizer = if artifact.exists() {
        // Safe resume: pin artifact to the exact sample TSV and sample contents.
        let expected = format!("{:x}", Sha256::digest(fs::read(root.join("tokenizer-sample.tsv"))?));
        if fs::read_to_string(root.join("tokenizer-sample.sha256"))?.trim() != expected {
            return Err("tokenizer sample changed; use a fresh output directory".into());
        }
        let tok = ByteBpe::load(&artifact)?;
        if fs::read_to_string(root.join("tokenizer-artifact.sha256"))?.trim() != tok.artifact_hash() {
            return Err("cached tokenizer artifact changed".into());
        }
        tok
    } else {
        eprintln!("BPE fit: {} train-only samples, {sample_bytes} bytes, target 8192", sample.len());
        let tok = ByteBpe::train(&sample, 8192, NormalizationSpec::default())?;
        tok.save(&artifact)?;
        fs::write(root.join("tokenizer-sample.sha256"), format!("{:x}\n", Sha256::digest(fs::read(root.join("tokenizer-sample.tsv"))?)))?;
        fs::write(root.join("tokenizer-artifact.sha256"), format!("{}\n", tok.artifact_hash()))?;
        tok
    };
    if tokenizer.requested_vocab_size() != 8192 || tokenizer.normalization().latin_fold {
        return Err("unexpected tokenizer settings".into());
    }
    eprintln!("BPE ready: vocab={} hash={} elapsed={:.1}s", tokenizer.vocab_size(), tokenizer.artifact_hash(), started.elapsed().as_secs_f64());
    let mut statistics = BufWriter::new(fs::File::create(root.join("tokenization.tsv"))?);
    writeln!(statistics, "split\tdocuments\ttext_bytes\ttokens_including_bos_eos\tshards\tmanifest_hash")?;
    let splits: BTreeSet<_> = docs.iter().map(|d| d.split.clone()).collect();
    for split in splits {
        let group: Vec<_> = docs.iter().filter(|d| d.split == split).cloned().collect();
        let mut bytes = 0;
        for doc in &group {
            let text = verified_text(doc)?;
            bytes += text.len();
            let tokens = tokenizer.encode(&text)?;
            if tokenizer.decode_bytes(&tokens)? != text.as_bytes() {
                return Err(format!("round-trip failed for {}", doc.id).into());
            }
        }
        eprintln!("write {split}: {} documents, {bytes} bytes; all tokenizer round-trips passed", group.len());
        let dir = root.join("shards").join(&split);
        let manifest = write_token_shards(&group, &tokenizer, &dir, 1_000_000)?;
        for context in [512, 1024] {
            let mut loader = TokenLoader::open(&dir.join("manifest.lmtxt"), 2, context, 0x5eed)?;
            let batch = loader.next_batch()?.ok_or("empty loader batch")?;
            if !batch.valid.iter().any(|&v| v) || batch.input_ids.iter().any(|&t| t as usize >= tokenizer.vocab_size()) {
                return Err("invalid loader batch".into());
            }
        }
        let tokens: u64 = manifest.shards.iter().map(|s| s.token_count).sum();
        writeln!(statistics, "{split}\t{}\t{bytes}\t{tokens}\t{}\t{}", group.len(), manifest.shards.len(), manifest.manifest_hash)?;
        statistics.flush()?;
        eprintln!("verified {split}: {tokens} tokens; loader contexts 512/1024 passed");
    }
    fs::write(root.join("SHARDS_COMPLETE"), format!("{}\n", tokenizer.artifact_hash()))?;
    eprintln!("pretraining data ready; no model training ran");
    Ok(())
}
