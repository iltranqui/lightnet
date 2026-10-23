//! Token shards, manifest, deterministic packing, and resumable loading.
//!
//! A shard is a little-endian binary file:
//!
//! ```text
//! "LMSHARD1" | version:u32 | doc_count:u32
//! doc_count * { id_len:u32, id, split_len:u32, split, start:u64, len:u64 }
//! token_count:u64 | token_count * u32
//! ```
//!
//! Documents are never split across shards; `shard_tokens` is a target except
//! for a single longer document, which gets its own shard.  The manifest is a
//! line-based text file whose last line is `manifest_hash <sha256>` over every
//! preceding line.  Text fields are hex-encoded UTF-8 so any source metadata is
//! safe in the line format.
//!
//! Each document becomes `[BOS, body..., EOS]`.  Windows of at most `context`
//! pairs tile the document: window inputs are `[o, o+len)` and targets are
//! `[o+1, o+len+1)`, so every eligible next-token target occurs exactly once
//! and EOS never predicts into the next document.  Windows are packed into
//! rows with a distinct segment per window and positions restarting at zero.
//! The final padding uses PAD, target `-100`, and `valid=false`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{BOS_ID, ByteBpe, EOS_ID, Error, PAD_ID, Result, RngState, TokenBatch, normalize_text};

const SHARD_MAGIC: &[u8; 8] = b"LMSHARD1";
const SHARD_VERSION: u32 = 1;
const MANIFEST_HEADER: &str = "LIGHTNET-LM-MANIFEST 1";
const MANIFEST_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.lmtxt";
const TOKENIZER_FILE: &str = "tokenizer.lmbt";

/// One UTF-8 plaintext export with provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusDocument {
    pub id: String,
    pub work_id: String,
    pub split: String,
    pub source: String,
    pub license: String,
    pub sha256: String,
    pub text_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardInfo {
    pub file: String,
    pub sha256: String,
    pub token_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentEntry {
    pub id: String,
    pub work_id: String,
    pub split: String,
    pub source: String,
    pub license: String,
    pub sha256: String,
    pub shard: u32,
    pub start: u64,
    pub len: u64,
}

/// Parsed (or freshly written) shard manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardManifest {
    pub format_version: u32,
    pub tokenizer_hash: String,
    pub tokenizer_file: String,
    pub split: String,
    pub manifest_hash: String,
    pub shards: Vec<ShardInfo>,
    pub documents: Vec<DocumentEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSlot {
    pub input: u32,
    pub target: i32,
    pub segment: u32,
    pub position: u32,
    pub valid: bool,
}

/// Everything needed to resume exactly at the next batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoaderState {
    pub manifest_hash: String,
    pub rng_state: u64,
    pub epoch: u64,
    pub order: Vec<u32>,
    pub order_cursor: usize,
    pub document: Option<u32>,
    pub window_offset: u64,
    pub pending: Vec<StateSlot>,
    pub pending_fragments: u32,
}

/// Write token shards and a manifest for one split.
///
/// Cross-split leakage is checked before the single-split rule, so a caller
/// that passes all documents gets a leakage error rather than a mixed-split
/// error.
pub fn write_token_shards(
    documents: &[CorpusDocument],
    tokenizer: &ByteBpe,
    out: &Path,
    shard_tokens: usize,
) -> Result<ShardManifest> {
    if documents.is_empty() {
        return Err(Error::InvalidConfig(
            "cannot write a manifest for an empty corpus".to_owned(),
        ));
    }
    if shard_tokens == 0 {
        return Err(Error::InvalidConfig(
            "shard_tokens must be positive".to_owned(),
        ));
    }

    let mut ids = BTreeSet::new();
    let mut work_splits: BTreeMap<String, String> = BTreeMap::new();
    let mut normalized_splits: BTreeMap<String, String> = BTreeMap::new();
    let mut texts = Vec::with_capacity(documents.len());
    let mut text_hashes: BTreeMap<String, String> = BTreeMap::new();
    for document in documents {
        if document.id.is_empty()
            || document.work_id.is_empty()
            || document.split.is_empty()
            || document.source.is_empty()
            || document.license.is_empty()
        {
            return Err(Error::InvalidConfig(format!(
                "document {} has empty provenance metadata",
                document.id
            )));
        }
        if !ids.insert(document.id.clone()) {
            return Err(Error::InvalidConfig(format!(
                "duplicate document id {}",
                document.id
            )));
        }
        let bytes = fs::read(&document.text_path)?;
        let actual = sha256_hex(&bytes);
        if !document.sha256.eq_ignore_ascii_case(&actual) {
            return Err(Error::InvalidConfig(format!(
                "document {} text checksum does not match {}",
                document.id,
                document.text_path.display()
            )));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|err| Error::Parse(format!("document {} is not UTF-8: {err}", document.id)))?;
        let normalized = normalize_text(text, &tokenizer.normalization());
        let normalized_hash = sha256_hex(normalized.as_bytes());
        if let Some(previous) = work_splits.get(&document.work_id) {
            if previous != &document.split {
                return Err(Error::InvalidConfig(format!(
                    "work {} appears in both {previous} and {} splits",
                    document.work_id, document.split
                )));
            }
        } else {
            work_splits.insert(document.work_id.clone(), document.split.clone());
        }
        if let Some(previous) = normalized_splits.get(&normalized_hash) {
            if previous != &document.split {
                return Err(Error::InvalidConfig(format!(
                    "normalized duplicate document {} crosses {previous} and {} splits",
                    document.id, document.split
                )));
            }
        } else {
            normalized_splits.insert(normalized_hash, document.split.clone());
        }
        text_hashes.insert(document.id.clone(), actual);
        texts.push(bytes);
    }

    let split = documents[0].split.clone();
    if let Some(other) = documents.iter().find(|document| document.split != split) {
        return Err(Error::InvalidConfig(format!(
            "one manifest must cover one split; found {split} and {}",
            other.split
        )));
    }

    fs::create_dir_all(out)?;
    tokenizer.save(&out.join(TOKENIZER_FILE))?;

    let mut shard_builds: Vec<ShardBuild> = Vec::new();
    let mut current = ShardBuild::default();
    for (document, bytes) in documents.iter().zip(texts) {
        let text = std::str::from_utf8(&bytes)
            .map_err(|err| Error::Parse(format!("document {} is not UTF-8: {err}", document.id)))?;
        let body = tokenizer.encode(text)?;
        let mut tokens = Vec::with_capacity(body.len() + 2);
        tokens.push(BOS_ID);
        tokens.extend(body);
        tokens.push(EOS_ID);
        if !current.tokens.is_empty()
            && current.tokens.len().saturating_add(tokens.len()) > shard_tokens
        {
            shard_builds.push(std::mem::take(&mut current));
        }
        let start = current.tokens.len() as u64;
        current.tokens.extend_from_slice(&tokens);
        current.documents.push(ShardDocument {
            id: document.id.clone(),
            split: document.split.clone(),
            start,
            len: tokens.len() as u64,
        });
    }
    if !current.tokens.is_empty() {
        shard_builds.push(current);
    }
    if shard_builds.is_empty() {
        return Err(Error::InvalidConfig(
            "no token shards were produced".to_owned(),
        ));
    }

    let tokenizer_hash = tokenizer.artifact_hash();
    let mut shards = Vec::with_capacity(shard_builds.len());
    let mut entries = Vec::with_capacity(documents.len());
    for (index, build) in shard_builds.iter().enumerate() {
        let bytes = encode_shard(build)?;
        let file = format!("shard-{index:06}.lmt");
        fs::write(out.join(&file), &bytes)?;
        shards.push(ShardInfo {
            file,
            sha256: sha256_hex(&bytes),
            token_count: build.tokens.len() as u64,
        });
        for document in &build.documents {
            let source = documents
                .iter()
                .find(|candidate| candidate.id == document.id)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!("missing metadata for document {}", document.id))
                })?;
            entries.push(DocumentEntry {
                id: document.id.clone(),
                work_id: source.work_id.clone(),
                split: document.split.clone(),
                source: source.source.clone(),
                license: source.license.clone(),
                sha256: text_hashes.get(&source.id).cloned().ok_or_else(|| {
                    Error::InvalidConfig(format!("missing text hash for document {}", source.id))
                })?,
                shard: index as u32,
                start: document.start,
                len: document.len,
            });
        }
    }

    let mut manifest = ShardManifest {
        format_version: MANIFEST_VERSION,
        tokenizer_hash,
        tokenizer_file: TOKENIZER_FILE.to_owned(),
        split,
        manifest_hash: String::new(),
        shards,
        documents: entries,
    };
    let content = manifest_content(&manifest);
    manifest.manifest_hash = sha256_hex(content.as_bytes());
    let mut text = content;
    text.push_str(&format!("manifest_hash {}\n", manifest.manifest_hash));
    fs::write(out.join(MANIFEST_FILE), text)?;
    Ok(manifest)
}

/// Read and validate a manifest (including its self-hash).
pub fn read_manifest(path: &Path) -> Result<ShardManifest> {
    let text = fs::read_to_string(path)?;
    parse_manifest(&text)
}

/// Resumable CPU token loader.
pub struct TokenLoader {
    manifest: ShardManifest,
    batch: usize,
    context: usize,
    vocab_size: usize,
    rng: RngState,
    epoch: u64,
    order: Vec<u32>,
    order_cursor: usize,
    document: Option<u32>,
    window_offset: u64,
    pending: Vec<StateSlot>,
    pending_fragments: u32,
    shards: Vec<ShardIndex>,
}

struct ShardIndex {
    path: PathBuf,
    data_offset: u64,
    token_count: u64,
}

impl TokenLoader {
    /// Open a manifest, verify the tokenizer and every shard, and start at
    /// epoch zero.  Validation splits keep document order; training splits are
    /// shuffled by `seed`.
    pub fn open(manifest_path: &Path, batch: usize, context: usize, seed: u64) -> Result<Self> {
        if batch == 0 || context == 0 {
            return Err(Error::InvalidConfig(
                "loader batch and context must be positive".to_owned(),
            ));
        }
        let manifest = read_manifest(manifest_path)?;
        if manifest.documents.is_empty() || manifest.shards.is_empty() {
            return Err(Error::InvalidConfig(
                "manifest contains no documents or shards".to_owned(),
            ));
        }
        let dir = manifest_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let tokenizer = ByteBpe::load(&dir.join(&manifest.tokenizer_file))?;
        if tokenizer.artifact_hash() != manifest.tokenizer_hash {
            return Err(Error::InvalidConfig(
                "tokenizer artifact hash does not match the manifest".to_owned(),
            ));
        }
        let vocab_size = tokenizer.vocab_size();
        let mut shards = Vec::with_capacity(manifest.shards.len());
        for (index, shard) in manifest.shards.iter().enumerate() {
            shards.push(read_shard_index(&dir, index, shard, &manifest)?);
        }

        let mut rng = RngState::new(seed);
        let order = build_order(&manifest.split, manifest.documents.len(), &mut rng);
        Ok(Self {
            manifest,
            batch,
            context,
            vocab_size,
            rng,
            epoch: 0,
            order,
            order_cursor: 0,
            document: None,
            window_offset: 0,
            pending: Vec::new(),
            pending_fragments: 0,
            shards,
        })
    }

    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>> {
        if self.manifest.documents.is_empty() {
            return Ok(None);
        }
        let mut rows: Vec<Vec<StateSlot>> = Vec::with_capacity(self.batch);
        while rows.len() < self.batch {
            if self.pending.len() == self.context {
                self.flush_row(&mut rows);
                continue;
            }
            match self.peek_fragment()? {
                Some(fragment) => {
                    if self.pending.len() + fragment.len > self.context {
                        self.flush_row(&mut rows);
                        continue;
                    }
                    self.consume(fragment)?;
                    if self.pending.len() == self.context {
                        self.flush_row(&mut rows);
                    }
                }
                None => {
                    self.flush_row(&mut rows);
                    self.start_epoch();
                }
            }
        }
        Ok(Some(self.build_batch(rows)?))
    }

    pub fn manifest(&self) -> &ShardManifest {
        &self.manifest
    }

    pub fn state(&self) -> LoaderState {
        LoaderState {
            manifest_hash: self.manifest.manifest_hash.clone(),
            rng_state: self.rng.state,
            epoch: self.epoch,
            order: self.order.clone(),
            order_cursor: self.order_cursor,
            document: self.document,
            window_offset: self.window_offset,
            pending: self.pending.clone(),
            pending_fragments: self.pending_fragments,
        }
    }

    /// Validate and install a state.  Nothing is changed on error.
    pub fn restore(&mut self, state: &LoaderState) -> Result<()> {
        self.validate_state(state)?;
        self.rng = RngState::new(state.rng_state);
        self.epoch = state.epoch;
        self.order = state.order.clone();
        self.order_cursor = state.order_cursor;
        self.document = state.document;
        self.window_offset = state.window_offset;
        self.pending = state.pending.clone();
        self.pending_fragments = state.pending_fragments;
        Ok(())
    }

    fn validate_state(&self, state: &LoaderState) -> Result<()> {
        if state.manifest_hash != self.manifest.manifest_hash {
            return Err(Error::InvalidConfig(
                "loader state belongs to a different manifest".to_owned(),
            ));
        }
        let count = self.manifest.documents.len();
        if state.order.len() != count {
            return Err(Error::InvalidConfig(
                "loader state order length does not match the manifest".to_owned(),
            ));
        }
        let mut seen = vec![false; count];
        for &index in &state.order {
            let index = index as usize;
            if index >= count || seen[index] {
                return Err(Error::InvalidConfig(
                    "loader state order is not a permutation".to_owned(),
                ));
            }
            seen[index] = true;
        }
        if state.order_cursor > count {
            return Err(Error::InvalidConfig(
                "loader state cursor is past the order".to_owned(),
            ));
        }
        if state.pending.len() > self.context {
            return Err(Error::InvalidConfig(
                "loader state pending row exceeds context".to_owned(),
            ));
        }
        if state.pending.is_empty() {
            if state.pending_fragments != 0 {
                return Err(Error::InvalidConfig(
                    "loader state has fragments without pending slots".to_owned(),
                ));
            }
        } else {
            if state.pending_fragments == 0 {
                return Err(Error::InvalidConfig(
                    "loader state has pending slots without fragments".to_owned(),
                ));
            }
            let mut current_segment = 0u32;
            let mut fragments = 1u32;
            for slot in &state.pending {
                if !slot.valid {
                    return Err(Error::InvalidConfig(
                        "loader state pending slot is not valid".to_owned(),
                    ));
                }
                if slot.input as usize >= self.vocab_size
                    || slot.target < 0
                    || slot.target as usize >= self.vocab_size
                {
                    return Err(Error::InvalidConfig(
                        "loader state pending slot has an invalid token".to_owned(),
                    ));
                }
                if slot.position as usize >= self.context {
                    return Err(Error::InvalidConfig(
                        "loader state pending position is out of range".to_owned(),
                    ));
                }
                if slot.segment == current_segment {
                    continue;
                }
                if slot.segment == current_segment + 1 {
                    current_segment += 1;
                    fragments += 1;
                    continue;
                }
                return Err(Error::InvalidConfig(
                    "loader state pending segments are inconsistent".to_owned(),
                ));
            }
            if fragments != state.pending_fragments {
                return Err(Error::InvalidConfig(
                    "loader state pending fragment count is inconsistent".to_owned(),
                ));
            }
        }
        match state.document {
            Some(document) => {
                let document = document as usize;
                if document >= count || state.order_cursor == 0 {
                    return Err(Error::InvalidConfig(
                        "loader state document is out of range".to_owned(),
                    ));
                }
                if state.order[state.order_cursor - 1] as usize != document {
                    return Err(Error::InvalidConfig(
                        "loader state document does not match the order cursor".to_owned(),
                    ));
                }
                let len = self.manifest.documents[document].len;
                if state.window_offset + 1 >= len {
                    return Err(Error::InvalidConfig(
                        "loader state window offset has no remaining pair".to_owned(),
                    ));
                }
            }
            None => {
                if state.window_offset != 0 {
                    return Err(Error::InvalidConfig(
                        "loader state has an offset without a document".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn peek_fragment(&mut self) -> Result<Option<Fragment>> {
        loop {
            if let Some(document) = self.document {
                let len = self.manifest.documents[document as usize].len;
                if self.window_offset + 1 < len {
                    let remaining = len - 1 - self.window_offset;
                    let length = remaining.min(self.context as u64) as usize;
                    return Ok(Some(Fragment {
                        document,
                        offset: self.window_offset,
                        len: length,
                    }));
                }
                self.document = None;
                self.window_offset = 0;
            }
            if self.order_cursor >= self.order.len() {
                return Ok(None);
            }
            self.document = Some(self.order[self.order_cursor]);
            self.order_cursor += 1;
            self.window_offset = 0;
        }
    }

    fn consume(&mut self, fragment: Fragment) -> Result<()> {
        let document = &self.manifest.documents[fragment.document as usize];
        let absolute = document.start + fragment.offset;
        let tokens = self.read_tokens(document.shard, absolute, fragment.len + 1)?;
        let segment = self.pending_fragments;
        for index in 0..fragment.len {
            let input = tokens[index];
            let target = tokens[index + 1];
            if input as usize >= self.vocab_size || target as usize >= self.vocab_size {
                return Err(Error::InvalidConfig(format!(
                    "shard {} contains token ids outside the tokenizer vocabulary",
                    document.shard
                )));
            }
            self.pending.push(StateSlot {
                input,
                target: i32::try_from(target).map_err(|_| {
                    Error::InvalidConfig("token id does not fit a target".to_owned())
                })?,
                segment,
                position: index as u32,
                valid: true,
            });
        }
        self.pending_fragments += 1;
        self.window_offset += fragment.len as u64;
        if self.window_offset + 1 >= document.len {
            self.document = None;
            self.window_offset = 0;
        }
        Ok(())
    }

    fn flush_row(&mut self, rows: &mut Vec<Vec<StateSlot>>) {
        if self.pending.is_empty() {
            return;
        }
        let mut slots = std::mem::take(&mut self.pending);
        slots.resize(
            self.context,
            StateSlot {
                input: PAD_ID,
                target: -100,
                segment: 0,
                position: 0,
                valid: false,
            },
        );
        rows.push(slots);
        self.pending_fragments = 0;
    }

    fn start_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.order = build_order(
            &self.manifest.split,
            self.manifest.documents.len(),
            &mut self.rng,
        );
        self.order_cursor = 0;
        self.document = None;
        self.window_offset = 0;
    }

    fn build_batch(&self, rows: Vec<Vec<StateSlot>>) -> Result<TokenBatch> {
        let mut input_ids = Vec::with_capacity(rows.len() * self.context);
        let mut targets = Vec::with_capacity(rows.len() * self.context);
        let mut positions = Vec::with_capacity(rows.len() * self.context);
        let mut segments = Vec::with_capacity(rows.len() * self.context);
        let mut valid = Vec::with_capacity(rows.len() * self.context);
        for row in rows {
            for slot in row {
                input_ids.push(slot.input);
                targets.push(slot.target);
                positions.push(slot.position);
                segments.push(slot.segment);
                valid.push(slot.valid);
            }
        }
        Ok(TokenBatch {
            batch: self.batch,
            time: self.context,
            input_ids,
            targets,
            positions,
            segments,
            valid,
        })
    }

    fn read_tokens(&self, shard: u32, start: u64, len: usize) -> Result<Vec<u32>> {
        let index = self
            .shards
            .get(shard as usize)
            .ok_or_else(|| Error::InvalidConfig(format!("shard index {shard} is out of range")))?;
        let end = start
            .checked_add(len as u64)
            .ok_or_else(|| Error::InvalidConfig("token range overflows u64".to_owned()))?;
        if end > index.token_count {
            return Err(Error::InvalidConfig(format!(
                "token range {start}..{end} exceeds shard token count {}",
                index.token_count
            )));
        }
        let byte_offset = start
            .checked_mul(4)
            .and_then(|value| value.checked_add(index.data_offset))
            .ok_or_else(|| Error::InvalidConfig("token byte offset overflows".to_owned()))?;
        let mut file = fs::File::open(&index.path)?;
        file.seek(SeekFrom::Start(byte_offset))?;
        let mut buffer = vec![0u8; len * 4];
        file.read_exact(&mut buffer)?;
        let mut tokens = Vec::with_capacity(len);
        let (words, remainder) = buffer.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        for word in words {
            tokens.push(u32::from_le_bytes(*word));
        }
        Ok(tokens)
    }
}

struct Fragment {
    document: u32,
    offset: u64,
    len: usize,
}

#[derive(Default)]
struct ShardBuild {
    tokens: Vec<u32>,
    documents: Vec<ShardDocument>,
}

struct ShardDocument {
    id: String,
    split: String,
    start: u64,
    len: u64,
}

fn build_order(split: &str, count: usize, rng: &mut RngState) -> Vec<u32> {
    let mut order: Vec<u32> = (0..count as u32).collect();
    if split != "validation" {
        for index in (1..count).rev() {
            let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
            order.swap(index, swap);
        }
    }
    order
}

fn encode_shard(shard: &ShardBuild) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(SHARD_MAGIC);
    bytes.extend_from_slice(&SHARD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(shard.documents.len() as u32).to_le_bytes());
    for document in &shard.documents {
        push_string(&mut bytes, &document.id)?;
        push_string(&mut bytes, &document.split)?;
        bytes.extend_from_slice(&document.start.to_le_bytes());
        bytes.extend_from_slice(&document.len.to_le_bytes());
    }
    bytes.extend_from_slice(&(shard.tokens.len() as u64).to_le_bytes());
    for token in &shard.tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(bytes)
}

fn push_string(bytes: &mut Vec<u8>, value: &str) -> Result<()> {
    let length = u32::try_from(value.len())
        .map_err(|_| Error::InvalidConfig("shard string is too long".to_owned()))?;
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

fn read_shard_index(
    dir: &Path,
    index: usize,
    shard: &ShardInfo,
    manifest: &ShardManifest,
) -> Result<ShardIndex> {
    let path = dir.join(&shard.file);
    let bytes = fs::read(&path)?;
    let digest = sha256_hex(&bytes);
    if !digest.eq_ignore_ascii_case(&shard.sha256) {
        return Err(Error::InvalidConfig(format!(
            "shard {} checksum mismatch",
            shard.file
        )));
    }
    let mut cursor = 0usize;
    if bytes.len() < SHARD_MAGIC.len() + 8 {
        return Err(Error::Parse(format!("shard {} is truncated", shard.file)));
    }
    if &bytes[..SHARD_MAGIC.len()] != SHARD_MAGIC {
        return Err(Error::Parse(format!(
            "shard {} has an unknown magic",
            shard.file
        )));
    }
    cursor += SHARD_MAGIC.len();
    let version = read_u32(&bytes, &mut cursor, &shard.file)?;
    if version != SHARD_VERSION {
        return Err(Error::Parse(format!(
            "shard {} has unsupported version {version}",
            shard.file
        )));
    }
    let doc_count = read_u32(&bytes, &mut cursor, &shard.file)? as usize;
    let mut documents = Vec::with_capacity(doc_count);
    for _ in 0..doc_count {
        let id = read_string(&bytes, &mut cursor, &shard.file)?;
        let split = read_string(&bytes, &mut cursor, &shard.file)?;
        let start = read_u64(&bytes, &mut cursor, &shard.file)?;
        let len = read_u64(&bytes, &mut cursor, &shard.file)?;
        documents.push(ShardDocument {
            id,
            split,
            start,
            len,
        });
    }
    let token_count = read_u64(&bytes, &mut cursor, &shard.file)?;
    if token_count != shard.token_count {
        return Err(Error::InvalidConfig(format!(
            "shard {} token count does not match the manifest",
            shard.file
        )));
    }
    let token_bytes = usize::try_from(token_count)
        .map_err(|_| Error::InvalidConfig("shard token count exceeds usize".to_owned()))?
        .checked_mul(4)
        .ok_or_else(|| Error::InvalidConfig("shard length overflows".to_owned()))?;
    let expected_len = cursor
        .checked_add(token_bytes)
        .ok_or_else(|| Error::InvalidConfig("shard length overflows".to_owned()))?;
    if expected_len != bytes.len() {
        return Err(Error::Parse(format!(
            "shard {} is truncated or has trailing bytes",
            shard.file
        )));
    }

    let expected: Vec<&DocumentEntry> = manifest
        .documents
        .iter()
        .filter(|document| document.shard as usize == index)
        .collect();
    if expected.len() != documents.len() {
        return Err(Error::InvalidConfig(format!(
            "shard {} document table does not match the manifest",
            shard.file
        )));
    }
    let mut next_start = 0u64;
    for (actual, expected) in documents.iter().zip(expected) {
        if actual.id != expected.id
            || actual.split != expected.split
            || actual.start != expected.start
            || actual.len != expected.len
            || actual.start != next_start
        {
            return Err(Error::InvalidConfig(format!(
                "shard {} document table does not match the manifest",
                shard.file
            )));
        }
        next_start = next_start
            .checked_add(actual.len)
            .ok_or_else(|| Error::InvalidConfig("document offsets overflow".to_owned()))?;
    }
    if next_start != token_count {
        return Err(Error::InvalidConfig(format!(
            "shard {} document lengths do not cover the token table",
            shard.file
        )));
    }

    Ok(ShardIndex {
        path,
        data_offset: cursor as u64,
        token_count,
    })
}

fn read_u32(bytes: &[u8], cursor: &mut usize, file: &str) -> Result<u32> {
    if *cursor + 4 > bytes.len() {
        return Err(Error::Parse(format!("shard {file} is truncated")));
    }
    let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().expect("four bytes"));
    *cursor += 4;
    Ok(value)
}

fn read_u64(bytes: &[u8], cursor: &mut usize, file: &str) -> Result<u64> {
    if *cursor + 8 > bytes.len() {
        return Err(Error::Parse(format!("shard {file} is truncated")));
    }
    let value = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().expect("eight bytes"));
    *cursor += 8;
    Ok(value)
}

fn read_string(bytes: &[u8], cursor: &mut usize, file: &str) -> Result<String> {
    let length = read_u32(bytes, cursor, file)? as usize;
    if *cursor + length > bytes.len() {
        return Err(Error::Parse(format!("shard {file} is truncated")));
    }
    let value = std::str::from_utf8(&bytes[*cursor..*cursor + length])
        .map_err(|err| Error::Parse(format!("shard {file} has invalid UTF-8: {err}")))?
        .to_owned();
    *cursor += length;
    Ok(value)
}

fn parse_manifest(text: &str) -> Result<ShardManifest> {
    let lines: Vec<&str> = text
        .lines()
        .map(|line| line.trim_end_matches('\r'))
        .filter(|line| !line.is_empty())
        .collect();
    if lines.first().copied() != Some(MANIFEST_HEADER) {
        return Err(Error::Parse(
            "manifest has an unknown header or version".to_owned(),
        ));
    }
    let hash_index = lines
        .iter()
        .position(|line| line.starts_with("manifest_hash "))
        .ok_or_else(|| Error::Parse("manifest has no manifest_hash line".to_owned()))?;
    if hash_index + 1 != lines.len() {
        return Err(Error::Parse(
            "manifest_hash must be the last line".to_owned(),
        ));
    }
    let mut content = String::new();
    for line in &lines[..hash_index] {
        content.push_str(line);
        content.push('\n');
    }
    let hash_token = lines[hash_index]
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| Error::Parse("manifest_hash has no value".to_owned()))?;
    let manifest_hash = parse_digest(hash_token)?;
    if sha256_hex(content.as_bytes()) != manifest_hash {
        return Err(Error::Parse("manifest checksum mismatch".to_owned()));
    }

    let mut tokenizer_hash = None;
    let mut tokenizer_file = None;
    let mut split = None;
    let mut shard_count = None;
    let mut shards: Vec<ShardInfo> = Vec::new();
    let mut documents: Vec<DocumentEntry> = Vec::new();
    for line in &lines[1..hash_index] {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["tokenizer_hash", value] => {
                if tokenizer_hash.replace(parse_digest(value)?).is_some() {
                    return Err(Error::Parse("duplicate tokenizer_hash".to_owned()));
                }
            }
            ["tokenizer_file", value] => {
                if tokenizer_file.replace(decode_hex(value)?).is_some() {
                    return Err(Error::Parse("duplicate tokenizer_file".to_owned()));
                }
            }
            ["split", value] => {
                if split.replace(decode_hex(value)?).is_some() {
                    return Err(Error::Parse("duplicate split".to_owned()));
                }
            }
            ["shard_count", value] => {
                if shard_count.replace(parse_usize(value)?).is_some() {
                    return Err(Error::Parse("duplicate shard_count".to_owned()));
                }
            }
            ["shard", file, sha, count] => {
                shards.push(ShardInfo {
                    file: decode_hex(file)?,
                    sha256: parse_digest(sha)?,
                    token_count: parse_u64_token(count)?,
                });
            }
            [
                "document",
                id,
                work,
                source,
                license,
                sha,
                split,
                shard,
                start,
                len,
            ] => {
                documents.push(DocumentEntry {
                    id: decode_hex(id)?,
                    work_id: decode_hex(work)?,
                    source: decode_hex(source)?,
                    license: decode_hex(license)?,
                    sha256: parse_digest(sha)?,
                    split: decode_hex(split)?,
                    shard: u32::try_from(parse_usize(shard)?).map_err(|_| {
                        Error::Parse("document shard index overflows u32".to_owned())
                    })?,
                    start: parse_u64_token(start)?,
                    len: parse_u64_token(len)?,
                });
            }
            _ => {
                return Err(Error::Parse(format!(
                    "unknown manifest line: {}",
                    line.split_whitespace().next().unwrap_or("")
                )));
            }
        }
    }

    let tokenizer_hash =
        tokenizer_hash.ok_or_else(|| Error::Parse("manifest has no tokenizer_hash".to_owned()))?;
    let tokenizer_file =
        tokenizer_file.ok_or_else(|| Error::Parse("manifest has no tokenizer_file".to_owned()))?;
    let split = split.ok_or_else(|| Error::Parse("manifest has no split".to_owned()))?;
    let shard_count =
        shard_count.ok_or_else(|| Error::Parse("manifest has no shard_count".to_owned()))?;
    if shard_count != shards.len() {
        return Err(Error::Parse(
            "manifest shard_count does not match its shard lines".to_owned(),
        ));
    }
    if documents.is_empty() || shards.is_empty() {
        return Err(Error::InvalidConfig(
            "manifest contains no documents or shards".to_owned(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut files = BTreeSet::new();
    for shard in &shards {
        if !files.insert(shard.file.clone()) {
            return Err(Error::Parse("duplicate shard file".to_owned()));
        }
        if shard.file.contains('/') || shard.file.contains('\\') || shard.file == ".." {
            return Err(Error::Parse("shard file must be a plain name".to_owned()));
        }
    }
    for document in &documents {
        if !ids.insert(document.id.clone()) {
            return Err(Error::InvalidConfig(format!(
                "duplicate document id {}",
                document.id
            )));
        }
        if document.split != split {
            return Err(Error::InvalidConfig(
                "document split does not match the manifest split".to_owned(),
            ));
        }
        if document.len < 2 {
            return Err(Error::InvalidConfig(format!(
                "document {} is shorter than BOS+EOS",
                document.id
            )));
        }
        let shard = shards.get(document.shard as usize).ok_or_else(|| {
            Error::InvalidConfig("document references an unknown shard".to_owned())
        })?;
        let end = document
            .start
            .checked_add(document.len)
            .ok_or_else(|| Error::InvalidConfig("document offsets overflow".to_owned()))?;
        if end > shard.token_count {
            return Err(Error::InvalidConfig(
                "document range exceeds its shard".to_owned(),
            ));
        }
    }
    if tokenizer_file.contains('/') || tokenizer_file.contains('\\') || tokenizer_file == ".." {
        return Err(Error::Parse(
            "tokenizer file must be a plain name".to_owned(),
        ));
    }

    Ok(ShardManifest {
        format_version: MANIFEST_VERSION,
        tokenizer_hash,
        tokenizer_file,
        split,
        manifest_hash,
        shards,
        documents,
    })
}

fn manifest_content(manifest: &ShardManifest) -> String {
    let mut text = String::new();
    text.push_str(MANIFEST_HEADER);
    text.push('\n');
    text.push_str(&format!("tokenizer_hash {}\n", manifest.tokenizer_hash));
    text.push_str(&format!(
        "tokenizer_file {}\n",
        hex_encode(manifest.tokenizer_file.as_bytes())
    ));
    text.push_str(&format!(
        "split {}\n",
        hex_encode(manifest.split.as_bytes())
    ));
    text.push_str(&format!("shard_count {}\n", manifest.shards.len()));
    for shard in &manifest.shards {
        text.push_str(&format!(
            "shard {} {} {}\n",
            hex_encode(shard.file.as_bytes()),
            shard.sha256,
            shard.token_count
        ));
    }
    for document in &manifest.documents {
        text.push_str(&format!(
            "document {} {} {} {} {} {} {} {} {}\n",
            hex_encode(document.id.as_bytes()),
            hex_encode(document.work_id.as_bytes()),
            hex_encode(document.source.as_bytes()),
            hex_encode(document.license.as_bytes()),
            document.sha256,
            hex_encode(document.split.as_bytes()),
            document.shard,
            document.start,
            document.len
        ));
    }
    text
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(value: &str) -> Result<String> {
    if !value.len().is_multiple_of(2) {
        return Err(Error::Parse("hex field has an odd length".to_owned()));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    for pair in pairs {
        let pair = std::str::from_utf8(pair)
            .map_err(|_| Error::Parse("hex field is not ASCII".to_owned()))?;
        bytes.push(
            u8::from_str_radix(pair, 16)
                .map_err(|_| Error::Parse("hex field has a non-hex digit".to_owned()))?,
        );
    }
    String::from_utf8(bytes).map_err(|_| Error::Parse("hex field is not valid UTF-8".to_owned()))
}

fn parse_digest(value: &str) -> Result<String> {
    let lower = value.to_ascii_lowercase();
    if lower.len() != 64 || !lower.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::Parse(
            "checksum must be 64 hexadecimal digits".to_owned(),
        ));
    }
    Ok(lower)
}

fn parse_usize(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| Error::Parse(format!("invalid integer {value}")))
}

fn parse_u64_token(value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| Error::Parse(format!("invalid integer {value}")))
}
