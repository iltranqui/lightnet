//! Reproducible byte-fallback BPE with text normalization.
//!
//! The base alphabet is fixed: four control IDs, then the 256 byte values, then
//! learned merges.  Encoding is defined on normalized UTF-8 bytes, so every
//! input string is representable and round-trips exactly.
//!
//! ```text
//! 0 PAD | 1 BOS | 2 EOS | 3 reserved UNK
//! 4..=259  byte values 0..=255
//! 260..    learned merges, in rank order
//! ```

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
/// ID of byte value `0`; byte `b` maps to `BYTE_BASE + b`.
pub const BYTE_BASE: u32 = 4;
/// ID of the first learned merge, `merge index 0`.
pub const FIRST_MERGE_ID: u32 = 260;
/// Base vocabulary size before any merge.
pub const BASE_VOCAB_SIZE: usize = FIRST_MERGE_ID as usize;

/// Artifact format version written by [`ByteBpe::to_artifact`].
pub const ARTIFACT_VERSION: u32 = 1;
const ARTIFACT_HEADER: &str = "lightnet-bpe";

/// Text policy recorded in the artifact and applied by [`normalize_text`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NormalizationSpec {
    /// Expand `æ`/`œ` ligatures and fold `v`/`j` to `u`/`i` (case preserved).
    pub latin_fold: bool,
}

/// Normalize line endings, optionally fold Latin orthography, then apply NFC.
///
/// Folding runs before NFC so `v` + combining macron recomposes to `u` +
/// macron.  Macrons themselves are never removed.
pub fn normalize_text(text: &str, spec: &NormalizationSpec) -> String {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let folded = if spec.latin_fold {
        let mut out = String::with_capacity(text.len());
        for ch in text.chars() {
            match ch {
                'æ' => out.push_str("ae"),
                'Æ' => out.push_str("AE"),
                'œ' => out.push_str("oe"),
                'Œ' => out.push_str("OE"),
                'v' => out.push('u'),
                'V' => out.push('U'),
                'j' => out.push('i'),
                'J' => out.push('I'),
                other => out.push(other),
            }
        }
        out
    } else {
        text
    };
    folded.as_str().nfc().collect()
}

/// A byte-fallback BPE tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteBpe {
    normalization: NormalizationSpec,
    requested_vocab_size: usize,
    merges: Vec<(u32, u32)>,
    token_bytes: Vec<Vec<u8>>,
    ranks: HashMap<(u32, u32), u32>,
}

impl ByteBpe {
    /// Learn merges over the documents.
    ///
    /// Each document must be UTF-8; it is normalized with `normalization` first.
    /// The highest-frequency adjacent pair is merged, ties breaking toward the
    /// lowest pair of IDs, and training stops when the requested vocabulary is
    /// reached or when no pair occurs at least twice.  `vocab_size()` therefore
    /// reports the achieved size, which may be smaller than the requested one.
    pub fn train(
        documents: &[Vec<u8>],
        vocab_size: usize,
        normalization: NormalizationSpec,
    ) -> Result<Self> {
        if vocab_size < BASE_VOCAB_SIZE {
            return Err(Error::InvalidConfig(format!(
                "vocabulary size {vocab_size} is below the base size {BASE_VOCAB_SIZE}"
            )));
        }
        if vocab_size > u32::MAX as usize {
            return Err(Error::InvalidConfig(format!(
                "vocabulary size {vocab_size} does not fit in u32 token IDs"
            )));
        }

        let mut token_bytes: Vec<Vec<u8>> = vec![Vec::new(); BASE_VOCAB_SIZE];
        for byte in 0..=255u32 {
            token_bytes[(BYTE_BASE + byte) as usize] = vec![byte as u8];
        }

        let mut documents_owned: Vec<Vec<u32>> = Vec::with_capacity(documents.len());
        for (index, document) in documents.iter().enumerate() {
            let text = std::str::from_utf8(document).map_err(|_| {
                Error::Parse(format!(
                    "tokenizer training document {index} is not valid UTF-8"
                ))
            })?;
            let normalized = normalize_text(text, &normalization);
            documents_owned.push(
                normalized
                    .bytes()
                    .map(|byte| BYTE_BASE + u32::from(byte))
                    .collect(),
            );
        }

        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();
        while BASE_VOCAB_SIZE + merges.len() < vocab_size {
            let Some((pair, _)) = best_pair(&documents_owned) else {
                break;
            };
            let new_id = u32::try_from(FIRST_MERGE_ID as usize + merges.len())
                .map_err(|_| Error::InvalidConfig("merge id overflows u32".to_owned()))?;
            let mut bytes = token_bytes[pair.0 as usize].clone();
            bytes.extend_from_slice(&token_bytes[pair.1 as usize]);
            token_bytes.push(bytes);
            ranks.insert(
                pair,
                u32::try_from(merges.len())
                    .map_err(|_| Error::InvalidConfig("merge rank overflows u32".to_owned()))?,
            );
            merges.push(pair);
            for document in &mut documents_owned {
                *document = merge_pair(document, pair, new_id);
            }
        }

        Ok(Self {
            normalization,
            requested_vocab_size: vocab_size,
            merges,
            token_bytes,
            ranks,
        })
    }

    /// Normalize and encode text.  Special-token spellings are plain text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = normalize_text(text, &self.normalization);
        Ok(self.encode_bytes(normalized.as_bytes()))
    }

    /// Encode arbitrary bytes with the learned merge table.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let length = bytes.len();
        let mut symbols: Vec<u32> = bytes
            .iter()
            .map(|&byte| BYTE_BASE + u32::from(byte))
            .collect();
        let mut prev: Vec<Option<usize>> = (0..length).map(|index| index.checked_sub(1)).collect();
        let mut next: Vec<Option<usize>> = (0..length)
            .map(|index| (index + 1 < length).then_some(index + 1))
            .collect();
        let mut alive = vec![true; length];
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        for index in 0..length.saturating_sub(1) {
            if let Some(rank) = self.ranks.get(&(symbols[index], symbols[index + 1])) {
                heap.push(Reverse((*rank, index)));
            }
        }

        while let Some(Reverse((rank, index))) = heap.pop() {
            if !alive[index] {
                continue;
            }
            let Some(right) = next[index] else {
                continue;
            };
            if !alive[right] {
                continue;
            }
            let pair = self.merges[rank as usize];
            if symbols[index] != pair.0 || symbols[right] != pair.1 {
                continue;
            }

            symbols[index] = FIRST_MERGE_ID + rank;
            alive[right] = false;
            let after = next[right];
            next[index] = after;
            if let Some(after) = after {
                prev[after] = Some(index);
            }
            if let Some((rank, before)) =
                prev[index]
                    .filter(|&before| alive[before])
                    .and_then(|before| {
                        self.ranks
                            .get(&(symbols[before], symbols[index]))
                            .map(|rank| (*rank, before))
                    })
            {
                heap.push(Reverse((rank, before)));
            }
            if let Some(rank) = after
                .filter(|&after| alive[after])
                .and_then(|after| self.ranks.get(&(symbols[index], symbols[after])).copied())
            {
                heap.push(Reverse((rank, index)));
            }
        }

        let mut output = Vec::new();
        let mut cursor = Some(0);
        while let Some(index) = cursor {
            if alive[index] {
                output.push(symbols[index]);
            }
            cursor = next[index];
        }
        output
    }

    /// Expand token IDs back to bytes.  Control IDs `0..=3` are not text.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for &id in ids {
            if id < BYTE_BASE {
                return Err(Error::Parse(format!(
                    "token id {id} is a control token, not text"
                )));
            }
            let bytes = self.token_bytes.get(id as usize).ok_or_else(|| {
                Error::Parse(format!(
                    "token id {id} is outside the vocabulary ({})",
                    self.token_bytes.len()
                ))
            })?;
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }

    /// Requested vocabulary size passed to [`ByteBpe::train`].
    pub fn requested_vocab_size(&self) -> usize {
        self.requested_vocab_size
    }

    /// Achieved vocabulary size, `260 + merge_count`.
    pub fn vocab_size(&self) -> usize {
        BASE_VOCAB_SIZE + self.merges.len()
    }

    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    pub fn normalization(&self) -> NormalizationSpec {
        self.normalization
    }

    /// Canonical text serialization of the artifact.
    pub fn to_artifact(&self) -> String {
        let mut out = String::new();
        out.push_str(ARTIFACT_HEADER);
        out.push('\n');
        out.push_str("version ");
        out.push_str(&ARTIFACT_VERSION.to_string());
        out.push('\n');
        out.push_str("latin_fold ");
        out.push(if self.normalization.latin_fold {
            '1'
        } else {
            '0'
        });
        out.push('\n');
        out.push_str("requested_vocab_size ");
        out.push_str(&self.requested_vocab_size.to_string());
        out.push('\n');
        out.push_str("merges ");
        out.push_str(&self.merges.len().to_string());
        out.push('\n');
        for (index, (left, right)) in self.merges.iter().enumerate() {
            out.push_str(&(FIRST_MERGE_ID as usize + index).to_string());
            out.push(' ');
            out.push_str(&left.to_string());
            out.push(' ');
            out.push_str(&right.to_string());
            out.push('\n');
        }
        out
    }

    /// Parse and validate a canonical artifact.
    pub fn from_artifact(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| Error::Parse("tokenizer artifact is empty".to_owned()))?;
        if header != ARTIFACT_HEADER {
            return Err(Error::Parse(format!("not a {ARTIFACT_HEADER} artifact")));
        }

        let version = parse_u32(field(lines.next(), "version")?, "version")?;
        if version != ARTIFACT_VERSION {
            return Err(Error::Parse(format!(
                "unsupported tokenizer artifact version {version}"
            )));
        }
        let latin_fold = match field(lines.next(), "latin_fold")? {
            "0" => false,
            "1" => true,
            other => {
                return Err(Error::Parse(format!(
                    "expected 0 or 1 for latin_fold, found \"{other}\""
                )));
            }
        };
        let requested_vocab_size = parse_usize(
            field(lines.next(), "requested_vocab_size")?,
            "requested_vocab_size",
        )?;
        if requested_vocab_size < BASE_VOCAB_SIZE {
            return Err(Error::Parse(format!(
                "requested_vocab_size {requested_vocab_size} is below {BASE_VOCAB_SIZE}"
            )));
        }
        let merge_count = parse_usize(field(lines.next(), "merges")?, "merges")?;
        let required_vocab = BASE_VOCAB_SIZE
            .checked_add(merge_count)
            .ok_or_else(|| Error::Parse("merge count overflows usize".to_owned()))?;
        if requested_vocab_size < required_vocab {
            return Err(Error::Parse(format!(
                "merge count {merge_count} exceeds requested_vocab_size {requested_vocab_size}"
            )));
        }

        let mut merges: Vec<(u32, u32)> = Vec::new();
        for index in 0..merge_count {
            let line = lines
                .next()
                .ok_or_else(|| Error::Parse(format!("artifact ends before merge {index}")))?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 3 {
                return Err(Error::Parse(format!(
                    "merge {index} must have three fields, found {}",
                    fields.len()
                )));
            }
            let id = parse_u32(fields[0], "merge id")?;
            let left = parse_u32(fields[1], "merge left")?;
            let right = parse_u32(fields[2], "merge right")?;
            let expected = u32::try_from(FIRST_MERGE_ID as usize + index)
                .map_err(|_| Error::Parse("merge id overflows u32".to_owned()))?;
            if id != expected {
                return Err(Error::Parse(format!(
                    "merge {index} has id {id}, expected {expected} (duplicate, reordered or foreign id)"
                )));
            }
            if left < BYTE_BASE || right < BYTE_BASE {
                return Err(Error::Parse(format!(
                    "merge {index} references a control token"
                )));
            }
            if left >= id || right >= id {
                return Err(Error::Parse(format!(
                    "merge {index} has a forward reference ({left}, {right}) to id {id}"
                )));
            }
            merges.push((left, right));
        }
        if let Some(extra) = lines.find(|line| !line.trim().is_empty()) {
            return Err(Error::Parse(format!(
                "trailing content after merges: \"{}\"",
                extra.trim()
            )));
        }

        let mut token_bytes: Vec<Vec<u8>> = vec![Vec::new(); BASE_VOCAB_SIZE];
        for byte in 0..=255u32 {
            token_bytes[(BYTE_BASE + byte) as usize] = vec![byte as u8];
        }
        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();
        for (index, (left, right)) in merges.iter().enumerate() {
            if ranks.contains_key(&(*left, *right)) {
                return Err(Error::Parse(format!(
                    "merge {index} repeats pair ({left}, {right})"
                )));
            }
            let mut bytes = token_bytes[*left as usize].clone();
            bytes.extend_from_slice(&token_bytes[*right as usize]);
            token_bytes.push(bytes);
            ranks.insert(
                (*left, *right),
                u32::try_from(index)
                    .map_err(|_| Error::Parse("merge rank overflows u32".to_owned()))?,
            );
        }

        Ok(Self {
            normalization: NormalizationSpec { latin_fold },
            requested_vocab_size,
            merges,
            token_bytes,
            ranks,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        fs::write(path, self.to_artifact())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_artifact(&text)
    }

    /// SHA-256 of the canonical artifact, lowercase hex.
    pub fn artifact_hash(&self) -> String {
        let digest = Sha256::digest(self.to_artifact().as_bytes());
        let mut hex = String::with_capacity(digest.len() * 2);
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        for &byte in digest.iter() {
            hex.push(DIGITS[(byte >> 4) as usize] as char);
            hex.push(DIGITS[(byte & 0x0f) as usize] as char);
        }
        hex
    }
}

fn merge_pair(document: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut output = Vec::with_capacity(document.len());
    let mut index = 0;
    while index < document.len() {
        if index + 1 < document.len() && document[index] == pair.0 && document[index + 1] == pair.1
        {
            output.push(new_id);
            index += 2;
        } else {
            output.push(document[index]);
            index += 1;
        }
    }
    output
}

/// Highest-frequency adjacent pair occurring at least twice.  `BTreeMap`
/// iteration order makes ties break toward the lowest `(left, right)` pair.
fn best_pair(documents: &[Vec<u32>]) -> Option<((u32, u32), u64)> {
    let mut counts: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for document in documents {
        for window in document.windows(2) {
            *counts.entry((window[0], window[1])).or_insert(0) += 1;
        }
    }
    let mut best: Option<((u32, u32), u64)> = None;
    for (&pair, &count) in &counts {
        if count < 2 {
            continue;
        }
        match best {
            Some((_, best_count)) if count <= best_count => {}
            _ => best = Some((pair, count)),
        }
    }
    best
}

fn field<'a>(line: Option<&'a str>, key: &str) -> Result<&'a str> {
    let line = line.ok_or_else(|| Error::Parse(format!("artifact is missing \"{key}\"")))?;
    let (found, value) = line
        .split_once(' ')
        .ok_or_else(|| Error::Parse(format!("artifact line for \"{key}\" is malformed")))?;
    if found != key {
        return Err(Error::Parse(format!(
            "expected \"{key}\", found \"{found}\""
        )));
    }
    Ok(value.trim())
}

fn parse_u32(value: &str, name: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|_| Error::Parse(format!("invalid {name} \"{value}\"")))
}

fn parse_usize(value: &str, name: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| Error::Parse(format!("invalid {name} \"{value}\"")))
}
