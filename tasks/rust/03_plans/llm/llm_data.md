# Latin Data and Tokenizer

Data notes for the Latin decoder (decoder Task 2 and recipe R1/R2). The corpus source list lives in [latin_datasets.md](latin_datasets.md); this file records the tokenizer artifact contract and the rules for the tokenizer-training sample. No corpus text is vendored here, and nothing below is a measured corpus statistic.

## Table of Contents

- [Scope](#scope)
- [Tokenizer artifact version 1](#tokenizer-artifact-version-1)
  - [ID layout](#id-layout)
  - [Normalization](#normalization)
  - [Merge learning](#merge-learning)
  - [Artifact format](#artifact-format)
- [Token shards and packing](#token-shards-and-packing)
  - [Shard file](#shard-file)
  - [Manifest](#manifest)
  - [Packing and splits](#packing-and-splits)
- [Encoding and decoding](#encoding-and-decoding)
- [Tokenizer-training sample](#tokenizer-training-sample)
- [Corpus provenance and splits](#corpus-provenance-and-splits)
- [Open items](#open-items)

## Scope

- The tokenizer is `lightnet_lm::tokenizer::ByteBpe`, implemented in `src/lightnet-lm/src/tokenizer.rs` and covered by `src/lightnet-lm/tests/lm_tokenizer.rs`.
- Token shards, packing, and the loader are `src/lightnet-lm/src/data.rs`, covered by `src/lightnet-lm/tests/lm_data.rs`.
- Chat control tokens are recipe task R2. Task 2 reserves IDs 0 to 3 only; the versioned top-of-vocabulary chat range is not implemented yet, so a tokenizer artifact must not assume those IDs exist.

## Tokenizer artifact version 1

### ID layout

| IDs | Meaning |
|---|---|
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3 | reserved (UNK) |
| 4 to 259 | byte values 0 to 255, ID = 4 + byte |
| 260 and up | learned merges, ID = 260 + merge rank |

The base vocabulary is 260 IDs. Learned merges are appended in rank order, so the achieved vocabulary is `260 + merge_count`.

### Normalization

`normalize_text` is part of the artifact identity and is applied before every encode and to every tokenizer-training document:

1. Canonical line endings: `\r\n` and lone `\r` become `\n`.
2. Optional Latin folding, disabled by default:
   - `æ`/`Æ` become `ae`/`AE`, `œ`/`Œ` become `oe`/`OE`.
   - `v`/`V` become `u`/`U`, `j`/`J` become `i`/`I`.
   - Case is otherwise preserved and macrons are never removed.
3. NFC composition.

Folding runs before NFC, so a decomposed macron on a folded letter recomposes (`v` + U+0304 becomes `ū`). Folding is lossy for medieval orthography and is a recorded ablation, not a default: the artifact stores `latin_fold` so a folded tokenizer can never be mistaken for an unfolded one.

### Merge learning

`ByteBpe::train(documents, requested_vocab_size, normalization)`:

- Each document must be valid UTF-8 and is normalized before merge learning.
- Pairs are counted only between adjacent symbols inside one document; a pair is never formed across the boundary between two documents.
- Each iteration merges the highest-frequency adjacent pair that occurs at least twice. Ties break toward the lowest `(left_id, right_id)` pair.
- Training stops at the requested vocabulary size, or earlier when no pair occurs twice. `requested_vocab_size()` and `vocab_size()` therefore differ when the corpus is too small; this is expected, not an error.
- Requesting fewer than 260 IDs is an error. IDs must fit in `u32`.
- Document order does not affect the result, so permuting the training documents yields the same artifact hash.

### Artifact format

Version 1 is line-based text:

```text
lightnet-bpe
version 1
latin_fold 0
requested_vocab_size 300
merges 2
260 4 5
261 260 6
```

Each merge line is `<id> <left> <right>`. Loading validates:

- the header and `version 1` (unknown versions are errors);
- `latin_fold` is 0 or 1;
- `requested_vocab_size` is at least 260 and at least `260 + merges`;
- `merges` matches the number of merge lines;
- merge IDs are exactly `260 + rank` (duplicate, reordered, or foreign IDs are errors);
- every operand is at least 4 (controls are not merge operands) and strictly lower than the merge's own ID (forward references are errors);
- no trailing content.

`artifact_hash()` is the lowercase hexadecimal SHA-256 of the canonical artifact text, so the hash covers the normalization policy, the requested size, and every merge.


## Token shards and packing

`write_token_shards(documents, tokenizer, out, shard_tokens)` writes one split
to `out`: `tokenizer.lmbt` (the canonical BPE artifact), `shard-NNNNNN.lmt`
files, and `manifest.lmtxt`. It verifies each document's declared SHA-256
against `text_path`, requires UTF-8 plaintext, rejects empty provenance fields,
duplicate document IDs, work IDs reused across splits, and normalized exact
duplicates across splits. One manifest covers exactly one split; write one
output directory per split.

### Shard file

All integers are little-endian. A shard is:

```text
"LMSHARD1"
version:u32 = 1
doc_count:u32
doc_count * { id_len:u32, id bytes, split_len:u32, split bytes, start:u64, len:u64 }
token_count:u64
token_count * u32
```

`shard_tokens` is a target, not a hard cap: a single longer document is kept in
one shard rather than split across shards. `start` and `len` locate the
document's tokens in the shard. The loader verifies the magic, version, every
document boundary, every token ID against the tokenizer vocabulary, the token
count, and the shard SHA-256 from the manifest before reading tokens.

### Manifest

The manifest is line-based text whose final line is the checksum. Every text
field is lowercase hex-encoded UTF-8, and every digest is lowercase hex:

```text
LIGHTNET-LM-MANIFEST 1
tokenizer_hash <sha256>
tokenizer_file <hex "tokenizer.lmbt">
split <hex split>
shard_count <n>
shard <hex file> <sha256> <token_count>
document <hex id> <hex work_id> <hex source> <hex license> <sha256> <hex split> <shard> <start> <len>
manifest_hash <sha256>
```

`manifest_hash` is SHA-256 over every preceding line, including its newline.
Loading rejects a missing, duplicated, reordered, or unknown field, an
out-of-order `manifest_hash`, a checksum mismatch, a tokenizer artifact hash
mismatch, and any shard hash/count mismatch.

### Packing and splits

- Each document becomes `[BOS, body..., EOS]`; an empty document is
  `[BOS, EOS]` and is valid.
- A window holds at most `context` next-token pairs. Inputs are
  `[o, o+len)` and targets are `[o+1, o+len+1)`, so every eligible body/EOS
  target occurs exactly once across the windows and EOS never predicts into the
  next document.
- Windows are packed into rows. Each window gets a distinct segment ID, and
  positions restart at zero in each window. Padding uses `PAD_ID`, target
  `-100`, and `valid=false`; padded positions contribute no loss.
- Training splits shuffle documents deterministically from the loader seed
  (SplitMix64/Fisher-Yates); the validation split uses stable document order.
  The epoch advances automatically when the order is exhausted.
- `LoaderState` records the manifest hash, shuffle RNG/epoch/order, document
  cursor, long-document offset, and pending packing fragments. `restore`
  rejects a state from a different manifest or with inconsistent lengths, so a
  resumed run reproduces the next batch.

## Encoding and decoding

- `encode(text)` normalizes with the stored policy, then runs the learned merges.
- `encode_bytes(bytes)` runs the same merges on arbitrary bytes, which is what token-shard preparation uses.
- Merges apply in increasing rank order; new pairs created by a merge always have a higher rank than the merge that created them. The implementation uses a rank/index priority queue and links, and a rank-order test pins the non-overlapping left-to-right result.
- `decode_bytes(ids)` expands merges back to bytes. Control IDs 0 to 3 are not text and return an error; callers strip BOS/EOS/PAD before display.
- Special-token spellings (`<|bos|>`, `<s>`, `</s>`, `[BOS]`, `<pad>`, `<unk>`) are ordinary text. Encoding never emits a control ID, so a literal in a corpus cannot silently become a control token.

## Tokenizer-training sample

Train the tokenizer on the training split only, and never load a whole general corpus to fit it. The intended policy:

- Select an explicit, documented sample of training documents before fitting (for example a bounded number of documents per work and source, capped in total bytes).
- Record the selection rule, the document IDs and hashes used, the normalization policy, the requested vocabulary, the achieved vocabulary, and the artifact hash.
- Do not include held-out or validation documents in the sample. Task 10 checks this split property.
- The sample composition is a proposal until measured; no selection rule has been validated for Latin quality yet.

## Corpus provenance and splits

Every unit of text must carry the metadata the loader and evaluator need: stable document id, work id, split, source, licence, SHA-256 of the original text, and the normalized text path. Whole works are assigned to one split; alternate editions and translations of the same work stay together. Normalized exact duplicates and near-duplicates are audited across splits before training.

Corpus rights are mostly unconfirmed. See [latin_datasets.md](latin_datasets.md) for the candidate sources and their licence status. Do not download, vendor, or train on a source marked unverified or noncommercial without the user's approval.

## Open items

- Chat control tokens and a tokenizer version bump (recipe R2) are not implemented.
- Shards and manifests are version 1 and line/LE based; no migration path is needed until a second format exists.
- No tokenizer-training sample has been selected or measured; vocabulary sizes 8192 and 16384 are plan targets, not validated choices.
- u/v and i/j folding remains an ablation to run, not a recommendation.
- Whether the final corpus can be represented as UTF-8 plaintext (rather than TEI/HTML that needs extraction) is unverified per source.
