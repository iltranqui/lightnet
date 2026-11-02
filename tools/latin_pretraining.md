# Latin pretraining data preparation

## Contents

- [Code and feature](#code-and-feature)
- [Run later](#run-later)
- [Preparation policy](#preparation-policy)
- [Outputs and verification](#outputs-and-verification)
- [Current validation](#current-validation)

## Code and feature

`prepare_latin.py` uses Python 3.10+ standard-library XML parsing to extract the downloaded pinned Latin snapshots. `test_prepare_latin.py` covers text filtering, annotated enclitic copies, author split rules, folded audit text and charter regrouping.

The Rust example `src/lightnet-examples/examples/llm_prepare_corpus.rs` is opt-in through the `lightnet-examples` crate's `llm` feature. Its optional dependencies are `lightnet-lm` and `sha2`. The example calls the existing byte-BPE tokenizer, version-1 shard writer and loader. It does not construct or train a language model.

## Run later

Commands below are for a later authorized execution, from the repository root. None were run after the code-only/no-build correction. Use a new output directory: interrupted preparations are deliberately not overwritten. The earlier `datasets/latin/prepared` attempt was stopped and must not be treated as ready.

```sh
systemd-run --user --scope -p MemoryHigh=3G \
  python3 tools/prepare_latin.py --out datasets/latin/pretraining-v1

systemd-run --user --scope -p MemoryHigh=3G \
  cargo run -p lightnet-examples --release -j 4 \
  --features llm --example llm_prepare_corpus -- datasets/latin/pretraining-v1
```

To run just the extraction fixtures later:

```sh
python3 -m unittest discover -s tools -p test_prepare_latin.py
```

## Preparation policy

- Extract surface text, not lemmas or serialized XML. Preserve case, macrons and u/v/i/j. Canonicalize NFC, line endings and whitespace. Strip notes, apparatus readings, stage/speaker labels, translations and duplicated `{word}` enclitic annotations.
- Prefer a single canonical Perseus edition per CTS work. Annotated mirrors and alternate editions are excluded by work ID. Further exclude exact duplicates under lowercase, punctuation-free, orthographically folded comparison text.
- Track explicit component licences; use the Perseus repository default only when no component licence is declared. Additional lascivaroma works without a recognized original-text licence are listed as exclusions. An annotation licence alone is insufficient for inclusion.
- Reserve Seneca for validation and Tacitus for test. Other CTS author IDs are assigned deterministically using SHA-256, with nominal 90/5/5 allocation by author count. This does not guarantee those proportions by bytes or tokens. Unknown authorship is not proof of author independence.
- Rejoin CoNLL-U `# text` sentences into works or charters across published splits, using passage positions. LLCT and CIRCSE are separate diagnostic manifests, never tokenizer-training sources. Seneca/Tacitus diagnostics intentionally overlap the corresponding held-out author domains; diagnostic results are not an independent aggregate test score.
- Audit training against all held-out and diagnostic works using exact five-word shingles in both original and folded views. Quarantine a complete training work when it shares at least 20 shingles and at least 20% of the smaller work's unique shingles. Lower overlaps of at least five shingles are reported for review. This heuristic is an audit, not proof that every near duplicate has been removed; validation-versus-test near-duplicate screening remains a separate review.
- Select tokenizer samples from training only: deterministic work order, round-robin authors, at most 4096 UTF-8 bytes per work and 512 KiB total. This is a bounded pilot sample, not a measured quality-optimal choice. Fit an 8192-ID BPE target with Latin folding disabled; record the achieved vocabulary and artifact hash.

## Outputs and verification

Preparation writes `documents.tsv`, per-split TSVs, UTF-8 `text/`, `tokenizer-sample.tsv`, `sample/`, `tokenizer-sample.json`, `provenance.jsonl`, `excluded.json`, `overlap-audit.json`, `report.json`, and `SHA256SUMS`. Every input to the Rust helper carries the seven fields expected by `lightnet-rs lm`: ID, work ID, split, source, licence, SHA-256, relative text path. Provenance additionally records original XML/CoNLL-U hashes and source revisions. `PREPARATION_COMPLETE` is written after extraction and audits finish; the helper refuses an incomplete extraction.

The Rust example validates global IDs, work/text split disjointness, main-corpus author disjointness, sample parent membership, sample hashes and byte bounds. It writes `tokenizer-8192.lmbt`, sample/artifact hash sidecars, `tokenization.tsv`, and `shards/<split>/manifest.lmtxt` plus native `.lmt` shards. It checks tokenizer round trips for every retained document and opens each manifest with the real loader at contexts 512 and 1024. Shard target size is one million tokens; long works remain intact. `SHARDS_COMPLETE` is written only after all splits pass those checks.

After a run, verify text preparation from its output directory with `sha256sum -c SHA256SUMS`. Native manifests separately authenticate the tokenizer and token shards. Use `shards/train/manifest.lmtxt` for pretraining and `shards/validation/manifest.lmtxt` for validation. Keep test and diagnostic manifests separate.

## Current validation

Five Python extraction fixtures passed before the final no-build instruction. The final Rust example and Cargo feature wiring have not been compiled or executed. No tokenizer fitting or model pretraining ran. Dataset processing was stopped during the five-word overlap audit; there is no completed pretraining dataset from that attempt.
