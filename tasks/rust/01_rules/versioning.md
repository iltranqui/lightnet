# Versioning

Version lines, commit messages, tags and the docs updated per version.

## Table of Contents

- [Versioning](versioning.md)

## Versioning

- For every commit, including refactors and docs-only commits, increment the patch version in the root `Cargo.toml` `[workspace.package]` and include the corresponding `Cargo.lock` changes in that commit. Workspace crates inherit this version; do not change independently versioned crates unless the work requires it. Keep this Cargo version separate from the `v0.M.NNN` release-line numbering below.
- Version lines: `v0.0.NNN` is closed at `v0.0.274`. New work uses minor lines, each numbered from 1 in three digits:
  - `v0.1.NNN`: Stage 1, CPU parity of all layers (missing layers, axis alignment, CPU Muon). Incomplete; latest `v0.1.044`, so the next free number is `v0.1.045`.
  - `v0.2.NNN`: Stage 2, all GPU paths complete (native kernels for every layer, GPU Muon). Active under the GPU-priority override; latest `v0.2.202`, so the next free number is `v0.2.203`.
  - `v0.3.NNN`: sparse MoE, CPU and GPU (`MOE.md`). Unstarted; default sequencing starts it only when `v0.2` is complete.
  Next version = highest number in `version-map.tsv` + 1; never reuse a number.
- Code commits are `v0.M.NNN: <what>`. Docs-only commits are `docs(rust): <what>` and take no number.
- Per version, append a `version-map.tsv` row (`idx<TAB>version<TAB>old_sha<TAB>new_sha`; `old_sha` is `-` when there is no pre-rewrite sha; `new_sha` is the full commit sha).
- Per version, also add one line with evidence to `implemented.md`, an entry to `STATUS.md` "Latest increments", and tick `todo.md`.
- Gap: `v0.0.248` (`2941a2ee`, CUDA toolkit path in `.cargo/config.toml`) was out of plan. Gaps in numbering are allowed; never reuse a number.
- Version commits are `git add`ed by path. Tracked files commit normally; local-only files in `tasks/` need `git add -f`.
- **Commit and tag on green.** This is a standing authorization for every agent (Claude or Codex) working here; do not ask first. It applies when a new feature is implemented and the tests that cover it were run and passed. Then, in this order:
  1. Commit the code files the task names (by path, never `git add -A`) with the message `v0.M.NNN: <what>`, using the next free number from `version-map.tsv`. Put the tests that were run and their result in the commit body. Comment-only source updates may go with the docs commit.
  2. Tag that code commit with the version, `git tag v0.M.NNN <sha>` (lightweight, local). Never reuse, move or delete a tag; if the tag already exists, stop and re-check the number.
  3. Commit the docs (`implemented.md`, `STATUS.md`, `todo.md`, `layer-support.md`, `gpu_layers_plan.md`, and `version-map.tsv`) as `docs(rust): <what>` after the tag. The version-map row uses `git rev-parse v0.M.NNN` as `new_sha`.
  Do not commit or tag when any covered test fails, times out (a `cfg_corpus` timeout is neither pass nor fail), or was not run; report it instead. Docs-only changes are committed as `docs(rust): <what>` with no tag. Commits and tags stay local: pushing, pull requests and history rewrites still need the user's approval.

