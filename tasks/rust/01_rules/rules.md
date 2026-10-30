# Rules

Hard rules for every agent working in `tasks/rust/` (branch `rust`). Read before any build, edit or commit.

## Table of Contents

- [Rules](#rules)
- [Doc Hygiene](#doc-hygiene)

## Rules

- Max 4 build threads (`JOBS=4`; repo scripts use `-j 2`). One build/test command at a time, never two in parallel.
- Cap memory: `systemd-run --user --scope -p MemoryHigh=4G <cmd>` (scripts and `cargo xtask` take `MEM=4G`). If no user systemd bus, run uncapped and say so.
- Make code edits and refactors in the main session with shell tools, `Edit` and `Write`: no Python scripts and no subagents unless the user asks.
- When moving layer code, run `JOBS=4 MEM=4G cargo xtask check` after each layer, not only at the end, and `gpu-check` plus the layer's test filter before each commit. Before replacing a line range in a large dispatcher, confirm the line after the range starts the next arm.
- `src/lightnet-cuda-oxide` is not a workspace member: build/test it only via `./tools/test_oxide.sh` (`cargo oxide`).
- Backward = exact derivative of the Rust forward. C++ deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Never claim C++ parity: no compiled C++ exists here. Write "hand-derived from source, no compiled-C++ golden".
- Reference C++ branches (`raptor*`, `Jev_laya`, `fog`, `v6-dev-*`) are readable in sibling repo `/home/kerrigan/src/DarkSuite/darknet`: use `git -C /home/kerrigan/src/DarkSuite/darknet show <branch>:<path>` and `git -C /home/kerrigan/src/DarkSuite/darknet grep` first, since plan paths go stale.
- No training runs unless the user asks.
- Commit only files the task names, never `git add -A`. Message `v0.M.NNN: <what>` (see [Versioning](versioning.md)). Leave the user's `.gitignore` edit alone.
- Commit and tag on green: when an agent (Claude or Codex) has implemented a feature and its named tests pass, it commits and tags it right away, without asking. Details in [Versioning](versioning.md), "Commit and tag on green".
- All files under `tasks/rust/` are tracked (checked 2026-10-27). A new file still needs `git add -f` if `tasks/` is ignored: check with `git check-ignore tasks/rust/<file>`.
- Do not run `cargo xtask fmt`: it reformats unrelated files. Format edited code by hand.
- `cargo xtask oxide` rewrites `src/lightnet-cuda-oxide/Cargo.lock`; revert that file before committing.
- Each edited `.md` needs a TOC.
- Clean up after each completed step: delete `.weights`, `.backup`, `.onnx`, optimizer sidecars, logs, A/B build trees (e.g. `build/base`, `build/cur`), and other big files once nothing needs them. Keep any file a doc or test still references, and record the evidence in the docs before deleting. `build/` is gitignored, so deleting there is safe, but look at the target first; if unsure, ask.


## Doc Hygiene

- Finished work goes in `implemented.md` (one line per version with evidence), not as prose in plan files.
- Plan files keep only open work. Update `todo.md`, `STATUS.md`, `version-map.tsv` per version.
