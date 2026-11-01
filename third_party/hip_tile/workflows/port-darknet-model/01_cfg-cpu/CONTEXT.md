# 01_cfg-cpu — the model runs correctly on the CPU

One job: `hip_darknet` loads the model and its CPU path detects what the labels say.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): the model's `.cfg`, `.weights`, `.names`, validation list and label files (paths in `runs/<model>/01_cfg-cpu.md`).
- Reference (every run): `_shared/rules.md`, `_shared/darknet-semantics.md`, `crates/hip_darknet/CONTEXT.md`.
- Porting source (read only): `~/src/DarkSuite/darknet/src-lib/` for any layer not yet in `darknet-semantics.md`.

Do NOT load: `kernels/`, GPU code, `/usr/bin/darknet` (never run it).

## Process

1. List the cfg's section types and keys. Every layer type not yet supported by `hip_darknet::cfg` is new work.
2. For each new layer: read its C++ forward, add its semantics to `_shared/darknet-semantics.md` with file:line, implement it in `cfg.rs`/`model.rs`/`reference.rs`, and add a hand-computed unit test.
3. Check the weights load consumes the file exactly to EOF.
4. Run `cargo run -p hip_darknet --release -- --cfg ... --weights ... --names ... <img>` and the ground-truth test (`tests/detect.rs`, pointed at this model's labelled images).

## Outputs

- Code in `crates/hip_darknet` (+ tests).
- `runs/<model>/01_cfg-cpu.md`: layer inventory, new semantics, unit-test count, ground-truth table (gt/det/matched/recall/precision/class errors), CPU ms/frame.

## Human check

Look at the ground-truth table: recall ≥ 0.9, no class errors, and spot-check one image's boxes by eye.
