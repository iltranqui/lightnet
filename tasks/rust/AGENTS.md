# Rust Port — Agent Handbook

Entry point for any agent working in `tasks/rust/` (branch `rust`). Read this first; open other files only when the map below says to.

## Table of Contents

- [Goal](#goal)
  - [Running this file as a goal](#running-this-file-as-a-goal)
- [Rules](#rules)
- [Layer Layout and Completion](#layer-layout-and-completion)
  - [Layout](#layout)
  - [Layer Completion Contract](#layer-completion-contract)
  - [Layout Move Queue](#layout-move-queue)
- [Commands](#commands)
- [Versioning](#versioning)
- [File Map](#file-map)
- [Current State](#current-state)
- [Open Work](#open-work)
- [Lessons](#lessons)
- [Doc Hygiene](#doc-hygiene)

## Goal

Reach full Rust parity with every C++ layer (branch `rust`), in three release lines (see [Versioning](#versioning)):

1. `v0.1.NNN`: every missing layer (`deform_conv`, `dsconv`, `graph_conv`, `wmhf`, `clifford`, `recursive_block`, `hyperbolic_foveation`, `foveation_film`, `depth_anything`, `dpt_head`, plus `yolo_3d` if `depth_3d` confirms scope) ported on CPU under the Layer Port Contract, with SGD, Adam and Muon arms, and every layer's axis/layout semantics pinned to the C++ source by a named test.
2. `v0.2.NNN`: every layer has a native GPU path with a CPU-parity test, host-fallback layers included, plus GPU Muon.
3. `v0.3.NNN`: opt-in true sparse MoE per `MOE.md`, inference first, never slower than dense.

Both halves of every layer must be complete: the host code (CPU reference) and the device code (native GPU runtime glue plus Oxide kernels). A layer counts as finished only when it meets the [Layer Completion Contract](#layer-completion-contract), and its host and device code live together in one folder ([Layout](#layout)).

Done when all three lines are complete, each version has its tests passing and its docs updated (`implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`), and no parity is claimed without a C++ golden. Default sequencing starts each line after the previous one is complete. **Current operator override: focus on GPU layers now, directly without subagents, while preserving all open CPU requirements.** Follow `gpu_layers_plan.md`; native GPU work is authorized before CPU completion, not evidence that CPU is complete. Details: [Open Work](#open-work).

### Running this file as a goal

Use it as `/goal tasks/rust/AGENTS.md`. When an agent is given this file as its goal, the goal is: make the host and the device code of every layer complete under the [Layer Completion Contract](#layer-completion-contract), in the [Layout](#layout), following [Open Work](#open-work). The agent loops until the stop conditions below hold:

1. Read `lessons.md`, `git status` and `git log -5`, then `STATUS.md` (top) and `layer-support.md` to see the current state.
2. Pick the first unfinished item: the [Layout Move Queue](#layout-move-queue) while it has open items, then the completion queue in [Open Work](#open-work). Work one layer or one slice at a time.
3. Write the change, then run, one command at a time, `JOBS=4 MEM=4G cargo xtask check`, `gpu-check`, the layer's test filter and, for device work, `cargo xtask oxide --test <suite>`.
4. When the covered tests pass, commit and tag as described in [Versioning](#versioning) ("Commit and tag on green") and update the docs in the same pass. A refactor move gets a `refactor(rust): ...` commit with no tag.
5. Go back to step 2.

Stop and report to the user, without continuing, when any of these happens:
- every layer meets the Completion Contract (done), or the next item needs a decision that is not in this file;
- a covered test fails, times out or cannot run (for example no GPU), and the cause is not clear after one root-cause pass;
- an action would be hard to reverse or outward-facing: pushing, pull requests, history rewrites, deleting data, training runs, anything in [Open Work](#open-work) marked as needing the user.

Never report a layer as complete from a primitive, a forward-only path or a passing subset; name the tests that ran and the ones that did not.

## Rules

- Max 4 build threads (`JOBS=4`; repo scripts use `-j 2`). One build/test command at a time, never two in parallel.
- Cap memory: `systemd-run --user --scope -p MemoryHigh=4G <cmd>` (scripts and `cargo xtask` take `MEM=4G`). If no user systemd bus, run uncapped and say so.
- Make code edits and refactors in the main session with shell tools, `Edit` and `Write`: no Python scripts and no subagents unless the user asks.
- When moving layer code, run `JOBS=4 MEM=4G cargo xtask check` after each layer, not only at the end, and `gpu-check` plus the layer's test filter before each commit. Before replacing a line range in a large dispatcher, confirm the line after the range starts the next arm.
- `src/lightnet-cuda-oxide` is not a workspace member: build/test it only via `./tools/test_oxide.sh` (`cargo oxide`).
- Backward = exact derivative of the Rust forward. C++ deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Never claim C++ parity: no compiled C++ exists here. Write "hand-derived from source, no compiled-C++ golden".
- Reference C++ lives on other branches (`raptor*`, `Jev_laya`, `fog`, `v6-dev-*`): read with `git show <branch>:<path>`; use `git grep` first, plan paths go stale.
- No training runs unless the user asks.
- Commit only files the task names, never `git add -A`. Message `v0.M.NNN: <what>` (see [Versioning](#versioning)). Leave the user's `.gitignore` edit alone.
- Commit and tag on green: when an agent (Claude or Codex) has implemented a feature and its named tests pass, it commits and tags it right away, without asking. Details in [Versioning](#versioning), "Commit and tag on green".
- `tasks/` is not uniformly excluded: most files are tracked. Check with `git ls-files tasks/rust/<file>`; only an untracked file needs `git add -f`.
- Do not run `cargo xtask fmt`: it reformats unrelated files. Format edited code by hand.
- `cargo xtask oxide` rewrites `src/lightnet-cuda-oxide/Cargo.lock`; revert that file before committing.
- Each edited `.md` needs a TOC.
- Clean up after each completed step: delete `.weights`, `.backup`, `.onnx`, optimizer sidecars, logs, A/B build trees (e.g. `build/base`, `build/cur`), and other big files once nothing needs them. Keep any file a doc or test still references, and record the evidence in the docs before deleting. `build/` is gitignored, so deleting there is safe, but look at the target first; if unsure, ask.

## Layer Layout and Completion

This section defines where each layer's code lives and when a layer is finished. It applies to every layer, and the moves it describes are the first part of the current work; the completion work in [Open Work](#open-work) builds on it.

### Layout

Each layer's host and device code are grouped by layer, in two crates because `lightnet-cuda-oxide` depends on `lightnet-core` and so cannot hold CPU code:

```
src/lightnet-core/src/layers/<layer>/
  mod.rs   re-exports cpu::*; lib.rs re-exports the layer at its old path
  cpu.rs   host reference: spec use, forward, exact backward, weights, optimizer arms
  gpu.rs   device runtime glue: GpuExecutableNetwork methods for full and compact
           forward, backward and update (present once the layer's arms are moved)
src/lightnet-cuda/src/host_ops.rs      host math shared with the backend trait defaults
src/lightnet-cuda-oxide/src/kernels/   device kernels, one file per layer as they are split
src/lightnet-cuda-oxide/tests/oxide_<layer>.rs   CPU-versus-GPU parity for that layer
```

- `src/lightnet-cuda-oxide` stays outside the workspace and builds only through `cargo xtask oxide`.
- `tasks/rust/layer-map.md` (to be written) lists, per layer, its CPU file, GPU glue, kernel file and parity test.
- Keep every old path working through re-exports, so moving a layer never forces edits elsewhere.
- Widen visibility only as far as needed (`pub(crate)`); never widen to `pub` just to compile.

### Layer Completion Contract

A layer is complete only when all of the following hold, with a named test or documented evidence for each:

1. Host: config and shape, `.weights` round trip, CPU forward, exact CPU backward, SGD and Adam arms, Muon where the layer has eligible matrices, and a named `axis_layout` test.
2. Device: native full forward and, where the f16 inference list includes the layer, native compact forward; native backward; native parameter update; no host round trip inside the arm, except a documented, tested fallback for variants the kernel cannot cover.
3. Parity: an Oxide test compares the device forward, backward and update with the CPU result within a stated tolerance, run through `cargo xtask oxide --test <suite>`.
4. Layout: the code lives in `layers/<layer>/` and has a row in `layer-support.md` and `layer-map.md`.
5. Docs: the version commit, `implemented.md`, `STATUS.md`, `todo.md` and `version-map.tsv` are updated, and nothing claims C++ parity without a compiled-C++ golden.

Native primitives or a native forward alone do not make a layer complete. `layer-support.md` records which parts exist today.

### Layout Move Queue

Status on 2026-10-24, each move a refactor commit without a version tag (`refactor(rust): ...`):

- Done, host code: every layer module that was in `lightnet-core/src` is under `layers/<layer>/cpu.rs`.
- Done, device glue in `layers/<layer>/gpu.rs`: `dcnv4`, `hyperbolic_foveation`, `laya_act_head`, `laya_encoder`, `laya_head`, `laya_scorer`, `vit`, `swin`, `tucker_attention`, `mambavision`, `kimi_kda`, `laya_marker_gather`, `channel_slice`, `channel_shuffle`, `route`, `pool` (max, local average, average), `upsample`, `reorg`, `softmax`, `scale_channels`, `sam`, `shortcut`, `crop`, `local`, `connected`, `deconvolutional`, `eml_convolutional`, `moe`, `foveation_film`, `dsconv`, `deform_conv`, `graph_conv`, `rnn`, `lstm`, `crnn`, `gru`, `convolutional` and `batch_norm` (with the shared batch-norm helpers). For the five transformer layers and `vit`, only the host-fallback forward arm and, for `vit`, the native inference path have moved so far.
- Done: no full forward arm remains inline in `runtime/gpu_forward.rs` except pass-throughs. The `yolo` family (`yolo`, `obb_yolo`, `gaussian_yolo`, `detection`, `cost`, `regression`, `omd_loss`, `contrastive`) has no forward arm to move: it is a pass-through.
- Done, compact arms in `runtime/gpu_forward_compact.rs` (`gpu.rs` `forward_<layer>_compact_gpu`): `convolutional`, `deconvolutional`, `graph_conv`, `connected`, `route`, `scale_channels`, `channel_slice`, `reorg`, `laya_marker_gather`, `laya_act_head`, `shortcut`, `sam`; the pool and `upsample` compact arms reuse the f32 methods. Open: the other compact arms (thin delegations). Done, backward arms (`backward_<layer>_gpu`): `batch_norm`, `maxpool`, `local_avgpool`, `avgpool`, `crop`, `route`, `laya_marker_gather`, `upsample`, `softmax`, `channel_slice`, `channel_shuffle`, `reorg`, `sam`, `scale_channels`. Open: the remaining backward arms in `runtime/gpu_backward.rs`. Move backward one layer per commit and run that layer's Oxide parity suite before committing.
- Open, device kernels and backend methods: split `kernels/layers_extra.rs` and the `OxideBackend` methods in `lightnet-cuda-oxide/src/lib.rs` per layer, keeping kernel names and signatures unchanged.
- Open, `layer-map.md`: write it once the moves above are done.

Order of work: finish the moves first, then work through the completion queue in [Open Work](#open-work). A move changes no behaviour; if a move needs a behaviour change, make it a separate versioned commit.

## Commands

```
cargo xtask check [pkg...]                  # default: lightnet-cli, lightnet-core
cargo xtask gpu-check                       # lightnet-cli with oxide
cargo xtask test [filter]                   # host tests; filter is much faster
cargo xtask oxide --test <name> [filter]    # one oxide suite
cargo xtask clippy [pkg...]
cargo xtask build [pkg...]
cargo test -p lightnet-core --test cfg_corpus -j 2   # run alone; >1 h under load
```

`cargo xtask` applies the memory and thread caps, and warns and runs uncapped when there is no user systemd bus. A `cfg_corpus` timeout is neither a pass nor a failure.

## Versioning

- Version lines: `v0.0.NNN` is closed at `v0.0.274`. New work uses minor lines, each numbered from 1 in three digits:
  - `v0.1.NNN`: Stage 1, CPU parity of all layers (missing layers, axis alignment, CPU Muon). Incomplete; latest `v0.1.042`.
  - `v0.2.NNN`: Stage 2, all GPU paths complete (native kernels for every layer, GPU Muon). Active under the GPU-priority override; latest `v0.2.198`, so the next free number is `v0.2.199`.
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

## File Map

| File | LOC | Role | Open when |
|---|---|---|---|
| `STATUS.md` | 1046 | Resume point: ground rules, per-version "Implemented" log, evidence gaps | Need history of a version or an evidence gap |
| `layer-support.md` | 363 | Per-`LayerKind` CPU/GPU/f16/axis matrix + gap lists; derived from code, not prose | Checking whether a layer has a CPU port, a GPU path, or an axis pin |
| `todo.md` | 808 | Deduplicated open items by area | Choosing next work |
| `implemented.md` | 2745 | Consolidated done ledger (sections 1–262) | Checking whether X exists |
| `plan-open-items.md` | 292 | Waves 0–2 done; remaining scope = layer parity + axis alignment (checklist, backlog) | Dispatching the next layer task |
| `layers.md` | 3396 | Layer Port Contract (lines 61–97) + Phase 0 TDD recipe + phase tables (end of file) | Porting a layer: read the contract and the tail only |
| `onnx.md` | 1238 | 21-step ONNX export plan, complete (Rust-runtime numeric checks passed) | Evidence/history for `export-onnx` |
| `carpn_map_plan.md` | 84 | CARPN sample-level mAP dispatch plan; API implementation complete | Evidence for caller-supplied sample behavior |
| `carpn_network_map_plan.md` | 134 | CARPN CPU network inference, `detect-image`, and direct `detector map`; shipped in v0.0.259; CARPN training/`train -map` deferred | CARPN training work |
| `gui_plan.md` | 265 | Merged egui GUI plan (M0–M6); partly implemented in v0.1.019 (JSON+txt sidecars, scan/filter/nav/undo, edit model, actions, project creation, class remap, `detect-server` + worker, launcher/canvas/views/class-editor/overlay). Open: class add/delete dialog, mapped-but-unimplemented hotkeys, train runner, DarkMark file diff | GUI work |
| `archive/` | - | Superseded `gui_iced_plan.md`, `gui_code_phase_plan.md`; the latter holds DarkMark ground-truth tables G1–G17/D1–D16 | Citing `Gn`/`Dn` |
| `MOE.md` | 150 | True sparse `[moe_convolutional]` plan (opt-in, CPU then GPU, dense-masked as oracle), literature, open questions | Sparse MoE work |
| `gpu_layers_plan.md` | 673 | Operator-authorized GPU-first continuation; deconvolution, dense-head loss and ViT inference progress through v0.2.198; remaining DCNv4 variants, GPU Muon and native updates/residency open; current resume point at top | GPU layer work and resume |
| `AGENTS_fp8.md` | 105 | Low-precision fp16/bf16/fp8 goal, plans, delegation and user gates | Low-precision work |
| `fp8_fp4.md` | 157 | FP8 GEMM / FP4 emulation design, sm_89 | FP8/FP4 design history |
| `vit_plan.md`, `vit_notes.md` | 71 | `[vit]` subplan and source audit | ViT / attention work |
| `augmentation-audit.md` | 59 | C++ detector augmentation semantics | Augmentation work |
| `carpn_task_plan.md`, `carpn_notes.md` | 812 | v65–v70 per-milestone plans and notes (`NORM_CHAN*` etc.), superseded by `implemented.md` | Almost never |
| `lessons.md` | 18 | Rules from past corrections | Session start |
| `version-map.tsv` | 517 | idx, version, old sha, new sha (`-` = none) | Mapping version to commit |

## Current State

- The active line is `v0.2` under the GPU-priority override. Latest version is `v0.2.198` (native f32 ViT inference forward, `implemented.md` §262); `v0.2.197` added native Segmentation loss/delta, `v0.2.196` native JevCe loss/delta, and `v0.2.193`–`v0.2.195` finished deconvolution channel normalization plus compact f32/f16 inference. Next free is `v0.2.199`.
- The layer-layout move is in progress: host code and most small device arms are under `lightnet-core/src/layers/<layer>/`; see the [Layout Move Queue](#layout-move-queue) for what is done and open. These moves are `refactor(rust):` commits with no version number.
- Per-kind CPU/GPU/f16/axis status is tabulated in `layer-support.md`; use it instead of re-deriving from plan prose.
- Waves 0–2 of `plan-open-items.md` are complete: Waves 0–1 (v0.0.247–253) closed the OBB CPU training gaps; Wave 2 implemented ONNX export (`v0.0.254`–`v0.0.257`) and completed docs as `2b998051`; v0.0.258 adds caller-supplied CARPN mAP dispatch.
- ONNX evidence: `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; `MEM=3G ./tools/test_host.sh export_onnx_args` passed 1 CLI test; CLI build passed. ONNX Runtime worst relative errors were `3.257e-06` fused, `3.908e-05` unfused, and `3.831e-06` Mish, all under `1e-4`. Rust-runtime parity only, not compiled-C++ parity.
- `MEM=3G` host tests ran in a user systemd scope. Full baseline remains non-green: four known `cfg_corpus` OOM/timeout cases remain unresolved (CSPX p7-mish and p7-mish_hp SIGKILL/OOM at ~7 GiB; `lstm.train.cfg` and `rnn.train.cfg` hit the 300 s CPU timeout). Do not claim a green host suite.
- Canonical LegoGears dataset assets used by the original ONNX plan are absent; checks used local matching cfg/weights and artwork. ONNX Runtime is installed, but Python `onnx` package is not required by the checker and is absent.
- Scope narrowed to layer parity and axis alignment; non-layer open items are not dispatched from the plan.
- The working tree currently has only the untracked `plan_deeplsd.md` and `plan_llm_recursive.md`; leave them alone unless a task names them. Check `git status` before staging.

## Open Work

The [Goal](#goal) is parity with every C++ `ELayerType` (compared on `Jev_laya`; the other branches add no kinds), axis/layout pinned by tests, and GPU support for every layer. Nothing else is in scope (`plan-open-items.md`, "Scope"). Each backlog stub must be expanded into its own plan before dispatch; each layer's Definition of Done includes the axis checklist.

Default order across release lines (the current GPU-priority operator override above supersedes the no-interleaving gate):

1. **v0.1 / Stage 1, CPU for all:** finish every required CPU layer/config/source variant (config, shape, `.weights`, forward, exact backward, axis tests) and CPU Muon before starting Stage 2.
2. **v0.2 / Stage 2, GPU for all:** give every layer a native GPU path, including layers that currently only have a host round-trip fallback (`swin`, `tucker_attention`, `mambavision`, `kimi_kda`, `dcnv4`, `laya_*`, ...), plus GPU Muon. A host fallback is only interim; completion requires native cuda-oxide kernels with CPU-parity tests (`./tools/test_oxide.sh`). Mark each layer's GPU status (native / host fallback / none) in `implemented.md`.
3. **v0.3 / Stage 3, sparse MoE:** implement the opt-in true sparse path per `MOE.md`, CPU then GPU, after v0.2 is complete.

Stage 1 remains incomplete; the latest version-mapped CPU increment is v0.1.042. The operator explicitly authorized GPU priority: continue bounded native work per `gpu_layers_plan.md`, retaining CPU gaps and leaving v0.3 unstarted. The latest version-mapped GPU increment is v0.2.198. Recent versioned work: v0.2.194 compact f32 deconvolution, v0.2.195 compact f16 deconvolution, v0.2.196 native JevCe loss/delta, v0.2.197 native Segmentation loss/delta, and v0.2.198 native f32 ViT inference forward. The WMHF primitives are not a runtime GPU path; keep its GPU support marked incomplete.

**Device completion queue.** Work through this list in order, after the open items in the [Layout Move Queue](#layout-move-queue). Each item ends with the layer meeting the [Layer Completion Contract](#layer-completion-contract). The current state of each layer is in `layer-support.md`.

1. Transformer family, which shares kernels (LayerNorm, GEMM, flash attention, mHC residual):
   - `vit`: device backward (flash dQ/dK/dV primitives exist), a `forward_compact` arm, and a training path that fills the cache natively, so the host fallback is no longer needed for training.
   - `swin`, `tucker_attention`, `mambavision`, `kimi_kda`: native forward and backward; today they are host round trips in both directions.
   - `depth_anything` and `dpt_head`: native forward after `vit`, then backward.
2. Layers with device primitives but no runtime path: `wmhf` (Haar DWT/IWT, scan, split, gate and high-frequency primitives exist; the seven convolution branches do not), then `clifford`, `recursive_block`.
3. Layers with no device path and a CPU gap: `yolo_3d` (needs the v0.1 loss and decode work first) and `obb_refine`/CARPN (CPU training first).
4. Hybrid layers that still use host steps: `hyperbolic_foveation` and `foveation_film` (host HF source and cache), `dcnv4` excluded variants, `deform_conv` with batch norm, `graph_conv` unsupported variants, `gru` training, and CenterNet, modern-YOLO and LayaLoss dense-head loss.
5. Optimizers: GPU Muon for the remaining matrix families, and native update and residency where parameters are still host-staged.

A host fallback listed in `layer-support.md` is interim until the layer passes the contract. Compact (`forward_compact`) coverage is part of the contract only where the layer is on the f16 inference list; `vit`, `swin`, `tucker_attention`, `mambavision` and `kimi_kda` have no compact arm today.

**Muon optimizer (also in scope).** Add Muon next to SGD/Adam wherever it applies: 2D weight matrices (convolution weights viewed as `[out, in*kh*kw]`, `connected`, deconvolutional, attention/MLP projections in `vit`/`swin`/`transformer`/`tucker_attention`/`kimi_kda`/`mambavision`, MoE/EML expert convs, `dcnv4` projections, `laya_*` projections). Biases, BN/norm scales, positional/embedding tables and 1D params stay on Adam/AdamW. Follows the same two stages: CPU first (momentum buffer + Newton-Schulz orthogonalisation, optimizer state saved in the DNOPT sidecar and round-trip tested like Adam moments), then a native GPU kernel. No Muon exists in the repo or on any C++ branch (checked with `git grep`), so the spec must be written first (update rule, NS coefficients/iterations, update-scale convention, per-layer eligibility); state "hand-derived from the paper, no C++ reference".

**Sparse MoE (also in scope):** `[moe_convolutional]` currently runs every expert and masks the output (same as C++). `MOE.md` plans an opt-in true sparse path (CPU first, then GPU) that must equal the dense-masked result.

Current gaps and completed CPU work (see `implemented.md`, `todo.md`, `layer-support.md`, and `STATUS.md`; latest version-mapped CPU increment is v0.1.042 and GPU increment is v0.2.198):

1. **`[yolo_3d]` remains a partial v0.1 layer, not a complete port.** The `depth_3d` source confirms scope, and `[dpt_head]` is CPU-ported. The accepted `yolo_3d_plan.md` supersedes the draft; S1–S13 are transcribed in `yolo_3d_notes.md`. v0.1.024 adds config/spec/shape helpers; v0.1.025 adds standalone label/truth helpers; v0.1.026 integrates network aliases/shared class-K validation and weightless raw CPU forward/exact supplied-seed backward, with checkpoint/optimizer and legacy consumer/GPU rejection tests (`implemented.md` §§38–40). The `yolo_3d` filter passes 42 core + 1 CLI tests. Next: projected math/assignments/loss and exact derivatives, standalone/mixed training APIs consuming truths and map/counter metadata, decode/evaluator and the A2 output-axis pin; then depth/metric CPU support. Seeded optimizer tests are not full loss-to-conv training evidence; all legacy detector consumers still reject cuboid networks. Axis status remains Open, GPU status none, no compiled-C++ golden. The plan's projected version numbers are stale; use `version-map.tsv` and never reuse them.
2. **DSConv multi-template `morph=0/1` is required for full source parity and remains an open v0.1 gap.** The executable C++ CPU forward implements multi-view averaging and training-time view dropout (`Jev_laya:src-lib/dsconv_layer.cpp:597-638`), although the header calls it experimental and the CPU backward is a stub. Rust v0.1.002 explicitly rejects it. Treat the variant as a separate Opus-planned task covering cached active views and exact gradients for all offset blocks; do not claim full DSConv parity yet.
3. The other previously missing Stage-1 CPU ports are implemented within their documented bounds: `deform_conv`, `graph_conv`, `wmhf`, `clifford`, `recursive_block`, `hyperbolic_foveation`, `foveation_film`, `depth_anything`, and `dpt_head`; CPU Muon is recorded in v0.1.005. Their focused tests and evidence boundaries are in `implemented.md` and `STATUS.md`. These do not establish compiled-C++ parity.
4. The bounded A1 family audit is complete in v0.1.012; DPT's named axis test is in v0.1.013. This is not all-layer axis completion: the current ledger has 46 pinned C++-backed variants, 14 open C++-backed variants (including partial Yolo3d), 5 Rust-only variants and Unsupported. See `axis-coverage.md`; `axis_layout` passes 50 tests. DSConv, DeformConv, GraphConv, WMHF, Clifford, DepthAnything, RNN, LSTM, CRNN, Contrastive, JevCe and OmdLoss source-derived axis pins are recorded in `implemented.md`; check `version-map.tsv` for whether a specific pin has landed. Resolve all applicable Open contracts as well as per-backlog A2 tests before closing Stage 1.
5. **v0.2 is incomplete:** every layer needs a native GPU implementation and CPU-parity test, including current host fallbacks; GPU Muon remains open. The current operator override authorizes GPU work now despite unresolved v0.1 requirements. Native JevCe loss/delta (v0.2.196, §260) and Segmentation loss/delta (v0.2.197, §261) are wired; CenterNet, modern-YOLO and LayaLoss still use the host dense-loss path. Deconvolution `NormalizeChannels*` and compact f32/f16 inference are native (v0.2.193–195, §§257–259). DCNv4 bounded native runtime math, BN adjoint/EMA and `NormalizeChannels*` activations remain as documented in `implemented.md` §§50–52; large/unsupported variants keep host caches, and parameter updates/residency remain open. WMHF primitives are not a runtime GPU path. Do not confuse native primitives or forward with a complete native layer.
6. **v0.3 is incomplete:** `[moe_convolutional]` still computes every expert; the opt-in sparse path in `MOE.md` is sequenced after v0.2.
7. **Act-head training gap discovered during GPU audit:** executable C++ accumulates/updates all four act-head parameter blocks, whereas Rust runtime uses only the returned CLS adjoint and ignores parameter gradients. The frozen-weight module description does not establish full source-training support. The operator question about including this training fix remains unanswered; do not infer approval from automatic continuation. Independent native math can progress without changing that behavior. Evidence: `implemented.md` §43. Shared-file act-head loading also remains unproven: an error hint naming a missing loader is not an artifact, and §45 tests only in-memory blob replacement. These gaps remain part of overall CPU/GPU completion.

**C++ enum boundary:** `NETWORK` is the `[net]`/`[network]` metadata section, not a layer (`Jev_laya:src-lib/darknet_cfg.cpp:817-820`; Rust `NetConfig::from_section` accepts both). `BLANK` and `EMPTY` are marked unused in `darknet_enums.hpp:51-52`; `BLANK` is an internal GPU-fusion sentinel (`darknet_network.cpp:2662-2672`), and the `[blank]` alias falls through the parser's unsupported default. `[empty]`/`[silence]` map to `EMPTY` and also hit that default (`darknet_enums.cpp:60-62`; `darknet_cfg.cpp:947-1157`). They are not missing Rust layer kinds; unsupported cfgs must still error explicitly.

Already done within documented CPU-reference bounds, do not blindly re-plan those ports (`vit` also has a native inference forward since v0.2.198): `vit`/`vc_attention`, `swin`/`transformer`, `tucker_attention`, `mambavision`, `kimi_kda` (MiniMax sparse attention is a shared option), `dcnv4`, `laya_*`. This does not close native GPU work or the act-head training/loading gaps above.

Not dispatched from this GPU-layer queue (still in `todo.md`): GUI, CARPN Stage-2 training and `detector train -map`, OBB GPU training/`-map`, CUDA graph capture (conditional; see `AGENTS_fp8.md`), and convergence runs. FP8/FP4 is a separate active goal in `AGENTS_fp8.md`, not part of this queue.

Blocked: `dcnv3` (no reference numerics); `detr_decoder` (C++ broken).

## Lessons

- One package/command per build; do not chain `-p` crates or bundle edit+build+restore.
- Use the exact model name the user types (codex: `gpt-6-luna`).
- Do not compare NaN with `PartialEq`; assert `.is_nan()`.
- When one loss spans several passes, each pass owns only its partial gradient.

Full list: `lessons.md`.

## Doc Hygiene

- Finished work goes in `implemented.md` (one line per version with evidence), not as prose in plan files.
- Plan files keep only open work. Update `todo.md`, `STATUS.md`, `version-map.tsv` per version.
- Candidates to archive or trim (not done, needs user approval since most are untracked): `task_plan.md`, `notes.md`, and the checked-off Phase 0 TDD body of `layers.md` (about lines 97–3213).
