# Rust Port — Agent Handbook

Entry point for any agent working in `tasks/rust/` (branch `rust`). Read this first; open other files only when the map below says to.

## Table of Contents

- [Goal](#goal)
- [Rules](#rules)
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

Done when all three lines are complete, each version has its tests passing and its docs updated (`implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`), and no parity is claimed without a C++ golden. Default sequencing starts each line after the previous one is complete. **Current operator override: focus on GPU layers now, directly without subagents, while preserving all open CPU requirements.** Follow `gpu_layers_plan.md`; native GPU work is authorized before CPU completion, not evidence that CPU is complete. Details: [Open Work](#open-work).

## Rules

- Max 4 build threads (`-j 4`; repo scripts use `-j 2`). One build/test command at a time, never two in parallel.
- Cap memory: `systemd-run --user --scope -p MemoryHigh=3G <cmd>` (scripts take `MEM=3G`). If no user systemd bus, run uncapped and say so.
- `src/darknet-cuda-oxide` is not a workspace member: build/test it only via `./tools/test_oxide.sh` (`cargo oxide`).
- Backward = exact derivative of the Rust forward. C++ deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Never claim C++ parity: no compiled C++ exists here. Write "hand-derived from source, no compiled-C++ golden".
- Reference C++ lives on other branches (`raptor*`, `Jev_laya`, `fog`, `v6-dev-*`): read with `git show <branch>:<path>`; use `git grep` first, plan paths go stale.
- No training runs unless the user asks.
- Commit only files the task names, never `git add -A`. Message `v0.M.NNN: <what>` (see [Versioning](#versioning)). Leave the user's `.gitignore` edit alone.
- `tasks/` is in `.git/info/exclude`: only `STATUS.md`, `todo.md`, `layers.md`, `lessons.md`, `fp8_fp4.md` are tracked; the rest are local-only and need `git add -f` to commit.
- Each edited `.md` needs a TOC.
- Clean up after each completed step: delete `.weights`, `.backup`, `.onnx`, optimizer sidecars, logs, A/B build trees (e.g. `build/base`, `build/cur`), and other big files once nothing needs them. Keep any file a doc or test still references, and record the evidence in the docs before deleting. `build/` is gitignored, so deleting there is safe, but look at the target first; if unsure, ask.

## Commands

```
MEM=3G ./tools/test_host.sh [filter]         # darknet-core + darknet-cli; filter is much faster
MEM=3G ./tools/test_oxide.sh --test <name>   # one oxide suite (~15 min for all)
cargo check -p darknet-cli --features oxide -j 4
cargo test -p darknet-core --test cfg_corpus -j 2   # run alone; >1 h under load
```

A `cfg_corpus` timeout is neither a pass nor a failure.

## Versioning

- Version lines: `v0.0.NNN` is closed at `v0.0.274` (last row of `version-map.tsv`). New work uses minor lines, each numbered from 1 in three digits:
  - `v0.1.NNN`: Stage 1, CPU parity of all layers (missing layers, axis alignment, CPU Muon).
  - `v0.2.NNN`: Stage 2, all GPU paths complete (native kernels for every layer, GPU Muon). Starts only when `v0.1` is complete.
  - `v0.3.NNN`: sparse MoE, CPU and GPU (`MOE.md`). Starts only when `v0.2` is complete.
  Next version = highest number in the active line + 1; the first is `v0.1.001`.
- Code commits are `v0.M.NNN: <what>`. Docs-only commits are `docs(rust): <what>` and take no number.
- Per version, append a `version-map.tsv` row (`idx<TAB>version<TAB>old_sha<TAB>new_sha`; `old_sha` is `-` when there is no pre-rewrite sha; `new_sha` is the full commit sha).
- Per version, also add one line with evidence to `implemented.md`, an entry to `STATUS.md` "Latest increments", and tick `todo.md`.
- Gap: `v0.0.248` (`2941a2ee`, CUDA toolkit path in `.cargo/config.toml`) was out of plan. Gaps in numbering are allowed; never reuse a number.
- Version commits are `git add`ed by path. Tracked files commit normally; local-only files in `tasks/` need `git add -f`.

## File Map

| File | LOC | Role | Open when |
|---|---|---|---|
| `STATUS.md` | 803 | Resume point: ground rules, per-version "Implemented" log, evidence gaps | Need history of a version or an evidence gap |
| `layer-support.md` | 310 | Per-`LayerKind` CPU/GPU/f16/axis matrix + gap lists; derived from code, not prose. Local-only (`git add -f` to track) | Checking whether a layer has a CPU port, a GPU path, or an axis pin |
| `todo.md` | 521 | Deduplicated open items by area | Choosing next work |
| `implemented.md` | ~1450 | Consolidated done ledger (sections 1–51) | Checking whether X exists |
| `plan-open-items.md` | 280 | Waves 0–2 done; remaining scope = layer parity + axis alignment (checklist, backlog) | Dispatching the next layer task |
| `layers.md` | 3396 | Layer Port Contract (lines 61–97) + Phase 0 TDD recipe + phase tables (end of file) | Porting a layer: read the contract and the tail only |
| `onnx.md` | 1238 | 21-step ONNX export plan, complete (Rust-runtime numeric checks passed) | Evidence/history for `export-onnx` |
| `carpn_map_plan.md` | 84 | CARPN sample-level mAP dispatch plan; API implementation complete | Evidence for caller-supplied sample behavior |
| `carpn_network_map_plan.md` | 134 | CARPN CPU network inference, `detect-image`, and direct `detector map`; shipped in v0.0.259; CARPN training/`train -map` deferred | CARPN training work |
| `gui_plan.md` | 265 | Merged egui GUI plan (M0–M6); partly implemented in v0.1.019 (JSON+txt sidecars, scan/filter/nav/undo, edit model, actions, project creation, class remap, `detect-server` + worker, launcher/canvas/views/class-editor/overlay). Open: class add/delete dialog, mapped-but-unimplemented hotkeys, train runner, DarkMark file diff | GUI work |
| `archive/` | - | Superseded `gui_iced_plan.md`, `gui_code_phase_plan.md`; the latter holds DarkMark ground-truth tables G1–G17/D1–D16 | Citing `Gn`/`Dn` |
| `MOE.md` | 150 | True sparse `[moe_convolutional]` plan (opt-in, CPU then GPU, dense-masked as oracle), literature, open questions | Sparse MoE work |
| `gpu_layers_plan.md` | ~350 | Operator-authorized GPU-first continuation; bounded deform_conv and DCNv4 native runtime math wired; DCNv4 BN math is native; remaining activations/large variants and native updates/residency remain open; NormalizeChannels* native (v0.2.015); current resume point at top | GPU layer work and resume |
| `runtime_modularization_plan.md` | 125 | Plan to split the 37k-line `runtime.rs` into a `runtime/` module tree (phases P0–P7, no API change) | Refactoring or adding a layer arm in `runtime.rs` |
| `fp8_fp4.md` | 125 | FP8 GEMM / FP4 emulation design, sm_89 | FP8/FP4 work |
| `vit_plan.md`, `vit_notes.md` | 71 | `[vit]` subplan and source audit | ViT / attention work |
| `augmentation-audit.md` | 59 | C++ detector augmentation semantics | Augmentation work |
| `task_plan.md`, `notes.md` | 812 | v65–v70 per-milestone plans and notes (`NORM_CHAN*` etc.), superseded by `implemented.md` | Almost never |
| `lessons.md` | 13 | Rules from past corrections | Session start |
| `version-map.tsv` | 260 | idx, version, old sha, new sha (`-` = none) | Mapping version to commit |

## Current State

- Per-kind CPU/GPU/f16/axis status is tabulated in `layer-support.md`; use it instead of re-deriving from plan prose.
- Waves 0–2 of `plan-open-items.md` are complete: Waves 0–1 (v0.0.247–253) closed the OBB CPU training gaps; Wave 2 implemented ONNX export (`v0.0.254`–`v0.0.257`) and completed docs as `2b998051`; v0.0.258 adds caller-supplied CARPN mAP dispatch.
- ONNX evidence: `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; `MEM=3G ./tools/test_host.sh export_onnx_args` passed 1 CLI test; CLI build passed. ONNX Runtime worst relative errors were `3.257e-06` fused, `3.908e-05` unfused, and `3.831e-06` Mish, all under `1e-4`. Rust-runtime parity only, not compiled-C++ parity.
- `MEM=3G` host tests ran in a user systemd scope. Full baseline remains non-green: four known `cfg_corpus` OOM/timeout cases remain unresolved (CSPX p7-mish and p7-mish_hp SIGKILL/OOM at ~7 GiB; `lstm.train.cfg` and `rnn.train.cfg` hit the 300 s CPU timeout). Do not claim a green host suite.
- Canonical LegoGears dataset assets used by the original ONNX plan are absent; checks used local matching cfg/weights and artwork. ONNX Runtime is installed, but Python `onnx` package is not required by the checker and is absent.
- Scope narrowed to layer parity and axis alignment; non-layer open items are not dispatched from the plan.
- Uncommitted user changes remain in `Cargo.toml`, `Cargo.lock`, `.gitignore`, `STATUS.md`, `fp8_fp4.md`, and `layers.md`; leave `.gitignore` alone.

## Open Work

The [Goal](#goal) is parity with every C++ `ELayerType` (compared on `Jev_laya`; the other branches add no kinds), axis/layout pinned by tests, and GPU support for every layer. Nothing else is in scope (`plan-open-items.md`, "Scope"). Each backlog stub must be expanded into its own plan before dispatch; each layer's Definition of Done includes the axis checklist.

Default order across release lines (the current GPU-priority operator override above supersedes the no-interleaving gate):

1. **v0.1 / Stage 1, CPU for all:** finish every required CPU layer/config/source variant (config, shape, `.weights`, forward, exact backward, axis tests) and CPU Muon before starting Stage 2.
2. **v0.2 / Stage 2, GPU for all:** give every layer a native GPU path, including layers that currently only have a host round-trip fallback (`swin`, `tucker_attention`, `mambavision`, `kimi_kda`, `dcnv4`, `laya_*`, ...), plus GPU Muon. A host fallback is only interim; completion requires native cuda-oxide kernels with CPU-parity tests (`./tools/test_oxide.sh`). Mark each layer's GPU status (native / host fallback / none) in `implemented.md`.
3. **v0.3 / Stage 3, sparse MoE:** implement the opt-in true sparse path per `MOE.md`, CPU then GPU, after v0.2 is complete.

Stage 1 remains incomplete at v0.1.026. The operator explicitly authorized GPU priority: v0.2.001–002 add native f32 marker gather and f16 compact inference; continue bounded native work per `gpu_layers_plan.md`, retaining CPU gaps and leaving v0.3 unstarted.

**Muon optimizer (also in scope).** Add Muon next to SGD/Adam wherever it applies: 2D weight matrices (convolution weights viewed as `[out, in*kh*kw]`, `connected`, deconvolutional, attention/MLP projections in `vit`/`swin`/`transformer`/`tucker_attention`/`kimi_kda`/`mambavision`, MoE/EML expert convs, `dcnv4` projections, `laya_*` projections). Biases, BN/norm scales, positional/embedding tables and 1D params stay on Adam/AdamW. Follows the same two stages: CPU first (momentum buffer + Newton-Schulz orthogonalisation, optimizer state saved in the DNOPT sidecar and round-trip tested like Adam moments), then a native GPU kernel. No Muon exists in the repo or on any C++ branch (checked with `git grep`), so the spec must be written first (update rule, NS coefficients/iterations, update-scale convention, per-layer eligibility); state "hand-derived from the paper, no C++ reference".

**Sparse MoE (also in scope):** `[moe_convolutional]` currently runs every expert and masks the output (same as C++). `MOE.md` plans an opt-in true sparse path (CPU first, then GPU) that must equal the dense-masked result.

Current gaps and completed CPU work (see `implemented.md`, `todo.md`, `layer-support.md`, and `STATUS.md`; latest CPU increment is v0.1.026; latest GPU increment is v0.2.015):

1. **`[yolo_3d]` remains a partial v0.1 layer, not a complete port.** The `depth_3d` source confirms scope, and `[dpt_head]` is CPU-ported. The accepted `yolo_3d_plan.md` supersedes the draft; S1–S13 are transcribed in `yolo_3d_notes.md`. v0.1.024 adds config/spec/shape helpers; v0.1.025 adds standalone label/truth helpers; v0.1.026 integrates network aliases/shared class-K validation and weightless raw CPU forward/exact supplied-seed backward, with checkpoint/optimizer and legacy consumer/GPU rejection tests (`implemented.md` §§38–40). The `yolo_3d` filter passes 42 core + 1 CLI tests. Next: projected math/assignments/loss and exact derivatives, standalone/mixed training APIs consuming truths and map/counter metadata, decode/evaluator and the A2 output-axis pin; then depth/metric CPU support. Seeded optimizer tests are not full loss-to-conv training evidence; all legacy detector consumers still reject cuboid networks. Axis status remains Open, GPU status none, no compiled-C++ golden. The plan's projected version numbers are stale; use `version-map.tsv` and never reuse them.
2. **DSConv multi-template `morph=0/1` is required for full source parity and remains an open v0.1 gap.** The executable C++ CPU forward implements multi-view averaging and training-time view dropout (`Jev_laya:src-lib/dsconv_layer.cpp:597-638`), although the header calls it experimental and the CPU backward is a stub. Rust v0.1.002 explicitly rejects it. Treat the variant as a separate Opus-planned task covering cached active views and exact gradients for all offset blocks; do not claim full DSConv parity yet.
3. The other previously missing Stage-1 CPU ports are implemented within their documented bounds: `deform_conv`, `graph_conv`, `wmhf`, `clifford`, `recursive_block`, `hyperbolic_foveation`, `foveation_film`, `depth_anything`, and `dpt_head`; CPU Muon is recorded in v0.1.005. Their focused tests and evidence boundaries are in `implemented.md` and `STATUS.md`. These do not establish compiled-C++ parity.
4. The bounded A1 family audit is complete in v0.1.012; DPT's named axis test is in v0.1.013. This is not all-layer axis completion: the current registry has 19 pinned variants, 41 C++-backed Open variants (including partial Yolo3d), 5 Rust-only variants and Unsupported. See `axis-coverage.md`; `axis_layout` passes 21 tests. Resolve all applicable Open contracts as well as per-backlog A2 tests before closing Stage 1.
5. **v0.2 is incomplete:** every layer needs a native GPU implementation and CPU-parity test, including current host fallbacks; GPU Muon remains open. The current operator override authorizes GPU work now despite unresolved v0.1 requirements; native f32 LAYA marker gather and f16 compact inference are implemented in v0.2.001–002 (`implemented.md` §§41–42); v0.2.003–005 add act-head feature/exact-erf primitives and native f32 full/compact forward. CPU backward, frozen/update behavior and f16 limits remain; full backward/update integration is open per `gpu_layers_plan.md`. v0.2.007 wires bounded deform_conv native forward/backward; v0.2.009–011 add DCNv4 aggregation host/native forward/adjoint primitives only (6 device tests), not runtime integration. v0.2.012 corrects a shared aggregation packing bug and composes native coarse-predictor/upsample/raw aggregation forward/backward; 3 chain tests cover real CPU outputs/input/parameter gradients (18 cases), device FD and guards. v0.2.013 wires bounded DCNv4 native f32 runtime forward/backward (no BN, <=25 active points, supported elementwise activation, no other fallback/host-only deform_conv). Two-step SGD/Adam parity passes, but parameter gradients/updates remain host-resident and offsets recompute on device. v0.2.014 extends this to native rolling-stat BN affine/adjoint and post-backward 0.9 EMA (including unseeded layers), preserving existing Rust CPU semantics, ignored ordinary bias/beta, and optimizer gates. Host DCNv4 26/runtime:: 282 and device BN 2/runtime DCNv4 3/raw composition 3/aggregate 6 pass. Large/unsupported/mixed variants keep host caches; compact/f16 unchanged. v0.2.015 adds native DCNv4 NormalizeChannels* activations/adjoints (§52). Next: remaining DCNv4 activations/large variants, native updates/residency. Evidence/limits in `implemented.md` §§50–51; full runtime-chain timed out at 600 s in ViT, not a full-suite pass. Do not confuse native primitives or forward with a complete native layer.
6. **v0.3 is incomplete:** `[moe_convolutional]` still computes every expert; the opt-in sparse path in `MOE.md` is sequenced after v0.2.
7. **Act-head training gap discovered during GPU audit:** executable C++ accumulates/updates all four act-head parameter blocks, whereas Rust runtime uses only the returned CLS adjoint and ignores parameter gradients. The frozen-weight module description does not establish full source-training support. The operator question about including this training fix remains unanswered; do not infer approval from automatic continuation. Independent native math can progress without changing that behavior. Evidence: `implemented.md` §43. Shared-file act-head loading also remains unproven: an error hint naming a missing loader is not an artifact, and §45 tests only in-memory blob replacement. These gaps remain part of overall CPU/GPU completion.

**C++ enum boundary:** `NETWORK` is the `[net]`/`[network]` metadata section, not a layer (`Jev_laya:src-lib/darknet_cfg.cpp:817-820`; Rust `NetConfig::from_section` accepts both). `BLANK` and `EMPTY` are marked unused in `darknet_enums.hpp:51-52`; `BLANK` is an internal GPU-fusion sentinel (`darknet_network.cpp:2662-2672`), and the `[blank]` alias falls through the parser's unsupported default. `[empty]`/`[silence]` map to `EMPTY` and also hit that default (`darknet_enums.cpp:60-62`; `darknet_cfg.cpp:947-1157`). They are not missing Rust layer kinds; unsupported cfgs must still error explicitly.

Already done within documented CPU-reference bounds, do not blindly re-plan those ports: `vit`/`vc_attention`, `swin`/`transformer`, `tucker_attention`, `mambavision`, `kimi_kda` (MiniMax sparse attention is a shared option), `dcnv4`, `laya_*`. This does not close native GPU work or the act-head training/loading gaps above.

Not dispatched (dropped, still in `todo.md`): GUI, CARPN Stage-2 training and `detector train -map`, OBB GPU training/`-map`, FP8/FP4, CUDA graph capture, convergence runs.

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
