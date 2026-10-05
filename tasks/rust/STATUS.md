# Porting progress — durable state

Read this first after a context compaction. It records what is done, what is
verified, and what is next. The authoritative plan is
`tasks/rust/layers.md`; the FP8/FP4 plan is `tasks/rust/fp8_fp4.md`.

## Ground rules (do not violate)

- Every cargo command uses `-j 4` or fewer. `tools/test_host.sh` and
  `tools/test_oxide.sh` use `-j 2`.
- `src/darknet-cuda-oxide` is **not** a workspace member; build/test it with
  `./tools/test_oxide.sh` (which wraps `cargo oxide`).
- Backward is always the **exact derivative of the Rust forward**. C++
  deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Commit only the files a change names.

## User decisions (2026-10-04) — all Open Questions answered YES

"port everything in rust". Recorded in full in `tasks/rust/layers.md`
("Open questions for the user — RESOLVED 2026-10-04"). Short version:
ship the cfgs, port LAYA, port the `flash_attention`/`vc_attention` variants,
port the whole OBB family, build a dense-truth trainer, keep the legacy layers,
and let Rust define the swin weights order. **Nothing is blocked on a question
any more** — the remaining blockers are evidence gaps only.

## DONE and verified

`tasks/rust/fp8_fp4.md` — every phase except the optional step 5:
- Phase A: `kernels/gemm_f8.rs` (m16n8k32 e4m3/e5m2 tiled GEMM, NN/NT/TN/TT,
  batched, split-K), per-tensor quantiser, `GemmMode::Fp8`.
- Phase B: FP8 dgrad/wgrad with the C++ asymmetric formats (e4m3 weights x
  e5m2 gradient), `DARKNET_OXIDE_GEMM_BWD=fp8`.
- Phase C: `kernels/quant_fp4.rs` + `kernels/gemm_fp4.rs` (packed NVFP4,
  `__dp4a`), **bit exact** against a CPU i32 reference, so no tensor core is
  on that path. `GemmMode::Fp4`.
- Phase D: bench table in `tasks/rust/todo.md`.
- Step 5 (FP8 conv/mAP vs the C++ build) not done: no C++ build available.

`tasks/rust/layers.md`:
- **Phase 0 (Tasks 1-12)** — commits `8e4b0f24`..`be71eb6e`.
- **Phase 1b `segmentation`** — `segmentation.rs`, 18 tests.
- **Phase 1b `omd_loss`** — `omd_loss.rs`, 14 tests.
- **1b deferred (c) `contrastive`**, classifier branch — `contrastive.rs`,
  11 tests. The detection branch self-reports as unported.
- **Phase 2A `yolox`** — `modern_yolo.rs`, 15 tests.
- **Phase 1b `[laya_scorer]`** — `laya_scorer.rs`, 14 tests, validated against the
  real `test_data/laya/laya_english.weights` blob (LayerNorm gamma mean 0.9045,
  beta rms 0.0425 - a wrong offset or blob order would be arbitrary).

Layers ported follow the Layer Port Contract in `tasks/rust/layers.md`:
spec + parse + shape in `model.rs`, pure math in its own module, thin CPU
arms + host-round-trip GPU arms in `runtime.rs`, and a `dense_loss` arm for
heads whose truth is not their own output shape.

## Just landed (verify + commit if still uncommitted)

`[laya_scorer]`: `src/darknet-core/src/laya_scorer.rs` + `LayaScorerSpec` +
`LayerParams::LayaScorer`. Note it needs `#![feature(float_erf)]` in
`darknet-core` (the erf GELU is not darknet's tanh approximation, and C++ uses
`std::erf`).
Evidence: 248 darknet-core unit tests pass, 0 unreachable-pattern warnings,
`cfg_corpus` green, full oxide suite 18/18 `ok`, and
`cargo check/build/run -j 4` green. Cfg keys: `classes`(20), `max`(200),
`jitter`(0.2), `resize`(1.0) — the C++ parser reads nothing else, so every
training knob keeps its `set_default_training_params` default.

Two real bugs it caught, both worth remembering:
- the FP8 quantisation factor was inverted (`amax/max` instead of `max/amax`),
  which mapped every operand onto the bottom of the format's range;
- `yolox_train_batch` sized its delta per image but indexed it with whole-batch
  strides.

## Verification commands

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G cargo test -j 4 -p darknet-core --lib segmentation   # the fast inner loop
MEM=3G ./tools/test_oxide.sh --test oxide_gemm_f8  # ~50 s
MEM=3G ./tools/test_oxide.sh                      # ~15 min, 18 suites
```

`cfg_corpus` needs ~145 s **alone**; five unrelated cfgs hit their 300 s
per-cfg timeout when the suite shares the machine, so run it on its own before
believing a `cfg_corpus` failure.

## NEXT (in dependency order, nothing blocked)

1. Phase 1b `laya_*` — `laya_scorer` is done. Next in the family:
   `laya_marker_gather` (S, two inputs via `layers=idxA,idxB`, checks
   `out_c(A)==hidden` and `out_c(B)>=2*max_options`), `laya_act_head` (S/M,
   three inputs), `laya_loss` (M, `ce | rlcd`).
   **Order matters:** `[laya_scorer]`'s `weights=` is only usable together with
   `[laya_encoder]` + `[laya_head]` (the shared-file offset is computed past
   their blobs), so those two come next even though the plan files them under
   Phase 4. They also need the Phase 3A prerequisite kernel set.
2. Phase 2A remainder — `ppyoloe` + `yolonas` (share the DFL decode already
   sketched in `modern_yolo.rs` comments), then `centernet`, plus the
   LegoGears nested-object fix (`class_quality`, smallest-area-wins,
   min-cell-size gate) that lives in the same C++ file.
3. Phase 3A prerequisite (2) — the shared oxide op set: LayerNorm fwd/bwd,
   batched-GEMM wrapper, row-softmax backward, residual/bias epilogues. Build
   it **once**; `vit`, `swin`, `tucker`, `mambavision`, `kimi_kda`, `minimax` and
   the LAYA encoder all need it.
4. Phase 2B `moe_convolutional` (weights order verified) and
   `eml_convolutional` (an activation, not a `LayerKind`; save order is
   `weights.cpp:775`, still `[report: not re-read]` — re-read it).
   Both need the container-conv reuse prerequisite first.
5. Phase 3B deformable family — start with `dcnv4` (real C++ forward and
   kernels; weights order `weights.cpp:428`).
6. Phase 2C OBB, then Phase 4.

## Known evidence gaps (not user questions)

- `[dcnv3]`: forward/backward/update are stubs in **all five** branches that
  carry it (`deform`, `Jev_laya`, `fog`, `jepa`, `depth_3d`), and
  `docs/dcnv3_layer.md` says so. Only its `.weights` order can be ported
  against C++; the numerics would have to come from InternImage. Details and
  the evidence are in `tasks/rust/layers.md`.
- `[contrastive]` detection branch needs the yolo head's per-box class ids;
  `YoloHeadOutput` only carries the raw output tensor.
- DETR is still broken in C++, so Phase 4's `detr_decoder` also has no
  reference numerics.
- No C++ build of any owning branch is available here, so C++ dump parity is
  "NOT RUN" everywhere; it is recorded per layer in `tasks/rust/todo.md`.
