# Porting progress — durable state

Read this first after a context compaction. It records what is verified and what
is next. The authoritative plan is `tasks/rust/layers.md`; the FP8/FP4 plan is
`tasks/rust/fp8_fp4.md`; the **full implemented inventory is
`tasks/rust/implemented.md`** (this file no longer repeats it).

## Contents

- [Ground rules (do not violate)](#ground-rules-do-not-violate)
- [User decisions (2026-10-04) — all Open Questions answered YES](#user-decisions-2026-10-04--all-open-questions-answered-yes)
- [Implemented](#implemented)
- [Axis audit](#axis-audit)
- [Verification commands](#verification-commands)
- [NEXT (general layer-port roadmap)](#next-general-layer-port-roadmap)
- [Known evidence gaps (not user questions)](#known-evidence-gaps-not-user-questions)
- [Task plans: ONNX complete; GUI partly implemented](#task-plans-onnx-complete-gui-partly-implemented)

## Ground rules (do not violate)

- Every cargo command uses `-j 4` or fewer. `tools/test_host.sh` and
  `tools/test_oxide.sh` use `-j 2`.
- `src/darknet-cuda-oxide` is **not** a workspace member; build/test it with
  `./tools/test_oxide.sh` (which wraps `cargo oxide`).
- Backward is always the **exact derivative of the Rust forward**. C++
  deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Commit only the files a change names.
- Per-kind CPU/GPU/f16/axis status lives in `layer-support.md` (code-derived).
  Read it before scheduling layer work; do not re-derive from plan prose.

## User decisions (2026-10-04) — all Open Questions answered YES

"port everything in rust". Recorded in full in `tasks/rust/layers.md`
("Open questions for the user — RESOLVED 2026-10-04"). Short version:
ship the cfgs, port LAYA, port the `flash_attention`/`vc_attention` variants,
port the whole OBB family, build a dense-truth trainer, keep the legacy layers,
and let Rust define the swin weights order. **Nothing is blocked on a question
any more** — the remaining blockers are evidence gaps only.

## Implemented

Everything delivered so far is consolidated in
[`tasks/rust/implemented.md`](implemented.md): the cuda-oxide backend (P0–P9),
FP8/FP4, layer-port Phases 0 / 1b / 2A / 2B-CPU and the Phase 3A shared
primitives, LegoGears_v2 inference parity, the training loop + data pipeline,
and housekeeping — each with its evidence.

**Unit baseline after v0.2.008 (`38cdb046`)**: `cargo test -p darknet-core --lib
-j 2` **950 passed**, `cargo test -p darknet-cli -j 2` **29 passed**. That is the
unit baseline only; the `cfg_corpus` integration suite is still not re-run and
its four known residuals (two ~7 GiB SIGKILL/OOM cases, two 300 s CPU-timeout
cases) remain unresolved. Do not read the 950/29 result as a green full suite.

**v0.2.007 (`15fc3f8c`)**: native `[deform_conv]` forward and backward on the
device, gated on `dilation == 1` without batch norm with the host path retained
for the excluded variants. Evidence: host 33, `oxide_deform_gather` 5,
`oxide_deform_conv_accumulate` 8, `oxide_deform_envelope` 14,
`oxide_deform_conv_chain` 3, `oxide_deform_conv_backward_chain` 3,
`oxide_runtime_chain` 26 (incl. the device-path and the BN host-fallback cases),
`test_host.sh deform_conv` 23. One host hop remains (the batch-coupled
normalization Jacobian). No f16, no diluted/BN device support, no benchmark or
compiled-C++ golden. Ledger: `implemented.md` §47; state: `layer-support.md`.

Latest increments:
- v0.2.146: GPU Muon now supports eligible HF predictor and FiLM matrices through native Oxide orthogonalization; two-step learned-anisotropic runtime parity passes. See `implemented.md` §182.
- v0.2.145: native HF inverse parameter adjoints now cover fixed-point and Newton iterations; isotropic/anisotropic finite differences and actual-Oxide runtime SGD/Adam parity pass. See `implemented.md` §181.
- v0.2.143: Oxide predictor source/W1/B1/W2/B2 adjoints pass finite-difference checks; see `implemented.md` §179.
- v0.2.142: native HF predictor backward now covers MLP parameter and source adjoints; actual-Oxide anisotropic SGD/Adam parity and direct input-gradient checks pass. The inverse-point parameter Jacobian remains host-side. See `implemented.md` §178.
- v0.2.141: opt-in GPU-backward input-gradient capture is now compared directly with CPU across learned-anisotropic two-step SGD and Newton Adam runtime tests. See `implemented.md` §177; native predictor-source and inverse-point parameter-Jacobian math remain host-side.
- v0.2.140: runtime consumes the native HF grid adjoint in host inverse-point parameter gradients; actual-Oxide SGD/Adam, 30 host HF, and 12 FiLM tests pass. Native predictor backward and direct runtime input-gradient inspection remain open. See `implemented.md` §176.
- v0.2.139: added native Oxide HF bilinear-warp input/grid adjoints and runtime input-gradient propagation; primitive reference/finite-difference plus actual-Oxide SGD/Adam parity tests pass. Runtime HF parameter/predictor backward and direct tensor inspection remain open. See `implemented.md` §175.
- v0.2.138: strengthened anisotropic Newton Adam runtime parity to use learned HF parameters and compare every predictor parameter block across two updates. Actual-Oxide test passes; see `implemented.md` §174.
- v0.2.137: native Oxide predictor plus fixed-point/Newton inverse grids now support isotropic and anisotropic HF forward. Anisotropic learned-HF SGD, anisotropic Newton Adam, primitive parity, and 12 focused host tests pass. HF adjoints and direct runtime input-gradient inspection remain open. See `implemented.md` §173.
- v0.2.136: native isotropic Newton inverse-grid generation is integrated; actual-Oxide Adam runtime parity and primitive parity pass. Anisotropic modes and HF backward remain incomplete. See `implemented.md` §172.
- v0.2.135: native HF predictor pooling/MLP/constraints now run for isotropic fixed-point GPU runtime; learned-HF two-step actual-device parity and 4 primitive tests pass. Other predictor/grid modes and HF backward remain host-side. See `implemented.md` §171.
- v0.2.134: added native isotropic fixed-point HF inverse-grid generation; learned predictor input/runtime parity passes through native grid and warp. Newton/anisotropic and backward remain host-managed. See `implemented.md` §170.
- v0.2.133: added native Oxide HF bilinear warp and wired GPU runtime forward; predictor/grid generation and backward remain host-side. Actual-device output parity and 2 primitive tests pass. See `implemented.md` §169.
- v0.2.132: added read-only host-fallback train-buffer inspection and actual-Oxide parity checks for every HF/FiLM gradient buffer under two-step SGD and Adam; both tests pass. Native convolution gradients remain checked indirectly by upstream parameter parity; direct input-gradient inspection remains open. See `implemented.md` §168.
- v0.2.131: actual-Oxide HF/FiLM `train_only_bn` gate parity passes; FiLM parameters remain unchanged after GPU-runtime SGD. Direct input-gradient/train-buffer inspection remains open. See `implemented.md` §167.
- v0.2.130: actual-Oxide HF/FiLM Adam `burnin_update=1` gate parity passes: FiLM is held at iteration 0 and updated at iteration 1. `train_only_bn` runtime gate and direct gradient-buffer inspection remain open. See `implemented.md` §166.
- v0.2.129: actual-Oxide HF/FiLM SGD `dont_update` gate parity passes for iteration 0 skip and iteration 1 update. Adam/burn-in/train-only-BN runtime gates and direct gradient-buffer inspection remain open. See `implemented.md` §165.
- v0.2.128: actual-Oxide rejects HF/FiLM compact f32, compact f16, and GPU Muon as intended; focused gate test passes. Direct runtime input-gradient/train-buffer inspection and SGD/Adam gate coverage remain open. See `implemented.md` §164.
- v0.2.127: extended actual-Oxide HF/FiLM SGD and Adam runtime parity to two complete forward/backward/update cycles; both focused tests pass. See `implemented.md` §164.
- v0.2.126: added host-fallback SGD/Adam optimizer updates for GPU-runtime HF/FiLM training; actual Oxide one-step SGD/Adam CPU parity passes, as do 12 focused host tests and the Muon rejection gate. Two-step runtime parity remains open. See `implemented.md` §163.
- v0.2.125: runtime FiLM backward now uses native Oxide input/projection adjoints and routes HF-source gradients through the host HF backward helper; actual-device runtime test plus 12 focused host tests pass. GPU optimizer updates remain rejected. See `implemented.md` §162.
- v0.2.124: actual-Oxide FoveationFilm full projection backward now finite-differences every HF, embedding, gamma, and beta gradient element against host forward; both focused tests pass. This remains primitive-only; runtime backward uses CPU fallback. See `implemented.md` §161.
- v0.2.123: integrated native Oxide FoveationFilm runtime forward with host-computed HF source parameters and explicit CPU backward fallback; actual-device runtime parity/backward-fallback plus all 12 focused host tests pass. GPU optimizer, compact/f16 paths, and native layer backward remain open; see `implemented.md` §160.
- v0.2.122: added native full FoveationFilm input/source/parameter adjoints with deterministic spatial and batch reductions; the 2 actual-device parity/FD tests and host reference test pass. Runtime integration remains open; see `implemented.md` §159.
- v0.2.121: added full FoveationFilm forward projection primitive (HF embedding/ReLU, gamma/beta projections, NCHW affine) through native Oxide operations; 2 actual-device parity/FD tests pass. Runtime integration and backward remain open; see `implemented.md` §158.
- v0.2.120: added native Oxide f32 FiLM per-sample channel affine and exact input/gamma/beta adjoints; host reference and 1 actual-device parity/finite-difference test pass. This is a primitive only; no FoveationFilm layer runtime path yet. See `implemented.md` §157 and `gpu_layers_plan.md`.
- v0.1.041: promoted the hand-computed FoveationFilm batch/channel affine test into the named `axis_layout` filter; `MEM=3G ./tools/test_host.sh axis_layout` passed 36 tests. Hand-derived from C++ source, no compiled-C++ golden; see `implemented.md` §156 and `axis-coverage.md`.
- v0.1.040: added a named batch-2, non-square Crop axis/layout pin for NCHW and centered-window source coordinates; `MEM=3G ./tools/test_host.sh axis_layout` passed 35 tests. Rust contract only, no C++ golden; see `implemented.md` §155 and `axis-coverage.md`.
- Verification alert (current): the `oxide_runtime_chain` suite run during v0.2.123 reported 70/72; the new FiLM test's config typo was fixed and its focused actual-device test passes. The DCNv4 non-BN Adam runtime test also failed independently on targeted rerun at size 15 (`delta[480725]`: `0.000032189768` vs `0.000032739248`), repeating the known near-threshold mismatch. Treat Adam runtime parity claims as unstable pending cause/fix and a clean rerun; no tolerance was changed.
- v0.2.119: extended DCNv4 two-step runtime CPU-reference parity to BN Adam at 19x19. Focused runtime test passed (626.18 s), with non-BN Adam through 23x23 and SGD through 31x31. See `implemented.md` §154. No C++ golden/full-suite claim.
- v0.2.118: extended DCNv4 two-step runtime CPU-reference parity to BN Adam at 17x17. Focused runtime test passed (603.29 s), with non-BN Adam through 23x23 and SGD through 31x31. See `implemented.md` §153. No C++ golden/full-suite claim.
- v0.2.117: extended DCNv4 two-step runtime CPU-reference parity to non-BN Adam at 23x23. The 593.08 s focused rerun passed after the known near-threshold Adam mismatch recurred at size 15 on the first attempt; tolerance unchanged, cause unresolved. SGD remains covered through 31x31, BN Adam through 15x15. See `implemented.md` §152. No C++ golden/full-suite claim.
- v0.2.116: extended DCNv4 two-step runtime CPU-reference parity to non-BN Adam at 21x21. Focused runtime test passed (584.18 s); SGD remains covered through 31x31 and BN Adam through 15x15. See `implemented.md` §151. No C++ golden/full-suite claim.
- v0.2.115: extended DCNv4 two-step runtime CPU-reference parity to non-BN Adam at 19x19. Focused runtime test passed (571.59 s); SGD remains covered through 31x31 and BN Adam through 15x15. See `implemented.md` §150. No C++ golden/full-suite claim.
- v0.2.114: extended DCNv4 two-step runtime CPU-reference parity to Adam at 17x17 without BN. Focused runtime test passed (562.82 s); SGD remains covered through 31x31 and BN Adam through 15x15. See `implemented.md` §149. No C++ golden/full-suite claim.
- v0.2.113: native Crop GPU training now applies saturation/exposure HSV transforms in addition to per-channel RGB shift and geometry; actual-Oxide CPU-reference parity covers forward/backward plus SGD/Adam. Crop primitive parity/FD and host crop (10 tests) pass. See `implemented.md` §148. No C++ golden/full-suite claim.
- v0.2.112: native Crop training now supports per-channel RGB shift in addition to geometry/affine scaling; saturation/exposure HSV adjustments still reject. Actual-Oxide runtime parity including backward and SGD/Adam updates, crop primitive parity/FD, and host crop (10 tests) pass. See `implemented.md` §147. No C++ golden/full-suite claim.
- v0.2.111: extended native DCNv4 to 31x31/961 active points; actual-Oxide composition and two-step runtime SGD parity pass, 33x33 falls back, and Adam parity remains through 15x15. See `implemented.md` §146. CPU-reference parity only.
- v0.2.110: extended native DCNv4 to 29x29/841 active points; actual-Oxide composition and two-step runtime SGD parity pass, 31x31 falls back, and Adam parity remains through 15x15. See `implemented.md` §145. CPU-reference parity only.
- v0.2.109: extended native DCNv4 to 27x27/729 active points; actual-Oxide composition and two-step runtime SGD parity pass, 29x29 falls back, and Adam parity remains through 15x15. See `implemented.md` §144. CPU-reference parity only.
- v0.2.108: extended native DCNv4 to 25x25/625 active points; actual-Oxide composition and two-step runtime SGD parity pass, 27x27 falls back, and Adam parity remains through 15x15. See `implemented.md` §143. First runtime attempt hit the known near-threshold Adam delta mismatch at size 15; rerun passed without tolerance change.
- v0.2.107: extended native DCNv4 to 23x23/529 active points; actual-Oxide composition and two-step runtime SGD parity pass, 25x25 falls back, and Adam parity remains through 15x15. See `implemented.md` §142. CPU-reference parity only.
- v0.2.106: extended native DCNv4 to 21x21/441 active points; actual-Oxide composition and two-step runtime SGD parity pass, 23x23 falls back, and Adam parity remains through 15x15. See `implemented.md` §141. CPU-reference parity only.
- v0.2.105: extended native DCNv4 to 19x19/361 active points; actual-Oxide composition and two-step runtime SGD parity pass, 21x21 falls back, and Adam parity remains through 15x15. See `implemented.md` §140. CPU-reference parity only.
- v0.2.104: extended native DCNv4 to 17x17/289 active points with actual-Oxide composition and runtime SGD parity; Adam runtime remains covered through 15x15, and 19x19 falls back. See `implemented.md` §139. CPU-reference parity only; one initial near-threshold Adam update mismatch passed on rerun, cause unresolved.
- v0.1.039: pinned SAM's elementwise NCHW named-source product and both input/source adjoints; focused test passed and full axis filter passed 34. See `implemented.md` §138. Hand-derived, no compiled-C++ golden.
- v0.1.038: pinned ScaleChannels named-source order and both channel/spatial broadcast axes plus their adjoints; focused test passed and full axis filter passed 33. See `implemented.md` §137. Hand-derived, no compiled-C++ golden.
- v0.1.037: added focused standard-Dropout per-element mask/inference/adjoint coverage; test passed. DropBlock remains open because the C++ GPU kernel's batch/channel plane-index expression is unresolved (axis ledger F3); see `implemented.md` §136. No compiled-C++ golden.
- v0.1.036: pinned Softmax's contiguous per-batch grouped reduction, stable output and grouped adjoint from `softmax_layer.cpp`/`blas.cpp`; focused softmax filter passed 12, full axis filter passed 32. See `implemented.md` §135. Hand-derived, no compiled-C++ golden.
- v0.1.035: pinned ChannelShuffle's group-row/group-column channel transpose and inverse adjoint across batch/spatial planes. Focused test passed; full axis filter passed 31; see `implemented.md` §134. Hand-derived, no compiled-C++ golden.
- v0.1.034: pinned Reorg normal-forward channel/offset packing and its adjoint; the full axis filter passed 30. C++ reverse source has an `out_c`/`in_c` divisor defect and remains unpinned; see `implemented.md` §133 and axis ledger F2.
- v0.1.033: pinned Upsample forward/reverse geometry, scale, adjoints and ragged-tail behavior from `upsample_layer.cpp`/`blas.cpp`; corrected CPU reverse mode to ignore incomplete stride blocks instead of indexing beyond output. Host upsample tests passed 6; full axis filter passed 29; see `implemented.md` §132. Hand-derived, no compiled-C++ golden.
- v0.1.032: pinned Route relative source order, group slicing and backward scatter from `Jev_laya:src-lib/route_layer.cpp:91-139`; named batch-2 non-square axis test passed. Full axis filter now passes 28 tests; ledger has 25 pinned / 35 open C++-backed variants. Hand-derived, no compiled-C++ golden; see `implemented.md` §131.
- v0.2.103: added actual-Oxide two-step connected-BN backward/optimizer parity with `leaky` activation for both SGD and Adam; outputs, weights, biases, BN scales and rolling stats match CPU. Focused Oxide test passed; see `implemented.md` §130.
- v0.2.102: expanded actual-Oxide connected-BN GPU Muon backward parity to `linear` and `leaky` activations; the focused case and all five `gpu_muon_connected` runtime-chain tests pass. Rust CPU-reference parity only; see `implemented.md` §129.
- v0.2.101: connected-layer BN backward now stays on the GPU path for supported activations, including BN adjoints and rolling-stat updates; BN optimizer accumulation/rolling-stat state remain host-managed. The actual-Oxide two-step Muon parity test, all five `gpu_muon_connected` Oxide tests, and focused host test passed. Rust CPU-reference parity only; see `implemented.md` §128.
- v0.2.100: compact f16 inference now accepts Detection and Regression identity layers; batch-2 actual-Oxide output matched f32 (relative max error 2.152e-4). Full f16 suite passed 18 tests; see `implemented.md` §127.
- v0.2.099: compact f16 inference accepts the Cost pass-through identity, enabling full ResNet101 parity through the connected classifier and final cost section. Full Oxide f16 suite passed 17 tests; ResNet101 relative max error was 9.775e-4. See `implemented.md` §126.
- v0.2.098: compact f16 Connected inference now uses native Oxide GEMM with NCHW flattening and folded rolling-stat BN. Full f16 suite passed 17 tests; batch-2 relative max errors vs f32 were 2.467e-4 (non-BN) and 3.655e-4 (BN). See `implemented.md` §125.
- v0.2.097: compact f16 convolution now supports all three `NORM_CHAN*` activations with native f32 reductions. Full Oxide f16 suite passed 15 tests; relative max errors vs f32 ranged from 4.418e-4 to 4.639e-4. Inference only; see `implemented.md` §124.
- v0.2.096: compact f16 inference now supports Dropout as a device-resident identity, matching inference semantics. Full Oxide f16 suite passed 15 tests; relative max error vs f32 was 3.890e-4. See `implemented.md` §123.
- v0.2.095: compact f16 inference now supports native ScaleChannels for per-channel and per-spatial scales. Full Oxide f16 suite passed 14 tests; batch-2 relative max errors vs f32 were 6.633e-4/1.701e-3. Inference only; see `implemented.md` §122.
- v0.2.094: compact f16 inference now supports native ChannelSlice through half-storage source selection/copy. Full Oxide f16 suite passed 13 tests; batch-2 output relative max error vs f32 was 2.353e-4. Inference only; see `implemented.md` §121.
- v0.2.093: compact f16 inference now supports native Softmax with f32 stable group reductions. The full Oxide f16 suite passed 12 tests; group-count 1/2 relative max errors vs f32 were 4.106e-4/5.549e-4. Inference only; see `implemented.md` §120.
- v0.2.092: compact f16 inference now supports Region's layout transpose, objectness sigmoid and optional class softmax. The full f16 Oxide suite passed 11; batch-2 Region relative max errors vs f32 were 2.324e-4/2.856e-4 with softmax off/on. f16 training/backward and truth-loss/decode remain unsupported; see `implemented.md` §119.
- v0.2.091: added native Oxide f32 Region layout/activation forward and exact supplied-seed adjoint; full/compact output and gradients matched CPU on batch-2 non-square cases with softmax off/on. Actual-Oxide focused test and `MEM=3G ./tools/test_host.sh region` passed. Truth-driven Region loss/decode remain open; no compiled-C++ golden; see `implemented.md` §118.
- v0.2.090: compact f16 inference now supports native Reorg forward/reverse permutations; batch-2 non-square odd-size actual-Oxide output matched f32 with relative max error 4.646e-4. Full f16 suite passed 10 before a shape refinement; the refined Reorg test passed. Inference only; see `implemented.md` §117.
- v0.2.089: compact f16 inference now supports native `[channel_shuffle]`; batch-2 actual-Oxide output matched f32 with relative max error 2.437e-4. Full `oxide_runtime_f16` suite passed 9 tests. Inference only; see `implemented.md` §116.
- v0.2.088: compact f16 inference now supports SAM for mapped activations using a native half-storage product/activation kernel. The full `oxide_runtime_f16` suite passed 8 tests; non-adjacent-source SAM relative max error was 6.223e-4. Also corrected compact GPU source lifetime accounting for SAM references. Inference only; see `implemented.md` §115.
- v0.2.087: compact Oxide f16 inference now supports `maxpool_depth` using native half-storage channel-group max pooling. The full `oxide_runtime_f16` suite passed 7; depth-pool f16 vs f32 relative max error was 2.488e-4. No f16 training/backward or C++ golden claim; see `implemented.md` §113.
- v0.1.031: pinned Connected's flattened NCHW input, `[out][in]` weight rows, batch-local forward rows, and supplied-seed adjoint with a hand-computed B=2 fixture. `MEM=3G ./tools/test_host.sh axis_layout` passed 27; `every_layer_kind_has_an_axis_pin_decision` passed. Ledger now has 24 pinned / 36 open C++-backed variants. Hand-derived from `Jev_laya:src-lib/connected_layer.cpp:186`; no compiled-C++ golden; see `implemented.md` §114.
- v0.1.030: pinned LocalAvgPool's window geometry, per-window averaging, batch/channel order and overlap-summed adjoint with a hand-computed non-square B=2 fixture. `MEM=3G ./tools/test_host.sh axis_layout` passed 26; `every_layer_kind_has_an_axis_pin_decision` passed. Ledger then had 23 pinned / 37 open C++-backed variants. No compiled-C++ golden; see `implemented.md` §112.
- v0.1.029: pinned AvgPool's per-sample/per-channel `H*W` reduction and supplied-seed adjoint with a hand-computed non-square B=2 fixture. `axis_layout` passed 25; the axis registry passed. Ledger now has 22 pinned / 38 open C++-backed variants. No compiled-C++ golden; see `implemented.md` §111.
- v0.1.028: pinned MaxPool depth and spatial axis contracts with hand-computed batch-2 fixtures, including odd kernel, unequal strides, padding, first-wins ties, and `zero_nonmax` backward. `axis_layout` passed 24 and the axis registry passed. Ledger now has 21 pinned / 39 open C++-backed variants. No compiled-C++ golden; see `implemented.md` §110.
- v0.2.086: native Oxide f32 `maxpool_depth` forward/backward is wired through the full GPU runtime; focused host and actual-Oxide maxpool tests pass. f16 depth pooling remains unsupported; no C++ parity/full-suite claim. See `implemented.md` §109.
- v0.2.085: CRNN convolution projection matrices are eligible for GPU Muon; actual-Oxide two-step CPU parity with BN enabled passes, including outputs, weights, biases and BN scales. Gradients, optimizer state and final updates remain host-managed. No C++ parity/full-suite claim; see `implemented.md` §108.
- v0.2.084: CRNN training with rolling-stat BN now uses native convolutional recurrent forward/backward. Actual-Oxide CPU output, gradients and parameter updates pass `batch_normalized_crnn_device_backward_matches_cpu_on_oxide`; the host FakeBackend BN regression also passes. No C++ parity/full-suite claim; see `implemented.md` §107.
- v0.2.083: LSTM BN training now uses the existing native Oxide recurrent forward/backward path with rolling-stat affine folding. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain lstm` passed 4 tests, including BN output/parameter-update CPU parity. No C++ parity/full-suite claim; see `implemented.md` §106.
- v0.2.082: RNN training with rolling-stat BN now uses native Oxide forward projections plus the existing native recurrent BN backward/update path. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain rnn` passed 6 tests, including full output and parameter-update parity against CPU for the BN case. Unsupported recurrent activations remain host fallback; no C++ parity/full-suite claim. See `implemented.md` §105.
- v0.2.081: native f32 LSTM inference now handles rolling-stat BN by folding it into all eight recurrent/input projections; BN training remains on the host fallback. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain lstm` passed 4 tests, including new multi-step BN inference CPU parity and existing LSTM training/Muon regressions. No C++ parity/full-suite claim; see `implemented.md` §104.
- v0.2.080: RNN GPU inference now keeps the recurrent projections on-device with rolling-stat BN folded into weights/biases; BN training remains on the host path. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain rnn` passed 6 tests, including the new multi-step BN inference parity test and existing RNN/CRNN training regressions. No C++ parity/full-suite claim; see `implemented.md` §103.
- v0.2.079: native f32 GPU inference for `[gru]`, including rolling-stat BN, now matches the Rust CPU reference across multiple sequence steps. GPU training/backward remains explicitly unsupported. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gru` passed 3 tests. No C++ parity/full-suite claim; see `implemented.md` §102.
- v0.1.027: Region CPU forward now follows C++ flatten/objectness logistic/optional class softmax and supplies the exact seeded adjoint; added a named C++-source-derived axis pin. `axis_layout` passed 22 tests, the axis registry passed, and all 12 `[region]` cfgs passed isolated CPU inference. GPU inference rejects explicitly; truth-driven loss/decode and native GPU remain open. No compiled-C++ golden/full-suite claim; see `implemented.md` §101.
- v0.2.078: GPU `[gru]` training rejects with `UnsupportedLayer` instead of a misleading configuration mismatch; focused Oxide regression remains green. Native f32 inference is added in v0.2.079; training/backward remains open. See `implemented.md` §100.
- v0.2.077: corrected the DCNv4 mixed-network test to use the matching native GPU backward cache instead of passing its host snapshot into CPU backward. Focused test and all 62 `oxide_runtime_chain` tests pass. This is not full-suite or compiled-C++ evidence; see `implemented.md` §99.
- v0.2.076: GPU Crop training now runs native random geometry/flip/rotation and matching affine sampling, with explicit rejection for training HSV/color adjustments. Actual-Oxide Conv→Crop SGD/Adam parity, primitive forward/backward/FD, and host crop tests pass. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain crop_native_training_augmentation_and_backward_match_cpu` passed 1; `oxide_pool_parity crop_bilinear_forward_backward_and_fd` passed 1; `MEM=3G ./tools/test_host.sh crop` passed 10. No compiled-C++ golden/full-suite/performance claim; see `implemented.md` §98.
- v0.2.075: standalone BatchNorm GPU training now runs native batch-statistics forward/backward with GPU cache and CPU-reference-compatible rolling statistics; scale/bias optimizer application remains host AdamW. Actual-Oxide CPU parity passed, and `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain batch_norm` passed 9 tests. No compiled-C++ golden/full-suite/performance claim; see `implemented.md` §97.
- v0.2.074: GraphConv BN training is now native for eligible edge modes/activations, with native BN forward/backward, GPU BN cache, and CPU-reference-compatible rolling stats. Actual-Oxide grouped learned-edge+self parity passes across SGD and Adam; the BN GPU Muon test also passes. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv` passed 9 tests, plus focused host inference and Muon-rejection checks. Unsupported variants retain fallback; no C++ golden/full-suite/performance claim. See `implemented.md` §96.
- v0.2.073: GraphConv GPU Muon now accepts BN variants through the existing CPU forward/backward fallback; Oxide still supplies matrix orthogonalization and parameter parity matches CPU across two updates, including BN affine and rolling stats. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_batch_norm_cpu_fallback_matches_cpu` passed 1 test. This is not native GraphConv BN training; no compiled-C++ golden/full-suite/performance claim. See `implemented.md` §95.
- v0.2.072: expanded GraphConv GPU Muon CPU parity, DNOPT6 resume, and bias coverage to `groups=2` learned-edge+self. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps` passes 1 actual-Oxide test. Other groups/configs and overall parity remain open; no compiled-C++ golden/full-suite/performance claim. See `implemented.md` §94.
- v0.2.071: verified GraphConv GPU Muon's `dont_update=1` gate on actual Oxide: outputs match CPU, layer parameters remain unchanged, and the global optimizer step advances. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_dont_update_gate_preserves_parameters` passed 1 test. Coverage is limited to no-BN learned-edge+self; no compiled-C++ golden/full-suite/performance claim. See `implemented.md` §93.
- v0.2.070: extended GraphConv GPU Muon parity through DNOPT6 save/load and a resumed third update. Fresh CPU/GPU executables match on resumed outputs and all tested GraphConv parameters. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps` passes 1 actual-Oxide test. This covers only learned-edge+self non-BN GraphConv; no compiled-C++ golden/full-suite/performance claim. See `implemented.md` §92.
- v0.2.069: hybrid GPU Muon now accepts non-BN GraphConv projection, self-projection, and learned-edge matrices. Actual-Oxide two-step CPU parity covers all matrix blocks and host BN rejection is verified. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps` and the focused `tools/test_host.sh` rejection test pass. Gradients, Muon state, and final updates remain host-staged; BN remains unsupported. See `implemented.md` §91. No compiled-C++ golden/full-suite/performance claim.
- v0.2.068: ordinary f32 GraphConv GPU inference now uses the native edge/self/projection path with rolling-stat BN; training BN remains fallback. FakeBackend verifies no CPU GraphConv cache is built, and actual Oxide full+compact outputs match CPU: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu` passes 2 tests; `MEM=3G ./tools/test_host.sh graph_conv_batchnorm_inference_uses_gpu_backend_without_cpu_cache` passes 1. See `implemented.md` §90. No compiled-C++ golden/full-suite/performance claim.
- v0.2.067: compact f32 GraphConv inference now applies BN from rolling stats as well as gamma/beta; training BN remains on fallback. Actual-Oxide CPU parity passes for learned edges+self with and without BN: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu` passes 2 tests. See `implemented.md` §89. No compiled-C++ golden/full-suite/performance claim.
- v0.2.066: added compact f32 GPU inference for eligible GraphConv modes (uniform/learned, optional self, no BN, supported activations). Actual-Oxide output matches CPU for learned edges plus self projection. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu_with_learned_edges_and_self` passes 1 test. See `implemented.md` §88. Unsupported variants and C++ golden remain open; no full-suite/performance claim.
- v0.2.065: integrated GraphConv self projection into native no-BN runtime forward/backward, including host-staged self-weight optimizer gradients. Actual-Oxide CPU output and SGD/Adam parameter parity pass. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_self_projection_runtime_forward_backward_and_sgd_adam_match_cpu` passes 1 test. See `implemented.md` §87. BN/unsupported variants/compact inference and C++ golden remain open; no full-suite/performance claim.
- v0.2.064: added native Oxide GraphConv center-reference gather and its adjoint, with checked host references, actual-device parity, and all-input finite differences. This is a primitive only; self-projection runtime remains open. `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` passes 5 tests. See `implemented.md` §86. No C++ golden/full-suite/performance claim.
- v0.2.063: integrated GraphConv learned-edge mode 1 forward/backward into the native Oxide runtime for no-self/no-BN supported activations; edge gradients stage to host optimizer buffers. Actual-Oxide output and SGD/Adam parameter parity pass against CPU. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_learned_runtime_forward_backward_and_updates_match_cpu` passes 1 test. See `implemented.md` §85. Unsupported cases retain fallback; no C++ golden/full-suite/performance claim.
- v0.2.062: added native Oxide GraphConv learned-edge input, edge-kernel, and edge-bias adjoints; actual-device outputs match the checked host adjoint and host finite differences pass. No runtime integration. `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` passes 4 tests. See `implemented.md` §84. No C++ golden/full-suite/performance claim.
- v0.2.061: added an actual-Oxide native GraphConv learned-edge mode 1 forward primitive with host-reference parity and invalid parameter/group rejection. Backward and runtime dispatch are still open; host FD covers the reference only. `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` passes 4 tests. See `implemented.md` §83. No C++ golden/full-suite/performance claim.
- v0.2.060: added a checked host contract for GraphConv learned-edge mode 1 masked-softmax aggregation and its input/kernel/bias adjoint, with finite differences and shape guards. This is not a native learned-edge GPU path. `oxide_graph_conv` passes 4 tests; direct `darknet-cuda` unit tests are blocked by an unrelated existing compile error in `host_ops/local.rs`. See `implemented.md` §82. No C++ golden/full-suite/performance claim.
- v0.2.059: native Oxide GraphConv runtime f32 forward/backward for uniform edge mode 0, no self projection/BN, with actual-device output and SGD/Adam update parity through an upstream convolution. Gradients download to host optimizer buffers; unsupported variants use CPU fallback and compact inference remains unsupported. Host GraphConv 15 and focused Oxide runtime test pass. See `implemented.md` §81. The broader runtime-chain run had 50 pass / 1 DCNv4 cache failure; no full-suite/C++-golden/performance claim.
- v0.2.058: hardened GraphConv primitive geometry validation against i64 coordinate overflow; the actual-Oxide GraphConv suite passes 3 tests, including malformed shape and overflow guards. Runtime dispatch remains unwired. See `implemented.md` §80. No C++ golden/full-suite/performance claim.
- v0.2.057: added native Oxide f32 uniform valid-neighbor aggregation forward/adjoint primitives for GraphConv edge mode 0; host-reference/FD, asymmetric geometry and malformed-shape tests passed. This is not runtime GraphConv GPU dispatch: learned edges, self projection, BN, runtime cache/epilogue and optimizer integration remain open. Existing host GraphConv filter passed 15. See `implemented.md` §79. No C++ golden/full-suite/performance claim.
- v0.2.056: GPU Muon now supports non-BN deform_conv main/offset/mask matrices. Native backward gradients are already staged into host buffers; Oxide performs orthogonalization and host state/final updates apply. Actual-Oxide two-step CPU parity with accumulated backwards passed; BN deform_conv remains rejected. See `implemented.md` §78. No C++ golden/full-suite/performance claim.
- v0.2.055: hybrid GPU Muon now supports the non-BN DCNv4 offset-projection matrix; device gradients are flushed per eligible layer, offset projection uses the existing Muon scale, and cached parameters refresh on the next forward. Actual-Oxide two-step CPU parity passed with accumulated backwards. BN DCNv4 remains rejected. See `implemented.md` §77. No C++ golden/full-suite/performance claim.
- v0.2.054: hybrid GPU Muon now supports the MoE router and expert matrices when expert BN is disabled; device gradients are staged at optimizer boundaries, expert caches refreshed, and consumed buffers cleared. Actual-Oxide two-step CPU parity passed with two accumulated backward calls per step. BN MoE remains rejected. See `implemented.md` §76. No C++ golden/full-suite/performance claim.
- v0.2.053: hybrid GPU Muon now supports both non-BN EML convolution branches with optimizer-boundary gradient staging, Oxide orthogonalization, cache refresh, and device-gradient reset. Actual-Oxide two-step CPU parity passed with two accumulated backward calls per step; BN EML remains rejected. Host `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. See `implemented.md` §75. No C++ golden/full-suite/performance claim.
- v0.2.052: hybrid GPU Muon now accepts standard non-BN convolution matrices; device weight/bias gradients are staged at optimizer boundaries, Oxide performs Newton–Schulz, host updates are applied, device parameters are refreshed and consumed gradients cleared. Actual-Oxide grouped-convolution CPU parity passed across two updates with two accumulated backward calls each; BN convolutions remain rejected. `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. See `implemented.md` §74. No C++ golden/full-suite/performance claim.
- v0.2.051: native f32 SAM forward/backward now uses backend pointwise multiply/activation and fans source-gradient contributions into the correct earlier layer. Actual-Oxide two-step CPU parity passed for both upstream convs and outputs; unsupported activations preserve host fallback. See `implemented.md` §73. No C++ golden/full-suite/performance claim.
- v0.2.050: extended hybrid GPU Muon to LSTM's four input and four recurrent matrices; actual-Oxide two-step CPU parity and host `gpu_muon_` filter (4) passed. Gradients/state/final updates remain host-staged; no C++ golden/full-suite/performance claim. See `implemented.md` §72.
- v0.2.049: extended hybrid GPU Muon to RNN input/recurrent/output matrices; actual-Oxide two-step CPU parity and host `gpu_muon_` filter (4) passed. RNN gradients/state/final updates remain host-staged; no C++ golden/full-suite/performance claim. See `implemented.md` §71.
- v0.2.048: added native Oxide f32 `[local]` forward/backward with supported-activation runtime dispatch and hybrid GPU Muon eligibility for each location-specific matrix. Oxide primitive parity (2), actual-GPU two-step SGD and Muon CPU parity, host `local` filter (11), and host `gpu_muon_` filter (4) passed. Local optimizer updates/state remain host-staged. See `implemented.md` §70; no C++ golden/full-suite/performance claim.
- v0.2.047 (`fb05dad0`): extended hybrid GPU Muon to KimiKda matrices; actual-Oxide two-step CPU parity and host `gpu_muon_` filter passed. KimiKda layer ops and optimizer state/final updates remain host-side. See `implemented.md` §69.
- v0.2.046 (`dd40ca39`): extended hybrid GPU Muon to MambaVision matrices; actual-Oxide two-step CPU parity and host `gpu_muon_` filter passed. MambaVision layer ops and optimizer state/final updates remain host-side. See `implemented.md` §69.
- v0.2.045 (`5bff2104`): extended hybrid GPU Muon to TuckerAttention matrices; actual-Oxide two-step CPU parity and host `gpu_muon_` filter passed. TuckerAttention layer ops and optimizer state/final updates remain host-side. See `implemented.md` §69.
- v0.2.044 (`32a111b7`): extended hybrid GPU Muon eligibility to Swin matrices; two-step actual-Oxide CPU parity and host `gpu_muon_` regression filter passed. Swin layer ops and optimizer state/final updates remain host-side. See `implemented.md` §69.
- v0.2.043 (`506dc9bd`): GPU Muon now supports eligible ViT matrices through native Oxide orthogonalization with host-staged gradients/state/final updates; two-step actual-device CPU parity and host `gpu_muon_` regression filter passed. ViT layer forward/backward remain host-side. See `implemented.md` §69.
- v0.2.042 (`56bfbe2f`): actual-device connected Muon `train_only_bn` gate matched CPU across two updates; parameters remain unchanged under the gate. Other gates/families remain open. See `implemented.md` §69.
- v0.2.041 (`4edd8860`): actual-Oxide connected GPU Muon with batch normalization matched CPU for outputs, parameters, scales and rolling stats across two updates. It exercises the host-backward fallback, not native connected BN backward; other Muon families/gates remain open. See `implemented.md` §69.
- v0.2.040 (`798d3389`): actual-Oxide deconvolution runtime-chain CPU parity passed for two-step SGD and Adam updates. Optimizer staging remains host-side; focused test only, no full-suite/C++ claim. See `implemented.md` §69.
- v0.2.039 (`b1fef3f9`): added native Oxide f32 deconvolution forward/backward runtime operations with legacy host routing for unmapped activations. Actual-GPU CPU-parity chain and host SGD/Adam path tests passed; optimizer staging remains host-side. No full-suite/C++ claim; see `implemented.md` §69.
- v0.2.038 (`fadfcb95`): actual-GPU CPU parity passed for connected Muon `burnin_update` gating (blocked before burn-in, update after). Other gate variants/connected BN remain open. See `implemented.md` §69.
- v0.2.037 (`281cdc7c`): deconvolutional GPU Muon eligibility uses Oxide Newton–Schulz with host fallback gradients/state/update; actual-GPU two-step CPU parity and host rejection regression passed. This is not native deconvolution forward/backward. See `implemented.md` §69.
- v0.2.036 (`86b2fbda`): actual-GPU two-layer `dont_update` gate test passed against CPU across two updates. Other gate variants, connected BN and eligible families remain open. See `implemented.md` §69.
- v0.2.035 (`cc0c00c1`): actual-GPU CPU parity passed for two connected Muon layers across two steps; FakeBackend `dont_update=1` gate test also passed. GPU-Muon host filter 4 tests and Muon regression filter 25 passed. Other eligible families and actual-device gate coverage remained open. See `implemented.md` §69.
- v0.2.034 (`f27b370c`): connected GPU Muon now uses native device momentum/lookahead and final direction+decay application, with momentum tensors cached between steps; host mirrors remain for DNOPT6 and weights stay host-canonical. Native `oxide_muon` 3 tests, host `darknet-cuda` Muon 4, GPU-Muon host filter 3, Muon regression filter 24, and actual-GPU connected two-step/resume parity 1 passed. Other eligible families remain unsupported. See `implemented.md` §69; no C++ golden/full-suite claim.
- v0.2.033 (`d73a87e7`): added FakeBackend CPU parity for the standalone BatchNorm AdamW fallback on the GPU Muon dispatch; focused GPU-Muon host filter 3 tests and Muon filter 24 tests passed. This adds no native layer coverage. See `implemented.md` §69.
- v0.2.032 (`2ddcaa6b`): connected-layer GPU Muon runtime slice uses the native Oxide Newton–Schulz primitive, host-side AdamW fallback and host-canonical state; DNOPT6 state round-trip and actual-GPU two-step plus resumed CPU parity passed. Other parameter families and device-resident updates remain unsupported/open. See `implemented.md` §69; no C++ golden/full-suite claim.
- v0.2.031 (`b46ad9c8`): standalone native Oxide f32 Muon Newton–Schulz orthogonalization is exposed via `F32TensorOpsBackend`; actual-GPU host-reference parity passed for square/tall/wide matrices, zero, and bad shapes (`oxide_muon`, 2 tests); the host reference's zero/non-finite and transpose-equivariance tests passed (2). This alone was not runtime integration. See `implemented.md` §68; no C++ golden/full-suite claim.
- v0.2.030 (`6433f57f`): native DCNv4 learned parameter tensors are cached per layer and reused across forwards and SGD/Adam updates; updates replace the cached device tensors while refreshing host-canonical mirrors. Exact host snapshots detect external parameter edits. Host DCNv4 29, cache invalidation/upload-reduction test and actual-GPU two-step CPU parity passed. Host downloads remain; no speedup/C++/full-suite claim; see `implemented.md` §67.
- v0.2.029 (`9b722d35`): DCNv4's per-forward cache retains the uploaded offset-projection weights for backward, eliminating its second weight upload and keeping the input adjoint tied to the forward snapshot. Host DCNv4 29 and actual-GPU two-step optimizer parity passed. Parameters still upload on each forward/update; no network-wide residency, speedup or C++ claim; see `implemented.md` §66.
- v0.2.028 (`d7b0e5cc`): eligible DCNv4 Adam moments persist on-device between updates and are materialized into the existing sidecar only on save; parameter tensors remain host-canonical. Host DCNv4 29 and sidecar round-trip passed; focused actual-GPU two-step parity passed on rerun after one near-threshold failure. No speedup/C++/full-suite claim; see `implemented.md` §65.
- v0.2.027 (`e5217f8a`): native DCNv4 now supports through 15x15/225 active points; GPU composition and runtime two-step SGD/Adam parity with CPU passed, and >225 retains host fallback. Host DCNv4 29; Oxide composition 3; focused actual-GPU runtime 1 passed. No speedup, compiled-C++ or full-suite claim; see `implemented.md` §64.
- v0.2.026 (`fafa95ae`): `Revleaky` now maps to the existing native Leaky f32 operation with matching Rust forward/backward semantics; DCNv4 actual-device two-step SGD/Adam CPU parity now includes Revleaky. Host DCNv4 29 and focused RTX 4060 runtime test passed. No speedup, compiled-C++ or full-suite claim; see `implemented.md` §63.
- v0.2.025 (`7c96d45a`): eligible native DCNv4 Adam now runs device optimizer operations while moments round-trip through the existing host-sidecar state; parameters remain host-canonical. Host DCNv4 29 and focused actual-GPU two-step SGD/Adam CPU-parity tests passed. No speedup, compiled-C++ or full-suite claim; see `implemented.md` §62.
- v0.2.024 (`0087d109`): eligible native DCNv4 SGD now uses device gradients and momentum for updates; Adam and canonical parameters remain host-side. Host DCNv4 28 and focused actual-GPU two-step SGD/Adam CPU-parity test passed. No speedup, compiled-C++ or full-suite claim; see `implemented.md` §61.
- v0.2.023 (`2ff4544e`): added the native Oxide full-sequence CLS-adjoint scatter for act-head gradients (NCHW token-zero additive scatter) and checked host reference. The complete `oxide_laya_act_backward` suite passed 5 on RTX 4060; host `laya_act` 11 and darknet-cuda helper 2 passed. This does not wire runtime backward/updates or resolve act-head training eligibility; no C++/performance claim.
- v0.2.022 (`27ab3cfb`): native DCNv4 bias/offset/BN-scale gradients accumulate as device tensors across microbatches; one gradient transfer occurs at the host optimizer boundary. SGD/Adam parameter updates/residency remain unchanged. Host DCNv4 28 and focused RTX 4060 two-step runtime SGD/Adam parity passed. No speedup or compiled-C++ claim.
- v0.2.021 (`7a262a13`): native DCNv4 per-forward GPU state now retains raw output and offsets for backward, eliminating repeated projection/upsampling/aggregation work. Host params/gradients/optimizer remain. Host dcnv4 28 and focused RTX 4060 runtime-chain test pass; no speedup claim.
- v0.2.020 (`c99f83d8`): DCNv4 native/runtime bound now covers 13x13 (169 points), with real-GPU composition output/gradient parity and two-step SGD/Adam CPU parity, BN on/off. Focused oxide tests and host DCNv4 28 pass; >169 remains host fallback. No compiled-C++ or performance claim.
- v0.2.019 (`76242245`): DCNv4 native/runtime bound is now 121 active points (11x11); real-GPU composition and two-step SGD/Adam CPU parity pass with BN on/off. >121 and unsupported variants retain host fallback; no compiled-C++/performance claim.
- v0.2.018 (`613b243e`): native DCNv4 supports through 9x9/81 points, with real-GPU composition and two-step runtime SGD/Adam CPU parity (BN on/off). Oxide composition, runtime-chain target and host `dcnv4` (28) pass. >81 and unsupported variants remain host fallback; no compiled-C++ or speedup claim.
- v0.2.017 (`8a3ef3b0`): actual GPU runtime 7x7 DCNv4 two-step SGD/Adam parity with CPU, BN on/off; native-path/fallback exclusion and parameter movement are covered. Targeted oxide runtime test passed (1, ~108 s). >7x7 remains host; no compiled-C++ or performance claim.
- v0.2.016 (`a96d6d95`): native DCNv4 fixed mask/kernel bound extended to 49 active points (7x7); runtime eligibility matches the bound. `oxide_dcnv4_chain` 3 passed including GPU composition parity for 7x7; `test_host.sh dcnv4` 28 passed. >49 and other unsupported variants remain on host. No 7x7 runtime optimizer parity, compiled-C++ parity, or performance claim.
- v0.2.015 (`1fc80f27`): DCNv4's bounded f32 native path now accepts `normalize_channels`, `_softmax` and `_softmax_maxval` activations (after optional rolling-stat BN) using the existing `normalize_channels_f32`/backward primitives; channel output is recomputed on device from the forward preactivation. Exact `SoftmaxMaxval` adjoint gained the max-chain term (epsilon sits outside exp(x-max); first max selected at ties; one-channel case is constant, zero translation derivative) in both host_ops and the Oxide kernel; `GradMode::Cpp` shortcut unchanged. Gates (serial, 3G): darknet-cuda lib 40, host dcnv4 28, `runtime::` 284, `oxide_dcnv4_normalization` 3 (3 norms x BN x SGD/Adam two-step CPU/GPU parity, GPU forward FD with fixed stats, mixed-network CPU cache), `oxide_layers_extra` 25. Not compiled-C++ parity; larger variants, other activations, compact/f16, device updates/residency, GPU Muon remain open.
- v0.2.014 (`ceef2037`): native DCNv4 rolling-stat BN affine/adjoint and post-backward 0.9 EMA (unbiased variance, including unseeded/gated layers), preserving the existing Rust CPU semantics and ignored bias/beta fields. Uses existing device primitives; no new kernels/state formats. Serial 3G gates: host `dcnv4` 26, mandatory `runtime::` 282, host-cuda 37; new device BN 2 (actual GPU FD with fixed rolling stats, zero/negative gamma, N=1/unseeded EMA), runtime DCNv4 3 (BN on/off × SGD/Adam × masks/center modes, two steps), raw composition 3 and aggregate 6; CLI oxide-feature check and targeted formatting/diff checks passed. Audit in `implemented.md` §51. Parameter gradients/updates remain host-side; uploads/raw recomputation persist. Mixed/unsupported/large variants keep host caches; compact/f16 and all CPU/GPU/Muon/MoE gaps remain. Next: native DCNv4 NormalizeChannels* activation/adjoint composition. No C++ training-BN parity or speedup claim; previous full runtime-chain ViT timeout remains unresolved, not a pass.
- v0.2.013 (`3f3c2226`): bounded DCNv4 native f32 runtime forward/backward (no BN, <=25 active points, supported elementwise activation, no other fallback/host-only deform_conv). Trait defaults plus native Oxide overrides; helper in `runtime/dcnv4_gpu.rs`. Backward recomputes offsets on device; parameter gradients/SGD/Adam remain host-resident, f16/compact unchanged. Serial 3G gates: host-cuda 37, host `dcnv4` 22, mandatory `runtime::` 278; device runtime `dcnv4` 3 (two-step SGD/Adam with upstream conv/all blocks moving, BN/mixed fallback), composition 3, aggregate 6; CLI oxide-feature check and targeted Rustfmt/diff checks passed. Full runtime-chain timed out at 600 s in ViT after 25 preceding tests reported ok; not a pass/full-suite green baseline. Audit in `implemented.md` §50. Next: native DCNv4 rolling-stat BN affine/adjoint (preserve CPU semantics); unsupported/large variants, compact/f16, device updates/residency, GPU Muon and broader CPU/GPU/MoE scope remain open. No performance/training/compiled-C++ claim.
- v0.2.012 (`c51b25f6`): standalone native DCNv4 coarse predictor/upsample + aggregation forward/backward composition (raw math, not runtime wiring). Corrected shared stage layout: K=size²-remove_center and padded row stride; v0.2.009–011's host/device consistency was not real CPU agreement. Real CPU oracle covers 18 cases and every output/input/offset parameter gradient; device FD covers all operand groups. Serial 3G gates: chain 3, aggregate 6, host-cuda 36, host `dcnv4` 17, Oxide fallback regression 1 passed; targeted Rustfmt/diff checks passed. Audit in `implemented.md` §49. Next: trait/default migration, coordinated runtime cache/forward/backward and epilogue, then two-step optimizer parity. Host fallback/optimizers/f16 unchanged; no benchmark/training/full-suite/compiled-C++ claim; overall CPU/GPU/Muon/MoE work remains open.
- v0.2.011 (`8fe32a09`): DCNv4 native f32 aggregation adjoint (offsets given), following host reference v0.2.009 (`e3cfb27d`) and native aggregation forward v0.2.010 (`7874c225`). Fixed batch-zero-only thread bound in the pending kernel; shared geometry rejects overflow before unsafe arithmetic. Serial 3G scopes: `oxide_dcnv4_aggregate` 6 passed (host parity, actual-device FD, all mask/center modes, signed seeds, batches/groups, stride/dilation/edge/error coverage), host-cuda 35, host `dcnv4` 17, Oxide runtime fallback `dcnv4` 1. Targeted Rustfmt/diff checks passed; audit in `implemented.md` §48. Runtime remains host fallback; full predictor/upsample/backward composition and parameter/update parity are next. No f16, training, full-suite, benchmark or compiled-C++ golden claim; CPU/GPU/Muon/sparse-MoE gaps remain open.
- v0.2.006 (`2ffc33e2`): standalone native f32 act-head gradient API returns all four batch-summed parameter gradients plus `[B,H]` dcls, composing native feature/dense/exact-erf/backward/route operations. No runtime/eligibility/update/f16 change. Immutable checked MLP descriptor and native operand/metadata guards added. RTX 4060/Fp32, serial 3G scopes: new backward suite 4 passed, forward regression 2, exact-erf regression 4, host act-head 11. CPU oracle compacts only ignored extra marker channels because its full-helper validator requires exact 2*K; details, GPU FD and allocator-lifetime audit in `implemented.md` §46. Targeted Rustfmt/diff checks passed. Next independent math: full-sequence CLS-adjoint scatter; runtime backward remains CPU and the training question is unanswered. No full-suite/performance/compiled-C++ parity claim.
- v0.2.005 (`79ce20bb`): act-head f32 full/compact forward now composes native feature, dense and exact-erf GELU ops instead of downloading inputs for host MLP math. Four parameter blocks upload each call (no residency/speedup claim), ensuring warm-call edits and in-memory blob replacement are visible. RTX 4060/Fp32 real tests 2 passed; exact-erf regression 4; host act-head 11 and runtime 273 passed under serial 3G scopes. CPU-call probe, errors, source alias/lifetime and unchanged CPU-backward/frozen-update evidence: `implemented.md` §45. Targeted Rustfmt/diff checks passed. GPU status native f32 forward only; CPU backward remains, f16 rejected, training question and shared-file loader gaps open. Next independent slice: native gradient math without changing optimizer eligibility/application. No full-suite/performance/compiled-C++ parity claim.
- v0.2.004 (`eec5eab0`): dedicated native f32 exact-erf GELU forward/seeded backward via supported NVIDIA libdevice device-FFI; no compiler changes or tanh/cfg/AMP/f16 semantics changes. RTX 4060, serial 3G scopes: `oxide_gelu_erf` 4 passed, `oxide_grad_methods` 2 passed/1 ignored timed benchmark, act-head feature regression 3 passed, host `gelu` 2 passed. Targeted Rustfmt/diff checks passed. Closed-form, actual GPU FD, nonfinite/error-path and native-linkage audit: `implemented.md` §44. This is a primitive, not runtime integration; act-head still uses host fallback. Next independent slice: native f32 act-head forward composition; preserve CPU backward and current training behavior while the unanswered source-training question is resolved. No full-suite/performance/compiled-C++ parity claim.
- v0.2.003 (`3de7c327`): standalone native f32 act-head CLS plus detached summary-feature primitive, not yet wired into runtime. RTX 4060, serial 3G scopes: new `oxide_laya_act_features` 3 passed, extra-layer regression 24, f16 gather 4, host `laya_act_head` 9. Targeted Rustfmt/diff checks passed; source/requirement audit in `implemented.md` §43. Audit confirmed C++ act-head trains all four parameter blocks while Rust runtime discards their gradients; the frozen-vs-trainable question remains unanswered, so training behavior is unchanged. Existing GPU GELU is tanh-approximate, not the required exact erf variant. Next independent prerequisite: native exact-erf GELU; full MLP/backward/update and runtime fallback removal remain open. Act-head GPU status still host fallback; no full-layer/full-suite/compiled-C++ parity claim.
- v0.2.002 (`32822894`): native f16 marker-gather compact inference, with bit-preserving feature copies, device-side half marker decode/round/clamp, checked layouts/storage and correct compact source lifetimes. RTX 4060 Laptop, serial 3G scopes: `oxide_marker_gather_f16` 4 passed, full `oxide_runtime_f16` 6 passed, f32 `oxide_marker_gather` 4 passed and host `marker_gather` 12 passed. Targeted Rustfmt/diff checks passed; audit/coverage limits in `implemented.md` §42. No full-state f16 forward or f16 training API added, no benchmark/full-host/corpus/full-Oxide/compiled-C++ golden. GPU priority continues; next is the native LAYA act-head contract audit/port per `gpu_layers_plan.md`. CPU gaps/parked activation work and all other GPU/GPU Muon/sparse-MoE requirements remain open.
- v0.2.001 (`f56342ca`): operator requested GPU focus now, overriding CPU-first ordering without closing CPU gaps. Native Oxide f32 `[laya_marker_gather]` full/compact forward and exact deterministic backward replace host round trips and whole-net CPU-backward fallback. RTX 4060 Laptop verification under serial 3G scopes: `oxide_marker_gather` 4 passed; `oxide_runtime_chain -- marker_gather` 1 passed; full `oxide_layers_extra` 24 passed; host `marker_gather` 12 and `runtime::` 271 passed. Targeted Rustfmt/diff checks passed. See `implemented.md` §41 for source/requirement audit, FD/rounding/routing evidence and limits. GPU status native f32; f16 explicitly unsupported, GPU Muon and other layers remain open. Parked CPU activation work was not included in this release. No benchmark, training run, full-host/corpus/full-Oxide or compiled-C++ golden. Resume GPU work via `gpu_layers_plan.md`, directly without subagents.
- v0.1.026 (`755e5145`): integrated `[yolo_3d]` cfg/network aliases and shared class/K validation, raw CPU forward with separate typed cuboid outputs, exact supplied-seed identity backward, zero-byte checkpoint head, no head Adam/Muon state, GPU rejection and fail-closed legacy detector APIs/CLI consumers. `MEM=3G ./tools/test_host.sh` filters: `yolo_3d` 42 core + 1 CLI, `model::` 90, `runtime::` 270, `yolo` 106 + 2, `map` 27 + 8, `weights` 57 + 2, `axis_layout` 21, `dpt_head` 18, `onnx` 12 + 1; all in 3G scopes. Oxide-feature CLI check (`-j 4`, 3G scope), targeted Rustfmt and `git diff --check` passed. Scoped audit and test limitations: `implemented.md` §40. Supplied-seed optimizer tests do not prove cuboid training loss; f16 guard is tested through shared preflight, not device execution. Projected loss/decode/evaluation, A2 axis pin and depth/metric terms remain open; GPU status none. Hand-derived from source, no compiled-C++ golden; no training runs, Oxide tests, corpus or full-host run. Stage 1 remains incomplete.
- v0.1.025 (`8f6178ed`): added pure/mixed cuboid label parsers/loaders and deterministic truth helpers with source row precedence, tracks, visibility hulls, crop/flip intrinsics, batch-major prefix/vertices/padding, mixed flags and source mosaic zeroing. `MEM=3G ./tools/test_host.sh yolo_3d` passed 31 core tests (19 new), `label` passed 43; both in 3G systemd scopes. Test-first missing-type/function failures, final targeted Rustfmt and `git diff --check` recorded in `implemented.md` §39 with the scoped coverage audit. Source mosaic copies only entries 0–4, so mixed ownership is also zeroed. No augmentation sampler or network execution is enabled; enum/network/loss/decode/evaluation/optimizer/guard integration and depth/metric terms remain open. Hand-derived from source, no compiled-C++ golden; GPU status none. Stage 1 remains incomplete; no training, Oxide, corpus or full-host run.
- v0.1.024 (`4506e46e`): prepared strict `[yolo_3d]` config types/parser helpers only (`model/yolo_3d_config.rs`); network/runtime dispatch remains unsupported. Test-first compile failed on missing types, then `MEM=3G ./tools/test_host.sh yolo_3d_cfg` passed 12 core tests and `model::` passed 88, both in 3G systemd scopes; targeted Rustfmt and `git diff --check` passed. Aliases/defaults, projected channel arithmetic, source-only vs Rust-strict validation, and staged depth/metric rejection are covered; this is not an axis/decode or full layer-port pass. Hand-derived from source, no compiled-C++ golden; GPU status none. See `implemented.md` §38 for the coverage audit and remaining contracts. No training, Oxide, corpus or full-host run; Stage 1 remains incomplete.
- v0.1.001 (`8f5a9a94495cbc376f0b91cbf12f5123fe7ee42c`): added the source-derived DCNv1/2 CPU reference/runtime and f32 GPU host fallback, including config/shape, source-order weights, seeded initialization, exact backward, BN, SGD/Adam persistence, optimizer gates, and explicit compact rejection. `MEM=3G ./tools/test_host.sh deform_conv` passed 23 core tests, `weights` passed 38, `runtime` passed 184, and the targeted Oxide runtime-chain fallback test passed 1. No native kernel or compiled-C++ golden/parity; CPU Muon remains a cross-cutting Stage-1 requirement.
- v0.1.002 (`f49cfc56f571329c14f6b43224d635199b6cca1b`): added bounded single-template DSConv CPU support for both aliases and morph 0/1/2, exact Rust-forward backward, source-order weights, seeded init, SGD/Adam persistence, and explicit GPU rejection. `MEM=3G ./tools/test_host.sh dsconv` passed 12 core tests, `weights` passed 40, and `runtime` passed 187; targeted formatting and `git diff --check` passed. Multi-template morph 0/1 remains unsupported; no native kernel or compiled-C++ golden/parity, and CPU Muon remains open.
- v0.1.003 (`f9a90bd5940f0dfce40e4e2ded0d7c1838f50e11`): integrated source-derived `[graph_conv]` / `[graph_convolutional]` CPU forward/backward, strict config/shape, source-order weights, initialization, BN/activation, SGD and persisted Adam. `MEM=3G ./tools/test_host.sh graph_conv` passed 14 tests; `weights` passed 42; `runtime` passed 192. Grouped/axis hand math, learned and uniform boundary finite differences, input/all-parameter gradients including bias, preceding-convolution accumulation, optional weight layouts, optimizer state, and explicit GPU rejection are covered. No Oxide tests or training; no compiled-C++ golden (the source macro is disabled). CPU Muon and Stage-2 GPU support remain open.
- v0.1.004 (`49ce73ccfc03d76c9a79e53389d283198d5d788e`): added the source-derived WMHF Stage-1 CPU path across config, shape, initialization, exact backward, BN, SGD/Adam, optimizer-state persistence, source-order weights, and gradient reporting. `MEM=3G ./tools/test_host.sh wmhf_` passed 17 tests, `weights` passed 43, and `runtime` passed 198. GPU execution/updates reject explicitly; no compiled-C++ golden (the source macro is disabled), native GPU path, Oxide test, or training run. CPU Muon is integrated in v0.1.005; `clifford` and `recursive_block` remained open at that increment.
- v0.1.005 (`2674f5741405e33371448bb9a664d482ffa02e8d`): delivered opt-in hybrid CPU Muon/AdamW with DNOPT6 state persistence. `MEM=3G ./tools/test_host.sh muon_` passed 13 tests, `weights` 43, and `runtime` 204, including `train_only_bn` gating. No training, Oxide tests, or compiled-C++ golden/parity; GPU Muon remains Stage 2.
- v0.1.006 (`1cc80d92b06a7b4fcbd90bea0f6a38931bb0d0ed`): delivered source-derived `[clifford]` Stage-1 CPU support across config, shape, initialization, exact Rust-forward backward, BN, source-order weights, SGD/AdamW/Muon, and explicit GPU/compact/update rejection. `MEM=3G ./tools/test_host.sh clifford_` passed 13 tests, `weights` 44, and `runtime` 208. Coverage includes input/all-parameter finite differences for avgpool+DropPath, Fourier, inner/abs/local, and wedge/global; weight byte-order/round-trip; optimizer state; and BN rolling stats. No training, Oxide tests, `cfg_corpus`, or compiled-C++ golden/parity. `recursive_block` remains Stage-1 CPU work.
- v0.1.007 (`94513d426c190ab8b7fb012ee3bf0b5f68ff5e55`): delivered source-derived `[recursive_block]` / `[recurrent_block]` Stage-1 CPU support with nested-section parsing, body-only weights, static and controller recurrence, lattice projection, per-pass caches/dropout masks, nested BN rolling stats, exact full-BPTT Rust backward, SGD/AdamW/Muon state handling, and explicit GPU/compact/update rejection. `MEM=3G ./tools/test_host.sh recursive_block` passed 11 tests, `weights` passed 45 core + 2 CLI tests, and `runtime` passed 219 tests; `MEM=3G cargo check -p darknet-core -j 4` passed. No training, Oxide tests, or feature-specific `cfg_corpus` run; the full host baseline/cfg-corpus status remains non-green/unverified under the memory cap. See `implemented.md` §24 and the evidence-gap record below. No compiled-C++ golden/parity.
- v0.1.008 (`f7b044bc84c6b19ed5b99cc97633fd859724db74`): integrated Stage-1 CPU `[hyperbolic_foveation]` / `[hf]` through strict cfg/shape, NCHW forward and exact finite-iteration backward, predictor/source-edge gradients, label regularizers, source-order weights, SGD/AdamW/Muon persistence, and axis-aligned/OBB/CARPN detection and mAP reprojection. GPU execution/backward/update paths reject explicitly; no GPU fallback or Oxide kernel. `MEM=3G ./tools/test_host.sh hyperbolic_foveation` passed 30 core tests, `runtime` 231, `weights` 47 core + 2 CLI, and `detect` 36 core + 10 CLI. Targeted rustfmt and `git diff --check` passed. No training, Oxide tests, feature-specific `cfg_corpus`, or compiled-C++ golden/parity; the source macro is disabled and the full host baseline/cfg-corpus remains non-green/unverified under the memory cap. See `implemented.md` §25.
- v0.1.009 (`51e381b8a94ae4e23ecc8054f6045a128f87fc8f`): integrated Stage-1 CPU `[foveation_film]` / `[hf_film]` with strict HF-source resolution, exact CPU forward/backward and learned-HF predictor-source fan-in, source-order weights, seeded initialization, SGD/AdamW/Muon updates/state validation, and explicit GPU/compact/f16/backward/update rejection. `MEM=3G ./tools/test_host.sh foveation_film` passed 12 core tests, `runtime` 237, and `weights` 49 core + 2 CLI. New math/runtime-test Rustfmt checks and `git diff --check` passed. No training, Oxide tests, or feature-specific `cfg_corpus`; the macro is disabled, so no compiled-C++ golden/parity is claimed. See `implemented.md` §26.
- v0.1.010 (`f0a3f2391655d120044bd13f6ded6ca6c655971e`): added source-derived Stage-1 CPU `[depth_anything]` with strict aliases/config, type-major and external/sequential weights, DINOv2 ViT-S/14 forward, complete Rust-mode input backward and `GradMode::Cpp` source no-op, frozen SGD/AdamW/Muon handling, and explicit GPU rejection. `MEM=3G ./tools/test_host.sh depth_anything` passed 16 tests, `runtime` 244, and `weights` 52; targeted Rustfmt and `git diff --check` passed. No compiled-C++ golden/parity, training, Oxide, or feature-specific `cfg_corpus` run. Full host/corpus status remains non-green/unverified; `[dpt_head]`, `[yolo_3d]`, native GPU, GPU Muon, and the broader roadmap remain open. See `implemented.md` §27.
- v0.1.011 (`a34aef8f`): added the initial named hand-computed axis/layout tests for ViT, MoE, EML, DCNv4, LAYA, channel slice, deconvolution, and shortcut. `axis_layout` passed 13 core tests; `laya_` passed 66. The remaining already-ported transformer-family axes were added in v0.1.012, completing A1 from `AGENTS.md`; missing-layer DoDs and the broader v0.1 roadmap remain open. No compiled-C++ golden/parity.
- v0.1.012 (`60db7b44`): completed the A1 axis audit across all already-ported families by pinning Swin window/shift order, Tucker batch/window/token/channel isolation, MambaVision NCHW/token and per-batch scan-state axes, and Kimi KDA raster scans with per-batch reset. `MEM=3G ./tools/test_host.sh axis_layout` passed 18 tests; focused suites passed `tucker_attention` 15, `mambavision` 17, and `kimi_kda` 11. Targeted edition-2024 Rustfmt and `git diff --check` passed. No behavior changes or compiled-C++ golden/parity; missing-layer axis DoDs and the v0.1–v0.3 roadmap remain open.
- v0.1.013 (`a995fe956907281affdf8b748b2862cecc39c03f`): added source-derived Stage-1 CPU `[dpt_head]` with strict aliases/config and shape validation, 66-slot source-order checkpointing, NCHW tap-major forward, analytical full-Rust backward, SGD/AdamW/Muon state handling, and explicit GPU/stream rejection. `MEM=3G ./tools/test_host.sh dpt_head` passed 18 tests, `runtime` 260, `weights` 55 core + 2 CLI, and `axis_layout` 19. Targeted edition-2024 Rustfmt and `git diff --check` passed. No training, Oxide tests, or full `cfg_corpus`; no compiled-C++ golden/parity. `[yolo_3d]`, native GPU, GPU Muon, and the broader roadmap remain open; see `implemented.md` §30.
- v0.1.014 (`fe3a554c`): added the source-derived parser regression `model::tests::unused_cpp_layer_enum_names_are_not_rust_noop_layers`, pinning `[network]` as metadata and rejecting `[blank]`, `[empty]`, and `[silence]`. `MEM=3G ./tools/test_host.sh unused_cpp_layer_enum_names_are_not_rust_noop_layers` passed 1 test; targeted edition-2024 Rustfmt and `git diff --check` passed. No runtime behavior change or compiled-C++ golden/parity; the broader roadmap remains open.
- v0.1.015 (`e848f4ef`): gated axis coverage. `AxisPin`/`AXIS_PINS` record one decision per `LayerKind` variant (64 plus `Unsupported`), and the exhaustive `LayerKind::axis_pin` match makes a new variant a build error until classified. Added `axis_fixture.rs` (`PROBE`, `axis_value` sentinel packing, `nchw_index`, `axis_tensor`) so pins share one notion of "distinct". `MEM=3G ./tools/test_host.sh every_layer_kind_has_an_axis_pin_decision` passed 1 test, `model::` 75, `axis_fixture` 3, and `axis_layout` 19 (unchanged). Baseline: 18 variants pinned, 41 C++-backed open, 5 Rust-only, 3 unported; ledger in `tasks/rust/axis-coverage.md`. No layer math changed; no compiled-C++ golden/parity.

- v0.1.016 (`051021fe`): added the first Tier 1 axis pins for `[convolutional]`: `convolutional_axis_layout_keeps_nchw_and_group_partitions` (channel-swapping 1x1 weights over `batch=2`, plus `groups=2` partitioning) and `convolutional_axis_layout_pins_dilated_extent_and_groups` (extent and `[out][in/groups]` weight cardinality). `MEM=3G ./tools/test_host.sh convolutional_axis_layout` passed 5, `axis_layout` 21 (was 19), and `runtime::` 261. Full CPU baseline: `cargo test -p darknet-core --lib -j 2` passed 897 and `-p darknet-cli` 28; GPU baseline `oxide_parity` 11 and `oxide_runtime_chain` 24. Corrects axis-coverage finding F1: `conv_dim` must keep ignoring dilation, because C++ `convolutional_out_width` does; adding the textbook term broke `cfg/resnet152_trident.cfg`. No compiled-C++ golden/parity.- v0.0.274 (`e24d7625`): matched `Jev_laya`'s config-side `try_fix_nan` default: deformable-sampling configs enable it when `[net]` omits the key, while explicit `try_fix_nan=0/1` is preserved. `MEM=3G ./tools/test_host.sh dcnv4` passed 16 tests. This closes a source-contract gap found in the DCNv4 audit; no compiled-C++ parity is claimed.

- v0.1.017 (`263fa448`): fixed the GPU convolution extent for `dilation > 1`. `Geom::new` used the textbook dilated span while the CPU/C++ `convolutional_out_width` ignores dilation, so the device produced a different map size and `expect_shape` rejected every dilated conv; the kernels already sampled and zero-filled correctly. f32 and f16 now use the CPU extent, and `oxide_cfg_shapes` asserts the CPU convention instead of the old `pad*dilation` workaround. Evidence: `oxide_conv_parity` 13, `oxide_cfg_shapes` 13 (6 dilated geometries, 12 runs, 0 failures), `oxide_gemm_mma` 5, `oxide_half` 13. No compiled-C++ golden/parity.
- v0.1.018 (`0ca7613e`): added `darknet-gui`, the opt-in egui/eframe annotator (`gui_plan.md` first slice): flat project scan, `*.names` class discovery, YOLO `.txt` load/save with 10-decimal round trip, drag-to-create and click-to-select boxes, per-image undo, and sidecar deletion when an image has no marks. `cargo build/test -p darknet-gui --offline` pass (7 tests). DarkMark JSON sidecars and the prediction overlay are not implemented.- v0.0.273 (`cb96aa49`): integrated `[dcnv4]` / `[dcnv4_convolutional]` with strict config/shape handling, source-order weights, seeded initialization, CPU reference forward and exact backward, BN, SGD/Adam with optimizer-state persistence, and GPU f32 host fallback. The focused `dcnv4`, `weights`, and `runtime` host filters and the targeted `oxide_runtime_chain` host-fallback test passed (test counts were not retained in the audit output). Coverage includes finite differences for input/offset projection, BN/optimizer behavior, exact-size round trips, and CPU/FakeBackend/Oxide runtime paths. No native kernel, training run, or compiled-C++ golden/parity.

- v0.1.019 (`fcd23bbd`): implemented the annotation GUI plan across three crates. `darknet-gui-project` (36 tests) covers DarkMark-compatible detector JSON with unknown-field preservation, `G4`/`G5` save rules, `G2` scanning, `D5` file preference, filters/sort, clamped navigation with unmarked walking, capped undo, the `G9`-`G16` mouse/edit model, interpolation/clipboard/snap/homography, project creation and an atomic class remap with backup. `darknet-gui-engine` (4 tests) speaks JSON lines to a child `detect-server` with a 30 s timeout. `darknet-cli` gained `detect-server` (28 CLI tests still pass). The `darknet-gui` shell wires the launcher, recents, image cache, canvas, views, class editor and prediction overlay. Not wired: add/delete classes from the dialog, delete-image/undo-delete, heatmaps, jump, screenshots, train runner, and the Task 1.5 DarkMark file diff.- v0.0.271: integrated the source-derived MambaVision reference path: strict cfg/shape handling, explicit per-layer recurrent-state API, exact CPU forward/backward, seeded initialization, SGD/Adam, source-order `.weights`, optimizer-state persistence, and GPU f32 host fallback. `MEM=3G ./tools/test_host.sh mambavision` passed 16 tests, `runtime` 171, and `weights` 33; coverage includes input/all-parameter/state finite differences, causal convolution, MSA clipping, initialization bounds, optimizer decay/persistence, serialization, and FakeBackend parity. No native MambaVision kernel or compiled-C++ golden/parity.

- v0.1.020 (`2640b902`): the GUI's "Load model" now resolves `DARKNET_RS_BIN`, then the current executable, then `darknet-rs` on `PATH`; it previously printed that advice without reading the variable. `cargo build -p darknet-gui --offline` passes.- v0.0.272 (`3d7202bf`): integrated source-derived Kimi KDA cfg/model support, exact CPU scan/MoE forward/backward including load-balance cost, seeded initialization, SGD/Adam with optimizer-state persistence, source-order weights, and GPU f32 host fallback. `MEM=3G ./tools/test_host.sh kimi_kda` passed 10 tests, `runtime` 173, and `weights` 34. Coverage includes hand math, input/all-parameter finite differences, norm-floor and top-k branches, serialization, initialization, and FakeBackend parity. No native kernel or compiled-C++ golden/parity.

- v0.1.021 (`8c536b6d`): re-transcribed the detector key map from `DarkMark/src-darkmark/DMContent.cpp::keyPressed()` instead of the `G16` summary and corrected the divergences (`r`/`R` sorts, `,`/`.` normalized delta, `o` mass-delete, `P` homography, `l`/`b`/`B` display modes, `d`/`D` snapping, `h`/`H` heatmaps, `s`/`S`/`F12` screenshots, `t`/`[`/`]`/`(`/`)` tiling, `y`/`Y` copy, `Shift+U` undo-delete, `Escape` cancel). `Tab` now skips invisible marks and records the recent class/size; `A` accepts only the selected prediction. `Project` discovers `<name>.cfg` and the usual `<name>_best/_final.weights` names and the GUI prefills the overlay paths. `darknet-gui-project` 38 tests, `darknet-gui-engine` 4, GUI and CLI build offline.- v0.0.270 (`ecd4e54e`): added source-derived Tucker local attention with strict cfg aliases/defaults, rank and MSA parsing, zero-padded window forward and exact backward, seeded initialization, SGD/Adam, `.weights` save/load, and GPU f32 host fallback. `MEM=3G ./tools/test_host.sh tucker_attention` passed 14 tests, `weights` 32, and `runtime` 168; coverage includes finite differences for all factor blocks/input/Tanh/MSA, a hand-computed 2×2 window, multi-head math, legacy-header round trip, and FakeBackend CPU forward/backward/update parity. MiniMax sparse attention is a shared option rather than a standalone layer in the audited C++ branch; it is wired for ViT and Tucker. No native Tucker kernel or compiled-C++ golden/parity.

- v0.1.022 (`7146ead2`): `Ctrl+C`/`Ctrl+V` and Ctrl/Shift+Insert now use the system clipboard via `arboard`, and the top bar's label control cycles DarkMark's on/auto/off mode instead of writing an unused setting.
- v0.1.023 (`da9d03ad`): sidecars are written with `G3`'s tab indentation, so a byte-level diff against DarkMark no longer fails on nested lines; `darknet-gui-project` 39 tests.- v0.0.269 (`4addcb07`): integrated strict Swin/transformer aliases, Rust-defined weights and modern save/load, seeded initialization, exact CPU block forward/backward, runtime accumulation and SGD/Adam, optimizer-state serialization, and GPU f32 host fallback. `MEM=3G ./tools/test_host.sh swin` passed 25 tests, `runtime` 166, and `weights` 29; FakeBackend/CPU output, gradient, and update parity includes a preceding-convolution chain. No native Swin kernel, C++ weights writer, or compiled-C++ golden. v0.0.268 (`0bd26bda`) added the source-derived reference helpers before this integration.
- v0.0.254 (`fc50aa57`): added the dependency-free ONNX protobuf wire encoder; varint, negative signed-int, and model/tensor wire tests pass.
- v0.0.255 (`1b632932`): implemented the ONNX exporter for supported convolution/BN, activations, maxpool, route, shortcut, upsample, and YOLO paths; unsupported layers fail explicitly. `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests.
- v0.0.256 (`4a0d99e7`): added `export-onnx` CLI parsing and command; `MEM=3G ./tools/test_host.sh export_onnx_args`, scoped host build, and fused/unfused local-model export passed.
- v0.0.257 (`778242ab`): added `tools/onnx_check.py`; Python syntax check and ONNX Runtime comparisons passed with worst relative errors `3.257e-06` fused, `3.908e-05` unfused, and `3.831e-06` Mish, all under `1e-4`. Rust-runtime parity only; no compiled-C++ golden.
- v0.0.258 (`7681f478`): `obb_detector_map` dispatches all-CARPN sample sets to the existing polygon evaluator, rejects mixed front-point/CARPN geometry, and preserves legacy behavior for empty prediction sets. `MEM=3G ./tools/test_host.sh obb_map` passed 5 core tests, `MEM=3G ./tools/test_host.sh carpn_map` passed 3, and `MEM=3G ./tools/test_host.sh carpn_polygon_iou` passed 2. Caller-supplied sample API only; network decode/CLI validation remains unsupported and is planned in `carpn_network_map_plan.md`; CARPN training/`train -map` remains deferred. No compiled-C++ parity is claimed.
- v0.0.259 (`29889c5c`): added CPU `[obb_refine]` forward, CARPN quadrilateral decoding/NMS, `detect-image`, full-forward validation, and direct cfg/weights-backed `detector map`. Focused config, weights, forward, decoder, proposal, map, CLI, and explicit training-rejection tests passed; `obb_` passed 138 core + 4 CLI. Scoped CLI build, targeted formatting check, and `git diff --check` passed. Workspace-wide `cargo fmt --check` remains blocked by pre-existing unrelated ONNX/CUDA formatting differences. Hand-derived from `raptor`; no compiled-C++ or canonical dataset parity.
- v0.0.260 (`16e95d9a`): standard flash attention now has recompute-based native Oxide dQ/dK/dV and runtime dispatch inside ViT's whole-network CPU backward fallback. `MEM=3G ./tools/test_host.sh vit` passed 28 tests; `MEM=3G ./tools/test_oxide.sh --test oxide_layernorm` passed 14; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain` passed 22, including CPU-matched parameter updates; `MEM=3G ./tools/test_host.sh runtime` passed 161 runtime tests. This is Rust backend evidence, not compiled-C++ parity; VC attention remains open.
- v0.0.261 (`7ca5a58e`): added host E4M3 encode/decode and ExpCast 0/1/2 primitives as VC groundwork. `MEM=3G ./tools/test_host.sh vc_attention` passed 4 source-derived tests. Not connected to the ViT cfg/runtime path; no compiled-C++ parity.
- v0.0.262 (`18ad3547`): added V-Smooth forward and exact piecewise backward with finite-difference, permutation, non-differentiability, and quantization-boundary tests. `MEM=3G ./tools/test_host.sh vc_attention` passed 7 tests. The full VC attention path remains unintegrated; no compiled-C++ parity.
- v0.0.263 (`8792f7b9`): added a source-shaped online quantized-row forward/backward with O(T+C*T) cache and exact inter-tile alpha derivative. `MEM=3G ./tools/test_host.sh vc_attention` passed 10 tests, including two-tile finite differences. Full head/cache and Q/K scale chaining remain open; no compiled-C++ parity.
- v0.0.264 (`9a085535`): added Q/K quantization and score-scale backward chaining through `sq` and centered-key `sk`. `MEM=3G ./tools/test_host.sh vc_attention` passed 12 tests, including permuted-key finite differences and score-clamp masking. Full head/cache remains open; no compiled-C++ parity.
- v0.0.265 (`fabcfba5`): composed a single batch-head VC forward/cache/backward with O(T*C) state and recomputed online rows; LSE is explicitly for centered/clamped scores. `MEM=3G ./tools/test_host.sh vc_attention` passed 13 tests, including Q/K/V finite differences across two KV tiles with key permutation. Batch/head dispatch, cfg/runtime and backend fallback remain open; no compiled-C++ parity.
- v0.0.266 (`fc47742d`): added B×H QKV packing/unpacking around VC head references with one O(T*C) cache per head. `MEM=3G ./tools/test_host.sh vc_attention` passed 14 tests, including B=2/H=2 forward and fused-gradient mapping. Config/runtime and backend fallback remain open; no compiled-C++ parity.
- v0.0.267 (`3c6f5c76`): wired `[vit]` VC config aliases/validation, ExpCast modes, and source 8×8 spatial token ordering into CPU and GPU host-round-trip dispatch. GPU paths deliberately bypass standard flash device kernels and use the Rust VC reference for forward; backward uses its per-head cache and exact piecewise derivative, with explicit errors at undefined points. `MEM=3G ./tools/test_host.sh vc_attention` passed 16 filtered tests; `MEM=3G ./tools/test_host.sh vit` passed 32, including config and CPU/FakeBackend runtime forward/backward. No native VC kernel or compiled-C++ parity is claimed.
- Wave 1 verification follow-up (commit `9931ccc3`): added a tiny OBB CPU CLI smoke test with `adam=1`, `subdivisions=2`, and `[net] exposure=1.5`; it asserts two iterations, final weights, and the Adam state sidecar. The focused test passed; direct `cargo test -p darknet-core -p darknet-cli -j 2 -- obb_` passed (135 core + 4 CLI). The earlier direct host filter was uncapped. After user-provided Oxide output confirmed systemd scopes start, `MEM=3G ./tools/test_host.sh obb_` passed 135 core + 4 CLI tests in scope `run-r5e0590e5177c4b5f9508c6e86ca4f304.scope`. This focused pass does not cover the full host baseline/cfg corpus. No compiled-C++ parity is claimed.
- v0.0.251 (`968c1beb`; gate incomplete): added source-derived OBB weighted NMS. Reference is `raptor:src-lib/box.cpp` (`diounms_sort`, not `yolo_layer.cpp`): `overlap = obb_iou - (d²/c²)^beta_nms`, where `obb_iou = detection_obb_iou(i,j)`, `d²` is center-distance squared, and `c²` is the enclosing AABB diagonal squared; merge weights are `w_i = prob_i[k]` and `w_j = prob_j[k] * obb_iou`, used to average box x/y/w/h and the keeper front point. Source enum spelling is `OBB_W_NMS`; cfg aliases are `obb_w` and `obb_wnms`. The four `weighted_merge_*` tests and the cfg/multi-head-beta tests passed. Its OBB-filtered run had 131/132 core tests and 3/3 CLI tests pass; the loader failure is resolved by v0.0.252 below. No compiled-C++ parity is claimed.
- v0.0.252 (`fae49267`): omitted OBB `jitter` now defaults to zero while other detector heads retain their existing default; configured unsupported geometric augmentation remains rejected. The seeded flip/loader tests pass, including the prior `obb_batch_loader_applies_configured_seeded_flips_to_images_and_labels` failure.
- v0.0.253 (`31219aba`): implemented the parsed `ciou_obb_mul` training route. Source `raptor:src-lib/box.cpp` defines `L = max(0, 1 - S)`, `S = clamp(CIoU_obb * cos_half_theta, 0, 1)`; `raptor:src-lib/yolo_layer_obb.cpp` and `dx_box_ciou_obb_multiplicative_loss` define six decoded-geometry central differences with `eps = 1e-3 * (abs(value) + 1)`, clamping perturbed width/height to at least `1e-6`. Rust chains these gradients through the box/front channels, scales/clips deltas, and routes front-anchor offsets through both passes. The cfg now builds both training configs instead of rejecting the loss. `ciou_obb_mul` tests and direct `cargo test -p darknet-core -p darknet-cli -j 2 -- obb_` passed (135 core + 3 CLI); uncapped because the user systemd bus was unavailable at that run; a later capped OBB filter passed 135 core + 4 CLI tests, while the full host baseline/cfg corpus remains unrerun under the cap. No compiled-C++ parity is claimed.
- Full host baseline attempt: direct uncapped `cargo test -p darknet-core -p darknet-cli -j 2` passed 562/562 core unit tests and 23/23 CLI tests; preceding integration targets passed. The `cfg_corpus` integration test failed after 3943.36 s on four entries. Triage: direct child execution of `cfg/cspx-p7-mish.cfg` exits by SIGKILL, and kernel logs show global OOM kills at ~6.98–7.11 GiB RSS for `cfg_corpus` processes; the same explanation fits the paired `cspx-p7-mish_hp.cfg` child failure. `cfg/lstm.train.cfg` and `cfg/rnn.train.cfg` hit the 300 s CPU inference timeout; both request batch 128, 576 time steps, and three 1024-unit recurrent layers. These are host resource/runtime limits, not a reproduced parser/model error. At the time of this baseline run, the `systemd-run --user` memory cap was unavailable. Later user-provided `./tools/test_oxide.sh` output shows `Running scope as unit: ...scope`, confirming the user scope now starts for that run. The focused OBB filter later passed under the cap (135 core + 4 CLI), but the full host baseline/cfg corpus has not been rerun under the cap. Full baseline remains non-green; do not rerun the >1-hour corpus blindly. **Disposition (2026-10-16): accept these four cfg-corpus failures as known environment/resource-limited residuals for scoped follow-on work, not as passes or fixed behavior.** Preserve the failure list and unit-test baseline; do not claim full-host green. Revisit only if the resource environment changes or a relevant change touches these cfg paths. `cspx-p7-mish_hp.cfg` shares the observed SIGKILL pattern but its OOM cause is not independently proven. W2.1 plan dispatchability was audited before Wave 2; the GUI gates remain deferred pending user confirmation and external prerequisites.
- v164: OBB NMS now supports source `greedynms` (fixed 0.6 DIoU exponent) and `diounms` (configurable `beta_nms`, default 0.6), using the source AABB-based formulas. Multi-head DIoU configs with mismatched betas and non-finite `beta_nms` are rejected. The NMS scoring/cfg/multi-head tests passed; `MEM=3G ./tools/test_host.sh obb_` passed 122 core + 2 CLI tests. Weighted OBB merge remains unsupported; no compiled-C++ parity is claimed.
- v163: OBB NMS now accepts source `default` and `cornersnms` kinds and routes both through the source AABB-IoU suppression path; existing `obb` remains orientation-aware, and source `obb_nms`/`obbnms` aliases parse. The NMS-kind cfg and overlap tests passed; `MEM=3G ./tools/test_host.sh obb_` passed 120 core + 2 CLI tests. `greedynms`, `diounms`, and weighted OBB NMS remain unsupported; no compiled-C++ parity is claimed.
- v162: `detect-image` now routes `[net] obb=1` inference through the orientation-aware OBB decoder and reports the legacy front point in human output and full-precision `det_obb` TSV rows; unsupported mixed OBB/modern heads fail explicitly. The dispatch/formatting CLI test passed, `detect_` passed 2 CLI tests, and `obb_` passed 119 core + 2 CLI tests. CARPN network decoding remains unsupported; no compiled-C++ parity is claimed.
- v161: CPU OBB training now routes `-map` through the orientation-aware `ObbValidationDataset` and shared training `MapSchedule`; interval and final evaluations write `_map.csv`, update loss-chart mAP, print reports, and save `_best.weights` on improvement. The CLI test checks `-map_every 2` at iterations 2 and final 3; `detector_train_` passed 4/4 CLI tests, `map_` passed 4 CLI + 11 core tests, and `obb_` passed 119 core + 1 CLI test. CARPN validation remains unsupported; no compiled-C++ parity is claimed.
- v160: the CPU OBB trainer now writes a separate `<cfg>_obb_metrics.csv` with subdivision-aggregated average OBB IoU/GIoU, class output, assignment counts, classification/box cost components, front loss, and orientation accuracy/error buckets. The end-to-end CLI test validates the CSV schema and two-subdivision counts; `detector_train_` passed 4/4 CLI tests and `obb_` passed 119 core plus 1 CLI test. This is source-mapped telemetry, not compiled-C++ parity.
- v159: `[net] obb=1` now routes CPU `detector train` through a bounded OBB SGD loop using source OBB losses, configured subdivisions, direct resize, seeded horizontal flips, checkpoints, and loss CSV logging. It explicitly rejects `-map`, Adam, dynamic loss scaling, and unsupported augmentation; OBB GPU training is rejected. The CLI regression test covers one iteration and `-map` rejection; all 119 core OBB-filtered tests plus the CLI test passed. `cargo check -p darknet-cli --features oxide -j 4` could not complete because the existing f16 `ForwardResult` initializer at `main.rs:309` omits current cache fields; no oxide-compiled or compiled-C++ parity is claimed. OBB `-map` cadence and broader optimizer/augmentation integration remain open.
- v158: OBB delta batches now expose source-style loss cost for MSE and supported non-MSE modes. Non-MSE cost uses source best/additional anchor `1 - box_ious` loss averages (truth-threshold-only cells do not enter the count) plus objectness-normalized non-box squared deltas; per-head and subdivision score-loss sums are aggregated. The focused cost tests and all 119 OBB-filtered host tests passed. `ciou_obb_mul` training and CLI/schedule integration remain open; no compiled-C++ parity is established.
- v157: OBB truth-threshold dispatch now permits distinct matching and box-delta score configs, required for weighted MSE matching with the source CIoU-OBB regression fallback. The MSE weighted end-to-end delta test and all 119 OBB-filtered host tests passed.
- v156: OBB delta batches now report the source objectness-normalized sum of squared deltas outside the six box channels as the classification/objectness loss component, including non-MSE modes. Coverage checks default and non-unit `obj_normalizer`; all 118 OBB-filtered host tests passed. The remaining non-MSE IoU scalar and CLI/schedule integration are open.
- v155: OBB delta batches now expose the source raw `sum(delta²)` cost when every OBB YOLO head uses MSE; other loss modes return `None` rather than a misleading partial scalar. The MSE cost test and all 118 OBB-filtered host tests passed. Full non-MSE loss/CLI integration remains open.
- v154: covariance and COBB matching scores now combine using their configured normalizer weights when both are active, matching the source independent-branch matcher; CIoU-OBB override and single-branch paths are unchanged. The weighted-score test and all 117 OBB-filtered host tests passed. No compiled-C++ parity is established.
- v153: when covariance and COBB are both active, box regression now adds their independent contributions instead of returning after covariance. A combined-vs-separate gradient test and all 117 OBB-filtered host tests passed. This does not establish compiled-C++ or full-loss parity.
- v152: separate OBB matching-score and box-delta dispatch for `iou_loss=mse`: weighted covariance/COBB scores still select matches, while box regression follows source CIoU-OBB fallback instead of applying the covariance analytic gradient. Added routing and regression-path tests; all 116 OBB-filtered host tests passed. This does not establish compiled-C++ or full-loss parity.
- v151: front-aware OBB training stats now aggregate source angle-error mean/buckets (`<5°`, `<20°`, `>30°`) across assignments, heads, and subdivisions, using configured anchor offsets. The focused angle/merge tests and all 114 OBB-filtered host tests passed.
- v150: `ciou_obb_mul` is distinguished from additive CIoU-OBB during parsing; OBB training config now rejects that unsupported variant instead of silently using the additive path. Focused coverage and all 113 OBB-filtered host tests passed.
- v149: OBB `iou_thresh_kind` now parses to the spec with the source `iou` default; a configured `diou` value is covered by the adapter test. This is config retention; training-threshold semantics remain part of broader integration.
- v148: standard OBB `iou_loss` values are now retained in the parsed YOLO spec and routed into front-point delta configuration; `ciou_obb` and `ciou_obb_mul` map to CIoU while retaining their OBB score-mode flag. The focused adapter tests and all 113 OBB-filtered host tests passed.
- v147: the parsed OBB `front_loss_normalizer` now feeds covariance matching-score length weighting instead of a hard-coded 0.35. The focused adapter test and all 113 OBB-filtered host tests passed.
- v146: `[obb_yolo] iou_loss` now parses CIoU-OBB/CIoU-OBB-mul and standard IoU names; per-head config selects the CIoU-OBB match branch when requested or when covariance/COBB weighting is absent. Unsupported names fail. The focused tests and all 113 OBB-filtered host tests passed; this is score-path coverage, not full loss parity.
- v145: parsed covariance/COBB normalizers, covariance length ratio, COBB weights, and clamped QBBV vIoU weight now populate both matching and box-delta score configs. The focused parser/adapter test and all 112 OBB-filtered host tests passed.
- v144: `[obb_yolo] map=` now resolves relative class-map files and routes parsed IDs into the existing assignment config, matching the OBB source loader path. The focused parser/adapter test and all 111 OBB-filtered host tests passed.
- v143: `[obb_yolo]` now parses `counters_per_class` into bounded class multipliers and routes them through the existing per-head assignment config; malformed class counts are rejected. The focused parser/adapter test and all 110 OBB-filtered host tests passed.
- v142: added `train_obb_yolo_subdivisions`, which accumulates OBB gradients across all configured subdivisions, merges per-head/source metrics, and applies one SGD update normalized by total images. Tests cover two-subdivision accumulation and update. The focused test and all 109 OBB-filtered host tests passed; no C++ parity is implied.
- v141: added `train_obb_yolo_batch_step`, a one-subdivision CPU SGD path from an `ObbTrainingBatch` through forward, OBB delta generation, accumulated backprop, and parameter update. It rejects Adam and multi-subdivision configurations rather than silently using incorrect semantics. The focused parameter-update test and all 108 OBB-filtered host tests passed; full loss scalar and CLI/schedule integration remain open.
- v140: added `backward_obb_yolo_training`, connecting OBB output deltas to `ExecutableNetwork::backward_seeded`; the test verifies finite input gradients and nonzero convolution weight gradients. It still stops before optimizer updates and a complete OBB loss scalar. The focused backward test and all 107 OBB-filtered host tests passed; no compiled-C++ parity is established.
- v139: added `compute_obb_yolo_training_deltas`, which consumes OBB `ForwardResult` heads and labels, builds per-head configs, and returns per-layer output-gradient seeds plus aggregated stats. It explicitly requires shared focal/class options across heads and does not compute the full loss scalar or run backward/update. The focused integration test and all 106 OBB-filtered host tests passed; no compiled-C++ parity is established.
- v138: multi-head front-aware dispatch now accepts a separate front-delta config per head; the shared-config API remains as a compatibility wrapper. The multi-head test compares per-head outputs/stats and rejects a config-count mismatch. The focused test and all 105 OBB-filtered host tests passed. Network-loop/backprop integration remains open; no compiled-C++ parity is established.
- v137: added a parsed-spec adapter that binds an `[obb_yolo]` head to runtime grid/network dimensions and produces matching, assignment, box-delta, and front-delta configs, including parsed directional offsets and front-loss normalization. The focused adapter/parser test and all 105 OBB-filtered host tests passed. It does not run a network trainer or backprop; no compiled-C++ parity is established.
- v136: parsed `[obb_yolo]` `front_loss_normalizer` and its `front_point_normalizer` alias with source default `0.35` and source precedence. The focused parser tests and all 105 OBB-filtered host tests passed. Routing this setting into a network trainer remains open; no compiled-C++ parity is established.
- v135: `[obb_yolo]` now parses source `obb_anchors` in width/height-pair and width/height/offset-quadruple forms, stores finite directional offsets in `ObbYoloSpec`, rejects ambiguous/invalid forms, and retains the Rust `anchors` alias. The focused parser tests and all 105 OBB-filtered host tests passed. Offset routing into training remains open; the inference decoder remains cell-relative. No compiled-C++ parity is established.
- v134: front-point writeback now reports source-normalized loss sums/counts for explicit-front matches, composed across truth-threshold and best/additional passes and aggregated across heads. The 10 front-point-filtered tests and all 104 OBB-filtered host tests passed. This is only one loss term; full OBB loss/metric aggregation and network-loop/backprop integration remain open. No compiled-C++ parity is established.
- v133: added the multi-head front-aware training dispatcher by applying the ordered one-head composition to each head and aggregating its stats. Tests compare multi-head outputs/stats with per-head calls; all 104 OBB-filtered host tests passed. Network-loop/backprop integration remains open; no compiled-C++ parity is established.
- v132: added an opt-in one-head training composition that inserts truth-threshold and best/additional front-point deltas after their corresponding box passes; the test checks the exact order even with `accumulate=false`. The focused composition test and all 104 OBB-filtered host tests passed. Multi-head front dispatch and network-loop/backprop integration remain open; no compiled-C++ parity is established.
- v131: added truth-threshold front-point delta writeback for matched batch/head cells with class-scaled normalization and source center-chain behavior. The focused writeback test and all 103 OBB-filtered host tests passed. Composing the front passes into the full head/network training loop remains open; no compiled-C++ parity is established.
- v130: added positive-assignment front-point delta writeback for best/additional anchors, including class-scaled normalization, preserved unrelated channels, and optional center-chain updates when offsets are configured. The focused writeback test and all 102 OBB-filtered host tests passed. Truth-threshold front writeback and composed network-loop integration remain open; no compiled-C++ parity is established.
- v129: added standalone source front-point CIoU channel deltas with grid chain factors, normalizer/clipping, absent-front gating, and optional center-chain contributions when directional priors are present. The focused test and all 101 OBB-filtered host tests passed. Batch/head writeback and network-loop integration remain open; no compiled-C++ parity is established.
- v128: one-/multi-head OBB stats now expose source `net.total_bbox` counts by summing truth-threshold cells and best/additional anchor box assignments. The focused composition tests and all 100 OBB-filtered host tests passed. Raw front-channel gradients and full network-loop/loss integration remain open.
- v127: multi-head OBB delta dispatch now returns both per-head stats and source-style aggregate `avg_cat` sums/counts, threshold-positive counts, and box-positive counts. The focused multi-head test and all 100 OBB-filtered host tests passed. Raw front-channel gradients and full network-loop/loss integration remain open.
- v126: routed optional per-anchor front offsets through full-head score matching and truth/positive box deltas, including the covariance center-chain contribution; the focused training test and all 100 OBB-filtered host tests passed. cfg ingestion/inference decode and raw front-channel gradients remain open.
- v125: added standalone `decode_obb_yolo_front_point` for legacy cell-relative or source directional-anchor-prior decoding, with finite-offset validation. The focused test and all 99 OBB-filtered host tests passed. Config loading and routing these offsets through matcher/trainer/network decode remain open.
- v124: added multi-head delta dispatch across separate head tensors with shared global anchors/classes/network dimensions and per-head stats; the focused dispatch test and all 98 OBB-filtered host tests passed. Network-loop/backprop integration remains open.
- v123: composed the one-head delta passes in source order with `apply_obb_yolo_batch_head_training_deltas`, returning the anchor `avg_cat` stats and separate threshold/anchor counts. The focused composition test and all 97 OBB-filtered host tests passed. Multi-head trainer dispatch, front-point deltas, and broader losses/metrics remain open.
- v122: added `apply_obb_yolo_batch_head_truth_threshold_deltas` for per-cell truth-threshold positives, applying mapped class deltas (including the source smoothed `best_score²` override) and class-scaled box deltas while leaving the previously emitted objectness entry intact. The focused test and all 97 OBB-filtered host tests passed. Trainer sequencing, front-point deltas, and broader losses/metrics remain open.
- v121: positive-anchor box-delta writeback now scales `iou_normalizer` by the mapped class multiplier, matching the source `obb_yolo_delta_training` call for COBB and covariance branches. The focused writeback test and all 96 OBB-filtered host tests passed. Multi-head/trainer integration remains open.
- v120: `apply_obb_yolo_batch_head_positive_class_deltas` now returns source `avg_cat` sum/count contributions for its best/additional anchor assignments, while still writing class deltas. The focused stats/writeback test and all 96 OBB-filtered host tests passed. Truth-threshold metric aggregation and multi-head/trainer integration remain open.
- v119: integrated the source covariance analytic-gradient branch into `obb_yolo_box_regression_deltas` and positive batch/head box writeback, including covariance/IoU normalizer scaling and YOLO width/coordinate chain factors. Focused tests and all 96 OBB-filtered host tests passed. `center_front_anchor` and trainer/loss-loop integration remain open.
- v118: ported `box_covariance_w2_obb_loss_and_gradient` with source forward loss terms and analytic center/size/front gradients. Analytic gradients match finite-difference checks across all six inputs, and returned terms match the existing decoded-geometry loss. Focused test and all 95 OBB-filtered host tests passed. Batch delta integration remains open.
- v117: added `obb_yolo_box_regression_deltas` and per-batch/head positive box-delta writeback for the source finite-difference COBB/legacy score path, including normalizer scaling, sigmoid chain factors, clipping, and accumulation. The focused tests and all 94 OBB-filtered host tests passed. Covariance analytic gradients, multi-head trainer integration, and full loss/loop wiring remain open.
- v116: added `apply_obb_yolo_batch_head_positive_class_deltas`, applying source class-channel deltas at best/additional positive anchors with class mapping, multipliers, label smoothing, and existing-target behavior while preserving other entries. The focused writeback test and all 91 OBB-filtered host tests passed. Multi-head integration, class metrics, box deltas/losses, and trainer wiring remain open.
- v115: added `ObbYoloPositiveObjectnessConfig` and `apply_obb_yolo_batch_head_positive_objectness_deltas` for source best/additional positive-anchor overrides, including class mapping/multipliers and smooth-mode preservation of nonzero deltas. Tests verify mask slots, assignments, smooth-mode behavior, and bad-map rejection; all 90 OBB-filtered host tests passed.
- v114: added `apply_obb_yolo_batch_head_objectness_deltas`, writing source-derived negative/ignore/positive objectness deltas into entry 6 of the per-head delta tensor while preserving all other entries. Tests cover positive/negative batch slots, exact write locations, and delta-shape rejection; all 88 OBB-filtered host tests passed. Class/box delta writeback and full trainer integration remain open.
- v113: integrated source objectness-delta generation into batch-cell matching and full head traversal, using the output objectness channel, `obj_normalizer`, smoothing, and positive/ignore gate precedence. Batch/head-focused tests and all 86 OBB-filtered host tests passed. Class/box deltas and full trainer integration remain open.
- v112: added `ObbYoloHeadMatchConfig` and `visit_obb_yolo_batch_head_matches` to traverse a complete batch-major head tensor in source batch/cell/anchor order with bounded callback memory, sanitize consumed prediction/class channels, and emit per-cell match gates. Tests cover score/channel extraction, batch tags, traversal ordering, and malformed tensor length; all 86 OBB-filtered host tests passed. Multi-head trainer dispatch, deltas, losses, and optimizer integration remain open.
- v111: added `match_obb_yolo_batch_cell_predictions` and tagged batch/head/anchor/cell match records, mapping caller-selected cells to the correct image truths and preserving input order with per-cell score config. Focused tests cover distinct batch truths, order, thresholds, and invalid batch indices; all 83 OBB-filtered host tests passed. Full head-tensor traversal, loss deltas, and trainer integration remain open.
- v110: added `ObbYoloHeadAssignmentSpec`, batch/head-tagged positive records, and `enumerate_obb_yolo_batch_positives` in source head-then-batch order. Tests verify primary/additional anchors, local truth indices, per-head masks, order, and empty-input rejection; all 81 OBB-filtered host tests passed. Prediction-conditioned batch matching and full loss/trainer integration remain open.
- v109: added `match_obb_yolo_cell_prediction_with_score_config`, composing per-truth covariance/COBB/angular branch scores with source best-truth, class-gated ignore, positive thresholds, truth order, and empty-row sentinel for one cell. All 79 OBB-filtered host tests passed. Batch/multi-head assignment and training integration remain open.
- v108: added `ObbYoloMatchScoreConfig`/`ObbYoloMatchScoreBranch` and `obb_yolo_match_score`, dispatching one cell through covariance-first, then COBB, then legacy-angular source branch priority. All 78 OBB-filtered host tests passed; batch/multi-truth assignment and training integration remain open.
- v107: added `obb_yolo_cobb_prediction_match_score_with_viou`, decoding one cell's six post-YOLO channels, enforcing covariance-priority/COBB branch gates, and applying the source COBB plus QBBV score. The raw-match test and all 77 OBB-filtered host tests passed. Batch assignment/loss integration and compiled-C++ parity remain open.
- v106: added source COBB decoded-geometry loss/quality helpers (`ObbCobbLossConfig`, `ObbCobbLossTerms`, `box_cobb_obb_loss_terms`, and QBBV-blended quality), including candidate type selection, analytic score rows, Smooth-L1 terms, and invalid-geometry fallback. Two focused COBB tests and all 76 OBB-filtered host tests passed. Raw-channel decode/training integration and compiled-C++ parity remain open.
- v105: added `box_ciou_angular_quality_score` and `obb_yolo_angular_prediction_match_score_with_viou` for the source legacy weighted-branch fallback, including raw-channel cell decoding and QBBV blending. Focused geometry/decode tests passed; all 74 OBB-filtered host tests passed. COBB scoring/integration and compiled-C++ parity remain open.
- v104: added `blend_obb_score_with_qbbv_viou` and `obb_yolo_covariance_prediction_match_score_with_viou`, applying the source weight clamp, default 24×24 sampling, and non-finite sanitization to the covariance match branch. QBBV blending and raw covariance integration tests passed; all 72 OBB-filtered host tests passed. Multi-branch/other-branch integration and compiled-C++ parity remain open.
- v103: added `obb_qbbv_heatmap_value` and `box_obb_qbbv_viou`, porting the source-oriented tent heatmap and center-sampled squared-heatmap intersection-over-union (sample count clamped to 4–128). Two focused QBBV tests and all 71 OBB-filtered host tests passed; score blending and compiled-C++ parity remain open.
- v102: added `obb_yolo_covariance_prediction_match_score`, decoding one cell's post-YOLO channels and scoring them against a truth with the covariance-W2 branch. The branch gate, direction-free front placeholder, and anchor/grid decode are covered. One focused test passed; weighted covariance/COBB aggregation, QBBV-vIoU blending, and compiled-C++ parity remain open.
- v101: added `box_covariance_w2_obb_loss_terms` and `box_covariance_w2_obb_quality_score` for already-decoded OBB geometry, porting the source covariance, center, and front-length terms with source epsilons/clamps. Two focused formula tests passed; raw matching decode, QBBV-vIoU blending, COBB scoring, and compiled-C++ parity remain open.
- v100: added `obb_yolo_class_deltas`, a source-derived class-channel delta primitive for positive OBB cells, including the existing-delta gate, focal/non-focal branches, label smoothing, optional class weights/normalization, and average-class output. Four focused class-delta tests and all 66 OBB-filtered host tests passed. Full training/loss integration and compiled-C++ parity remain open.
- v99: added `match_obb_yolo_cell_prediction` to compose per-truth CIoU scoring with the source best-truth, class-gated ignore, and positive-threshold gates for one cell. It preserves truth order, first-on-tie selection, and the `truth.x == 0` empty-slot sentinel. All 62 OBB-filtered host tests passed; weighted branches, batch/positive-anchor integration, losses, and compiled-C++ parity remain open.
- v98: added `obb_yolo_ciou_prediction_match_score`, decoding one cell's post-YOLO six-channel output for the source CIoU/default-score branch and passing its normalized geometry to v97's quality metric. Legacy/new-coordinates, nonzero-cell, absent-front, and invalid-input cases are covered. All 60 OBB-filtered host tests passed; weighted covariance/COBB scoring, batch-loop integration, and compiled-C++ parity remain open.
- v97: added `box_ciou_obb_quality_score`, matching the source `box_ciou_obb_loss_terms` score for already-decoded OBB geometries: rotated IoU, CIoU center/aspect penalties, and the cosine half-angle orientation factor. Two focused source-formula tests passed; raw-channel matching, covariance/COBB scoring, and training integration remain open. No compiled-C++ parity is claimed.
- v96: added `obb_yolo_cell_objectness_delta` to translate the v93 per-cell match selection into the source OBB YOLO negative, ignore, and positive objectness deltas. It preserves the smoothed ignore max, positive-target override, non-finite output/delta sanitization, and leaves raw score calculation plus class/box loss and trainer integration open. All 55 OBB-filtered host tests passed; no compiled-C++ parity is claimed.
- v95: audited the fp16 convolution-backward TODO against `mixed.rs` and the real-device `oxide_amp` suite. AMP convolution dgrad uses f16 GEMM with f32 accumulation before f16 storage; wgrad uses f16 inputs with f32 accumulation and returns f32 weight gradients. All 11 `oxide_amp` tests passed, including grouped and multi-chunk convolution. This closes the tracked fp16 dgrad/wgrad follow-up; half2 elementwise remains unnecessary.
- v94: added `carpn_detector_map` for normalized CARPN quadrilateral detections
  and front-point OBB labels. It reuses oriented polygon IoU and the shared AP
  report/matching policy, rejects non-CARPN predictions and missing-front labels,
  and reconstructs ground-truth vertices in the source BoxOBB order. The
  evaluation tests passed 12/12 (including all three CARPN tests); the OBB-focused
  suite passed 53/53. CARPN network
  decoding/validation integration and compiled-C++ parity remain open.
- v93: added `select_obb_yolo_cell_truth_match` for validated single-cell
  selection from precomputed source-style scores. It preserves the first best
  truth, gates ignore matching on any class output strictly above 0.25, and
  reports strict `ignore_thresh` and `truth_thresh` outcomes. Focused OBB tests
  passed 53/53. Raw prediction-match score math, delta/loss generation, trainer
  integration, and compiled-C++ parity remain open.
- v92: added `enumerate_obb_yolo_head_positives` with typed best-anchor and
  additional-anchor matches. It preserves truth and global-anchor order, emits
  extra masked anchors only under the source `iou_thresh < 1` and strict score
  comparison, and can return qualifying extras even when the global best is not
  in the head mask. Focused OBB tests passed 49/49. Prediction-conditioned
  ignore/truth-threshold matching, loss/deltas, trainer integration, and
  compiled-C++ parity remain open.
- v91: added `ObbTrainingBatch::encode_truths` and public seven-float OBB YOLO
  truth packing in the source `[batch][box-slot][7]` layout. Rows preserve label
  order, use the center placeholder for absent front points, and zero-pad unused
  slots; invalid shapes/classes/capacity fail explicitly. Focused OBB tests passed
  44/44. The package-wide host test attempt timed out after 20 minutes in the
  existing `laya_scorer::real_weights` test, so its final result is inconclusive.
  Complete anchor assignment, OBB losses/trainer integration, and compiled-C++
  parity remain open.
- v90: `DetectorAugmentation` now accepts `[obb_yolo]` cfg sections and exposes
  a deterministic per-sample horizontal-flip mask matching its existing Rust
  augmentation draw order. Added `load_obb_training_batch_with_flips_from_paths`,
  which applies the configured flips to resized NCHW images and labels together.
  Crop/resize jitter, HSV, mosaic, training/loss integration, and C++ RNG parity
  remain unsupported. The focused OBB suite passed 42/42 and all 465
  `darknet-core` library tests passed.
- v89: added public direction-free OBB YOLO anchor-match scoring and
  best-anchor/cell assignment primitives in `obb_training.rs`. Matching compares
  normalized anchor size with truth size; global-best selection, first-on-tie
  ordering, head-mask handling, and cell clamping follow the audited source.
  This is not a complete OBB truth assignment/loss/training path, and no
  compiled-C++ parity is claimed. The focused OBB suite passed 40/40 and all 463
  `darknet-core` library tests passed.
- v88: added `ObbTrainingBatch::apply_horizontal_flips`, which reflects selected
  NCHW samples and their OBB labels together using a validated per-image mask.
  The focused OBB suite passed 36/36 and all 459 `darknet-core` library tests
  passed. Flip selection remains caller-owned; config randomness and crop/mosaic
  augmentation are still open, with no compiled-C++ parity claim.
- v87: added public `flip_obb_labels`, an exact horizontal reflection of label
  centers, front points, and stored bounds that preserves OBB dimensions and
  absent-front state. The focused OBB suite passed 35/35 and all 458
  `darknet-core` library tests passed. This is label geometry only: loader
  integration and crop/mosaic transforms remain open; no compiled-C++ parity is
  claimed.
- v86: added typed `[obb_yolo] nms_kind` parsing with source default `obb`,
  case-insensitive explicit `obb`, and an explicit error for unwired alternate
  kinds. The OBB decoder dispatches the configured oriented kind to its
  orientation-aware NMS; focused OBB tests passed 33/33 and all 456 `darknet-core`
  library tests passed. No compiled-C++ parity is claimed.
- v85: added a standalone CPU `detector map` command with optional weights,
  confidence/IoU/NMS thresholds, and letterbox preprocessing. It routes standard
  graphs through `ValidationDataset` and OBB graphs through
  `ObbValidationDataset`; `darknet-cli` tests passed 19/19 and temporary standard
  and OBB cfg/data command smoke tests both completed. Training `-map` cadence,
  GPU evaluation, CARPN mAP, and compiled-C++ parity remain out of scope.
- v84: added `ObbValidationDataset`, including OBB-only head/class validation,
  OBB label preflight, one inference/decode per image, pixel-to-normalized
  conversion for boxes/front points, and orientation-aware AP evaluation. The
  existing axis-aligned evaluator still refuses OBB graphs; the CPU CLI route was
  added in v85, while training `-map` cadence and CARPN mAP remain open. The focused
  OBB host suite passed 33/33 and all 456 `darknet-core` library tests passed; no
  compiled-C++ parity is claimed.
- v83: added `obb_detector_map` for normalized legacy front-point predictions,
  using orientation-aware matching and existing AP/report semantics. Five-field
  labels are treated as axis-aligned truth; CARPN polygon detections are rejected
  until their truth-vertex encoder is wired. This is a standalone scorer, not
  network/CLI evaluation. The focused OBB host suite passed 31/31 and all 454
  `darknet-core` library tests passed; no compiled-C++ parity is claimed.
- v82: added a separate OBB training-batch type/loader using direct image resize
  and the existing five/seven-field parser. It preserves optional front-point
  state; class IDs are bounded by the smallest OBB head and optional data class
  count. Augmentation and OBB losses remain unsupported. The focused OBB host
  suite passed 29/29 and all 452 `darknet-core` unit tests passed; no compiled-C++
  parity is claimed.
- v81: wired `[net] obb=1` and `[obb_yolo]` cfg parsing into a typed
  six-coordinate head with filter/mask/anchor validation; CPU/runtime head
  collection feeds a network-level multi-head decoder and orientation-aware
  NMS. AABB YOLO training and mAP explicitly reject OBB networks rather than
  silently using axis-aligned semantics. The focused OBB host suite passed
  26/26, and all 449 `darknet-core` unit tests passed. Batch
  loading/augmentation, OBB truth assignment/loss, NMS-kind config wiring, and
  mAP remain open; no compiled-C++ golden/parity claim.
- v80: extended `ObbLabel` with source-clamped AABB bounds and added a
  source-derived normalized-center label rotation utility. Explicit front points
  rotate oriented rectangle corners while preserving OBB sizes; missing or
  degenerate front vectors use the source AABB fallback. The focused OBB suite
  passed 21/21, including both rotation paths and the source no-op threshold.
  Image augmentation/loader/training integration and compiled-C++ parity remain
  open.
- v79: added standalone OBB label-file parsing for five-field box rows and
  seven-field rows with explicit front points. The parser tracks `has_front`,
  defaults absent front coordinates to the center placeholder, clamps finite
  fields to `[0,1]`, and validates class bounds and positive dimensions. The
  focused OBB suite passed 17/17. This is not `TrainingBatch`, augmentation,
  truth assignment, OBB loss, or mAP integration; no compiled-C++ parity claim.
- v78: added a source-derived per-head legacy `[obb_yolo]` decoder for six box
  channels, objectness, class scores, and raw linear front offsets. It handles
  `new_coords`, `scale_x_y`, strict objectness/class thresholds, output
  sanitization, integer-rounded C++ letterbox dimensions, and image-pixel front
  points; OBB NMS is deliberately separate. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb` suite passed 12/12,
  covering logistic and `new_coords` decode, letterbox correction, integer
  resize rounding, malformed shapes, geometry, and NMS. At the v78 point this
  remained a standalone primitive; cfg/runtime integration was added in v81.
  No compiled-C++ golden or parity claim is available.
- v77: added an OBB detection wrapper/geometry enum, CARPN-vs-front-point IoU
  dispatch, and classwise greedy OBB NMS with descending per-class confidence,
  exact-zero skip behavior, and strict `IoU > threshold`. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb` suite passed 8/8, including
  orientation-sensitive suppression, mixed CARPN/front-point dispatch, class
  isolation, and zero-objectness/zero-score cases. This is not decoder, cfg, or
  dataset/mAP integration; no compiled-C++ golden or parity claim is available.
- v76: added host-only CARPN convex-quad IoU for four interleaved vertex pairs,
  reusing the pure-Rust convex clipper and rejecting degenerate/non-finite inputs.
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh carpn_polygon_iou` passed 2/2
  for hand-derived overlap, containment, disjoint, diamond, reversed-winding, and
  invalid cases. This is geometry only; CARPN detection dispatch, OBB NMS/mAP,
  cfg/layer integration, and losses remain open. No compiled-C++ golden/parity
  claim is available.
- v75: added host-only vertices and convex IoU for legacy front-point OBB rectangles,
  with source-ordered corners and explicit invalid-geometry behavior. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb_geometry` tests pass 3/3,
  covering rotated/partial overlaps, containment, disjoint boxes, and malformed
  inputs. This is only a geometry prerequisite: OBB cfg/layer/data loading,
  CARPN explicit polygons, losses, NMS, and mAP remain open. No compiled-C++
  golden or parity claim is available.
- v74: EML Adam now applies both convolution branches and BN scales on-device,
  retains per-branch moments, and supports deferred host synchronization. Adam
  moments are persisted in additive DNOPT5 sidecars while DNOPT1–4 remain
  readable; DNOPT3 EML host moments migrate to device state, and host fallback
  remains active when another layer requires CPU backward. Focused host EML tests
  pass 12/12, optimizer-state tests pass 5/5, and `oxide_runtime_chain` passes
  20/20, including no-BN Adam update parity, BN scale updates, deferred sync,
  and optimizer-state output. No compiled-C++ golden is available; CPU EML BN still
  uses rolling statistics while native GPU training uses batch statistics, so
  CPU/C++ training parity is not claimed.
- v73: EML SGD now accumulates per-branch weight/beta/BN-scale gradients and
  momentum buffers on-device, applies both branch updates through the backend,
  and supports deferred host synchronization. Device finiteness checks cover
  EML branch and BN-scale buffers. Focused host EML tests pass 9/9; the full
  Oxide `oxide_runtime_chain` suite passes 19/19, including two-step momentum
  parity, BN/no-BN updates, and deferred sync. At that point Adam remained
  host-side for EML. No CPU/C++ BN-training parity claim.
- v72: Batch-normalized EML now stays on native Oxide forward/backward: each
  branch retains batch-stat BN caches, computes exact scale/beta gradients and
  convolution adjoints, and applies device-produced rolling EMA stats to host
  params. Gradients still accumulate into host train buffers and optimizers
  remain host-side. Focused host EML tests pass 3/3; the real Oxide
  `oxide_runtime_chain` suite passes 19/19, including both branch EMA updates,
  finite BN gradients, and host optimizer/cache refresh. CPU EML training still
  uses rolling statistics while native GPU training uses batch statistics, so
  CPU/C++ training parity is not claimed. Device optimizers and a compiled-C++
  golden remain open.
- v71: No-BN EML GPU backward now uses backend EML adjoints and both branch
  convolution input/weight-gradient ops, adds the eligible residual gradient,
  and accumulates parameter gradients into host train buffers for SGD/Adam.
  BN EML retains the explicit CPU backward fallback; host EML gradients now
  participate in dynamic-loss-scale finiteness checks. `oxide_runtime_chain`
  passes 19/19, including two accumulated no-BN backwards and BN fallback.
  Optimizer updates remain host-side; no C++ parity claim is made.
- v70: EML f32 GPU forward now runs both branch convolutions, folded rolling-stat
  BN parameters, bias addition, and the combine op through the backend; both
  branch parameter blocks use separate per-owner device-cache slots. CPU branch
  caches remain available for the explicit CPU backward/optimizer fallback.
  `./tools/test_oxide.sh --test oxide_runtime_chain` passed 19/19, including BN
  and no-BN EML outputs plus device-cache invalidation after host updates.
  Native backward/optimizer and C++ batch-stat/
  EMA training behavior remain open; this is not a C++ parity claim.
- v69: Integrated `[eml_convolutional]` cfg/shape parsing, two branch-convolution
  parameter and training-buffer blocks, C++ `input_layer` then `self_layer`
  `.weights` order, CPU forward/backward, SGD/Adam updates, and an explicit GPU
  host-fallback path. Focused EML host tests pass 3/3, including finite-difference
  parameter gradients and FakeBackend CPU/GPU-fallback SGD parity; the EML Adam
  fallback test passes. The EML weights round-trip is covered by
  `recurrent_weight_blocks_round_trip_in_darknet_order`. Remaining: native Oxide
  whole-layer execution and batch-stat/rolling-EMA BN training behavior; the
  existing host convolution helper uses rolling-stat BN. The standalone full
  `cfg_corpus` run exceeded a 40-minute timeout without a final result. No
  compiled-C++ golden is available, so no C++ parity claim is made.
- v68: Added host-reference and native Oxide f32 EML branch-combine forward
  and backward ops with optional residual, shape checks, exact branch
  derivatives, and the source-derived nonpositive clamp/epsilon defaults.
  `MEM=3G PKGS=darknet-cuda ./tools/test_host.sh` passed 12/12; real-device
  `oxide_layers_extra` passed 24/24. Config/layer/runtime/optimizer wiring and
  compiled-C++ golden parity remain open.
- v67: Added `eml_conv_math.rs`, a source-derived pure reference for the
  `[eml_convolutional]` two-branch combine, stable softplus/sigmoid, optional
  residual, and exact default branch gradients. Hand-reference, shape/default,
  strict-boundary, and finite-difference tests passed (4/4). This is only the
  math prerequisite; cfg/layer/runtime/backend/optimizer support and compiled-C++
  parity remain open; no compiled-C++ golden was available.
- v66: Enabled `normalize_channels`, `normalize_channels_softmax`, and
  `normalize_channels_softmax_maxval` on convolution layers in CPU and f32 GPU
  inference/training forward/backward. Exact gradients remain default;
  `DARKNET_GRAD_MODE=cpp` selects the source-derived CUDA diagonal shortcut.
  Other layer types and f16 inference reject the activation explicitly. The
  FakeBackend CPU/GPU training parity test, Oxide chained forward/backward/SGD
  (including compact inference), C++-mode Oxide runtime, and f16 rejection tests
  pass. This is source-derived runtime parity, not a compiled-C++ golden.
- v65: Added shape-aware host-reference and native Oxide primitives for
  `normalize_channels`, `normalize_channels_softmax`, and
  `normalize_channels_softmax_maxval`, including exact normalization-chain
  backward and the legacy C++ CUDA diagonal shortcut when requested. Source
  inspection found C++ CPU and CUDA gradients differ; `cpp_grad` intentionally
  follows the CUDA kernels. Direct host finite differences/parity and Oxide
  forward/backward/C++ tests pass; full `oxide_layers_extra` passed (23/23).
  At the v65 boundary, cfg parsing/runtime integration remained open; v66 below
  closes the supported f32 convolution path without claiming compiled-C++ parity.
- v64: Weighted-shortcut backward now assembles per-source `delta * activation`
  products and applies exact None/ReLU/softmax normalization-chain derivatives
  through `F32TensorOpsBackend`; `GradMode::Cpp` still returns raw products.
  Oxide transfers only the final per-input/per-slot weight-gradient vector to
  host train buffers; parameter updates remain host-side. Host finite-difference
  math passed (4/4), FakeBackend runtime parity passed (1/1), the direct Oxide
  product/normalization/C++-mode test passed (1/1) and full `oxide_layers_extra`
  passed (22/22), and real-device runtime forward/backward/SGD parity passed
  (1/1), including short-source prefixes.
- v63: Audited convolution `grad_centr` against the C++ GPU update path and
  removed it from the open parity gaps. `gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate`
  passed (host 1/1), and `gradient_centralization_matches_reference_and_is_noop_off_warp_sizes`
  passed on real Oxide (1/1), covering BN gating, per-filter subtraction, and
  C++'s non-warp-size skip.
- v62: Replaced approximate mosaic pasting/box remapping with the source-level
  C++ quadrant crop alignment, flip-aware shifts, optional corner relocation,
  integer-pixel box clipping, and strict filtering. Rust RNG remains deterministic
  but not C++-identical; no compiled-C++ golden claim. Augmentation/loader host
  filter passed (12/12), including OpenCV single-image fixtures and hand-derived
  mosaic pixel/box geometry tests.
- v61: BN RNN/LSTM/CRNN scale-gradient reductions now stay on backend using
  existing centering, channel-scale, pointwise-multiply, and channel-reduction
  ops; Oxide no longer transfers raw projection/convolution outputs and deltas to
  host just for dscale. Forward/cache, rolling-stat update, and optimizer behavior
  remain unchanged. Host recurrent filters passed (RNN 7/7, LSTM 3/3, CRNN 3/3),
  all three real Oxide BN recurrent parity tests passed (3/3), and full real
  Oxide `oxide_runtime_chain` passed (17/17).
- v60: BN CRNN backward now uses backend convolution/BPTT ops with rolling-stat
  affine input/beta derivatives; forward/cache capture, scale-gradient reductions,
  and optimizer remain host-side. FakeBackend covers three projection blocks,
  scales, rolling stats, upstream convolution updates, and loss-scale overflow;
  host CRNN tests passed (3/3), host `rnn` filter (7/7), host LSTM tests (3/3),
  and full real Oxide `oxide_runtime_chain` passed (17/17).
- v59: BN LSTM backward now uses backend gate/BPTT and projection operations,
  with rolling-stat affine input/beta derivatives; CPU forward/cache capture and
  optimizer remain host-side, and scale-gradient reductions transfer raw
  projections/deltas to host. BN CRNN remains on CPU backward. Host LSTM tests
  passed (3/3), host RNN regression tests passed (6/6), and real Oxide
  `oxide_runtime_chain` passed (16/16), including BN LSTM parameter, rolling-stat,
  and upstream-convolution parity.
- v58: BN RNN backward uses backend projection/BPTT ops and rolling-stat affine
  input/beta derivatives; scale-gradient reduction still transfers raw projections
  and deltas to host. Forward/cache capture and optimizer remain host-side; BN
  LSTM/CRNN still use CPU backward. Dynamic loss-scale checks now inspect
  host-buffered connected/recurrent gradients. Host RNN tests passed (6/6),
  including FakeBackend parameter/stat parity; focused real Oxide parity passed
  (1/1); full `oxide_runtime_chain` passed
  (15/15).
- v57: `[maxpool] maxpool_zero_nonmax=1` now stays on the GPU backward path.
  The existing kernel writes gradients only for input argmaxes, so non-argmax
  deltas are already zero. Host maxpool tests passed (8/8); focused real Oxide
  parity passed (1/1), and the full `oxide_runtime_chain` suite passed (14/14).
- v56: BN MoE router/expert SGD, including BN scale updates and momentum, now
  runs on-device. Deferred sync preserves the expert cache and copies raw params
  plus BN stats to host. FakeBackend MoE tests passed (21/21); the full real-device
  `oxide_runtime_chain` suite passed (13/13), including BN backward/SGD parity.
- v55: BN MoE router/expert Adam now updates raw weights, beta, and BN scales
  on-device; scale moments are persisted in DNOPT4 while DNOPT1-3 remain
  readable, including migration of nested host BN moments from DNOPT3. Deferred
  sync now retains device-updated BN expert caches and copies raw params plus
  BN stats back to host. FakeBackend MoE tests passed (20/20), including host-to-
  device scale-moment migration and v4 resume; optimizer-state compatibility
  tests passed (4/4). The full real-device `oxide_runtime_chain` suite passed
  (13/13), including BN Adam parity and deferred synchronization.
- v54: BN MoE expert backward is no longer routed through whole-network CPU
  fallback. The backend computes BN affine/input/weight derivatives and
  accumulates beta/scale gradients with router/expert gradients on device;
  host-side SGD/Adam consume them after flush. No-BN router/expert optimizer
  paths remain device-native. FakeBackend MoE tests passed (19/19), including BN
  gradient/SGD parity, BN Adam host fallback, and BN-scale overflow detection;
  the full real-device `oxide_runtime_chain` suite passed (12/12). The focused
  `moe_batchnorm_expert_device_backward_matches_cpu_on_oxide` test passed (1/1),
  comparing upstream conv, router, expert, and BN scale updates against CPU.
  BN expert Adam remains host-side.
- v53: moved no-BN MoE router/expert Adam updates and moment storage onto the
  device; DNOPT3 persists those moments, DNOPT1/2 remain readable, and legacy
  DNOPT2 host moments migrate on the first device update. Fixed host nested-Adam
  child resizing so later MoE experts retain moments across steps. The focused
  host MoE suite passed (18/18), host optimizer compatibility passed, and the
  full real-device `oxide_runtime_chain` suite passed (11/11), including
  `moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients`.
- Added no-BN LSTM device backward; v59 adds its BN rolling-stat projection
  derivative path using an uploaded CPU forward cache. Gate derivatives, connected projection
  gradients, and recurrent/input-gradient accumulation use backend ops. The
  training path retains BPTT caches on-device rather than downloading projection,
  gate, and cell caches, and concatenates timestep outputs directly on-device.
  LSTM and RNN forward/backward sequence slicing and timestep input-gradient
  assembly stay on-device; RNN projection gradients also accumulate on-device
  across steps, and no-BN RNN/CRNN training pre-activation/state caches now
  remain device-resident for backward (the host cache is captured only for
  host-facing forwards). CRNN forward input slicing and output assembly also
  stay on-device. CRNN backward sequence slicing and input-gradient assembly
  also stay on-device; convolution parameter gradients accumulate on-device across
  steps. Recurrent cache/state and parameter-gradient transfers, host
  accumulation/SGD, and some cache handling remain host-side. Targeted host
  RNN/CRNN tests and real-device `oxide_runtime_chain` tests
  `rnn_recurrent_projections_and_training_backward_match_cpu` and
  `crnn_recurrent_convolutions_and_training_backward_match_cpu` pass. LSTM
  device-backward parity also passes, including upstream convolution updates.
- Added an oxide Stair activation kernel matching Darknet's floor-based
  alternating-step formula and parity gradient, including negative odd intervals.
  Verified by the 391-test core library suite, `oxide_parity` (11 passed,
  including finite differences away from integer discontinuities), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (195 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. Every
  oxide test target also compiled with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide PLSE activation kernel with the C++ piecewise-linear sigmoid
  and exact ±4 endpoint gradients. Verified by the 391-test core library suite,
  `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (189 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests.
- Added an oxide Lhtan activation kernel with its exact 0.001 tail slopes and
  Darknet's endpoint gradients at 0 and 1. Verified by the 391-test core library
  suite, `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (183 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests. Every oxide test target also compiled
  with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide Relie activation kernel (leaky ReLU with slope 0.01), including
  the exact zero-point derivative. Verified by `oxide_parity` (11 passed,
  including finite differences and input/output derivative parity), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (177 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. The
  391-test core library suite also passed.
- Added an oxide HardTanh kernel with Darknet's inclusive clamp and zero
  derivative at exactly -1 and 1. Verified by `oxide_parity` (11 passed,
  including finite differences and endpoint parity), `oxide_half` (13 passed, 1
  ignored), all 13 `oxide_cfg_shapes` tests (171 convolution-activation pairs),
  the real-device chained-convolution activation test, the 391-test core
  library suite, and a no-run build of every oxide test target.
- Added an oxide loggy activation kernel and exact derivative, wiring it through
  cfg parsing, f32/f16 convolution activation paths, and the real-device runtime.
  Verified by `oxide_parity` (11 passed, including finite differences and
  input-vs-output derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (165 convolution-activation pairs), all 7 real-device
  `oxide_runtime_chain` tests, and the 391-test core library suite. The oxide
  `--no-run` build compiled every test target.
- Added an oxide Hard Mish activation kernel and exact derivative, including
  Darknet's strict threshold behavior at -2 and 0. Verified by `oxide_parity`
  (11 passed, including finite differences and explicit x=-2), `oxide_half` (13
  passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (159 forward/backward
  convolution-activation pairs), and all 7 real-device `oxide_runtime_chain`
  tests. The core library suite passed (391 tests).
- Added an oxide SELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), all 13 `oxide_cfg_shapes` tests (including
  153 forward/backward convolution-activation pairs), and all 7 real-device
  `oxide_runtime_chain` tests. `cargo test -j 4 -p darknet-core --lib` passed
  (391 tests).
- Added an oxide ELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), and the real-device
  `oxide_runtime_chain::chained_conv_activation_stacks_match_cpu_backward` test.
  `MEM=3G ./tools/test_oxide.sh --no-run` also compiled every oxide test target.
- YOLO IoU/GIoU box geometry and objectness/class sigmoid-BCE deltas have
  central finite-difference coverage; the 394-test core library suite passed at
  that milestone. The box test records Darknet's legacy 0.5 width/height
  comparison factor. DIoU/CIoU deltas now have source-derived parity coverage
  for contained, edge-crossing/tied, and horizontal/vertical disjoint boxes;
  all 24 `yolo_`-filtered host tests pass. Do not use naive forward finite
  differences as their C++ reference: `box_diou` uses `(d/c)^0.6`, while `dx_box_iou`
  implements the legacy derivative of `d/c`; CIoU approximates the `alpha`
  contribution.
- f16 standalone batchnorm (via channel-affine), bias + activation, and reverse
  upsample forward are already implemented. Verified by
  `oxide_half::f16_shortcut_avgpool_affine_reverse_match_f64_reference` and
  `oxide_runtime_f16::tiny_net_with_avgpool_local_avgpool_batchnorm_reverse_upsample_matches_f32`
  (13 + 5 tests passed; 1 ignored in the half suite).
- `NORM_CHAN*` f32 convolution forward/backward is wired through CPU references
  and native Oxide ops; exact f32 adjoints are default and C++ mode uses the
  inspected CUDA diagonal shortcuts (C++ CPU custom gradients differ). The cfg
  parser permits these activations only on `[convolutional]`; f16 inference and
  other layer types reject them explicitly.
- Weighted `[shortcut]` GPU forward composes oxide channel-scale + add. Input and
  source gradients use backend elementwise multiply + shortcut-fit ops. Oxide
  assembles raw `delta * source` products and applies exact None/ReLU/softmax
  normalization derivatives on-device, including the `GradMode::Cpp` raw-product
  behavior. Only the final per-input/per-slot weight-gradient vector transfers
  to host train buffers; parameter updates/SGD remain host-resident. FakeBackend
  and real-device
  `weighted_shortcut_runtime_matches_cpu_forward_backward_and_sgd` parity pass.
- AMP runtime with dropout and dropblock now has real-device output/mask/backward/
  SGD comparison against f32 (`oxide_runtime_chain` test).
- Deferred optimizer accumulation has a real-device `subdivisions=2` CPU parity
  test across two iterations, including pending-gradient flush.
- Device convolution params support per-owner expert slots for container reuse;
  warm-up and host-update invalidation are covered by the
  `container_convolution_cache_is_keyed_by_owner_and_expert_slot` FakeBackend test.
  MoE computes router pooling/logits/top-k softmax, expert convolutions, and
  per-sample gate blending on Oxide. No-BN router connected-backward,
  device-selected gate columns, expert gate-dot reductions, load-balance plus
  top-k softmax-Jacobian backward, router-pool input-gradient broadcast, and
  expert convolution backward use backend ops. Gates and pooled/expert
  output/activation caches stay device-resident; no gate/dot values cross the
  host in normal Oxide training. Router/expert parameter gradients accumulate
  on-device across backward calls. For SGD, no-BN router/expert parameter
  updates plus momentum run on-device without gradient transfers. Router values
  sync to host after each step for the next forward; expert values support
  deferred host sync. No-BN MoE Adam updates also run on-device without gradient
  transfers; DNOPT3 persists their moments and legacy DNOPT2 host moments migrate
  on first use. DNOPT4 adds BN MoE scale moments; DNOPT1-3 remain readable, and
  nested host BN moments migrate on the first BN Adam device update. BN expert
  backward uses backend ops; BN expert SGD updates weights/beta/scales with
  device-resident momentum, and BN Adam uses DNOPT4 scale moments. The dynamic-
  loss-scale finiteness check covers dirty router/expert/BN-scale buffers.
  FakeBackend
  parity and real-device
  `moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  pass, including two backward calls accumulated before the first deferred SGD
  step, followed by a second step with upstream convolution updates. Real-device
  `moe_batchnorm_expert_device_backward_matches_cpu_on_oxide` also checks BN
  expert gradients and updates against CPU, including upstream convolution
  updates. `moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients`
  checks BN expert/scale Adam and deferred synchronization. The FakeBackend test
  `gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check` verifies
  dirty router/expert/BN-scale accumulators are checked for overflow; FakeBackend
  SGD parity also checks router momentum residency and `dont_update` gradient
  retention. `oxide_layers_extra`
  verifies soft/tied top-k, pool and pool-backward, non-multiple gate-dot
  reductions, and router backward parity with load balancing.
- Overlapping-window maxpool backward, including `maxpool_zero_nonmax`, matches
  central finite differences; real-device maxpool forward/backward/SGD parity
  also passes for `maxpool_zero_nonmax=1`.
- No-BN RNN and CRNN forward/backward use backend ops: RNN projections and
  CRNN recurrent convolutions, activation gradients, state accumulation, and
  shortcuts. No-BN LSTM forward runs gates/cell products through backend ops,
  including `vector_mul_f32`; its backward now uses a backend gate-derivative
  operation plus connected ops for projection gradients, time-accumulated
  parameters, and recurrent/input-gradient accumulation. Device training keeps
  BPTT caches resident instead of downloading projection/gate/cell caches,
  and timestep outputs are concatenated on-device. RNN/LSTM forward/backward
  sequence slicing and timestep input-gradient assembly also stay on-device;
  RNN projection gradients accumulate on-device across steps; CRNN forward
  input slicing and output assembly also stay on-device. CRNN backward sequence
  slicing and input-gradient assembly also stay on-device, with convolution
  parameter gradients accumulated across steps. Recurrent cache/state and
  parameter-gradient transfers, host accumulation/SGD, and some cache handling
  still cross the host. BN RNN/LSTM backward uses backend projection/BPTT ops;
  BN CRNN backward uses convolution/BPTT ops. All use rolling-stat affine
  input/beta derivatives; scale-gradient reductions use backend centering,
  channel-scale, multiply, and channel-reduction ops (native on Oxide), so raw
  recurrent values/deltas no longer return to host for dscale. Recurrent BN
  forward/cache capture, rolling-stat update, and optimizer remain host-side.
  FakeBackend and real-device
  `oxide_runtime_chain::batch_normalized_rnn_device_backward_matches_cpu_on_oxide`,
  `oxide_runtime_chain::batch_normalized_lstm_device_backward_matches_cpu_on_oxide`,
  and `oxide_runtime_chain::batch_normalized_crnn_device_backward_matches_cpu_on_oxide`
  compare parameter and rolling-stat updates with CPU, including upstream
  connected/convolution updates. Dynamic loss-scale finiteness checks include
  host-buffered connected/recurrent gradients.
- Host Adam updates standalone batchnorm, local, connected, weighted shortcut,
  MoE fallback, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT parameters with
  distinct moment slots and per-layer rate scaling. DNOPT3 persists no-BN device
  MoE moments and host moments; DNOPT4 adds device BN scale moments. DNOPT1-3
  remain loadable, with nested BN host moments migrating on first BN device Adam
  update. The ViT reference test covers C++ weight
  order, decay groups, clipping, and reduced mHC rate; host resume has a
  multi-step continuation test.
- `[vit]` reference path: cfg/shape, C++-ordered header-aware weights, seeded init,
  CPU analytical forward/backward + SGD, and GPU f32 host round-trip with native
  standard-flash forward/dQKV backward; 28 focused tests pass. Other transformer
  backward math remains host-side.
- Host suites: `darknet-core --lib` 390/390 and `darknet-cli` 18/18 pass. The
  real-device `oxide_runtime_chain` suite passes all 7 tests.
  `cfg_corpus` started but the combined host run hit its 30-minute command timeout;
  this is not evidence of a code failure. Historical `raptor-fp4` C++ commit
  `ceb7c974` was located for MSA and flash/VC source audit.
- `[moe_convolutional]` CPU path (commit `cdf68d00`) — 13 tests green.
- `[laya_marker_gather]` — 10 tests; `[laya_act_head]` — 9 tests;
  `[laya_loss]` — 11 tests; `[laya_encoder]` — 9 tests; `[laya_head]` — 8 tests;
  `darknet-core --lib` 354/354.
- Docs reconciled: `todo.md` now tracks open work only; `eml_convolutional`
  save order verified; `onnx.md` / `gui_plan.md` marked not started.

## Axis audit

Audited the reference implementations on branch `Jev_laya` using `git grep` and `git show`. The tests below pin source-level axis contracts with hand-computed Rust expectations; **no compiled-C++ golden exists, so this is not a C++ parity claim**. `MEM=3G ./tools/test_host.sh axis_layout` passed 18 core tests, and `MEM=3G ./tools/test_host.sh laya_` passed 66.

| Layer | Source-derived axis contract | Pinning test |
|---|---|---|
| `[vit]` | `src-lib/vit_layer.cpp::vit_patchify_cpu`: NCHW input, batch-major row-major `(patch_y, patch_x)` tokens, and `(patch_y, patch_x, channel)` patch features (channel fastest). | `vit::tests::patchify_axis_layout_keeps_channel_fastest_order_across_batches` — B=2, C=2, H=2, W=3. |
| `[swin]` | `src-lib/transformer_layer.cpp::window_partition` converts NCHW to `[B*nH*nW,T,C]` in batch/window/raster-token/channel order; `cyclic_shift` reads source `(y-dy,x-dx)` with wrap independently per NCHW plane. | `swin::tests::window_partition_axis_layout_is_batch_major_raster_then_channel`; `cyclic_shift_axis_layout_wraps_each_nchw_plane_with_source_signs`. |
| `[tucker_attention]` | `src-lib/tucker_attention_layer.cpp::forward_tucker_attention_layer` enumerates batch/window-y/window-x, packs raster tokens as `[T,C]` from NCHW, zero-pads partial windows, then scatters output back to NCHW. | `tucker_attention::tests::axis_layout_keeps_batch_window_token_and_channel_axes_independent` — B=2, C=2, H=1, W=2; per-batch attention and channel sentinel. |
| `[mambavision]` | `src-lib/mambavision_layer.cpp::forward_mambavision_layer` maps NCHW to `[B,T,C]` rows and writes NCHW output; `scan_forward` starts each batch's `[D,S]` state independently and advances in token order. | `mambavision::tests::axis_layout_maps_nchw_rows_and_scan_state_per_batch` — B=2, T=2, distinct initial states. |
| `[kimi_kda]` | `src-lib/kimi_kda_layer.cpp::flatten_cpu` maps NCHW to `[B,T,C]`; the delta scan loops per batch and raster token (no prior state at `t=0`), then `deflatten_cpu` restores NCHW. | `kimi_kda::tests::axis_layout_keeps_nchw_raster_scans_independent_per_batch` — B=2, C=2, H=1, W=2. |
| `[moe_convolutional]` | `src-lib/moe_convolutional_layer.cpp::compute_cpu_router` pools each image's C×HW plane independently; `forward_moe_convolutional_layer` applies its `[B,E]` gates to the same sample's expert outputs. | `runtime::tests::cpu_moe_convolutional_axis_layout_keeps_router_and_blend_per_batch` — B=2, C=2, H=2, W=3, distinct image means and hand-computed blends. |
| `[eml_convolutional]` | `src-lib/convolutional_layer.cpp::forward_eml_convolutional_layer` runs two NCHW convolution branches and combines matching flattened `[B,C,H,W]` elements; the residual uses the same input index when enabled. | `runtime::tests::eml_convolutional_axis_layout_preserves_batch_channel_and_spatial_order` — B=2, C=2, H=2, W=3, identity/swap branch sentinels. |
| `[dcnv4]` | `src-lib/dcnv4_layer.cpp::dcnv4_deformable_forward_cpu` partitions channels by group and enumerates kernel points kw-major/kh-minor; each point consumes `(dx,dy)` before its mask and samples x from w, y from h. | `dcnv4::tests::offset_axis_layout_reads_dx_then_dy_and_keeps_batches_nchw`; `groups_keep_channel_partitions_and_sampling_uses_zero_padding`. |
| `[laya_encoder]` | `src-lib/laya_encoder_layer.cpp::la_forward_impl` reads channel-major `[B,2,T]` token IDs/masks, transforms rows as `[B,T,H]`, then writes channel-major `[B,H,T]`. | `laya_encoder::tests::encoder_axis_layout_keeps_batch_and_token_rows_independent` — B=2, T=2 with distinct token embeddings. |
| `[laya_head]` | `src-lib/laya_head_layer.cpp::lh_forward_impl` reads `[B,C,T]`, selects qtype per sample from its qtype channel, and writes `[B,H,T]`. | `laya_head::tests::head_axis_layout_preserves_batch_sequence_and_qtype_channels` — B=2, T=2, distinct per-sample qtypes. |
| `[laya_marker_gather]` | `src-lib/laya_marker_gather_layer.cpp::forward_laya_marker_gather_layer` reads marker channel `2*j` at sequence column 0 for each batch row and gathers each hidden channel at that row's position into `[B,H,K]`. | `laya_marker_gather::tests::marker_gather_axis_layout_uses_sample_specific_positions` — two rows gather different columns. |
| `[laya_act_head]` | `src-lib/laya_act_head_layer.cpp::forward_laya_act_head_layer` builds features per batch row and stores logits at `output[b*A+a]`. | `laya_act_head::tests::act_head_axis_layout_is_batch_major_over_the_two_layer_mlp` — B=2, K=3, T=2; hand-checks both output rows. |
| `[laya_scorer]` | `src-lib/laya_scorer_layer.cpp::sc_forward_impl` converts `[B,H,K]` to B×K hidden rows, normalizes each option over H, and writes `[B,K]`. | `laya_scorer::tests::scorer_axis_layout_keeps_batches_channels_and_options_independent` — B=2, H=3, K=2. |
| `[laya_loss]` | `src-lib/laya_loss_layer.cpp::loss_forward_host` passes N and K to the row-wise loss and sums per-sample loss over the batch; invalid/padded options do not contribute. | `laya_loss::tests::loss_axis_layout_excludes_padding_rows_from_batch_mean` — B=2, K=2 with the second sample padded. |
| `[channel_slice]` | `src-lib/channel_slice_layer.cpp::forward_channel_slice_layer` copies the selected contiguous channel range independently for each batch, with each channel spanning H×W. | `channel_slice::tests::slice_axis_layout_preserves_batch_channel_and_spatial_order` — B=2, C=4, H=2, W=3. |
| `[deconvolutional]` | `src-lib/deconvolutional_layer.cpp::forward_deconvolutional_layer` applies the channel/filter GEMM per batch, scatters with `col2im`, and adds filter biases into NCHW output; weights are input-channel then filter/kernel. | `deconvolutional::tests::deconvolutional_axis_layout_maps_input_channels_to_filters_and_batch` — B=2, C=2, three filters. |
| `[connected]` | `src-lib/connected_layer.cpp:186` treats each flattened NCHW sample as an independent input row and stores weights as `[out][in]`. | `runtime::tests::connected_axis_layout_keeps_batch_rows_and_output_weight_axes` — B=2, C=2, H=2, W=3, hand-computed forward and supplied-seed adjoint. |
| `[shortcut]` | `src-lib/shortcut_layer.cpp::forward_shortcut_layer` dispatches to `src-lib/blas.cpp::shortcut_multilayer_cpu`, whose `src_i/step` selects channel/spatial weight slots while `src_b` keeps source rows separate. | `shortcut_weights::tests::shortcut_axis_layout_uses_nchw_channel_slots_per_batch` — B=2, C=2, three spatial cells/channel. |
| `[dpt_head]` | `Jev_laya:src-lib/dpt_head_layer.cpp::dpt_forward_impl` consumes four contiguous tap-major feature blocks per image and writes a batch-local NCHW depth plane through the reassembly/fusion path. | `runtime::tests::dpt_head_runtime_tests::dpt_head_axis_layout_batched_nchw_forward_matches_independent_non_square_samples` — B=2, non-square maps, distinct tap/channel markers. |

A1 from `AGENTS.md` is complete for the already-ported layer families listed here. Per-backlog axis checks remain part of the DoD for missing-layer ports. This audit is source-derived, not compiled-C++ parity.

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

## NEXT (general layer-port roadmap)

**Open-items plan gate:** OBB CPU tasks 1.1–1.4 are complete, but Task 0.2 is not green: the full uncapped host suite reports the `cfg_corpus` failures recorded above. Do not start Wave 2 until those failures are triaged or explicitly accepted. At earlier test runs the systemd cap was unavailable; the focused OBB host filter has since passed under the cap (135 core + 4 CLI). The full host baseline/cfg corpus has not been rerun under the cap and remains non-green.

1. The LAYA family is **complete within its CPU-reference scope**, not all-native GPU scope (`laya_scorer`, `laya_marker_gather`,
   `laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head`). The `[vit]`
   reference path, standard flash primitives, and VC config/runtime reference
   dispatch are wired. VC uses its Rust reference on CPU and GPU host-round-trip
   paths; native VC device kernels and compiled-C++ parity are unavailable.
   Swin/transformer and Tucker attention are now integrated in the Rust
   reference runtime. MambaVision is integrated in v0.0.271 and Kimi KDA in
   v0.0.272; other Phase 3A layers remain open. MiniMax sparse attention is an auxiliary
   utility/config option, not a standalone C++ layer; Rust wires it into ViT,
   Tucker, and MambaVision, with remaining attention callers audited alongside
   their layer ports.
   **Order matters:** `[laya_scorer]`'s `weights=` is only usable together with
   `[laya_encoder]` + `[laya_head]` (the shared-file offset is computed past
   their blobs), so those two come next even though the plan files them under
   Phase 4. They also need the Phase 3A prerequisite kernel set.
2. Phase 2A is **complete** (yolox, ppyoloe/yolonas shared DFL, centernet,
   nested-object fix, `detect::decode_modern_detections`, `detect-image` merge).
   Phase 2B `moe_convolutional` has a verified CPU path and device
   forward/backward plus device SGD/Adam (v0.0.273–v72 area; see
   `implemented.md` §7 and `layer-support.md` for the remaining partials).
   `eml_convolutional` is integrated (`LayerKind::EmlConvolutional`) with
   device forward/backward and device SGD/Adam; C++ batch-stat BN training
   remains open.
   Remaining across the phases: `cfg/LegoGears_centernet.cfg` end to end (the
   cfg lives on `Jev_laya`), 2C OBB (CARPN Stage-2 training; broader GPU OBB),
   remaining 3A/3B native GPU work, and 4.
3. **Phase 3A prerequisite (2) is complete**: LayerNorm fwd/bwd, the fused
   residual/bias epilogue, the row-softmax backward and the batched-GEMM wrapper
   (`linear_f32`) are all in, each with a host default plus GPU parity tests.
   It supports the ViT, Swin/transformer, Tucker, MambaVision, and Kimi KDA
   reference paths; other Phase 3A work and the LAYA encoder/head remain. MiniMax sparse
   attention is a shared utility, not a standalone layer. Standard
   `flash_attention` has native Oxide forward/backward primitives. VC now has
   strict config/runtime integration and uses its centered/clamped-score Rust
   LSE contract with exact piecewise backward; GPU execution intentionally falls
   back to this host reference for every device/head dimension. Native VC kernels,
   broader Phase 3A layers, and compiled-C++ parity remain open.
4. Phase 2B `moe_convolutional` (weights order verified) has device
   forward/backward and device SGD/Adam; see `implemented.md` §7. Ported next
   was `eml_convolutional`, now a real `LayerKind` (not an activation) with
   save order **re-read 2026-10-05**: `input_layer` then `self_layer`, two full
   conv blocks. `[dcnv3]` remains blocked (no reference numerics anywhere).
5. Phase 3B CPU family — DCNv1/2 CPU reference is integrated in v0.1.001
   (GPU is host fallback only); bounded single-template DSConv CPU support is
   integrated in v0.1.002 (`morph=0/1/2`), while multi-template morph 0/1
   remains open; graph-conv CPU support is integrated in v0.1.003 and WMHF in
   v0.1.004. CPU Muon is integrated in v0.1.005, Clifford in v0.1.006, and
   `recursive_block` in v0.1.007. The bounded Phase 3B CPU layer ports are
   complete; DSConv multi-template morph 0/1 remains unsupported. Stage-2 native
   GPU work for `dcnv4` and `deform_conv` and GPU Muon remains open.
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
  "NOT RUN" everywhere.

## Task plans: ONNX complete; GUI partly implemented

- `tasks/rust/onnx.md` (`darknet-rs export-onnx`): all 21 checklist steps are complete (v0.0.254–257; docs committed as `2b998051`). Core/CLI tests, host CLI build, fused/unfused exports, Python syntax, and ONNX Runtime numerical gates are recorded above and in `tasks/rust/todo.md`. Canonical external dataset paths were absent, so available matching local assets were used.
- `tasks/rust/carpn_map_plan.md`: the bounded caller-supplied sample dispatcher is implemented in v0.0.258. `tasks/rust/carpn_network_map_plan.md`'s CPU inference, `detect-image`, and direct `detector map` milestone is implemented in v0.0.259 with generated cfg/weights-backed polygon-mAP tests. CARPN Stage-2 training and scheduled `detector train -map` remain deferred; no compiled-C++ parity or canonical-dataset evidence is available.
- `tasks/rust/gui_plan.md` (egui DarkMark replacement, merged 2026-10-17):
  **partly implemented in v0.1.019–v0.1.021** — `darknet-gui-project`,
  `darknet-gui-engine` and `darknet-gui` exist, as does the `detect-server` CLI
  arm (`src/darknet-cli/src/main.rs:433`), together with JSON+txt sidecars,
  scan/filter/nav/undo, the `G9`–`G16` edit model, actions, project creation,
  atomic class remap, `detect-server` + worker, and the
  launcher/canvas/views/class-editor/overlay shell. Still open: class
  add/delete from the dialog (engine supports it), the behaviours behind mapped
  but unimplemented keys (mass-delete, homography, merge, heatmaps, screenshots,
  tiling, jump, delete-image/undo-delete), the train runner, and the Task 1.5
  DarkMark file diff. Dataset and GPU smoke gates remain environment-dependent.

The remaining ONNX bookkeeping and the GUI plan do not depend on each other.

## Current Muon optimizer status

- v0.1.005 (`2674f5741405e33371448bb9a664d482ffa02e8d`) delivers the opt-in Stage-1 CPU Muon/AdamW hybrid and DNOPT6 momentum persistence. `muon_` passed 13 tests, `weights` 43, and `runtime` 204, including `train_only_bn` gating. No training, Oxide tests, or compiled-C++ golden/parity were run. Clifford is integrated in v0.1.006 and `recursive_block` in v0.1.007; GPU Muon is Stage 2.
