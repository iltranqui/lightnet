# Todo (branch rust)

Plan: ~/.claude/plans/https-github-com-nvidia-cuda-rust-port-t-imperative-pumpkin.md
Goal: cuda-oxide backend `bridges/darknet-cuda-oxide` implementing `F32TensorOpsBackend` with Rust kernels ported from `src-lib/*.cu`.

Build/test (one command at a time, repo root):
`systemd-run --user --scope -p MemoryHigh=3G --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 target/debug/cargo-oxide oxide test -- --manifest-path bridges/darknet-cuda-oxide/Cargo.toml -j 4`

## Done
- [x] Flatten rust/ to repo root, vendor cuda-oxide, merged workspace, nightly-2026-08-28 pin (85fb0a91)
- [x] Drop darknet-cli cutile dep (cutile-rs-aot checkout missing) (85fb0a91)
- [x] Verify: cpp_parity, cuda-macros/device/host, darknet-core, `cargo oxide run vecadd` on RTX 4060
- [x] P0 scaffold bridges/darknet-cuda-oxide, upload/zeros/download (e4a42d27)
- [x] P1 bias_add/channel_scale/bias_backward, 7 activations fwd+bwd, vector_add/shortcut/saxpy, sgd (e4a42d27), 7 parity tests pass

## P2 Conv (src-lib: im2col_kernels.cu, col2im_kernels.cu, blas_kernels.cu gemm, convolutional_kernels.cu)
- [x] `kernels/gemm.rs`: smem-tiled GEMM NN/NT/TN with alpha/beta; parity vs CPU incl. non-tile-multiple shapes
- [x] `kernels/im2col.rs`: im2col + col2im with stride, dilation, padding, groups
- [x] `kernels/conv_direct.rs`: fwd / dgrad / wgrad direct kernels (ref semantics: cuTile conv2d_flat)
- [x] Wire `conv2d_f32`, `conv2d_backward_input_f32`, `conv2d_backward_weights_f32`
- [x] Heuristic `auto` + env override `DARKNET_OXIDE_CONV={direct,gemm,auto}`
- [x] Cached conv workspace in backend (check Send/Sync needs of `GpuExecutableNetwork`)
- [x] Tests: direct vs gemm vs CPU; groups, depthwise, dilation, stride>1, padding, 1x1, 3x3
- [ ] Decide bias_backward block reduction with `SharedArray` (current: 2-stage partials)

## Real-cfg shape coverage (scan agent writes tests/data/*.csv; tests added AFTER P3 to avoid concurrent builds)
- [ ] Scan all cfg/*.cfg (+ LegoGears_v2 cfgs) with shape propagation -> `tests/data/conv_shapes.csv` (+ maxpool/upsample/route CSVs, summary.md) via `tools/cfg_conv_shapes.py`
- [x] Test `conv_real_cfg_shapes`: every unique conv geometry, fwd + dgrad + wgrad, direct/gemm/(tf32) vs CPU reference, batch 1 (+ batch 2 sample); heavy layers use capped spatial dims or sampled output windows, documented -> DONE in `tests/oxide_cfg_shapes.rs`: `cfg_conv_direct`/`cfg_conv_gemm`/`cfg_conv_gemm_tf32` (+ batch 2 every 50th geometry)
- [x] Same for maxpool / upsample / route (grouped) geometries found in cfgs -> DONE: `cfg_maxpool_shapes`, `cfg_upsample_shapes`, `cfg_route_shapes`, `cfg_shortcut_shapes`
- [x] Explicit list of cfg geometries the oxide conv does NOT support (return Err, tested) vs must-support (LegoGears_v2 = hard requirement) -> DONE: `cfg_conv_dilated_and_non_rule_geometries` (dilation>1: Err under darknet-core dims, parity with pad*dilation); LegoGears_v2 at real size in every mode test
- [ ] Re-run with P6 fp16 and P7 bench shapes reusing the same CSV

### Scan results (done): 101 cfgs, 5845 conv layers, 990 unique geometries; C++ darknet layer table cross-check: 3822 convs, 0 mismatches
Data: `bridges/darknet-cuda-oxide/tests/data/{conv_shapes,conv_activation_pairs,activations,maxpool_shapes,upsample_shapes,route_shapes,shortcut_shapes}.csv` + `conv_shapes_summary.md`; script `tools/cfg_conv_shapes.py [--crosscheck]`.
- Convs: k1 (571 geoms), k3 (~374), k5, k7, one k11/s4; strides 1/2/4 with stride_x==stride_y always; padding==size/2 except alexnet k11; 67 grouped geoms (36 depthwise, groups up to 1152, resnext 32); dilation 2/3 only in resnet152_trident (6 geoms); xnor (74 layers) and share_index (54) NOT supported/excluded.
- Dilation caveat: BOTH C++ and Rust output-dim formulas ignore dilation -> decide: oxide mirrors that (verify how C++ applies dilation) or keep dilation out of scope and test it explicitly as Err/skip.
- Maxpool combos (size/stride/pad): 2/1/1, 2/2/1, 3/2/0, 3/2/2, 5/1/4, 9/1/8, 13/1/12 (SPP). Upsample: stride 2 (x91), 4 (x1). Routes: 1-6 sources, groups=2/group_id=1 (x30). Shortcuts: 1120, all single-source; 30 have MISMATCHED source shape (runtime currently CPU-falls-back; decide oxide support or keep fallback + test).
- Activations used: leaky 2809, mish 2005, linear 1348, swish 674, logistic 94, relu6 52 (1 cfg), relu 48, ramp 22 (1 cfg); NOT covered by oxide: relu6, ramp (tanh supported, unused).
- Test rule: dilation=1 & stride_x==stride_y & padding==size/2 & groups|channels = 984 geoms, batch 1, fwd+dgrad+wgrad; MACs>~50M geoms: keep channels/k/s/p/groups, shrink in_h/in_w to ~32-64; CPU-ref cap a few hundred MMAC/case; 3-5 full-size GPU-only layers vs cuDNN/darknet; mandatory picks per (k,s), depthwise, groups 2/32/odd filters-per-group, dilation 2&3, stem in_c 1/3, large-filters 1x1; cross with activation/BN from conv_activation_pairs.csv (one activation per geom + full coverage on a small set).
- [ ] 42 cfgs gave no C++ layer table (old cfg/unsupported/*, yolov3-openimages, LegoGears_v2_fp4sim*): not investigated
- [x] Implement `conv_real_cfg_shapes` test from the rule above (reads tests/data/conv_shapes.csv) -> DONE (`tests/oxide_cfg_shapes.rs`, ~50 s; LegoGears layers from `tests/data/legogears_v2_convs.csv` via `tools/cfg_conv_shapes.py --cfg`)
- [ ] Add relu6 (+ ramp?) to trait/oxide or assert CPU fallback (test asserts relu6/ramp are NOT expressible via F32ActivationKind; still open: add kernels or keep CPU fallback)
- [ ] Mismatched-shape shortcut decision (30 layers in cfgs) (tested: `shortcut_add_f32` returns Err for all 30 layers since flat lengths differ, no silent equal-length case in the cfgs; decision still open)

### Findings from real-cfg tests (committed): 1009 conv geoms x {direct, gemm} + 192 tf32 + LegoGears 21 layers real size, 0 failures
- DILATION: C++/darknet-core output dims ignore dilation; C++ cuDNN pads `pad*dilation`; darknet-core CPU uses plain pad (shifted window, differs from C++ GPU). Oxide uses true dilated extent: called with core convention -> Err; called with `pad*dilation` (cuDNN way) matches. If runtime routes dilated conv to oxide it must pass pad*dilation. Only resnet152_trident uses dilation.
- Mismatched-shape shortcuts (30 layers): `shortcut_add_f32` returns Err (length mismatch), runtime keeps CPU fallback. Decide in P5.
- relu(NaN)=0 on oxide vs NaN in C++ (`x*(x>0)`); mish/swish at -inf -> NaN. Decide if matching C++ matters.
- Large geoms only tested shrunk; full-size only for LegoGears. No cuDNN cross-check yet (P4/P7).
- darknet-cli main.rs has dead `cfg(feature="cutile")` blocks (unexpected_cfgs warnings) -> clean up in P4 when adding the `oxide` feature.

## Activation coverage from real cfgs (C++ has ~21 fwd + 21 bwd kernels in activation_kernels.cu; trait `F32ActivationKind` has 7: Linear, Logistic, ReLU, Leaky, Mish, Swish, Tanh)
- [ ] Scan agent also emits `tests/data/activations.csv`: per activation name, count, which cfgs/layer types (conv/connected/shortcut/yolo/...) use it
- [ ] Gap list: activations used by any cfg but missing from `F32ActivationKind`/oxide (candidates: loggy, relu6, elu, selu, gelu, hard_mish, hardtan, lhtan, plse, ramp, stair, relie, normalize_channels, normalize_channels_softmax, normalize_channels_softmax_maxval) — decide per item: add to trait+runtime (`gpu_activation_kind` mapping in darknet-core runtime.rs) + oxide kernel, or keep CPU fallback with a test that asserts the fallback is taken
- [x] For each supported activation: forward + backward parity vs CPU reference over edge values (0, +-tiny, +-large, Mish/Swish thresholds, NaN/Inf inputs behave like C++), finite-difference check, and C++ parity vs `activate_array`/`gradient_array` semantics (input vs output form, as verified for P1) -> DONE: `cfg_activations_supported_and_unsupported` (edge values, many sizes, NaN/Inf mirror; FD + output-form parity stay in oxide_parity.rs)
- [x] Real-cfg activation x conv pairing test: each (activation, conv geometry) combination that occurs in the scanned cfgs runs conv + bias + activation fwd/bwd chain on oxide vs CPU -> DONE: `cfg_conv_bias_activation_chain` (123 chains: 60 csv sample, 42 activation x geometry cross, 21 LegoGears real size)
- [ ] fp16 (P6): same activation set on f16 storage with documented tolerance; TF32/fp16 conv followed by activation
- [ ] LegoGears_v2 uses leaky (19) + linear (2): those two are hard requirements, everything else best-effort by cfg frequency

## P2b Tensor-core GEMM (after P2 baseline passes; cuda-device `wmma.rs`/`mma_frag.rs`, sm_80+, works on sm_89)
- [x] `kernels/gemm_mma.rs`: mma.sync + ldmatrix GEMM, tf32 (or f16) inputs, f32 accumulate
- [x] Flag-gated `DARKNET_OXIDE_GEMM={fp32,tf32,auto}`; default fp32 until accuracy accepted
- [x] Parity vs fp32 GEMM and CPU at relaxed tolerance (document it); conv fwd/dgrad/wgrad through it
- [x] Interleaved perf A/B vs fp32 tiled GEMM
- [ ] Keep GEMM behind an interface so wgmma (sm_90) / tcgen05 (sm_100 Blackwell) kernels can be added per arch

## Forward + backward verification (applies to every phase)
- [ ] Each op with a backward gets: forward parity vs CPU, backward parity vs CPU
- [ ] Finite-difference gradient check (central diff, f64 CPU reference of forward) on small shapes for: activations (7), conv dgrad + wgrad, bias, pooling, upsample, route, shortcut, scale
- [ ] Chained test: forward -> backward on a tiny conv+bn-free+activation stack, grads match CPU `ExecutableNetwork` backward
- [x] Add finite-difference test for existing P1 activations / bias now
- [x] Verify `activation_backward_f32` input-vs-output semantics against the runtime (darknet-core gradient uses outputs for some kinds)
- [ ] channel_scale backward (trait has none; check runtime scale-channels path in P5)

## P3 Pooling / resample / route (maxpool_layer_kernels.cu, avgpool_layer_kernels.cu, crop_layer_kernels.cu)
- [x] maxpool fwd/bwd (argmax index)
- [x] avgpool + local_avgpool fwd/bwd (check trait batch/stride semantics vs C++ first)
- [x] upsample fwd/bwd
- [x] route_concat / route_backward
- [x] crop_bilinear fwd/bwd (overridden, trait default panics)
- [x] Parity tests per op

## P3b YOLO head elementwise (src-lib/yolo_layer.cpp forward_yolo_layer_gpu / backward_yolo_layer_gpu)
- [ ] Check how `darknet-core` runtime/train.rs applies yolo activation today (CPU? skipped?) and where delta from CPU loss is consumed
- [ ] Trait method (default host impl) e.g. `yolo_activate_f32(x, batch, anchors, classes, w, h, scale_x_y)`: logistic on x,y and obj+classes slices, scale_x_y = scale*v - 0.5*(scale-1); Gaussian-yolo variant only if the runtime supports it
- [ ] Oxide kernel `kernels/yolo.rs` + fwd parity vs CPU and vs C++ head outputs; backward: delta accumulation (axpy * loss_scale * delta_normalizer) into previous-layer delta
- [ ] Loss/target assignment stays CPU (same as C++: output pulled to host, delta pushed back), keep that D2H/H2D path working with OxideTensor
- [ ] `mean_alpha` output_avg EMA only if a cfg uses it (LegoGears_v2 check)
- [ ] Finite-difference/parity test for activation+scale_x_y backward

## FINAL TARGET: match LegoGears_v2 (C++/CUDA Darknet inference is the ground truth)
Network: `DarkSuite/datasets/LegoGears_v2/LegoGears_v2.cfg` (+ `LegoGears.names`, weights e.g. `LegoGears_augmatch_best.weights`; 224x160x3, batch 16 in cfg, YOLOv4-tiny style).
Layers it needs: 21 convolutional (19 with batch_normalize, 3 with `groups=2`, leaky x19 / linear x2), 11 route (incl. grouped routes), 3 maxpool, 1 upsample, 2 yolo. No softmax/connected/dropout/reorg/avgpool, so P5 is NOT on the critical path for this net.
- [ ] Confirm BN path for inference on oxide (runtime composes bias_add/channel_scale, or fold BN into conv weights) and that all 19 BN convs work
- [x] Grouped route (`groups`/`group_id`) supported by `route_concat_f32` (P3)
- [ ] YOLO layers (2): C++ `forward_yolo_layer_gpu` (src-lib/yolo_layer.cpp:1439) runs on GPU: copy input, logistic on x,y and on objectness+classes per anchor, `scal_add` for scale_x_y, optional `mean_array` (EMA); then pulls output to CPU for decode/targets/loss and pushes delta back; backward = GPU axpy. See P3b. Verify the Rust runtime's head outputs equal C++ (activation applied, scale_x_y); mismatch = blocker
- [ ] Layer-by-layer: dump every layer output from C++ darknet (CUDA build, same weights/image) vs Rust oxide, report max abs/rel diff per layer (tolerance 1e-3 fp32 path)
- [ ] Final detections on a LegoGears_v2 image set: same boxes/classes/scores vs C++ CUDA (box IoU > 0.99, score diff < 1e-3), tolerances documented
- [ ] mAP on the LegoGears_v2 validation set: Rust oxide == C++ CUDA within run noise (same weights, same thresholds)
- [ ] Also run with `DARKNET_OXIDE_GEMM=tf32` and record the delta (must not silently become default)
- [ ] Watch the predict stream-sync pitfall (memory: A/B same weights across binaries first)
- [ ] Perf: latency/FPS oxide vs C++ CUDA (cuDNN) on this net, interleaved A/B

## P4 CLI + end to end
- [ ] `oxide` feature / backend selection in darknet-cli (mirror old cutile wiring at main.rs:395/557; cutile code is cfg-gated on missing feature)
- [ ] Inference on LegoGears_v2 via darknet-cli (see FINAL TARGET), detections vs CPU within tolerance
- [ ] A few SGD steps: loss trajectory vs CPU on same weights (no long training runs)
- [ ] Interleaved perf A/B conv throughput (laptop noise ~, no JIT startup)

## P5 CPU-fallback gaps (trait + runtime.rs)
- [ ] Extend `F32TensorOpsBackend` with default host impls: softmax, reorg, dropout, scale_channels, connected
- [ ] `kernels/layers_extra.rs`: softmax (reduction), reorg, dropout (counter-hash RNG), scale_channels, connected via gemm
- [ ] `darknet-core/src/runtime.rs`: dispatch to backend instead of CPU fallback
- [ ] SAM / recurrent stay CPU (note in TODO.md)

## P6 FP16 (C++ reference: cuda_f32_to_f16/cuda_f16_to_f32 in blas_kernels.cu, cuDNN half conv path; cuda-device has `f16.rs`, `bf16.rs`, `f16x2.rs`)
Design decision first: trait `F32TensorOpsBackend` is f32-only. Pick (a) new `F16TensorOpsBackend` trait / mixed-precision mode where weights+activations stored as f16 with f32 accumulate and f32 master weights, or (b) internal fp16 compute inside f32 API (convert at GEMM boundary). Recommend (a) for inference storage savings; write the decision here before coding.
- [ ] Decide API (a vs b), document in plan + here
- [ ] `kernels/convert.rs`: f32<->f16 (and bf16) conversion kernels, round-to-nearest, NaN/Inf handling (C++ fix_nan_and_inf)
- [ ] `OxideTensorF16` (half storage) + upload/download helpers
- [ ] FP16 GEMM via `mma.sync` m16n8k16 f16 inputs, f32 accumulate (`gemm_mma_f16`), `DARKNET_OXIDE_GEMM=fp16` (+ `with_gemm_mode`), NN/NT/TN/TT, alpha/beta, batched
- [ ] Conv fwd/dgrad/wgrad through fp16 GEMM; fp16 activations / bias / elementwise (half2 vectorized where easy); pooling/upsample/route on f16
- [ ] Training path: f32 master weights, loss scaling (`loss_scale`) and nan/inf reset like C++ mixed precision (only if training on oxide is in scope; else inference-only and say so)
- [ ] Tests: parity vs f64 CPU with documented fp16 tolerance (11-bit mantissa, overflow/underflow cases), fp16 vs f32 layer agreement, finite-difference NOT required (explain)
- [ ] LegoGears_v2 fp16 inference: detections vs fp32 C++ (expect tiny score deltas; document), mAP delta
- [ ] Bench fp16 vs tf32 vs fp32 (see P7)

## P7 Benchmarks (no training runs; interleave A/B, ~0.5 noise on laptop; reuse style of tools/fp8_train_bench.py)
- [ ] `bridges/darknet-cuda-oxide/benches/` (or `tools/oxide_bench.py` + Rust bench bin): warmup, N iters, median/min/p95, cudaEvent timing, JSON+CSV output with git hash, GPU name, clocks, mode env vars
- [ ] Micro: GEMM (fp32/tf32/fp16, sizes 256..4096 + conv-shaped), im2col, elementwise, activations, bias, pooling, upsample, route: GFLOP/s or GB/s and % of roofline (RTX 4060: ~15 TFLOPs fp32, ~272 GB/s)
- [ ] Conv layers: all 21 LegoGears_v2 conv shapes (fwd, dgrad, wgrad) direct vs gemm vs tf32 vs fp16, vs C++ cuDNN where measurable
- [ ] Layer-wise + end-to-end inference latency/FPS on LegoGears_v2 (batch 1 and 16): oxide {fp32,tf32,fp16} vs C++ CUDA (cuDNN, and cuDNN off) vs Rust CPU vs (cuTile if restored); interleaved A/B, same weights/image, stream-sync correct (memory: predict stream-sync bug)
- [ ] Training step time (fwd+bwd+sgd) for LegoGears_v2, a few steps only, oxide vs C++
- [ ] Memory: peak device memory, workspace size
- [ ] Nsight Compute (ncu) pass on hottest kernels: occupancy, achieved bandwidth, bank conflicts; record findings and drive optimization (pipelined GEMM with cp.async, batched im2col, fused bias+act)
- [ ] Results table committed as docs/OXIDE_BENCH.md (git add -f) and summarized in root TODO.md

## Housekeeping
- [ ] Update root TODO.md: add oxide backend status, cuTile disabled note
- [ ] Re-enable cuTile dep once bridges/cutile-rs-aot is restored (or remove bridge)
- [ ] Fix rustup `clippy-preview` conflict (run with RUSTUP_TOOLCHAIN until fixed)
- [ ] `cargo oxide` alias unusable from bridge dir; keep README note in bridges/darknet-cuda-oxide
- [ ] Build race: `cargo oxide ... -j 4` can SIGBUS in `opt` (lib + lib-test codegen share darknet_cuda_oxide*.ll/.ptx in the crate dir); use -j 1 until fixed (all tests then pass)
- [ ] TF32 perf: only ~1.2x fp32 on 1024^3 (no cp.async double buffering); conv wgrad ~30% SLOWER with TF32; conv time dominated by im2col/col2im + per-batch launches. Optimize later (pipelined GEMM, batched im2col), not blocking
- [ ] Commit minimally per phase; add Review section here when done

## Review
(pending)
