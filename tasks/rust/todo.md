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
- [ ] maxpool fwd/bwd (argmax index)
- [ ] avgpool + local_avgpool fwd/bwd (check trait batch/stride semantics vs C++ first)
- [ ] upsample fwd/bwd
- [ ] route_concat / route_backward
- [ ] crop_bilinear fwd/bwd (overridden, trait default panics)
- [ ] Parity tests per op

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
- [ ] Grouped route (`groups`/`group_id`) supported by `route_concat_f32` (P3)
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
