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
- [ ] `kernels/gemm_mma.rs`: mma.sync + ldmatrix GEMM, tf32 (or f16) inputs, f32 accumulate
- [ ] Flag-gated `DARKNET_OXIDE_GEMM={fp32,tf32,auto}`; default fp32 until accuracy accepted
- [ ] Parity vs fp32 GEMM and CPU at relaxed tolerance (document it); conv fwd/dgrad/wgrad through it
- [ ] Interleaved perf A/B vs fp32 tiled GEMM
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

## P4 CLI + end to end
- [ ] `oxide` feature / backend selection in darknet-cli (mirror old cutile wiring at main.rs:395/557; cutile code is cfg-gated on missing feature)
- [ ] yolov4-tiny inference: detections vs CPU within tolerance
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
- [ ] Commit minimally per phase; add Review section here when done

## Review
(pending)
