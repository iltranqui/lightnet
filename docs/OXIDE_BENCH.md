# cuda-oxide backend benchmarks (P7)

Measured 2026-10-01 on an RTX 4060 Laptop (sm_89, 24 SM, driver 610.62, WSL2), git `fcc4d842` + uncommitted P7 work,
release builds, LegoGears_v2 (224x160x3, 21 convs) with `LegoGears_augmatch_best.weights`.
Raw data: `tools/oxide_bench.py --out <dir>` writes `results.json` / `results.csv` (git hash, GPU, clocks, driver, `DARKNET_*` env).
Roofline used for percentages: 15 TFLOP/s fp32, 272 GB/s (marketing peaks; the laptop part does not sustain them, see caveats).
No training runs were made.

## Methodology

* Harness: `src/darknet-cuda-oxide/tests/oxide_bench.rs` (ignored tests `bench_micro`, `bench_conv`) + `tools/oxide_bench.py` (driver) +
  `tools/oxide_bench_cpp.cpp` (C++ CUDA Darknet probe linking `build/src-lib/libdarknet.so`; built on demand into `target/oxide_bench/`).
  End-to-end Rust numbers come from `DARKNET_BENCH_ITERS` / `DARKNET_BENCH_BATCH` in `darknet-rs detect-image` (`BENCH_JSON` line).
* Micro/conv: warm-up launch, then per sample the mean device time of N back-to-back launches measured with CUDA events on the backend
  stream (N calibrated to ~20 ms), 7 samples, median / min / p95 over the samples. Modes (fp32/tf32, direct/gemm) are interleaved sample by
  sample because laptop clocks drift. Because launches are back to back, the figures include the per-launch gap: sub-30 us items are launch bound.
* End to end: per variant a fresh process does 5 (oxide: 5; C++: 10) warm-up forwards then N=100 timed, fully synchronised forwards
  (oxide `forward_compact` downloads the yolo heads; C++ `network_predict` pulls the output). 5 interleaved rounds (each variant once per
  round), statistics over all 500 samples. Same weights; the oxide process uses the PNG, the C++ probe a fixed random tensor (timing is data
  independent). Both include H2D of the input and D2H of the heads/outputs, not image decode or NMS.
* Memory: oxide `cuMemGetInfo` delta after load+warm-up up to the end of the run (incl. buffer pool), workspace = cached im2col buffer; C++ delta
  of free device memory across load+run (includes cuDNN workspaces and the fp16 shadow weights).

## 1. End-to-end LegoGears_v2 inference

| variant | batch 1 median / min / p95 ms | FPS | batch 16 median / min / p95 ms | img/s |
|---|---|---|---|---|
| oxide fp32 | 2.36 / 1.85 / 4.39 | 424 | 23.0 / 20.7 / 26.4 | 696 |
| oxide tf32 | 2.33 / 1.80 / 4.47 | 429 | 19.8 / 16.7 / 23.0 | 808 |
| oxide fp16 (f16 storage, P6) | 1.79 / 1.33 / 3.36 | 557 | 18.2 / 15.5 / 21.0 | 878 |
| C++ CUDA, cuDNN fp16 (default) | 5.40 / 3.69 / 11.0 | 185 | 14.1 / 10.9 / 17.6 | 1132 |
| C++ CUDA, cuDNN, `cudnn_half=0` | 4.61 / 3.01 / 9.23 | 217 | 14.2 / 10.8 / 17.4 | 1127 |
| Rust CPU (`darknet-core`, 1 run per round, 2 rounds) | 3371 | 0.3 | 54098 | 0.3 |

* fp16 row (added with P6, separate interleaved run of 5 rounds x 100 iters together with oxide fp32/tf32 and both C++ rows; fp32 2.31 / 23.1, tf32 2.33 / 19.6,
  C++ cuDNN fp16 5.06 / 14.0 ms in that run): weights + activations stored as f16, f32 accumulate, bias + activation fused into the GEMM epilogue (no f32 activation
  round trip). Batch 1: 1.29x faster than oxide fp32/tf32 and 2.8x faster than C++ cuDNN fp16; batch 16: 1.27x vs fp32, 1.08x vs tf32, still 1.3x behind C++
  (the per-image im2col + GEMM loop was still in use then; batched since, see 1.1). Device memory delta: b1 +28 MB (f32 weights are kept next to the f16 copies, f16 im2col scratch 0 reported
  because `workspace_mb` only counts the f32 workspace), b16 +86 MB (f32 path +184 MB: f16 activations halve the pool).
* Batch 1: oxide is ~2x faster than C++ (launch-light, no cuDNN heuristics); round medians are stable (oxide 2.2-2.5 ms).
  Batch 16: C++ is ~1.4-1.6x faster (cuDNN implicit-GEMM vs our per-image im2col + GEMM loop, see hotspots).
* "cuDNN off" is not switchable at run time (compile-time `CUDNN`); the second C++ row sets `net.cudnn_half = 0` after load (cuDNN fp32 path). Measured
  difference is within noise at b16, so on this net cuDNN fp16 buys little. A true cuDNN-less build was not made.
* Rust CPU is the unoptimised reference path (single thread), not a performance target.
* Memory (MB, device): oxide b1 +18 (workspace 4.9, 12.3 cached in the pool), b16 +184 (workspace 4.9, 150 pool); C++ b1 76, b16 348 (332 with `cudnn_half=0`).
  Pool cap is `DARKNET_OXIDE_POOL_MB` (default 1024).

### 1.1 fp16 inference: batched im2col + one GEMM per chunk, more f16 layers, bf16 GEMM (2026-10-01)

Batched f16 conv (`conv2d_bias_act_f16`: one im2col launch and one GEMM launch per chunk of images, instances = image x group, column buffer capped at 32 MB, 1x1 convs with
aligned K / H*W skip im2col) was already in the tree when this section was written, so the P6 remark above ("per-image im2col + GEMM loop is unchanged") and the b16 fp16 row
(18.2 ms) are stale. A/B with the existing switch `DARKNET_OXIDE_CONV_CHUNK=1` (= old one-launch-per-image loop), release `darknet-rs`, LegoGears_v2, interleaved, 5 rounds x 100 iters
(`tools/oxide_bench.py --suites e2e --batches 1,16 --variants oxide-fp16,oxide-fp16-loop,cpp-cudnn-half`, results in `target/oxide_bench/f16_batched/`):

| variant | batch 1 median / min / p95 ms | FPS | batch 16 median / min / p95 ms | img/s | device MB delta b1 / b16 |
|---|---|---|---|---|---|
| oxide fp16, batched (default) | 2.04 / 1.44 / 4.10 | 489 | 9.55 / 9.06 / 10.9 | 1676 | 28 / 110 |
| oxide fp16, per-image loop (`CONV_CHUNK=1`) | 1.93 / 1.43 / 4.22 | 519 | 19.0 / 16.0 / 22.0 | 842 | 28 / 86 |
| C++ CUDA, cuDNN fp16 | 5.20 / 3.70 / 8.82 | 192 | 14.8 / 11.3 / 18.4 | 1078 | - |

Batch 16: 2.0x faster than the per-image loop and 1.55x faster than C++ cuDNN fp16 (it was 1.3x behind); batch 1 is unchanged (a single image is one chunk; 2.04 vs 1.93 ms is
within the ~5% laptop noise, round medians 1.83 - 2.20 vs 1.78 - 2.33). The batched buffer costs +24 MB at b16.

Layers added to the f16 path (so non-LegoGears nets run under `forward_compact_f16` instead of `UnsupportedLayer`): shortcut (same-shaped sources, and the mismatched crop /
zero-pad source, activation fused into the last add so there is one rounding), global avgpool, local_avgpool, standalone batch norm (folded scale + bias) and standalone
bias + activation (one `channel_affine_act_f16` kernel), and `reverse` upsample. Still `UnsupportedLayer` in f16: connected,
reorg, softmax, scale_channels, sam, crop, recurrent layers, shortcut without sources. Accuracy vs the f32 path (`tests/oxide_runtime_f16.rs`, random He-initialised weights and BN
statistics, scale-normalised max error of the yolo heads / final output): tiny nets 2.8e-4 .. 8.8e-4; yolov3-tiny 1.3e-3, yolov4-tiny 1.3e-3, yolov3 (75 convs, 23 shortcuts, 3 heads, 128x128)
6.2e-4, resnet101 up to the global avgpool (33 shortcuts, 4 resized) 6.7e-4; tolerance 4e-3 (tiny) / 5e-3 (real cfgs). The element ops are tested against f64 references to
5.2e-4 |ref| + 1e-7 (one binary16 rounding) in `tests/oxide_half.rs`. LegoGears_v2 detections with `DARKNET_OXIDE_GEMM=fp16` on the 80 validation images are byte-identical to the
recorded fp16 run (mAP 99.6685%).

bf16 GEMM: `OxideBackend::gemm_bf16` (bf16 operands, f32 accumulate / output, `mma.sync m16n8k16 .bf16`) reuses the f16 kernel through a `bf16` flag; the tensor-core step is
instantiated twice (macro, branch outside the unrolled k loop). A plain in-loop `if bf16 { mma_bf16 } else { mma_f16 }` cost 40% of the f16 GEMM (4096^3: 5.33 -> 9.18 ms) and was
dropped; with the macro split 4096^3 is 5.33 -> 5.38 ms (25.8 -> 25.5 TF/s), 2048^3 0.79 -> 0.80 ms, 21 LegoGears conv shapes 0.63 - 0.67 -> 0.66 - 0.68 ms (all within noise,
`f16_gemm_bench`). bf16 is a GEMM only (no bf16 conv / activations); error model as for f16 (|err| <= 2e-5 S + 1e-6, measured worst err/S 1.3e-7). Not done: 3-stage cp.async
pipeline / bigger tiles (b16 e2e is now 9.5 ms, GEMM already runs at 25.5 TF/s on 4096^3; no quick measured win expected, so not attempted).

## 2. Micro-benchmarks (device time, median of 7)

GEMM (fp32 vs tf32 vs fp16, `GemmMode` forced; % of the 15 TF fp32 peak, tensor-core modes shown against the same number, so >100% is possible; fp16 = native f16 operands, f32 out,
n and k padded to multiples of 8 as in the conv path, so the timed problem is slightly larger for the lego shapes):

| shape | fp32 ms | fp32 GF/s (%) | tf32 ms | tf32 GF/s (%) | fp16 ms | fp16 GF/s (%) |
|---|---|---|---|---|---|---|
| 256^3 | 0.034 | 983 (6.6) | 0.031 | 1086 (7.2) | 0.033 | 1004 (6.7) |
| 512^3 | 0.078 | 3452 (23) | 0.045 | 5960 (40) | 0.038 | 7092 (47) |
| 1024^3 | 0.454 | 4726 (32) | 0.179 | 11991 (80) | 0.098 | 22000 (147) |
| 2048^3 | 3.49 | 4918 (33) | 1.30 | 13223 (88) | 0.681 | 25217 (168) |
| 4096^3 | 27.6 | 4977 (33) | 11.3 | 12196 (81) | 5.31 | 25908 (173) |
| lego m64 n2240 k576 | 0.052 | 3198 (21) | 0.038 | 4359 (29) | 0.035 | 4712 (31) |
| lego m256 n140 k3456 | 0.079 | 3136 (21) | 0.041 | 6053 (40) | 0.036 | 6896 (46) |
| lego m512 n35 k2304 | 0.046 | 1811 (12) | 0.037 | 2240 (15) | 0.032 | 2568 (17) |
| lego m30 n35 k512 (head) | 0.029 | 37 (0.2) | 0.029 | 38 (0.3) | 0.027 | 40 (0.3) |

All 21 conv-shaped GEMMs are in `results.csv` (`gemm_lego00..20`): the 21 take 0.03-0.08 ms each, i.e. a 25-30 us launch/event floor for most
(m<=128 or n<=140). fp32 sustains ~33% of peak on big squares (4.9 TF/s); tf32 reaches 12-13 TF/s; fp16 (m16n8k16, ldmatrix, BK=32) reaches 25-26 TF/s, 2.1x tf32.

Memory-bound ops (64 MB tensors unless noted; % of 272 GB/s):

| op | GB/s | % | note |
|---|---|---|---|
| vector_add / saxpy / shortcut_add | 230 | 84 | |
| bias_add / channel_scale | 239 | 88 | |
| activations (leaky, relu, logistic, mish, swish, tanh) | 229-239 | 84-88 | all bandwidth bound, math is free |
| activation_backward (leaky) | 242 | 89 | |
| sgd_update | 229 | 84 | |
| bias_backward | 205 | 75 | |
| maxpool 2x2 s2 | 245 | 90 | |
| maxpool 3x3 s1 pad 2 | 104 | 38 | window re-reads, no shared-memory tiling |
| upsample x2 | 163 | 60 | |
| route_concat (2 x 16 MB -> 33 MB) | 324 | 119 | L2 hits (same input twice); fine |
| global avgpool [8,64,128,128] | 20 | 7.5 | one thread/plane, 1.64 ms |
| softmax 8192 x 1000 | 39 | 14 | one thread per group, 1.67 ms |
| BN train fwd / bwd [8,64,64,64] | 146 / 221 | 54 / 81 | |
| im2col (lego layers, 1 image) | 5-129 | 2-47 | 14-50 us, launch bound except layers 1-3, 19 |

## 3. LegoGears_v2 conv layers, direct vs gemm (batch 1, device ms, sums over the 21 layers)

| pass | direct | gemm fp32 | gemm tf32 | gemm fp16 (f16 storage, fused bias epilogue) |
|---|---|---|---|---|
| fwd | 7.67 | 1.24 | 1.04 | 1.01 |
| dgrad | 9.57 | 1.30 | 1.10 | - (inference only) |
| wgrad | 58.2 | 1.50 | 1.28 | - (inference only) |

(fp16 fwd re-measured in the P6 run next to fp32 1.24 / tf32 1.07 ms: the layer sums are launch-bound, so the 21 convs only gain 5-20%; end to end the gain is larger
because bias + activation are fused away and activations move as halves.)

Worst direct kernels: wgrad of layer 14 = 18.3 ms (gemm 0.11), layers 16/19 wgrad 9 ms. Direct only wins for the tiny 1x1 head convs (layers 17, 18, 20:
0.03-0.07 ms vs 0.04-0.05 ms), i.e. `Auto` threshold is right. Per-layer numbers: `results.csv` rows `conv00_fwd` ... `conv20_wgrad`. tf32 beats fp32 by
10-35% on the big layers and ties on the rest (launch floor of ~40 us per conv = im2col + GEMM [+ split-K reduce]).

## 4. Hotspots (nsys kernel summary, oxide fp32; ncu on batch 1)

Per forward at batch 1 the GPU is busy ~1.1 ms of the 2.3 ms wall (the rest is host launch gaps, 6 launches per conv layer: im2col, sgemm, split-K reduce,
bias, activation + routes). Kernel share of GPU time, batch 1 / batch 16:

| kernel | b1 | b16 | findings (ncu, 256-thread blocks) |
|---|---|---|---|
| `sgemm_tiled` | 53.6% | 56.2% | 122 regs/thread, ~30% achieved warp occupancy at 96 blocks, tiny layers launch 4-16 blocks (SM throughput 9-36%, ~11 us); large layers 76-80% SM throughput but 5-38 K shared bank conflicts per launch |
| `im2col` | 27.7% | 29.2% | one launch per image per conv (11781 launches at b16 over 36 runs); layer 19 (1890 blocks) 22 us at 73% SM, small layers are 2-3 us launch bound |
| `gemm_splitk_reduce` | 5.9% | 6.3% | 3-7 us, DRAM bound (52-62% of peak DRAM) because partials go through global memory |
| bias_add / route_copy / activation | 4.5 / 4.0 / 3.2% | 2.9 / 2.7 / 2.0% | 756/648/756 small launches |
| maxpool, upsample | 1.2% | 0.8% | |

Findings:

1. Batch 1 is launch-bound: GPU time ~1.1 ms vs 2.3 ms wall, ~106 launches per forward.
2. Batch 16 is GEMM + im2col bound (~85% of GPU time) and issues 16x the launches because conv loops over images (im2col + GEMM per image). It is 1.6x slower than cuDNN.
3. fp32 GEMM reaches only 33% of the fp32 roofline; tf32 80-88% on large problems, but conv-sized GEMMs (n <= 2240) are far from either.
4. Pure elementwise ops are already at 84-90% of memory bandwidth; the way to speed them up is fusion (bias+activation), not tuning. Outliers: global avgpool, softmax, maxpool 3x3 s1.

## 5. Next optimisation steps (ordered by expected gain)

1. CUDA graph capture of the whole forward (static shapes after the pool warm-up): removes the ~1.2 ms/forward of launch gaps at batch 1 (106 launches, ~10 us each) and most of the
   16x-launch cost at batch 16. C++ Darknet has `use_cuda_graph` for the same reason. Needs a graph-capable launch path in `cuda-core` (stream capture + the buffer pool must
   not allocate during capture).
2. Batched im2col and a batched/grouped GEMM over the image loop (one launch per layer instead of one per image), or an implicit-GEMM conv that skips the column buffer entirely
   (removes the 28% im2col share and the 4.9 MB workspace).
3. Fuse bias + activation (and BN-folded scale) into the GEMM epilogue: saves 2 launches and 2 full passes over the output per conv.
4. GEMM: reduce shared-memory bank conflicts and registers (122 regs caps occupancy at ~30%), make tf32 the default for large problems (`DARKNET_OXIDE_GEMM=auto`),
   keep split-K partials in shared/L2 (an atomics-free two-pass reduce already exists; fuse it into the last block).
5. Small kernels: block-parallel softmax and global avgpool (14% / 7.5% of bandwidth), tiled maxpool 3x3.
6. ~~fp16 activations and weights pre-packed per layer~~ done (P6, see the fp16 rows); bf16 only has convert kernels. Batched im2col + one GEMM per chunk is done (1.1: b16 9.5 ms, 1.55x faster than C++ cuDNN fp16), bf16 GEMM exists (1.1); still open: 3-stage cp.async pipeline, implicit-GEMM conv.

## 6. Not done / caveats

* Training-step time (fwd + bwd + SGD) is now measured with a synthetic driver, see section 7; opt-in mixed-precision training (`DARKNET_OXIDE_AMP=1`) is section 7.2 (no convergence run).
* fp16 (P6) inference: f16 storage for conv / maxpool / route / upsample (the f16 dgrad/wgrad exist only in the AMP training mode, 7.2), f16 shortcut / avgpool / local_avgpool / batch norm / `reverse` upsample were added later (1.1); connected, reorg, softmax, scale_channels, sam, crop and recurrent layers still make the CLI
  error with `UnsupportedLayer` under `DARKNET_OXIDE_GEMM=fp16`. Accuracy and detection deltas are in `tasks/rust/todo.md` (P6). cuTile is disabled, so no cuTile column.
* Laptop GPU: clocks at benchmark start were idle (210 MHz, boosting to ~3 GHz under load); sustained clocks and thermals vary between runs, so treat differences below ~5% as noise
  (round medians in `results.json` show the spread, e.g. oxide b1 2.18-2.54 ms).
* % of roofline uses marketing peaks; fp32 sustained clock may be lower than boost.
* ncu was run on batch 1 only, kernels 100-130 of the launch stream (second forward), with a limited metric set.

## 7. Training step, synthetic (no dataset, no real training; added 2026-10-01)

Driver: `darknet-rs bench-train <cfg> [steps] [warmup]` (env `DARKNET_BACKEND=cpu|oxide`, `DARKNET_BENCH_BATCH`) and the `train` mode of `tools/oxide_bench_cpp.cpp`
(`oxide_bench_cpp <cfg> - <batch> <iters> <warmup> 0 train`, batch from the cfg). Random He-initialised weights, random input in [0,1), SGD lr 1e-4, momentum/decay from the
cfg. LegoGears_v2 (21 convs + BN, 224x160, 2 yolo heads), batch 1 and 16, 20 timed steps after 3 warm-up, 3 interleaved rounds (each side once per round), fp32 everywhere.

What a "step" is on each side (they are not identical work, read the table with this):
* Rust (oxide and CPU): `forward_training` + `backward_accumulate_seeded` + `update_weights`. The loss is not computed; instead random output deltas (+-0.1) are injected
  on both yolo layers (C++ and Rust both compute the yolo loss on the host in real training, so it is excluded here). The oxide forward downloads every layer output to the host
  (`forward_training_with_state` keeps a CPU copy of all activations for the host loss / fallback paths) and the update downloads the updated weights/BN stats to refresh the
  host parameters; both are included in the wall clock and are a large part of the gap below.
* C++: `train_network_datum_gpu` (device forward + backward, includes the yolo loss kernel with an all-zero truth) + `update_network_gpu` + `cudaDeviceSynchronize`.
  cuDNN `cudnn_half` is whatever the cfg parser picks for the GPU (1 on sm_89): mixed precision tensor-core convs. Overriding it after parsing breaks the cuDNN descriptors,
  so an fp32 C++ row was not made. BN is not fused (training buffers).

| variant | batch 1 step ms (round medians) | batch 16 step ms (round medians) | peak device MB b1 / b16 |
|---|---|---|---|
| oxide fp32 (Rust kernels) | 65.2 (65.2 / 65.8 / 64.9) | 198 (198.0 / 199.4 / 196.8) | 230 / 800 |
| C++ CUDA (cuDNN, cudnn_half=1) | 15.7 (15.5 / 15.7 / 16.5) | 45.6 (45.7 / 44.2 / 45.6) | 130 / 616 |
| Rust CPU (single thread) | ~15 400 (4 steps: 16.3, 15.3, 15.5, 13.6 s) | not run (one step would take ~4 min; expect ~16x batch 1) | host RSS peak 117 MB (b1) |

Phase split of the oxide step (median over the 3 rounds): batch 1 = fwd 13.6 / bwd 20.3 / update 31.4 ms; batch 16 = fwd 97.0 / bwd 67.0 / update 31.6 ms.
The update phase is a constant ~31 ms because it round-trips every parameter through the host (`sync_convolution_owner_from_device` after each optimizer step) and
re-folds BN into the inference weights; the forward is dominated by the per-layer output downloads at batch 16. The transfers were not timed separately, so the device-only share is unknown (not verified);
C++ is 4.2x (b1) / 4.3x (b16) faster end to end here, and the host round trips above are the suspected main cause rather than kernel throughput
(inference kernels reach parity with C++ at b16, see section 1). Peak device memory is measured as the `cuMemGetInfo` free-memory delta after the backward and after the update
(includes the buffer pool, so it is an upper bound of live tensors): oxide 230 / 800 MB vs C++ 130 / 616 MB.
Not measured: C++ fp32 training (see above), cuDNN-less, a long run (convergence).

### 7.1 Training-step optimisation (2026-10-01): where the time went and what was fixed

nsys (13 steps, `nsys profile --stats=false` + `nsys stats -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum`) of the step above, before the fixes:
GPU kernel time was only ~6 ms/step at batch 1 (of 65) and ~52 ms/step at batch 16 (of 198). The rest was host work:
b1 = 200 small H2D + 200 D2H copies per step and 5200 stream syncs / 13 steps; b16 = 3.6 GB of D2H in 13 steps (every layer output + activation cache downloaded
by the training forward: 1.4 s of cuMemcpyDtoH = 51% of the API time), and per BN layer per step a full host fold of the BN weights + 4 uploads (backward rolling-stat refresh and update).

Fixes (all in `src/darknet-core/src/runtime.rs`, semantics unchanged, GPU-vs-CPU tests untouched):
1. `forward_training_device`: the CLI/bench training forward downloads only the yolo head outputs (what the host loss reads). Networks that need the CPU backward fallback keep the full host copies. `forward_training_with_state` still returns everything.
2. BN fold on the device: after an update the BN-folded inference weights are `channel_scale(raw_w, gamma/sqrt(var+eps))` on the GPU (same arithmetic as the host fold) instead of download + host fold + upload; only the small bias vector round-trips.
3. Lazy fold: `folded_stale` flag on the cached device params; the update and the rolling-stat refresh only mark the folded copies stale, the training forward/backward use the raw tensors, and the fold runs the next time an inference forward needs it.
4. `update_weights_gpu_deferred` + `sync_host_weights_from_device`: no host weight download per step; the CLI trainer syncs before every `save_weights`. `update_weights_gpu` keeps the eager sync (used by the tests, same host-visible result). The deferred variant falls back to eager when the CPU backward fallback reads host weights.

Result (oxide fp32, LegoGears_v2, 20 steps, 3 rounds, median step ms; C++ numbers from the table above):

| step | before b1 | after b1 | before b16 | after b16 |
|---|---|---|---|---|
| fwd | 13.6 | 6.2 | 97.0 | 20.1 |
| bwd | 20.3 | 9.0 | 67.0 | 49.2 |
| update | 31.4 | 5.4 | 31.6 | 5.8 |
| total | 65.2 | 20.4 (20.4 / 19.4 / 20.4) | 198 | 76.3 (76.3 / 76.0 / 79.7) |
| C++ cuDNN | 15.7 | 15.7 | 45.6 | 45.6 |

Oxide is now 1.3x (b1) / 1.7x (b16) of the C++ cuDNN step (was 4.2x / 4.3x). The "after" rounds were run back to back (oxide only, C++ not re-run); the "before" column is the interleaved table above.

What remains: b1 is launch bound (~730 kernel launches and ~280 sync copies per step vs ~6 ms of kernels; the update does sanitize (alloc + copy) + sanitize + sgd per tensor, i.e. 3 launches and 3 allocations per weight/bias/scale; BN training uploads gamma/rolling stats per layer per forward). At b16 the step is kernel bound (sgemm_tiled 57%, im2col 20%, col2im 6%), the gap to cuDNN is GEMM throughput, not host traffic.
Next: a fused sanitize+SGD kernel (and a single multi-tensor launch), keep BN rolling stats/gamma on the device, CUDA graph capture of the step, tensor-core GEMM for the f32 conv fwd/bwd.

### 7.2 Mixed-precision (AMP) training step (2026-10-01)

Opt-in: `DARKNET_OXIDE_AMP=1` (implies the f16 tensor-core GEMM; default stays f32), `OxideBackend::with_amp(true)`. f32 master weights and f32 optimizer state, f16 working copies
of the conv weights (refreshed per update), f16 activations / gradients, f32 accumulation, BN statistics and reductions in f32, f32 weight / bias / BN-scale gradients; conv fwd /
dgrad / wgrad on the f16 `mma.sync` GEMM; `[net] loss_scale` as before. Ops without an f16 implementation widen to f32 at the boundary (see `src/darknet-cuda-oxide/src/mixed.rs`).
Same driver and conditions as section 7 (`bench-train`, LegoGears_v2, 20 steps after 3 warm-up, 3 rounds interleaved f32 / AMP, run back to back, medians of the per-round medians):

| variant | b1 step ms (rounds) | b1 fwd / bwd / update | b16 step ms (rounds) | b16 fwd / bwd / update | peak device MB b1 / b16 |
|---|---|---|---|---|---|
| oxide fp32 | 26.8 (24.4 / 27.7 / 26.8) | 8.8 / 9.5 / 7.3 | 82.7 (82.6 / 90.3 / 81.9) | 21.2 / 54.1 / 6.9 | 230 / 800 |
| oxide AMP (fp16 GEMM + f16 storage) | 25.9 (25.4 / 26.5 / 25.9) | 9.0 / 8.6 / 6.8 | 62.9 (62.9 / 67.5 / 60.3) | 17.6 / 37.8 / 6.7 | 238 / 584 |

AMP is 1.3x faster at b16 (bwd 1.43x) and uses 27% less device memory; at b1 it is within noise (launch-bound, see 7.1). Not optimised yet: wgrad launches one GEMM per image
(K = H*W per launch), dY has an unaligned row pitch (H*W not a multiple of 8) so the dgrad / wgrad GEMMs take the element-loop load path, BN / bias / activation are standalone
passes, the f32 `[net]` update path is unchanged. The f32 row here (82.7 ms) is the same code as the 76.3 ms "after" row of 7.1 re-measured later (laptop drift, about +8%).
Accuracy of the mode (no training run, same init / input / seeds, `DARKNET_BENCH_DUMP` = conv weight deltas after 1-3 SGD steps, `lr 1e-3`): with the trained
`LegoGears_augmatch_best.weights` the head conv delta differs from f32 by 8.8e-4 (max-normalised) and the first conv by 3.9e-2, exactly the profile of the TF32 f32-API mode
(5.8e-4 / 3.0e-2), i.e. the AMP error is operand rounding that already exists in the compute-only modes, not an AMP-specific loss; with random He init + random
output deltas (the default synthetic bench, BN gradient explosion) all reduced-precision modes (tf32, fp16 compute, AMP) differ from f32 by 10-30% in the early layers and
AMP == tf32 there too. Tests: `src/darknet-cuda-oxide/tests/oxide_amp.rs` (per-op vs f64, tiny net gradients and SGD tracking).

### 7.3 Training-step host/launch overhead (2026-10-01)

Same driver and conditions as 7.1 (`bench-train`, LegoGears_v2, 20 steps after 3 warm-up, 3 rounds, before/after binaries interleaved per round, oxide f32; AMP rows with `DARKNET_OXIDE_AMP=1`).
Changes (semantics unchanged):
1. Fused sanitize + SGD / Adam: `sanitized_sgd_update` / `sanitized_adam_update` kernels (reset_nan_and_inf(updates), optional fix_nan_and_inf(weights), the optimizer, and the zeroed updates for Adam) behind the new defaulted trait methods `sanitized_sgd_update_f32` / `sanitized_adam_update_f32` (default = the old composition, so other backends are untouched). 3 launches + 2 temporaries per tensor become 1 launch.
2. BN gamma and rolling stats live on the device (`GpuConvolutionLayerParams.bn`): the training forward no longer uploads them, the backward stores the device EMA instead of downloading it, `update_gpu_batch_norm_scales` steps the device gamma, and the inference fold is a device kernel (`batchnorm_fold_f32`: scale and folded bias) instead of a host fold + uploads. The deferred path (`update_weights_gpu_deferred`) now also leaves biases, gamma and rolling stats on the device; `sync_host_weights_from_device` (called before every save, dump and by the bench) copies weights, biases, gamma, mean, variance to the host. The first backward after a fresh `update_weights_gpu` (eager) still downloads the rolling stats; `update_weights_gpu` and the CPU-fallback networks keep the old eager host-visible behaviour.
3. No zero-fill where every element is overwritten: conv dgrad `dx` (direct kernel and the col2im gather both store), conv wgrad `dw` (direct block per element, GEMM `beta = 0` on the first image, `beta == 0` never reads C), full-coverage route gradients. BN `mean` fold fused into `bn_mean_finalize` (same fixed-order sum). Fusing the variance fold changed the rolling variance by 1 ulp (FMA contraction), so it was reverted to keep bit-exactness.

| step (round medians ms) | before b1 | after b1 | before b16 | after b16 | before AMP b16 | after AMP b16 |
|---|---|---|---|---|---|---|
| total | 20.8 (22.3 / 19.8 / 20.8) | 11.9 (11.9 / 11.6 / 12.4) | 74.5 (74.5 / 80.2 / 74.2) | 63.2 (62.9 / 63.2 / 63.6) | 60.8 (59.2 / 62.1 / 60.8) | 46.5 (47.1 / 44.5 / 46.5) |
| update phase | 5.5 | 0.8 | 5.8 | 1.0 | 6.8 | 0.85 |
| AMP b1 total | 20.8 | 12.0 | | | | |

b1 is 1.75x faster (now below the C++ cuDNN 15.7 ms of section 7), b16 1.18x (AMP 1.3x; C++ 45.6 ms). The phase split moves because work is no longer forced to complete inside the update (the yolo-head download of the next forward is the sync point), read totals, not phases.
nsys (13 steps, per step): b1 launches 690 -> 589, cuMemcpyDtoH 80 -> ~2 steady state (121 total over 13 steps: the first eager step plus the head downloads), cuMemcpyHtoD 87 -> ~3 (195 total), stream syncs 167 -> 24, cuMemsetD8 94 -> ~3 steady state (498 total, 459 of them the zero-fill of fresh pool allocations), kernel time unchanged (5.9 ms). b16: 28 067 -> 26 754 launches, DtoH 1040 -> 121, syncs 2166 -> ~300 total.
Correctness (no training run): `DARKNET_PARITY_DUMP` of biases, BN gamma / BN bias / rolling mean / rolling variance and weights after steps 1 and 5, batch 2, same init / input / seeds (the step is deterministic: two before runs are byte-identical): before vs after files are byte-identical for f32 and for AMP, and the two yolo head outputs are identical. New tests in `oxide_batchnorm_adam.rs`: fused SGD / Adam bit-identical to the separate launches (with NaN / Inf in updates and weights, `fix_weights` on and off), device BN fold vs the host fold (1e-6). All oxide suites pass individually and in the full run, `cargo test -p darknet-core --lib`: 119 passed.
Not done: CUDA graph capture. `cuda-core 0.3.1` only exposes raw `cuStreamBeginCapture` / `cuStreamEndCapture` (no instantiate / launch wrapper), and a replayable graph needs static buffer addresses (the step allocates from the pool, downloads the yolo heads for the host loss between forward and backward, and bakes lr / Adam step into kernel arguments), so it is not a cheap change. Next launch cuts if wanted: one multi-tensor update launch per layer (-40 launches, about 0.5 ms), fold of the gradient accumulation (`saxpy`, 61 launches) into the update kernel, a fused conv bias + BN + activation epilogue. Remaining host cost at b1: 589 launches x ~16 us = 9 ms of the 12 ms step.

### 7.4 Multi-tensor update, folded gradient accumulation, fused bias + activation (2026-10-01)

Same driver as 7.1 / 7.3 (`bench-train`, LegoGears_v2, 20 steps after 3 warm-up, 3 interleaved rounds, previous binary vs new binary, oxide f32). Inference: `detect-image` (the command `tools/oxide_bench.py --suites e2e` runs), 100 iterations, 3 interleaved rounds, median of the last 90 iterations per round.
Changes (all bit-exact, only measured wins kept):
1. Multi-tensor update: `sanitized_sgd_update_multi_f32` / `sanitized_adam_update_multi_f32` (defaulted trait methods = per-tensor composition) with a oxide override `sanitized_{sgd,adam}_update3`: one launch per layer for weights + biases + BN scales (thread `i` of the concatenation picks its segment; per-segment decay / `fix_weights`; the three outputs are separate tensors, so no copies). 2-3 launches per layer become 1.
2. Folded gradient accumulation: on the deferred training path the backward no longer runs three `saxpy` per conv layer; the gradient is kept as `pending` in the layer's train buffers and the update kernel computes `u + g` itself (the same addition, so the same bits). A second backward before the update (subdivisions) first flushes the earlier pending gradient with a `saxpy`, so the order is `(u + g1) + g2` as before. `grad_centr` layers flush their weight gradient first. Eager `update_weights_gpu` and every non-deferred reader still see accumulated buffers (nothing is pending there).
3. Fused inference epilogue: `bias_activation_f32` (default = `bias_add` then `activation`; oxide kernel `bias_activation` does `activate(x + bias[c])`) used by `forward_compact` (BN is already folded into weights / bias there). Saves one launch and one activation-sized allocation per conv layer.

| step (round medians) | before | after |
|---|---|---|
| train b1 total | 11.1 (11.1 / 10.8 / 11.5) ms | 9.6 (9.4 / 9.6 / 9.9) ms (-14%) |
| train b1 update phase | 0.72 ms | 0.35 ms |
| train b16 total | 63.7 (63.4 / 64.4 / 63.7) ms | 62.4 (60.7 / 65.8 / 62.4) ms (-2%, within the noise of the rounds) |
| train b16 update phase | 0.9 ms | 0.5 ms |
| train launches / step (b1, nsys, 13 steps) | 589 | 492.7 (about 97 fewer: -61 saxpy, -36 update); kernel time 5.94 -> 5.73 ms |
| inference b1 fp32 | 2.27 (2.49 / 2.27 / 2.21) ms | 1.96 (1.96 / 2.03 / 1.91) ms (-14%) |
| inference b1 tf32 | 2.36 (2.23 / 2.36 / 2.49) ms | 1.95 (2.06 / 1.95 / 1.93) ms (-17%) |
| inference b16 fp32 | 14.76 (14.68 / 14.84 / 14.76) ms | 14.59 (14.57 / 14.71 / 14.59) ms (-1%) |
| inference b16 tf32 | 10.72 (10.65 / 11.01 / 10.72) ms | 10.51 (10.51 / 10.49 / 10.58) ms (-2%) |
| inference device memory delta b1 / b16 | 18 / 208 MB | 16 / 168 MB |

Inference b1 is launch bound, so the saved launches show (about 0.35 ms); b16 is kernel bound and moves by 1-2%. The f16 inference path already had the fused epilogue and is untouched.
Correctness (no training run): `DARKNET_PARITY_DUMP` after steps 1 and 5, batch 2, same init / input / seeds / deltas, previous binary vs new binary: biases, BN gamma / bias, rolling mean / variance, weights and both yolo head outputs are byte-identical for SGD (f32 GEMM), AMP and Adam (`adam=1`). Detections of the 80 validation images (`tools/oxide_map.py --save-detections`, fp32): mAP 99.6685%, `oxide_det_compare.py` vs `target/oxide_f16/det_fp32.tsv` 155 matched, 0 unmatched, max score diff 0, min IoU 1. New tests in `oxide_batchnorm_adam.rs`: multi-tensor SGD / Adam (1-3 tensors, with and without pending gradient, NaN / Inf inputs, coupled and decoupled decay) bit-identical to the per-tensor calls, fused bias + activation bit-identical for all nine activations. All oxide suites pass individually and in the full run; `cargo test -p darknet-core --lib`: 125 passed.
Not done: fusing BN apply + activation in the training forward (the backward needs the pre-activation tensor `biased`, so the fused kernel would have to write both outputs, which saves the read but not the launch's second store; BN statistics need their own pass anyway); the conv GEMM epilogue itself (bias + activation inside the f32 GEMM store like the f16 path) was not attempted, the standalone fused pass is already one launch. A subdivision > 1 run was verified only at the kernel level (pending gradient folded in the multi-tensor kernels, flush order by construction), not end to end. CUDA graph capture stays skipped (7.3).

### 7.5 Batched weight-gradient im2col (2026-10-02)
`conv_backward_weights` (f32) ran one im2col + one GEMM per image. It now batches the im2col across a chunk (same `COL_CAP_BYTES` cap as the forward) and passes the per-image column offset to the GEMM (`b: (bi*col_len, kg*ohw, groups)`), so im2col launches drop from `batch` to `ceil(batch/chunk)` per conv; the per-image GEMM (accumulating `beta = 1` into `dW`) is unchanged. Interleaved A/B with the release `bench-train`, 3 rounds x 40 steps + 5 warmup, medians (noisy laptop):

| batch | baseline | batched wgrad | delta |
| --- | --- | --- | --- |
| 1 | 10.19 ms | 10.05 ms | -1.4% |
| 16 | 62.88 ms | 60.29 ms | -4.1% |

p10 at b16: 54.13 -> 51.39 ms. Correctness: `oxide_conv_parity` (12), `oxide_cfg_shapes` (13, wgrad over ~1000 geometries), `oxide_amp` (11) and `oxide_runtime_chain` (3) all pass. Note the release build is not faster than the debug build for training (both ~10 ms b1 / ~61 ms b16) because the step is GPU/launch bound, not host-code bound. The remaining per-image GEMM instances (wgrad, and dgrad `W^T * dy_b`) are not batched: the GEMM instance addressing `(base, stride, modulus)` has no second level for the image x group product, so a batched dgrad/wgrad GEMM or implicit-GEMM conv is the next step.

### 7.6 Batched dgrad/wgrad GEMM, implicit-GEMM forward (2026-10-05)

Both backward convolutions now run one GEMM per chunk instead of one per image, using the existing batched GEMM kernel and the split-K reduction as a general image-sum, so no new GEMM kernel was needed for the backward.  The implicit-GEMM forward was implemented and validated but is **not** a win on this GPU, so it is opt-in only.

Changes (semantics unchanged, all suites green):
1. **dgrad**: instances are `(image, group)` image-major (`count = nb * groups`), so `b` (dy) and `c` (dcol workspace) use plain instance strides; then one col2im per image reads its `col_base = bi * col_len` block of the shared workspace.  The f32 `col2im` kernel gained the `col_base` parameter.  Was `batch` GEMMs, now `ceil(batch/chunk)`.
2. **wgrad**: one batched GEMM writes per-`(image, group)` partial `dW` into a workspace (`PARTIAL_CAP_BYTES = 64 MB`), then `gemm_splitk_reduce` (with `splits = nb`, `count = groups`) sums the images in fixed order into `dw`; across chunks `dw` accumulates with `beta = 1`.  This replaced `batch` GEMMs plus the `beta = 1` chain.
3. **implicit forward** (`DARKNET_OXIDE_CONV=implicit`, `kernels/conv_implicit.rs`, `OxideBackend::gemm_conv_implicit`): generates the im2col `B` tile from `x` on the fly, removing the column buffer and the im2col launch.  Correctness is covered by the new `conv_parity_implicit` (all 13 parity cases + finite differences).
4. **im2col** (`kernels/im2col.rs`): was one thread per column element with ~6 integer divisions each; now one block per column row `(image, c, ky, kx)` (`blocks_of(rows, ...)`) with threads sweeping the output positions, so the row decode is uniform per block and only two divisions remain per element.  nsys (13-step profile): im2col 129.8 -> 106.3 ms (~18% faster).
5. **f32 GEMM shared loads** (`kernels/gemm.rs`): the inner loop now reads 4 contiguous `k` of `A` and 4 contiguous `n` of `B` per outer step (per-thread column tile `tx*4 .. +3`), which the compiler widens to 128-bit shared loads — 8 FMA per shared load instead of 2.
6. **buffer-pool auto-size + OOM retry** (`tensor.rs`, `alloc`): see 7.7.

A/B (release `bench-train`, LegoGears_v2, 30 steps + 5 warm-up, medians; the "before" baseline is the same binary with `git checkout`-restored sources):

| variant | b1 step ms | b16 step ms | b16 peak device MB |
| --- | --- | --- | --- |
| before (per-image dgrad/wgrad GEMM) | 10.82 | 61.74 | 776 |
| + batched dgrad/wgrad | 11.01 | 54.45 (-12%) | 972 |
| + streamlined im2col | 10.63 | 52.3-53.2 | 972 |
| + vectorised GEMM shared loads | 10.63 | **48.99** (-21% vs before) | 972 |
| implicit forward (opt-in) | – | 57.4-57.7 (+5% vs materialised) | 970 |

Phase attribution shifts because the batched kernel moves the stream sync point, so read the totals.  The implicit forward is slower because the materialised `B` tile is read with 16-byte `cp.async` (`ldb = ohw`, aligned) while the implicit tile can only issue 4-byte scalar copies: the im2col source column is an arbitrary input-column offset and `cp.async` 16-byte requires 16-byte alignment.  It stays behind `DARKNET_OXIDE_CONV=implicit`, is not picked by `Auto`, and did not replace the materialised path.  `DARKNET_OXIDE_CONV=gemm` still selects the (batched) materialised GEMM path.

### 7.7 Batch scaling and the buffer-pool cap (2026-10-05)

`bench-train`, LegoGears_v2, release, medians:

| batch | total ms | fwd | bwd | ms / image | peak device MB |
| --- | --- | --- | --- | --- | --- |
| 1 | 10.63 | 4.42 | 5.37 | 10.63 | 204 |
| 16 | 48.99 | 38.95 | 9.57 | 3.06 | 972 |
| 32 | 95.77 | 78.11 | 17.56 | 2.99 | 1596 |
| 64 | 192.43 | 132.32 | 58.78 | 3.01 | 2564 |

Scales linearly (~3.0 ms/image) from batch 16 up.  The `BufPool` cap used to be a fixed 1 GiB (`DARKNET_OXIDE_POOL_MB` default 1024), chosen in the 2026-10-01 allocator fix for inference where peaks were 18/184 MB.  Once the live+free footprint exceeds the cap the pool cannot keep freed buffers, so every step re-`cuMemAlloc`/`cuMemFree`d the working set: batch 32 measured **125.05 ms** instead of 95.77 (~30% slower), batch 64 similar.  It is now auto-sized when `DARKNET_OXIDE_POOL_MB` is unset: half of the free device memory (`cuMemGetInfo`), capped at 4 GiB, with the 1 GiB fallback if the query fails.  Because a generous cache can then starve a later larger allocation, `alloc_zeroed` now flushes the pool and retries once on OOM (`BufPool::flush`); this was caught by `cfg_shortcut_shapes` and is covered by the full suite.

Correctness: `oxide_conv_parity` (13, incl. `conv_parity_implicit`), `oxide_cfg_shapes` (13), `oxide_runtime_chain` (3), `oxide_layers_extra` (17), `oxide_pool_parity` (9), `oxide_parity` (10), `oxide_gemm_mma` (5), `oxide_grad_methods` (2), `oxide_half` (13), `oxide_runtime_f16` (5), `oxide_amp` (11) and `oxide_batchnorm_adam` (13) all pass.

### 7.8 fp8 GEMM overhead (2026-10-22)

`gemm_bench_precisions` (`DARKNET_BENCH_MODES=fp32,fp8`), end to end through the GEMM seam, so fp8 includes amax, quantise and temporaries. `prec-lego` is the 21 LegoGears_v2 conv GEMMs; `prec-sq` is 512, 1024 and 2048 cubed.

| Step | `prec-lego` fp8 | `prec-lego` fp32 | `prec-sq` fp8 | `prec-sq` fp32 |
| --- | --- | --- | --- | --- |
| Before (device scales, weight cache) | 21.97 ms | 2.52 ms | | |
| v0.2.167 pooled temporaries | 3.84 ms | 1.12 ms | 4.44 ms | 4.43 ms |
| v0.2.168 scale fused into quantise | 2.95 ms | 1.02 ms | 4.68 ms | 4.60 ms |
| v0.2.169 A+B fused (block-reduced amax) | 2.23 ms | 0.98 ms | 4.16 ms | 4.21 ms |

`nsys` after v0.2.168 (2402 GEMMs): 14022 launches (238 ms host API time) and 4906 memsets (71 ms) against 166 ms of GPU kernel time, so the path was launch-bound; `sgemm_fp8_mma` was 76% of GPU time, amax and quantise about 20%. A first A+B fusion with one atomic per element on adjacent slots doubled the lego time (5.89 ms; the amax kernel averaged 140 us against 3.4 us) and was replaced by the block-reduced version. Numbers are single runs; the fp32 column moved by up to 20% between runs. Remaining ideas and status: `tasks/rust/plan_fp8.md`, "Overhead optimisation". No training run was made.

## Gradient-method shootout (2026-10-01)

Harness: `src/darknet-cuda-oxide/tests/oxide_grad_methods.rs` (`grad_method_shootout`, ignored; a tiny-shape version of the same
cases runs in the normal suite). RTX 4060 Laptop, debug-profile test binary (kernels are the same), no training runs.
Per cell: median over 5 interleaved rounds of the mean device time (CUDA events, `time_gpu_ms`, launch gaps included) of one
full gradient `dL/dx` for `L = sum(dy * f(x))`; error = max abs difference to an independent f64 host reference.
FD runs entirely on the device (perturb with `saxpy`, forward op, loss = `scale_channels` product + `bias_backward` reduction; no D2H
inside the timed region), h = 1e-2, forward difference = n+1 forwards, central = 2n forwards. "C++ hand-written" is listed only where C++
differs from the exact derivative (softmax: `dy` passed through, a copy); for activations (leaky/logistic/mish), bias, maxpool and the
unweighted shortcut the C++ backward is the same function as the analytical kernel, so (a) == (b). Memory was not measured (FD only
needs n transient loss scalars plus one perturbed input copy; analytical needs one output-sized tensor).

| op | inputs n | analytical ms (err) | C++ hand-written ms (err) | FD forward n+1 ms (err) | FD central 2n ms (err) |
|---|---|---|---|---|---|
| activation Leaky | 64 | 0.0154 (3.0e-9) | = analytical | 5.72 (3.1e-5) | 9.78 (1.8e-5) |
| activation Logistic | 64 | 0.0174 (3.6e-8) | = analytical | 5.26 (4.2e-4) | 9.83 (2.2e-5) |
| activation Mish | 64 | 0.0162 (9.3e-8) | = analytical | 5.06 (2.9e-3) | 9.92 (3.9e-5) |
| softmax (groups of 8) | 64 | 0.0362 (4.4e-8) | 0.0292 (9.5e-1, not the Jacobian) | 4.61 (6.7e-4) | 8.74 (1.6e-5) |
| bias (d/dbias) | 16 | 0.0296 (1.6e-6) | = analytical | 1.41 (3.5e-4) | 2.53 (2.1e-4) |
| maxpool 2x2/2 | 64 | 0.0145 (0) | = analytical | 4.76 (1.2e-5) | 10.27 (3.7e-6) |
| shortcut x+y | 64 | 0.0289 (0) | = analytical | 5.47 (4.4e-5) | 9.81 (2.1e-5) |
| activation Leaky | 512 | 0.0154 (4.5e-9) | = analytical | 35.40 (1.8e-4) | 71.67 (8.2e-5) |
| activation Logistic | 512 | 0.0155 (5.9e-8) | = analytical | 38.53 (5.6e-4) | 73.18 (9.4e-5) |
| activation Mish | 512 | 0.0144 (1.6e-7) | = analytical | 38.24 (3.0e-3) | 72.59 (1.3e-4) |
| softmax (groups of 8) | 512 | 0.0304 (4.6e-8) | 0.0303 (9.8e-1, not the Jacobian) | 38.85 (6.9e-4) | 77.19 (2.3e-5) |
| bias (d/dbias) | 128 | 0.0466 (2.3e-6) | = analytical | 10.04 (1.8e-3) | 19.01 (8.3e-4) |
| maxpool 2x2/2 | 512 | 0.0146 (0) | = analytical | 38.20 (9.4e-4) | 78.32 (4.3e-4) |
| shortcut x+y | 512 | 0.0280 (0) | = analytical | 37.13 (1.8e-4) | 76.73 (9.0e-5) |

Conclusion: analytical is the default for every op in the set. It is 48x to 2500x faster than the cheapest finite difference (forward, n+1) (0.014-0.047 ms,
launch bound, vs 1.4-78 ms), is exact to f32 round-off (<= 2.3e-6) while FD is at best 3.7e-6 (central) and typically 1e-5..3e-3
(forward difference is O(h) biased: 3e-3 on mish; both degrade with n and with f32 loss round-off / h), and FD cost grows linearly
with n (n+1 or 2n forwards, each ~5 launches here) while analytical stays one or two launches. The smallest FD case (bias, 16 inputs) is the closest at 48x. FD stays selectable only as a verification tool (the tests above), never as a default. The C++
softmax shortcut is a plain copy and the same speed as the exact Jacobian kernel (0.029 vs 0.036 ms at n=64, 0.030 vs 0.030 at 512),
so there is no speed reason for it; it exists only for training parity with a cost layer that already folds the Jacobian in
(`DARKNET_GRAD_MODE=cpp`, default `analytical`).

## 8. Short GPU training parity (LegoGears_v2, 2026-10-01; 20 SGD steps, no convergence run)

User-approved short run (exception to the no-training-runs rule). Same init weights (`LegoGears_augmatch_best.weights`), same 2 fixed training images (224x160 PNGs, no augmentation, 4 + 2 labels), batch 2,
subdivisions 1, constant lr 0.00261 (cfg value, burn-in and policy bypassed), momentum 0.9 / decay 0.0005 from the cfg, SGD. Data generated by `tools/oxide_parity_data.py`
(cfg copy with `batch=2`, `input.bin`, `labels.txt`, `truth.bin`, fixed random yolo deltas in [-0.05, 0.05)).
Tools: `darknet-rs bench-train` with `DARKNET_BENCH_WEIGHTS / _INPUT / _LABELS / _DELTAS`, `DARKNET_PARITY_DUMP=<prefix>`, `DARKNET_PARITY_STEPS=0,1,5,20` (dumps conv biases / BN scales / rolling stats / weights),
`oxide_bench_cpp <cfg> <weights> 0 0 0 0 parity <input> <truth> <steps> <lr> <dump prefix> <steps csv> [delta prefix]`, comparison `tools/oxide_parity_compare.py <A> <B> <step>`.
Variants: oxide f32 (`DARKNET_OXIDE_GEMM=fp32`), oxide tf32, oxide AMP (`DARKNET_OXIDE_AMP=1`), Rust CPU (5 steps only, ~26 s/step at b2), C++ CUDA (`train_network_datum_gpu`, cudnn_half=1, BN not fused = batch statistics).

### 8.1 Real loss: not comparable (finding; superseded by 8.6 after the yolo fixes)
A first run used the real yolo loss on both sides (Rust `compute_yolo_training_loss_with_state` vs C++ `forward_yolo_layer`). Step 1 on the same weights/input: Rust loss 70.78 vs C++ cost 0.191 (different
normalisation/definition, not comparable), yolo delta L2 8.41 vs 4.25 (x1.98) and the weight deltas after step 1 are uncorrelated (relL2 ~1.0). The Rust loss is a re-implementation, not a line-for-line port
(e.g. it assigns the truth per cell/anchor with its own rules, whereas C++ ciou/iou_normalizer/ignore handling, `delta_normalizer` and `*cost` differ), so identical real-loss training is NOT achievable without
porting the loss; this is a pre-existing, separate item from the oxide backend. Not debugged further (no more runs). The numbers below therefore inject IDENTICAL fixed synthetic deltas on both yolo heads (C++: yolo layers
run forward-only, `delta_gpu` overwritten after the forward, then the unmodified `backward_network_gpu`; Rust: `backward_accumulate_seeded[_gpu]` seeds), which isolates forward + backward + BN + optimizer.
Forward check at step 1 (same weights, input): yolo outputs after applying the yolo activations to the Rust logits, oxide f32 vs C++: relL2 2.0e-4 / 1.8e-4 (max abs 7.5e-4) on the two heads.

### 8.2 Parameter deltas after N steps, relative to C++ CUDA (cudnn_half=1)
relL2 = ||dA - dC++|| / ||dC++|| of the parameter change since the init, per tensor, then max / mean over the 21 convs. In brackets: max-normalised max error (max|dA-dC++| / max|dC++|).

| variant vs C++ | step | conv weights max / mean | biases (BN beta) max / mean | BN scales max / mean |
|---|---|---|---|---|
| oxide f32 | 1 | 2.6e-2 / 1.4e-2 (1.9e-1) | 2.8e-2 / 1.4e-2 | 2.1e-2 / 1.1e-2 |
| oxide f32 | 5 | 3.1e-2 / 1.8e-2 (2.3e-1) | 3.1e-2 / 2.0e-2 | 2.4e-2 / 1.5e-2 |
| oxide f32 | 20 | 1.1e-1 / 7.9e-2 (2.2e-1) | 1.5e-1 / 8.2e-2 | 1.2e-1 / 6.5e-2 |
| oxide tf32 | 1 / 5 / 20 | 3.8e-2 / 2.0e-2; 3.6e-2 / 2.2e-2; 1.2e-1 / 8.2e-2 | 5.7e-2 / 2.2e-2; 5.2e-2 / 2.6e-2; 2.1e-1 / 9.0e-2 | 3.3e-2 / 1.5e-2; 3.3e-2 / 1.9e-2; 1.2e-1 / 6.9e-2 |
| oxide AMP | 1 / 5 / 20 | 4.3e-2 / 2.6e-2; 3.3e-2 / 2.2e-2; 1.1e-1 / 7.7e-2 | 4.6e-2 / 2.7e-2; 3.9e-2 / 2.6e-2; 1.6e-1 / 7.8e-2 | 3.5e-2 / 2.0e-2; 3.0e-2 / 1.8e-2; 1.1e-1 / 6.1e-2 |
| oxide f32 vs oxide tf32 (self) | 1 / 20 | 3.6e-2 / 2.1e-2; 1.3e-1 / 8.3e-2 | | |
| oxide f32 vs oxide AMP (self) | 1 / 20 | 4.5e-2 / 2.6e-2; 1.2e-1 / 8.1e-2 | | |

Per conv layer, oxide f32 (AMP at step 1 for comparison); the error grows from the heads towards the input (typical for reduced-precision backward accumulated through depth):

| conv layer | mean abs weight delta (C++) step 1 | relL2 f32 step 1 | max-norm f32 step 1 | relL2 f32 step 20 | relL2 AMP step 1 |
|---|---|---|---|---|---|
| 0 | 3.4e-04 | 7.4e-03 | 6.5e-03 | 9.2e-02 | 1.7e-02 |
| 1 | 1.2e-04 | 1.2e-02 | 2.6e-02 | 9.0e-02 | 2.8e-02 |
| 2 | 1.0e-04 | 1.5e-02 | 1.8e-02 | 1.0e-01 | 3.3e-02 |
| 4 | 1.4e-04 | 1.8e-02 | 2.3e-02 | 1.1e-01 | 3.9e-02 |
| 5 | 1.1e-04 | 2.3e-02 | 6.7e-02 | 1.1e-01 | 4.3e-02 |
| 7 | 1.2e-04 | 1.7e-02 | 3.4e-02 | 1.1e-01 | 3.7e-02 |
| 10 | 6.4e-05 | 2.0e-02 | 2.7e-02 | 9.8e-02 | 3.5e-02 |
| 12 | 8.6e-05 | 2.2e-02 | 4.5e-02 | 1.1e-01 | 4.2e-02 |
| 13 | 7.2e-05 | 2.3e-02 | 4.8e-02 | 1.1e-01 | 3.7e-02 |
| 15 | 8.1e-05 | 2.3e-02 | 1.0e-01 | 1.0e-01 | 3.3e-02 |
| 18 | 4.4e-05 | 2.6e-02 | 1.9e-01 | 8.7e-02 | 2.5e-02 |
| 20 | 8.5e-05 | 1.2e-02 | 1.9e-02 | 7.1e-02 | 1.8e-02 |
| 21 | 6.7e-05 | 1.4e-02 | 5.6e-02 | 6.7e-02 | 1.8e-02 |
| 23 | 8.3e-05 | 1.5e-02 | 5.9e-02 | 5.6e-02 | 2.1e-02 |
| 26 | 2.5e-05 | 1.8e-02 | 1.4e-01 | 9.3e-02 | 3.9e-02 |
| 27 | 5.6e-05 | 7.2e-03 | 9.1e-03 | 7.9e-02 | 3.7e-02 |
| 28 | 2.9e-05 | 1.4e-02 | 1.4e-01 | 4.5e-02 | 2.1e-02 |
| 29 (head) | 1.9e-04 | 2.4e-04 | 3.8e-04 | 1.0e-02 | 5.0e-04 |
| 32 | 7.3e-05 | 2.9e-03 | 3.7e-03 | 6.4e-02 | 3.6e-03 |
| 35 | 5.3e-05 | 4.7e-03 | 3.1e-02 | 3.8e-02 | 6.3e-03 |
| 36 (head) | 3.4e-04 | 3.7e-04 | 3.9e-04 | 6.0e-03 | 4.9e-04 |

Reading: the per-step weight deltas agree with C++ to ~1e-2 relL2 at step 1 (heads 3e-4), not the 1e-3 hoped for. The two oxide modes and the C++ fp16 run all sit in the same ~2e-2 band
against each other (f32 vs tf32 and f32 vs AMP differ by 2-3e-2 too), so the reduced precision of the C++ cudnn_half path (a C++ fp32 run is impossible, see section 7) and BN-backward cancellation explain it; I did not
separate the two (not verified). At step 20 the error grows to ~8e-2 mean for every pair (momentum 0.9 and lr 2.6e-3 on pretrained weights amplify rounding differences, same growth between f32 and tf32/AMP), which is
chaotic divergence of the trajectory, not a systematic bias: no sign of a missing or mis-scaled gradient (a systematic error would show at step 1 and in the heads).
Per-step loss trajectory: with fixed deltas the loss is not defined; the first-order surrogate sum(delta*output) is printed by both (Rust prints it on raw logits, so only the C++ column is meaningful in absolute terms, C++: 1.956 at step 1
-> 118.2 at step 20 as the weights follow the fixed deltas). The real-loss trajectory (C++: 0.191, 0.113, 0.227, 0.085, 0.032 ... 0.025 over 20 steps) cannot be reproduced by Rust, see 8.1.

### 8.3 BN rolling statistics: do they follow? Partly, with a real mismatch (diagnosed statically, not fixed)
The oxide batch-statistics BN training forward matches C++ (same step-1 forward, same deltas). The rolling mean / variance DO follow the batch statistics, but the update is 10x too fast:
the delta of the rolling mean / variance after step 1 is exactly 10.00x the C++ one in every layer (norm ratio 9.996-10.007 / 9.999-10.002), and oxide/10 matches C++ to 2.0e-3 (mean) / 4.0e-4 (var) relL2.
Cause: `BATCH_NORM_ROLLING_KEEP = 0.9` in `src/darknet-core/src/runtime.rs` (EMA keep factor, comment says it mirrors the host updates), whereas C++ conv+BN uses 0.99 / 0.01
(`scal_ongpu(.99)` in `batchnorm_layer.cpp`, cuDNN factor `.01` in `convolutional_kernels.cu`). The Rust CPU path uses the same 0.9 (CPU/oxide rolling-delta ratio 1.0 in the first layers). Consequence: weights files saved after Rust
training carry rolling stats that track the last ~10 batches instead of ~100, which changes inference of a short-trained net (noisier) but not training. Fixed afterwards (2026-10-01): GPU path keep factor 0.99 (host/CPU path keeps 0.9, which is what C++ `forward_batchnorm_layer` on the CPU does), pin test `batch_norm_rolling_keep_matches_cpp`; see `tasks/rust/todo.md` P8 follow-up.
(The "bn_bias" dump column is irrelevant: Rust keeps the BN beta in the conv bias, the BN-struct beta is unused.)

Update (2026-10-01, static diff only): the yolo-loss mismatch of 8.1 has a concrete cause for the x1.98 delta L2: Rust multiplied the iou/ciou box deltas by `(2 - w*h)` (~1.98 for these small boxes) while C++ `delta_yolo_box` uses that scale only in the MSE branch. Fixed in `train.rs` (`apply_yolo_box_delta`) with a hand-computed test. The loss VALUE 70.78 vs 0.191 is a definition difference: Rust `sum(delta^2)` over all deltas vs C++ `iou_normalizer*mean(1-iou) + obj_normalizer*||non-box delta||^2` (not changed). Full 12-row difference table (incl. open items) in `tasks/rust/todo.md` under P8, "P8 follow-up". Not re-run, so the post-fix delta ratio (expected ~1.0) is unmeasured.

### 8.4 Rust CPU path (5 steps; ~26.5 s/step at b2: fwd 6.7 s / bwd 19.7 s)
CPU vs C++ after 1 step: weight delta relL2 mean 1.65 (max 8.1), BN scale delta mean 1.47 (so also different in the heads' neighbourhood); CPU vs oxide f32 the same (1.65 / 8.1). Expected and not a bug of the oxide backend: the Rust CPU training forward
normalises conv+BN with the ROLLING statistics (inference-style, the pretrained running stats), whereas C++ CUDA and the oxide GPU path use the batch statistics, so activations and every gradient differ from layer 1 on (CPU forward output differs from oxide already
at step 1: surrogate on logits 3.41 vs 6.15). The CPU and oxide rolling-mean deltas have ratio 1.0 in the first conv and up to 17x later (activations diverge with depth). A CPU path with batch-stat BN (to match C++) is not implemented; it was not asked for.

### 8.5 Verdict
oxide f32 / tf32 / AMP training follows C++ CUDA step by step to ~1-3e-2 relL2 on weight/bias/scale deltas (heads <1e-3), growing to ~8e-2 by step 20 through chaotic amplification; consistent with the fp16 cuDNN path of C++ and the
rounding spread between the oxide precision modes themselves. Real findings: (1) rolling-stat EMA keep 0.9 vs C++ 0.99 (x10 too fast), (2) the Rust yolo training loss is not the C++ loss (different value and delta scale), (3) CPU training uses rolling-stat BN (differs from GPU / C++ by design).
Not done: loss-trajectory parity (needs 2), >20 steps, C++ fp32 reference, batch-stat CPU path, multiple seeds/images.

### 8.6 Re-run after the yolo fixes (2026-10-01): real loss now comparable and matching

Fixes since 8.1/8.3: no `(2-w*h)` scale on iou-kind box deltas, GPU BN rolling keep 0.99, Rust loss output = C++ `cost` (`train.rs` `yolo_head_cost`: per head `iou_normalizer*mean(1-iou)` [giou kind: `1-giou`; MSE kind: `||delta||^2`]
`+ obj_normalizer*||delta without the 4 box channels||^2`, mean over yolo heads like `get_network_cost`; `YoloTrainingLoss` also carries `avg_iou`/`box_count`; deltas unchanged), `scale_x_y` now applied to x,y with `new_coords` as in C++.
Same data / commands as section 8 (data regenerated with `tools/oxide_parity_data.py`; release `darknet-rs` rebuilt; one run each: oxide f32 20 steps, Rust CPU 5 steps, C++ CUDA 20 steps; synthetic-delta oxide f32 rerun, the C++ synthetic dump is unchanged).

Real loss (printed `cost` / yolo delta L2), oxide f32 vs C++ (before the fixes: 70.78 vs 0.191 and 8.41 vs 4.25):

| step | oxide loss | C++ cost | oxide delta L2 | C++ delta L2 | | step | oxide loss | C++ cost |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.191247 | 0.191136 | 4.254964 | 4.254946 (ratio 1.000004) | | 9 | 0.2035 | 0.1980 |
| 2 | 0.112967 | 0.112957 | 4.7517 | 4.7506 | | 12 | 0.0296 | 0.0289 |
| 3 | 0.226005 | 0.226554 | 4.0950 | 4.0960 | | 16 | 0.0337 | 0.0355 |
| 4 | 0.084476 | 0.084781 | 4.3358 | 4.3360 | | 19 | 0.0317 | 0.0292 |
| 5 | 0.032052 | 0.032046 | 4.2046 | 4.2080 | | 20 | 0.024089 | 0.024884 |

Step-1 loss agrees to 6e-4 relative, delta L2 to 4e-6; over 20 steps loss within ~0-8% (mean |dev| of steps 1-20 ~3%), same shape (spikes at 3 and 9, plateau ~0.03). Rust CPU (rolling-stat BN forward, 8.4): step 1 loss 0.0790 / delta L2 4.392, steps 2-5 loss 0.156, 0.158, 0.055, 0.044 (differs: different forward).

Parameter deltas vs C++ (relL2 max / mean over the 21 convs; BN-stat columns also the norm ratio of the rolling-stat delta, oxide/C++):

| run | step | conv weights | biases | BN scales | rolling mean (ratio) | rolling var (ratio) |
|---|---|---|---|---|---|---|
| real loss, oxide f32 | 1 | 4.2e-2 / 1.5e-2 | 7.0e-2 / 1.9e-2 | 3.6e-2 / 1.3e-2 | 5.7e-3 / 2.0e-3 (1.000-1.001) | 1.1e-3 / 4.0e-4 (1.000) |
| real loss, oxide f32 | 5 | 6.7e-2 / 3.2e-2 | 7.6e-2 / 3.7e-2 | 3.9e-2 / 2.2e-2 | 5.9e-3 / 2.7e-3 (0.999-1.001) | 2.6e-3 / 1.8e-3 (1.000-1.001) |
| real loss, oxide f32 | 20 | 8.3e-1 / 2.3e-1 | 8.7e-1 / 2.5e-1 | 4.7e-1 / 1.8e-1 | 5.7e-2 / 3.2e-2 (0.990-1.002) | 2.3e-2 / 1.7e-2 (0.992-1.011) |
| synthetic deltas, oxide f32 | 1 | 2.6e-2 / 1.4e-2 | 2.8e-2 / 1.4e-2 | 2.1e-2 / 1.1e-2 | 5.7e-3 / 2.0e-3 (1.000-1.001) | 1.1e-3 / 4.0e-4 (1.000) |
| synthetic deltas, oxide f32 | 5 | 3.1e-2 / 1.8e-2 | 3.1e-2 / 2.0e-2 | 2.4e-2 / 1.5e-2 | 1.7e-2 / 6.2e-3 (0.998-1.001) | 5.6e-3 / 2.7e-3 (0.999-1.001) |
| synthetic deltas, oxide f32 | 20 | 1.1e-1 / 7.9e-2 | 1.5e-1 / 8.2e-2 | 1.2e-1 / 6.5e-2 | 6.5e-2 / 4.3e-2 (0.992-1.010) | 4.5e-2 / 2.5e-2 (0.990-1.003) |
| real loss, Rust CPU | 1 | 1.5 / 1.0 | 2.1 / 1.3 | 1.3 / 0.97 | ratio 10.0-172 | ratio 10.0-27.8 |
| real loss, Rust CPU | 5 | 3.6 / 1.2 | 3.4 / 1.7 | 1.6 / 1.1 | ratio 8.3-79 | ratio 8.0-15.2 |

Reading. (1) Rolling-stat delta ratio is 1.000 at steps 1 and 5 (was 10.00x before the keep fix, 8.3): fixed. (2) The step-1/5 weight deltas under the real loss (1.5e-2 / 3.2e-2 mean) are now as close as with injected deltas
(1.4e-2 / 1.8e-2), i.e. the previous "uncorrelated, relL2 ~1" of 8.1 was entirely the yolo delta scale/definition. (3) At step 20 the real-loss run diverges more (mean 0.23 vs 0.08 synthetic): with a real loss the deltas depend on the
weights, so the ~1e-2 fp16-vs-fp32 differences feed back (C++ uses cuDNN fp16 conv, oxide f32); same chaotic amplification as before, now also visible in the loss (steps 8-19 differ by up to 8%), not a systematic bias (step-1 loss and delta L2 agree to 1e-3 / 4e-6).
(4) Rust CPU remains different on purpose of the older design: its conv+BN training forward uses the ROLLING stats (8.4) so weights/loss differ from C++ from step 1 (loss 0.079 vs 0.191), and its rolling EMA keeps 0.9, which is what the C++ CPU path does (ratio 10x vs the C++ GPU 0.99 is therefore expected).
Update (2026-10-01, static, no run): the formerly remaining differences #3 (iou_thresh extra anchors of another head) and #4 (best-anchor search over all `num` anchors instead of the head masks) are now ported to the C++ behaviour, as are the `delta_normalizer` scaling, the bad-label/equidistant filter order (`ep_loss_threshold` from the pre-update rolling average) and the new_coords inference decode (`detect.rs`: no sigmoid on x/y/objectness/class). None is active in the LegoGears parity cfg (iou_thresh=1, partitioned masks, no new_coords, default delta_normalizer), so the step-1 numbers above are unchanged; `cargo test -p darknet-core --lib` 125 passed. Still open: adversarial branch, `rewritten_bbox`/`total_bbox`/`labels` counters (see todo table).
The residual 6e-4 on the step-1 loss is the fp16 (C++ cudnn_half) vs f32 forward. Not run: oxide AMP/tf32 real-loss, >20 steps, multiple images.

## Reproduce

```
RUSTUP_TOOLCHAIN=nightly-2026-08-28 target/debug/cargo-oxide oxide build -- --release -p darknet-cli --features oxide -j 1   # darknet-rs for e2e
python3 tools/oxide_bench.py --suites micro,conv --rounds 7 --out target/oxide_bench/micro_conv
python3 tools/oxide_bench.py --suites e2e --rounds 5 --iters 100 --out target/oxide_bench/e2e   # variants incl. oxide-fp16
# synthetic training step (section 7)
DARKNET_BACKEND=oxide DARKNET_BENCH_BATCH=16 target/release/darknet-rs bench-train ../datasets/LegoGears_v2/LegoGears_v2.cfg 20 3
DARKNET_OXIDE_AMP=1 DARKNET_BACKEND=oxide DARKNET_BENCH_BATCH=16 target/release/darknet-rs bench-train ../datasets/LegoGears_v2/LegoGears_v2.cfg 20 3   # AMP (7.2); DARKNET_BENCH_DUMP=<file> / _WEIGHTS / _LR for A/B of weight deltas
target/oxide_bench/oxide_bench_cpp_train ../datasets/LegoGears_v2/LegoGears_v2.cfg - 16 20 3 0 train   # g++ command as in tools/oxide_bench.py ensure_cpp_probe, source tools/oxide_bench_cpp.cpp
# gradient-method shootout
RUSTUP_TOOLCHAIN=nightly-2026-08-28 target/debug/cargo-oxide oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 2 --test oxide_grad_methods -- --ignored --nocapture --test-threads=1
# profiles
DARKNET_BACKEND=oxide DARKNET_BENCH_ITERS=30 nsys profile --stats=false -o nsys target/release/darknet-rs detect-image <cfg> <weights> <names> <png> 0.25 0.45 resize-exact
nsys stats -r cuda_gpu_kern_sum nsys.nsys-rep
```

## 9. P9 convergence run (LegoGears_v2, 3000 iterations, 2026-10-02)

Same cfg (batch 16, subdivisions 1, max_batches 3000, burn_in 1000, steps 2400/2700) and the same init weights (`LegoGears_augmatch_best.weights`) for all three runs, each `-clear -seed 42` (C++ `-clear`), isolated backup dirs, the same 80 train / 80 valid images, mAP at conf 0.005 / IoU 0.5 (`-map` in the Rust CLI, `detector map` for C++). Training ran with `DARKNET_BACKEND=oxide`; C++ is the CUDA build (`build/src-cli/darknet`, cuDNN half ENABLED).

| run | build | median step | last loss | final mAP@0.50 |
| --- | --- | --- | --- | --- |
| oxide f32 | debug | 129.1 ms | 0.0296 | 100.0000% |
| oxide AMP (`DARKNET_OXIDE_AMP=1`) | debug | 113.0 ms | 0.0319 | 100.0000% |
| C++ CUDA (cudnn_half=1) | build/src-cli/darknet | ~40-50 ms train + ~10 ms load | 0.019 | 100.0000% |

Acceptance ("final mAP within ~0.5 mAP") is satisfied: every run reaches 100.0000% mAP@0.50 with all five per-class APs at 100.00. Data order and augmentation RNG are not bit-identical to C++ (deterministic SplitMix64 vs C++ `std::random`; the augmentation geometry/HSV path is validated to 1 LSB against OpenCV), so the loss curves differ in detail (oxide average loss ~0.041 vs C++ ~0.021, both decreasing and ending near zero); the final mAP is identical at the resolution this 80-image set provides.

Step time: oxide debug median 129 ms (f32) / 113 ms (AMP) vs C++ ~50-60 ms including loading. The oxide figure is the debug profile (no release run in this pass) and includes the per-layer host round trips during training described in section 7. This validates the new P9 trainer plumbing end to end: `<name>_loss.csv`, `<name>_map.csv`, `chart.png`, `_last.weights` (with `-save_every`), the Adam resume sidecar `<name>_optimizer.state`, and resume from `_last.weights` (a 30-step run resumed at iteration 31 from `seen=480`).

Deferred from the "go full" scope in this pass (documented in `tasks/rust/todo.md`): mosaic/cutmix, background loader prefetch, weighted shortcut, `reverse`/`deform` conv, device connected/SAM/recurrent backward + Adam with host-fallback layers, adversarial yolo, implicit-GEMM conv, f32 GEMM epilogue.

## 10. fp8 and bf16 against fp32 training, LegoGears_v2 (2026-10-23)

`cargo xtask train_legogears [-fp8|-fp16|-bf16]`, seed 42, 3000 iterations, batch 16, release build (4 build threads), RTX 4060 Laptop, one run at a time, each run in its own backup directory (`DATA` override). Rows are parsed from each run's log and `_loss.csv` by `bench/log_train.sh` into `bench/results_train.csv`. Step time is the median of the per-iteration milliseconds, total is their sum (mAP evaluation excluded). The same seed reproduces the loss exactly and total time agrees within 0.6% between repeats.

| run | step median | total | final avg loss | mAP@0.50 (best) | F1 |
|---|---|---|---|---|---|
| fp32 | 48.9 ms | 147.2 s, 147.0 s, 146.4 s (3 runs) | 0.0597 | 99.55% (99.80%) | 0.9935 |
| fp8 (`-fp8`, v0.2.177: e4m3 forward, e5m2 gradients) | 47.2 ms | 142.9 s, 142.1 s (2 runs) | 0.0578 | 99.80% (99.85%) | 0.9902 |
| bf16 AMP (`-bf16`) | 46.0 ms | 138.6 s, 138.6 s (2 runs) | 0.0524 | 99.63% (99.63%) | 0.9935 |
| fp16 AMP (`-fp16`) | 67.5 ms | 208.8 s (1 run) | 0.0525 | 99.63% (99.63%) | 0.9902 |

fp8 is 3.0% faster than fp32 (142.5 s against 146.9 s, mean of the runs) and bf16 5.7% faster; fp16 AMP is 42% slower than fp32 (it is launch and conversion bound; bf16 shares its kernels but the step is 32% faster than fp16, because fp16 here runs with dynamic loss scaling and the overflow scan). The fp32 step moves with the laptop state (53.4 ms in an earlier session, 48.9 ms in this one), so only same-session numbers are compared. Earlier fp8 variants, same protocol: v0.2.170 57.0 ms / 171.5 s; v0.2.171 byte im2col 50.5 ms / 153.7 s (other session); v0.2.172 e4m3 gradients 47.4 ms / 142.8 s and 143.0 s, mAP 99.55%, F1 0.9902; `DARKNET_OXIDE_GEMM_BWD=fp8` before the v0.2.177 wiring fix (wgrad fed `dy` as e4m3 and the column as e5m2, no byte column) 51.0 ms / 153.1 s, mAP 99.55%, F1 0.9935. `bench-train` (batch 16, 30 steps, 5 interleaved rounds, medians): fp32 50.9 ms, fp8 49.0 ms (v0.2.172), bf16 47.1 ms.

Accuracy against `results/chart_fp8_optimize_yolo_gpu.png` (mAP 99.39%, best 99.48%, F1 0.9935, loss 0.0718) and the goal's targets (mAP >= 99.0%, F1 >= 0.99, loss within 10% of fp32): every run meets all three. fp8 has the highest mAP (99.80%) and a final loss 3% below fp32. Its F1 is 0.9902 against the chart's 0.9935: at confidence 0.25 it has TP 152, FP 3, FN 0, fp32 and bf16 have FP 2; fp16 AMP also lands on 0.9902. That is one false positive out of 155 detections on an 80-image validation set, not a measurable accuracy difference, and the recipe was not tuned to it.

The chart uses the C++ Darknet chart style (title "Darknet/YOLO v6.0.117") and says 3000 iterations in 88.2 s (about 29 ms per iteration). There is no C++ fp8 build here, so that time is not comparable to a Rust step time and the 88.2 s target is not reached (the Rust fp8 and bf16 runs take 142 s and 139 s). Remaining fp8 cost: the fp8 GEMM kernel is only about 1.3x faster than fp32 `sgemm_tiled` here, `im2col` (8 ms per step) and `col2im` are shared with fp32, and fp8 still adds a quantise pass for `dy`. fp8 on top of bf16 storage (plan_fp8 Task 6) is not done.
