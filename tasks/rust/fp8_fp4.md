# FP8 tensor-core GEMM + FP4 CUDA-core emulation in cuda-oxide (sm_89)

Approved 2026-10-02. Branch `rust`.

## Context
`bridges/darknet-cuda-oxide` has TF32 (`gemm_mma.rs`) and FP16/BF16 (`gemm_f16.rs`) mma GEMMs, but no FP8/FP4. Findings:
- `crates/cuda-device/src/generated/register_mma.rs` already exposes dense FP8 mma, "Requires PTX 8.7 and `sm_89+`", two shapes (verified 2026-10-02):
  - **k32 (use this):** `mma_m16n8k32_fp8_{f16,f32}_{e4m3,e5m2}_{e4m3,e5m2}` (lines ~1171-1283) -> `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`; A=[u32;4], B=[u32;2], C=[f32;4] (f16 acc: C=[u32;2]). Same fragment shape as the existing `mma_m16n8k32_s32_s8` in wmma.rs, so int8 layout code is the template.
  - k16: `mma_m16n8k16_fp8_*` (A=[u32;2], B=u32) - fallback only.
  - **Do not use** `mma_m16n8k32_{f16,f32}_e4m3_*` (no `fp8_` in name): those lower to `kind::f8f6f4` and need sm_120a/121a.
  - Caveat: overlay `intrinsics/overlay/register_mma_fp8.toml` says `runtime_validation = "unexecuted"` -> step 1 smoke test on the real sm_89 GPU is mandatory.
- e4m3/e5m2 <-> f16 packed converts exist in `generated/convert.rs` (cvt, saturating f16x2->e4m3x2).
- Native FP4 mma (`mma_m16n8k32_*_e2m1_*`) needs sm_120a/121a: NOT usable on sm_89. `dp4a`/`dp2a` exist in `cuda-device/src/dotprod.rs`.
- Per user memory: "FP4 on Ada" = CUDA-core emulation of Blackwell NVFP4 math (packed E2M1 + E4M3 block-16 scales, exact block dot, FP32 acc, dp4a), not FP16 WMMA quant sim. cuDNN has no FP8 dgrad/wgrad on sm89, so own kernel is the only backward route.
- C++ references (not on `rust`; read via `git show <branch>:<path>`):
  - `raptor-fp4`: `README_fp4_fp8.md`; `src-fp4sim/fp4_sim_gemm.cu` (SIMT packed NVFP4 GEMM `fp4sim_gemm_nt_packed_kernel`, `__dp4a`, line ~178-290, plus one-thread-per-output reference kernel = test oracle), `fp4_sim_kernels.cu` (packed quantize), `fp4_sim_gemm.hpp` (contract: NT, codes+scales tightly packed, block-16, scale = 0.25*dec(sA)*dec(sB), global scales gA/gB device floats); tests `src-test/test_fp4_sim_gemm.cpp`, `test_fp4_sim.cpp`.
  - `v6-dev-fp8` / `fp8`: `src-fp8/fp8_gemm.cpp` (cublasLt TN plan: Forward W:E4M3 x X:E4M3; WGrad dy:E5M2 x X:E4M3; DGrad W:E4M3 x dy:E5M2), `fp8_scaling.cpp` (delayed scaling, amax history `kFp8AmaxHistoryLength`), `fp8_kernels.cu` (quantize), tests `test_fp8_gemm.cpp`; results in `tasks/v6-dev-fp8/todo.md` (mAP 99.55% LegoGears_v2).
  - Rust intrinsic usage example: `crates/rustc-codegen-cuda/examples/sparse_mma_fp8`.
- Explore-agent claim of FP8 `m16n8k32` on sm_89 is CONFIRMED (the `mma_m16n8k32_fp8_*` family above), after an earlier wrong look at the `kind::f8f6f4` family.

## Project tree (relevant)
```
darknet/
├── crates/
│   ├── cuda-device/src/
│   │   ├── generated/register_mma.rs   # FP8 mma m16n8k32 + k16 (exists)
│   │   ├── generated/convert.rs        # e4m3/e5m2 cvt (exists)
│   │   ├── wmma.rs                     # f16/bf16/tf32/s8 mma (exists)
│   │   └── dotprod.rs                  # dp4a (exists)
│   └── darknet-core/                   # F32TensorOpsBackend trait, runtime.rs
├── bridges/darknet-cuda-oxide/
│   ├── src/
│   │   ├── lib.rs                      # backend; dispatches gemm_mma / gemm_f16 (EDIT)
│   │   ├── conv.rs, mixed.rs           # conv->GEMM callers (EDIT, later)
│   │   └── kernels/
│   │       ├── gemm_f16.rs             # template for FP8 kernel (read)
│   │       ├── convert.rs              # add f32->e4m3/e5m2 quantize (EDIT)
│   │       ├── gemm_f8.rs              # NEW
│   │       ├── gemm_fp4.rs             # NEW
│   │       ├── quant_fp4.rs            # NEW (pack + block scales)
│   │       └── mod.rs                  # register kernels (EDIT)
│   └── tests/
│       ├── oxide_gemm_f8.rs            # NEW
│       └── oxide_gemm_fp4.rs           # NEW
└── tasks/rust/fp8_fp4.md               # this plan
```

## Components
| Component | Role / responsibilities | Depends on / used by | Constraints / open Qs |
|---|---|---|---|
| `kernels/convert.rs` quantize | f32 -> e4m3/e5m2 with per-tensor scale; amax reduction kernel | cvt intrinsics; used by gemm_f8, backend | scale = amax/448 (e4m3) ; saturating; delayed vs current amax |
| `kernels/gemm_f8.rs` | row-major NN/NT/TN/TT batched GEMM, FP8 in, FP32 acc, epilogue applies sA*sB, writes f32 (or f16) | `mma_m16n8k32_fp8_f32_*`, cp.async, smem; used by backend | k32 fragments with 1-byte elems (A 4 regs, B 2 regs; same layout as the s8 k32 mma): `ldmatrix` is b16-based, may need manual smem loads or ldmatrix + repack; K tile multiple of 32; TN/TT need transpose path |
| `kernels/quant_fp4.rs` | pack E2M1 pairs/byte + E4M3 scale per 16-block along K; round-to-nearest-even; tensor-level FP32 scale | cvt e4m3; used by gemm_fp4 | match NVFP4 layout & C++ reference rounding exactly |
| `kernels/gemm_fp4.rs` | CUDA-core emulated GEMM: unpack codes -> int8 (x2, range ±12) via LUT/`prmt`, `dp4a` into i32 per 16-block, scale by sA*sB/4, FP32 acc | dp4a; used by backend `fp4_sim_gemm=simt` | SASS must show IDP.4A, no HMMA; slow-ish by design |
| Backend wiring (`lib.rs`) | new precision selector (env/cfg e.g. `DARKNET_GEMM=fp8|fp4|tf32|f16`), quantize->GEMM->dequant path, fallback to f32 | above kernels | default stays current; no API twins, extend existing dispatch |
| Tests | numeric parity vs CPU f32 ref and vs C++ FP8/FP4 output on same inputs | all | tolerance from format (e4m3 ~6% rel), exact for FP4 block dot |

## Implementation plan
Phase A – FP8 forward GEMM
1. Smoke kernel: single warp m16n8k32 e4m3 mma (and k16 for comparison), validate fragment layout against CPU (reuse `examples/sparse_mma_fp8` pattern). Test `oxide_gemm_f8.rs::fragment_layout`.
2. Quantize + amax kernels in `convert.rs`; test round-trip vs CPU e4m3 reference (incl. saturation, NaN).
3. `gemm_f8.rs` tiled kernel modelled on `gemm_f16.rs` (32x32 warp tile, cp.async 2-stage), NN first, then NT/TN/TT + batch.
4. Wire into backend behind selector; run `oxide_cfg_shapes` geometries (990 shapes) for correctness; bench vs `gemm_f16` (`oxide_gemm_bench.rs`).
5. Optional: FP8 conv via im2col path (`conv.rs`), mAP on LegoGears_v2 vs C++ FP8 (interleave A/B; ~0.5 mAP noise).

Phase B – FP8 backward
6. dgrad/wgrad as FP8 GEMMs (e5m2 for grads, e4m3 for weights/acts) using TN/NT variants; per-tensor scales; compare convergence to FP32 (static analysis/short runs only—no debug training runs per memory).

Phase C – FP4 emulation
7. `quant_fp4.rs`: port `fp4_sim_kernels.cu` packed quantize (raptor-fp4); unit test vs CPU reference and C++ codes. Port `fp4_sim_gemm_nt_packed_reference_gpu` first as the oracle kernel, then the tiled kernel.
8. `gemm_fp4.rs` dp4a kernel; exactness test (block dot bit-exact vs CPU i32 ref); check SASS (`cuobjdump -sass | grep IDP.4A`, no HMMA).
9. Selector `fp4` in backend; shape tests; document that it is emulation, not tensor-core.

Phase D – docs/bench: results table (f32/tf32/f16/fp8/fp4 GEMM TFLOPs on sm_89) into `tasks/rust/todo.md`.

## Verification
- Build: `systemd-run --user --scope -p MemoryHigh=3G` + `RUSTUP_TOOLCHAIN=nightly-2026-08-28 target/debug/cargo-oxide oxide test -- --manifest-path bridges/darknet-cuda-oxide/Cargo.toml -j 1` (one build at a time, ≤2 threads).
- Tests: `oxide_gemm_f8`, `oxide_gemm_fp4`, existing parity tests unchanged (no regression to default path).
- SASS check: HMMA.16816.F32.E4M3.E4M3 present for FP8; IDP.4A and no HMMA for FP4.
- Bench: `oxide_gemm_bench`, interleaved A/B (laptop noise).
- End-to-end: LegoGears_v2 detections/mAP vs C++ FP8 path.

## Open questions
- Expected FP8 speedup over FP16 mma on Ada is up to ~2x peak; memory-bound layers gain less.
- Quantization granularity (per-tensor vs per-row) – start per-tensor.

---

# STATUS (2026-10-04)

Phases A (steps 1-4) and C (steps 7-9) are implemented, tested and green on the
real sm_89 GPU (RTX 4060 Laptop).  Phase B (FP8 backward) and Phase D
(bench table) are **not** done.  Nothing below changes the default path:
`DARKNET_OXIDE_GEMM` still defaults to `fp32`, and the full oxide suite is
unchanged (`./tools/test_oxide.sh` -> `EXIT=0`, all suites `ok`).

## What exists

| Plan item | Where | Status |
|---|---|---|
| A1 FP8 fragment-layout smoke test | `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs` `fp8_mma_probe` (m16n8k32) and `fp8_mma_probe16` (m16n8k16), host wrappers `OxideBackend::fp8_mma_probe{,16}`, tests `oxide_gemm_f8.rs::fragment_layout_*` | done |
| A1 layout sweep over candidate wirings | `oxide_gemm_f8.rs::fragment_layout_sweep_finds_the_documented_wiring` and `::fragment_layout_sweep_k16` | done |
| A2 quantize + amax | `kernels/gemm_f8.rs` `fp8_amax_f32_kernel`, `fp8_quantize_kernel`, `fp8_dequantize_kernel`; host reference in `src/fp8.rs` | done |
| A3 tiled FP8 GEMM | `kernels/gemm_f8.rs` `sgemm_fp8_mma` (64x64 tile, BK=32, 128 threads, NN/NT/TN/TT, batched views, split-K from the host) | done |
| A4 backend selector | `GemmMode::Fp8` (`DARKNET_OXIDE_GEMM=fp8`/`e4m3`), `OxideBackend::gemm_f32_via_fp8` | done |
| C7 NVFP4 quantiser | `kernels/quant_fp4.rs` (`fp4_quantize_1x16_kernel`, `fp4_amax_f32_kernel`, `fp4_finalize_global_scale_kernel`, `fp4_dequantize_1x16_kernel`); host reference in `src/fp4.rs` | done |
| C7/C8 packed GEMM + oracle | `kernels/gemm_fp4.rs` `fp4_gemm_nt_packed` (tiled) and `fp4_gemm_nt_packed_reference` (one thread per output) | done |
| C8 exactness | `oxide_gemm_fp4.rs` - both kernels are **bit exact** against `fp4::gemm_nt_packed`, whose block dot is an i32 `dp4a` sum | done |
| C9 selector | `GemmMode::Fp4` (`DARKNET_OXIDE_GEMM=fp4`/`nvfp4`), `OxideBackend::gemm_f32_via_fp4` | done |
| B6 FP8 dgrad / wgrad | - | **not done** |
| D bench table | - | **not done** |

## Findings that correct the plan

1. **The FP8 `mma` family does work on this sm_89 part**, but the plan's
   mandatory smoke test was worth having: the *documented* layout in
   `branches/cuda-device/src/wmma.rs` (int8 `mma_m16n8k32`) has the A fragment's
   `row` selector as `group` for elements 0-4 and 8-12, `group + 8` for 4-8 and
   12-16 - **not** "elements >= 8".  Writing it as "elements >= 8" produces a
   wrong-but-plausible result rather than an obvious failure.  The sweep test
   now pins all five wiring axes (which register holds row group+8, which K
   band each register covers, and the byte order inside a register) against an
   exact host GEMM; the documented wiring is the only k32 candidate that matches
   (max relative error 0).  `m16n8k16` has additional degenerate candidates
   (the K band is only 16 wide), also fine because the documented one matches.
2. **B fragments cannot be read one aligned word at a time from a `[k][n]`
   shared tile**: an mma B register holds four *K* values of one column, which
   in the natural `[k][n]` layout sit 32 bytes apart.  The kernel therefore
   stages B transposed in shared memory (`[n][k]`, stride 40 -> 36), which
   makes the four bytes consecutive again.  Getting this wrong shows up as
   `CUDA_ERROR_MISALIGNED_ADDRESS` from `ld.shared.b32`.
3. **e4m3 / e5m2 details the plan did not pin** (all verified against the
   hardware by `quantize_matches_the_host_reference_over_a_dense_sweep`):
   * e4m3's exponent bias is **7**, not `2^(bits-1)-1`, so its smallest subnormal
     is `2^-9` and the subnormal step is `2^(1-bias-mantissa_bits)`.
   * e5m2 has a **5-bit** exponent field (bias 15); its largest finite code is
     `0x7b` (57344), `0x7c` is infinity and `0x7d..0x7f` are NaN.  e4m3 spends
     the whole top binade on finite values: `0x7e` is 448 and only `0x7f` is NaN.
   * `cvt.rn.satfinite.<fmt>x2.f32` maps **every** NaN (signed or not) to the
     *unsigned* NaN code; the sign is dropped.
   * Round-to-nearest ties go to the even *code*, and saturation has to be
     applied before the search (`max - x` is flat in f32 far above the grid, so
     the tie rule would otherwise pick a sub-maximal code).
4. **Tensor-core accumulation order** puts a floor of about `5e-5` (relative to
   the row norm `sum_k |a_k b_k|`) on any comparison with an f64 CPU reference,
   far above `2^-24`.  All FP8 tests compare against that row norm, not against
   `|sum|`, which cancels.
5. **`__dp4a` couples 4 bytes of a word in lane order 0..3**, exactly as
   `fp4simgemm_decode_row_words` in the C++ assumes; the device matches the C++
   bit for bit.  The *host* reference in `src/fp4.rs` initially had an `as i32`
   precedence bug that silently dropped every +-0.5 code - the C++ LUT does not
   have it, which is why the device and the (buggy) host disagreed only on
   inputs containing 0.5.  `fp4::tests::block_dot_includes_the_half_codes` pins it.

## How to run

```
./tools/test_oxide.sh --test oxide_gemm_f8     # 19 tests
./tools/test_oxide.sh --test oxide_gemm_fp4    # 10 tests
cargo test --manifest-path src/darknet-cuda-oxide/Cargo.toml --lib   # 14 host reference tests
```
