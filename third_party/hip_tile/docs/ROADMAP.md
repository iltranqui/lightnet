# ROADMAP

Two tracks. **Track A** trains a Darknet YOLO model from scratch on the AMD GPU and proves the result. **Track B** grows the platform under it (runtime, tile model, packaging, async) toward the layered shape of NVIDIA's `cuda-rust`, using the lessons that project's PRs and issues already paid for. Both tracks obey the same constraints: pure Rust built by plain cargo, no C/C++, no `hipcc`, kernels lowered by the LLVM backend without hand-written AMDGCN asm.

## Contents

- [Track A — training `hip_darknet` end to end](#track-a--training-hip_darknet-end-to-end)
  - [Acceptance](#acceptance)
  - [Decisions](#decisions)
  - [Stages](#stages)
  - [Status](#status)
  - [Risks](#risks)
- [Track B — platform, `cuda-rust` style](#track-b--platform-cuda-rust-style)
  - [Target layout](#target-layout)
  - [Where the platform stands](#where-the-platform-stands)
  - [Lessons from cuda-rust](#lessons-from-cuda-rust)
  - [Milestones](#milestones)
  - [Platform decisions](#platform-decisions)
  - [Out of scope](#out-of-scope)
  - [Platform risks](#platform-risks)

## Track A — training `hip_darknet` end to end

The goal: train a Darknet YOLO model **from scratch** on a labelled dataset and prove the result, entirely in Rust on the AMD GPU — no C/C++, no `hipcc`, no Darknet binary. The first target is `LegoGears_v2` (5 classes, 224×160, YOLOv4-tiny-shaped cfg).

Implementation lives in `crates/hip_darknet` (grown, not a new crate) with every new GPU primitive as a tile op in `crates/hip-tile`. Process lives in [`workflows/train-darknet-model/`](../workflows/train-darknet-model/CONTEXT.md); the current run is `workflows/train-darknet-model/runs/legogears_v2/`.

### Acceptance

Measured on the 80-image validation split of `~/src/DarkSuite/datasets/LegoGears_v2/`, compared against that dataset's own recorded training history (`LegoGears_v2_loss.csv`, `LegoGears_v2_map.csv`):

| Gate | Bar |
|---|---|
| mAP@0.5 (final) | ≥ 0.99 (reference 0.9955) |
| mAP@0.5:0.95 | recorded and stable (reference only logs 0.5) |
| Loss trajectory | same band as the reference at equal images-seen |
| Learning-rate curve | matches the reference csv by construction (hard check of the schedule port) |
| Stability | no NaN/inf in loss, weights or deltas across the run |
| Reproducibility | fixed seed, command line recorded, resume produces the same weights |
| Round trip | the produced `.weights` loads through the inference path and passes the ground-truth detection test |
| Performance | measured per `_shared/benchmarking.md`, GPU faster than the CPU reference trainer by a stated factor |

### Decisions

Recorded because they constrain every stage; change them here, not in code.

| Decision | Value | Why |
|---|---|---|
| Init | from scratch | reproduces the reference run's starting point |
| Batch | 4 | user decision; comfortable memory on the 680M iGPU at 224×160 |
| Iterations | 12000 with `steps=9600,10800`, `scales=0.2,0.1`, `burn_in=1000`, `max_batches=12000` | matches the reference's 48 000 images seen and the same LR fractions |
| Optimizer | SGD with momentum only | user decision; no Adam, no per-parameter adaptive anything |
| Precision | f32 end to end | user decision; no f64 master weights, no mixed precision, no f64 accumulation |
| Checkpoint format | Darknet `.weights`, BN unfolded on save | loadable by DarkSuite and by our own loader |
| Resume format | our own, documented | Darknet's `optimizer.state` layout is unpublished; no compatibility claimed |
| Augmentation | only what this cfg enables: crop jitter + resize, hue/saturation/exposure, and flip (off by default) | mosaic/mixup/cutmix/angle are rejected, never silently ignored |
| Ops | new primitives in `crates/hip-tile`, kernels stay thin | user decision; keeps `kernels/` declarative |
| Loss comparison | per-image loss column, never raw `loss` across batch sizes | Darknet's cost is a batch-wide sum |
| Verification | f32 CPU reference, two independent CIoU derivations, and finite differences at a stated tolerance | no f64 oracle exists under the f32 decision |

### Stages

| # | Stage | Gate before moving on | Contract |
|---|---|---|---|
| 01 | training semantics | facts cited from Darknet's C++, loss scaling pinned | [`01_train-semantics/`](../workflows/train-darknet-model/01_train-semantics/CONTEXT.md) |
| 02 | CPU backward (f32) | CIoU two-formula agreement + finite differences + analytic identities, all at stated tolerances | [`02_cpu-backward/`](../workflows/train-darknet-model/02_cpu-backward/CONTEXT.md) |
| 03 | tile ops + GPU kernels | GPU vs CPU parity per op/kernel, FD where it applies, batch-1 inference suites unchanged | [`03_gpu-backward/`](../workflows/train-darknet-model/03_gpu-backward/CONTEXT.md) |
| 04 | GPU training step | GPU gradients == CPU gradients, loss match, timing probe accepted | [`04_executor/`](../workflows/train-darknet-model/04_executor/CONTEXT.md) |
| 05 | CLI, checkpoints, metrics | weights round trip, csv schema matches the reference, resume equality | [`05_cli-checkpoints/`](../workflows/train-darknet-model/05_cli-checkpoints/CONTEXT.md) |
| 06 | the run | the acceptance table above | [`06_run/`](../workflows/train-darknet-model/06_run/CONTEXT.md) |
| 07 | bench + docs | numbers per method, every CONTEXT.md true again | [`07_bench-docs/`](../workflows/train-darknet-model/07_bench-docs/CONTEXT.md) |

Nothing moves to the next stage until a person has read the previous stage's run file.

### Status

| Stage | State | File |
|---|---|---|
| 01 training semantics | written, awaiting the human read | `runs/legogears_v2/01_train-semantics.md` |
| 02 CPU backward | **done** — 11 numeric gradient checks + 49 unit tests + learning check pass | `runs/legogears_v2/02_cpu-backward.md` |
| 03 tile ops + kernels | **done except one open case** — `store_tile_add` + 12 kernels GPU-verified (util, maxpool/upsample/route gradients, BN stats/backward, `[yolo]` delta for the CIoU path); `conv_bwd_dw/dx` verified at batch 1 and 2 (batched `dX` bug fixed); the device MSE box path has a 4-entry mismatch, unused by this cfg | `runs/legogears_v2/03_gpu-backward.md` |
| 04 GPU training step | not started — unblocked (needs batched forward + `TrainModel`) | — |
| 05 CLI, checkpoints, metrics | **done** except split-run resume (needs a long run) — the CPU-side of it | `runs/legogears_v2/05_cli-checkpoints.md` |
| 06 the run | blocked on 03/04 for the 12 000-iteration budget; dataset caveats recorded in stage 05 | — |
| 07 bench + docs | not started | — |

Working today: `hip_darknet train --data …/LegoGears_v2.data --iters N --batch N --out DIR` trains on the CPU with Darknet-faithful deltas, writes Darknet-format `.weights` and reference-compatible `loss.csv`/`map.csv`, and can resume. The CPU path is the reference implementation for the GPU trainer; at ~5.5 s/image it is not the run owner.

Two dataset facts that change stage 06 (recorded in `runs/legogears_v2/05_cli-checkpoints.md`): the train and valid lists are the same 80 images, and the only shipped weights file (`…_last.weights`) produces no detections, so there is no trustworthy checkpoint to score against.

### Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| CIoU delta in f32 | likeliest way to "train but worse" | two independent derivations must agree, FD at a stated f32 tolerance, GPU-vs-CPU exactness — all before any long run |
| f32 momentum drift over 12000 iterations | no f64 master copy | `max|w|` and a weight-norm checksum per checkpoint, so drift shows as a discontinuity rather than a mystery |
| Batch-4 batch norm statistics | noisier than the reference's batch 16 | documented as an expected shift; fallback is 2-step accumulation to effective batch 8, recorded as a decision |
| No atomics/accumulate primitive in `hip-tile` | scatter-based `dX`/`dW` would need them | gather-`dX` and one-block-per-tile `dW`; block-partial + reduction pass is the deterministic alternative if a spec needs it |
| Batched forward touching inference kernels | could regress the passing parity suites | batch-1 path kept byte-identical, suites re-run unchanged in stage 03 |
| Wall clock on 12 CUs | ~48 000 forward-equivalents plus backward | 200-iteration timing probe at stage 04 precedes the full run |
| Loss-scale confusion across batch sizes | would make stage 06 unreadable | per-image loss column decided in stage 01 |

## Track B — platform, `cuda-rust` style

`NVIDIA/cuda-rust` groups its crates by layer: driver bindings, a core runtime, an async layer, then one folder per programming model (`cuda-oxide` for SIMT, `cutile-rs` for tiles), all sharing one host runtime. hip-tile already has the same pieces in a flat `crates/` folder, minus the async layer and the SIMT model. Track B regroups what exists, then fills the gaps in the order that removes the most risk first.

Survey basis: about 300 PRs and 581 issues of `NVIDIA/cuda-rust`, read on 2026-10-28 (titles and labels for most, bodies for a few). Issue and PR numbers below refer to that repo.

### Target layout

Directories move; package names stay (`hip-sys`, `hip-rt`, `hip-tile`, …), so no `use` line changes. Folders for layers that have no code yet are not created until they do.

```
hip_tile/
├── hip-bindings/          bindings     ← crates/hip-sys
├── hip-core/              runtime      ← crates/hip-rt (device, stream, event, buffer, module, rocblas)
├── hip-core-derive/       derives      (later: only when a derive exists, e.g. KernelArg)
├── hip-async/             async        (later: milestone B6)
├── hiptile-rs/            tile model
│   ├── hip-tile/          ← crates/hip-tile
│   ├── hip-tile-macros/   ← crates/hip-tile-macros
│   ├── hip-tile-build/    ← crates/hip-tile-build   (the compiler driver)
│   └── kernels/           ← kernels
├── apps/
│   ├── runner/            ← apps/runner
│   └── hip_darknet/       ← crates/hip_darknet      (a consumer, not a layer)
└── xtask/
```

| Layer | cuda-rust | hip-tile today | Gap |
|---|---|---|---|
| bindings | `cuda-bindings` | `hip-sys` | none |
| core runtime | `cuda-core` | `hip-rt` | subviews, `mem_info`, external streams, pools, pinned mapping |
| derives | `cuda-core-derive` | `unsafe trait KernelArg` with hand impls | optional |
| async | `cuda-async` (futures, graphs) | none | whole layer |
| SIMT model | `cuda-oxide` | none | not planned (see [Out of scope](#out-of-scope)) |
| tile model | `cutile-rs` (via closed `tileiras`) | `hip-tile` + macros + build (pure Rust via LLVM) | generics, more primitives |

### Where the platform stands

Facts checked in the code on 2026-10-28, so the milestones start from the truth:

- `hip-rt` has `Device`, `Stream`, `Event`, `Module`, `Function`, `DeviceBuffer<T: Copy>`. `Function::launch` is already `unsafe`, which is what cuda-rust had to retrofit (#1522, #1524).
- `DeviceBuffer` has no borrowed subview, no `mem_info`, no way to wrap a foreign `hipStream_t`, no pool, no pinned host memory.
- Kernel arguments go through `unsafe trait KernelArg` (`crates/hip-tile/src/arg.rs`), implemented for primitives, raw pointers, `Tensor`, `TensorMut`.
- LDS is reached through raw pointers sized by `crates/hip-tile/src/config.rs` (`LDS_PAD`, `*_lds_bytes`). The block size is a fixed 256 threads; no wave-size parameter exists.
- `crates/hip-tile-build/src/lib.rs:72` finds `hip-tile` by joining `crates/hip-tile/src` onto a parent directory, and both `build.rs` files pass `"../../kernels"`. Any regrouping breaks these at kernel-build time, not at `cargo check`.

### Lessons from cuda-rust

What their trackers say, and what it means here.

| Lesson | Evidence | Consequence for hip-tile |
|---|---|---|
| Address spaces are the biggest bug source. Rust assumes one flat 64-bit pointer; shared memory is 32-bit in address space 3 | #886, #874, #885, #883, #1334, #1344, #1453, packed-aggregate series #1000–#1094, #1661 | AMD LDS is also addrspace(3) with 32-bit pointers. Wrap LDS access in a typed pointer before more ops depend on raw `*mut T` |
| Warp size is baked into collectives | checked Block/Warp collectives #1589 family, #1311, #1313 | gfx1031 is RDNA2 (wave32, wave64 also possible); CDNA is wave64. Make wave size a parameter, never a literal |
| Unchecked argument pushing is unsound | #1522, #1524 | already `unsafe` here; keep any safe launch typed from the `#[kernel]` signature |
| Safe casts need a bit-validity rule | #1503, #1506, #1513, #1516 (`AnyBitPattern`; `bool` is copyable but not castable) | add the same rule before `DeviceBuffer` gets reinterpret/cast helpers |
| Zero sizes panic deep inside | #1373 (zero launch dims), #1538 (empty stream pool) | validate at the API edge, return `Error` |
| Two host runtimes had to be merged later | #1460, #1461, #1603, #1604, #96 | one `hip-core` for every model, from the start |
| Module discovery breaks in libraries | #1284 (loader reads the process executable, misses a `.so`), #1365 (generic-only library has no modules), #1464 (prebuilt artifacts), #1285 | embedded-kernel loading must work from a library crate, not only from the binary that ran `build.rs` |
| Arch floors and target detection drift | #1266 (build defaults to sm_80, run auto-detects), #1462, #1623, #1628, #1631 | one source of truth for the target (`gfx1031` is hard-coded in two `build.rs` files today) |
| Async drop must not block | #1417 (reaper for in-flight futures), #1413 (one spin count can't fit short and long waits) | design `hip-async` drop and polling from these, not from scratch |
| Graphs need a stated contract | #107, #1429 (capture modes), #1416 (replay safety), #1071 (conditional nodes need device-runtime symbols) | capture first, conditional nodes not at all (see [Out of scope](#out-of-scope)) |
| Loop and inline attributes need real metadata | #1375 (`#[unroll(N)]` emits none, backend unrolls twice), #1494, #1472, #1552 | if `hip-tile-macros` adds unroll control, emit LLVM loop metadata and test the IR |
| Float atomics can silently become CAS loops | #1234 | AMDGPU has native float atomics; check the emitted ISA when the atomics op lands |
| Generic tile typing is hard | #1587, #1659, #1378, #1407, #1412, #1596 | element-generic kernels are a milestone of their own, with const-generic tests |
| Tile primitive gaps users ask for | #1432 (bitwise reductions), #1658 (sort/rank-by-key), #1426 (raw-offset memory ops), #1558 (atomic aggregation) | candidate ops for `workflows/add-kernel` |

### Milestones

Each milestone is a stage: its output is read by a person before the next one starts. New ops go through [`workflows/add-kernel/`](../workflows/add-kernel/CONTEXT.md).

| # | Milestone | Scope | Gate |
|---|---|---|---|
| B0 | layout | `git mv` into the [target layout](#target-layout); fix `hip-tile-build` path discovery (`lib.rs:72`) to come from Cargo, not a directory walk; one kernel-dir and target constant for both `build.rs` files; update every `CONTEXT.md`, `CLAUDE.md`, `_templates/` and TOC that names the old paths | `cargo check --workspace`, all parity suites and `hip_darknet` inference tests pass unchanged |
| B1 | address space + wave size | typed LDS pointer/region in `hip-tile` replacing raw `*mut T`; wave size as a const parameter carried by the arch config | gemm/transpose/reduction parity unchanged; benchmark per `_shared/benchmarking.md` shows no regression; LLVM IR shows addrspace(3) loads/stores |
| B2 | core memory API | borrowed `DeviceBuffer` subviews; `Device::mem_info` (`hipMemGetInfo`); `Stream::borrow_raw`; pinned host buffer; bit-validity trait for casts; zero-size checks on launch dims and buffers | unit tests per item; misuse cases return `Error`, never panic |
| B3 | typed launch | `#[kernel]` emits a typed launcher from its signature; raw `Function::launch` stays `unsafe`; audit every `KernelArg` impl against the trait's safety contract | a wrong argument type fails to compile (trybuild-style test) |
| B4 | packaging | kernel modules loadable from a library crate; a library of only generic kernels still yields modules; target detection shared by build and run | `hip_darknet` built as a dependency of a second crate loads its kernels |
| B5 | tile ops | atomics/accumulate (also closes the Track A risk), bitwise reductions, raw-offset memory ops, then sort/rank-by-key; element-generic kernels | per-op GPU vs CPU parity; native float atomics confirmed in the ISA |
| B6 | `hip-async` | futures over `Event`/`Stream`, non-blocking drop via a reaper, adaptive completion polling; then `hipGraph` stream capture with a written replay-safety contract | ordering tests across two streams; drop of an in-flight future neither blocks nor leaks; graph replay equals eager execution |

Order rationale: B0 is cheap and unblocks clean docs. B1 fixes the class of bug that cost cuda-rust the most, while few ops depend on raw LDS pointers. B2–B4 are API work with no kernel risk. B5 feeds Track A. B6 is the largest new layer and benefits from everything before it.

### Platform decisions

| Decision | Value | Why |
|---|---|---|
| Layout move | directories only, package names unchanged | zero code churn; renames can follow as their own commit if wanted |
| Empty layer folders | not created | a folder promises code; `hip-async`, `hip-core-derive` appear with their first crate |
| Toolchain | pure Rust through `rustc`'s AMDGPU LLVM backend and its linker | standing project rule; rules out `comgr` (C++) as the cuda-rust `nvJitLink` analogue |
| Host runtime | one `hip-core` shared by every kernel model | cuda-rust is paying to merge two (#1460) |
| Arguments | `unsafe` raw launch, typed safe launch only | #1522 |

### Out of scope

- **A SIMT model (`hip-oxide`).** cuda-oxide is a full MIR-to-LLVM compiler; hip-tile's tile model already covers its kernels. Revisit only if a kernel cannot be expressed as tiles.
- **Matrix-core paths (MFMA/WMMA).** The target GPU, gfx1031 (RDNA2), has no matrix cores; WMMA starts with RDNA3, MFMA is CDNA. Revisit with new hardware.
- **Graph conditional nodes and other device-runtime calls.** They need device-side runtime symbols (#1071) that a pure-Rust link does not provide.
- **Windows and aarch64 hosts.** cuda-rust spends real effort here (#139, #1605, #8, #698); this project targets Linux/WSL2 on one machine.

### Platform risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Layout move breaks kernel builds silently | path discovery is a directory walk, failing only when `build.rs` runs | B0 replaces it before moving anything; the gate runs real kernel builds |
| Typed LDS pointer costs performance | an abstraction can block LLVM from folding addresses | B1 gate is a benchmark, plus an IR check for addrspace(3) |
| Wave-size parameter spreads through every op | collectives and reductions assume one size today | introduce it in the arch config first, convert ops one at a time with parity per op |
| Async layer grows past its use | no current caller needs futures or graphs | start B6 only when `hip_darknet` training or inference has a measured overlap win |
| Survey ages | cuda-rust moves weekly (v0.5.0 in progress) | the numbers above are dated 2026-10-28; re-survey before B6 |
