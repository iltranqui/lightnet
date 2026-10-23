# lightnet

Rust port of Darknet/YOLO (branch `rust`) with CPU, cuTile and cuda-oxide GPU backends. This file only routes; read the file it points to for the task at hand.

## Table of Contents

- [Where things live](#where-things-live)
- [Which file for which task](#which-file-for-which-task)
- [Commands](#commands)
- [Hard rules](#hard-rules)
- [Learned rules](#learned-rules)

## Where things live

| Zone | Path | Entry file |
|---|---|---|
| Product (Darknet port) | `src/darknet-*` | `src/<crate>/CONTEXT.md` |
| cuda-oxide (submodule) | `third_party/cuda-oxide` | `third_party/CONTEXT.md` |
| Roadmap and plans | `tasks/rust/` | `tasks/rust/CONTEXT.md` |
| Reference docs | `docs/` | `docs/CONTEXT.md` |
| Model configs | `cfg/` | `cfg/CONTEXT.md` |
| Scripts | `tools/` | `tools/CONTEXT.md` |

Whole-repo map: [CONTEXT.md](CONTEXT.md).

## Which file for which task

- Port or change a layer: `tasks/rust/AGENTS.md`, then `tasks/rust/layer-support.md` and `tasks/rust/layers.md`.
- GPU layer work: `tasks/rust/gpu_layers_plan.md`.
- Current state after a context reset: `tasks/rust/STATUS.md`.
- Touch a crate: its `CONTEXT.md` first.
- Touch cuda-oxide: read `third_party/CONTEXT.md` first; it is a pinned submodule, not editable here.

## Commands

`cargo xtask <cmd>` is the preferred way to check, build and test. It applies the memory cap and the thread limit for you, and it falls back to running uncapped, with a warning, when there is no user systemd bus. Use it instead of typing raw `cargo` or `systemd-run` commands. Run `cargo xtask help` for the list.

| Command | Alias | Runs |
|---|---|---|
| `cargo xtask check [pkg...]` | `c` | `cargo check` (default: `darknet-cli`, `darknet-core`) |
| `cargo xtask gpu-check` | `gc` | `cargo check -p darknet-cli --features oxide` |
| `cargo xtask build [pkg...]` | `b` | `cargo build` (default: `darknet-cli`) |
| `cargo xtask clippy [pkg...]` | `l` | `cargo clippy` (default: `darknet-cli`, `darknet-core`) |
| `cargo xtask test [filter...]` | `t`, `host` | host tests, same as `tools/test_host.sh` (`PKGS` overrides the packages) |
| `cargo xtask oxide [args...]` | `o` | `tools/test_oxide.sh` (GPU tests; args pass through) |
| `cargo xtask train_legogears [-fp8\|-fp16\|-bf16\|-fp8 -fp16\|-fp8 -bf16] [weights] [flags...]` | `train`, `tr` | `tools/train_legogears_oxide.sh`: release oxide build, then LegoGears_v2 training. `-fp8` sets `DARKNET_OXIDE_GEMM=fp8` and `DARKNET_OXIDE_GEMM_BWD=fp8` (e5m2 gradients), `-fp16` sets `DARKNET_OXIDE_AMP=1` (mixed precision) and adds `-dyn_scale -loss_scale 1024` unless supplied; `-bf16` sets `DARKNET_OXIDE_AMP=bf16` (no loss scaling). Plain `-fp8` and the no-flag default keep f32 storage (an inherited `DARKNET_OXIDE_AMP` is unset for the child); `-fp8 -fp16` and `-fp8 -bf16` (either order) set both groups, i.e. fp8 convolution GEMMs on 16-bit activation storage, with the `-fp16` scaler rule. Default is fp32 |
| `cargo xtask fmt` | `f` | `cargo fmt --all` |

Every command exports the detected GPU arch as the advisory `CUDA_OXIDE_DEVICE_ARCH` (unless `CUDA_OXIDE_TARGET` or `CUDA_OXIDE_DEVICE_ARCH` is already set).

Environment: `MEM` (default `3G`), `JOBS` (default 2, clamped to 4; `train_legogears` also defaults to 4, per the operator's 4-thread cap). Start training only when the user asks. Add new commands as arms in `xtask/src/main.rs` and to this table.

## Hard rules

- Max 4 build threads, wrap builds in `systemd-run --user --scope -p MemoryHigh=3G`; `cargo xtask` does both (see [Commands](#commands)).
- `src/darknet-cuda-oxide` is not a workspace member; build it only with `cargo xtask oxide` (`./tools/test_oxide.sh`).
- Commit only the files a task names, never `git add -A`.

Full rule set lives in `tasks/rust/AGENTS.md`.

<!-- claude-reflect: learned rules (saved from the band; edit freely) -->
## Learned rules
- Delete model weights and other big files from the repository when they are no longer needed.
- Do not archive or remove completed plans in tasks/rust/; keep them, since they are the project roadmap needed for development.
- Prefer `cargo xtask` subcommands over raw commands when running or testing this project; they are documented in CLAUDE.md.
- After testing a new feature and confirming it works correctly, commit the change with the feature tag, as specified in AGENTS.md and AGENTS_fp8.md.
- Limit all builds and checks to a maximum of 4 threads in this project.
<!-- end claude-reflect -->
