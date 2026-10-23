# GPU GEMM benchmark results

Table of contents:

- [Purpose](#purpose)
- [Files](#files)
- [CSV schema](#csv-schema)
- [Adding a ref](#adding-a-ref)
  - [Step by step](#step-by-step)
- [Rules](#rules)
- [Reading the CSVs](#reading-the-csvs)
- [Open decisions](#open-decisions)

## Purpose

One place that records the GPU GEMM benchmark numbers for each git tag or branch of lightnet, so the speed of the cuda-oxide backend can be compared across versions. One row per (ref, bench case). Rows are produced only by `run_bench.sh`, which parses the bench output; numbers are never typed by hand.

## Files

- `run_bench.sh`: runs one ref's benches in a scratch worktree and appends parsed rows.
- `results_kernels.csv`: rows from `gemm_bench_low_precision_kernels` (fp32, fp8-mma, fp4-dp4a).
- `results_lego.csv`: rows from `gemm_bench_lego` and `gemm_bench_precisions` (LegoGears_v2 conv shapes).
- `results_train.csv`: one row per LegoGears_v2 training run (3000 iterations): median step ms, total seconds, final avg loss, mAP@0.50, best mAP, F1. Columns: `ref,commit,dirty,date,mode,iters,median_ms,total_s,final_avg_loss,map50,best_map50,f1` (`dirty` = 1 when tracked files were modified).
- `log_train.sh`: parses a training log and its `_loss.csv` and appends one row to `results_train.csv`. Usage: `bench/log_train.sh <mode> <train.log> <loss.csv>`.
- `failures.csv`: one row per ref and bench whose build or run failed, with the reason.
- `.work/`: scratch space (worktrees, logs). Created and removed by the script; logs are kept for debugging.
- `README.md`: this file.

## CSV schema

Header line of `results_kernels.csv` and `results_lego.csv` (exact):

```
ref,ref_type,commit,date,bench,mode,m,n,k,ms,gflops
```

| Column | Meaning |
|---|---|
| `ref` | Tag or branch name, e.g. `v0.2.166`. |
| `ref_type` | `tag` or `branch`. |
| `commit` | Full 40-character commit sha the ref points to. |
| `date` | Commit date of that sha, ISO 8601 (`%cI`). |
| `bench` | `low_precision_kernels`, `lego`, `precisions`, or `lego_total` / `precisions_total` for TOTAL rows. |
| `mode` | As printed by the bench: `fp32`, `fp8-mma`, `fp4-dp4a` (kernels) or `fp32`, `fp8`, `tf32`, ... (lego, precisions). For TOTAL rows, the mode name. |
| `m`, `n`, `k` | GEMM shape, integers. Blank on TOTAL rows (`*_total`), which sum over all shapes. |
| `ms` | Median time per GEMM in milliseconds. For TOTAL rows, the summed time. |
| `gflops` | Throughput in GF/s as printed. Blank on TOTAL rows (the bench does not print it). |

`failures.csv` header (exact):

```
ref,commit,bench,reason
```

`reason` is the first `error` or `panicked` line of the log (truncated to 200 characters, commas replaced by `;`), or `no BENCH lines parsed`.

Example format only (not a data row): tag `v0.2.166` is commit `420dfd930b02d6d010cfcdc6d4e12cbabfa2b2e4`.

## Adding a ref

Run from the repository root:

```
bench/run_bench.sh <tag-or-branch> [low_precision_kernels|lego|precisions ...]
```

With no bench names it runs `low_precision_kernels` then `lego`. Environment:

- `LIGHTNET_BENCH_MODES`: modes for lego and precisions. Default `fp32,fp8` (the script sets this if unset).
- `BENCH_SHARE_OXIDE=1`: required for now, see [Open decisions](#open-decisions). Without it the script exits with code 2 before touching anything.

### Step by step

1. Resolve the name: a tag wins over a branch with the same name. Unknown name, exit 1.
2. Resolve the commit (`<ref>^{commit}`) and its commit date.
3. Create `results_*.csv` and `failures.csv` with their headers if missing or empty. Existing rows are never truncated.
4. For each bench: if `results_*.csv` already has rows for this ref and bench, skip it (idempotent).
5. Prune stale worktree registrations, then `git worktree add --detach` a scratch worktree at `bench/.work/<ref>-<sha12>-<bench>`.
6. Link the main checkout's `third_party/cuda-oxide` into the worktree (the submodule is pinned, not per ref).
7. Run `cargo xtask oxide --test oxide_gemm_bench -- --ignored <test>` in the worktree, with `LIGHTNET_BENCH_MODES` for lego and precisions. Output goes to `bench/.work/<ref>-<bench>.log`. `cargo xtask` applies the memory cap and thread limit.
8. Parse the printed lines with awk: `KERNELBENCH m.. n.. k..: mode ms ms gf GF/s | ...` for kernels, `BENCH <label> m.. n.. k..: ...` for lego and precisions, and `BENCH <label> TOTAL: mode ms ms | ...` into `*_total` rows.
9. If the run exited non-zero or no rows parsed: append one line to `failures.csv` (skipped if the same ref and bench already failed) and continue. Otherwise append the rows to the CSV.
10. Remove the symlink, remove the worktree (`git worktree remove --force`, then `rm -rf` as fallback), prune, and remove `bench/.work` if empty. A `trap` on EXIT does the same on error or interrupt.

## Rules

- One bench at a time. Do not start a second `run_bench.sh` while one runs: the GPU and the timings would interfere.
- GPU benches must not run concurrently with training. Training changes GPU clocks and load and invalidates the numbers.
- Numbers are only comparable on the same machine, with the same GPU, driver and background load. Do not compare rows from different machines or from a loaded laptop without saying so.
- Never hand-type numbers into the CSVs. Re-run the script instead, or delete the bad rows and re-run.
- Do not commit scratch files: `bench/.work/` stays out of git.

## Reading the CSVs

- Filter by `bench` and `mode` first, then compare `gflops` for the same `m,n,k` across refs.
- `ms` is the median of several rounds (`LIGHTNET_BENCH_ROUNDS`, default 5). Use `ms` for time and `gflops` for throughput; they are not independent.
- TOTAL rows (`lego_total`, `precisions_total`) give the sum of the per-shape `ms` across the LegoGears_v2 convs. Use them for whole-network estimates, and do not add them to per-shape rows.
- Rows from `failures.csv` mean that ref has no numbers for that bench. Check the log in `.work/` if it still exists.
- Commit date (`date`) is the commit's date, not the tag's creation date.

## Open decisions

- How `third_party/cuda-oxide` reaches the scratch worktree. The script currently uses a symlink into the main checkout, behind `BENCH_SHARE_OXIDE=1`. This has not been approved. Alternatives: copy the directory, or `git submodule update --init` inside the worktree (needs the pinned commit to be fetchable).
