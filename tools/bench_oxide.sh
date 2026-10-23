#!/usr/bin/env bash
# Kernel benchmarks (ignored tests in src/lightnet-cuda-oxide/tests/oxide_*bench.rs).
# Usage: ./tools/bench_oxide.sh <micro|conv|gemm-square|gemm-lego|all>
# Env passed through: LIGHTNET_BENCH_MODES, LIGHTNET_BENCH_FILTER, LIGHTNET_BENCH_ROUNDS,
#   LIGHTNET_BENCH_CONV_BATCH, LIGHTNET_BENCH_TARGET_MS (see tests/oxide_bench.rs header).
# Modes are interleaved per round (laptop clocks drift); results are BENCHJSON lines,
# also saved to target/bench/<kind>_<timestamp>.log.  Override: MEM (default 3G).
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

MEM="${MEM:-3G}"
KIND="${1:-all}"
export RUSTUP_TOOLCHAIN=nightly-2026-08-28

case "$KIND" in
    micro)       RUNS=("oxide_bench bench_micro") ;;
    conv)        RUNS=("oxide_bench bench_conv") ;;
    gemm-square) RUNS=("oxide_gemm_bench gemm_bench_square") ;;
    gemm-lego)   RUNS=("oxide_gemm_bench gemm_bench_lego") ;;
    all)         RUNS=("oxide_bench bench_micro" "oxide_bench bench_conv"
                       "oxide_gemm_bench gemm_bench_square" "oxide_gemm_bench gemm_bench_lego") ;;
    *)
        printf 'Usage: %s <micro|conv|gemm-square|gemm-lego|all>\n' "$0" >&2
        exit 1
        ;;
esac

source tools/lib_cap.sh  # cwd is the repo root here
oxide_arch_hint

if [[ ! -x target/debug/cargo-oxide ]]; then
    capped "$MEM" cargo build -p cargo-oxide -j 2
fi

mkdir -p target/bench
LOG="target/bench/${KIND}_$(date +%Y%m%d_%H%M%S).log"
for run in "${RUNS[@]}"; do
    read -r file name <<<"$run"
    capped "$MEM" \
        target/debug/cargo-oxide oxide test -- \
        --manifest-path src/lightnet-cuda-oxide/Cargo.toml -j 2 --release --test "$file" \
        -- --ignored --nocapture --test-threads=1 "$name" 2>&1 | tee -a "$LOG"
done
printf 'Saved %s\n' "$LOG" >&2
