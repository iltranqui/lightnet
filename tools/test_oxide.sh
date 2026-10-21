#!/usr/bin/env bash
# cuda-oxide GPU kernel + bridge tests (needs the GPU and the cargo-oxide codegen build).
# Usage: ./tools/test_oxide.sh [--test <file>] [-- test-name-filters...]
#   ./tools/test_oxide.sh                          all bridge tests (slow)
#   ./tools/test_oxide.sh --test oxide_parity      one test file
#   ./tools/test_oxide.sh --test oxide_amp -- amp_adam
# Override: MEM (default 3G). Tests run with --test-threads=1.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

source tools/lib_cap.sh  # cwd is the repo root here
oxide_arch_hint
MEM="${MEM:-3G}"
export RUSTUP_TOOLCHAIN=nightly-2026-08-28

if [[ ! -x target/debug/cargo-oxide ]]; then
    printf 'Building cargo-oxide helper (missing after a clean build)...\n'
    capped "$MEM" cargo build -p cargo-oxide -j 2
fi

CARGO_ARGS=()
TEST_ARGS=()
while (($#)); do
    if [[ "$1" == "--" ]]; then
        shift
        TEST_ARGS=("$@")
        break
    fi
    CARGO_ARGS+=("$1")
    shift
done

capped "$MEM" \
    target/debug/cargo-oxide oxide test -- \
    --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 2 "${CARGO_ARGS[@]}" \
    -- --test-threads=1 --nocapture "${TEST_ARGS[@]}"
