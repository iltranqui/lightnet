#!/usr/bin/env bash
# Build and train LegoGears_v2 using the Rust oxide backend.
# Usage: ./tools/train_legogears_oxide.sh [initial.weights] [training flags...]
# Overrides: CFG, DATA, SEED, DARKNET_OXIDE_AMP=1
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-../datasets/LegoGears_v2/LegoGears_v2.cfg}"
DATA="${DATA:-../datasets/LegoGears_v2/LegoGears_v2.data}"
SEED="${SEED:-42}"

for path in "$CFG" "$DATA"; do
    if [[ ! -f "$path" ]]; then
        printf 'Missing training file: %s\n' "$path" >&2
        exit 1
    fi
done
if [[ ! -x target/debug/cargo-oxide ]]; then
    printf 'Building cargo-oxide helper (missing after a clean build)...\n'
    systemd-run --user --scope -p MemoryHigh=3G \
        --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
        cargo build -p cargo-oxide -j 4
fi

systemd-run --user --scope -p MemoryHigh=3G \
    --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
    target/debug/cargo-oxide oxide build -- \
    -p darknet-cli --features oxide -j 4

export DARKNET_BACKEND=oxide
exec target/debug/darknet-rs detector train \
    "$DATA" "$CFG" -seed "$SEED" -dont_show "$@"
