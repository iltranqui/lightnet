#!/usr/bin/env bash
# Build and train LegoGears_v2 using the Rust oxide backend.
# Usage: ./tools/train_legogears_oxide.sh [initial.weights] [training flags...]
# Overrides: CFG, DATA, SEED, DARKNET_OXIDE_AMP=1, PROFILE=release|debug
#
# PROFILE defaults to `release`: the trainer's host code (image loader, phase
# bookkeeping, launch preparation) is a large part of the measured step time,
# and a debug host build inflates it several-fold.  Set PROFILE=debug when you
# need a debuggable binary (CUDA-GDB, backtraces).  cuda-oxide compiles the
# device code release-like either way.
#
# -map evaluates every valid= image once using the current training weights.
# Optional: -map_every 100 -thresh 0.005 -iou_thresh 0.5
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-../datasets/LegoGears_v2/LegoGears_v2.cfg}"
DATA="${DATA:-../datasets/LegoGears_v2/LegoGears_v2.data}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-release}"

case "$PROFILE" in
    release)
        CARGO_PROFILE_FLAG=(--release)
        BIN_DIR="target/release"
        ;;
    debug)
        CARGO_PROFILE_FLAG=()
        BIN_DIR="target/debug"
        ;;
    *)
        printf 'Invalid PROFILE=%s (expected release or debug)\n' "$PROFILE" >&2
        exit 1
        ;;
esac

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

# Everything after `--` is forwarded to cargo, so `--release` selects the host
# profile.  Without it the host (loader + phase bookkeeping) is an unoptimized
# debug build and the profile numbers are not comparable to C++.
systemd-run --user --scope -p MemoryHigh=3G \
    --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
    target/debug/cargo-oxide oxide build -- \
    -p darknet-cli --features oxide -j 4 "${CARGO_PROFILE_FLAG[@]}"

printf 'Running %s/%s (PROFILE=%s)\n' "$BIN_DIR" "darknet-rs" "$PROFILE" >&2

export DARKNET_BACKEND=oxide
exec "$BIN_DIR/darknet-rs" detector train \
    "$DATA" "$CFG" -seed "$SEED" -dont_show -map "$@"
