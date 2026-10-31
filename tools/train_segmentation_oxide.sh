#!/usr/bin/env bash
# Build and train a semantic-segmentation dataset (SEG_DATASET=semantic_binary|semantic_multi) on the Rust oxide backend.
# Usage: SEG_DATASET=semantic_binary ./tools/train_segmentation_oxide.sh [initial.weights] [training flags...]
# Overrides: CFG, DATA, SEED, JOBS (build threads, default 4), LIGHTNET_OXIDE_AMP=1, PROFILE=release|debug
#
# PROFILE defaults to `release`: the trainer's host code (image loader, phase
# bookkeeping, launch preparation) is a large part of the measured step time,
# and a debug host build inflates it several-fold.  Set PROFILE=debug when you
# need a debuggable binary (CUDA-GDB, backtraces).  cuda-oxide compiles the
# device code release-like either way.
#
# No -map: it is a detection metric.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source tools/lib_cap.sh  # cwd is the repo root here
oxide_arch_hint
SEG_DATASET="${SEG_DATASET:?set SEG_DATASET=semantic_binary or semantic_multi}"
CFG="${CFG:-../datasets/${SEG_DATASET}/darknet/semantic_segmentation.cfg}"
DATA="${DATA:-../datasets/${SEG_DATASET}/darknet/semantic_segmentation.data}"
SEED="${SEED:-42}"
JOBS="${JOBS:-4}"
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
    capped 3G env RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
        cargo build -p cargo-oxide -j 4
fi

# Everything after `--` is forwarded to cargo, so `--release` selects the host
# profile.  Without it the host (loader + phase bookkeeping) is an unoptimized
# debug build and the profile numbers are not comparable to C++.
capped 3G env RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
    target/debug/cargo-oxide oxide build -- \
    -p lightnet-cli --features oxide -j "$JOBS" "${CARGO_PROFILE_FLAG[@]}"

printf 'Running %s/%s (PROFILE=%s)\n' "$BIN_DIR" "lightnet-rs" "$PROFILE" >&2

export LIGHTNET_BACKEND=oxide
exec "$BIN_DIR/lightnet-rs" detector train \
    "$DATA" "$CFG" -seed "$SEED" -dont_show "$@"
