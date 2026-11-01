#!/usr/bin/env bash
# Check that the exported ONNX heads of the LegoGears_v2 detector match the Rust
# runtime layer dumps (tools/onnx_check.py; raw YOLO heads, CPU ONNX Runtime).
# Usage: ./tools/onnx_match.sh [image] [onnx_check.py flags...]
#   --amd   use the AMD GPU (hip_tile, third_party/hip_tile) forward as the reference;
#           HIP_BIN overrides the hip_darknet binary, built on demand. The ONNX export still
#           comes from lightnet-rs (the CPU-only build is used when no release binary exists).
# Overrides: CFG, WEIGHTS, NAMES, BIN. IMAGE defaults to the first line of the valid list.
# Needs python3 with numpy, opencv-python and onnxruntime.
# This is Rust-runtime parity only, not a compiled Darknet golden.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/lib_cap.sh

DIR="../datasets/LegoGears_v2"
CFG="${CFG:-$DIR/LegoGears_v2.cfg}"
NAMES="${NAMES:-$DIR/LegoGears.names}"
BIN="${BIN:-target/release/lightnet-rs}"
AMD=0
ARGS=()
for a in "$@"; do
    if [[ "$a" == --amd ]]; then AMD=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]}"
if [[ "$AMD" == 1 && ! -x "$BIN" && -x target/cpu-release/release/lightnet-rs ]]; then
    BIN="target/cpu-release/release/lightnet-rs"
fi
if [[ -z "${WEIGHTS:-}" ]]; then
    for suffix in final last best; do
        [[ -f "$DIR/LegoGears_v2_${suffix}.weights" ]] && { WEIGHTS="$DIR/LegoGears_v2_${suffix}.weights"; break; }
    done
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
    IMAGE="$1"
    shift
else
    IMAGE="${IMAGE:-$(head -n1 "$DIR/LegoGears_v2_valid.txt")}"
fi
for path in "$CFG" "${WEIGHTS:-}" "$NAMES" "$IMAGE" "$BIN"; do
    if [[ -z "$path" || ! -e "$path" ]]; then
        printf 'Missing: %s\n' "${path:-<no weights found in $DIR>}" >&2
        exit 1
    fi
done
if [[ "$AMD" == 1 ]]; then
    HIP_BIN="${HIP_BIN:-third_party/hip_tile/target/release/hip_darknet}"
    if [[ ! -x "$HIP_BIN" ]]; then
        printf "Building hip_darknet (release, AMD GPU)...\n" >&2
        (cd third_party/hip_tile && capped 3G cargo build --release -p hip_darknet -j 4)
    fi
    exec python3 tools/onnx_check.py "$CFG" "$WEIGHTS" "$NAMES" "$IMAGE" --binary "$BIN" --amd-binary "$HIP_BIN" "$@"
fi
exec python3 tools/onnx_check.py "$CFG" "$WEIGHTS" "$NAMES" "$IMAGE" --binary "$BIN" "$@"
