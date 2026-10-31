#!/usr/bin/env bash
# Check that the exported ONNX heads of the LegoGears_v2 detector match the Rust
# runtime layer dumps (tools/onnx_check.py; raw YOLO heads, CPU ONNX Runtime).
# Usage: ./tools/onnx_match.sh [image] [onnx_check.py flags...]
# Overrides: CFG, WEIGHTS, NAMES, BIN. IMAGE defaults to the first line of the valid list.
# Needs python3 with numpy, opencv-python and onnxruntime.
# This is Rust-runtime parity only, not a compiled Darknet golden.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DIR="../datasets/LegoGears_v2"
CFG="${CFG:-$DIR/LegoGears_v2.cfg}"
NAMES="${NAMES:-$DIR/LegoGears.names}"
BIN="${BIN:-target/release/lightnet-rs}"
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
exec python3 tools/onnx_check.py "$CFG" "$WEIGHTS" "$NAMES" "$IMAGE" --binary "$BIN" "$@"
