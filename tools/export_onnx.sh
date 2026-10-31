#!/usr/bin/env bash
# Export a trained model to ONNX with `lightnet-rs export-onnx`.
# Usage: ./tools/export_onnx.sh <legogears|segmentation_bin|segmentation_multi> [export flags...]
# Flags pass to export-onnx: -fp16|-fp32, -nhwc|-nchw, -decode|-raw (detection),
# -activation|-noactivation (segmentation), --no-fuse.
# Overrides: CFG, WEIGHTS, OUT, BIN (default target/release/lightnet-rs).
# WEIGHTS defaults to the first existing of <name>_final, <name>_last, <name>_best.
# Runs on the CPU; the binary is built here (release, 4 jobs) only if it is missing.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/lib_cap.sh

TARGET="${1:?usage: export_onnx.sh <legogears|segmentation_bin|segmentation_multi> [flags...]}"
shift
DATASETS="../datasets"
case "$TARGET" in
    legogears)
        DIR="$DATASETS/LegoGears_v2"
        BASE="LegoGears_v2"
        DEFAULT_CFG="$DIR/$BASE.cfg"
        ;;
    segmentation_bin | segmentation_multi)
        case "$TARGET" in
            segmentation_bin) SET="semantic_binary" ;;
            *) SET="semantic_multi" ;;
        esac
        BASE="semantic_segmentation"
        DEFAULT_CFG="$DATASETS/$SET/darknet/$BASE.cfg"
        DIR="$DATASETS/$SET/darknet/backup"
        ;;
    *)
        printf 'Unknown export target: %s\n' "$TARGET" >&2
        exit 1
        ;;
esac

CFG="${CFG:-$DEFAULT_CFG}"
if [[ -z "${WEIGHTS:-}" ]]; then
    for suffix in final last best; do
        if [[ -f "$DIR/${BASE}_${suffix}.weights" ]]; then
            WEIGHTS="$DIR/${BASE}_${suffix}.weights"
            break
        fi
    done
fi
OUT="${OUT:-$DIR/$BASE.onnx}"
BIN="${BIN:-target/release/lightnet-rs}"

for path in "$CFG" "${WEIGHTS:-}"; do
    if [[ -z "$path" || ! -f "$path" ]]; then
        printf 'Missing file: %s (set CFG / WEIGHTS)\n' "${path:-<no weights found in $DIR>}" >&2
        exit 1
    fi
done
if [[ ! -x "$BIN" ]]; then
    printf 'Building %s (release, 4 jobs)...\n' "$BIN" >&2
    capped 3G cargo build --release -p lightnet-cli -j 4
fi

mkdir -p "$(dirname "$OUT")"
printf 'export-onnx %s %s -> %s\n' "$CFG" "$WEIGHTS" "$OUT" >&2
exec "$BIN" export-onnx "$CFG" "$WEIGHTS" "$OUT" "$@"
