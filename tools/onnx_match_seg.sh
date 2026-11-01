#!/usr/bin/env bash
# Check that an exported semantic-segmentation ONNX model matches the native forward
# (tools/onnx_check_seg.py: segmentation head, CPU ONNX Runtime, resize-exact input).
#
# Usage: ./tools/onnx_match_seg.sh <bin|multi|both> [image] [flags...]
#   bin | multi   semantic_binary or semantic_multi (both = one after the other, no image)
#   image         one image; default is the first line of the dataset's test.txt
# Flags:
#   -all, --all   run every image in the dataset's test.txt (model loaded once)
#   --variants    check all 8 exports: FP32/FP16 x NCHW/NHWC x activated/raw logits
#   --fp16 --nhwc --raw   the single export to check (default FP32, NCHW, activated)
#   --cpu         force the CPU native forward (default: oxide GPU, CPU fallback)
#   --amd         use the AMD GPU (hip_tile, third_party/hip_tile) forward as the reference;
#                 HIP_BIN overrides the hip_darknet binary, built on demand
#   --no-fuse     keep batch norm unfused        --tol X   max_abs tolerance
#   -help, --help, -h   this text
# Overrides (environment): CFG, WEIGHTS, BIN, IMAGE, LIGHTNET_BACKEND=cpu|oxide|auto.
# WEIGHTS defaults to the first existing of <name>_final, <name>_last, <name>_best in backup/.
# If the binary lacks `segment-image`, the oxide release binary is rebuilt (GPU reference);
# only if that build fails is a CPU-only one built into target/cpu-release (CPU reference).
# Needs python3 with numpy, opencv-python and onnxruntime.
# This is Rust-runtime parity only, not a compiled Darknet golden.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source tools/lib_cap.sh

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}
for arg in "$@"; do
    case "$arg" in -h | -help | --help) usage; exit 0 ;; esac
done

# `both` runs the binary and the multiclass set in turn (flags only, no image).
if [[ "${1:-}" == both ]]; then
    shift
    status=0
    for set in bin multi; do
        printf "\n=== segmentation_%s ===\n" "$set" >&2
        "$0" "$set" "$@" || status=1
    done
    exit "$status"
fi

case "${1:-}" in
    bin) SET="semantic_binary" ;;
    multi) SET="semantic_multi" ;;
    *) printf 'usage: onnx_match_seg.sh <bin|multi|both> [image] [flags...]  (-help for details)\n' >&2; exit 1 ;;
esac
shift

DIR="../datasets/$SET"
BASE="semantic_segmentation"
CFG="${CFG:-$DIR/darknet/$BASE.cfg}"
BIN="${BIN:-target/release/lightnet-rs}"
if [[ -z "${WEIGHTS:-}" ]]; then
    for suffix in final last best; do
        [[ -f "$DIR/darknet/backup/${BASE}_${suffix}.weights" ]] && { WEIGHTS="$DIR/darknet/backup/${BASE}_${suffix}.weights"; break; }
    done
fi

# -all / --all: every image of the test list instead of one.
FLAGS=()
ALL=0
AMD=0
for arg in "$@"; do
    case "$arg" in
        -all | --all) ALL=1 ;;
        --amd) AMD=1 ;;
        *) FLAGS+=("$arg") ;;
    esac
done
if [[ "$ALL" == 1 ]]; then
    TARGET=(--list "$DIR/test.txt")
    CHECK_IMAGE="$DIR/test.txt"
elif [[ ${#FLAGS[@]} -gt 0 && "${FLAGS[0]}" != -* ]]; then
    TARGET=("${FLAGS[0]}")
    CHECK_IMAGE="${FLAGS[0]}"
    FLAGS=("${FLAGS[@]:1}")
else
    IMAGE="${IMAGE:-$(head -n1 "$DIR/test.txt")}"
    TARGET=("$IMAGE")
    CHECK_IMAGE="$IMAGE"
fi
CHECKS=("$CFG" "${WEIGHTS:-}" "$CHECK_IMAGE")
# AMD mode still exports the ONNX with lightnet-rs (CPU export), so the binary is required;
# use the CPU-only build when no release binary exists.
if [[ "$AMD" == 1 && ! -x "$BIN" && -x target/cpu-release/release/lightnet-rs ]]; then
    BIN="target/cpu-release/release/lightnet-rs"
fi
CHECKS+=("$BIN")
for path in "${CHECKS[@]}"; do
    if [[ -z "$path" || ! -e "$path" ]]; then
        printf 'Missing: %s\n' "${path:-<no weights found in $DIR/darknet/backup>}" >&2
        exit 1
    fi
done

# The binary must know `segment-image`. A stale one is rebuilt as the oxide release binary
# (the same build as training, so the native reference runs on the GPU). Only if that build
# fails is a CPU-only copy built into its own target dir; the reference is then CPU.
HIP_BIN="${HIP_BIN:-third_party/hip_tile/target/release/hip_darknet}"
if [[ "$AMD" == 1 ]]; then
    if [[ ! -x "$HIP_BIN" ]]; then
        printf "Building hip_darknet (release, AMD GPU)...\n" >&2
        (cd third_party/hip_tile && capped 3G cargo build --release -p hip_darknet -j 4)
    fi
    exec python3 tools/onnx_check_seg.py "$CFG" "$WEIGHTS" "${TARGET[@]}" --binary "$BIN" --amd-binary "$HIP_BIN" "${FLAGS[@]}"
fi
if ! "$BIN" help 2>&1 | grep -qF "segment-image <cfg> <weights> <image>..."; then
    printf "Rebuilding the oxide release binary (first build takes minutes)...\n" >&2
    if SEG_DATASET="$SET" BUILD_ONLY=1 ./tools/train_segmentation_oxide.sh && "$BIN" help 2>&1 | grep -qF "segment-image <cfg> <weights> <image>..."; then
        :
    else
        printf "oxide build failed; falling back to a CPU-only build (CPU reference).\n" >&2
        BIN="target/cpu-release/release/lightnet-rs"
        capped 3G cargo build --release -p lightnet-cli -j 4 --target-dir target/cpu-release
    fi
fi
exec python3 tools/onnx_check_seg.py "$CFG" "$WEIGHTS" "${TARGET[@]}" --binary "$BIN" "${FLAGS[@]}"
