#!/usr/bin/env python3
"""Compare an exported segmentation ONNX model with the Rust runtime.

Usage: tools/onnx_check_seg.py CFG WEIGHTS (IMAGE | --list FILE)
       [--binary target/release/lightnet-rs] [--cpu] [--variants]
       [--fp16] [--nhwc] [--raw] [--no-fuse] [--tol 1e-4] [-help]

Exports the model, runs it in ONNX Runtime (CPU), runs `segment-image` once for
the native forward (GPU oxide when the binary has it and a device is available,
else CPU; `--cpu` forces CPU), and compares the segmentation head (the last
layer): max error on the values and agreement of the per-pixel class.
Preprocessing is resize-exact, as in the Rust inference path.

  IMAGE | --list FILE   one image, or every image listed in FILE (one path per line).
                        With a list the native model is loaded once and each export is
                        run on every image; the table shows the worst case over images.
  --variants            check every export: FP32/FP16 x NCHW/NHWC x activated/raw
                        (8 variants) instead of the one chosen by --fp16/--nhwc/--raw.
  --fp16 --nhwc --raw   the single variant to check (default FP32, NCHW, activated).
                        Raw variants export logits (-noactivation); only the class
                        decision is compared (threshold 0 for one class, else argmax).
  --tol X               max_abs tolerance on activated values (default 1e-4, 2e-2 for FP16).
  --cpu                 force the CPU native forward.   --no-fuse   keep batch norm unfused.

Exit status: 0 all variants pass, 1 a variant failed, 2 a tool error.
This verifies Rust-runtime parity only; it is not a compiled-C++ golden.
"""

import argparse
import itertools
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def net_size(cfg: Path) -> tuple[int, int]:
    text = cfg.read_text()
    width = re.search(r"^\s*width\s*=\s*(\d+)", text, flags=re.M)
    height = re.search(r"^\s*height\s*=\s*(\d+)", text, flags=re.M)
    if width is None or height is None:
        raise ValueError(f"{cfg}: missing net width/height")
    return int(width.group(1)), int(height.group(1))


def layer_count(cfg: Path) -> int:
    sections = re.findall(r"^\s*\[([^\]]+)\]", cfg.read_text(), flags=re.M)
    return len([s for s in sections if s.strip().lower() != "net"])


def classes(values: np.ndarray, logits: bool = False) -> np.ndarray:
    """Per-pixel class of a [1, C, H, W] tensor: threshold for C == 1, else argmax.

    The binary threshold is 0.5 on probabilities and 0 on logits (sigmoid(0) = 0.5).
    """
    if values.shape[1] == 1:
        return (values[:, 0] > (0.0 if logits else 0.5)).astype(np.int64)
    return values.argmax(axis=1)


def load_images(path_list: Path) -> list[Path]:
    base = path_list.parent
    images = []
    for line in path_list.read_text().splitlines():
        line = line.strip()
        if line:
            path = Path(line)
            images.append(path if path.is_absolute() else base / path)
    if not images:
        raise ValueError(f"{path_list}: no images listed")
    return images


def preprocess(path: Path, width: int, height: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
    return resized.astype(np.float32) / 255.0  # HWC


def check_variant(args, tmp: Path, images: np.ndarray, references: list[np.ndarray],
                  fp16: bool, nhwc: bool, raw: bool):
    """Export one variant and compare it with the native head on every image.

    Returns (worst class agreement, worst max_abs or None, mean class agreement,
    failing image count, ok).
    """
    onnx_path = tmp / f"model_{'fp16' if fp16 else 'fp32'}_{'nhwc' if nhwc else 'nchw'}_{'raw' if raw else 'act'}.onnx"
    export = [args.binary, "export-onnx", str(args.cfg), str(args.weights), str(onnx_path),
              "-fp16" if fp16 else "-fp32", "-nhwc" if nhwc else "-nchw"]
    if raw:
        export.append("-noactivation")
    if args.no_fuse:
        export.append("--no-fuse")
    subprocess.run(export, check=True, stdout=subprocess.DEVNULL)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tol = args.tol if args.tol is not None else (2e-2 if fp16 else 1e-4)

    agreements, errors, failing = [], [], 0
    for index, (image, reference) in enumerate(zip(images, references)):
        tensor = image[None] if nhwc else image.transpose(2, 0, 1)[None]
        if fp16:
            tensor = tensor.astype(np.float16)
        outputs = session.run(None, {input_name: tensor})
        if len(outputs) != 1:
            raise ValueError(f"expected one segmentation output, model has {len(outputs)}")
        actual = outputs[0].astype(np.float32)
        if nhwc:
            actual = actual.transpose(0, 3, 1, 2)
        if reference.size != actual.size:
            raise ValueError(f"ONNX output has {actual.size} values, native head has {reference.size}")
        expected = reference.reshape(actual.shape)
        agree = float(np.mean(classes(actual, logits=raw) == classes(expected)))
        agreements.append(agree)
        passed = agree >= 0.999
        if not raw:
            error = float(np.max(np.abs(actual - expected)))
            errors.append(error)
            passed = passed and error <= tol
        failing += 0 if passed else 1
        if args.list is not None and (index + 1) % 10 == 0:
            print(f"    ... {index + 1}/{len(images)} images", flush=True)
    onnx_path.unlink()
    return (min(agreements), max(errors) if errors else None,
            float(np.mean(agreements)), failing, failing == 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        add_help=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-h", "-help", "--help", action="help", help="show this help and exit")
    parser.add_argument("cfg", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--list", type=Path, default=None)
    parser.add_argument("--binary", default="target/release/lightnet-rs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--nhwc", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--no-fuse", action="store_true")
    parser.add_argument("--tol", type=float, default=None)
    args = parser.parse_args()
    if (args.image is None) == (args.list is None):
        parser.error("give exactly one of IMAGE or --list FILE")
    paths = load_images(args.list) if args.list is not None else [args.image]

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        dump = tmp / "dump"
        dump.mkdir()
        env = dict(os.environ, LIGHTNET_LAYER_DUMP_DIR=str(dump))
        if args.cpu:
            env["LIGHTNET_BACKEND"] = "cpu"
        native = subprocess.run(
            [args.binary, "segment-image", str(args.cfg), str(args.weights), *map(str, paths)],
            check=True, env=env, capture_output=True, text=True,
        )
        # "segment-image backend oxide|cpu: ..." tells which forward is the reference.
        print("native reference:", native.stdout.strip().splitlines()[-1] if native.stdout.strip() else "?")

        head = layer_count(args.cfg) - 1
        references = []
        for index in range(len(paths)):
            where = dump if len(paths) == 1 else dump / f"{index:05}"
            reference_path = where / f"layer_{head:03}.f32"
            if not reference_path.is_file():
                raise FileNotFoundError(f"Rust layer dump missing: {reference_path}")
            references.append(np.fromfile(reference_path, dtype=np.float32))

        width, height = net_size(args.cfg)
        images = np.stack([preprocess(path, width, height) for path in paths])

        if args.variants:
            variants = list(itertools.product((False, True), (False, True), (False, True)))
        else:
            variants = [(args.fp16, args.nhwc, args.raw)]

        rows, failed = [], False
        for fp16, nhwc, raw in variants:
            worst, error, mean, failing, ok = check_variant(args, tmp, images, references, fp16, nhwc, raw)
            failed |= not ok
            rows.append((fp16, nhwc, raw, worst, error, mean, failing, ok))

        print(f"\nsegmentation ONNX vs native, layer {head}, {len(paths)} image(s):")
        print("  precision layout output    worst agree  mean agree  worst max_abs  failing  result")
        print("  --------- ------ --------- ----------- ----------- ------------- -------- -------")
        for fp16, nhwc, raw, worst, error, mean, failing, ok in rows:
            print(
                f"  {'fp16' if fp16 else 'fp32':<9} {'NHWC' if nhwc else 'NCHW':<6} "
                f"{'logits' if raw else 'activated':<9} {worst * 100:>10.4f}% {mean * 100:>10.4f}% "
                f"{'n/a' if error is None else f'{error:.3e}':>13} {failing:>5}/{len(paths):<2} {'OK' if ok else 'FAIL':>7}"
            )
        print("ALL OK" if not failed else "FAILED")
        return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"onnx_check_seg: {error}", file=sys.stderr)
        sys.exit(2)
