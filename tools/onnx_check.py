#!/usr/bin/env python3
"""Compare exported ONNX raw YOLO heads with the Rust runtime's layer dumps.

Usage: tools/onnx_check.py CFG WEIGHTS NAMES IMAGE
       [--binary target/release/darknet-rs] [--no-fuse] [--tol 1e-4]

Preprocessing is resize-exact, matching the Rust CPU inference input path.
This verifies Rust-runtime parity only; it is not a compiled-C++ golden.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def yolo_layer_indices(cfg: Path) -> list[int]:
    sections = re.findall(r"^\s*\[([^\]]+)\]", cfg.read_text(), flags=re.M)
    layers = [section.strip().lower() for section in sections if section.strip().lower() != "net"]
    return [index for index, section in enumerate(layers) if section == "yolo"]


def net_size(cfg: Path) -> tuple[int, int]:
    text = cfg.read_text()
    width = re.search(r"^\s*width\s*=\s*(\d+)", text, flags=re.M)
    height = re.search(r"^\s*height\s*=\s*(\d+)", text, flags=re.M)
    if width is None or height is None:
        raise ValueError(f"{cfg}: missing net width/height")
    return int(width.group(1)), int(height.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("names", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--binary", default="target/release/darknet-rs")
    parser.add_argument("--no-fuse", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        dump = Path(tmp) / "dump"
        dump.mkdir()
        cmd = [args.binary, "export-onnx", str(args.cfg), str(args.weights), str(onnx_path)]
        if args.no_fuse:
            cmd.append("--no-fuse")
        subprocess.run(cmd, check=True)
        env = dict(os.environ, DARKNET_LAYER_DUMP_DIR=str(dump))
        subprocess.run(
            [
                args.binary,
                "detect-image",
                str(args.cfg),
                str(args.weights),
                str(args.names),
                str(args.image),
                "0.25",
                "0.45",
                "resize-exact",
            ],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
        )

        width, height = net_size(args.cfg)
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read image: {args.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        output_names = [output.name for output in session.get_outputs()]
        outputs = session.run(None, {"input": tensor})
        indices = yolo_layer_indices(args.cfg)
        if len(indices) != len(output_names):
            print(f"expected {len(indices)} yolo outputs, model has {output_names}", file=sys.stderr)
            return 1
        worst = 0.0
        for name, layer_index, actual in zip(output_names, indices, outputs):
            reference_path = dump / f"layer_{layer_index:03}.f32"
            if not reference_path.is_file():
                raise FileNotFoundError(f"Rust layer dump missing: {reference_path}")
            reference = np.fromfile(reference_path, dtype=np.float32)
            if reference.size != actual.size:
                raise ValueError(
                    f"{name} (layer {layer_index}): output has {actual.size} values, "
                    f"Rust dump has {reference.size}"
                )
            reference = reference.reshape(actual.shape)
            error = float(np.max(np.abs(actual - reference)))
            scale = max(float(np.max(np.abs(reference))), 1.0)
            relative = error / scale
            print(
                f"{name} (layer {layer_index}): shape {actual.shape} "
                f"max_abs {error:.3e} rel_to_max {relative:.3e}"
            )
            worst = max(worst, relative)
        print("OK" if worst <= args.tol else "FAIL", f"(worst {worst:.3e}, tol {args.tol:g})")
        return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"onnx_check: {error}", file=sys.stderr)
        sys.exit(2)
