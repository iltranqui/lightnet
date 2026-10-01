#!/usr/bin/env python3
"""Evaluate darknet-rs detector output with Darknet detector-map AP semantics."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def parse_detections(stdout: str) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    detections = []
    for line in stdout.splitlines():
        if not line.startswith("det\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"unexpected detection row: {line!r}")
        detections.append((fields[1], float(fields[2]), tuple(map(float, fields[3:7]))))
    return detections


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", data[16:24])
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"unsupported image format: {path}")
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        segment_length = struct.unpack(">H", data[offset:offset + 2])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
            return width, height
        offset += segment_length
    raise ValueError(f"could not read JPEG dimensions: {path}")


def ap_for_class(
    ranked: list[tuple[float, bool]], ground_truth_count: int
) -> float:
    if not ranked or ground_truth_count == 0:
        return 0.0
    tp = fp = 0
    points: list[tuple[float, float]] = []
    for _, matched in ranked:
        if matched:
            tp += 1
        else:
            fp += 1
        points.append((tp / (tp + fp), tp / ground_truth_count))

    area = 0.0
    last_recall = points[-1][1]
    last_precision = points[-1][0]
    for precision, recall in reversed(points[:-1]):
        area += (last_recall - recall) * last_precision
        last_recall = recall
        last_precision = max(last_precision, precision)
    area += last_recall * last_precision
    return area


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/darknet-rs"))
    parser.add_argument(
        "--detections-json", type=Path,
        help="reuse Darknet detector-test JSON instead of running darknet-rs",
    )
    parser.add_argument("--backend", choices=("cpu", "oxide"), default="cpu")
    parser.add_argument("--gemm", choices=("fp32", "tf32", "fp16", "auto"), default="fp32")
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU (detector map uses 0.45)")
    parser.add_argument("--save-detections", type=Path, help="write per-image detections TSV")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = [line.strip() for line in args.names.read_text().splitlines() if line.strip()]
    class_ids = {name: i for i, name in enumerate(names)}
    image_paths = [Path(line.strip()) for line in args.valid.read_text().splitlines() if line.strip()]
    gt_by_image: dict[Path, list[tuple[int, tuple[float, float, float, float]]]] = {}
    gt_counts = [0] * len(names)
    predictions: list[list[tuple[float, Path, tuple[float, float, float, float]]]] = [
        [] for _ in names
    ]

    env = os.environ.copy()
    env["DARKNET_BACKEND"] = args.backend
    env["DARKNET_OXIDE_GEMM"] = args.gemm
    env["DARKNET_DETECT_TSV"] = "1"
    json_by_image: dict[Path, list[dict]] = {}
    if args.detections_json:
        json_by_image = {
            Path(row["filename"]): row["objects"]
            for row in json.loads(args.detections_json.read_text())
        }
    for index, image in enumerate(image_paths, 1):
        label_path = image.with_suffix(".txt")
        ground_truth = []
        for row in label_path.read_text().splitlines():
            fields = row.split()
            if not fields:
                continue
            cls = int(fields[0])
            x, y, w, h = map(float, fields[1:5])
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            ground_truth.append((cls, box))
            gt_counts[cls] += 1
        gt_by_image[image] = ground_truth

        width, height = image_size(image)
        if args.detections_json:
            detections = []
            for obj in json_by_image.get(image, []):
                coords = obj["relative_coordinates"]
                cx, cy = coords["center_x"] * width, coords["center_y"] * height
                bw, bh = coords["width"] * width, coords["height"] * height
                detections.append((
                    obj["name"], obj["confidence"],
                    (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                ))
        else:
            cmd = [
                str(args.binary), "detect-image", str(args.cfg), str(args.weights), str(args.names),
                str(image), str(args.conf), str(args.nms), "resize-exact",
            ]
            completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
            if completed.returncode:
                sys.stderr.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                raise SystemExit(completed.returncode)
            detections = parse_detections(completed.stdout)
        if args.save_detections:
            with args.save_detections.open("a") as dump:
                for name, score, box in detections:
                    dump.write(f"{image}\t{name}\t{score:.9f}\t" + "\t".join(f"{v:.6f}" for v in box) + "\n")
        for name, score, box in detections:
            cls = class_ids.get(name)
            if cls is None:
                raise ValueError(f"unknown class name in detector output: {name!r}")
            normalized_box = (box[0] / width, box[1] / height,
                              box[2] / width, box[3] / height)
            predictions[cls].append((score, image, normalized_box))
        if index % 10 == 0 or index == len(image_paths):
            print(f"processed {index}/{len(image_paths)}", file=sys.stderr, flush=True)

    ap_values = []
    with args.output.open("w") as out:
        out.write("class\tgt\tdetections\tAP\n")
        for cls, name in enumerate(names):
            ranked = sorted(predictions[cls], key=lambda p: p[0], reverse=True)
            used: dict[Path, set[int]] = defaultdict(set)
            matches = []
            for score, image, predicted_box in ranked:
                best_iou, best_index = 0.0, -1
                for gt_index, (gt_cls, gt_box) in enumerate(gt_by_image[image]):
                    if gt_cls != cls:
                        continue
                    overlap = iou_xyxy(predicted_box, gt_box)
                    if overlap > args.iou and overlap > best_iou:
                        best_iou, best_index = overlap, gt_index
                if best_index >= 0 and best_index not in used[image]:
                    used[image].add(best_index)
                    matches.append((score, True))
                else:
                    matches.append((score, False))
            ap = ap_for_class(matches, gt_counts[cls])
            ap_values.append(ap)
            out.write(f"{name}\t{gt_counts[cls]}\t{len(ranked)}\t{ap * 100:.4f}\n")
        mean_ap = sum(ap_values) / len(names) if names else 0.0
        out.write(f"mAP@{args.iou:.2f}\t-\t-\t{mean_ap * 100:.4f}\n")
    print(f"mAP@{args.iou:.2f}={mean_ap * 100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
