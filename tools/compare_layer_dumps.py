#!/usr/bin/env python3
"""Compare two DARKNET_LAYER_DUMP_DIR exports layer by layer."""

from __future__ import annotations

import argparse
from array import array
import csv
from pathlib import Path


def read_layers(directory: Path) -> dict[int, dict[str, str]]:
    with (directory / "layers.tsv").open(newline="") as stream:
        return {int(row["index"]): row for row in csv.DictReader(stream, delimiter="\t")}


def read_f32(path: Path) -> array:
    values = array("f")
    values.frombytes(path.read_bytes())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--label", default="left vs right")
    parser.add_argument("--relative-floor", type=float, default=1.0e-6)
    args = parser.parse_args()
    left_meta, right_meta = read_layers(args.left), read_layers(args.right)
    if left_meta.keys() != right_meta.keys():
        raise SystemExit("layer indices differ")
    print(f"comparison: {args.label}")
    print("layer\ttype\tcount\tmax_abs\tmax_rel")
    for index in left_meta:
        left_row, right_row = left_meta[index], right_meta[index]
        dims = ("w", "h", "c", "batch", "count")
        if any(left_row[key] != right_row[key] for key in dims):
            raise SystemExit(f"layer {index}: shape/count mismatch: {left_row} != {right_row}")
        left = read_f32(args.left / f"layer_{index:03}.f32")
        right = read_f32(args.right / f"layer_{index:03}.f32")
        if len(left) != len(right) or len(left) != int(left_row["count"]):
            raise SystemExit(f"layer {index}: raw tensor size does not match metadata")
        max_abs = 0.0
        max_rel = 0.0
        for a, b in zip(left, right):
            diff = abs(a - b)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / max(abs(a), abs(b), args.relative_floor))
        layer_type = right_row["type"]
        print(f"{index}\t{layer_type}\t{len(left)}\t{max_abs:.8g}\t{max_rel:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
