#!/usr/bin/env python3
"""Compare two `oxide_map.py --save-detections` TSV files (image, class, score, x1,y1,x2,y2).

Detections are matched per image and class by best IoU (> 0.5, greedy by score); reports matched /
unmatched counts, max |score diff|, mean |score diff|, min IoU and max box coordinate diff (px).
Usage: oxide_det_compare.py A.tsv B.tsv [--thr 0.25]
"""
import argparse
from collections import defaultdict


def load(path, thr):
    d = defaultdict(list)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) != 7 or float(f[2]) < thr:
            continue
        d[(f[0], f[1])].append((float(f[2]), tuple(map(float, f[3:7]))))
    return d


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--thr", type=float, default=0.25)
    args = ap.parse_args()
    A, B = load(args.a, args.thr), load(args.b, args.thr)
    matched = 0
    only_a = only_b = 0
    sdiff = []
    min_iou = 1.0
    max_coord = 0.0
    for key in set(A) | set(B):
        la, lb = sorted(A.get(key, []), reverse=True), sorted(B.get(key, []), reverse=True)
        used = set()
        for sa, ba in la:
            best, bi = 0.0, -1
            for j, (sb, bb) in enumerate(lb):
                if j in used:
                    continue
                v = iou(ba, bb)
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best > 0.5:
                used.add(bi)
                matched += 1
                sb, bb = lb[bi]
                sdiff.append(abs(sa - sb))
                min_iou = min(min_iou, best)
                max_coord = max(max_coord, max(abs(x - y) for x, y in zip(ba, bb)))
            else:
                only_a += 1
        only_b += len(lb) - len(used)
    mean = sum(sdiff) / len(sdiff) if sdiff else 0.0
    print(f"thr {args.thr}: matched {matched}, only-A {only_a}, only-B {only_b}, max score diff {max(sdiff, default=0):.3e}, "
          f"mean score diff {mean:.3e}, min IoU {min_iou:.5f}, max coord diff {max_coord:.3f} px")


if __name__ == "__main__":
    main()
