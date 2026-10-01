#!/usr/bin/env python3
"""Fixed inputs for the training-parity run (docs/OXIDE_BENCH.md section 8).
usage: oxide_parity_data.py <outdir>   -> lego_b2.cfg (batch=2), input.bin, labels.txt, truth.bin, delta.<layer>.bin"""
import re, sys
from pathlib import Path
import numpy as np
from PIL import Image

D = Path(__file__).resolve().parents[2] / "datasets" / "LegoGears_v2"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cfg = (D / "LegoGears_v2.cfg").read_text().replace("\nbatch=16\n", "\nbatch=2\n", 1)
(out / "lego_b2.cfg").write_text(cfg)
imgs = (D / "LegoGears_v2_train.txt").read_text().split()[1:3]   # 2 images, 4 + 2 labels, already 224x160
xs, lab, tr, k = [], [], np.zeros((2, 1200), np.float32), [0, 0]   # truth: 200 boxes x (x,y,w,h,class,track)
for b, p in enumerate(imgs):
    xs.append((np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).transpose(2, 0, 1))
    for line in Path(p).with_suffix(".txt").read_text().strip().splitlines():
        c, x, y, w, h = line.split(); lab.append(f"{b} {c} {x} {y} {w} {h}")
        tr[b, k[b] * 6:k[b] * 6 + 6] = [float(x), float(y), float(w), float(h), float(c), 0]; k[b] += 1
np.stack(xs).astype("<f4").tofile(out / "input.bin")
(out / "labels.txt").write_text("\n".join(lab) + "\n")
tr.astype("<f4").tofile(out / "truth.bin")
secs = re.findall(r"^\[(\w+)\]", cfg, re.M)[1:]
rng = np.random.default_rng(1234)
for i, t in enumerate(secs):
    if t == "yolo":   # outputs per sample: 3 anchors x 10 channels x grid (7x5 at layer 30, 14x10 at layer 37), batch 2
        n = 2 * 30 * (35 if i == 30 else 140)
        ((rng.random(n) - 0.5) * 0.1).astype("<f4").tofile(out / f"delta.{i}.bin")
