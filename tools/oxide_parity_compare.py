#!/usr/bin/env python3
"""Compare parameter dumps of the training-parity run (docs/OXIDE_BENCH.md section 8).
usage: oxide_parity_compare.py <prefixA> <prefixB> <step> [ref_step=0]
Compares the parameter DELTA (step - ref_step) of A against B per tensor kind: relative L2 error
||dA-dB|| / ||dB|| and max |dA-dB| / max|dB| (B is the reference), plus mean |d|. Dump layout per conv
layer (see darknet-cli dump_parity / oxide_bench_cpp parity): biases, [bn biases, scales, mean, var], weights."""
import sys
import numpy as np

def load(prefix, step):
    lens = [tuple(int(v) for v in t.split(":")) for t in open(prefix + ".layers").read().split(",")]
    raw = np.fromfile(f"{prefix}.step{step}.bin", dtype="<f4")
    out, o = [], 0
    for idx, nb, nbn, nw in lens:
        seg = {}
        for name, n in (("bias", nb), ("bn_bias", nbn), ("bn_scale", nbn), ("bn_mean", nbn), ("bn_var", nbn), ("weights", nw)):
            seg[name] = raw[o:o + n]; o += n
        out.append((idx, seg))
    assert o == raw.size, (o, raw.size)
    return out

def main():
    a, b, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ref = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    A1, A0 = load(a, step), load(a, ref)
    B1, B0 = load(b, step), load(b, ref)
    kinds = ["weights", "bias", "bn_bias", "bn_scale", "bn_mean", "bn_var"]
    print(f"{'layer':>5} " + " ".join(f"{k:>34}" for k in kinds))
    print(f"{'':>5} " + " ".join(f"{'relL2 / max/maxref / mean|dB|':>34}" for _ in kinds))
    agg = {k: [0.0, 0.0] for k in kinds}
    for (i, a1), (_, a0), (_, b1), (_, b0) in zip(A1, A0, B1, B0):
        cells = []
        for k in kinds:
            if a1[k].size == 0:
                cells.append(f"{'-':>34}"); continue
            da = a1[k].astype(np.float64) - a0[k]; db = b1[k].astype(np.float64) - b0[k]
            nb = np.linalg.norm(db)
            rel = np.linalg.norm(da - db) / nb if nb > 0 else (0.0 if np.linalg.norm(da) == 0 else float("inf"))
            mx = np.abs(da - db).max() / max(np.abs(db).max(), 1e-30)
            cells.append(f"{rel:9.2e} /{mx:9.2e} /{np.abs(db).mean():9.2e}")
            agg[k][0] = max(agg[k][0], rel); agg[k][1] += rel
        print(f"{i:>5} " + " ".join(f"{c:>34}" for c in cells))
    n = len(A1)
    print("max relL2 over layers : " + ", ".join(f"{k}={agg[k][0]:.2e}" for k in kinds))
    print("mean relL2 over layers: " + ", ".join(f"{k}={agg[k][1]/n:.2e}" for k in kinds))

main()
