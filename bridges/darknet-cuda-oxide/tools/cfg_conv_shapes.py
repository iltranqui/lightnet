#!/usr/bin/env python3
"""Scan darknet cfgs, propagate (c,h,w) layer by layer, and dump the convolution
geometries (plus maxpool / upsample / route / shortcut / activation inventories)
that the cfgs use.  Read-only over the cfgs; writes CSV/MD into tests/data.

Shape rules mirror src-lib/darknet_cfg.cpp + crates/darknet-core/src/model.rs:
  conv   : pad=1 -> padding=size/2 else `padding`; out=(in+2*pad-size)/stride+1
           (darknet ignores dilation in the output size -- flagged below)
  maxpool: out=(in+padding-size)/stride+1, padding default size-1
  route  : concat channels of `layers` (negative = relative, else absolute), / groups
  shortcut/sam/dropout/softmax/yolo/...: shape unchanged (scale_channels: from-layer shape)

Usage:  cfg_conv_shapes.py [--out-dir DIR] [--crosscheck]
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import threading
from collections import Counter, defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
OXIDE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(OXIDE, "..", ".."))
EXTRA_DIRS = [os.path.abspath(os.path.join(REPO, "..", "datasets", "LegoGears_v2"))]

SUPPORTED_ACTS = {"linear", "logistic", "relu", "leaky", "mish", "swish", "tanh"}
# activation default when the key is absent (darknet parser)
ACT_DEFAULT = {
    "convolutional": "logistic", "connected": "logistic", "deconvolutional": "logistic",
    "local": "logistic", "crnn": "logistic", "rnn": "logistic", "conv_lstm": "linear",
    "shortcut": "linear", "sam": "linear", "scale_channels": "linear",
}
ALIASES = {"conv": "convolutional", "max": "maxpool", "avg": "avgpool", "conn": "connected",
           "network": "net", "reorg3d": "reorg", "local_avg": "local_avgpool",
           "Gaussian_yolo": "gaussian_yolo"}
SAME_SHAPE = {"yolo", "gaussian_yolo", "region", "detection", "dropout", "softmax", "cost",
              "batchnorm", "sam", "contrastive", "silence"}


class ShapeFail(Exception):
    pass


def parse_cfg(path):
    sections = []
    cur = None
    with open(path, errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line[0] in "#;":
                continue
            line = re.split(r"\s[#;]", line)[0].strip()
            if line.startswith("["):
                name = line.strip("[] \t").strip()
                name = ALIASES.get(name, name)
                name = ALIASES.get(name.lower(), name.lower())
                cur = {"type": name, "kv": OrderedDict(), "line": ln}
                sections.append(cur)
            elif "=" in line and cur is not None:
                k, v = line.split("=", 1)
                cur["kv"][k.strip()] = v.strip()
    return sections


def gi(kv, key, default):
    try:
        return int(float(kv[key])) if key in kv else default
    except ValueError:
        return default


def glist(kv, key):
    if key not in kv:
        return []
    return [int(x) for x in re.split(r"[,\s]+", kv[key].strip()) if x != ""]


def conv_like_geom(kv, kind):
    size = gi(kv, "size", 1 if kind != "crnn" else 3)
    stride = gi(kv, "stride", 1)
    sx, sy = gi(kv, "stride_x", -1), gi(kv, "stride_y", -1)
    if sx < 1:
        sx = stride
    if sy < 1:
        sy = stride
    dil = gi(kv, "dilation", 1)
    if size == 1:
        dil = 1
    pad = gi(kv, "padding", 0)
    if gi(kv, "pad", 0):
        pad = size // 2
    return size, sx, sy, dil, pad


def propagate(path, recs):
    secs = parse_cfg(path)
    cfgname = os.path.relpath(path, REPO) if path.startswith(REPO) else path
    if not secs or secs[0]["type"] != "net":
        return "failed", "no [net] first section"
    net = secs[0]["kv"]
    h, w, c = gi(net, "height", 0), gi(net, "width", 0), gi(net, "channels", 3)
    inputs = gi(net, "inputs", h * w * c)
    batch = gi(net, "batch", 1)
    if h == 0 and w == 0 and inputs:
        c, h, w = inputs, 1, 1
    outs = []
    cur = (c, h, w)
    for idx, s in enumerate(secs[1:]):
        t, kv = s["type"], s["kv"]
        ci, hi, wi = cur

        def rec_act():
            a = kv.get("activation")
            if a is None and t in ACT_DEFAULT:
                a = ACT_DEFAULT[t]
            if a is not None:
                recs["act"].append((a.lower(), t, cfgname))

        try:
            if t == "convolutional":
                n = gi(kv, "filters", 1)
                size, sx, sy, dil, pad = conv_like_geom(kv, t)
                groups = gi(kv, "groups", 1)
                if not (hi and wi and ci):
                    raise ShapeFail("layer before convolutional must output image")
                oh = (hi + 2 * pad - size) // sy + 1
                ow = (wi + 2 * pad - size) // sx + 1
                bn = gi(kv, "batch_normalize", 0)
                if gi(kv, "cbn", 0):
                    bn = 2
                act = kv.get("activation", "logistic").lower()
                flags = []
                for k in ("antialiasing", "binary", "xnor", "deform", "share_index",
                          "assisted_excitation", "sway", "rotate", "stretch", "stretch_sway"):
                    if gi(kv, k, 0):
                        flags.append(k)
                recs["conv"].append(dict(
                    in_c=ci, in_h=hi, in_w=wi, filters=n, size=size, stride_x=sx, stride_y=sy,
                    dilation=dil, padding=pad, groups=groups, bn=bn, act=act, out_h=oh,
                    out_w=ow, cfg=cfgname, layer=idx, batch=batch, flags=flags))
                rec_act()
                cur = (n, oh, ow)
            elif t in ("deconvolutional", "local", "crnn", "conv_lstm"):
                n = gi(kv, "filters", gi(kv, "output", 1))
                size, sx, sy, dil, pad = conv_like_geom(kv, t)
                if t == "deconvolutional":
                    oh = sy * (hi - 1) + size - 2 * pad
                    ow = sx * (wi - 1) + size - 2 * pad
                else:
                    oh = (hi + 2 * pad - size) // sy + 1
                    ow = (wi + 2 * pad - size) // sx + 1
                if t in ("conv_lstm", "crnn"):
                    n = gi(kv, "output", 1)
                recs["unsup"].append(dict(kind=t, in_c=ci, in_h=hi, in_w=wi, filters=n, size=size,
                                          stride_x=sx, stride_y=sy, padding=pad,
                                          out_h=oh, out_w=ow, cfg=cfgname, layer=idx))
                rec_act()
                cur = (n, oh, ow)
            elif t == "maxpool":
                stride = gi(kv, "stride", 1)
                sx, sy = gi(kv, "stride_x", stride), gi(kv, "stride_y", stride)
                size = gi(kv, "size", stride)
                padding = gi(kv, "padding", size - 1)
                depth = gi(kv, "maxpool_depth", 0)
                if depth:
                    oc, oh, ow = gi(kv, "out_channels", 1), hi, wi
                else:
                    oh = (hi + padding - size) // sy + 1
                    ow = (wi + padding - size) // sx + 1
                    oc = ci
                recs["maxpool"].append(dict(size=size, stride_x=sx, stride_y=sy, padding=padding,
                                            in_c=ci, in_h=hi, in_w=wi, out_h=oh, out_w=ow,
                                            cfg=cfgname, layer=idx,
                                            antialiasing=gi(kv, "antialiasing", 0), depth=depth))
                cur = (oc, oh, ow)
            elif t == "local_avgpool":
                stride = gi(kv, "stride", 1)
                size = gi(kv, "size", stride)
                padding = gi(kv, "padding", size - 1)
                cur = (ci, (hi + padding - size) // stride + 1, (wi + padding - size) // stride + 1)
            elif t == "avgpool":
                cur = (ci, 1, 1)
            elif t == "upsample":
                st = gi(kv, "stride", 2)
                if st == 0:
                    raise ShapeFail("upsample stride 0")
                cur = (ci, hi // -st, wi // -st) if st < 0 else (ci, hi * st, wi * st)
                recs["upsample"].append(dict(stride=st, in_c=ci, in_h=hi, in_w=wi,
                                             out_h=cur[1], out_w=cur[2], cfg=cfgname, layer=idx))
            elif t == "reorg":
                st = gi(kv, "stride", 1)
                if gi(kv, "reverse", 0):
                    cur = (ci // (st * st), hi * st, wi * st)
                else:
                    cur = (ci * st * st, hi // st, wi // st)
            elif t == "route":
                ls = glist(kv, "layers")
                if not ls:
                    raise ShapeFail("route without layers")
                groups, gid = gi(kv, "groups", 1), gi(kv, "group_id", 0)
                res = [(idx + l) if l < 0 else l for l in ls]
                if any(r < 0 or r >= idx for r in res):
                    raise ShapeFail("route index out of range %s" % ls)
                first = outs[res[0]]
                tot = 0
                for r in res:
                    if outs[r][1:] != first[1:]:
                        raise ShapeFail("route mixes spatial shapes %s vs %s" % (first, outs[r]))
                    tot += outs[r][0]
                oc = tot // groups
                recs["route"].append(dict(layers=res, groups=groups, group_id=gid,
                                          src_c=[outs[r][0] for r in res], h=first[1], w=first[2],
                                          out_c=oc, cfg=cfgname, layer=idx))
                cur = (oc, first[1], first[2])
            elif t == "shortcut":
                fr = glist(kv, "from")
                res = [(idx + l) if l < 0 else l for l in fr]
                if any(r < 0 or r >= idx for r in res):
                    raise ShapeFail("shortcut index out of range %s" % fr)
                recs["shortcut"].append(dict(cur=cur, srcs=[outs[r] for r in res], nsrc=len(res),
                                             wt=kv.get("weights_type", "none"),
                                             wn=kv.get("weights_normalization", "none"),
                                             cfg=cfgname, layer=idx))
                rec_act()
            elif t == "scale_channels":
                fr = gi(kv, "from", 0)
                r = idx + fr if fr < 0 else fr
                if r < 0 or r >= idx:
                    raise ShapeFail("scale_channels from out of range")
                cur = outs[r]
                rec_act()
            elif t == "connected":
                cur = (gi(kv, "output", 1), 1, 1)
                rec_act()
            elif t in ("rnn", "gru", "lstm"):
                cur = (gi(kv, "output", 1), 1, 1)
                rec_act()
            elif t == "crop":
                cur = (ci, gi(kv, "crop_height", hi), gi(kv, "crop_width", wi))
            elif t in SAME_SHAPE:
                if t == "sam":
                    rec_act()
            else:
                raise ShapeFail("unknown layer type [%s]" % t)
        except ShapeFail as e:
            return "failed", "shape-propagation-failed at layer %d (%s): %s" % (idx, t, e)
        outs.append(cur)
    return "ok", ""


def find_cfgs():
    cfgs = sorted(glob.glob(os.path.join(REPO, "cfg", "**", "*.cfg"), recursive=True))
    for d in EXTRA_DIRS:
        cfgs += sorted(glob.glob(os.path.join(d, "*.cfg")))
    return cfgs


def crosscheck(cfgs, conv_by_cfg):
    """Compare conv rows with the built C++ darknet's own layer table (CPU, no weights)."""
    exe = os.path.join(REPO, "build", "src-cli", "darknet")
    pat = re.compile(r"^\s*(\d+)\s+\d+\s+convolutional\s+(\d+)\s+(\d+) x (\d+) / (\d+)\s+"
                     r"(\d+) x (\d+) x (\d+)\s+->\s+(\d+) x (\d+) x (\d+)")
    tot = bad = skipped = 0
    for cfg in cfgs:
        rel = os.path.relpath(cfg, REPO) if cfg.startswith(REPO) else cfg
        rows = {r["layer"]: r for r in conv_by_cfg.get(rel, [])}
        if not rows:
            continue
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryHigh=3G", exe, "-nogpu",
               "-verbose", "speed", cfg]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", cwd="/tmp")
        table = {}
        timer = threading.Timer(60, p.kill)
        timer.start()
        try:
            for line in p.stdout:
                line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                m = pat.match(line)
                if m:
                    table[int(m.group(1))] = tuple(int(x) for x in m.groups()[1:])
                if "Total BFLOPS" in line:
                    break
        finally:
            p.kill()
            timer.cancel()
            p.wait()
        if not table:
            skipped += 1
            print("crosscheck SKIP (no table): %s" % rel)
            continue
        for lyr, r in rows.items():
            t = table.get(lyr)
            if t is None:
                continue
            tot += 1
            mine = (r["filters"], r["size"], r["size"], r["stride_x"], r["in_w"], r["in_h"],
                    r["in_c"], r["out_w"], r["out_h"], r["filters"])
            if mine != t:
                bad += 1
                print("crosscheck MISMATCH %s layer %d: mine=%s darknet=%s" % (rel, lyr, mine, t))
    print("crosscheck: %d conv layers compared, %d mismatches, %d cfgs without table"
          % (tot, bad, skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(OXIDE, "tests", "data"))
    ap.add_argument("--crosscheck", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cfgs = find_cfgs()
    recs = dict(conv=[], unsup=[], maxpool=[], upsample=[], route=[], shortcut=[], act=[])
    ok, failed = [], []
    conv_by_cfg = defaultdict(list)
    for p in cfgs:
        before = len(recs["conv"])
        try:
            st, msg = propagate(p, recs)
        except Exception as e:  # noqa
            st, msg = "failed", "shape-propagation-failed: internal error %r" % (e,)
        rel = os.path.relpath(p, REPO) if p.startswith(REPO) else p
        for r in recs["conv"][before:]:
            conv_by_cfg[rel].append(r)
        (ok if st == "ok" else failed).append((rel, msg))

    keyf = ("in_c", "in_h", "in_w", "filters", "size", "stride_x", "stride_y", "dilation",
            "padding", "groups")
    geoms = OrderedDict()
    for r in recs["conv"]:
        k = tuple(r[f] for f in keyf)
        g = geoms.setdefault(k, dict(rows=[], bn=set(), acts=set(), out=(r["out_h"], r["out_w"]),
                                     ex="%s#%d" % (r["cfg"], r["layer"])))
        g["rows"].append(r)
        g["bn"].add(r["bn"])
        g["acts"].add(r["act"])
    gid = {k: i for i, k in enumerate(geoms)}

    with open(os.path.join(a.out_dir, "conv_shapes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(list(keyf) + ["batch_normalize_any", "activations", "out_h", "out_w", "count",
                                  "example_cfg"])
        for k, g in geoms.items():
            wr.writerow(list(k) + [int(any(b for b in g["bn"])), "|".join(sorted(g["acts"])),
                                   g["out"][0], g["out"][1], len(g["rows"]), g["ex"]])
    pairs = Counter()
    for r in recs["conv"]:
        pairs[(gid[tuple(r[f] for f in keyf)], r["act"], r["bn"])] += 1
    gk = list(geoms)
    with open(os.path.join(a.out_dir, "conv_activation_pairs.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["geom_id"] + list(keyf) + ["activation", "batch_normalize", "count"])
        for (g, act, bn), n in sorted(pairs.items()):
            wr.writerow([g] + list(gk[g]) + [act, bn, n])

    def dump(name, header, table, exkey="example_cfg"):
        with open(os.path.join(a.out_dir, name), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header + ["count", exkey])
            for k, v in table.items():
                wr.writerow(list(k) + [len(v), "%s#%d" % (v[0]["cfg"], v[0]["layer"])])

    mp = defaultdict(list)
    for r in recs["maxpool"]:
        mp[(r["size"], r["stride_x"], r["stride_y"], r["padding"], r["in_c"], r["in_h"], r["in_w"],
            r["out_h"], r["out_w"])].append(r)
    dump("maxpool_shapes.csv", ["size", "stride_x", "stride_y", "padding", "in_c", "in_h", "in_w",
                                "out_h", "out_w"], mp)
    up = defaultdict(list)
    for r in recs["upsample"]:
        up[(r["stride"], r["in_c"], r["in_h"], r["in_w"], r["out_h"], r["out_w"])].append(r)
    dump("upsample_shapes.csv", ["stride", "in_c", "in_h", "in_w", "out_h", "out_w"], up)
    rt = defaultdict(list)
    for r in recs["route"]:
        rt[(len(r["layers"]), "|".join(map(str, r["src_c"])), r["groups"], r["group_id"], r["h"],
            r["w"], r["out_c"])].append(r)
    dump("route_shapes.csv", ["n_sources", "src_channels", "groups", "group_id", "h", "w", "out_c"], rt)
    sc = defaultdict(list)
    for r in recs["shortcut"]:
        srcs = "|".join("x".join(map(str, s)) for s in r["srcs"])
        sc[("x".join(map(str, r["cur"])), srcs, r["wt"], r["wn"])].append(r)
    dump("shortcut_shapes.csv", ["cur_chw", "from_chw_list", "weights_type", "weights_normalization"], sc)

    acts = defaultdict(lambda: dict(n=0, types=Counter(), cfgs=set(), ex=None))
    for act, t, cfg in recs["act"]:
        d = acts[act]
        d["n"] += 1
        d["types"][t] += 1
        d["cfgs"].add(cfg)
        d["ex"] = d["ex"] or cfg
    with open(os.path.join(a.out_dir, "activations.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["activation", "count", "section_types", "cfg_count", "example_cfg"])
        for act, d in sorted(acts.items(), key=lambda kv: -kv[1]["n"]):
            wr.writerow([act, d["n"], "|".join("%s:%d" % kv for kv in d["types"].most_common()),
                         len(d["cfgs"]), d["ex"]])

    def macs(r):
        return r["out_h"] * r["out_w"] * r["filters"] * (r["in_c"] // max(r["groups"], 1)) * r["size"] ** 2

    L = []
    L.append("# Conv shapes scanned from darknet cfgs\n")
    L.append("Generated by `tools/cfg_conv_shapes.py`. Shapes propagated layer by layer with the "
             "darknet rules (see script docstring); output dims follow `convolutional_out_height/width` "
             "= `(in + 2*pad - size)/stride + 1` (darknet and `crates/darknet-core` both IGNORE dilation "
             "in the output size). `geom_id` in conv_activation_pairs.csv = 0-based row index of "
             "conv_shapes.csv.\n")
    L.append("## Scan\n")
    L.append("- cfgs scanned: %d (cfg/**/*.cfg incl. cfg/unsupported/* + datasets/LegoGears_v2/*.cfg)" % len(cfgs))
    L.append("- propagated fully: %d; failed: %d" % (len(ok), len(failed)))
    for rel, msg in failed:
        L.append("  - FAILED `%s`: %s" % (rel, msg))
    L.append("- convolutional layers seen (total, incl. layers before any failure): %d; unique geometries: %d"
             % (len(recs["conv"]), len(geoms)))
    uk = Counter(u["kind"] for u in recs["unsup"])
    L.append("- unsupported-but-seen conv-like layers: %s"
             % (", ".join("%s x%d" % kv for kv in sorted(uk.items())) or "none"))
    for u in recs["unsup"]:
        L.append("  - %s %s#%d: in c%d h%d w%d -> %d filters k%d s%d/%d p%d -> %dx%d"
                 % (u["kind"], u["cfg"], u["layer"], u["in_c"], u["in_h"], u["in_w"], u["filters"],
                    u["size"], u["stride_x"], u["stride_y"], u["padding"], u["out_h"], u["out_w"]))
    L.append("")
    L.append("## Distinct (size, stride_x, stride_y, dilation, groups) combos\n")
    L.append("| size | stride_x | stride_y | dilation | groups | layers | unique geoms |")
    L.append("|---|---|---|---|---|---|---|")
    combo_l, combo_g = Counter(), Counter()
    for k, g in geoms.items():
        c5 = (k[4], k[5], k[6], k[7], k[9])
        combo_l[c5] += len(g["rows"])
        combo_g[c5] += 1
    for c5, n in sorted(combo_l.items()):
        L.append("| %d | %d | %d | %d | %d | %d | %d |" % (c5 + (n, combo_g[c5])))
    L.append("")
    L.append("Padding values seen per kernel size: " + "; ".join(
        "k%d: %s" % (s, sorted({k[8] for k in geoms if k[4] == s}))
        for s in sorted({k[4] for k in geoms})))
    L.append("")
    allr = recs["conv"]
    if allr:
        L.append("## Ranges\n")
        for name in ("in_c", "filters", "in_h", "in_w", "out_h", "out_w", "groups"):
            vals = [r[name] for r in allr]
            L.append("- %s: min %d, max %d" % (name, min(vals), max(vals)))
        L.append("- [net] batch values (not part of the geometry key): %s" % sorted({r["batch"] for r in allr}))
        L.append("")
        L.append("## Heaviest unique geometries by MACs (per image)\n")
        L.append("| MACs (M) | in (c,h,w) | filters | k | s | groups | count | example |")
        L.append("|---|---|---|---|---|---|---|---|")
        gm = sorted(geoms.items(), key=lambda kv: -macs(kv[1]["rows"][0]))
        for k, g in gm[:15]:
            r = g["rows"][0]
            L.append("| %.1f | %d,%d,%d | %d | %d | %d | %d | %d | %s |"
                     % (macs(r) / 1e6, r["in_c"], r["in_h"], r["in_w"], r["filters"], r["size"],
                        r["stride_x"], r["groups"], len(g["rows"]), g["ex"]))
        L.append("")
        ms = sorted(macs(g["rows"][0]) for g in geoms.values())
        L.append("Unique-geometry MACs/image: min %.3fM, median %.1fM, max %.1fM; geometries > 100M MACs: %d, > 500M: %d; "
                 "> 1G: %d\n" % (ms[0] / 1e6, ms[len(ms) // 2] / 1e6, ms[-1] / 1e6,
                                 sum(m > 100e6 for m in ms), sum(m > 500e6 for m in ms),
                                 sum(m > 1e9 for m in ms)))

    L.append("## Layers NOT fitting the plain oxide conv trait\n")
    L.append("Darknet kernels are always square with symmetric padding, so non-square kernel / asymmetric "
             "padding cannot occur. Conditions checked (unique geoms / layers):\n")
    odd = OrderedDict()

    def add(label, cond):
        gs = [g for k, g in geoms.items() if cond(k, g)]
        odd[label] = (len(gs), sum(len(g["rows"]) for g in gs), [g["ex"] for g in gs[:3]])
    add("stride_x != stride_y", lambda k, g: k[5] != k[6])
    add("dilation > 1", lambda k, g: k[7] > 1)
    add("groups > 1", lambda k, g: k[9] > 1)
    add("groups does not divide in_c or filters", lambda k, g: k[0] % k[9] or k[3] % k[9])
    add("depthwise (groups == in_c == filters)", lambda k, g: k[9] > 1 and k[9] == k[0] == k[3])
    add("padding != size//2", lambda k, g: k[8] != k[4] // 2)
    add("size even", lambda k, g: k[4] % 2 == 0)
    add("stride > 1", lambda k, g: k[5] > 1 or k[6] > 1)
    add("stride > size", lambda k, g: max(k[5], k[6]) > k[4])
    add("batch_normalize == 2 (cbn)", lambda k, g: 2 in g["bn"])
    for label, (ng, nl, ex) in odd.items():
        L.append("- %s: %d geoms / %d layers%s" % (label, ng, nl, (" e.g. " + ", ".join(ex)) if ex else ""))
    flagc = Counter(fl for r in allr for fl in r["flags"])
    L.append("- special per-layer flags (antialiasing/binary/xnor/deform/share_index/assisted_excitation/sway...): %s"
             % (dict(flagc) or "none"))
    dil = [r for r in allr if r["dilation"] > 1]
    if dil:
        L.append("- NOTE dilation>1 layers exist; darknet out dims ignore dilation: "
                 + ", ".join("%s#%d" % (r["cfg"], r["layer"]) for r in dil[:5]))
    L.append("")

    L.append("## Maxpool (size, stride_x, stride_y, padding) combos\n")
    mc, mg = Counter(), {}
    for r in recs["maxpool"]:
        k = (r["size"], r["stride_x"], r["stride_y"], r["padding"], r["depth"], r["antialiasing"])
        mc[k] += 1
        mg.setdefault(k, "%s#%d" % (r["cfg"], r["layer"]))
    L.append("| size | sx | sy | padding | maxpool_depth | antialiasing | count | example |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, n in sorted(mc.items()):
        L.append("| %s | %d | %s |" % (" | ".join(map(str, k)), n, mg[k]))
    L.append("")
    L.append("## Upsample strides\n")
    L.append(", ".join("stride %d x%d" % (k, n)
                       for k, n in sorted(Counter(r["stride"] for r in recs["upsample"]).items())) or "none")
    L.append("")
    L.append("## Route configs\n")
    rc = Counter((len(r["layers"]), r["groups"], r["group_id"]) for r in recs["route"])
    L.append("| n_sources | groups | group_id | count |")
    L.append("|---|---|---|---|")
    for k, n in sorted(rc.items()):
        L.append("| %d | %d | %d | %d |" % (k + (n,)))
    L.append("")
    L.append("Distinct route channel patterns (src channels, out_c, h x w): %d (see route_shapes.csv)\n" % len(rt))
    L.append("## Shortcut shapes\n")
    nmis = sum(1 for r in recs["shortcut"] if any(s != r["cur"] for s in r["srcs"]))
    L.append("- shortcut layers: %d; with >1 source: %d; any source shape != current shape: %d; "
             "weights_type values: %s; distinct (cur,srcs) shapes: %d"
             % (len(recs["shortcut"]), sum(r["nsrc"] > 1 for r in recs["shortcut"]), nmis,
                dict(Counter(r["wt"] for r in recs["shortcut"])), len(sc)))
    L.append("")

    L.append("## Activations vs oxide F32ActivationKind\n")
    L.append("Supported by oxide: Linear, Logistic, ReLU, Leaky, Mish, Swish, Tanh. Counts include the "
             "parser default when `activation=` is absent (conv/connected/rnn/crnn/local: logistic; "
             "shortcut/sam/scale_channels: linear).\n")
    L.append("| activation | uses | section types | cfgs | oxide covered |")
    L.append("|---|---|---|---|---|")
    unc = []
    for act, d in sorted(acts.items(), key=lambda kv: -kv[1]["n"]):
        cov = act in SUPPORTED_ACTS
        if not cov:
            unc.append(act)
        L.append("| %s | %d | %s | %d | %s |" % (act, d["n"], ", ".join("%s:%d" % kv for kv in d["types"].most_common()),
                                                 len(d["cfgs"]), "yes" if cov else "**NO**"))
    L.append("")
    L.append("NOT covered: %s" % (", ".join(unc) or "none"))
    L.append("Supported but unused: %s" % (", ".join(sorted(SUPPORTED_ACTS - set(acts))) or "none"))
    L.append("")
    with open(os.path.join(a.out_dir, "conv_shapes_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print("cfgs=%d ok=%d failed=%d conv_layers=%d unique_geoms=%d unsup=%s"
          % (len(cfgs), len(ok), len(failed), len(recs["conv"]), len(geoms), dict(uk)))
    for rel, msg in failed:
        print("FAILED", rel, msg)
    if a.crosscheck:
        crosscheck(cfgs, conv_by_cfg)


if __name__ == "__main__":
    sys.exit(main())
