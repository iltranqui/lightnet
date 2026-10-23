#!/usr/bin/env python3
"""P7 benchmark driver for the cuda-oxide backend (no training runs).

Suites
  micro  GEMM (square + LegoGears conv shapes), im2col, elementwise, activations, bias, pooling,
         upsample, route, softmax, batch-norm: GFLOP/s or GB/s and % of roofline.
         Rust side: src/lightnet-cuda-oxide/tests/oxide_bench.rs (CUDA-event timed).
  conv   the 21 LegoGears_v2 conv layers, fwd / dgrad / wgrad, direct vs gemm-fp32 vs gemm-tf32 vs gemm-fp16 (fwd only, native f16 storage + fused epilogue).
  e2e    LegoGears_v2 inference latency/FPS at batch 1 and 16: oxide fp32/tf32/fp16, C++ CUDA Darknet
         (cuDNN fp16 and cuDNN fp32), Rust CPU.  Variants are INTERLEAVED round by round
         (laptop clocks drift); each round is a fresh process doing `--iters` synchronised forwards.

Method: warm-up, N iterations, median / min / p95; micro and conv samples are CUDA-event timed
mean-of-iters per round, e2e samples are wall clock around a fully synchronised forward (host
copies of the heads / outputs force the stream sync).  Output: <out>/results.json and results.csv
with git hash, GPU name, clocks, driver and the mode env vars.

Examples
  tools/oxide_bench.py --suites micro,conv                      # ~10 min
  tools/oxide_bench.py --suites e2e --rounds 5 --iters 100      # needs the release lightnet-rs
  tools/oxide_bench.py --suites e2e --batches 1 --variants oxide-fp32,cpp-cudnn-half

Build prerequisites (one build at a time, see tasks/rust/todo.md):
  RUSTUP_TOOLCHAIN=nightly-2026-08-28 target/debug/cargo-oxide oxide build -- --release -p lightnet-cli --features oxide -j 1
The Rust micro/conv suites are built and run through `cargo-oxide oxide test`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO.parent / "datasets" / "LegoGears_v2"
PEAK_FP32_GFLOPS = 15000.0  # RTX 4060 laptop, ~15 TFLOP/s fp32 (marketing)
PEAK_GBS = 272.0  # 128-bit GDDR6 @ 17 Gbps
CARGO_OXIDE = REPO / "target" / "debug" / "cargo-oxide"
TOOLCHAIN = os.environ.get("RUSTUP_TOOLCHAIN", "nightly-2026-08-28")
CAP = ["systemd-run", "--user", "--scope", "-p", "MemoryHigh=2G"]


def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def pct(v: float, peak: float) -> float:
    return round(100.0 * v / peak, 2)


def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    p95 = s[min(len(s) - 1, max(0, -(-len(s) * 95 // 100) - 1))]
    return {"median_ms": statistics.median(s), "min_ms": s[0], "p95_ms": p95, "n": len(s)}


def environment() -> dict:
    q = lambda f: sh(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"]).splitlines()[0]
    env = {k: v for k, v in os.environ.items() if k.startswith("LIGHTNET_")}
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(sh(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])),
        "gpu": q("name"),
        "driver": q("driver_version"),
        "clocks_sm_mhz": q("clocks.sm"),
        "clocks_mem_mhz": q("clocks.mem"),
        "clocks_max_sm_mhz": q("clocks.max.sm"),
        "power_w": q("power.draw"),
        "temp_c": q("temperature.gpu"),
        "toolchain": TOOLCHAIN,
        "roofline": {"fp32_gflops": PEAK_FP32_GFLOPS, "gbs": PEAK_GBS},
        "env": env,
    }


# --------------------------------------------------------------------------- micro / conv


def run_rust_bench(test: str, extra_env: dict) -> list[dict]:
    env = dict(os.environ, RUSTUP_TOOLCHAIN=TOOLCHAIN, **extra_env)
    cmd = CAP + [str(CARGO_OXIDE), "oxide", "test", "--", "--manifest-path",
                 str(REPO / "src/lightnet-cuda-oxide/Cargo.toml"), "--release", "--test", "oxide_bench",
                 "-j", "1", "--", "--ignored", "--nocapture", "--test-threads=1", test]
    print("+", shlex.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    rows = [json.loads(l.split("BENCHJSON ", 1)[1]) for l in proc.stdout.splitlines() if "BENCHJSON " in l]
    if proc.returncode != 0 or not rows:
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise SystemExit(f"rust bench {test} failed (rc={proc.returncode})")
    for r in rows:
        r["pct_peak_gflops"] = pct(r["gflops"], PEAK_FP32_GFLOPS) if r["gflops"] else None
        r["pct_peak_gbs"] = pct(r["gbs"], PEAK_GBS) if r["gbs"] else None
    return rows


# --------------------------------------------------------------------------- end to end


def ensure_cpp_probe(out_dir: Path) -> Path:
    exe = REPO / "target" / "oxide_bench" / "oxide_bench_cpp"
    src = REPO / "tools" / "oxide_bench_cpp.cpp"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    exe.parent.mkdir(parents=True, exist_ok=True)
    lib = REPO / "build" / "src-lib"
    cmd = CAP + ["g++-15", "-O2", "-std=gnu++17", "-fopenmp", "-DCUDNN", "-DCUDNN_HALF", "-DLIGHTNET_GPU",
                 "-DLIGHTNET_GPU_CUDA", "-DLIGHTNET_HAS_PROTOBUF", "-DLIGHTNET_OPENMP", "-DNDEBUG",
                 f"-I{REPO}/src-lib", f"-I{REPO}/src-other", f"-I{REPO}/src-onnx",
                 "-isystem", "/usr/local/cuda/targets/x86_64-linux/include",
                 "-isystem", "/usr/local/cuda/targets/x86_64-linux/include/cccl",
                 "-isystem", "/usr/include/opencv4", str(src), "-o", str(exe),
                 f"-L{lib}", "-ldarknet", "-L/usr/local/cuda/lib64", "-lcudart", f"-Wl,-rpath,{lib}"]
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return exe


def last_json(stdout: str, tag: str | None) -> dict:
    for line in reversed(stdout.splitlines()):
        if tag and line.startswith(tag):
            return json.loads(line[len(tag):])
        if not tag and line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("no result line in output:\n" + stdout[-2000:])


def e2e_variants(args) -> dict:
    cfg = DATASET / "LegoGears_v2.cfg"
    weights = DATASET / "LegoGears_augmatch_best.weights"
    names = DATASET / "LegoGears.names"
    image = Path(args.image)
    rs = REPO / "target" / "release" / "lightnet-rs"
    cpp = ensure_cpp_probe(Path(args.out))

    def rust(env_extra: dict, backend: str):
        def run(batch: int, iters: int) -> dict:
            env = dict(os.environ, LIGHTNET_BACKEND=backend, LIGHTNET_BENCH_ITERS=str(iters),
                       LIGHTNET_BENCH_BATCH=str(batch), **env_extra)
            p = subprocess.run([str(rs), "detect-image", str(cfg), str(weights), str(names), str(image),
                                "0.25", "0.45", "resize-exact"], env=env, capture_output=True, text=True, cwd=REPO)
            return last_json(p.stdout, "BENCH_JSON ")
        return run

    def cpp_run(half: int):
        def run(batch: int, iters: int) -> dict:
            p = subprocess.run([str(cpp), str(cfg), str(weights), str(batch), str(iters), "10", str(half)],
                               capture_output=True, text=True, cwd=REPO)
            return last_json(p.stdout, None)
        return run

    return {
        "oxide-fp32": rust({"LIGHTNET_OXIDE_GEMM": "fp32"}, "oxide"),
        "oxide-tf32": rust({"LIGHTNET_OXIDE_GEMM": "tf32"}, "oxide"),
        "oxide-fp16": rust({"LIGHTNET_OXIDE_GEMM": "fp16"}, "oxide"),
        # LIGHTNET_OXIDE_CONV_CHUNK=1: old per-image im2col + GEMM loop (A/B for the batched path)
        "oxide-fp32-loop": rust({"LIGHTNET_OXIDE_GEMM": "fp32", "LIGHTNET_OXIDE_CONV_CHUNK": "1"}, "oxide"),
        "oxide-tf32-loop": rust({"LIGHTNET_OXIDE_GEMM": "tf32", "LIGHTNET_OXIDE_CONV_CHUNK": "1"}, "oxide"),
        "oxide-fp16-loop": rust({"LIGHTNET_OXIDE_GEMM": "fp16", "LIGHTNET_OXIDE_CONV_CHUNK": "1"}, "oxide"),
        "oxide-auto": rust({"LIGHTNET_OXIDE_GEMM": "auto"}, "oxide"),
        "cpp-cudnn-half": cpp_run(1),
        "cpp-cudnn-fp32": cpp_run(0),
        "rust-cpu": rust({}, "cpu"),
    }


def run_e2e(args) -> list[dict]:
    variants = e2e_variants(args)
    names = [v for v in args.variants.split(",") if v]
    rows = []
    for batch in [int(b) for b in args.batches.split(",")]:
        samples: dict[str, list[float]] = {n: [] for n in names}
        meta: dict[str, dict] = {n: {} for n in names}
        round_medians: dict[str, list[float]] = {n: [] for n in names}
        for rnd in range(args.rounds):
            for n in names:  # interleaved A/B: every variant once per round
                if n == "rust-cpu" and rnd >= args.cpu_rounds:
                    continue
                iters = args.cpu_iters if n == "rust-cpu" else args.iters
                res = variants[n](batch, iters)
                samples[n] += res["ms"]
                round_medians[n].append(statistics.median(res["ms"]))
                meta[n] = {k: v for k, v in res.items() if k != "ms"}
                print(f"batch={batch} round={rnd} {n:15s} median {round_medians[n][-1]:8.3f} ms", flush=True)
        for n in names:
            st = stats(samples[n])
            rows.append({"group": "e2e", "name": f"legogears_v2_b{batch}", "mode": n, "batch": batch,
                         **st, "fps": round(batch * 1000.0 / st["median_ms"], 1),
                         "round_medians_ms": [round(v, 3) for v in round_medians[n]], **meta[n]})
    return rows


# --------------------------------------------------------------------------- output


def write_outputs(out: Path, env: dict, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"environment": env, "results": rows}, indent=1))
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["git_hash", "gpu", "clocks_sm_mhz"] + keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"git_hash": env["git_hash"], "gpu": env["gpu"], "clocks_sm_mhz": env["clocks_sm_mhz"], **r})
    print(f"wrote {out}/results.json and results.csv ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", default="micro,conv,e2e")
    ap.add_argument("--out", default=str(REPO / "target" / "oxide_bench" / time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--rounds", type=int, default=5, help="interleaved rounds (e2e) / samples per item (micro, conv)")
    ap.add_argument("--iters", type=int, default=100, help="e2e iterations per round per variant")
    ap.add_argument("--cpu-iters", type=int, default=2, help="e2e iterations for rust-cpu")
    ap.add_argument("--cpu-rounds", type=int, default=2, help="rounds in which rust-cpu runs (it is ~50 s/iter at batch 16)")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--variants", default="oxide-fp32,oxide-tf32,oxide-fp16,cpp-cudnn-half,cpp-cudnn-fp32,rust-cpu")
    ap.add_argument("--image", default=str(DATASET / "darkmark_image_cache" / "resize" / "00000082.png"))
    ap.add_argument("--filter", default="", help="substring filter for micro/conv item names")
    ap.add_argument("--conv-batch", type=int, default=1)
    args = ap.parse_args()

    env = environment()
    rows: list[dict] = []
    extra = {"LIGHTNET_BENCH_ROUNDS": str(max(args.rounds, 3))}
    if args.filter:
        extra["LIGHTNET_BENCH_FILTER"] = args.filter
    suites = args.suites.split(",")
    if "micro" in suites:
        rows += run_rust_bench("bench_micro", dict(extra, LIGHTNET_BENCH_MODES="fp32,tf32,fp16"))
    if "conv" in suites:
        rows += run_rust_bench("bench_conv", dict(extra, LIGHTNET_BENCH_MODES="direct,gemm-fp32,gemm-tf32,gemm-fp16",
                                                 LIGHTNET_BENCH_CONV_BATCH=str(args.conv_batch)))
    if "e2e" in suites:
        rows += run_e2e(args)
    env["clocks_sm_mhz_end"] = sh(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"])
    write_outputs(Path(args.out), env, rows)


if __name__ == "__main__":
    main()
