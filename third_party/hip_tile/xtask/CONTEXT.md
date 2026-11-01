# xtask — project automation

One job: `cargo xtask <task>` runs the repeatable project commands, with no dependencies.

## Contents

- [Files](#files)
- [Tasks](#tasks)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `src/main.rs`: the task dispatcher and `train_hip_legogears`.
- `../.cargo/config.toml`: the alias `xtask = "run --release -p xtask --"`.

## Tasks

| Task | What it does |
|---|---|
| `train_hip_legogears` | builds `hip_darknet` (release), then runs `hip_darknet detector train CFG DATA -gpu`: LegoGears_v2 from scratch on the AMD GPU, with batch, iterations and the learning-rate schedule taken from the `.cfg` (`batch=16`, `max_batches=3000`) unless an option overrides them. Output goes to `target/legogears_v2_gpu` unless `--out`. Both steps run under `systemd-run --user --scope -p MemoryHigh=3G` when it exists. |

`cargo xtask --help` lists the options (`--iters`, `--batch`, `--subdivisions`, `--seed`, `--weights`, `--map`, `--save-every`, `--cpu`).

## Rules for changes

- Read `_shared/rules.md` first. No dependencies: keep it `std` only so it builds in seconds.
- A task wraps a command that already works by hand; it never holds logic the crates should hold.

## Human check

`cargo xtask train_hip_legogears --iters 20 --out /tmp/x` prints `device: AMD GPU (hip-tile)`, 20 iteration lines, and writes `iter_000020.weights`, `last.state`, `loss.csv`, `map.csv`.
