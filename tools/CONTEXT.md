# tools

Helper scripts for testing, benchmarking and parity checks.

## Table of Contents

- [Scripts](#scripts)
- [Constraints](#constraints)

## Scripts

| Script | Job |
|---|---|
| `test_host.sh` | CPU test run (`-j 2`) |
| `test_oxide.sh` | Build and test `darknet-cuda-oxide` through `cargo oxide` |
| `bench_oxide.sh`, `oxide_bench.py` | cuda-oxide benchmarks; results go to `docs/OXIDE_BENCH.md` |
| `oxide_parity_data.py`, `oxide_parity_compare.py` | Training-parity run (`docs/OXIDE_BENCH.md` section 8) |
| `oxide_det_compare.py`, `oxide_map.py` | Detection comparison and mAP for the oxide backend |
| `compare_layer_dumps.py` | Diff per-layer dumps |
| `augmentation_reference.py`, `onnx_check.py` | Reference checks for augmentation and ONNX export |
| `train_legogears_oxide.sh` | Training run; start only when the user asks |

## Constraints

- Scripts cap build threads at `-j 2` and memory via `MEM=3G`.
- Delete weights, logs and sidecars a script produces once nothing needs them.
