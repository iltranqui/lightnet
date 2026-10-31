#!/usr/bin/env bash
# Host-only tests (lightnet-core + model + layers + runtime + train + lightnet-cli, no GPU kernels, no cuda-oxide codegen).
# Usage: ./tools/test_host.sh [test-name-filters...]     e.g. weights map_report
# Overrides: MEM (default 3G), PKGS ("lightnet-core lightnet-model lightnet-layers lightnet-runtime lightnet-train lightnet-cli")
# No filter = whole suite (several minutes); with filters only matching tests run.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

source tools/lib_cap.sh  # cwd is the repo root here
MEM="${MEM:-3G}"
PKGS=(${PKGS:-lightnet-core lightnet-model lightnet-layers lightnet-runtime lightnet-train lightnet-cli})
PKG_FLAGS=()
for p in "${PKGS[@]}"; do PKG_FLAGS+=(-p "$p"); done

capped "$MEM" \
    cargo test "${PKG_FLAGS[@]}" -j 2 -- "$@"
