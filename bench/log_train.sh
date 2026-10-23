#!/usr/bin/env bash
# Append one parsed LegoGears_v2 training row to bench/results_train.csv.
# Usage: bench/log_train.sh <mode> <train.log> <loss.csv>
#   mode       fp32, fp8, fp16, bf16 (a label only)
#   train.log  stdout of `cargo xtask train_legogears ...`
#   loss.csv   <name>_loss.csv written by the run (only its last 3000 rows are used)
# Numbers come from the log and the CSV; never type them by hand.
set -euo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$BENCH_DIR" rev-parse --show-toplevel)"
OUT="$BENCH_DIR/results_train.csv"
HEADER="ref,commit,dirty,date,mode,iters,median_ms,total_s,final_avg_loss,map50,best_map50,f1"

mode="${1:?usage: log_train.sh <mode> <train.log> <loss.csv>}"
log="${2:?missing train.log}"
loss="${3:?missing loss.csv}"

[[ -s "$OUT" ]] || printf '%s\n' "$HEADER" > "$OUT"

commit="$(git -C "$REPO" rev-parse HEAD)"
ref="$(git -C "$REPO" describe --tags --always)"
dirty=0
[[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] && dirty=1
date="$(git -C "$REPO" show -s --format=%cI "$commit")"

text="$(tr '\r' '\n' < "$log")"
map50="$(grep -a 'mean average precision (mAP@0.50)=' <<<"$text" | tail -1 | sed 's/.*=\([0-9.]*\)%.*/\1/')"
best="$(grep -a '^3000: mAP@0.50' <<<"$text" | tail -1 | sed 's/.*best = \([0-9.]*\)%.*/\1/')"
f1="$(grep -a 'F1 score=' <<<"$text" | tail -1 | sed 's/.*F1 score=\([0-9.]*\).*/\1/')"

stats="$(python3 -I - "$loss" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))[-3000:]
ms = [float(r["milliseconds"]) for r in rows]
print(f'{len(rows)},{statistics.median(ms):.3f},{sum(ms)/1000:.1f},{rows[-1]["avg_loss"]}')
PY
)"

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$ref" "$commit" "$dirty" "$date" "$mode" "$stats" "$map50" "$best" "$f1" >> "$OUT"
printf 'appended %s row to %s\n' "$mode" "$OUT"
