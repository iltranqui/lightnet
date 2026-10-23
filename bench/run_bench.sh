#!/usr/bin/env bash
# Run GPU GEMM benches for one git ref and append parsed rows to bench/*.csv.
# Usage: bench/run_bench.sh <tag-or-branch> [low_precision_kernels|lego|precisions ...]
# Default benches: low_precision_kernels lego.
# Env:
#   DARKNET_BENCH_MODES  lego/precisions modes (default fp32,fp8)
#   BENCH_SHARE_OXIDE    1 = symlink the main checkout's third_party/cuda-oxide into the
#                        scratch worktree (OPEN DECISION, default 0: script aborts early)
# Idempotent: a ref+bench that already has rows is skipped.
set -euo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$BENCH_DIR" rev-parse --show-toplevel)"
WORK="$BENCH_DIR/.work"
KERNELS_CSV="$BENCH_DIR/results_kernels.csv"
LEGO_CSV="$BENCH_DIR/results_lego.csv"
FAIL_CSV="$BENCH_DIR/failures.csv"
HEADER="ref,ref_type,commit,date,bench,mode,m,n,k,ms,gflops"
FAIL_HEADER="ref,commit,bench,reason"
SHARE_OXIDE="${BENCH_SHARE_OXIDE:-0}"

ref="${1:?usage: run_bench.sh <tag-or-branch> [bench...]}"
shift
benches=("$@")
if [[ ${#benches[@]} -eq 0 ]]; then benches=(low_precision_kernels lego); fi

for f in "$KERNELS_CSV" "$LEGO_CSV"; do
    [[ -s "$f" ]] || printf '%s\n' "$HEADER" > "$f"
done
[[ -s "$FAIL_CSV" ]] || printf '%s\n' "$FAIL_HEADER" > "$FAIL_CSV"

if git -C "$REPO" show-ref --verify --quiet "refs/tags/$ref"; then
    ref_type=tag
elif git -C "$REPO" show-ref --verify --quiet "refs/heads/$ref"; then
    ref_type=branch
else
    echo "unknown tag or branch: $ref" >&2
    exit 1
fi
commit="$(git -C "$REPO" rev-parse "$ref^{commit}")"
cdate="$(git -C "$REPO" show -s --format=%cI "$commit")"

if [[ "$SHARE_OXIDE" != 1 ]]; then
    echo "BENCH_SHARE_OXIDE is not 1: the worktree's third_party/cuda-oxide would be empty." >&2
    echo "Set BENCH_SHARE_OXIDE=1 to link the main checkout's submodule (open decision, see README.md)." >&2
    exit 2
fi
if [[ ! -e "$REPO/third_party/cuda-oxide/Cargo.toml" ]]; then
    echo "main checkout has no third_party/cuda-oxide; run: git submodule update --init third_party/cuda-oxide" >&2
    exit 2
fi

wt=""
cleanup() {
    if [[ -n "$wt" && -d "$wt" ]]; then
        # Unlink the submodule symlink first so the main checkout is never touched.
        [[ -L "$wt/third_party/cuda-oxide" ]] && rm -f "$wt/third_party/cuda-oxide"
        git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    fi
    git -C "$REPO" worktree prune 2>/dev/null || true
    wt=""
    rmdir "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

# Turn printed bench lines into CSV rows.
#   low_precision_kernels: KERNELBENCH m<M> n<N> k<K>: <mode> <ms> ms <gf> GF/s | ...
#   lego/precisions:       BENCH <label> m<M> n<N> k<K>: <mode> <ms> ms <gf> GF/s | ...
#   lego/precisions total: BENCH <label> TOTAL: <mode> <ms> ms | ...  -> bench=<label>_total,
#                          m/n/k and gflops left blank.
parse_rows() {
    local bench="$1" log="$2"
    awk -v bench="$bench" -v ref="$ref" -v rt="$ref_type" -v commit="$commit" -v cdate="$cdate" '
        BEGIN { OFS = "," }
        {
            if (bench == "low_precision_kernels") {
                if ($0 !~ /^KERNELBENCH m/) next
                body = substr($0, length("KERNELBENCH ") + 1)
            } else {
                if ($0 !~ /^BENCH /) next
                body = substr($0, length("BENCH ") + 1)
                sub(/^[^ ]+ /, "", body)          # drop the label
            }
            if (body ~ /^TOTAL:/) {
                if (bench == "low_precision_kernels") next
                sub(/^TOTAL: */, "", body)
                nseg = split(body, seg, /\|/)
                for (i = 1; i <= nseg; i++) {
                    nf = split(seg[i], t, " ")
                    if (nf < 2) continue
                    print ref, rt, commit, cdate, bench "_total", t[1], "", "", "", t[2], ""
                }
                next
            }
            colon = index(body, ":")
            if (colon == 0) next
            split(substr(body, 1, colon - 1), s, " ")
            m = substr(s[1], 2); n = substr(s[2], 2); k = substr(s[3], 2)
            nseg = split(substr(body, colon + 1), seg, /\|/)
            for (i = 1; i <= nseg; i++) {
                nf = split(seg[i], t, " ")
                if (nf < 5) continue
                print ref, rt, commit, cdate, bench, t[1], m, n, k, t[2], t[4]
            }
        }' "$log"
}

record_failure() {
    local bench="$1" reason="$2"
    if ! awk -F, -v r="$ref" -v b="$bench" 'NR>1 && $1==r && $3==b {f=1} END{exit !f}' "$FAIL_CSV"; then
        printf '%s,%s,%s,%s\n' "$ref" "$commit" "$bench" "$reason" >> "$FAIL_CSV"
    fi
}

run_one() {
    local bench="$1" test csv
    case "$bench" in
        low_precision_kernels) test=gemm_bench_low_precision_kernels; csv="$KERNELS_CSV" ;;
        lego)                  test=gemm_bench_lego;                  csv="$LEGO_CSV" ;;
        precisions)            test=gemm_bench_precisions;            csv="$LEGO_CSV" ;;
        *) echo "unknown bench: $bench" >&2; return 1 ;;
    esac

    if awk -F, -v r="$ref" -v b="$bench" 'NR>1 && $1==r && $5==b {f=1} END{exit !f}' "$csv"; then
        echo "skip $ref $bench: rows already in ${csv##*/}"
        return 0
    fi

    mkdir -p "$WORK"
    local safe="${ref//\//_}"
    wt="$WORK/$safe-${commit:0:12}-$bench"
    local log="$WORK/$safe-$bench.log"
    git -C "$REPO" worktree prune
    rm -rf "$wt"
    echo "worktree $ref ($commit) -> $wt"
    git -C "$REPO" worktree add --detach --quiet "$wt" "$commit"
    # Submodule pin is shared across refs; link the main checkout's copy (read only use).
    rmdir "$wt/third_party/cuda-oxide" 2>/dev/null || true
    ln -s "$REPO/third_party/cuda-oxide" "$wt/third_party/cuda-oxide"

    echo "run $ref $bench ($test)"
    local status=0
    if [[ "$bench" == lego || "$bench" == precisions ]]; then
        (cd "$wt" && DARKNET_BENCH_MODES="${DARKNET_BENCH_MODES:-fp32,fp8}" \
            cargo xtask oxide --test oxide_gemm_bench -- --ignored "$test") > "$log" 2>&1 || status=$?
    else
        (cd "$wt" && cargo xtask oxide --test oxide_gemm_bench -- --ignored "$test") > "$log" 2>&1 || status=$?
    fi

    local rows
    rows="$(parse_rows "$bench" "$log")"
    if [[ $status -ne 0 || -z "$rows" ]]; then
        local reason
        if [[ $status -ne 0 ]]; then
            reason="exit $status: $(grep -m1 -E '^error|panicked' "$log" | tr ',' ';' | cut -c1-200 || true)"
        else
            reason="no BENCH lines parsed"
        fi
        reason="${reason//$'\n'/ }"
        echo "FAILED $ref $bench: $reason" >&2
        record_failure "$bench" "$reason"
    else
        printf '%s\n' "$rows" >> "$csv"
        echo "appended $(printf '%s\n' "$rows" | wc -l) rows to ${csv##*/}"
    fi

    cleanup
}

for b in "${benches[@]}"; do
    run_one "$b"
done
