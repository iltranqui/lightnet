#!/usr/bin/env bash
# Guard lower-layer imports while the crate split is in progress.
set -eu
cd "$(dirname "$0")/.."
status=0
while IFS= read -r -d '' file; do
    if grep -nE 'crate::(train|detect)(::|\b)' "$file"; then
        printf 'Forbidden layer dependency in %s\n' "$file" >&2
        status=1
    fi
    if [[ $file == */cpu.rs ]] && grep -nPz 'crate::runtime::Gpu|use\s+crate::runtime::\{[^}]*\bGpu' "$file"; then
        printf 'Forbidden GPU runtime dependency in %s\n' "$file" >&2
        status=1
    fi
done < <(find src/lightnet-core/src/layers -type f \( -name cpu.rs -o -name gpu.rs \) -print0)
exit "$status"
