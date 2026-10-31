#!/usr/bin/env bash
# Guard the crate layering: model <- layers <- runtime <- train <- core.
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
done < <(find src/lightnet-runtime/src/layers -type f \( -name cpu.rs -o -name gpu.rs \) -print0)

if [[ ! -d src/lightnet-model/src ]]; then
    printf 'Missing lightnet-model source directory\n' >&2
    exit 1
fi
while IFS= read -r -d '' file; do
    if grep -nE 'crate::(runtime|train|detect)(::|\b)|crate::layers::([[:alnum:]_]+::)+cpu(::|\b)' "$file"; then
        printf 'Forbidden model dependency in %s\n' "$file" >&2
        status=1
    fi
done < <(find src/lightnet-model/src -type f -name '*.rs' -print0)
if [[ ! -d src/lightnet-layers/src ]]; then
    printf 'Missing lightnet-layers source directory\n' >&2
    exit 1
fi
while IFS= read -r -d '' file; do
    if grep -nE 'crate::(runtime|train|detect)(::|\b)|GpuExecutableNetwork' "$file"; then
        printf 'Forbidden layers dependency in %s\n' "$file" >&2
        status=1
    fi
done < <(find src/lightnet-layers/src -type f -name '*.rs' -print0)
if [[ ! -d src/lightnet-runtime/src ]]; then
    printf 'Missing lightnet-runtime source directory\n' >&2
    exit 1
fi
while IFS= read -r -d '' file; do
    if grep -nE 'crate::(train|detect)(::|\b)|lightnet_(train|core)\b' "$file"; then
        printf 'Forbidden runtime dependency in %s\n' "$file" >&2
        status=1
    fi
done < <(find src/lightnet-runtime/src -type f -name '*.rs' -print0)
exit "$status"
