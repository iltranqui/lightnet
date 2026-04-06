#!/usr/bin/env bash
set -euo pipefail

status=0

pick_cuda_path() {
  if [[ -n "${CUDA_TOOLKIT_PATH:-}" ]]; then
    printf '%s\n' "${CUDA_TOOLKIT_PATH}"
    return 0
  fi

  for candidate in /usr/local/cuda-13.2 /usr/local/cuda /usr/local/cuda-13; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

pick_llvm_config() {
  if [[ -n "${LLVM_CONFIG_PATH:-}" ]]; then
    printf '%s\n' "${LLVM_CONFIG_PATH}"
    return 0
  fi

  if command -v llvm-config >/dev/null 2>&1; then
    command -v llvm-config
    return 0
  fi

  if command -v llvm-config-21 >/dev/null 2>&1; then
    command -v llvm-config-21
    return 0
  fi

  return 1
}

pick_llvm_prefix() {
  if [[ -n "${CUDA_TILE_USE_LLVM_INSTALL_DIR:-}" ]]; then
    printf '%s\n' "${CUDA_TILE_USE_LLVM_INSTALL_DIR}"
    return 0
  fi

  for candidate in /usr/lib/llvm-21 /usr/lib/llvm-20 /usr/lib/llvm-19; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

echo "cutile bridge environment check"
echo

cuda_path="$(pick_cuda_path || true)"
if [[ -z "${cuda_path}" ]]; then
  echo "CUDA: missing"
  echo "Set CUDA_TOOLKIT_PATH to your CUDA 13.2 install root."
  status=1
else
  echo "CUDA: ${cuda_path}"
  echo "export CUDA_TOOLKIT_PATH=${cuda_path}"
fi

echo

llvm_config="$(pick_llvm_config || true)"
if [[ -z "${llvm_config}" ]]; then
  echo "LLVM: missing"
  echo "Install LLVM 21 and make llvm-config available on PATH or set LLVM_CONFIG_PATH."
  status=1
else
  llvm_version="$("${llvm_config}" --version 2>/dev/null || true)"
  echo "llvm-config: ${llvm_config}"
  echo "llvm-config version: ${llvm_version:-unknown}"

  if [[ -z "${llvm_version}" || "${llvm_version%%.*}" -lt 21 ]]; then
    echo "LLVM 21 is required by cutile-rs."
    status=1
  fi
fi

echo

llvm_prefix="$(pick_llvm_prefix || true)"
if [[ -n "${llvm_prefix}" ]]; then
  echo "CUDA_TILE_USE_LLVM_INSTALL_DIR=${llvm_prefix}"
else
  echo "CUDA_TILE_USE_LLVM_INSTALL_DIR is not set"
  echo "Set it to the LLVM 21 install prefix, for example /usr/lib/llvm-21."
  status=1
fi

echo

if [[ -n "${llvm_prefix}" ]]; then
  mlir_header="${llvm_prefix}/include/mlir-c/AffineExpr.h"
  if [[ -f "${mlir_header}" ]]; then
    echo "MLIR headers: found"
  else
    echo "MLIR headers: missing"
    echo "Install the MLIR development package, for example libmlir-21-dev."
    status=1
  fi
else
  echo "MLIR headers: unknown"
  status=1
fi

echo

if [[ -n "${llvm_prefix}" ]]; then
  if [[ -f "${llvm_prefix}/lib/libPolly.a" && -f "${llvm_prefix}/lib/libPollyISL.a" ]]; then
    echo "Polly static libs: found"
  else
    echo "Polly static libs: missing"
    echo "Install the Polly development package, for example libpolly-21-dev."
    status=1
  fi
else
  echo "Polly static libs: unknown"
  status=1
fi

echo

if [[ "${status}" -eq 0 ]]; then
  echo "Environment looks ready for the cutile bridge build."
else
  echo "Environment is not ready yet."
fi

exit "${status}"
