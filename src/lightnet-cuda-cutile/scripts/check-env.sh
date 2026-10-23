#!/usr/bin/env bash
set -euo pipefail

status=0

pick_cuda_path() {
  if [[ -n "${CUDA_TOOLKIT_PATH:-}" ]]; then
    printf '%s\n' "${CUDA_TOOLKIT_PATH}"
    return 0
  fi

  for candidate in /usr/local/cuda-13.3 /usr/local/cuda-13.2 /usr/local/cuda /usr/local/cuda-13; do
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
  echo "Set CUDA_TOOLKIT_PATH to your CUDA 13.2+ install root."
  status=1
else
  echo "CUDA: ${cuda_path}"
  echo "export CUDA_TOOLKIT_PATH=${cuda_path}"
fi

echo

if [[ -n "${cuda_path}" && -x "${cuda_path}/bin/nvcc" ]]; then
  echo "nvcc: ${cuda_path}/bin/nvcc"
else
  echo "nvcc: missing"
  echo "Install the CUDA toolkit or set CUDA_TOOLKIT_PATH."
  status=1
fi

echo

if [[ -n "${cuda_path}" && -f "${cuda_path}/include/cuda.h" ]]; then
  echo "CUDA headers: found"
else
  echo "CUDA headers: missing"
  echo "Expected ${cuda_path:-CUDA_TOOLKIT_PATH}/include/cuda.h."
  status=1
fi

echo

if [[ -n "${cuda_path}" && -f "${cuda_path}/include/curand.h" ]]; then
  echo "cuRAND headers: found"
else
  echo "cuRAND headers: missing"
  echo "Install the CUDA development headers, including curand.h."
  status=1
fi

echo

libclang_found=0
if [[ -n "${LIBCLANG_PATH:-}" ]]; then
  shopt -s nullglob
  libclang_files=("${LIBCLANG_PATH}"/libclang*.so*)
  shopt -u nullglob
  if [[ "${#libclang_files[@]}" -gt 0 ]]; then
    libclang_found=1
  fi
elif command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libclang'; then
  libclang_found=1
fi
if [[ "${libclang_found}" -eq 1 ]]; then
  echo "libclang: found"
else
  echo "libclang: missing"
  echo "Install libclang or set LIBCLANG_PATH to its containing directory."
  status=1
fi

echo

if [[ "${status}" -eq 0 ]]; then
  echo "Environment looks ready for the cutile v0.2.0 bridge build."
else
  echo "Environment is not ready yet."
fi

exit "${status}"
