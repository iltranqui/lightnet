# Source me: `capped <mem> <cmd...>` runs cmd under a soft cgroup RAM cap
# (systemd-run --user --scope -p MemoryHigh=<mem>); without a user systemd bus
# (e.g. WSL without systemd --user) it warns and runs uncapped.
capped() {
    local mem="$1"
    shift
    if systemd-run --user --scope -q -p MemoryHigh="$mem" true 2>/dev/null; then
        systemd-run --user --scope -q -p MemoryHigh="$mem" "$@"
    else
        printf 'warning: no user systemd bus, running %s UNCAPPED\n' "$1" >&2
        "$@"
    fi
}

# Export the local GPU arch (from nvidia-smi) as the advisory
# CUDA_OXIDE_DEVICE_ARCH hint. The backend builds for this GPU only when the
# GPU can run the kernel; kernels that need a newer arch still get it. Unlike
# CUDA_OXIDE_TARGET it never forces a too-low arch. An explicit
# CUDA_OXIDE_TARGET or CUDA_OXIDE_DEVICE_ARCH wins.
oxide_arch_hint() {
    [[ -n "${CUDA_OXIDE_TARGET:-}" || -n "${CUDA_OXIDE_DEVICE_ARCH:-}" ]] && return 0
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r')" || return 0
    if [[ "$cap" =~ ^[0-9]+$ ]]; then
        export CUDA_OXIDE_DEVICE_ARCH="sm_$cap"
        printf 'oxide: CUDA_OXIDE_DEVICE_ARCH=%s (detected)\n' "$CUDA_OXIDE_DEVICE_ARCH" >&2
    fi
}
