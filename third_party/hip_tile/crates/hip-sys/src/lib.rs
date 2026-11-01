//! Hand-written FFI to libamdhip64 (and rocBLAS under the `rocblas` feature).
//!
//! Signatures and enum values are taken from `hip_runtime_api.h` / `driver_types.h`
//! of ROCm 7.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use core::ffi::{c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int;
pub const hipSuccess: hipError_t = 0;

pub type hipDevice_t = c_int;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum hipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// `hipDeviceAttribute_t` values used by `hip-rt` (`hipDeviceAttributeWarpSize`,
/// `hipDeviceAttributeMultiprocessorCount`), counted from the enum in the header.
pub type hipDeviceAttribute_t = c_int;
pub const hipDeviceAttributeMultiprocessorCount: hipDeviceAttribute_t = 63;
pub const hipDeviceAttributeWarpSize: hipDeviceAttribute_t = 87;

/// Size of `hipDeviceProp_tR0600`, measured with clang against the ROCm 7 header.
pub const DEVICE_PROP_SIZE: usize = 1472;
/// Offset of `char name[256]` in `hipDeviceProp_tR0600`.
pub const DEVICE_PROP_NAME_OFFSET: usize = 0;
/// Offset of `char gcnArchName[256]` in `hipDeviceProp_tR0600`.
pub const DEVICE_PROP_GCN_ARCH_NAME_OFFSET: usize = 1160;

/// Opaque, over-sized, aligned stand-in for `hipDeviceProp_tR0600`.
#[repr(C, align(16))]
pub struct hipDeviceProp_t(pub [u8; 2048]);

unsafe extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device: c_int) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: hipDevice_t) -> hipError_t;
    pub fn hipDeviceGetAttribute(
        value: *mut c_int,
        attr: hipDeviceAttribute_t,
        device: c_int,
    ) -> hipError_t;
    /// `hipGetDeviceProperties` is a macro for this versioned symbol.
    #[link_name = "hipGetDevicePropertiesR0600"]
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device: c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;

    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> hipError_t;

    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, stop: hipEvent_t) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;

    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    pub fn hipGetLastError() -> hipError_t;
}

#[cfg(feature = "rocblas")]
pub mod rocblas {
    //! rocBLAS subset. Values from `rocblas/internal/rocblas-types.h`.
    use core::ffi::{c_int, c_void};

    pub type rocblas_handle = *mut c_void;
    pub type rocblas_status = c_int;
    pub const rocblas_status_success: rocblas_status = 0;

    pub type rocblas_operation = c_int;
    pub const rocblas_operation_none: rocblas_operation = 111;
    pub const rocblas_operation_transpose: rocblas_operation = 112;

    unsafe extern "C" {
        pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
        pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
        pub fn rocblas_set_stream(handle: rocblas_handle, stream: super::hipStream_t)
        -> rocblas_status;
        pub fn rocblas_sgemm(
            handle: rocblas_handle,
            trans_a: rocblas_operation,
            trans_b: rocblas_operation,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: *const f32,
            c: *mut f32,
            ldc: c_int,
        ) -> rocblas_status;
    }
}
