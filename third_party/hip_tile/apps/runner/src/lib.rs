//! Embeds the device code object and loads it.

pub use kernels;

/// The kernels crate compiled for the GPU (an AMDGPU code object).
pub static KERNELS_ELF: &[u8] = include_bytes!(env!("HIP_TILE_KERNELS_ELF"));

/// Initialise HIP on device 0 and load [`KERNELS_ELF`].
pub fn load() -> hip_rt::Result<(hip_rt::Device, hip_rt::Module)> {
    let device = hip_rt::Device::init()?;
    let module = hip_rt::Module::from_elf(KERNELS_ELF)?;
    Ok((device, module))
}

/// Deterministic pseudo-random values in `[0, 1)` (xorshift), so relative errors are well defined.
pub fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// Launch the tile gemm: `c (m x n) = a (m x k) * b (k x n)`, all tightly packed row-major.
/// Asynchronous on the default stream.
pub fn launch_gemm(
    module: &hip_rt::Module,
    a: &hip_rt::DeviceBuffer<f32>,
    b: &hip_rt::DeviceBuffer<f32>,
    c: &mut hip_rt::DeviceBuffer<f32>,
    (m, n, k): (usize, usize, usize),
) -> hip_rt::Result<()> {
    use kernels::gemm::{BM, BN, gemm};
    let grid = (
        (n as u32).div_ceil(BN as u32),
        (m as u32).div_ceil(BM as u32),
    );
    gemm::launch(
        module,
        None,
        grid,
        a.tensor(m as u32, k as u32),
        b.tensor(k as u32, n as u32),
        c.tensor_mut(m as u32, n as u32),
    )
}

/// Maximum relative error of `got` (`m x n`) against an f64 CPU product of `a` and `b`,
/// over at most about `samples` elements spread over the matrix (all of them if small).
pub fn max_rel_err_vs_cpu(
    (a, b, got): (&[f32], &[f32], &[f32]),
    (m, n, k): (usize, usize, usize),
    samples: usize,
) -> f64 {
    let step = (m * n / samples).max(1);
    let mut worst = 0f64;
    // `+ 1` keeps the last element (a ragged corner) in the sample.
    let idxs = (0..m * n).step_by(step).chain([m * n - 1]);
    for idx in idxs {
        let (i, j) = (idx / n, idx % n);
        let want: f64 = (0..k)
            .map(|p| f64::from(a[i * k + p]) * f64::from(b[p * n + j]))
            .sum();
        worst = worst.max((want - f64::from(got[idx])).abs() / want.abs().max(1e-9));
    }
    worst
}
