//! Parity of the shared element kernels (`lightnet-kernels-shared`) on the NVIDIA GPU
//! against the same function run on the CPU. The AMD twin is
//! `third_party/hip_tile/apps/runner/tests/shared_ops.rs`.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use lightnet_cuda_oxide::kernels::shared_ops;

fn check_bias_add(batch: usize, channels: usize, spatial: usize) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = shared_ops::kernels::load(&ctx).unwrap();

    let n = batch * channels * spatial;
    let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let bias: Vec<f32> = (0..channels).map(|c| c as f32 * 0.5 - 1.0).collect();

    let dx = DeviceBuffer::<f32>::from_host(&stream, &x).unwrap();
    let db = DeviceBuffer::<f32>::from_host(&stream, &bias).unwrap();
    let mut dout = DeviceBuffer::<f32>::from_host(&stream, &vec![0.0f32; n]).unwrap();
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one thread per element; buffers cover all accesses.
    unsafe { module.shared_bias_add(&stream, cfg, &dx, &db, &mut dout, spatial, channels) }
        .unwrap();

    let got = dout.to_host_vec(&stream).unwrap();
    let mut max_err = 0.0f32;
    for i in 0..n {
        let want = lightnet_kernels_shared::bias_add_elem(i, &x, &bias, spatial, channels);
        max_err = max_err.max((got[i] - want).abs());
        assert_eq!(got[i], want, "i={i}");
    }
    println!("bias_add b={batch} c={channels} hw={spatial}: max_err={max_err}");
}

#[test]
fn bias_add_small() {
    check_bias_add(1, 3, 2);
}

#[test]
fn bias_add_yolo_like() {
    check_bias_add(2, 255, 13 * 13);
}
