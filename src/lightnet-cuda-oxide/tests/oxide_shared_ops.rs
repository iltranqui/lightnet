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

/// Route copy through `route_copy_index`: only the sliced channels of `dst` change.
fn check_route_copy(batch: usize, channels: usize, stride: usize, (src_total, src_offset): (usize, usize), (dst_total, dst_offset): (usize, usize)) {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = lightnet_cuda_oxide::kernels::route::kernels::load(&ctx).unwrap();

    let src: Vec<f32> = (0..batch * src_total * stride).map(|i| i as f32 + 1.0).collect();
    let init = vec![-1.0f32; batch * dst_total * stride];
    let dsrc = DeviceBuffer::<f32>::from_host(&stream, &src).unwrap();
    let mut ddst = DeviceBuffer::<f32>::from_host(&stream, &init).unwrap();
    let n = batch * channels * stride;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one thread per copied element; `dst` covers `batch * dst_total * stride`.
    unsafe {
        module.route_copy(&stream, cfg, &dsrc, &mut ddst, batch, channels, src_total, src_offset, dst_total, dst_offset, stride)
    }
    .unwrap();

    let mut want = init;
    for i in 0..n {
        let (from, to) = lightnet_kernels_shared::route_copy_index(i, channels, stride, (src_total, src_offset), (dst_total, dst_offset));
        want[to] = src[from];
    }
    assert_eq!(ddst.to_host_vec(&stream).unwrap(), want);
    println!("route_copy b={batch} c={channels} hw={stride}: ok");
}

#[test]
fn route_copy_group_slice() {
    check_route_copy(2, 2, 5, (4, 2), (6, 3));
}

#[test]
fn route_copy_yolo_like() {
    check_route_copy(1, 128, 26 * 26, (256, 128), (384, 0));
}

/// Native `tanh` (the `OxideMath` override) against the shared exp-based default, which is
/// the formula the AMD kernels run, and both against the f64 value. The numbers decide
/// whether the override is worth keeping (see `tasks/hip_tile/todo.md`).
#[test]
fn tanh_native_vs_exp_based() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = shared_ops::kernels::load(&ctx).unwrap();

    let n = 1 << 20;
    // Dense sweep of [-12, 12] plus tiny values around zero.
    let x: Vec<f32> = (0..n)
        .map(|i| if i % 4 == 0 { ((i / 4) as f32 / 65536.0 - 2.0) * 1e-3 } else { (i as f32 / n as f32 - 0.5) * 24.0 })
        .collect();
    let dx = DeviceBuffer::<f32>::from_host(&stream, &x).unwrap();
    let mut dn = DeviceBuffer::<f32>::from_host(&stream, &vec![0.0f32; n]).unwrap();
    let mut de = DeviceBuffer::<f32>::from_host(&stream, &vec![0.0f32; n]).unwrap();
    let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: one thread per element; buffers cover all accesses.
    unsafe { module.shared_tanh_pair(&stream, cfg, &dx, &mut dn, &mut de) }.unwrap();
    let native = dn.to_host_vec(&stream).unwrap();
    let exp_based = de.to_host_vec(&stream).unwrap();

    let (mut d_abs, mut d_rel, mut n_abs, mut e_abs, mut n_rel, mut e_rel) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let truth = (x[i] as f64).tanh();
        let (a, b) = (native[i] as f64, exp_based[i] as f64);
        d_abs = d_abs.max((a - b).abs());
        n_abs = n_abs.max((a - truth).abs());
        e_abs = e_abs.max((b - truth).abs());
        if truth.abs() > 1e-6 {
            d_rel = d_rel.max((a - b).abs() / truth.abs());
            n_rel = n_rel.max((a - truth).abs() / truth.abs());
            e_rel = e_rel.max((b - truth).abs() / truth.abs());
        }
    }
    println!("tanh n={n}: native-vs-exp max_abs={d_abs:.3e} max_rel={d_rel:.3e}");
    println!("tanh vs f64: native max_abs={n_abs:.3e} max_rel={n_rel:.3e}; exp-based max_abs={e_abs:.3e} max_rel={e_rel:.3e}");
    assert!(d_abs < 1e-6, "native and exp-based tanh differ by {d_abs}");
}
