//! Parity of the shared element kernels (`lightnet-kernels-shared`) on the AMD GPU
//! against the same function run on the CPU.

use hip_rt::DeviceBuffer;
use kernels::shared_ops::{bias_add, route_copy};

fn check_bias_add(batch: usize, channels: usize, spatial: usize) {
    let (_device, module) = runner::load().unwrap();
    let n = batch * channels * spatial;
    let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
    let bias: Vec<f32> = (0..channels).map(|c| c as f32 * 0.5 - 1.0).collect();

    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let db = DeviceBuffer::from_slice(&bias).unwrap();
    let mut dout = DeviceBuffer::from_slice(&vec![0.0f32; n]).unwrap();
    let grid = (n as u32).div_ceil(bias_add::BLOCK_THREADS);
    bias_add::launch(
        &module,
        None,
        grid,
        dx.tensor(1, n as u32),
        db.tensor(1, channels as u32),
        dout.tensor_mut(1, n as u32),
        spatial as u32,
        channels as u32,
    )
    .unwrap();

    let got = dout.copy_to_vec().unwrap();
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

/// Copy `channels` channels of a `src_total`-channel tensor, from `src_offset`, into a
/// `dst_total`-channel tensor at `dst_offset`; the rest of `dst` must stay untouched.
fn check_route_copy(batch: usize, channels: usize, stride: usize, (src_total, src_offset): (usize, usize), (dst_total, dst_offset): (usize, usize)) {
    let (_device, module) = runner::load().unwrap();
    let src: Vec<f32> = (0..batch * src_total * stride).map(|i| i as f32 + 1.0).collect();
    let dst_len = batch * dst_total * stride;
    let ddst_init = vec![-1.0f32; dst_len];

    let dsrc = DeviceBuffer::from_slice(&src).unwrap();
    let mut ddst = DeviceBuffer::from_slice(&ddst_init).unwrap();
    let n = batch * channels * stride;
    route_copy::launch(
        &module,
        None,
        (n as u32).div_ceil(route_copy::BLOCK_THREADS),
        dsrc.tensor(1, src.len() as u32),
        ddst.tensor_mut(1, dst_len as u32),
        n as u32,
        channels as u32,
        stride as u32,
        src_total as u32,
        src_offset as u32,
        dst_total as u32,
        dst_offset as u32,
    )
    .unwrap();

    let mut want = ddst_init;
    for i in 0..n {
        let (from, to) = lightnet_kernels_shared::route_copy_index(i, channels, stride, (src_total, src_offset), (dst_total, dst_offset));
        want[to] = src[from];
    }
    assert_eq!(ddst.copy_to_vec().unwrap(), want);
    println!("route_copy b={batch} c={channels} hw={stride}: ok");
}

#[test]
fn route_copy_group_slice() {
    // Second half of a 4-channel input into channels 3.. of a 6-channel output.
    check_route_copy(2, 2, 5, (4, 2), (6, 3));
}

#[test]
fn route_copy_yolo_like() {
    check_route_copy(1, 128, 26 * 26, (256, 128), (384, 0));
}
