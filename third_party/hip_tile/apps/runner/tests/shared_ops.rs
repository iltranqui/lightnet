//! Parity of the shared element kernels (`lightnet-kernels-shared`) on the AMD GPU
//! against the same function run on the CPU.

use hip_rt::DeviceBuffer;
use kernels::shared_ops::bias_add;

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
