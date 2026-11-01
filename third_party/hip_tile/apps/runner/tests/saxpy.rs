use hip_rt::DeviceBuffer;
use kernels::saxpy::saxpy;

fn check(n: usize) {
    let (_device, module) = runner::load().unwrap();
    let a = -1.5f32;
    let x: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32 + 0.5).collect();

    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::from_slice(&y).unwrap();
    let grid = (n as u32).div_ceil(saxpy::BLOCK_THREADS);
    saxpy::launch(
        &module,
        None,
        grid,
        a,
        dx.tensor(1, n as u32),
        dy.tensor_mut(1, n as u32),
    )
    .unwrap();

    let got = dy.copy_to_vec().unwrap();
    for i in 0..n {
        assert_eq!(got[i], a * x[i] + y[i], "n={n} i={i}");
    }
}

#[test]
fn saxpy_n1() {
    check(1);
}

#[test]
fn saxpy_n1000() {
    check(1000);
}

#[test]
fn saxpy_non_multiple_of_block() {
    check((1 << 20) + 7);
}
