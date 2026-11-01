//! Template: copy into apps/runner/tests/ and adapt. GPU vs CPU reference for `name`.

use hip_rt::DeviceBuffer;

fn cpu_reference(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

fn check(rows: usize, cols: usize) {
    let (_device, module) = runner::load().expect("load kernels");
    let x = runner::random_vec(rows * cols, 1);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::<f32>::zeroed(rows * cols).unwrap();
    let grid = ((cols as u32).div_ceil(64), (rows as u32).div_ceil(64));
    kernels::name::name::launch(
        &module,
        None,
        grid,
        dx.tensor(rows as u32, cols as u32),
        dy.tensor_mut(rows as u32, cols as u32),
    )
    .unwrap();
    let got = dy.copy_to_vec().unwrap();
    assert_eq!(got, cpu_reference(&x), "{rows}x{cols}");
}

#[test]
fn name_matches_cpu() {
    for (r, c) in [(64, 64), (1000, 1000), (77, 130)] {
        check(r, c);
    }
}
