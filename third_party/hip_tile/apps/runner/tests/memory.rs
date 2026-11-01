use hip_rt::DeviceBuffer;
use kernels::{copy::copy, transpose::transpose};
use runner::random_vec;

fn grid(rows: usize, cols: usize) -> (u32, u32) {
    (cols.div_ceil(32) as u32, rows.div_ceil(32) as u32)
}

fn check_copy(rows: usize, cols: usize) {
    let (_device, module) = runner::load().unwrap();
    let x = random_vec(rows * cols, 3);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::from_slice(&vec![-1.0f32; rows * cols]).unwrap();
    copy::launch(
        &module,
        None,
        grid(rows, cols),
        dx.tensor(rows as u32, cols as u32),
        dy.tensor_mut(rows as u32, cols as u32),
    )
    .unwrap();
    assert!(
        dy.copy_to_vec().unwrap() == x,
        "copy {rows}x{cols} mismatch"
    );
}

fn check_transpose(rows: usize, cols: usize) {
    let (_device, module) = runner::load().unwrap();
    let x = random_vec(rows * cols, 4);
    let dx = DeviceBuffer::from_slice(&x).unwrap();
    let mut dy = DeviceBuffer::from_slice(&vec![-1.0f32; rows * cols]).unwrap();
    transpose::launch(
        &module,
        None,
        grid(rows, cols),
        dx.tensor(rows as u32, cols as u32),
        dy.tensor_mut(cols as u32, rows as u32),
    )
    .unwrap();
    let y = dy.copy_to_vec().unwrap();
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                y[j * rows + i],
                x[i * cols + j],
                "transpose {rows}x{cols} at ({i},{j})"
            );
        }
    }
}

#[test]
fn copy_1000x1000() {
    check_copy(1000, 1000);
}

#[test]
fn copy_77x130() {
    check_copy(77, 130);
}

#[test]
fn transpose_1000x1000() {
    check_transpose(1000, 1000);
}

#[test]
fn transpose_77x130() {
    check_transpose(77, 130);
}

#[test]
fn transpose_1x33() {
    check_transpose(1, 33);
}
