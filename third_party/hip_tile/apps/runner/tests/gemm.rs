use hip_rt::DeviceBuffer;
use hip_rt::rocblas::Handle;
use runner::{launch_gemm, max_rel_err_vs_cpu, random_vec};

/// Runs the tile gemm and checks it against an f64 CPU reference and rocBLAS.
fn check(m: usize, n: usize, k: usize) {
    let (_device, module) = runner::load().unwrap();
    let a = random_vec(m * k, 1);
    let b = random_vec(k * n, 2);
    let (da, db) = (
        DeviceBuffer::from_slice(&a).unwrap(),
        DeviceBuffer::from_slice(&b).unwrap(),
    );
    // Poison C so unwritten elements fail the comparison.
    let mut dc = DeviceBuffer::from_slice(&vec![-1.0f32; m * n]).unwrap();
    launch_gemm(&module, &da, &db, &mut dc, (m, n, k)).unwrap();
    let got = dc.copy_to_vec().unwrap();

    let err = max_rel_err_vs_cpu((&a, &b, &got), (m, n, k), 20_000);
    assert!(err <= 1e-5, "{m}x{n}x{k}: max rel err vs f64 CPU = {err:e}");

    let mut dref = DeviceBuffer::<f32>::zeroed(m * n).unwrap();
    Handle::new()
        .unwrap()
        .sgemm(m, n, k, &da, &db, &mut dref)
        .unwrap();
    let want = dref.copy_to_vec().unwrap();
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| f64::from((g - w).abs()) / f64::from(w.abs()).max(1e-9))
        .fold(0.0, f64::max);
    assert!(
        worst <= 1e-4,
        "{m}x{n}x{k}: max rel diff vs rocBLAS = {worst:e}"
    );
}

#[test]
fn gemm_1024() {
    check(1024, 1024, 1024);
}

#[test]
fn gemm_1000() {
    check(1000, 1000, 1000);
}

#[test]
fn gemm_77() {
    check(77, 77, 77);
}

#[test]
fn gemm_130x77x300() {
    check(130, 77, 300);
}

#[test]
fn gemm_k_smaller_than_tile() {
    check(129, 130, 5);
}
