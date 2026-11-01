use hip_rt::DeviceBuffer;
use hip_rt::rocblas::Handle;

#[test]
fn sgemm_64x64_matches_cpu() {
    const N: usize = 64;
    let _device = hip_rt::Device::init().unwrap();
    let a: Vec<f32> = (0..N * N).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let b: Vec<f32> = (0..N * N).map(|i| ((i * 3) % 13) as f32 - 6.0).collect();

    let (da, db) = (
        DeviceBuffer::from_slice(&a).unwrap(),
        DeviceBuffer::from_slice(&b).unwrap(),
    );
    let mut dc = DeviceBuffer::<f32>::zeroed(N * N).unwrap();
    Handle::new()
        .unwrap()
        .sgemm(N, N, N, &da, &db, &mut dc)
        .unwrap();
    let got = dc.copy_to_vec().unwrap();

    for i in 0..N {
        for j in 0..N {
            let want: f32 = (0..N).map(|k| a[i * N + k] * b[k * N + j]).sum();
            assert_eq!(got[i * N + j], want, "C[{i}][{j}]");
        }
    }
}
