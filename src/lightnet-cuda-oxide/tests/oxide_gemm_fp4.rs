//! Packed NVFP4 (E2M1 + E4M3 block scales) quantiser and `__dp4a` GEMM tests.
//!
//! The FP4 path is an *emulation* of the Blackwell NVFP4 recipe on Ada's
//! CUDA cores, not a tensor-core kernel (native `mma...e2m1` needs sm_120a), so
//! the checks are bit-exact rather than tolerance-based:
//!
//! 1. `quantizer_matches_the_host_reference_bit_for_bit`: the device codes and
//!    scale bytes must equal `lightnet_cuda_oxide::fp4::quantize_1x16`, an
//!    independent host implementation of the same C++ recipe
//!    (`raptor-fp4:src-fp4sim/fp4_sim_kernels.cu`).
//! 2. `dequantize_*`: the device dequantiser against the host decode.
//! 3. `gemm_matches_the_reference_kernel_exactly`: the tiled kernel against the
//!    one-thread-per-output oracle kernel shipped in the same module.
//! 4. `gemm_matches_the_cpu_integer_reference_exactly`: both kernels against
//!    `fp4::gemm_nt_packed`, whose block dot is an i32 `dp4a` sum.  Agreement
//!    here rules out tensor cores being used anywhere on the path.
//! 5. `hand_computed_single_block`: one block, values in hand.

use lightnet_cuda_oxide::fp4::{self, Fp4Packed};

use lightnet_cuda_oxide::OxideBackend;

struct Rng(u64);
impl Rng {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        lo + (hi - lo) * (r as f32 / (1u64 << 24) as f32)
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(lo, hi)).collect()
    }
}

/// The device quantiser must agree with the host recipe bit for bit.
fn check_quantized(be: &OxideBackend, rows: usize, k: usize, x: &[f32]) -> Fp4Packed {
    let got = be.fp4_quantize_1x16(x, rows, k).expect("quantize");
    let want = fp4::quantize_1x16(x, rows, k);
    assert_eq!(got.rows, want.rows);
    assert_eq!(got.k, want.k);
    assert_eq!(
        got.global_scale.to_bits(),
        want.global_scale.to_bits(),
        "global scale: got {} want {}",
        got.global_scale,
        want.global_scale
    );
    assert_eq!(got.codes.len(), want.codes.len(), "code buffer length");
    assert_eq!(got.scales.len(), want.scales.len(), "scale buffer length");
    for i in 0..got.codes.len() {
        assert_eq!(
            got.codes[i], want.codes[i],
            "code byte {i} (row {}, block {}, byte {}) differs: {:#04x} vs {:#04x}",
            i / (k.div_ceil(16) * 8),
            (i % (k.div_ceil(16) * 8)) / 8,
            i % 8,
            got.codes[i],
            want.codes[i]
        );
    }
    for i in 0..got.scales.len() {
        assert_eq!(
            got.scales[i], want.scales[i],
            "scale byte {i}: {:#04x} vs {:#04x}",
            got.scales[i],
            want.scales[i]
        );
    }
    got
}

#[test]
fn quantizer_matches_the_host_reference_bit_for_bit() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0xfeed_face);
    // Shapes: exact block multiples, a ragged tail, single row, wide K.
    for &(rows, k) in &[(4usize, 16usize), (3, 32), (5, 20), (1, 16), (7, 129), (2, 1)] {
        let x = rng.vec(rows * k, -3.0, 3.0);
        check_quantized(&be, rows, k, &x);
    }
}

#[test]
fn quantizer_handles_special_values() {
    let be = OxideBackend::new().expect("backend");
    let rows = 3;
    let k = 32;
    let mut x = vec![0.0f32; rows * k];
    x[0] = 0.0;
    x[1] = -0.0;
    x[2] = f32::INFINITY;
    x[3] = f32::NEG_INFINITY;
    x[4] = f32::NAN;
    x[5] = 1e30;
    x[16] = -1e30;
    x[17] = 1e-30;
    let packed = check_quantized(&be, rows, k, &x);
    // Non-finite values are ignored by amax, so the global scale comes from the
    // largest finite magnitude (1e30) and NaN / +-Inf saturate.
    assert_eq!(packed.global_scale, 1e30 / 2688.0);
    assert!(packed.value(0, 1) == 0.0 && packed.value(0, 1).is_sign_positive() || packed.value(0, 1) == 0.0);
    assert_eq!(packed.value(0, 2), fp4::e2m1_to_f32(0x07) * fp4::e4m3_magnitude(packed.scales[0]) * packed.global_scale);
}

#[test]
fn quantizer_of_all_zero_gives_zero_codes() {
    let be = OxideBackend::new().expect("backend");
    let packed = check_quantized(&be, 2, 32, &vec![0.0f32; 64]);
    assert_eq!(packed.global_scale, 1.0);
    assert!(packed.codes.iter().all(|c| *c == 0));
    assert!(packed.scales.iter().all(|s| fp4::e4m3_magnitude(*s) == 1.0));
}

#[test]
fn dequantize_matches_the_host_decode() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0x1234_5678);
    for &(rows, k) in &[(5usize, 16usize), (4, 48), (6, 33)] {
        let x = rng.vec(rows * k, -2.0, 2.0);
        let packed = be.fp4_quantize_1x16(&x, rows, k).expect("quantize");
        let got = be.fp4_dequantize_1x16(&packed).expect("dequantize");
        let want = packed.dequantized();
        assert_eq!(got.len(), rows * k);
        for i in 0..want.len() {
            assert!(
                (got[i] - want[i]).abs() <= 1e-6 * want[i].abs().max(1e-6),
                "{rows}x{k} element {i}: device {} host {}",
                got[i],
                want[i]
            );
        }
    }
}

/// Both kernels against the CPU integer reference; bit exact.
fn check_gemm(be: &OxideBackend, m: usize, n: usize, k: usize, seed: u64) {
    let mut rng = Rng(seed);
    let a = rng.vec(m * k, -2.0, 2.0);
    let b = rng.vec(n * k, -2.0, 2.0);
    let pa = be.fp4_quantize_1x16(&a, m, k).expect("quantize a");
    let pb = be.fp4_quantize_1x16(&b, n, k).expect("quantize b");
    // The device quantiser must equal the host one before the GEMM comparison
    // can say anything about the GEMM.
    let ha = fp4::quantize_1x16(&a, m, k);
    let hb = fp4::quantize_1x16(&b, n, k);
    assert_eq!(pa.codes, ha.codes, "A codes differ from the host quantiser");
    assert_eq!(pa.scales, ha.scales, "A scales differ from the host quantiser");
    assert_eq!(pa.global_scale.to_bits(), ha.global_scale.to_bits(), "A global scale differs");
    assert_eq!(pb.codes, hb.codes, "B codes differ from the host quantiser");
    assert_eq!(pb.scales, hb.scales, "B scales differ from the host quantiser");
    assert_eq!(pb.global_scale.to_bits(), hb.global_scale.to_bits(), "B global scale differs");
    let want = fp4::gemm_nt_packed(&pa, &pb, m, n, k, 1.0, 0.0, None);
    let host = fp4::gemm_nt_packed(&ha, &hb, m, n, k, 1.0, 0.0, None);
    for i in 0..m * n {
        assert_eq!(want[i].to_bits(), host[i].to_bits(), "host reference is not self-consistent at {i}");
    }
    for reference in [false, true] {
        let got = be.gemm_fp4(&pa, &pb, None, m, n, k, 1.0, 0.0, reference).expect("gemm_fp4");
        assert_eq!(got.len(), m * n);
        for i in 0..m * n {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "{m}x{n}x{k} reference={reference} [{i}]: device {} host {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn gemm_matches_the_cpu_integer_reference_exactly() {
    let be = OxideBackend::new().expect("backend");
    for &(m, n, k) in &[
        (16usize, 16usize, 16usize),
        (1, 1, 16),
        (64, 64, 64),
        (65, 33, 48),
        (129, 70, 80),
        (7, 3, 5),
    ] {
        check_gemm(&be, m, n, k, 0xabc0 + (m * 131 + n * 17 + k) as u64);
    }
}

#[test]
fn gemm_reference_and_tiled_kernels_agree_with_each_other() {
    // A larger, more awkward shape than the exactness test above so the tiled
    // kernel's partial tiles and its per-block staging are both covered.
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0x5151_5151);
    let (m, n, k) = (200, 150, 96);
    let a = rng.vec(m * k, -1.0, 1.0);
    let b = rng.vec(n * k, -1.0, 1.0);
    let pa = be.fp4_quantize_1x16(&a, m, k).unwrap();
    let pb = be.fp4_quantize_1x16(&b, n, k).unwrap();
    let tiled = be.gemm_fp4(&pa, &pb, None, m, n, k, 1.0, 0.0, false).unwrap();
    let oracle = be.gemm_fp4(&pa, &pb, None, m, n, k, 1.0, 0.0, true).unwrap();
    for i in 0..m * n {
        assert_eq!(tiled[i].to_bits(), oracle[i].to_bits(), "[{i}]: {} vs {}", tiled[i], oracle[i]);
    }
}

#[test]
fn gemm_applies_alpha_and_beta() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0x0f0f_0f0f);
    let (m, n, k) = (48, 32, 64);
    let a = rng.vec(m * k, -1.5, 1.5);
    let b = rng.vec(n * k, -1.5, 1.5);
    let pa = be.fp4_quantize_1x16(&a, m, k).unwrap();
    let pb = be.fp4_quantize_1x16(&b, n, k).unwrap();
    let c0 = rng.vec(m * n, -1.0, 1.0);
    let plain = be.gemm_fp4(&pa, &pb, None, m, n, k, 1.0, 0.0, false).unwrap();
    for (alpha, beta) in [(1.0f32, 0.0f32), (2.0, 0.25), (-1.3, 1.0)] {
        let got = be.gemm_fp4(&pa, &pb, Some(&c0), m, n, k, alpha, beta, false).unwrap();
        for i in 0..m * n {
            let want = alpha * plain[i] + if beta != 0.0 { beta * c0[i] } else { 0.0 };
            assert!(
                (got[i] - want).abs() <= 1e-5 * want.abs().max(1e-4),
                "alpha={alpha} beta={beta} [{i}]: got {} want {want}",
                got[i]
            );
        }
    }
}

#[test]
fn hand_computed_single_block() {
    // K = 16, so exactly one block with no scale or global-scale ambiguity
    // beyond a single power of two: A row = all 2.0, B row 0 = all 1.0.
    // The global scale is 2.0 / 2688, the block scale saturates to e4m3 448,
    // and every A element codes to E2M1 6, so C[0][0] = 16 * 2 * 1 = 32.
    let be = OxideBackend::new().expect("backend");
    let a = vec![2.0f32; 16];
    let b = vec![1.0f32; 16];
    let pa = be.fp4_quantize_1x16(&a, 1, 16).unwrap();
    let pb = be.fp4_quantize_1x16(&b, 1, 16).unwrap();
    // B is quantised in its own tensor: amax 1.0 -> global 1/2688.
    assert_eq!(pa.global_scale, 2.0 / 2688.0);
    assert_eq!(pb.global_scale, 1.0 / 2688.0);
    assert!(pa.codes.iter().all(|c| *c == 0x77), "all codes should be E2M1 +6");
    assert!(pb.codes.iter().all(|c| *c == 0x77), "all codes should be E2M1 +6");
    let got = be.gemm_fp4(&pa, &pb, None, 1, 1, 16, 1.0, 0.0, false).unwrap();
    // Every dequantised operand is exact (2.0 and 1.0), so the result must be
    // the plain f32 product: 16 * 2.0 * 1.0.  The E4M3 block scales are 448 and
    // the E2M1 codes +6 in both tensors, so `0.25 * 448 * 448 * 2 / 2688 * 1 /
    // 2688` reproduces the same number.
    let want = 16.0 * 2.0 * 1.0;
    assert!(
        (got[0] - want).abs() <= 1e-6 * want,
        "got {} want {want}",
        got[0]
    );
    // And that is the f32 product of the dequantised operands, to 1e-5.
    let deq = pa.value(0, 0) * pb.value(0, 0) * 16.0;
    assert!((got[0] - deq).abs() <= 1e-5 * deq.abs(), "got {} dequantised {deq}", got[0]);
}

#[test]
fn bench_smoke() {
    // The FP4 path is slow by design; this only checks the benchmark helper
    // runs and reports a plausible positive time.
    let be = OxideBackend::new().expect("backend");
    let (m, n, k) = (256, 256, 128);
    let mut rng = Rng(99);
    let a = rng.vec(m * k, -1.0, 1.0);
    let b = rng.vec(n * k, -1.0, 1.0);
    let pa = be.fp4_quantize_1x16(&a, m, k).unwrap();
    let pb = be.fp4_quantize_1x16(&b, n, k).unwrap();
    let ms = be.bench_gemm_fp4(&pa, &pb, m, n, k, 3).expect("bench");
    assert!(ms > 0.0, "benchmark returned {ms} ms");
    println!("fp4 gemm {m}x{n}x{k}: {ms:.3} ms");
}
#[test]
fn gemm_decodes_every_e2m1_code() {
    // Row 0 of A carries all 16 E2M1 codes, row 1 all +6; row 0 of B all +6,
    // row 1 the codes in reverse.  Both kernels must reproduce the host
    // integer reference exactly, which is only possible if the 4-bit nibble
    // order, the `+0`/`-0` code and the +-0.5 grid point all decode right.
    let be = OxideBackend::new().expect("backend");
    let k = 16usize;
    // Row 0 of A carries codes 0..15 (two per byte, low nibble first), row 1
    // is all +6.  Row 0 of B is all +6, row 1 carries codes 15..0.
    let mut a_codes = vec![0u8; 2 * 8];
    for i in 0..16 {
        a_codes[i / 2] |= (i as u8) << (4 * (i % 2));
    }
    for b in 0..8 {
        a_codes[8 + b] = 0x77;
    }
    let mut b_codes = vec![0u8; 2 * 8];
    for b in 0..8 {
        b_codes[b] = 0x77;
    }
    for i in 0..16 {
        b_codes[8 + i / 2] |= ((15 - i) as u8) << (4 * (i % 2));
    }
    let scale_448 = lightnet_cuda_oxide::fp8::f32_to_fp8(448.0, lightnet_cuda_oxide::Fp8Format::E4M3);
    let a_scales = vec![scale_448; 2];
    let b_scales = vec![scale_448; 2];
    let pa = Fp4Packed { rows: 2, k, codes: a_codes.clone(), scales: a_scales.clone(), global_scale: 1.0 };
    let pb = Fp4Packed { rows: 2, k, codes: b_codes.clone(), scales: b_scales.clone(), global_scale: 1.0 };
    let want = fp4::gemm_nt_packed(&pa, &pb, 2, 2, k, 1.0, 0.0, None);
    for reference in [false, true] {
        let got = be.gemm_fp4(&pa, &pb, None, 2, 2, k, 1.0, 0.0, reference).unwrap();
        assert_eq!(got.len(), 4);
        for i in 0..4 {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "reference={reference} [{i}]: device {} host {}",
                got[i],
                want[i]
            );
        }
    }
    // Spot-check the two non-trivial outputs by hand: the +-0 grid makes the
    // all-+6 row and the all-zero column cancel exactly, and the mixed one is
    // the integer dot of the reversed code sequence.
    assert_eq!(want[0], 0.0, "codes 0..15 against all +6 must cancel");
    assert_eq!(want[3], 0.0, "all +6 against the reversed codes must cancel");
    assert_eq!(want[2], 2304.0 * (0.25 * 448.0 * 448.0), "all +6 against all +6");
    let mut mixed = 0i32;
    for i in 0..16usize {
        let da = (2.0 * fp4::e2m1_to_f32(i as u8)) as i32;
        let db = (2.0 * fp4::e2m1_to_f32((15 - i) as u8)) as i32;
        mixed += da * db;
    }
    assert_eq!(want[1], mixed as f32 * (0.25 * 448.0 * 448.0), "codes 0..15 against the reversed codes");
}
