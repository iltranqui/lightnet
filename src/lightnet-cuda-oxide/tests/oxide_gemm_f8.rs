//! FP8 (e4m3 / e5m2) tensor-core GEMM tests.
//!
//! Three layers of checks, from the hardware upwards:
//!
//! 1. `fragment_layout_*`: the raw `mma.m16n8k32` probe.  The host builds A
//!    (16x32) and B (32x8) fragments with the layout documented in
//!    `kernels/gemm_f8.rs`, uploads them, and compares the 16x8 f32 result
//!    with a CPU product of the matrices those fragments represent.  This is
//!    the mandatory smoke test of `tasks/rust/fp8_fp4.md` step 1: the FP8 mma
//!    family is declared `sm_89+` but its runtime support was unverified, so
//!    the layout has to be pinned on the real GPU rather than assumed.
//! 2. `quantize_*`: the device quantiser against the host reference in
//!    `lightnet_cuda_oxide::fp8`, over a dense sweep that includes the
//!    saturation boundary and NaN.
//! 3. `gemm_fp8_*`: the tiled GEMM against a CPU f32 reference on the
//!    *dequantised* operands, all four transposes and odd shapes.
//!
//! Tolerance.  e4m3 keeps 3 explicit mantissa bits, so a relative error of up
//! to 2^-4 is introduced by the operand rounding alone; e5m2 keeps 2, so
//! 2^-3.  The tests therefore compare against the exact f32 product of the
//! *quantised* operands, with only the f32 accumulation order allowed to
//! differ (`2^-22 * K` relative).

use lightnet_cuda_oxide::fp8::{Fp8Format, amax_to_scale, f32_to_fp8, fp8_to_f32};
use lightnet_cuda_oxide::{Fp8Scaling, GemmMode, HalfFormat, OxideBackend, OxideTensorF32};

/// Relative slack against the *row norm* (`sum_k |a_k| |b_k|`) for the f32
/// accumulation-order difference between the tensor core and the f64 CPU
/// reference.  32 f32 additions of products bounded by the row norm give
/// ~2e-5; 1e-3 leaves ample margin while still being far tighter than the
/// ~6% error the e4m3 / e5m2 operand formats introduce on their own (a wrong
/// fragment layout moves the result by O(1) of the row norm - see
/// `fragment_layout_sweep_finds_the_documented_wiring`).
const TOL_ACC: f64 = 1e-3;

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

fn fmt_code(fmt: Fp8Format) -> u32 {
    fmt.code()
}

/// Host A fragment (16x32, row-major) for `mma.m16n8k32`: with
/// `group = lane / 4` and `thread = lane % 4`,
/// `row = group + (i >= 8) * 8`, `col = thread * 4 + (i & 3) + (i >= 8) * 16`.
fn a_fragment(a: &[f32], lane: usize, fmt: Fp8Format) -> [u8; 16] {
    // 32 lanes x 16 bytes = the whole warp's 16x32 e4m3 A tile.
    let group = lane / 4;
    let thread = lane % 4;
    let mut out = [0u8; 16];
    for i in 0..16 {
        // Registers 1 and 3 hold row group+8 (elements 4..8 and 12..16);
        // registers 2 and 3 carry the +16 K band.
        let row = group + if (i >= 4 && i < 8) || i >= 12 { 8 } else { 0 };
        let col = thread * 4 + (i & 3) + if i >= 8 { 16 } else { 0 };
        out[i] = f32_to_fp8(a[row * 32 + col], fmt);
    }
    out
}

/// Host B fragment (32x8, column-major, i.e. `b[k][n]`): with
/// `group = lane / 4` and `thread = lane % 4`,
/// `row = thread * 4 + (i & 3) + (i >= 4) * 16`, `col = group`.
fn b_fragment(b: &[f32], lane: usize, fmt: Fp8Format) -> [u8; 8] {
    // 32 lanes x 8 bytes = the whole warp's 32x8 B tile.
    let group = lane / 4;
    let thread = lane % 4;
    let mut out = [0u8; 8];
    for i in 0..8 {
        let row = thread * 4 + (i & 3) + if i >= 4 { 16 } else { 0 };
        out[i] = f32_to_fp8(b[row * 8 + group], fmt);
    }
    out
}

fn pack_a_fragments(a: &[f32], fmt: Fp8Format) -> Vec<u8> {
    let mut out = Vec::with_capacity(512);
    for lane in 0..32 {
        out.extend_from_slice(&a_fragment(a, lane, fmt));
    }
    out
}

fn pack_b_fragments(b: &[f32], fmt: Fp8Format) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for lane in 0..32 {
        out.extend_from_slice(&b_fragment(b, lane, fmt));
    }
    out
}

/// C fragment layout: `c[i * 32 + lane]` with
/// `row = group + (i >= 2) * 8`, `col = thread * 2 + (i & 1)`.
fn probe_to_matrix(c: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 16 * 8];
    for lane in 0..32 {
        let group = lane / 4;
        let thread = lane % 4;
        for i in 0..4 {
            let row = group + if i >= 2 { 8 } else { 0 };
            let col = thread * 2 + (i & 1);
            out[row * 8 + col] = c[i * 32 + lane];
        }
    }
    out
}

/// One `mma.m16n8k32` on random matrices, compared with the CPU product.
fn probe_once(be: &OxideBackend, fmt_a: Fp8Format, fmt_b: Fp8Format, seed: u64) {
    let mut rng = Rng(seed);
    // Small magnitudes so the e5m2 (2 mantissa bit) grid is exercised densely.
    let a = rng.vec(16 * 32, -1.0, 1.0);
    let b = rng.vec(32 * 8, -1.0, 1.0);
    let a_q: Vec<f32> = a.iter().map(|v| fp8_to_f32(f32_to_fp8(*v, fmt_a), fmt_a)).collect();
    let b_q: Vec<f32> = b.iter().map(|v| fp8_to_f32(f32_to_fp8(*v, fmt_b), fmt_b)).collect();
    let fa = pack_a_fragments(&a, fmt_a);
    let fb = pack_b_fragments(&b, fmt_b);
    let got = be.fp8_mma_probe(&fa, &fb, fmt_a, fmt_b).expect("probe");
    assert_eq!(got.len(), 128);
    let got = probe_to_matrix(&got);
    let mut worst = 0.0f64;
    for row in 0..16 {
        for col in 0..8 {
            let mut want = 0.0f64;
            let mut norm = 0.0f64;
            for k in 0..32 {
                let av = a_q[row * 32 + k] as f64;
                let bv = b_q[k * 8 + col] as f64;
                want += av * bv;
                norm += av.abs() * bv.abs();
            }
            let scale = norm.max(1e-6);
            let err = (got[row * 8 + col] as f64 - want).abs();
            worst = worst.max(err / scale);
            assert!(
                err <= TOL_ACC * scale,
                "{fmt_a:?}x{fmt_b:?} probe[{row},{col}]: got {}, want {want}, err {err}, scale {scale}",
                got[row * 8 + col]
            );
        }
    }
    println!("{fmt_a:?}x{fmt_b:?} probe worst err / row norm = {worst:e}");
    let _ = fmt_code(fmt_a);
}

#[test]
fn fragment_layout_e4m3_x_e4m3() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E4M3, Fp8Format::E4M3, 0x51ed_2701);
}

#[test]
fn fragment_layout_mixed_e4m3_x_e5m2() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E4M3, Fp8Format::E5M2, 0x1234_5678);
}

#[test]
fn fragment_layout_e5m2_x_e4m3_and_e5m2_x_e5m2() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E5M2, Fp8Format::E4M3, 0x0bad_c0de);
    probe_once(&be, Fp8Format::E5M2, Fp8Format::E5M2, 0xfeed_face);
}

#[test]
fn fragment_layout_hand_computed_tile() {
    // A single-lane-decodable case: A = [row r has value r+1] and
    // B = [col c has value c+1] give C[r][c] = 32 * (r+1) * (c+1).
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).map(|r| (r + 1) as f32).flat_map(|v| [v; 32]).collect();
    // B is the 32x8 tile in [k][n] order, as `b_fragment` reads it.
    let b: Vec<f32> = (0..32).flat_map(|_| (0..8).map(|c| (c + 1) as f32)).collect();
    let fa = pack_a_fragments(&a, Fp8Format::E4M3);
    let fb = pack_b_fragments(&b, Fp8Format::E4M3);
    let got = probe_to_matrix(&be.fp8_mma_probe(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap());
    for row in 0..16 {
        for col in 0..8 {
            let want = 32.0 * (row as f32 + 1.0) * (col as f32 + 1.0);
            assert!(
                (got[row * 8 + col] - want).abs() <= 1e-3 * want,
                "[{row},{col}]: got {} want {want}",
                got[row * 8 + col]
            );
        }
    }
}

#[test]
fn quantize_matches_the_host_reference_over_a_dense_sweep() {
    let be = OxideBackend::new().expect("backend");
    let mut values: Vec<f32> = Vec::new();
    // Dense sweep across the interesting magnitudes, including both
    // saturation boundaries and every representable e4m3 / e5m2 value.
    for i in 0..4000 {
        let t = i as f32 / 4000.0;
        values.push(-600.0 + t * 1200.0);
    }
    for e in -12..4 {
        let s = (2.0f32).powi(e);
        for m in 0..16 {
            values.push((1.0 + m as f32 / 16.0) * s);
            values.push(-(1.0 + m as f32 / 16.0) * s);
        }
    }
    values.push(0.0);
    values.push(-0.0);
    values.push(f32::NAN);
    values.push(-f32::NAN);
    values.push(f32::INFINITY);
    values.push(f32::NEG_INFINITY);
    values.push(1e30);
    values.push(-1e30);
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let got = be.quantize_f32_to_fp8(&values, 1.0, fmt).expect("quantize");
        assert_eq!(got.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            let want = f32_to_fp8(*v, fmt);
            assert_eq!(
                got[i], want,
                "{fmt:?}: value {v:e} (bits {:#x}) -> device {:#x}, host {want:#x}",
                v.to_bits(), got[i]
            );
        }
    }
}

#[test]
fn dequantize_is_the_inverse_of_quantize_on_finite_values() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0xabcd_ef01);
    let values: Vec<f32> = (0..4096).map(|_| rng.next(-2.0, 2.0)).collect();
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let bytes = be.quantize_f32_to_fp8(&values, 1.0, fmt).unwrap();
        let back = be.dequantize_fp8_to_f32(&bytes, fmt).unwrap();
        assert_eq!(back.len(), values.len());
        for i in 0..values.len() {
            let want = fp8_to_f32(bytes[i], fmt);
            assert_eq!(back[i], want, "{fmt:?} element {i}");
            // Relative slack from the mantissa plus the format's absolute
            // floor (half of the smallest subnormal step), which is what
            // dominates for values near the bottom of the range.
            let slack = 2f32.powi(-(4 - fmt.mantissa_bits() as i32));
            let floor = 2f32.powi(-(11 - fmt.mantissa_bits() as i32));
            assert!(
                (back[i] - values[i]).abs() <= slack * values[i].abs() + floor,
                "{fmt:?} round trip {i}: {} vs {}",
                back[i],
                values[i]
            );
        }
    }
}

#[test]
fn amax_ignores_non_finite_values() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(7);
    let mut values = rng.vec(100_000, -3.0, 3.0);
    values.push(f32::NAN);
    values.push(f32::INFINITY);
    values.push(-f32::INFINITY);
    let mut want = 0.0f32;
    for v in &values {
        if v.is_finite() {
            want = want.max(v.abs());
        }
    }
    let got = be.fp8_amax(&values).expect("amax");
    assert_eq!(got, want, "device amax {got} vs host {want}");
    assert_eq!(be.fp8_amax(&[]).unwrap(), 0.0);
    assert_eq!(amax_to_scale(want, Fp8Format::E4M3), want / 448.0);
}

/// Exact CPU reference plus the "row norm" `sum_p |a_ip| |b_pj|` of each
/// output: the tensor core accumulates in f32 in its own order, so the error
/// scales with the row norm rather than with the (possibly cancelling) sum.
fn reference_fp8_gemm(
    a_q: &[f32],
    b_q: &[f32],
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut norm = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            let mut rows = 0.0f64;
            for p in 0..k {
                let av = if trans_a { a_q[p * m + i] } else { a_q[i * k + p] } as f64;
                let bv = if trans_b { b_q[j * k + p] } else { b_q[p * n + j] } as f64;
                acc += av * bv;
                rows += av.abs() * bv.abs();
            }
            out[i * n + j] = alpha as f64 * acc;
            norm[i * n + j] = rows;
        }
    }
    (out, norm)
}

/// Quantise on the host, run the tiled kernel, compare with the CPU product of
/// the *quantised* operands.
fn gemm_case(
    be: &OxideBackend,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    fmt_a: Fp8Format,
    fmt_b: Fp8Format,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
    let (br, bc) = if trans_b { (n, k) } else { (k, n) };
    let a = rng.vec(ar * ac, -2.0, 2.0);
    let b = rng.vec(br * bc, -2.0, 2.0);
    let q = |v: &[f32], fmt: Fp8Format| -> (Vec<u8>, Vec<f32>) {
        let bytes: Vec<u8> = v.iter().map(|x| f32_to_fp8(*x, fmt)).collect();
        let back = bytes.iter().map(|x| fp8_to_f32(*x, fmt)).collect();
        (bytes, back)
    };
    let (aq, a_dq) = q(&a, fmt_a);
    let (bq, b_dq) = q(&b, fmt_b);
    let out = be.gemm_fp8(&aq, &bq, m, n, k, trans_a, trans_b, alpha, fmt_a, fmt_b).expect("gemm_fp8");
    let got = download(&be, &out);
    let (want, norm) = reference_fp8_gemm(&a_dq, &b_dq, m, n, k, trans_a, trans_b, alpha);
    let mut worst = 0.0f64;
    for idx in 0..m * n {
        let err = (got[idx] as f64 - want[idx]).abs();
        let scale = norm[idx].max(1e-4);
        worst = worst.max(err / scale);
        assert!(
            err <= TOL_ACC * scale,
            "{m}x{n}x{k} ta={trans_a} tb={trans_b} {fmt_a:?}x{fmt_b:?} [{idx}]: got {}, want {}, err {err}, scale {scale}",
            got[idx], want[idx]
        );
    }
    println!("{m}x{n}x{k} ta={trans_a} tb={trans_b}: worst err / row norm = {worst:e}");
}

fn download(be: &OxideBackend, t: &OxideTensorF32) -> Vec<f32> {
    use lightnet_cuda::F32TensorOpsBackend;
    be.download_f32(t).expect("download")
}

#[test]
fn gemm_fp8_e4m3_all_transposes() {
    let be = OxideBackend::new().expect("backend");
    for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
        gemm_case(&be, 96, 80, 64, ta, tb, 1.0, Fp8Format::E4M3, Fp8Format::E4M3, 0x1111 + ta as u64 + 10 * tb as u64);
    }
}

#[test]
fn gemm_fp8_transposed_large_k_matches_reference() {
    // Long K walks the vectorised `[k][n]` staging path (aligned 4x4 byte
    // blocks) over 128 tiles per block; n = 66 / 35 force the guarded edge
    // blocks and an unaligned row pitch.
    let be = OxideBackend::new().expect("backend");
    for &(m, n, k, ta, tb) in &[
        (96usize, 80usize, 4096usize, false, false),
        (70, 66, 4096, false, false),
        (65, 35, 4096, true, false),
        (64, 64, 4096, false, true),
    ] {
        gemm_case(&be, m, n, k, ta, tb, 1.0, Fp8Format::E4M3, Fp8Format::E4M3, 0x5000 + n as u64);
    }
}

#[test]
fn gemm_fp8_nn_is_bitwise_equal_to_nt_on_the_same_data() {
    // The same B, stored `[k][n]` (NN, register-transposed staging) and `[n][k]`
    // (NT, plain K-contiguous staging), must give identical bits: both
    // accumulate in the same K order.
    let be = OxideBackend::new().expect("backend");
    for &(m, n, k) in &[(96usize, 80usize, 200usize), (70, 66, 129), (33, 19, 4096), (130, 7, 31)] {
        let mut rng = Rng(0x6000 + k as u64);
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let aq: Vec<u8> = a.iter().map(|x| f32_to_fp8(*x, Fp8Format::E4M3)).collect();
        let bq: Vec<u8> = b.iter().map(|x| f32_to_fp8(*x, Fp8Format::E4M3)).collect();
        let mut bt = vec![0u8; k * n];
        for p in 0..k {
            for j in 0..n {
                bt[j * k + p] = bq[p * n + j];
            }
        }
        let nn = be
            .gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3)
            .expect("nn");
        let nt = be
            .gemm_fp8(&aq, &bt, m, n, k, false, true, 1.0, Fp8Format::E4M3, Fp8Format::E4M3)
            .expect("nt");
        let (nn, nt) = (download(&be, &nn), download(&be, &nt));
        let same = nn.iter().zip(&nt).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{m}x{n}x{k}: NN and NT differ");
    }
}

#[test]
fn fp8_e4m3_specialised_kernel_matches_generic() {
    // e4m3 x e4m3 normally runs `sgemm_fp8_mma_e4m3`; the seam forces the
    // generic-format kernel on the same operands. Same tiling, same K order:
    // the bits must be identical, in all four transposes, with a K tail, a
    // partial tile and the split-K path (65 x 3 x 200).
    let be = OxideBackend::new().expect("backend");
    for &(m, n, k) in &[(96usize, 80usize, 256usize), (65, 3, 200), (130, 70, 129), (64, 64, 4096)] {
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            let mut rng = Rng(0x7000 + (m * 7 + k) as u64);
            let a = rng.vec(m * k, -2.0, 2.0);
            let b = rng.vec(k * n, -2.0, 2.0);
            let aq: Vec<u8> = a.iter().map(|x| f32_to_fp8(*x, Fp8Format::E4M3)).collect();
            let bq: Vec<u8> = b.iter().map(|x| f32_to_fp8(*x, Fp8Format::E4M3)).collect();
            be.set_fp8_generic_kernel(false);
            let special = be
                .gemm_fp8(&aq, &bq, m, n, k, ta, tb, 0.75, Fp8Format::E4M3, Fp8Format::E4M3)
                .expect("specialised");
            be.set_fp8_generic_kernel(true);
            let generic = be
                .gemm_fp8(&aq, &bq, m, n, k, ta, tb, 0.75, Fp8Format::E4M3, Fp8Format::E4M3)
                .expect("generic");
            be.set_fp8_generic_kernel(false);
            let (s, g) = (download(&be, &special), download(&be, &generic));
            assert!(
                s.iter().zip(&g).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{m}x{n}x{k} ta={ta} tb={tb}: specialised and generic kernels differ"
            );
        }
    }
}

#[test]
fn gemm_fp8_odd_shapes_and_mixed_formats() {
    let be = OxideBackend::new().expect("backend");
    // Shapes that exercise the K tail (< 32, not a multiple of 32), partial
    // tiles in M and N, and the split-K path (few tiles, long K).
    for &(m, n, k) in &[(16usize, 8usize, 32usize), (1, 1, 32), (17, 19, 5), (65, 3, 200), (130, 70, 129)] {
        gemm_case(&be, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3, 0x2000 + k as u64);
        gemm_case(&be, m, n, k, false, false, 0.5, Fp8Format::E4M3, Fp8Format::E5M2, 0x3000 + k as u64);
        gemm_case(&be, m, n, k, true, true, 1.0, Fp8Format::E5M2, Fp8Format::E5M2, 0x4000 + k as u64);
    }
}

#[test]
fn gemm_fp8_matches_the_f32_reference_on_unquantised_inputs() {
    // Operands already on the e4m3 grid, so the kernel result must match the
    // plain f32 GEMM to accumulation noise: this catches scaling mistakes in
    // the epilogue that the quantised comparison above would absorb.
    let be = OxideBackend::new().expect("backend");
    let m = 128;
    let n = 96;
    let k = 256;
    let mut rng = Rng(0x9999_1111);
    let a: Vec<f32> = rng.vec(m * k, -1.0, 1.0);
    let b: Vec<f32> = rng.vec(k * n, -1.0, 1.0);
    let aq: Vec<u8> = a.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    let bq: Vec<u8> = b.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    let a_dq: Vec<f32> = aq.iter().map(|x| fp8_to_f32(*x, Fp8Format::E4M3)).collect();
    let b_dq: Vec<f32> = bq.iter().map(|x| fp8_to_f32(*x, Fp8Format::E4M3)).collect();
    let out = be
        .gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3)
        .expect("gemm_fp8");
    let got = download(&be, &out);
    for i in 0..m {
        for j in 0..n {
            let mut want = 0.0f64;
            let mut norm = 0.0f64;
            for p in 0..k {
                let av = a_dq[i * k + p] as f64;
                let bv = b_dq[p * n + j] as f64;
                want += av * bv;
                norm += av.abs() * bv.abs();
            }
            let scale = norm.max(1.0);
            assert!(
                (got[i * n + j] as f64 - want).abs() <= TOL_ACC * scale,
                "[{i},{j}]: got {} want {want}",
                got[i * n + j]
            );
        }
    }
}
// ---------------------------------------------------------------------------
// Layout sweep
// ---------------------------------------------------------------------------
//
// `tasks/rust/fp8_fp4.md` step 1 makes pinning the fragment layout on the real
// GPU mandatory: the FP8 `mma` family is declared `sm_89+` but its runtime
// support is `unexecuted` in `intrinsics/overlay/register_mma_fp8.toml`. Rather
// than trust the documentation, `fragment_layout_sweep` runs the raw mma once
// per candidate wiring and keeps the ones that reproduce an exact host GEMM.
// Only the true wiring can, because every axis (which register holds row
// group+8, which K band it covers, and the byte order inside a register)
// changes the sum.

/// A values that depend on row *and* on the K band, so swapping A registers,
/// A bands or bytes inside a register changes the product.
fn sweep_a(r: u32, c: u32) -> f32 {
    (1 + (3 * r + 5 * c) % 7 + c / 16) as f32
}

/// B values that depend on K (row and band) *and* on the column.
fn sweep_b(k: u32, n: u32) -> f32 {
    (1 + (2 * k + 3 * n + 4 * (k / 16)) % 8) as f32
}

fn a_row_offset(rp: u32, n: u32) -> u32 {
    match rp {
        0 => n & 1,
        1 => n / 2,
        2 => (n & 1) ^ 1,
        _ => (n / 2) ^ 1,
    }
}

fn a_band(ab: u32, n: u32) -> u32 {
    match ab {
        0 => 16 * (n / 2),
        1 => 4 * (n / 2),
        2 => 8 * n,
        3 => 4 * n,
        4 => 8 * (n / 2),
        _ => 16 * n,
    }
}

fn b_band(bb: u32, n: u32) -> u32 {
    match bb {
        0 => 16 * n,
        1 => 8 * n,
        2 => 4 * n,
        3 => 16 * (2 * (n & 1) + n / 2),
        4 => 16 * (n / 2),
        _ => 16 * (n & 1),
    }
}

fn byte_order(o: u32, i: u32) -> u32 {
    if o == 0 { i } else { 3 - i }
}

#[test]
fn fragment_layout_sweep_finds_the_documented_wiring() {
    const A_ROW_OPTS: u32 = 4;
    const A_BAND_OPTS: u32 = 6;
    const A_ORD_OPTS: u32 = 2;
    const B_BAND_OPTS: u32 = 6;
    const B_ORD_OPTS: u32 = 2;
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).flat_map(|r| (0..32).map(move |c| sweep_a(r, c))).collect();
    let b: Vec<f32> = (0..32).flat_map(|k| (0..8).map(move |n| sweep_b(k, n))).collect();
    let mut want = vec![0.0f64; 128];
    for r in 0..16usize {
        for n in 0..8usize {
            want[r * 8 + n] = (0..32).map(|k| (a[r * 32 + k] * b[k * 8 + n]) as f64).sum();
        }
    }
    let mut hits: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    for rp in 0..A_ROW_OPTS {
        for ab in 0..A_BAND_OPTS {
            // Skip candidates whose K band leaves the 32-wide tile.
            if (0..4).any(|n| a_band(ab, n) + 15 > 31) {
                continue;
            }
            for ao in 0..A_ORD_OPTS {
                let mut fa = Vec::with_capacity(512);
                for lane in 0..32u32 {
                    let g = lane / 4;
                    let t = lane % 4;
                    for n in 0..4u32 {
                        let r = g + 8 * a_row_offset(rp, n);
                        for i in 0..4u32 {
                            let c = 4 * t + a_band(ab, n) + byte_order(ao, i);
                            fa.push(f32_to_fp8(a[(r * 32 + c) as usize], Fp8Format::E4M3));
                        }
                    }
                }
                for bb in 0..B_BAND_OPTS {
                    if (0..2).any(|n| b_band(bb, n) + 15 > 31) {
                        continue;
                    }
                    for bo in 0..B_ORD_OPTS {
                        let mut fb = Vec::with_capacity(256);
                        for lane in 0..32u32 {
                            let g = lane / 4;
                            let t = lane % 4;
                            for n in 0..2u32 {
                                for i in 0..4u32 {
                                    let k = b_band(bb, n) + 4 * t + byte_order(bo, i);
                                    fb.push(f32_to_fp8(b[(k * 8 + g) as usize], Fp8Format::E4M3));
                                }
                            }
                        }
                        let got = probe_to_matrix(
                            &be.fp8_mma_probe(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap(),
                        );
                        let err: f64 = (0..128)
                            .map(|i| (got[i] as f64 - want[i]).abs() / want[i].abs().max(1e-3))
                            .fold(0.0, f64::max);
                        if (rp, ab, ao, bb, bo) == (0, 0, 0, 0, 0) {
                            println!("documented candidate: max rel err {err:e}");
                        }
                        if err <= 1e-3 {
                            hits.push((rp, ab, ao, bb, bo));
                        }
                    }
                }
            }
        }
    }
    println!("matching fragment layouts (a_row, a_band, a_order, b_band, b_order): {hits:?}");
    // Candidate 0 on every axis is the documented wiring.
    assert!(
        hits.contains(&(0, 0, 0, 0, 0)),
        "the documented fragment layout did not reproduce the exact GEMM; matches were {hits:?}"
    );
    assert!(!hits.is_empty(), "no candidate layout reproduced the exact GEMM");
}

#[test]
fn fragment_layout_sweep_k16() {
    // The m16n8k16 FP8 family (A = 2 registers, B = 1) swept over the same
    // candidate axes as the k32 probe.
    const A_ROW_OPTS: u32 = 4;
    const A_BAND_OPTS: u32 = 6;
    const A_ORD_OPTS: u32 = 2;
    const B_BAND_OPTS: u32 = 6;
    const B_ORD_OPTS: u32 = 2;
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).flat_map(|r| (0..16).map(move |c| sweep_a(r, c))).collect();
    let b: Vec<f32> = (0..16).flat_map(|k| (0..8).map(move |n| sweep_b(k, n))).collect();
    let mut want = vec![0.0f64; 128];
    for r in 0..16usize {
        for n in 0..8usize {
            want[r * 8 + n] = (0..16).map(|k| (a[r * 16 + k] * b[k * 8 + n]) as f64).sum();
        }
    }
    let mut hits: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    let mut best = (f64::INFINITY, (0, 0, 0, 0, 0));
    for rp in 0..A_ROW_OPTS {
        for ab in 0..A_BAND_OPTS {
            if (0..2).any(|n| a_band(ab, n) + 15 > 15) {
                continue;
            }
            for ao in 0..A_ORD_OPTS {
                let mut fa = Vec::with_capacity(256);
                for lane in 0..32u32 {
                    let g = lane / 4;
                    let t = lane % 4;
                    for n in 0..2u32 {
                        let r = g + 8 * a_row_offset(rp, n);
                        for i in 0..4u32 {
                            let c = 4 * t + a_band(ab, n) + byte_order(ao, i);
                            fa.push(f32_to_fp8(a[(r * 16 + c) as usize], Fp8Format::E4M3));
                        }
                    }
                }
                for bb in 0..B_BAND_OPTS {
                    if (0..1).any(|n| b_band(bb, n) + 15 > 15) {
                        continue;
                    }
                    for bo in 0..B_ORD_OPTS {
                        let mut fb = Vec::with_capacity(128);
                        for lane in 0..32u32 {
                            let g = lane / 4;
                            let t = lane % 4;
                            for i in 0..4u32 {
                                let k = b_band(bb, 0) + 4 * t + byte_order(bo, i);
                                fb.push(f32_to_fp8(b[(k * 8 + g) as usize], Fp8Format::E4M3));
                            }
                        }
                        let got = probe_to_matrix(
                            &be.fp8_mma_probe16(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap(),
                        );
                        let err: f64 = (0..128)
                            .map(|i| (got[i] as f64 - want[i]).abs() / want[i].abs().max(1e-3))
                            .fold(0.0, f64::max);
                        if err < best.0 {
                            best = (err, (rp, ab, ao, bb, bo));
                        }
                        if err <= 1e-3 {
                            hits.push((rp, ab, ao, bb, bo));
                        }
                    }
                }
            }
        }
    }
    println!("k16 best {best:?}, matching layouts {hits:?}");
    let ones = be
        .fp8_mma_probe16(&vec![0x38u8; 256], &vec![0x38u8; 128], Fp8Format::E4M3, Fp8Format::E4M3)
        .unwrap();
    println!("k16 all-ones raw (first 8): {:?} (16 expected)", &ones[..8]);
    assert!(!hits.is_empty(), "no k16 candidate reproduced the exact GEMM");
}

#[test]
fn f16_control_probe_uses_the_whole_warp() {
    // Same harness as the FP8 probes but with the f16 mma that the existing
    // fp16 GEMM already uses.  Every output must be 16 (16 products of 1.0);
    // if only a couple of lanes came back, the harness would be at fault
    // rather than the FP8 instruction.
    let be = OxideBackend::new().expect("backend");
    let raw = be.f16_mma_probe16(&vec![0x3C00u16; 256], &vec![0x3C00u16; 128]).unwrap();
    let nonzero: Vec<usize> = (0..128).filter(|i| raw[*i] != 0.0).collect();
    println!("f16 control: {} nonzero slots", nonzero.len());
    assert_eq!(nonzero.len(), 128);
    assert_eq!(raw.iter().filter(|v| **v == 16.0).count(), 128, "all 128 outputs must be 16");
}

#[test]
fn gemm_fp8_hand_computed_ones() {
    // A = 1 (16x32), B = 1 (8x32) -> C must be 32 everywhere.  The simplest
    // possible tiled-GEMM check: one output tile, one k step.
    let be = OxideBackend::new().expect("backend");
    let one = f32_to_fp8(1.0, Fp8Format::E4M3);
    for &(m, n, k) in &[(16usize, 8usize, 32usize), (64, 64, 32), (65, 67, 96)] {
        let aq = vec![one; m * k];
        let bq = vec![one; n * k];
        let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
        let out = download(&be, &out);
        println!("{m}x{n}x{k} ones -> {:?}", &out[..8]);
        for (i, v) in out.iter().enumerate() {
            assert!((v - k as f32).abs() <= 1e-3 * k as f32, "{m}x{n}x{k} [{i}] = {v}, want {k}");
        }
    }
}

#[test]
fn gemm_fp8_normal_operands_only() {
    // Same check with values that are all *normal* e4m3 (|v| >= 2^-6).  If the
    // tensor core flushed e4m3 subnormals to zero, this would pass while the
    // random-data tests would not.
    let be = OxideBackend::new().expect("backend");
    let grid: [f32; 8] = [0.25, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0];
    let codes: Vec<u8> = grid.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    for i in 0..8 {
        assert!(fp8_to_f32(codes[i], Fp8Format::E4M3).abs() >= 2f32.powi(-6));
    }
    let (m, n, k) = (16usize, 8usize, 32usize);
    let aq: Vec<u8> = (0..m * k).map(|i| codes[i % 8]).collect();
    let bq: Vec<u8> = (0..n * k).map(|i| codes[(i / 3) % 8]).collect();
    let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
    let got = download(&be, &out);
    let av: Vec<f32> = aq.iter().map(|c| fp8_to_f32(*c, Fp8Format::E4M3)).collect();
    let bv: Vec<f32> = bq.iter().map(|c| fp8_to_f32(*c, Fp8Format::E4M3)).collect();
    for idx in 0..m * n {
        let mut want = 0.0f64;
        for p in 0..k {
            want += av[idx / n * k + p] as f64 * bv[p * n + idx % n] as f64;
        }
        let scale = want.abs().max(1e-3);
        assert!(
            (got[idx] as f64 - want).abs() <= TOL_ACC * scale,
            "[{idx}]: got {} want {want}",
            got[idx]
        );
    }
}

#[test]
fn gemm_fp8_subnormal_operands_are_reported() {
    // Quantise a value whose e4m3 code is subnormal (0.5 * 2^-9) and check
    // what the tensor core does with it.
    let be = OxideBackend::new().expect("backend");
    let tiny = 2f32.powi(-9); // 0.001953125: the smallest e4m3 subnormal
    let code = f32_to_fp8(tiny, Fp8Format::E4M3);
    println!("e4m3 code for 2^-9 = {code:#04x} -> {}", fp8_to_f32(code, Fp8Format::E4M3));
    let (m, n, k) = (16usize, 8usize, 32usize);
    let aq = vec![code; m * k];
    let one = f32_to_fp8(1.0, Fp8Format::E4M3);
    let bq = vec![one; n * k];
    let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
    let got = download(&be, &out);
    println!("subnormal x ones -> {}", got[0]);
}

// ---------------------------------------------------------------------------
// Backend precision selector (`LIGHTNET_OXIDE_GEMM=fp8` / `fp4`)
// ---------------------------------------------------------------------------
//
// The kernels above are only useful if the network GEMM seam can reach them, so
// `GemmMode::Fp8` / `GemmMode::Fp4` route `gemm_f32` through
// quantise -> low-precision GEMM -> f32.  Both tests compare against a CPU
// reference built from the *dequantised* operands, which pins the whole path
// (including the per-tensor / per-block scale recipe) rather than just the mma.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend as _};

fn gemm_f32_reference(a: &[f32], b: &[f32], c0: &[f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut norm = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            let mut rows = 0.0f64;
            for p in 0..k {
                let av = a[i * k + p] as f64;
                let bv = b[p * n + j] as f64;
                acc += av * bv;
                rows += av.abs() * bv.abs();
            }
            out[i * n + j] = alpha as f64 * acc + beta as f64 * c0[i * n + j] as f64;
            norm[i * n + j] = rows;
        }
    }
    (out, norm)
}

#[test]
fn gemm_mode_fp8_selector_matches_the_dequantised_reference() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let mut rng = Rng(0x5eed_1234);
    for &(m, n, k) in &[(96usize, 80usize, 64usize), (65, 33, 129), (129, 70, 96)] {
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let c0 = rng.vec(m * n, -1.0, 1.0);
        let ta = be.upload_f32(&[m, k], &a).unwrap();
        let tb = be.upload_f32(&[k, n], &b).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        let got = download(&be, &be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 0.7, 0.3).unwrap());

        // Reference: quantise both operands exactly as the selector does and
        // dequantise them again.  `amax_to_scale` is the C++ *divisor*
        // (`amax / 448`), so the quantiser multiplies by its reciprocal and the
        // dequantiser multiplies by the divisor.
        let factor = |v: &[f32]| {
            let scale = lightnet_cuda_oxide::fp8::amax_to_scale(lightnet_cuda_oxide::fp8::amax_f32(v), Fp8Format::E4M3);
            1.0 / scale
        };
        let (sa, sb) = (factor(&a), factor(&b));
        let q = |v: &[f32], scale: f32| -> Vec<f32> {
            v.iter()
                .map(|x| fp8_to_f32(f32_to_fp8(x * scale, Fp8Format::E4M3), Fp8Format::E4M3) / scale)
                .collect()
        };
        let (want, norm) = gemm_f32_reference(&q(&a, sa), &q(&b, sb), &c0, m, n, k, 0.7, 0.3);
        let mut worst = 0.0f64;
        for i in 0..m * n {
            let err = (got[i] as f64 - want[i]).abs();
            worst = worst.max(err / norm[i].max(1e-4));
            // e4m3 keeps 3 explicit mantissa bits, so each operand carries up to
            // 2^-4 of relative error and the row norm bounds the sum's error at
            // about 2^-3.  0.25 is a comfortable factor of three over that and
            // still far tighter than the ~6% a wrong wiring would produce.
            assert!(err <= 0.25 * norm[i].max(1e-4), "{m}x{n}x{k} [{i}]: got {} want {}", got[i], want[i]);
        }
        println!("fp8 selector {m}x{n}x{k}: worst err / row norm = {worst:e}");
    }
}

#[test]
fn gemm_mode_fp4_selector_matches_the_dequantised_reference() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp4);
    let mut rng = Rng(0xfeed_4321);
    for &(m, n, k) in &[(64usize, 48usize, 64usize), (33, 17, 32), (96, 96, 80)] {
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let c0 = rng.vec(m * n, -1.0, 1.0);
        let ta = be.upload_f32(&[m, k], &a).unwrap();
        let tb = be.upload_f32(&[k, n], &b).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        let got = download(&be, &be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 0.7, 0.3).unwrap());

        // Reference: the same NVFP4 recipe the kernel uses (see `fp4`).  The
        // packed layout is [rows, k] for both operands, so B (stored [k, n] by
        // `gemm_f32`) is transposed first, exactly as the selector does.
        let pa = lightnet_cuda_oxide::fp4::quantize_1x16(&a, m, k);
        let mut b_k = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                b_k[j * k + p] = b[p * n + j];
            }
        }
        let pb = lightnet_cuda_oxide::fp4::quantize_1x16(&b_k, n, k);
        let packed = lightnet_cuda_oxide::fp4::gemm_nt_packed(&pa, &pb, m, n, k, 1.0, 0.0, None);
        for i in 0..m * n {
            let want = 0.7 * packed[i] as f64 + 0.3 * c0[i] as f64;
            let err = (got[i] as f64 - want).abs();
            let scale = (packed[i] as f64).abs().max(1e-3);
            assert!(
                err <= 1e-4 * scale,
                "{m}x{n}x{k} [{i}]: got {} want {want}",
                got[i]
            );
        }
    }
}

#[test]
fn gemm_mode_from_env_parses_bf16() {
    for value in ["bf16", "BF16", "bfloat16"] {
        // SAFETY: single-threaded test; the variable is removed below.
        unsafe { std::env::set_var("LIGHTNET_OXIDE_GEMM", value) };
        assert_eq!(GemmMode::from_env(), GemmMode::Bf16, "LIGHTNET_OXIDE_GEMM={value}");
        // SAFETY: see above.
        unsafe { std::env::remove_var("LIGHTNET_OXIDE_GEMM") };
    }
}

#[test]
fn gemm_mode_from_env_parses_fp8_and_fp4() {
    // `from_env` reads a process-wide variable, so exercise the mapping through
    // the documented spellings by comparing the parse result of a child-free
    // helper: the enum is `Copy`, so a scoped set/restore is enough.
    for (value, expect) in [
        ("fp8", GemmMode::Fp8),
        ("e4m3", GemmMode::Fp8),
        ("fp4", GemmMode::Fp4),
        ("nvfp4", GemmMode::Fp4),
        ("fp16", GemmMode::Fp16),
        ("tf32", GemmMode::Tf32),
        ("auto", GemmMode::Auto),
        ("nonsense", GemmMode::Fp32),
    ] {
        // SAFETY: single-threaded test; the variable is restored below.
        unsafe { std::env::set_var("LIGHTNET_OXIDE_GEMM", value) };
        assert_eq!(GemmMode::from_env(), expect, "LIGHTNET_OXIDE_GEMM={value}");
        // SAFETY: see above.
        unsafe { std::env::remove_var("LIGHTNET_OXIDE_GEMM") };
        assert_eq!(GemmMode::from_env(), GemmMode::Fp32);
    }
}

// ---------------------------------------------------------------------------
// FP8 backward (Phase B step 6)
// ---------------------------------------------------------------------------
//
// The C++ recipe (`v6-dev-fp8`) is asymmetric: the forward is e4m3 x e4m3, while
// dgrad is `W:E4M3 x dy:E5M2` and wgrad is `dy:E5M2 x col:E4M3`.  e5m2 has one
// mantissa bit less but 128x the dynamic range, which is what gradients need.
// `with_fp8_backward(Some((E4M3, E5M2)))` (or `LIGHTNET_OXIDE_GEMM_BWD=fp8`)
// switches just the two backward GEMMs; the forward keeps its own mode.
//
// The checks are against the *device f32* backward (same kernels, fp32 GEMM), so
// what is measured is exactly the quantisation error of the FP8 pair.

/// Row norm `sum_k |a_ik| |b_kj|` is not directly available for conv, so the
/// tolerance is scaled by the magnitude of the f32 result's own row: the test
/// uses data with a positive mean so the dot products do not cancel, and then
/// bounds the relative error per element.
fn relative_error(got: &[f32], want: &[f32], label: &str) -> f64 {
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let scale = w.abs().max(1e-3) as f64;
        let err = (*g as f64 - *w as f64).abs() / scale;
        worst = worst.max(err);
        assert!(
            err <= 0.25,
            "{label}[{i}]: fp8 {g} vs f32 {w}, relative {err:.4}"
        );
    }
    worst
}

#[test]
fn fp8_backward_dgrad_and_wgrad_track_the_f32_backward() {
    let be = OxideBackend::new().expect("backend");
    let fp8 = OxideBackend::new().expect("backend").with_fp8_backward(Some((Fp8Format::E4M3, Fp8Format::E5M2)));
    assert_eq!(fp8.fp8_backward_formats(), Some((Fp8Format::E4M3, Fp8Format::E5M2)));
    assert_eq!(be.fp8_backward_formats(), None, "default stays off");

    let cfg = ElementwiseLaunchConfig::default();
    let mut rng = Rng(0xdead_beef);
    // batch 2, c 32 -> f 64 on a 16x16 input, 3x3, stride 2, pad 1.  The
    // geometry is deliberately above `TINY_MACS` with `groups != c`, so
    // `ConvMode::Auto` picks the GEMM arm and the backward really goes through
    // the FP8 GEMM seam rather than the direct elementwise kernel.
    let (batch, c, h, w, f, k, sx, sy, pad) = (2usize, 32, 16, 16, 64, 3, 2, 2, 1);
    let oh = (h + 2 * pad - k) / sy + 1;
    let ow = (w + 2 * pad - k) / sx + 1;
    let x: Vec<f32> = (0..batch * c * h * w).map(|_| rng.next(0.2, 1.0)).collect();
    let dy: Vec<f32> = (0..batch * f * oh * ow).map(|_| rng.next(0.2, 1.0)).collect();
    let wt: Vec<f32> = (0..f * c * k * k).map(|_| rng.next(0.2, 1.0)).collect();
    let tx = be.upload_f32(&[batch, c, h, w], &x).unwrap();
    let tdy = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
    let twt = be.upload_f32(&[f, c, k, k], &wt).unwrap();

    let dx32 = download(&be, &be.conv2d_backward_input_f32(&tdy, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let dx8 = download(&fp8, &fp8.conv2d_backward_input_f32(&tdy, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let worst_d = relative_error(&dx8, &dx32, "dgrad");
    assert!(worst_d > 0.0, "the FP8 dgrad must actually differ from the f32 one");

    let dw32 = download(&be, &be.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let dw8 = download(&fp8, &fp8.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let worst_w = relative_error(&dw8, &dw32, "wgrad");
    assert!(worst_w > 0.0, "the FP8 wgrad must actually differ from the f32 one");
    println!("fp8 backward worst relative error: dgrad {worst_d:.4}, wgrad {worst_w:.4}");
}

#[test]
fn fp8_backward_quantises_dy_once() {
    // dgrad (dy = operand B) and wgrad (dy = operand A) quantise the same f32
    // buffer to the same format; one conv backward must quantise it once.
    // Reference: two identical copies of dy in distinct buffers, which cannot
    // share, give the previous behaviour (one quantise per GEMM).
    let fp8 = || {
        OxideBackend::new()
            .expect("backend")
            .with_gemm_mode(GemmMode::Fp8)
            .with_fp8_backward(Some((Fp8Format::E4M3, Fp8Format::E5M2)))
    };
    let cfg = ElementwiseLaunchConfig::default();
    let mut rng = Rng(0x5ba_2ed1);
    let (batch, c, h, w, f, k, sx, sy, pad) = (2usize, 32, 16, 16, 64, 3, 2, 2, 1);
    let oh = (h + 2 * pad - k) / sy + 1;
    let ow = (w + 2 * pad - k) / sx + 1;
    let x: Vec<f32> = (0..batch * c * h * w).map(|_| rng.next(0.2, 1.0)).collect();
    let dy: Vec<f32> = (0..batch * f * oh * ow).map(|_| rng.next(0.2, 1.0)).collect();
    let wt: Vec<f32> = (0..f * c * k * k).map(|_| rng.next(0.2, 1.0)).collect();

    let run = |be: &OxideBackend, dy_w: &OxideTensorF32, dy_d: &OxideTensorF32| {
        let tx = be.upload_f32(&[batch, c, h, w], &x).unwrap();
        let twt = be.upload_f32(&[f, c, k, k], &wt).unwrap();
        // Warm the weight cache (with a throwaway dy) so only the dy
        // quantisations of the measured pair are counted.
        let warm = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
        be.conv2d_backward_input_f32(&warm, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap();
        let before = be.fp8_quantize_launches();
        let dw = be.conv2d_backward_weights_f32(&tx, dy_w, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap();
        let dx = be.conv2d_backward_input_f32(dy_d, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap();
        (download(be, &dx), download(be, &dw), be.fp8_quantize_launches() - before)
    };

    let be = fp8();
    let shared = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
    let (dx_s, dw_s, n_shared) = run(&be, &shared, &shared);

    let be = fp8();
    let copy_w = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
    let copy_d = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
    let (dx_r, dw_r, n_separate) = run(&be, &copy_w, &copy_d);

    assert_eq!(n_separate, 2, "reference: one dy quantise per GEMM");
    assert_eq!(n_shared, 1, "dy must be quantised once per conv backward");
    assert_eq!(dx_s, dx_r, "dgrad changed bitwise");
    assert_eq!(dw_s, dw_r, "wgrad changed bitwise");
}

#[test]
fn fp8_backward_shares_dy_under_delayed_scaling() {
    let be = OxideBackend::new()
        .expect("backend")
        .with_gemm_mode(GemmMode::Fp8)
        .with_fp8_scaling(Fp8Scaling::Delayed)
        .with_fp8_backward(Some((Fp8Format::E4M3, Fp8Format::E5M2)));
    let cfg = ElementwiseLaunchConfig::default();
    let mut rng = Rng(0xde1_a7ed);
    let (batch, c, h, w, f, k, sx, sy, pad) = (2usize, 32, 16, 16, 64, 3, 2, 2, 1);
    let oh = (h + 2 * pad - k) / sy + 1;
    let ow = (w + 2 * pad - k) / sx + 1;
    let tx = be.upload_f32(&[batch, c, h, w], &(0..batch * c * h * w).map(|_| rng.next(0.2, 1.0)).collect::<Vec<_>>()).unwrap();
    let twt = be.upload_f32(&[f, c, k, k], &(0..f * c * k * k).map(|_| rng.next(0.2, 1.0)).collect::<Vec<_>>()).unwrap();
    let dy: Vec<f32> = (0..batch * f * oh * ow).map(|_| rng.next(0.2, 1.0)).collect();
    let step = || {
        // A fresh gradient tensor per step, as in training.
        let tdy = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
        let before = be.fp8_quantize_launches();
        let dw = download(&be, &be.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
        let dx = download(&be, &be.conv2d_backward_input_f32(&tdy, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
        assert!(dw.iter().chain(&dx).all(|v| v.is_finite()));
        be.fp8_quantize_launches() - before
    };
    step(); // history empty, weight cache cold
    assert_eq!(step(), 1, "one keyed dy quantise per backward");
    assert_eq!(step(), 1, "one keyed dy quantise per backward");
}

#[test]
fn fp8_gemm_does_not_read_back_to_host() {
    // The fp8 GEMM derives its per-tensor scales on the device; a host
    // round trip per call would stall the stream once per operand.
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (256, 256, 256);
    let mut rng = Rng(0xf8_5ca1_e0);
    let a = rng.vec(m * k, -2.0, 2.0);
    let b = rng.vec(k * n, -2.0, 2.0);
    let c0 = vec![0.0f32; m * n];
    let ta = be.upload_f32(&[m, k], &a).unwrap();
    let tb = be.upload_f32(&[k, n], &b).unwrap();
    let tc = be.upload_f32(&[m, n], &c0).unwrap();
    let before = be.host_readbacks();
    let _out = be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 1.0, 0.0).unwrap();
    assert_eq!(be.host_readbacks() - before, 0, "fp8 GEMM read data back to host");
}

#[test]
fn fp8_pair_quantise_ignores_elements_beyond_the_gemm_extent() {
    // The fused A+B quantiser splits the operands at their GEMM extents, not
    // at the buffer lengths: a longer buffer must give the same bytes and
    // scale as an exact one, and its tail must not leak into the amax.
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (48, 40, 96);
    let mut rng = Rng(0x9a1_4e7);
    let a = rng.vec(m * k, -1.0, 1.0);
    let b = rng.vec(k * n, -1.0, 1.0);
    let c0 = vec![0.0f32; m * n];
    let run = |a_in: &[f32], b_in: &[f32]| {
        let ta = be.upload_f32(&[a_in.len()], a_in).unwrap();
        let tb = be.upload_f32(&[b_in.len()], b_in).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        download(&be, &be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 1.0, 0.0).unwrap())
    };
    let exact = run(&a, &b);
    let mut a_long = a.clone();
    a_long.extend(std::iter::repeat_n(1.0e6f32, 300));
    let mut b_long = b.clone();
    b_long.extend(std::iter::repeat_n(1.0e6f32, 300));
    let padded = run(&a_long, &b_long);
    assert_eq!(exact, padded, "tail beyond the extent changed the result");
}

#[test]
fn fp8_weight_cache_hits_on_second_forward() {
    // The weight operand is quantised once and served from the cache; only
    // the activation operand is quantised on the second forward.
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (64, 96, 128);
    let mut rng = Rng(0x7e1_9a7);
    let w = be.upload_f32(&[m, k], &rng.vec(m * k, -1.0, 1.0)).unwrap();
    let x = be.upload_f32(&[k, n], &rng.vec(k * n, -1.0, 1.0)).unwrap();
    let c = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();

    let before = be.fp8_quantize_launches();
    be.gemm_f32_weights_a(&w, &x, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    let first = be.fp8_quantize_launches() - before;

    let before = be.fp8_quantize_launches();
    be.gemm_f32_weights_a(&w, &x, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    let second = be.fp8_quantize_launches() - before;

    assert_eq!(first, 2, "first forward: weight miss + activation");
    assert_eq!(second, 1, "second forward: activation only, weights cached");
}

#[test]
fn fp8_weight_cache_refreshes_after_update() {
    // An optimizer step returns a new weight tensor. The old tensor is
    // dropped, so its cache entry must never be served for the new values.
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (64, 96, 128);
    let mut rng = Rng(0x0_9da7e);
    let x = be.upload_f32(&[k, n], &rng.vec(k * n, -1.0, 1.0)).unwrap();
    let c = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();

    let w_old = be.upload_f32(&[m, k], &rng.vec(m * k, -1.0, 1.0)).unwrap();
    let out_old = be.gemm_f32_weights_a(&w_old, &x, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    let out_old = download(&be, &out_old);
    drop(w_old);

    let w_new = be.upload_f32(&[m, k], &rng.vec(m * k, -1.0, 1.0)).unwrap();
    let before = be.fp8_quantize_launches();
    let out_new = be.gemm_f32_weights_a(&w_new, &x, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    let launches = be.fp8_quantize_launches() - before;
    let out_new = download(&be, &out_new);

    assert_eq!(launches, 2, "updated weights were not requantised");
    assert_ne!(out_old, out_new, "output did not change with the weights");
    let want = download(&be, &be.gemm_f32(&w_new, &x, &c, m, n, k, false, false, 1.0, 0.0).unwrap());
    assert_eq!(out_new, want, "cached weights differ from the uncached fp8 path");
}

#[test]
fn fp8_cached_weights_serve_forward_and_transposed_dgrad() {
    // One cached copy of the weights must match the uncached fp8 GEMM for both
    // the plain (forward) and the transposed (dgrad) use.
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (48, 40, 64);
    let mut rng = Rng(0xd9ad_c4c4);
    let w = be.upload_f32(&[m, k], &rng.vec(m * k, -1.0, 1.0)).unwrap();
    let c = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();
    let x_fwd = be.upload_f32(&[k, n], &rng.vec(k * n, -1.0, 1.0)).unwrap();
    let x_bwd = be.upload_f32(&[m, n], &rng.vec(m * n, -1.0, 1.0)).unwrap();

    let fwd_cached = be.gemm_f32_weights_a(&w, &x_fwd, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    let fwd_plain = be.gemm_f32(&w, &x_fwd, &c, m, n, k, false, false, 1.0, 0.0).unwrap();
    assert_eq!(download(&be, &fwd_cached), download(&be, &fwd_plain), "forward");

    // dgrad: dcol (k x n) = W^T (k x m) * dy (m x n), with W stored m x k.
    let c_bwd = be.upload_f32(&[k, n], &vec![0.0f32; k * n]).unwrap();
    let dg_cached = be.gemm_f32_weights_a(&w, &x_bwd, &c_bwd, k, n, m, true, false, 1.0, 0.0).unwrap();
    let dg_plain = be.gemm_f32(&w, &x_bwd, &c_bwd, k, n, m, true, false, 1.0, 0.0).unwrap();
    assert_eq!(download(&be, &dg_cached), download(&be, &dg_plain), "transposed dgrad");
}

// ---- Delayed scaling (plan_fp8 Task 3) ------------------------------------

#[test]
fn fp8_scaling_defaults_to_current_and_parses_env_spelling() {
    assert_eq!(Fp8Scaling::from_name(""), Fp8Scaling::Current);
    assert_eq!(Fp8Scaling::from_name("current"), Fp8Scaling::Current);
    assert_eq!(Fp8Scaling::from_name("Delayed"), Fp8Scaling::Delayed);
    assert_eq!(Fp8Scaling::from_name("bogus"), Fp8Scaling::Current);
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    assert_eq!(be.fp8_scaling(), Fp8Scaling::Delayed);
}

#[test]
fn delayed_scaling_matches_current_scaling_on_stationary_data() {
    let cur = OxideBackend::new().expect("backend");
    let del = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0xde1a_7ed);
    let x = rng.vec(3000, -3.0, 3.0);
    let tx_cur = cur.upload_f32(&[x.len()], &x).unwrap();
    let tx_del = del.upload_f32(&[x.len()], &x).unwrap();
    let (want_bytes, want_scale) = cur.fp8_quantize_probe(&tx_cur, Fp8Format::E4M3, None).unwrap();
    for call in 0..20 {
        let (bytes, scale) = del.fp8_quantize_probe(&tx_del, Fp8Format::E4M3, None).unwrap();
        // Same tensor every call: the history max equals the tensor's amax, so
        // the factor, and with it every byte, matches current scaling.  (The
        // plan allows one e4m3 ulp; stationary data gives exact equality.)
        assert_eq!(scale, want_scale, "call {call}: scale pair");
        assert_eq!(bytes, want_bytes, "call {call}: bytes");
    }

    // Same through the GEMM: 20 calls on fixed operands, outputs equal current.
    let cur = cur.with_gemm_mode(GemmMode::Fp8);
    let del = del.with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (64, 48, 96);
    let a = rng.vec(m * k, -2.0, 2.0);
    let b = rng.vec(k * n, -2.0, 2.0);
    let c0 = vec![0.0f32; m * n];
    let run = |be: &OxideBackend, ta: &OxideTensorF32, tb: &OxideTensorF32, tc: &OxideTensorF32| {
        download(be, &be.gemm_f32(ta, tb, tc, m, n, k, false, false, 1.0, 0.0).unwrap())
    };
    let (ca, cb, cc) = (
        cur.upload_f32(&[m, k], &a).unwrap(),
        cur.upload_f32(&[k, n], &b).unwrap(),
        cur.upload_f32(&[m, n], &c0).unwrap(),
    );
    let (da, db, dc) = (
        del.upload_f32(&[m, k], &a).unwrap(),
        del.upload_f32(&[k, n], &b).unwrap(),
        del.upload_f32(&[m, n], &c0).unwrap(),
    );
    let want = run(&cur, &ca, &cb, &cc);
    for call in 0..20 {
        assert_eq!(run(&del, &da, &db, &dc), want, "gemm call {call}");
    }
}

#[test]
fn delayed_scaling_saturates_on_a_spike() {
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0x5b1_4e);
    let base = rng.vec(1024, -1.0, 1.0);
    let amax = base.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // One explicit key stands for one call site whose buffer changes.
    let key = Some(0x5eed_u64);
    for _ in 0..4 {
        let t = be.upload_f32(&[base.len()], &base).unwrap();
        be.fp8_quantize_probe(&t, Fp8Format::E4M3, key).unwrap();
    }
    let mut spiked = base.clone();
    spiked[7] = 100.0 * amax;
    spiked[8] = -100.0 * amax;
    let t = be.upload_f32(&[spiked.len()], &spiked).unwrap();
    let (bytes, scale) = be.fp8_quantize_probe(&t, Fp8Format::E4M3, key).unwrap();
    let back = |b: u8| fp8_to_f32(b, Fp8Format::E4M3);
    assert_eq!(back(bytes[7]), 448.0, "positive spike saturates to max finite");
    assert_eq!(back(bytes[8]), -448.0, "negative spike saturates to -max finite");
    assert!(bytes.iter().all(|&b| back(b).is_finite()), "NaN or inf in the output");
    assert!((scale[0] - 448.0 / amax).abs() <= 1e-3 * scale[0], "factor comes from the history, not the spike");

    // The spike enters the history for the next call.
    let (_, next) = be.fp8_quantize_probe(&t, Fp8Format::E4M3, key).unwrap();
    assert!((next[0] - 448.0 / (100.0 * amax)).abs() <= 1e-3 * next[0], "history picked up the spike");
}

#[test]
fn delayed_scaling_first_call_uses_current_amax() {
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0xf1_257);
    let x = rng.vec(700, -5.0, 5.0);
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let t = be.upload_f32(&[x.len()], &x).unwrap();
    let (bytes, scale) = be.fp8_quantize_probe(&t, Fp8Format::E4M3, Some(1)).unwrap();
    let factor = 448.0 / amax;
    assert!((scale[0] - factor).abs() <= 1e-6 * factor, "factor {} want {factor}", scale[0]);
    assert!((scale[1] * scale[0] - 1.0).abs() <= 1e-6);
    for (i, (&b, &v)) in bytes.iter().zip(&x).enumerate() {
        assert_eq!(b, f32_to_fp8(v * scale[0], Fp8Format::E4M3), "byte {i}");
    }
    // The first call recorded its amax: a second call with a tiny tensor still
    // scales by the first tensor's amax.
    // (Same length: the key includes the element count.)
    let tiny = vec![0.001f32; x.len()];
    let tt = be.upload_f32(&[x.len()], &tiny).unwrap();
    let (_, s2) = be.fp8_quantize_probe(&tt, Fp8Format::E4M3, Some(1)).unwrap();
    assert!((s2[0] - factor).abs() <= 1e-6 * factor, "second call uses history {} want {factor}", s2[0]);
}

#[test]
fn delayed_scaling_keeps_weights_on_current_scaling_and_forgets_after_16_calls() {
    // Weights go through the cache with current scaling even in Delayed mode.
    let cur = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let del = OxideBackend::new()
        .expect("backend")
        .with_gemm_mode(GemmMode::Fp8)
        .with_fp8_scaling(Fp8Scaling::Delayed);
    let (m, n, k) = (64, 96, 128);
    let mut rng = Rng(0x77_aa);
    let w = rng.vec(m * k, -1.0, 1.0);
    let x = rng.vec(k * n, -1.0, 1.0);
    let c0 = vec![0.0f32; m * n];
    let go = |be: &OxideBackend| {
        let tw = be.upload_f32(&[m, k], &w).unwrap();
        let tx = be.upload_f32(&[k, n], &x).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        download(be, &be.gemm_f32_weights_a(&tw, &tx, &tc, m, n, k, false, false, 1.0, 0.0).unwrap())
    };
    // First call: empty history, so the activation also uses its own amax.
    assert_eq!(go(&cur), go(&del));

    // A spike ages out of the 16-call window: after 16 calls with small data
    // the factor returns to the small tensor's own amax.
    let be = &del;
    let small = vec![1.0f32; 256];
    let big = vec![50.0f32; 256];
    let key = Some(9u64);
    let t = be.upload_f32(&[256], &big).unwrap();
    be.fp8_quantize_probe(&t, Fp8Format::E4M3, key).unwrap();
    let t = be.upload_f32(&[256], &small).unwrap();
    let mut last = [0.0f32; 2];
    for _ in 0..20 {
        last = be.fp8_quantize_probe(&t, Fp8Format::E4M3, key).unwrap().1;
    }
    assert!((last[0] - 448.0).abs() <= 1e-3, "old spike aged out, factor {}", last[0]);
}

/// Quantising 16-bit storage directly must give the bytes and scale pair that
/// widening to f32 first gives (the widening is exact, so nothing may differ).
/// The data holds a spike that overflows f16 to inf and a zero run.
fn fp8_from_half_matches_widened(hfmt: HalfFormat) {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0x16b1_7f8 + hfmt.code() as u64);
    let mut x = rng.vec(5000, -3.0, 3.0);
    x[11] = 1.0e6; // inf in f16, finite in bf16
    x[12] = -1.0e-9;
    x[100..140].fill(0.0);
    let xh = be.to_half(&be.upload_f32(&[x.len()], &x).unwrap(), hfmt).unwrap();
    let wide = be.from_half(&xh, hfmt).unwrap();
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let (want_bytes, want_scale) = be.fp8_quantize_probe(&wide, fmt, None).unwrap();
        let (bytes, scale) = be.fp8_quantize_half_probe(&xh, fmt, hfmt).unwrap();
        assert_eq!(scale.map(f32::to_bits), want_scale.map(f32::to_bits), "{fmt:?} {hfmt:?}: scale pair");
        assert_eq!(bytes, want_bytes, "{fmt:?} {hfmt:?}: bytes");
    }

    // Through the GEMM: half operands against the same values widened into the
    // f32 fp8 selector, all four transposes, odd shapes.
    let fp8 = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
        let (m, n, k) = (65usize, 33usize, 129usize);
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let ah = be.to_half(&be.upload_f32(&[m, k], &a).unwrap(), hfmt).unwrap();
        let bh = be.to_half(&be.upload_f32(&[k, n], &b).unwrap(), hfmt).unwrap();
        let (aw, bw) = (be.from_half(&ah, hfmt).unwrap(), be.from_half(&bh, hfmt).unwrap());
        let got = download(
            &be,
            &be.gemm_fp8_half(&ah, &bh, m, n, k, ta, tb, 0.7, Fp8Format::E4M3, Fp8Format::E4M3, hfmt).unwrap(),
        );
        let c0 = fp8.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();
        let (taw, tbw) = (fp8.upload_f32(&[m, k], &download(&be, &aw)).unwrap(), fp8.upload_f32(&[k, n], &download(&be, &bw)).unwrap());
        let want = download(&fp8, &fp8.gemm_f32(&taw, &tbw, &c0, m, n, k, ta, tb, 0.7, 0.0).unwrap());
        assert_eq!(got, want, "{hfmt:?} gemm ta={ta} tb={tb}");
    }
}

#[test]
fn fp8_from_f16_matches_fp8_from_widened_f32() {
    fp8_from_half_matches_widened(HalfFormat::F16);
}

#[test]
fn fp8_from_bf16_matches_fp8_from_widened_f32() {
    fp8_from_half_matches_widened(HalfFormat::Bf16);
}

// ---- Delayed scaling on 16-bit sources ------------------------------------

fn half_tensor(be: &OxideBackend, x: &[f32], hfmt: HalfFormat) -> lightnet_cuda_oxide::OxideTensorF16 {
    be.to_half(&be.upload_f32(&[x.len()], x).unwrap(), hfmt).unwrap()
}

/// Same tensor every call: the history max is the tensor's amax, so after the
/// call-0 current-scaling fallback and through the wrap of the 16-slot history
/// every call gives current scaling's bytes and scale pair.
fn delayed_half_matches_current_on_stationary_data(hfmt: HalfFormat) {
    let cur = OxideBackend::new().expect("backend");
    let del = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0xde1a_7ed + hfmt.code() as u64);
    let x = rng.vec(3000, -3.0, 3.0);
    let (hc, hd) = (half_tensor(&cur, &x, hfmt), half_tensor(&del, &x, hfmt));
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let (want_bytes, want_scale) = cur.fp8_quantize_half_probe(&hc, fmt, hfmt).unwrap();
        for call in 0..20 {
            let (bytes, scale) = del.fp8_quantize_half_probe(&hd, fmt, hfmt).unwrap();
            assert_eq!(scale.map(f32::to_bits), want_scale.map(f32::to_bits), "{hfmt:?} {fmt:?} call {call}: scale");
            assert_eq!(bytes, want_bytes, "{hfmt:?} {fmt:?} call {call}: bytes");
        }
    }
}

#[test]
fn delayed_half_matches_current_on_stationary_data_f16() {
    delayed_half_matches_current_on_stationary_data(HalfFormat::F16);
}

#[test]
fn delayed_half_matches_current_on_stationary_data_bf16() {
    delayed_half_matches_current_on_stationary_data(HalfFormat::Bf16);
}

fn delayed_half_saturates_on_a_spike(hfmt: HalfFormat) {
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0x5b1_4e + hfmt.code() as u64);
    let base = rng.vec(1024, -1.0, 1.0);
    let amax = base.iter().fold(0.0f32, |m, v| m.max(half_round(hfmt, *v).abs()));
    let key = Some(0x5eed_u64);
    for _ in 0..4 {
        be.fp8_quantize_half_probe_keyed(&half_tensor(&be, &base, hfmt), Fp8Format::E4M3, hfmt, key).unwrap();
    }
    let mut spiked = base.clone();
    spiked[7] = 100.0 * amax;
    spiked[8] = -100.0 * amax;
    let t = half_tensor(&be, &spiked, hfmt);
    let (bytes, scale) = be.fp8_quantize_half_probe_keyed(&t, Fp8Format::E4M3, hfmt, key).unwrap();
    let back = |b: u8| fp8_to_f32(b, Fp8Format::E4M3);
    assert_eq!(back(bytes[7]), 448.0, "{hfmt:?}: positive spike saturates to max finite");
    assert_eq!(back(bytes[8]), -448.0, "{hfmt:?}: negative spike saturates to -max finite");
    assert!(bytes.iter().all(|&b| back(b).is_finite()), "{hfmt:?}: NaN or inf in the output");
    assert!((scale[0] - 448.0 / amax).abs() <= 1e-3 * scale[0], "{hfmt:?}: factor comes from the history");
    let (_, next) = be.fp8_quantize_half_probe_keyed(&t, Fp8Format::E4M3, hfmt, key).unwrap();
    assert!((next[0] - 448.0 / (100.0 * amax)).abs() <= 2e-3 * next[0], "{hfmt:?}: history picked up the spike");
}

fn half_round(hfmt: HalfFormat, v: f32) -> f32 {
    use lightnet_cuda_oxide::half::*;
    match hfmt {
        HalfFormat::F16 => f16_bits_to_f32(f32_to_f16_bits(v)),
        HalfFormat::Bf16 => bf16_bits_to_f32(f32_to_bf16_bits(v)),
    }
}

#[test]
fn delayed_half_saturates_on_a_spike_f16() {
    delayed_half_saturates_on_a_spike(HalfFormat::F16);
}

#[test]
fn delayed_half_saturates_on_a_spike_bf16() {
    delayed_half_saturates_on_a_spike(HalfFormat::Bf16);
}

fn delayed_half_first_call_uses_current_amax(hfmt: HalfFormat) {
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0xf1_257 + hfmt.code() as u64);
    let x = rng.vec(700, -5.0, 5.0);
    let amax = x.iter().fold(0.0f32, |m, v| m.max(half_round(hfmt, *v).abs()));
    let t = half_tensor(&be, &x, hfmt);
    let (bytes, scale) = be.fp8_quantize_half_probe_keyed(&t, Fp8Format::E4M3, hfmt, Some(1)).unwrap();
    let factor = 448.0 / amax;
    assert!((scale[0] - factor).abs() <= 1e-6 * factor, "{hfmt:?}: factor {} want {factor}", scale[0]);
    assert!((scale[1] * scale[0] - 1.0).abs() <= 1e-6);
    for (i, (&b, &v)) in bytes.iter().zip(&x).enumerate() {
        assert_eq!(b, f32_to_fp8(half_round(hfmt, v) * scale[0], Fp8Format::E4M3), "{hfmt:?} byte {i}");
    }
    // Call 0 recorded its amax: a tiny tensor of the same length still scales by it.
    let tt = half_tensor(&be, &vec![0.001f32; x.len()], hfmt);
    let (_, s2) = be.fp8_quantize_half_probe_keyed(&tt, Fp8Format::E4M3, hfmt, Some(1)).unwrap();
    assert!((s2[0] - factor).abs() <= 1e-6 * factor, "{hfmt:?}: second call uses history {} want {factor}", s2[0]);
}

#[test]
fn delayed_half_first_call_uses_current_amax_f16() {
    delayed_half_first_call_uses_current_amax(HalfFormat::F16);
}

#[test]
fn delayed_half_first_call_uses_current_amax_bf16() {
    delayed_half_first_call_uses_current_amax(HalfFormat::Bf16);
}

/// The pool recycles an address for a new tensor: delayed scaling must start a
/// fresh history instead of inheriting the previous owner's amax.
fn delayed_half_forgets_history_when_pool_recycles_address(hfmt: HalfFormat) {
    let be = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let n = 1024;
    let first = half_tensor(&be, &vec![8.0f32; n], hfmt);
    let ptr = first.device_ptr();
    let (_, s1) = be.fp8_quantize_half_probe(&first, Fp8Format::E4M3, hfmt).unwrap();
    let first_amax = half_round(hfmt, 8.0);
    assert!((s1[0] - 448.0 / first_amax).abs() <= 1e-6 * s1[0], "{hfmt:?}: first history");
    drop(first);

    let second = half_tensor(&be, &vec![0.01f32; n], hfmt);
    assert_eq!(second.device_ptr(), ptr, "{hfmt:?}: pool did not recycle the address");
    let before = be.fp8_amax_passes();
    let (_, s2) = be.fp8_quantize_half_probe(&second, Fp8Format::E4M3, hfmt).unwrap();
    assert_eq!(be.fp8_amax_passes() - before, 1, "{hfmt:?}: recycled buffer must start at call 0");
    let second_amax = half_round(hfmt, 0.01);
    assert!((s2[0] - 448.0 / second_amax).abs() <= 1e-6 * s2[0], "{hfmt:?}: second history");
}

#[test]
fn delayed_half_forgets_history_when_pool_recycles_address_f16() {
    delayed_half_forgets_history_when_pool_recycles_address(HalfFormat::F16);
}

#[test]
fn delayed_half_forgets_history_when_pool_recycles_address_bf16() {
    delayed_half_forgets_history_when_pool_recycles_address(HalfFormat::Bf16);
}

/// After call 0 the delayed path is one launch: no stand-alone amax pass, one
/// quantise.  Current scaling runs the amax pass on every call.
fn delayed_half_quantises_in_one_launch(hfmt: HalfFormat) {
    let cur = OxideBackend::new().expect("backend");
    let del = OxideBackend::new().expect("backend").with_fp8_scaling(Fp8Scaling::Delayed);
    let mut rng = Rng(0x01e_1a + hfmt.code() as u64);
    let x = rng.vec(2048, -2.0, 2.0);
    let (hc, hd) = (half_tensor(&cur, &x, hfmt), half_tensor(&del, &x, hfmt));
    for call in 0..6u64 {
        let (a0, q0) = (del.fp8_amax_passes(), del.fp8_quantize_launches());
        del.fp8_quantize_half_probe(&hd, Fp8Format::E4M3, hfmt).unwrap();
        assert_eq!(del.fp8_quantize_launches() - q0, 1, "{hfmt:?} call {call}: quantise launches");
        let want_amax = if call == 0 { 1 } else { 0 };
        assert_eq!(del.fp8_amax_passes() - a0, want_amax, "{hfmt:?} call {call}: stand-alone amax passes");

        let (a0, q0) = (cur.fp8_amax_passes(), cur.fp8_quantize_launches());
        cur.fp8_quantize_half_probe(&hc, Fp8Format::E4M3, hfmt).unwrap();
        assert_eq!(cur.fp8_amax_passes() - a0, 1, "{hfmt:?} call {call}: current scaling is two-pass");
        assert_eq!(cur.fp8_quantize_launches() - q0, 1);
    }
}

#[test]
fn delayed_half_quantises_in_one_launch_f16() {
    delayed_half_quantises_in_one_launch(HalfFormat::F16);
}

#[test]
fn delayed_half_quantises_in_one_launch_bf16() {
    delayed_half_quantises_in_one_launch(HalfFormat::Bf16);
}

// ---- Non-finite propagation and tiny amax (codex review, v0.2.188 / v0.2.190) --

const NAN_BYTE: u8 = 0x7f;

/// Data with finite values plus NaN / +-Inf entries (and a finite 16-bit-safe
/// spike), quantised through every GEMM quantiser: non-finite input must reach
/// the byte stream as the NaN code (so the GEMM output is non-finite and the
/// loss scaler sees it), while amax and the scale pair keep ignoring it.
#[test]
fn fp8_quantisers_propagate_non_finite_input() {
    for scaling in [Fp8Scaling::Current, Fp8Scaling::Delayed] {
        let be = OxideBackend::new().expect("backend").with_fp8_scaling(scaling);
        let mut rng = Rng(0x0ff1_ce);
        let mut x = rng.vec(600, -2.0, 2.0);
        x[3] = f32::INFINITY;
        x[4] = f32::NEG_INFINITY;
        x[5] = f32::NAN;
        x[299] = 1.0e3; // finite spike: saturates, never NaN
        let amax = x.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
            let key = Some(0xbad_u64 + fmt.code() as u64);
            let t = be.upload_f32(&[x.len()], &x).unwrap();
            let (bytes, scale) = be.fp8_quantize_probe(&t, fmt, key).unwrap();
            let tag = format!("f32 {scaling:?} {fmt:?}");
            for i in [3usize, 4, 5] {
                assert!(fp8_to_f32(bytes[i], fmt).is_nan(), "{tag}: element {i} must be NaN, byte {:#x}", bytes[i]);
            }
            for (i, &b) in bytes.iter().enumerate() {
                if ![3, 4, 5].contains(&i) {
                    assert!(fp8_to_f32(b, fmt).is_finite(), "{tag}: finite input {i} became non-finite");
                }
            }
            if scaling == Fp8Scaling::Current {
                assert!((scale[0] - fmt_max(fmt) / amax).abs() <= 1e-3 * scale[0], "{tag}: amax must ignore non-finite");
            }
            for hfmt in [HalfFormat::F16, HalfFormat::Bf16] {
                let xh = half_tensor(&be, &x, hfmt);
                let (hb, _) = be.fp8_quantize_half_probe_keyed(&xh, fmt, hfmt, key.map(|k| k + 77)).unwrap();
                for i in [3usize, 4, 5] {
                    assert!(fp8_to_f32(hb[i], fmt).is_nan(), "{tag} {hfmt:?}: element {i} must be NaN, byte {:#x}", hb[i]);
                }
                assert!(
                    hb.iter().enumerate().all(|(i, &b)| [3, 4, 5].contains(&i) || fp8_to_f32(b, fmt).is_finite()),
                    "{tag} {hfmt:?}: finite input became non-finite"
                );
            }
        }
    }
}

fn fmt_max(fmt: Fp8Format) -> f32 {
    match fmt {
        Fp8Format::E4M3 => 448.0,
        Fp8Format::E5M2 => 57344.0,
    }
}

/// The two-operand (pair) quantiser of the f32 GEMM selector propagates too.
#[test]
fn fp8_gemm_with_inf_operand_is_non_finite() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (16usize, 16usize, 32usize);
    let mut rng = Rng(0x1ff);
    let mut a = rng.vec(m * k, -1.0, 1.0);
    let b = rng.vec(k * n, -1.0, 1.0);
    a[0] = f32::INFINITY;
    let (ta, tb) = (be.upload_f32(&[m, k], &a).unwrap(), be.upload_f32(&[k, n], &b).unwrap());
    let c0 = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();
    let c = download(&be, &be.gemm_f32(&ta, &tb, &c0, m, n, k, false, false, 1.0, 0.0).unwrap());
    assert!(c[..n].iter().all(|v| !v.is_finite()), "row 0 holds the Inf operand: {:?}", &c[..4]);
    assert!(c[n..].iter().all(|v| v.is_finite()), "other rows must stay finite");
}


/// bf16 tensor whose values are ~1e-36 (and zeros): `448 / amax` overflows f32,
/// which used to make the factor Inf, the zeros 0 * Inf = NaN and the
/// reciprocal 0.  The factor must clamp to a finite value instead.
fn tiny_amax_gives_finite_scale_pair(hfmt: HalfFormat) {
    for scaling in [Fp8Scaling::Current, Fp8Scaling::Delayed] {
        let be = OxideBackend::new().expect("backend").with_fp8_scaling(scaling);
        let mut x = vec![0.0f32; 512];
        for (i, v) in x.iter_mut().enumerate() {
            if i % 3 != 0 {
                *v = if i % 2 == 0 { 1.0e-36 } else { -0.7e-36 };
            }
        }
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
            // several calls so the delayed path leaves its current-scaling start
            for call in 0..4 {
                let xh = half_tensor(&be, &x, hfmt);
                let (bytes, scale) = be.fp8_quantize_half_probe_keyed(&xh, fmt, hfmt, Some(0x71_47)).unwrap();
                let tag = format!("{hfmt:?} {scaling:?} {fmt:?} call {call}");
                assert!(scale[0].is_finite() && scale[0] > 0.0, "{tag}: factor {}", scale[0]);
                assert!(scale[1].is_finite() && scale[1] > 0.0, "{tag}: reciprocal {}", scale[1]);
                assert!(bytes.iter().all(|&b| fp8_to_f32(b, fmt).is_finite()), "{tag}: NaN or inf byte");
                assert!(bytes.iter().zip(&x).all(|(&b, &v)| v != 0.0 || b & 0x7f == 0), "{tag}: zeros stay zero");
            }
        }
    }
    // f32 source, same story
    let be = OxideBackend::new().expect("backend");
    let x: Vec<f32> = (0..256).map(|i| if i % 4 == 0 { 0.0 } else { 1.0e-36 }).collect();
    let t = be.upload_f32(&[x.len()], &x).unwrap();
    let (bytes, scale) = be.fp8_quantize_probe(&t, Fp8Format::E4M3, None).unwrap();
    assert!(scale[0].is_finite() && scale[1].is_finite() && scale[1] > 0.0, "f32 scale {scale:?}");
    assert!(bytes.iter().all(|&b| fp8_to_f32(b, Fp8Format::E4M3).is_finite()), "f32: NaN or inf byte");
    assert!(bytes.iter().zip(&x).any(|(&b, &v)| v != 0.0 && b & 0x7f != 0), "f32: tiny values flushed to zero");
}

#[test]
fn tiny_amax_gives_finite_scale_pair_bf16() {
    tiny_amax_gives_finite_scale_pair(HalfFormat::Bf16);
}

#[test]
fn tiny_amax_gives_finite_scale_pair_f16() {
    // f16 flushes 1e-36 to zero (amax 0 -> factor 1): still finite, no NaN.
    tiny_amax_gives_finite_scale_pair(HalfFormat::F16);
}

/// Through the GEMM: ~1e-36 operands give a finite, correct product.
#[test]
fn fp8_gemm_with_tiny_operands_is_finite() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let (m, n, k) = (16usize, 16usize, 32usize);
    let a = vec![1.0e-36f32; m * k];
    let b = vec![1.0f32; k * n];
    let (ta, tb) = (be.upload_f32(&[m, k], &a).unwrap(), be.upload_f32(&[k, n], &b).unwrap());
    let c0 = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();
    let c = download(&be, &be.gemm_f32(&ta, &tb, &c0, m, n, k, false, false, 1.0, 0.0).unwrap());
    let want = 1.0e-36f64 * k as f64;
    assert!(c.iter().all(|v| v.is_finite()), "non-finite output {:?}", &c[..4]);
    assert!(c.iter().all(|&v| ((v as f64) - want).abs() <= 0.1 * want), "got {:?}, want {want:e}", &c[..4]);
}

#[test]
fn fp8_backward_formats_follow_the_operand_role() {
    // Wgrad has `dy` as operand A and the im2col column as operand B.  The
    // byte-column path (`GemmMode::Fp8`) and the f32-column path (forward in
    // fp32, backward fp8) must quantise `dy` to e5m2 and the column to e4m3
    // alike, so they agree bitwise.  Formats following the operand position
    // would give `dy` e4m3 and the column e5m2 on the f32-column path only.
    let cfg = ElementwiseLaunchConfig::default();
    let mut rng = Rng(0x701e_5);
    let (batch, c, h, w, f, k, sx, sy, pad) = (2usize, 32, 16, 16, 64, 3, 2, 2, 1);
    let oh = (h + 2 * pad - k) / sy + 1;
    let ow = (w + 2 * pad - k) / sx + 1;
    let x: Vec<f32> = (0..batch * c * h * w).map(|_| rng.next(0.2, 1.0)).collect();
    let dy: Vec<f32> = (0..batch * f * oh * ow).map(|_| rng.next(0.2, 1.0)).collect();
    let fmts = Some((Fp8Format::E4M3, Fp8Format::E5M2));
    let wgrad = |be: &OxideBackend| {
        let tx = be.upload_f32(&[batch, c, h, w], &x).unwrap();
        let tdy = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
        download(be, &be.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap())
    };
    let bytes = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8).with_fp8_backward(fmts);
    let f32col = OxideBackend::new().expect("backend").with_fp8_backward(fmts);
    assert_eq!(wgrad(&f32col), wgrad(&bytes), "wgrad formats differ between the column paths");
}
