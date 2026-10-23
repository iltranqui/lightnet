//! Host-side reference for the packed NVFP4 format the kernels in
//! [`crate::kernels::quant_fp4`] and [`crate::kernels::gemm_fp4`] consume.
//!
//! The layout and the rounding rules are the device ones; this module exists so
//! `tests/oxide_gemm_fp4.rs` can check the GPU quantiser *bit for bit* against
//! an independent host implementation of the same C++ recipe
//! (`raptor-fp4:src-fp4sim/fp4_sim_kernels.cu`, `fp4_sim_quantize_1x16_gpu`).

use crate::fp8::{Fp8Format, amax_f32, f32_to_fp8, fp8_to_f32};

/// E2M1 grid magnitudes, indexed by bits 2:0 of a code.
pub const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;
/// Elements per E4M3 block scale along K.
pub const BLOCK: usize = 16;
/// Packed code bytes per block (two 4-bit codes per byte).
pub const CODE_BYTES_PER_BLOCK: usize = 8;

/// Packed NVFP4 operand: `[rows, k]` row-major f32 quantised along K.
///
/// `codes` holds `rows * ceil(k / 16) * 8` bytes (low nibble = the even K of
/// the pair), `scales` holds `rows * ceil(k / 16)` raw e4m3 bytes, and
/// `global_scale` is the f32 tensor scale the recipe divides by first.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp4Packed {
    pub rows: usize,
    pub k: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_scale: f32,
}

impl Fp4Packed {
    /// `ceil(k / 16)`: the number of E4M3 scale blocks per row.
    pub fn blocks_per_row(&self) -> usize {
        self.k.div_ceil(BLOCK)
    }

    /// Decode a single value (global scale included).
    pub fn value(&self, row: usize, kk: usize) -> f32 {
        let blocks = self.blocks_per_row();
        let block = kk / BLOCK;
        let byte = self.codes
            [row * blocks * CODE_BYTES_PER_BLOCK + block * CODE_BYTES_PER_BLOCK + (kk % 16) / 2];
        let code = if kk % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        e2m1_to_f32(code) * e4m3_magnitude(self.scales[row * blocks + block]) * self.global_scale
    }

    /// The full `[rows, k]` dequantised matrix.
    pub fn dequantized(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.rows * self.k);
        for row in 0..self.rows {
            for kk in 0..self.k {
                out.push(self.value(row, kk));
            }
        }
        out
    }
}

/// E2M1 code -> f32.
pub fn e2m1_to_f32(code: u8) -> f32 {
    let magnitude = E2M1_GRID[(code & 7) as usize];
    if code & 8 != 0 { -magnitude } else { magnitude }
}

/// f32 -> E2M1 code: sign bit plus the nearest grid magnitude, ties to the
/// even index, `|x|` clamped to 6, NaN encoded as `+6`.
pub fn f32_to_e2m1(value: f32) -> u8 {
    if value.is_nan() {
        return 0x07;
    }
    let sign: u8 = if value.to_bits() >> 31 != 0 {
        0x08
    } else {
        0x00
    };
    let magnitude = if value.is_infinite() {
        E2M1_MAX
    } else {
        let m = value.abs();
        if m > E2M1_MAX { E2M1_MAX } else { m }
    };
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (idx, grid) in E2M1_GRID.iter().enumerate() {
        let distance = (magnitude - grid).abs();
        if distance < best_distance || (distance == best_distance && (idx & 1) == 0) {
            best = idx;
            best_distance = distance;
        }
    }
    sign | best as u8
}

/// Raw e4m3 byte -> f32 (`code <= 0x7e`, which is all a scale byte can be).
pub fn e4m3_magnitude(code: u8) -> f32 {
    fp8_to_f32(code, Fp8Format::E4M3)
}

/// `amax / (6 * 448)`, or 1 for a zero / non-finite amax.
pub fn global_scale_from_amax(amax: f32) -> f32 {
    if amax > 0.0 && amax.is_finite() {
        amax / (E2M1_MAX * 448.0)
    } else {
        1.0
    }
}

/// Decoded e4m3 block scale for `block_amax` (`TwoLevel` mode).
pub fn block_scale(block_amax: f32, global_scale: f32) -> f32 {
    if !(block_amax > 0.0) || !(global_scale > 0.0) {
        return 1.0;
    }
    let raw = block_amax / (global_scale * E2M1_MAX);
    let clamped = if raw > 448.0 { 448.0 } else { raw };
    e4m3_magnitude(f32_to_fp8(clamped, Fp8Format::E4M3))
}

/// Row-major `[rows, k]` f32 -> packed NVFP4 (the reference quantiser).
pub fn quantize_1x16(x: &[f32], rows: usize, k: usize) -> Fp4Packed {
    assert_eq!(x.len(), rows * k, "input must be exactly rows * k");
    let blocks = k.div_ceil(BLOCK);
    let global_scale = global_scale_from_amax(amax_f32(x));
    let mut codes = vec![0u8; rows * blocks * CODE_BYTES_PER_BLOCK];
    let mut scales = vec![0u8; rows * blocks];
    for row in 0..rows {
        for block in 0..blocks {
            let mut amax = 0.0f32;
            for i in 0..BLOCK {
                let kk = block * BLOCK + i;
                if kk < k {
                    let v = x[row * k + kk];
                    let m = if v.is_finite() { v.abs() } else { 0.0 };
                    if m > amax {
                        amax = m;
                    }
                }
            }
            let scale = block_scale(amax, global_scale);
            scales[row * blocks + block] = f32_to_fp8(scale, Fp8Format::E4M3);
            let divisor = global_scale * scale;
            let base = row * blocks * CODE_BYTES_PER_BLOCK + block * CODE_BYTES_PER_BLOCK;
            for i in 0..BLOCK {
                let kk = block * BLOCK + i;
                if kk < k {
                    let code = f32_to_e2m1(x[row * k + kk] / divisor);
                    let slot = base + i / 2;
                    if i % 2 == 0 {
                        codes[slot] = (codes[slot] & 0xF0) | code;
                    } else {
                        codes[slot] = (codes[slot] & 0x0F) | (code << 4);
                    }
                }
            }
        }
    }
    Fp4Packed {
        rows,
        k,
        codes,
        scales,
        global_scale,
    }
}

/// Exact CPU block dot of two packed rows: the i32 `dp4a` result the kernel
/// computes, i.e. `sum_k (2 * a_k) * (2 * b_k)` over one 16-element block.
/// `a_row` / `b_row` are *row* indices; elements past `k` are zero coded.
pub fn block_dot_i32(
    a: &Fp4Packed,
    a_row: usize,
    b: &Fp4Packed,
    b_row: usize,
    block: usize,
) -> i32 {
    let mut total = 0i32;
    for i in 0..BLOCK {
        let ka = block * BLOCK + i;
        let kb = block * BLOCK + i;
        let ca = if ka < a.k {
            packed_code(a, a_row, block, i)
        } else {
            0
        };
        let cb = if kb < b.k {
            packed_code(b, b_row, block, i)
        } else {
            0
        };
        // `as i32` binds tighter than `*`, so the doubling must happen in f32
        // first: `2 * (e2m1 as i32)` would truncate 0.5 to 0 and silently drop
        // every +-0.5 code from the dot product.
        let da = (2.0 * e2m1_to_f32(ca)) as i32;
        let db = (2.0 * e2m1_to_f32(cb)) as i32;
        total += da * db;
    }
    total
}

fn packed_code(packed: &Fp4Packed, row: usize, block: usize, i: usize) -> u8 {
    let blocks = packed.blocks_per_row();
    let byte =
        packed.codes[row * blocks * CODE_BYTES_PER_BLOCK + block * CODE_BYTES_PER_BLOCK + i / 2];
    if i % 2 == 0 { byte & 0x0F } else { byte >> 4 }
}

/// Exact CPU reference of the packed NVFP4 GEMM:
/// `alpha * gA * gB * sum_blocks dot * 0.25 * sA * sB + beta * C`.
pub fn gemm_nt_packed(
    a: &Fp4Packed,
    b: &Fp4Packed,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    c: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(a.rows, m, "A rows must be m");
    assert_eq!(b.rows, n, "B rows must be n");
    assert_eq!(a.k, k);
    assert_eq!(b.k, k);
    let blocks = k.div_ceil(BLOCK);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for block in 0..blocks {
                let dot = block_dot_i32(a, i, b, j, block);
                let sa = e4m3_magnitude(a.scales[i * blocks + block]);
                let sb = e4m3_magnitude(b.scales[j * blocks + block]);
                acc += dot as f32 * (0.25 * sa * sb);
            }
            let value = alpha * a.global_scale * b.global_scale * acc;
            out[i * n + j] = match (c, beta) {
                (Some(c), b) if b != 0.0 => value + beta * c[i * n + j],
                _ => value,
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_grid_round_trips_exactly() {
        for (idx, v) in E2M1_GRID.iter().enumerate() {
            assert_eq!(f32_to_e2m1(*v), idx as u8);
            assert_eq!(f32_to_e2m1(-*v), idx as u8 | 8);
        }
        assert_eq!(f32_to_e2m1(f32::NAN), 0x07);
        assert_eq!(f32_to_e2m1(f32::INFINITY), 0x07);
        assert_eq!(f32_to_e2m1(-f32::INFINITY), 0x0F);
        assert_eq!(f32_to_e2m1(100.0), 0x07);
        // Grid index, not value: 1.0 is index 2 and 1.5 is index 3.
        assert_eq!(f32_to_e2m1(1.2), 2, "1.2 is nearer 1.0 than 1.5");
        assert_eq!(f32_to_e2m1(1.3), 3, "1.3 is nearer 1.5 than 1.0");
        // Ties land on the even grid index.
        assert_eq!(f32_to_e2m1(0.25), 0);
        assert_eq!(f32_to_e2m1(0.75), 2);
        assert_eq!(f32_to_e2m1(1.75), 4);
        assert_eq!(f32_to_e2m1(2.5), 4);
    }

    #[test]
    fn quantizer_hand_computed_single_block() {
        // One row, K = 16, amax = 4 -> global = 4 / 2688 = 1/672.
        let x: Vec<f32> = (0..16)
            .map(|i| if i == 0 { 4.0 } else { (i as f32 - 8.0) * 0.5 })
            .collect();
        let packed = quantize_1x16(&x, 1, 16);
        assert_eq!(packed.global_scale, 4.0 / 2688.0);
        assert_eq!(packed.codes.len(), 8);
        assert_eq!(packed.scales.len(), 1);
        let want_scale = block_scale(4.0, packed.global_scale);
        assert_eq!(packed.scales[0], f32_to_fp8(want_scale, Fp8Format::E4M3));
        // With divisor = global * scale ~= amax / 6 the largest element codes
        // to +-6 and the rest to the grid.
        assert_eq!(packed.codes[0] & 0x0F, 0x07);
        assert_eq!(packed.value(0, 0), 4.0);
    }

    #[test]
    fn zero_input_is_all_zero_codes() {
        let packed = quantize_1x16(&[0.0; 32], 2, 16);
        assert_eq!(packed.global_scale, 1.0);
        assert_eq!(packed.scales, vec![f32_to_fp8(1.0, Fp8Format::E4M3); 2]);
        assert!(packed.codes.iter().all(|c| *c == 0));
        assert!(packed.dequantized().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn ragged_tail_is_zero_padded() {
        // K = 20: the second block only has 4 real elements.
        let x: Vec<f32> = (0..20).map(|i| i as f32 + 1.0).collect();
        let packed = quantize_1x16(&x, 1, 20);
        assert_eq!(packed.blocks_per_row(), 2);
        assert_eq!(packed.codes.len(), 16);
        for kk in 20..32 {
            // The padded lanes decode as +0 through the block scale.
            let blocks = packed.blocks_per_row();
            let block = kk / BLOCK;
            let byte = packed.codes[block * CODE_BYTES_PER_BLOCK + (kk % 16) / 2];
            let code = if kk % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            assert_eq!(code, 0, "padded element {kk} must be +0");
            assert_eq!(blocks, 2);
        }
    }

    #[test]
    fn reference_gemm_matches_a_float_product_of_the_dequantised_operands() {
        // B row j holds a single 6.0 at k = j, so the exact FP4 result is
        // A_dequant[i][j] * B_dequant[j][j]; compare the block-scaled integer
        // path against a plain f32 product of the dequantised matrices.
        let k = 16;
        let a: Vec<f32> = (0..16)
            .flat_map(|i| [(i as f32 + 1.0) * 0.25; 16])
            .collect();
        let mut b = vec![0.0f32; 16 * k];
        for j in 0..16 {
            b[j * k + j] = 6.0;
        }
        let pa = quantize_1x16(&a, 16, k);
        let pb = quantize_1x16(&b, 16, k);
        let c = gemm_nt_packed(&pa, &pb, 16, 16, k, 1.0, 0.0, None);
        for i in 0..16 {
            for j in 0..16 {
                let want = pa.value(i, j) * pb.value(j, j);
                let scale = want.abs().max(1e-3);
                assert!(
                    (c[i * 16 + j] - want).abs() <= 1e-5 * scale,
                    "[{i},{j}]: got {} want {want}",
                    c[i * 16 + j]
                );
                // The block-scaled operand must stay within one E2M1 step
                // (25% of the local magnitude) of the original.
                let deq = pa.value(i, j);
                assert!(
                    (deq - a[i * 16 + j]).abs() <= 0.25 * a[i * 16 + j].abs() + 1e-3,
                    "[{i},{j}]: dequantised {} vs {}",
                    deq,
                    a[i * 16 + j]
                );
            }
        }
    }

    #[test]
    fn reference_gemm_applies_alpha_and_beta() {
        let k = 32;
        let a: Vec<f32> = (0..(4 * k)).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..(4 * k)).map(|i| (i % 5) as f32 - 2.0).collect();
        let pa = quantize_1x16(&a, 4, k);
        let pb = quantize_1x16(&b, 4, k);
        let c0: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let got = gemm_nt_packed(&pa, &pb, 4, 4, k, 2.0, 0.25, Some(&c0));
        let plain = gemm_nt_packed(&pa, &pb, 4, 4, k, 1.0, 0.0, None);
        for i in 0..16 {
            let want = 2.0 * plain[i] + 0.25 * c0[i];
            assert!(
                (got[i] - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "[{i}]: {} vs {want}",
                got[i]
            );
        }
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;

    /// `as i32` binds tighter than `*`: the dot product has to double the E2M1
    /// values in f32, otherwise the +-0.5 codes truncate to zero.
    #[test]
    fn block_dot_includes_the_half_codes() {
        let k = 16;
        // Row 0 of A is all +0.5 (code 1), row 0 of B all +0.5: the doubled dot
        // is 16 * 1 * 1 = 16.
        let mut codes = vec![0x11u8; 16];
        codes[8..16].fill(0x77);
        let scale = crate::fp8::f32_to_fp8(448.0, crate::fp8::Fp8Format::E4M3);
        let packed = Fp4Packed {
            rows: 2,
            k,
            codes,
            scales: vec![scale; 2],
            global_scale: 1.0,
        };
        assert_eq!(block_dot_i32(&packed, 0, &packed, 0, 0), 16);
        assert_eq!(block_dot_i32(&packed, 0, &packed, 1, 0), 16 * 12);
        assert_eq!(block_dot_i32(&packed, 1, &packed, 0, 0), 16 * 12);
    }

    /// Every code must contribute the value the CUDA-core kernel decodes.
    #[test]
    fn block_dot_matches_a_hand_sum_over_all_codes() {
        let k = 16;
        let mut a_codes = vec![0u8; 8];
        for i in 0..16 {
            a_codes[i / 2] |= (i as u8) << (4 * (i % 2));
        }
        let b_codes = vec![0x11u8; 8];
        let scale = crate::fp8::f32_to_fp8(448.0, crate::fp8::Fp8Format::E4M3);
        let pa = Fp4Packed {
            rows: 1,
            k,
            codes: a_codes,
            scales: vec![scale],
            global_scale: 1.0,
        };
        let pb = Fp4Packed {
            rows: 1,
            k,
            codes: b_codes,
            scales: vec![scale],
            global_scale: 1.0,
        };
        let mut want = 0i32;
        for code in 0u8..16 {
            want += (2.0 * e2m1_to_f32(code)) as i32 * 1;
        }
        assert_eq!(block_dot_i32(&pa, 0, &pb, 0, 0), want);
    }
}
