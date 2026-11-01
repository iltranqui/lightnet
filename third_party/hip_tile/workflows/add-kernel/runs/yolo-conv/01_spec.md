---
status: done
stage: workflows/add-kernel/01_spec
date: 2026-10-25
---

# yolo-conv — spec

Contract: [`workflows/add-kernel/01_spec/CONTEXT.md`](../../01_spec/CONTEXT.md).

## Contents

- [Math](#math)
- [Shapes](#shapes)
- [Tile config and LDS](#tile-config-and-lds)
- [Launcher](#launcher)
- [Reference and tolerance](#reference-and-tolerance)

## Math

`out[co][oy][ox] = act(bias[co] + Σ w[co][ci][ky][kx] · in[ci][oy·s+ky−p][ox·s+kx−p])`, zero outside; act linear or leaky 0.1. As GEMM: A = weights `[Cout × Cin·k·k]`, B = im2col `[Cin·k·k × OH·OW]` gathered on the fly, C = `[Cout × OH·OW]`.

## Shapes

LegoGears layers: k 3 (pad 1) or 1; stride 1/2; Cin 3–512; Cout 30–512; OH·OW 8960 down to 35. Ragged: 19→30 on 5×7, 7→41 k3 s2 on 9×13; writes into a channel slice with `ld`.

## Tile config and LDS

Configs C64x64 (10240 B), C128x64 (14336), C32x128 (8832), C64x32 (7168), C32x32 (4608), all BK=16. `Config::choose(cout, n_out)` picks by output size.

## Launcher

`kernels::yolo::conv::launch_conv(module, stream, x, w, bias, y, &ConvParams::new(cin, h, w, k, stride, pad, act))`. Also maxpool, upsample, concat in `kernels::yolo::*`.

## Reference and tolerance

f64 CPU reference in the test; conv max relative error ≤ 1e-4 (floor 1% of the output's max magnitude); maxpool, upsample, concat exact.
