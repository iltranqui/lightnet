# Todo (branch rust)
- [x] flatten rust/ to root
- [x] vendor cuda-oxide crates, merge workspace
- [x] root nightly-2026-08-28 pin
- [x] verify: cpp_parity, cuda-macros/device/host, darknet-core
- [ ] decide: restore bridges/cutile-rs-aot or drop optional cutile dep (pre-existing break)
- [ ] commit (flatten, then vendor)

## Phase: cuda-oxide kernel backend (plan: ~/.claude/plans/https-github-com-nvidia-cuda-rust-port-t-imperative-pumpkin.md)
- [ ] P0 scaffold bridges/darknet-cuda-oxide + cargo oxide smoke
- [ ] P1 elementwise/bias/activations/sgd
- [ ] P2 conv direct + im2col/gemm + heuristic
- [ ] P3 pooling/upsample/route/crop
- [ ] P4 cli `oxide` feature + e2e
- [ ] P5 trait/runtime for softmax/reorg/dropout/scale_channels/connected
