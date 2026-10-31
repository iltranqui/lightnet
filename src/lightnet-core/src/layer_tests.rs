//! Layer tests that need the executor and the CPU reference together, so they
//! cannot live in `lightnet-layers`.

#[path = "layers/carpn/tests.rs"]
mod carpn;
#[path = "layers/gru/tests.rs"]
mod gru;
#[path = "layers/laya_scorer/wiring_tests.rs"]
mod laya_scorer;
