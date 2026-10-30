# Layer Contract

Where layer code lives and when a layer counts as finished. Applies to every layer.

## Table of Contents

- [Layout](#layout)
- [Layer Completion Contract](layer-contract.md)

## Layout

The layout contract (folder shape, where each piece plugs in, the add-or-finish recipe and the verify commands) lives in `src/lightnet-core/src/layers/CONTEXT.md`. Read it before touching a layer; it is not repeated here.

- `tasks/rust/02_state/layer-map.md` lists, per layer, its CPU file, GPU glue, kernel file and parity test.

## Layer Completion Contract

A layer is complete only when all of the following hold, with a named test or documented evidence for each:

1. Host: config and shape, `.weights` round trip, CPU forward, exact CPU backward, SGD and Adam arms, Muon where the layer has eligible matrices, and a named `axis_layout` test.
2. Device: native full forward and, where the f16 inference list includes the layer, native compact forward; native backward; native parameter update; no host round trip inside the arm, except a documented, tested fallback for variants the kernel cannot cover.
3. Parity: an Oxide test compares the device forward, backward and update with the CPU result within a stated tolerance, run through `cargo xtask oxide --test <suite>`.
4. Layout: the code lives in `layers/<layer>/` and has a row in `layer-support.md` and `layer-map.md`.
5. Docs: the version commit, `implemented.md`, `STATUS.md`, `todo.md` and `version-map.tsv` are updated, and nothing claims C++ parity without a compiled-C++ golden.

Native primitives or a native forward alone do not make a layer complete. `layer-support.md` records which parts exist today.

