# Commands

Build, check and test commands. `cargo xtask` applies the memory and thread caps.

## Table of Contents

- [Commands](#commands)

## Commands

```
cargo xtask check [pkg...]                  # default: lightnet-cli, lightnet-core
cargo xtask gpu-check                       # lightnet-cli with oxide
cargo xtask test [filter]                   # host tests; filter is much faster
cargo xtask oxide --test <name> [filter]    # one oxide suite
cargo xtask clippy [pkg...]
cargo xtask build [pkg...]
cargo test -p lightnet-core --test cfg_corpus -j 2   # run alone; >1 h under load
```

`cargo xtask` applies the memory and thread caps, and warns and runs uncapped when there is no user systemd bus. A `cfg_corpus` timeout is neither a pass nor a failure.

