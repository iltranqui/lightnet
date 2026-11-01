# hip_tile branch lessons

## Table of Contents

- [Lessons](#lessons)

## Lessons

- `codex exec "<prompt>"` waits on stdin ("Reading additional input from stdin...") when run in the background; always add `< /dev/null`.
- Never leave a bare `python3 -` in a shell command: it blocks on stdin.
- hip_tile launch grid is `(blocks along x, blocks along y)` and kernels read `block_id().1` for x: a 1-D launch must be `(nblocks, 1)`. Codex wrote `(1, nblocks)`; review every Codex launcher for this.
- The Bash tool already runs every command under `flock /tmp/claude-shell.flock`. Do not add `flock` to a command: it waits on its own lock and hangs for good (a cargo test hung 10 minutes).
