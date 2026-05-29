# rust-cargo

### Overview
- **Environment ID**: `rust-cargo`
- **Short description**: Rust coding tasks scored with `cargo build`, `cargo clippy`, and `cargo test` feedback.
- **Tags**: rust, cargo, coding, single-turn, RL, train, eval

### Task
The model receives a Rust programming prompt and should return exactly one fenced
Rust code block containing:

- the requested function implementation,
- a `#[cfg(test)] mod tests` module,
- multiple `assert!` / `assert_eq!` checks,
- no `main` function.

### Quickstart
Requires a local Rust toolchain with `cargo` available on `PATH` for executable
Cargo rewards.

```bash
prime env install environments/rust_cargo
prime eval run rust-cargo -n 1 -r 1
```

### Rubric
The reward combines static and executable Cargo feedback:

- response format / one Rust block,
- required function signature present,
- test assertions present,
- `cargo build --quiet`,
- `cargo clippy --quiet -- -D warnings`,
- `cargo test --quiet` with extra weight.

This ports the core idea from the Algora reference implementation: use Cargo as
an objective signal for Rust code generation.
