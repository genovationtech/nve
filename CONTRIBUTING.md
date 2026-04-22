# Contributing to NVE

Thanks for your interest in the Neural Virtualization Engine. NVE is
experimental research software, so contributions that improve correctness,
measurement rigor, documentation, or portability are especially welcome.

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproduction, the model /
  hardware / tier configuration you used, and the relevant log output.
- **Feature proposals** — open an issue first to discuss scope before
  sending a PR, especially for changes that affect the profiler, pager,
  clusterer, or tier manager.
- **Benchmarks / evidence** — add raw results to a new dated folder under
  `evidence/YYYY-MM-DD/` (see `evidence/README.md` for the layout).
- **Documentation** — README, `docs/`, and inline comments all count.

## Development setup

NVE is a Rust core with a Python SDK on top.

```bash
# Rust core
cargo build --release
cargo test

# Python SDK (editable install)
cd python
pip install -e '.[dev]'
pytest
```

CUDA kernels under `cuda/` are built via `build.rs` when a CUDA toolkit is
detected. CPU-only builds are supported and are the default for CI.

## Pull request checklist

Before opening a PR:

1. `cargo fmt` and `cargo clippy --all-targets -- -D warnings` pass.
2. `cargo test` passes.
3. Python changes: `ruff check python/` and `pytest python/tests` pass.
4. Public API changes are reflected in `README.md` and/or `docs/`.
5. New benchmark runs land under `evidence/YYYY-MM-DD/` with raw outputs,
   not only summaries.
6. Commits have descriptive messages — explain *why*, not just *what*.

PRs that change behavior should include either a test or a benchmark
demonstrating the change.

## Commit and branch conventions

- Branch from `main`; name branches `<area>/<short-description>`
  (e.g. `pager/ssd-prefetch-window`, `docs/readme-quickstart`).
- Keep commits focused. Prefer several small commits over one large one
  when the steps are independently reviewable.
- Do **not** commit build artifacts, model weights, or cache directories —
  `.gitignore` already covers the common cases; add to it if you find a
  new one.

## Code style

- **Rust**: edition from `Cargo.toml`, `rustfmt` defaults, `clippy` clean.
- **Python**: type hints on public functions, `ruff` defaults, docstrings
  on public modules and classes.
- Avoid adding comments that restate the code. Reserve comments for
  non-obvious invariants, workarounds, or references to papers / issues.

## Reporting security issues

Please do **not** file public issues for security vulnerabilities. Email the
maintainers directly (see `Cargo.toml` / `pyproject.toml` for contact
addresses) with a description and reproduction steps. We will acknowledge
receipt within a reasonable window and coordinate a fix before any public
disclosure.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License that covers the project (see [LICENSE](LICENSE)).
