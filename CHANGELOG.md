# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and GitHub issue /
  pull-request templates for public-release readiness.
- `.gitignore` covering Rust `target/`, Python caches, virtualenvs, build
  artifacts, native binaries, and model weight formats.
- Chronological layout for `evidence/` — runs are now grouped by capture
  date under `evidence/YYYY-MM-DD/`.

### Changed
- License switched to MIT.
- README `Project Structure` section rewritten to match the real tree;
  added an experimental-status disclaimer.
- Root-level integration harnesses (`test_*.py`, `benchmark_hot_only.py`)
  moved into `tests/integration/`.

### Removed
- Previously-tracked build artifacts and caches (`target/`, `__pycache__/`,
  `.nve_build_version`). Tracked file count dropped from ~11.1k to ~230.

## [0.2.0]

Initial public-facing snapshot of the Neural Virtualization Engine: Rust
core (profiler, pager, clusterer, tier manager), Python SDK, CUDA kernels,
and the first round of benchmark evidence across GPT-2, Qwen2.5, Llama-3.2
1B, and Llama-3.2 3B.
