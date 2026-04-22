# Security Policy

## Supported Versions

NVE is experimental research software. Only the latest `main` branch receives
security fixes. Tagged releases below the current minor version are not
actively supported.

## Reporting a Vulnerability

Please do **not** open public GitHub issues for security vulnerabilities.

Instead, report them privately by either:

- Using GitHub's private vulnerability reporting
  (**Security** tab → **Report a vulnerability**), or
- Emailing the maintainers at the address listed in the project's
  `Cargo.toml` / `pyproject.toml` metadata.

When reporting, please include:

1. A description of the issue and its potential impact.
2. Steps to reproduce (minimal example preferred).
3. The affected commit / version.
4. Any suggested mitigations, if known.

## Disclosure Timeline

- We aim to acknowledge reports within **5 business days**.
- We aim to provide an initial assessment within **14 days**.
- Fixes are coordinated with the reporter before any public disclosure.

## Scope

In scope:

- The Rust core engine (`src/`, `cuda/`) and its FFI surface.
- The Python SDK (`python/nve/`) and the `nve-serve` entry point.

Out of scope:

- Vulnerabilities in third-party dependencies (report upstream; we'll bump
  versions once a fix is available).
- Issues that require an already-compromised host or privileged local
  access to exploit.
- Research and benchmarking scripts under `evidence/`, `paper/`, and
  `google_colab/` — these are illustrative, not production code.
