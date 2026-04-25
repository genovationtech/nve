# Neural Virtualization Engine (NVE)

NVE is a Rust + CUDA inference engine for large language models that performs
load-time layer profiling to drive both precision selection and weight
placement.

The system enables operation under constrained memory budgets by combining:

- per-layer mixed precision (W4A8 / W4A16)
- three-tier weight paging (GPU / CPU RAM / SSD)

In evaluated configurations, NVE achieves:

- 1.5–1.8× higher decode throughput than llama.cpp Q4_0 on NVIDIA T4
- execution of 8B-class models within ~4 GB GPU memory via paging
- no measurable degradation on WikiText-2 and HellaSwag at tested scales

- Paper: [arXiv:2604.21026](https://arxiv.org/abs/2604.21026)
- Code:  https://github.com/genovationtech/nve

## Overview

Most LLM inference systems fix precision and memory placement offline. NVE
instead moves both decisions to load time.

A short forward pass over a small set of calibration prompts produces a
per-layer importance signal (MCAP). This signal is then used to:

1. Route layers to precision tiers (W4A8 or W4A16).
2. Assign weights to memory tiers (GPU, RAM, SSD).

The same model weights can therefore operate across different memory budgets
without modification. A profile is a 338-byte JSON keyed by architecture, not a
new set of weights; each target computes its own profile on load.

## Results

Benchmarks on NVIDIA T4, batch = 1, greedy decode:

- NVE W4A8 vs. llama.cpp Q4_0: 1.5–1.8× higher decode throughput across
  Llama-3.2-1B, Llama-3.2-3B, and Llama-3.1-8B.
- W4A8 vs. W4A16 within NVE: 2.3× improvement at 1B, widening to 2.9× at 8B.
- Llama-3.2-3B (~6.4 GB BF16) runs in 2 GB at 2.31 tok/s via paging.
- Llama-3.1-8B (~16 GB BF16) runs in 4 GB at 1.25 tok/s via paging.

Quality (same models, same tasks):

- WikiText-2 perplexity: no measurable difference across W4A16, W4A8, and
  MCAP Mixed.
- HellaSwag: within binomial CI of AutoAWQ at matched 4-bit precision
  (54% vs. 52%, n = 50).
- Cross-silicon determinism: a single 338-byte profile is SHA-256–identical
  across six parallel containers on T4 (sm_75) and A10G (sm_86); task accuracy
  87.5% on all 12 runs at 0 ms on-device profiling cost.

See the paper for full methodology, statistical caveats, and the failure modes
(Section 7, "Where the method breaks").

## Method

### MCAP: Monte Carlo Activation Profiling

A load-time, gradient-free, weight-free estimator of per-layer importance. A
60-second forward pass over 12 stratified calibration prompts produces a
normalized per-layer score:

- high-score layers → higher precision (W4A16)
- low-score layers  → lower precision (W4A8)

Scores are hardware-independent and cached per architecture. Evaluated across
eight architectures from GPT-2 (0.1B) through Qwen2-7.6B.

### Precision dispatch

Each layer is routed at decode time from its MCAP score:

- W4A16 for high-importance layers (kept in the CUDA graph-captured decode path)
- W4A8 for the remainder (custom INT8 kernel)

The routing reduces to a single branch in the per-layer decode path.

### Virtual weight paging

Weights are assigned to memory tiers:

- **GPU (hot)** — top layers by MCAP score, resident in VRAM
- **CPU RAM (warm)** — staged for on-demand transfer
- **SSD (cold)** — memory-mapped, paged in when needed

Placement is initialized from MCAP scores and maintained at runtime.

## Repository layout

```
src/
  profiler.rs             MCAP profiler (Section 3)
  importance_cache.rs     Architecture-keyed profile cache
  quantize.rs             W4A8 / W4A16 dispatch (Section 4)
  cuda_kernels.rs         CUDA kernels
  decode_graph.rs         CUDA graph-captured decode path
  pager.rs, tier.rs,      Three-tier weight pager (Section 5)
  paged_model.rs
  arch.rs,                Generic transformer front-end (8 architectures)
  generic_model.rs
  cli/                    `nve` binary (profile / run / bench)
benchmarks/, src/benchmark.rs, src/abc_benchmark.rs
                          Throughput, PPL, HellaSwag, 8-task harness
evidence/                 Dated run captures (2026-04-12 → 2026-04-18):
                          Modal runner scripts, raw JSON logs, figures
tests/, examples/         Integration tests and minimal runnable examples
docs/                     Architecture, quantization, streaming-profiler notes
```

Hugging Face access tokens are read from the `HF_TOKEN` environment variable;
no credentials are committed.

## Quick start

Build the engine:

```bash
cargo build --release
```

GPU builds:

```bash
cargo build --release --features cuda              # NVIDIA CUDA
cargo build --release --features cuda,flash-attn   # + Flash Attention v2 (sm_80+)
cargo build --release --features metal             # Apple Metal
cargo build --release --features vulkan            # Vulkan / WebGPU
```

Run a model end-to-end:

```bash
./target/release/nve run --model meta-llama/Llama-3.2-1B --prompt "..."
```

Profile + inspect the 338-byte signal:

```bash
./target/release/nve profile --model meta-llama/Llama-3.2-1B --out profile.json
```

Python usage (experimental):

```python
from nve import NVEEngine, TierConfig

engine = NVEEngine(
    model_path="path/to/model.safetensors",
    tiers=TierConfig(gpu_fraction=0.2, ram_fraction=0.3),
)
engine.profile(prompts=[
    "Solve x^2 + 3x = 0",
    "Write a Python sort",
    "Explain eigenvalues",
])
out = engine.infer("What is the integral of sin(x)?")
```

## Limitations

- MCAP quality depends on the calibration prompt distribution; the default
  12-prompt set is stratified (science, code, history, math) but not
  domain-adapted.
- Layer importance is a heuristic proxy for quantization sensitivity. It
  tracks sensitivity well in the evaluated regime but is not a proof.
- Sparse execution and paging patterns do not always map efficiently to GPU
  throughput; paged operating points trade latency for reach.
- Aggressive profile-guided quantization fails at the 1B scale (reported in
  the paper).
- Quality results are limited to the benchmarks and sample sizes evaluated
  (WikiText-2 PPL, HellaSwag, 8-task harness). Broader validation on Jetson,
  Apple Silicon, and consumer RTX is follow-up work.

## Status

Experimental research software. APIs, configuration, and on-disk formats may
change without notice. See `docs/` and the [paper](https://arxiv.org/abs/2604.21026)
for design notes and measurements.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for issues, pull requests, and the
development workflow (Rust + Python toolchains, testing, code style).

## License

MIT. See [LICENSE](LICENSE).

## Author

Anurita Das — [Genovation Technological Solutions Pvt Ltd](mailto:anurita@genovationsolutions.com)
