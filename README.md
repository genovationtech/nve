# Neural Virtualization Engine (NVE)

**Monte Carlo Guided Virtual Weight Paging for Neural Networks**

NVE is a runtime system that uses Monte Carlo sampling to estimate weight importance distributions across neural network layers, then *virtualizes* — not prunes — weights across memory tiers (GPU / RAM / SSD). It dynamically loads weight clusters based on inferred activation patterns, achieving the performance of a smaller model with the capacity of a larger one.

## Core Idea

> Use Monte Carlo not to remove weights — but to learn which parts of the model deserve to exist in fast memory.

Traditional pruning permanently removes weights, destroying generality. NVE treats model weights like virtual memory pages: hot weights live on GPU, warm weights in RAM, cold weights on SSD — and they migrate dynamically based on runtime activation profiles.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Python SDK Layer                   │
│         (PyTorch integration, ML workflows)          │
├─────────────────────────────────────────────────────┤
│                    Rust Core Engine                   │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │   Profiler   │ │    Pager     │ │  Clusterer   │  │
│  │   (MCAP)     │ │  (3-Tier)   │ │(Co-activate) │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘  │
│         │                │                │          │
│  ┌──────▼────────────────▼────────────────▼───────┐  │
│  │              Tier Manager                       │  │
│  │   GPU (hot) ◄──► RAM (warm) ◄──► SSD (cold)   │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## How It Works

### Phase 1: Monte Carlo Activation Profiling (MCAP)

Sample diverse prompts across domains (math, code, reasoning, conversation), run forward passes, and accumulate activation scores per weight:

```
Î(Wᵢ) = (1/N) Σₖ activationₖ(Wᵢ)
```

This is Monte Carlo estimation of weight importance — grounded in the Lottery Ticket Hypothesis and activation-based pruning literature, but applied *dynamically at runtime*.

### Phase 2: Weight Clustering

Weights are clustered by co-activation patterns into latent feature groups:

```
W = ⋃ₖ Wₖ
```

Each cluster `Wₖ` represents a coherent computational unit that activates together for certain input distributions.

### Phase 3: Virtual Paging

Weight clusters are assigned to memory tiers based on importance:

| Tier | Storage | Weights | Access Pattern |
|------|---------|---------|----------------|
| Hot  | GPU VRAM | Top ~20% | Always resident |
| Warm | System RAM | Next ~30% | Loaded on demand |
| Cold | SSD/NVMe | Remaining ~50% | Paged in when needed |

### Phase 4: Runtime Dispatch

At inference time, activation patterns are matched against known clusters to predict which weight groups are needed — and page them in *before* they're accessed:

```python
# No explicit labels needed — inferred from activation patterns
engine.infer(prompt)  # Automatically loads math-heavy clusters for math prompts
```

### Phase 5: Continuous Online Learning

Importance estimates are updated continuously, so the system adapts as usage patterns shift.

## Key Differentiators

| Approach | Static? | Reversible? | Runtime-aware? | Distribution-aware? |
|----------|---------|-------------|----------------|---------------------|
| Traditional Pruning | Yes | No | No | No |
| MoE | Partially | N/A | Yes | No |
| Quantization | Yes | Lossy | No | No |
| **NVE** | **No** | **Yes** | **Yes** | **Yes** |

NVE implements **Stochastic Sparse Execution of Dense Models** — probabilistic compute selection that preserves the full model while executing only what's needed.

## Known Challenges

1. **Coverage**: Monte Carlo sampling must cover all domains and edge cases
2. **Stability**: Rarely-activated weights may still be critical (safety weights, edge-case handlers)
3. **Fragmentation**: Too many small weight chunks can cause memory fragmentation
4. **GPU Efficiency**: Sparse compute patterns don't always map efficiently to GPU architectures

## Project Structure

```
nve/
├── Cargo.toml          # Rust workspace configuration
├── build.rs            # Build script (CUDA detection, FFI)
├── src/                # Rust core engine (profiler, pager, clusterer, tiers)
├── cuda/               # CUDA kernels for GPU tier operations
├── python/             # Python SDK (pyproject.toml, `nve` package, tests)
├── examples/           # End-to-end usage examples
├── tests/              # Rust integration tests
├── docs/               # Architecture, benchmarks, quantization, usage
├── paper/              # Research paper drafts and figures
├── evidence/           # Benchmark harnesses and raw measurement data
├── reports/            # Generated benchmark reports
└── google_colab/       # Colab notebooks for reproducible demos
```

## Quick Start

### Build the Rust core

```bash
cd nve
cargo build --release
```

### Install the Python SDK

```bash
cd nve/python
pip install -e .
```

### Run the example

```python
from nve import NVEEngine, TierConfig

engine = NVEEngine(
    model_path="path/to/model.safetensors",
    tiers=TierConfig(gpu_fraction=0.2, ram_fraction=0.3)
)

# Profile with diverse prompts
engine.profile(prompts=["Solve x^2 + 3x = 0", "Write a Python sort", "Tell me a joke"])

# Inference with automatic weight paging
result = engine.infer("What is the integral of sin(x)?")
```

## Theoretical Foundations

- **Lottery Ticket Hypothesis**: Networks contain smaller subnetworks that perform equally well
- **Activation-based Pruning**: Neurons that rarely activate are removable
- **Monte Carlo Estimation**: Sampling approximates importance distributions
- **Virtual Memory Systems**: OS-level paging applied to neural network weights

NVE combines all four into a unified runtime system.

## Status

NVE is **experimental research software**. APIs, configuration, and on-disk
formats may change without notice. See `docs/` and `paper/` for current design
notes and measurements.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on filing issues, submitting pull requests, and the development
workflow (Rust + Python toolchains, testing, code style).

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Author

Anurita Das — [Genovation Technological Solutions Pvt Ltd](mailto:anurita@genovationsolutions.com)
