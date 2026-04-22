# NVE Usage Guide

## Installation

```bash
cd nve
cargo build --release
```

Single binary at `target/release/nve`. No Python, no pip, no conda.

## Commands

### Generate Text

```bash
# Basic generation (loads full model into RAM)
nve generate -m meta-llama/Llama-3.2-1B -p "Hello world" -n 100

# Paged mode (works with models larger than RAM)
nve generate -m meta-llama/Llama-3.2-3B --paged -p "Hello" -n 50

# Hot-only mode (skip cold layers for speed, lower quality)
nve generate -m meta-llama/Llama-3.2-3B --paged --hot-only \
  --hot-budget-mb 250 --warm-budget-mb 750 -p "Hello" -n 50

# Profiled hot-only (select best layers by importance)
nve generate -m meta-llama/Llama-3.2-3B --paged --hot-only --profile \
  --hot-budget-mb 250 --warm-budget-mb 750 -p "Hello" -n 50

# With quantization (optional, reduces memory)
nve generate -m meta-llama/Llama-3.2-3B --paged --quantize q4 -p "Hello"
nve generate -m meta-llama/Llama-3.2-3B --paged --quantize q8 -p "Hello"

# Profile-guided quantization (non-uniform, AWQ-protected)
nve generate -m meta-llama/Llama-3.2-3B --paged --hot-only --profile \
  --quantize pg:2.0 -p "Hello" -n 50

# Auto-detect memory budget (uses 80% of available RAM)
nve generate -m model --paged --auto-budget -p "Hello"

# Control active layers explicitly
nve generate -m model --paged --hot-only --active-layers 10 -p "Hello"

# Temperature and top-p sampling
nve generate -m model -p "Once upon a time" -n 200 -t 0.8 --top-p 0.95
```

### Model Information

```bash
# Show model details (works with any HuggingFace model)
nve info -m meta-llama/Llama-3.2-3B
nve info -m microsoft/Phi-3.5-mini-instruct
nve info -m ./my-local-model
```

### Download Models

```bash
# Download from HuggingFace Hub
nve download meta-llama/Llama-3.2-3B
nve download microsoft/Phi-3.5-mini-instruct -o ./models/phi3
```

### Benchmark

```bash
# Compare baseline vs paged inference
nve benchmark -m meta-llama/Llama-3.2-3B -n 50 \
  --hot-budget-mb 250 --warm-budget-mb 750

# With quantization
nve benchmark -m model --quantize q4 --output results.json
```

### List Architectures

```bash
nve architectures
```

## Supported Models

Any HuggingFace model with safetensors weights and one of these architectures:

| Architecture | Example Models |
|---|---|
| Llama | Llama 2/3, CodeLlama, Llama 3.2 1B/3B |
| Mistral | Mistral-7B-v0.1 |
| Qwen2 | Qwen2.5-7B |
| Phi-3 | Phi-3.5-mini-instruct |
| Gemma | Gemma-2B, Gemma2-9B |
| GPT-NeoX | Pythia, RedPajama |
| GPT-2 | GPT-2 (all sizes) |
| Falcon | Falcon-7B, Falcon-40B |
| StableLM | StableLM-2-1.6B |
| StarCoder2 | StarCoder2-3B |
| InternLM2 | InternLM2-7B |
| OLMo | OLMo-1B, OLMo-7B |
| DeepSeek | DeepSeek-7B |

## Quantization Options

| Flag | Description | Quality | Memory |
|---|---|---|---|
| `--quantize none` | bf16 (default) | Best | Full |
| `--quantize q8` | 8-bit uniform | Near-lossless | 1.8x smaller |
| `--quantize q4` | 4-bit uniform | Good | 3.2x smaller |
| `--quantize q3` | 3-bit uniform | Degraded | 4x smaller |
| `--quantize q2` | 2-bit uniform | Significant loss | 5.3x smaller |
| `--quantize q1` | 1-bit ternary | Heavy loss | 5.3x smaller |
| `--quantize pg:2.0` | Profile-guided 2 bpw | Mixed (best layers high, worst pruned) | ~7.5x smaller |
| `--quantize pg:0.5` | Profile-guided 0.5 bpw | Extreme (3 layers kept) | ~30x smaller |

Profile-guided modes require `--profile` and `--paged --hot-only`.

## Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace API token for gated models (e.g., Llama) |
| `RUST_LOG` | Logging level: `info`, `debug`, `warn` |

## Memory Budget Guide

| System RAM | Recommended Budget | Suitable Models |
|---|---|---|
| 2 GB | 250 hot + 500 warm | 1B paged, 3B hot-only |
| 4 GB | 500 hot + 1500 warm | 3B paged, 7B hot-only |
| 8 GB | 1000 hot + 4000 warm | 7B paged, 13B hot-only |
| 16 GB | 2000 hot + 8000 warm | 13B paged, 30B hot-only |
| 32 GB+ | Use --auto-budget | Most models fit in RAM |

With `--quantize q4`, effective model size is ~3x smaller, so a 7B model needs only ~4.5 GB instead of ~14 GB.

With `--quantize pg:2.0`, effective model size is ~7x smaller, so a 7B model needs only ~2 GB.
