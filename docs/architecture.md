# NVE Architecture

NVE (Neural Virtualization Engine) is a pure Rust inference engine that uses Monte Carlo profiling to intelligently manage neural network weights across memory tiers and quantization levels.

## System Overview

```
                        ┌──────────────────────────────────┐
                        │           NVE CLI Binary          │
                        │     (single cargo build, no deps) │
                        └──────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
   ┌──────────▼──────────┐ ┌──────▼───────┐ ┌──────────▼──────────┐
   │  Streaming Profiler  │ │  Bit Allocator│ │   Paged Inference   │
   │  (layer importance   │ │  (importance  │ │   (hot/warm/cold    │
   │   + AWQ saliency)    │ │   → bit rate) │ │    tier management) │
   └──────────┬───────────┘ └──────┬───────┘ └──────────┬──────────┘
              │                    │                     │
              └────────────────────┼─────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Quantization Engine      │
                    │  Q8 Q4 Q3 Q2 Q1 Sparse AWQ  │
                    │      (AVX2 SIMD kernels)     │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
   ┌──────────▼──────┐  ┌─────────▼────────┐  ┌────────▼────────┐
   │  Generic Model   │  │   Weight Map     │  │   Safetensors   │
   │  (13 archs,      │  │   (per-arch      │  │   (mmap loader, │
   │   fused QKV/FFN)  │  │    weight names)  │  │    sharded)     │
   └──────────────────┘  └──────────────────┘  └─────────────────┘
```

## Core Components

### Streaming Profiler (`paged_model.rs`)

Profiles a model's layer importance without holding the full model in RAM. Loads one layer at a time, measures its contribution, evicts, moves to the next. Also collects AWQ per-channel activation saliency during the same pass.

- **Input:** 1 prompt token, model weights on disk
- **Output:** per-layer importance scores + per-channel saliency
- **Peak memory:** embedding + 1 layer + hidden state
- **Time:** ~20-30s for a 3B model with 28 layers

### Paged Model (`paged_model.rs`)

Manages weights across three memory tiers:

| Tier | Storage | Access Pattern |
|------|---------|----------------|
| Hot | RAM (always loaded) | Direct access |
| Warm | RAM (LRU managed) | On-demand with eviction |
| Cold | Disk (mmap'd safetensors) | Page fault → load → evict LRU |

Hot-only mode skips cold layers entirely for speed at the cost of quality. The profiler selects which layers to keep.

### Quantization Engine (`quantize.rs`)

Seven quantization formats, all with AVX2 SIMD acceleration:

| Format | Bits/Weight | Block Layout | Use Case |
|--------|-------------|-------------|----------|
| bf16 | 16 | CompactTensor | Full precision baseline |
| Q8 | ~9 | f32 scale + 32 i8 | High quality, mild compression |
| Q4 | ~5 | f32 scale + 16 packed nibbles | Standard compression |
| Q3 | ~4 | f32 scale + 12 packed bytes | Medium compression |
| Q2 | ~3 | f32 scale + 8 packed bytes | Aggressive compression |
| Q1 | ~3 | f32 scale + nonzero mask + sign bits | Ternary {-1,0,+1} |
| Sparse | <1 | block bitmap + Q4 active blocks | Extreme compression |

### Bit Allocation Algorithm (`quantize.rs:allocate_bits`)

Given profiler importance scores and a target bits-per-weight budget, assigns each layer an optimal quantization level:

1. Rank layers by importance (descending)
2. Start all at 0 bits (pruned)
3. Greedily upgrade the most efficient layer: `efficiency = importance / bit_cost`
4. Continue until budget exhausted

### AWQ Integration (`generic_model.rs:quantize_adaptive`)

Activation-Aware Weight Quantization protects important weights within each layer:

1. During profiling, collect `mean(|activation_j|)` per input channel
2. Compute scaling: `α_j = saliency_j^0.5`
3. Before quantizing: `W[:,j] *= α_j` (important weights scaled up)
4. Store `inv_α = 1/α` for inference correction
5. At inference: `x_j *= inv_α_j` before matmul

This ensures important weights land on higher quantization levels regardless of bit width.

### Generic Model Support (`generic_model.rs`, `weight_map.rs`, `arch.rs`)

Supports 13 transformer architectures:

| Architecture | Model Types | Special Handling |
|---|---|---|
| Llama | Llama 2/3, CodeLlama | Standard (separate Q/K/V) |
| Mistral | Mistral-7B | Llama-style |
| Qwen2 | Qwen2.5 series | Q/K/V biases |
| Phi-3 | Phi-3.5-mini | Fused QKV + fused gate_up |
| Gemma | Gemma, Gemma2 | GeGLU FFN |
| GPT-NeoX | Pythia, RedPajama | Fused QKV + parallel attn+FFN |
| GPT-2 | GPT-2 | Conv1D weights + learned position embeds |
| Falcon | Falcon-7B/40B | Fused QKV + parallel attn+FFN |
| StableLM | StableLM-2 | Llama-style |
| StarCoder2 | StarCoder2 | Full biases |
| InternLM2 | InternLM2 | Fused QKV, different naming |
| OLMo | OLMo | Llama-style |
| DeepSeek | DeepSeek | Llama-style |

Each architecture maps to canonical weight names via `weight_map.rs`, with automatic fused tensor splitting for QKV and gate_up projections.

## Data Flow

### Standard Inference
```
prompt → tokenize → embed → [layer 0 → ... → layer N] → norm → logits → sample → token
```

### Profiled Hot-Only Inference
```
prompt → tokenize
  → streaming profile (1 layer at a time, collect importance + AWQ saliency)
  → allocate bits (importance → per-layer QuantMode)
  → select active layers (top-N by importance)
  → hot-only inference (skip inactive layers, AWQ-quantize active layers on load)
  → generate tokens
```

### Profile-Guided Quantization Flow
```
1. profile_layer_importance()
   ├── For each layer 0..N:
   │   ├── Load layer (bf16 from safetensors)
   │   ├── Forward: compute FFN + Q/V projections
   │   ├── Record importance (L2 norm of output)
   │   ├── Record AWQ saliency (mean |activation| per channel)
   │   └── Evict layer (free RAM)
   └── Return: importance[], AwqCalibration

2. allocate_bits(importance, target_bpw)
   └── Return: per-layer QuantMode (Q8/Q4/Q3/Q2/Q1/None)

3. For each layer during inference:
   ├── Load layer (bf16 from safetensors)
   ├── quantize_adaptive(assigned_mode, awq_saliency)
   │   ├── Compute AWQ scales: α = saliency^0.5
   │   ├── Scale weights: W[:,j] *= α[j]
   │   ├── Quantize at assigned bit rate
   │   └── Store inv_α for inference correction
   ├── Drop bf16 original (save RAM)
   └── Run inference with quantized weights
```

## File Map

```
nve/src/
├── main.rs              # CLI entry point, command dispatch
├── paged_model.rs       # PagedModel: tiered inference + streaming profiler
├── generic_model.rs     # GenericModel: all-in-RAM inference, GenericBlockWeights
├── quantize.rs          # All quantization formats, WeightStorage, AWQ, bit allocation
├── arch.rs              # Architecture detection, UnifiedConfig
├── weight_map.rs        # Per-architecture weight name mapping
├── attention.rs         # GQA attention with WeightStorage dispatch
├── ops.rs               # RMSNorm, LayerNorm, SwiGLU, GELU, RoPE, sampling
├── tensor.rs            # Tensor (f32), CompactTensor (bf16), matmul ops
├── safetensors.rs       # Safetensors file loading (mmap, sharded)
├── tokenizer.rs         # BPE tokenizer (HuggingFace tokenizer.json)
├── hub.rs               # HuggingFace Hub model download
├── model.rs             # Legacy Llama-specific model (deprecated)
├── config.rs            # Legacy Llama config parser
├── tier.rs              # Memory tier data structures
├── cluster.rs           # Co-activation weight clustering
├── profiler.rs          # MCAP profiler (Rust core)
├── pager.rs             # Weight paging system
├── benchmark.rs         # Benchmark harness
└── lib.rs               # Library root + FFI exports
```
