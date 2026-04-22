# NVE Streaming Profiler

The streaming profiler is NVE's core differentiator. It measures per-layer importance and per-channel activation saliency for any model, using less RAM than the model itself.

## Problem

A 7B model is ~14 GB in bf16. On a machine with 4 GB RAM, you can't load all layers to profile them. Traditional profiling loads the entire model, runs calibration data, and records statistics — requiring the full model in memory.

## Solution: Stream One Layer at a Time

```
For each layer i = 0..N:
  1. Load layer i from safetensors (mmap'd, ~200-400 MB)
  2. Forward: compute FFN output + attention projections
  3. Measure: L2 norm of residual delta (importance)
  4. Record: per-channel mean |activation| (AWQ saliency)
  5. Evict layer i from RAM
  6. Move to layer i+1

Peak memory: embedding + 1 layer + hidden state
```

For a 3B model with 192 MB/layer: peak memory ~1 GB to profile all 28 layers of a 6 GB model.

## What It Measures

### Layer Importance

For each layer, the profiler measures the magnitude of the layer's contribution to the residual stream:

```
importance[i] = L2_norm(hidden_after_layer_i - hidden_before_layer_i)
```

Specifically, it runs the FFN path (the dominant computation) and measures:
- FFN output magnitude (SwiGLU/GELU/GeGLU output norm)
- Attention proxy: Q and V projection output norms

These are combined into a single importance score per layer.

### AWQ Channel Saliency

For each layer, the profiler also collects per-input-channel activation magnitudes:

```
hidden_saliency[j] = mean(|activation_j|) across tokens
                     where j indexes the hidden dimension
                     
intermediate_saliency[j] = mean(|ffn_intermediate_j|) across tokens
                            where j indexes the intermediate (FFN) dimension
```

These are used by the AWQ quantization to determine which weight channels to protect.

## Importance Distributions by Architecture

Different architectures have fundamentally different importance distributions. The profiler discovers these automatically.

### Llama 3.2 3B (28 layers) — U-shaped

```
Layer  0:  31.4  ██
Layer  1: 631.8  ████████████████████████████████████████████████████  ← peak
Layer  2:  17.2  █
...middle layers: 12-18 (flat)...
Layer 26:  69.3  █████
Layer 27: 689.7  ██████████████████████████████████████████████████████  ← peak
```

- Max/min ratio: **56x**
- Top-2 layers hold **80%** of total signal
- Middle 50% of layers hold only **8%**
- Profiler action: keep first + last layers, prune middle

### Phi-3.5-mini (32 layers) — Monotonically Increasing

```
Layer  0:  40.8  ████
Layer  8:  87.0  █████████
Layer 16: 121.5  ████████████
Layer 24: 216.5  ██████████████████████
Layer 31: 528.9  ████████████████████████████████████████████████████  ← peak
```

- Max/min ratio: **13x**
- Later layers always more important than earlier ones
- Profiler action: keep last N layers, prune earliest

### Key Insight

A static heuristic ("keep first and last layers") works for Llama but fails for Phi-3. The profiler adapts automatically to any architecture because it measures actual activation patterns, not assumed distributions.

## Integration with Quantization

The profiler output drives two downstream systems:

### 1. Layer-Level Bit Allocation

```
profile_layer_importance()
  → importance scores: [31.4, 631.8, 17.2, ..., 689.7]
  → allocate_bits(importance, target_bpw=2.0)
  → assignments: [Q4, Q8, None, ..., Q8]
```

Each layer gets a different quantization level. Most important layers get Q8 (8 bits), least important get pruned (0 bits).

### 2. Channel-Level AWQ Protection

```
profile_layer_importance()
  → AWQ saliency per layer: {hidden: [0.3, 12.1, 0.1, ...], intermediate: [...]}
  → compute_awq_scales(saliency)
  → awq_scales: [0.55, 3.48, 0.32, ...]  (α = s^0.5)
  → quantize_with_awq(weights, scales, Q4)
```

Within each layer, important channels get scaled up before quantization, preserving them at higher effective precision.

### Combined Effect

```
Layer 1 (importance 632):
  → Assigned Q8 (high precision)
  → AWQ protects channels with high saliency
  → Result: critical weights at ~8-bit effective precision

Layer 17 (importance 11):
  → Assigned None (pruned)
  → Not loaded at all
  → Result: zero memory, zero compute

Layer 25 (importance 30):
  → Assigned Q4 (medium precision)
  → AWQ protects channels with high saliency
  → Important channels at ~6-7 bit effective precision
  → Unimportant channels at ~3-4 bit effective precision
```

## Usage

### Basic Profiling (hot-only layer selection)

```bash
nve generate -m model --paged --hot-only --profile \
  --hot-budget-mb 250 --warm-budget-mb 774 \
  -p "The theory of general relativity"
```

This profiles all layers, selects the most important ones that fit in the RAM budget, and runs inference using only those layers.

### Profile-Guided Quantization

```bash
nve generate -m model --paged --hot-only --profile \
  --quantize pg:2.0 \
  -p "The theory of general relativity"
```

This profiles, allocates bits per layer, applies AWQ-scaled quantization on load, and runs inference with mixed precision.

## Performance

| Model | Layers | Profiling Time | Peak RAM During Profile |
|---|---|---|---|
| Llama 3.2 1B | 16 | ~10s | ~500 MB |
| Llama 3.2 3B | 28 | ~21s | ~950 MB |
| Phi-3.5-mini | 32 | ~110s | ~850 MB |

Profiling is a one-time cost per model (or per session). The importance scores can be cached for subsequent runs.

## Experimental Results

### At 1 GB RAM, Llama 3B with 5/28 Layers

| Method | English Words per Prompt | Subword Loops |
|---|---|---|
| **Profiled** [0,1,25,26,27] | 6-12 | 2/8 prompts |
| Evenly-spaced [0,7,14,21,27] | 0-1 | **8/8 prompts** |

The profiled model retains vocabulary, grammar, punctuation, and even thematic awareness. The evenly-spaced model produces only subword repetition loops.

### Why Profiled Wins

The profiler selects layers where 96% of the signal energy lives (first 2 + last 3 for Llama). Evenly-spaced wastes 3 of 5 layer slots on the flat middle where each layer contributes only ~1% of signal.

At 1 GB with 5 layers, you get 18% of the model. The profiler ensures that 18% captures 96% of the important computation. Evenly-spaced captures ~35%.
