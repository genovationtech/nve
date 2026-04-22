# NVE Benchmark Results

All benchmarks run on: Linux x86_64, 3.9 GB RAM, 2 CPU cores, no GPU, SSD storage.

## NVE Engine Speed (No Paging)

When the full model fits in RAM, NVE's Rust bf16 engine is competitive:

| Framework | Model | Tok/s | RAM | Language |
|---|---|---|---|---|
| **NVE** | Llama 3.2 1B (bf16) | **2.2** | 2,400 MB | Rust |
| BitNet.cpp | BitNet-2B (1.58-bit) | 1.07 | 1,125 MB | C++ |
| HuggingFace | Llama 3.2 1B (bf16) | 0.35 | 2,853 MB | Python |

NVE is 2x faster than BitNet.cpp and 6.3x faster than HuggingFace Python on the same hardware.

## Paged Inference at Memory Constraints

### 500 MB RAM Cap — Llama 1B vs 3B

| | 1B (16 layers) | 3B (28 layers) |
|---|---|---|
| Layers in RAM | 4 (25%) | 2 (7%) |
| Full paged tok/s | 0.07-0.14 | 0.03 |
| Hot-only tok/s | 4.3-4.6 | 2.5-3.5 |
| Full paged quality | Excellent | Excellent |
| Hot-only quality | Gibberish | Gibberish |

**Finding:** 1B wins at 500 MB. Fewer cold layers to page from SSD.

### 1 GB RAM Cap — Profiled vs Evenly-Spaced Hot-Only

**Llama 3.2 3B, 5/28 layers active:**

| Prompt | Profiled | Evenly-Spaced |
|---|---|---|
| General relativity | "production effect light full" | "Cardiff anhanhanhawk" |
| Photosynthesis | "useful covering full News Harrison changing" | "Override enactment □□□" |
| Water boils | "self sea possible purchase small excellent green" | "eenieeenieeenieeenie" |
| Shakespeare | "will use turn soul person live" | "amen enga enga enga" |

**Profiled: 48+ distinct English words across 8 prompts. Evenly-spaced: single-subword loops on 8/8.**

### Multi-Architecture: Phi-3.5 vs Llama 3B

| Metric | Phi-3.5 (32L) | Llama 3B (28L) |
|---|---|---|
| Importance distribution | Monotonic increasing | U-shaped |
| Max/min ratio | 13x | 51x |
| Profiled avg tok/s | 6.9 | 4.1 |
| Profiled quality | English fragments | English phrases |
| Evenly-spaced quality | Subword noise | Subword loops |

The profiler automatically adapts: tail-heavy selection for Phi-3, endpoint selection for Llama.

## Profile-Guided Quantization

**Llama 3.2 3B with `--quantize pg:X.X`:**

### pg:2.0 (2 bits/weight average)

```
Layer assignments:
  Layer  1 (imp 632) → Q8    Layer  0 (imp 31) → Q4
  Layer 27 (imp 690) → Q8    Layer 25 (imp 30) → Q4
  Layer 26 (imp  69) → Q8    Layer 12 (imp 19) → Q4
  Layers 4-11, 14-23 → None (pruned)

Active: 10/28 layers | Effective model size: ~800 MB
```

### pg:0.5 (0.5 bits/weight average)

```
Layer assignments:
  Layer 27 (imp 690) → Q8
  Layer  1 (imp 632) → Q4
  Layer 26 (imp  69) → Q2
  All other 25 layers → None (pruned)

Active: 3/28 layers | Effective model size: ~200 MB
Output: "n and and in in in in..." (still produces English)
```

## Importance Profile Data

### Llama 3.2 3B (28 layers)

```
Layer  0:  31.4    Layer 14:  15.7
Layer  1: 631.8 ★  Layer 15:  14.4
Layer  2:  17.2    Layer 16:  13.0
Layer  3:  17.3    Layer 17:  12.4 (min)
Layer  4:  16.6    Layer 18:  13.8
Layer  5:  16.7    Layer 19:  13.4
Layer  6:  17.3    Layer 20:  13.1
Layer  7:  15.5    Layer 21:  12.8
Layer  8:  15.6    Layer 22:  13.0
Layer  9:  16.1    Layer 23:  15.1
Layer 10:  17.0    Layer 24:  19.4
Layer 11:  16.5    Layer 25:  30.4
Layer 12:  18.5    Layer 26:  69.3
Layer 13:  20.7    Layer 27: 689.7 ★

Top-2 share: 80% | Middle 50% share: 8% | Max/min: 56x
```

### Phi-3.5-mini (32 layers)

```
Layer  0:  40.8    Layer 16: 121.5
Layer  1:  57.2    Layer 17: 130.3
Layer  2:  47.7    Layer 18: 131.5
Layer  3:  48.9    Layer 19: 145.9
Layer  4:  63.7    Layer 20: 152.6
Layer  5:  73.0    Layer 21: 164.5
Layer  6:  86.1    Layer 22: 174.0
Layer  7:  78.3    Layer 23: 204.2
Layer  8:  87.0    Layer 24: 216.5
Layer  9:  90.4    Layer 25: 244.7
Layer 10:  94.7    Layer 26: 259.3
Layer 11:  90.1    Layer 27: 297.1
Layer 12: 107.0    Layer 28: 316.9
Layer 13: 113.4    Layer 29: 343.5
Layer 14: 113.2    Layer 30: 378.4
Layer 15: 112.8    Layer 31: 528.9 ★

Top-4 share: 32% | Bottom-4 share: 4% | Max/min: 13x
```

## DeepSpeed and BitNet.cpp Comparison

Tested on the same hardware (3.9 GB RAM, CPU-only):

| Framework | Setup | Tok/s | RAM | Quality | Notes |
|---|---|---|---|---|---|
| NVE (no paging) | Llama 1B bf16 | **2.2** | 2,400 MB | Excellent | Pure Rust |
| NVE (paged, 500 MB) | Llama 1B bf16 | 0.08 | 464 MB | Excellent | SSD-paged |
| BitNet.cpp | BitNet-2B 1.58-bit | 1.07 | 1,125 MB | Excellent | Custom C++ |
| HuggingFace | Llama 1B bf16 | 0.35 | 2,853 MB | Excellent | PyTorch CPU |
| DeepSpeed | Llama 1B | N/A | N/A | N/A | Failed (torch version conflict) |

**Key finding:** The initial "13x gap" between NVE and BitNet was an unfair comparison (500 MB vs 1.1 GB). When both have enough RAM, NVE is 2x faster.

## Test Results

64 tests passing (as of 2026-04-09):

```
Quantization tests:
  test_q4_matvec, test_q4_memory_savings, test_quantize_block_q4_roundtrip
  test_q8_matvec, test_q8_memory_savings, test_quantize_block_q8_roundtrip
  test_q3_matvec, test_q3_roundtrip
  test_q2_matvec, test_q2_roundtrip
  test_q1_matvec, test_q1_roundtrip
  test_sparse_matvec
  test_awq_scaling
  test_bit_allocation, test_bit_allocation_high_budget, test_bit_allocation_zero_budget
  test_weight_storage_dispatch, test_weight_storage_all_variants
  test_quant_mode_from_str, test_quant_mode_new_variants

Architecture tests:
  test_detect_llama, test_detect_gpt2, test_detect_mistral

Inference tests:
  test_gqa_single_token, test_kv_cache
  test_rms_norm, test_silu, test_softmax, test_argmax, test_rope_basic
  test_compact_linear_vec, test_compact_matvec, test_dot_product, test_dot_large
  test_matmul_2x3_3x2, test_matmul_t
  ... and more
```
