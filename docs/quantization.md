# NVE Quantization System

Pure Rust quantization engine with AVX2 SIMD acceleration. No Python, PyTorch, or external dependencies.

## Quantization Formats

### Q8_0 (8-bit)

```
Block (36 bytes for 32 weights):
┌─────────────┬────────────────────────────────────┐
│ f32 scale   │ 32 × i8 values                     │
│ (4 bytes)   │ (32 bytes)                          │
└─────────────┴────────────────────────────────────┘
```

- **Levels:** 256 (-128 to +127)
- **Effective bpw:** ~9 bits (36/32 × 8)
- **Compression vs bf16:** 1.78x
- **Dequant:** `value = i8_val * scale`
- **Quality:** Near-lossless. Almost indistinguishable from bf16.

### Q4_0 (4-bit)

```
Block (20 bytes for 32 weights):
┌─────────────┬────────────────────────────────────┐
│ f32 scale   │ 16 bytes packed nibbles             │
│ (4 bytes)   │ (2 weights per byte, lo|hi nibble)  │
└─────────────┴────────────────────────────────────┘
```

- **Levels:** 16 (-8 to +7, stored as unsigned 0-15)
- **Effective bpw:** 5 bits (20/32 × 8)
- **Compression vs bf16:** 3.2x
- **Dequant:** `value = (nibble - 8) * scale`
- **Quality:** Good for most tasks. Standard quantization level.
- **Compatible with:** llama.cpp Q4_0 block format

### Q3 (3-bit)

```
Block (16 bytes for 32 weights):
┌─────────────┬────────────────────────────────────┐
│ f32 scale   │ 12 bytes packed (3 bits/weight)     │
│ (4 bytes)   │ (96 bits = 32 × 3)                  │
└─────────────┴────────────────────────────────────┘
```

- **Levels:** 8 (-4 to +3, stored as unsigned 0-7)
- **Effective bpw:** 4 bits (16/32 × 8)
- **Compression vs bf16:** 4x
- **Packing:** Linear bit packing. Weight i occupies bits `[i*3, i*3+3)` across the 12-byte data region.
- **Quality:** Noticeable degradation on complex tasks. Good for less important layers.

### Q2 (2-bit)

```
Block (12 bytes for 32 weights):
┌─────────────┬────────────────────────────────────┐
│ f32 scale   │ 8 bytes packed (2 bits/weight)      │
│ (4 bytes)   │ (4 weights per byte)                │
└─────────────┴────────────────────────────────────┘
```

- **Levels:** 4 (-2, -1, 0, +1, stored as unsigned 0-3)
- **Effective bpw:** 3 bits (12/32 × 8)
- **Compression vs bf16:** 5.3x
- **Packing:** 4 weights per byte. Weight at position `p` in byte: `(byte >> (p*2)) & 0x03`
- **Quality:** Significant degradation. Best paired with AWQ to protect important channels.

### Q1 (1-bit Ternary)

```
Block (12 bytes for 32 weights):
┌─────────────┬──────────────┬──────────────┐
│ f32 scale   │ nonzero mask │ sign bits    │
│ (4 bytes)   │ (4 bytes)    │ (4 bytes)    │
└─────────────┴──────────────┴──────────────┘
```

- **Values:** {-1, 0, +1} × scale
- **Effective bpw:** 3 bits (12/32 × 8), but information content is ~1.58 bits (same as BitNet)
- **Compression vs bf16:** 5.3x
- **Encoding:**
  - `nonzero_mask`: bit j = 1 if weight j is nonzero
  - `sign_bits`: bit j = 1 if weight j is negative (only meaningful when nonzero)
  - `scale`: mean absolute value of nonzero weights in the block
- **Threshold:** weights with `|w| < 0.33 * max(|w|)` in the block are zeroed
- **Quality:** BitNet-level. Only for least important layers.

### Sparse (Sub-1-bit)

```
Structure:
┌──────────────────────────────────────────────┐
│ active_bitmap: 1 bit per block               │
│ active_data: only active blocks (Q4 format)  │
│ inactive blocks → zero (no storage)          │
└──────────────────────────────────────────────┘
```

- **Effective bpw:** `keep_fraction × 5` (e.g., 6% kept → 0.3 bpw)
- **Compression vs bf16:** up to 50x+
- **How it works:**
  1. Compute L2 norm of each 32-weight block
  2. Sort blocks by norm (importance proxy)
  3. Keep top `keep_fraction` blocks, quantize to Q4
  4. Zero all other blocks
- **Inference:** bitmap scan → skip zero blocks → Q4 dot for active blocks
- **Quality:** Only useful at extreme compression with profiler guidance.

## AWQ (Activation-Aware Weight Quantization)

AWQ protects important weights **within each layer** by scaling them up before quantization so they occupy higher quantization levels.

### The AWQ Trick

Standard quantization treats all weights equally:
```
W_original:  [0.8, 0.002, 0.7, 0.001]   # important and unimportant mixed
W_quantized: [1.0, 0.0,   1.0, 0.0  ]   # both important and unimportant rounded same way
```

AWQ scales important weights up before quantizing:
```
saliency:    [high, low, high, low]
awq_scales:  [2.0,  0.5, 2.0,  0.5]      # α = saliency^0.5
W_scaled:    [1.6, 0.001, 1.4, 0.0005]   # important weights amplified
W_quantized: [2.0, 0.0,   1.0, 0.0  ]    # important weights get better precision
inv_scales:  [0.5, 2.0,   0.5, 2.0  ]    # stored for inference correction
```

At inference: `x_corrected[j] = x[j] * inv_scales[j]` before matmul. The scaling cancels out mathematically, but the important weights were quantized at higher effective precision.

### AWQ Scale Computation

```
saliency[j] = mean(|activation[j]|) across calibration tokens
awq_scales[j] = saliency[j]^0.5            # per AWQ paper's optimal exponent
inv_scales[j] = 1.0 / awq_scales[j]
```

The 0.5 exponent balances the quantization error between weight and activation channels. It's the analytical optimum from the AWQ paper (MIT Han Lab, 2023).

### Per-Projection AWQ

Different projections receive different AWQ scales based on their input dimensionality:

| Projection | AWQ Scale Source | Dimension |
|---|---|---|
| Q, K, V, O | hidden_saliency | hidden_size |
| gate, up | hidden_saliency | hidden_size |
| down | intermediate_saliency | intermediate_size |

## Profile-Guided Quantization

The combined algorithm that makes NVE unique.

### What It Does

Assigns different quantization levels to different layers based on profiled importance, with AWQ protecting important channels within each layer.

### How It Works

```
Step 1: Stream-profile all layers (21s for 3B model)
  └── Per layer: importance score + per-channel AWQ saliency

Step 2: Allocate bits given target budget (e.g., 2.0 bpw)
  └── Greedy: most important layers get highest precision
  
  Example for Llama 3.2 3B at pg:2.0:
    Layer  1 (importance 632) → Q8   (8 bits, best precision)
    Layer 27 (importance 690) → Q8
    Layer 26 (importance  69) → Q8
    Layer  0 (importance  31) → Q4   (4 bits)
    Layer 25 (importance  30) → Q4
    Layer 13 (importance  21) → Q4
    Layer 12 (importance  19) → Q4
    Layer 24 (importance  19) → Q4
    Layers 4-11, 14-23       → None  (0 bits, pruned)

Step 3: Quantize each layer on load with AWQ
  └── Load bf16 → apply AWQ scaling → quantize at assigned rate → drop bf16

Step 4: Inference with per-layer mixed precision
  └── Important layers: Q8 with AWQ channel protection
  └── Medium layers: Q4 with AWQ  
  └── Unimportant layers: skipped entirely
```

### Bit Allocation Algorithm

```rust
allocate_bits(layer_importance, target_bpw, layer_params) → Vec<QuantMode>
```

Greedy optimization:
1. All layers start at 0 bits (pruned)
2. Available rates: [Q8=8, Q4=4, Q3=3, Q2=2, Q1=1, None=0]
3. Total budget = target_bpw × total_params
4. Loop:
   - For each layer not at max rate: compute `efficiency = importance / bit_cost_of_upgrade`
   - Upgrade the most efficient layer by one step
   - Stop when budget exhausted or no upgrades fit

This is optimal in the greedy sense: each bit is spent where it produces the most value.

### Extreme Compression Examples

**pg:2.0 (2 bits/weight, 3B model → ~800 MB)**
```
3 layers at Q8 + 5 layers at Q4 + 20 layers pruned
AWQ protects important channels in active layers
```

**pg:0.5 (0.5 bits/weight, 3B model → ~200 MB)**
```
1 layer at Q8 + 1 layer at Q4 + 1 layer at Q2 + 25 layers pruned
Only 3 of 28 layers kept — profiler picks the most critical
```

**pg:0.25 (0.25 bits/weight, 3B model → ~100 MB)**
```
1 layer at Q4 + 1 layer at Q1 + 26 layers pruned
Entire model in ~100 MB, still produces English words
```

## CLI Usage

```bash
# Standard quantization (uniform, all layers same precision)
nve generate -m model --paged --quantize q4    # uniform Q4
nve generate -m model --paged --quantize q8    # uniform Q8
nve generate -m model --paged --quantize q2    # uniform Q2
nve generate -m model --paged --quantize q1    # uniform Q1 (ternary)

# Profile-guided non-uniform quantization (different per layer + AWQ)
nve generate -m model --paged --hot-only --profile --quantize pg:2.0
nve generate -m model --paged --hot-only --profile --quantize pg:1.0
nve generate -m model --paged --hot-only --profile --quantize pg:0.5
nve generate -m model --paged --hot-only --profile --quantize pg:0.25
```

## SIMD Implementation

All quantized formats have AVX2+FMA SIMD dot product kernels with scalar fallbacks.

**Runtime detection pattern:**
```rust
fn q4_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q4_dot_row_avx2(row_data, x, blocks_per_row) };
    }
    q4_dot_row_scalar(row_data, x, blocks_per_row)
}
```

**AVX2 strategy per format:**
- **Q8:** Load 8 i8 → sign-extend to i32 → cvt to f32 → FMA with x → scale
- **Q4:** Unpack nibbles from 4 bytes → 8 i32 → subtract 8 → cvt to f32 → FMA
- **Q3:** Extract 3-bit values from packed bytes → i32 → subtract 4 → cvt to f32 → FMA
- **Q2:** Mask 2-bit values (4 per byte) → i32 → subtract 2 → cvt to f32 → FMA
- **Q1:** Extract nonzero/sign bits → conditional add/subtract x values → scale

All use `_mm256_fmadd_ps` for fused multiply-add accumulation with horizontal reduction at the end.

## Memory Savings

For a 3B model (Llama 3.2 3B, 28 layers, 6.0 GB bf16):

| Mode | Model Size | Compression | Notes |
|---|---|---|---|
| bf16 | 6.0 GB | 1x | Baseline |
| Q8 | 3.4 GB | 1.8x | Near-lossless |
| Q4 | 1.9 GB | 3.2x | Standard |
| Q2 | 1.1 GB | 5.3x | Aggressive |
| Q1 | 1.1 GB | 5.3x | Ternary |
| pg:2.0 | ~0.8 GB | 7.5x | Profile-guided, 8 layers active |
| pg:0.5 | ~0.2 GB | 30x | Profile-guided, 3 layers active |
| pg:0.25 | ~0.1 GB | 60x | Profile-guided, 2 layers active |

## Comparison with Other Frameworks

| Feature | NVE | llama.cpp | BitNet.cpp | GPTQ | AWQ |
|---|---|---|---|---|---|
| Language | Rust | C++ | C++ | Python | Python |
| Formats | Q1-Q8+Sparse | Q2-Q8+K-quants | 1.58-bit | 4-bit | 4-bit |
| Per-layer bit allocation | **Yes** (profiler-guided) | No | No | No | No |
| AWQ channel protection | **Yes** | No | No | No | Yes |
| Combined profiler+AWQ | **Yes** | No | No | No | No |
| Runtime profiling | **Yes** (streaming) | No | No | Offline | Offline |
| Works on any safetensors | **Yes** | GGUF only | BitNet models only | Per-model | Per-model |
| Paged inference | **Yes** | mmap | No | No | No |
| SIMD | AVX2+FMA | AVX2/AVX512/NEON | AVX2/NEON | CUDA | CUDA |
| GPU required | No | No | No | Yes | Yes |
