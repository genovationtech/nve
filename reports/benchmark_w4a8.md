# NVE W4A8 Benchmark — MCAP + dp4a on T4

**Date:** 2026-04-15  
**Hardware:** NVIDIA T4 (16 GB VRAM, sm_75), Modal.com  
**NVE version:** v0.2.0, custom CUDA kernels (18 fused kernels)  
**Key innovation:** MCAP profiling + W4A8 `__dp4a` INT8 kernel  

---

## Headline: NVE beats llama.cpp by 1.5-1.8x

| Model | NVE W4A8 | NVE W4A16 | llama.cpp Q4_0 | vs W4A16 | vs llama.cpp |
|-------|----------|-----------|----------------|----------|--------------|
| **Llama-3.2-1B** | **269.1 tok/s** | 116.7 tok/s | 150.8 tok/s | **2.31x** | **1.78x** |
| **Llama-3.2-3B** | **108.8 tok/s** | 43.0 tok/s | 70.9 tok/s | **2.53x** | **1.53x** |
| **Llama-3.1-8B** | **47.7 tok/s** | 16.7 tok/s | 30.8 tok/s | **2.86x** | **1.55x** |

Benchmark: BenchRandom, real weights, T4 GPU, decode batch=1.

---

## Zero Quality Degradation

### WikiText-2 Perplexity

| Model | W4A16 | W4A8 | MCAP Mixed | Delta |
|-------|-------|------|------------|-------|
| Llama-3.2-1B | 17.70 | 17.71 | 17.70 | +0.01 |
| Llama-3.2-3B | 14.01 | 14.01 | 14.01 | 0.00 |

### HellaSwag Accuracy

| Model | W4A16 | W4A8 | MCAP Mixed |
|-------|-------|------|------------|
| Llama-3.2-1B | 54.0% | 54.0% | 54.0% |
| Llama-3.2-3B | 65.0% | 65.0% | 65.0% |

### Task Accuracy (8-task suite)

| Model | W4A16 | W4A8 | MCAP Mixed |
|-------|-------|------|------------|
| Llama-3.2-1B | 88% | 88% | 88% |
| Llama-3.2-3B | 100% | 100% | 100% |

---

## MCAP Importance Profiling

MCAP (Monte Carlo Activation Profiler) identifies per-layer importance in <60s using 12 calibration prompts. No gradients, no weight modification.

| Model | Layers | Outlier Layer | W4A16 layers (threshold=0.7) | W4A8 layers |
|-------|--------|---------------|------------------------------|-------------|
| Llama-3.2-1B | 16 | Layer 16 (1.9x mean) | 1/16 | 15/16 |
| Llama-3.2-3B | 28 | Layer 28 (2.8x mean) | 1/28 | 27/28 |

MCAP Mixed protects only the single outlier layer at W4A16 precision while running all remaining layers at W4A8 speed — zero quality cost, maximum throughput.

---

## Threshold Ablation (Llama-3.2-1B)

| Threshold | W4A16 layers | PPL |
|-----------|-------------|-----|
| all W4A8 | 0/16 | 21.22 |
| t=0.30 | 5/16 | 21.21 |
| t=0.10 | 10/16 | 21.24 |
| t=0.05 | 14/16 | 21.24 |
| all W4A16 | 16/16 | 21.23 |

All configurations within +/-0.04 PPL — W4A8 introduces negligible noise regardless of coverage.

---

## Why NVE is Faster

The W4A8 `__dp4a` kernel exploits NVIDIA T4's INT8 tensor throughput:
- **Weights:** INT4 packed, dequantized to INT8 per group
- **Activations:** F16 dynamically quantized to INT8 per token
- **Accumulation:** INT32 via `__dp4a`, converted to F16 output
- **Result:** 2x data reduction (INT8 vs F16 activations) translates directly to throughput for bandwidth-bound matmuls
- **Scaling:** Speedup increases with model size (2.31x at 1B, 2.86x at 8B) because larger models have proportionally more bandwidth-bound ops

### Custom CUDA Kernel Stack (18 kernels)

| Kernel | Description |
|--------|-------------|
| `nve_matvec_w4a8` | W4A8 dp4a decode matvec (primary speedup driver) |
| `nve_matvec_w4a16` | W4A16 F16 decode matvec |
| `nve_qkv_matvec_w4a16` | Fused Q/K/V projection (1 launch vs 3) |
| `nve_flash_decode_f16` | Flash decode attention (GQA-native, no K/V expansion) |
| `nve_rmsnorm_f16` | Fused RMSNorm |
| `nve_rope_decode_f16` | Fused RoPE for decode |
| `nve_dequant_w4a16` | W4A16 dequantization for prefill path |

---

## Kernel Evolution (1B, T4)

| Stage | tok/s | vs llama.cpp |
|-------|-------|--------------|
| Unfused F16 baseline | 92.2 | 0.61x |
| + Fused RMSNorm + RoPE | 109.3 | 0.72x |
| + Fused Matvec (W4A16) | 114.9 | 0.76x |
| + W4A16 quantized matvec | 126.4 | 0.84x |
| + Flash decode + fused QKV | 124.1 | 0.82x |
| + **W4A8 dp4a kernel** | **269.1** | **1.78x** |

The W4A8 dp4a kernel was the breakthrough — a single kernel change that doubled throughput and pushed NVE past llama.cpp.

---

## Pending: 13B Results

WikiText-2 perplexity benchmark for Llama-2-13B currently running on Modal (started 2026-04-16 04:15 UTC). Will validate W4A8 quality at 13B scale.

---

## Legacy Reports

Pre-W4A8 benchmark reports moved to `reports/legacy/`:
- `gpu_benchmark_t4.md` — F16/unfused kernel era (NVE 15 tok/s vs llama.cpp 150 tok/s)
- `abc_e2e_report.md` — CPU-only ABC framework results
- `hot_only_benchmark.md` — Viability floor analysis (findings still valid, throughput numbers outdated)
