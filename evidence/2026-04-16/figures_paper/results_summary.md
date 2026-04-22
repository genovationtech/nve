# NVE Experiment Results Summary

Generated: 2026-04-16T17:42:08

---

## Table 1: ABC Configuration Results Through 8B

| Model | Config | Task Acc. | Tok/s | Peak Mem (MB) | Source | Notes |
|-------|--------|-----------|-------|---------------|--------|-------|
| GPT-2 (0.1B) | Baseline | 62% | 30.9 | 411 | local |  |
| GPT-2 (0.1B) | A: Unif. Q4 | 38% | 24.1 | 420 | local |  |
| GPT-2 (0.1B) | B: Profiled Hot | 62% | 30.2 | 895 | local |  |
| GPT-2 (0.1B) | C: PG+AWQ | 75% | 27.3 | 838 | local |  |
| Qwen2.5 (0.5B) | Baseline | 100% | 7.1 | 999 | local |  |
| Qwen2.5 (0.5B) | A: Unif. Q4 | 88% | 8.3 | 1001 | local |  |
| Qwen2.5 (0.5B) | B: Profiled Hot | 100% | 9.3 | 1942 | local |  |
| Qwen2.5 (0.5B) | C: PG+AWQ | 75% | 8.7 | 1731 | local |  |
| Llama-3.2 (1.2B) | Baseline | 88% | 5.6 | 2699 | clean Modal |  |
| Llama-3.2 (1.2B) | A: Unif. Q4 | 88% | 4.2 | 1426 | clean Modal |  |
| Llama-3.2 (1.2B) | B: Profiled Hot | 88% | 5.3 | 5055 | clean Modal |  |
| Llama-3.2 (1.2B) | C: PG+AWQ | 0% | 4.8 | 4427 | clean Modal | ⚠ degenerate at 2.0 bpw (confirmed clean hardware) |
| Llama-3.2 (3.2B) | Baseline | 100% | 1.9 | 7109 | clean Modal |  |
| Llama-3.2 (3.2B) | A: Unif. Q4 | 100% | 1.6 | 3423 | clean Modal |  |
| Llama-3.2 (3.2B) | B: Profiled Hot | 100% | 1.9 | 13240 | clean Modal |  |
| Llama-3.2 (3.2B) | C: PG+AWQ | 100% | 2.1 | 11409 | clean Modal |  |


## Table 2: Layer Sweep — Llama-3.2-1B, Config B

| N (active layers) | Layer Fraction | Task Accuracy | Coherent? |
|-------------------|----------------|---------------|-----------|
| 2 | 12% | 0% | ✗ |
| 4 | 25% | 0% | ✗ |
| 6 | 38% | 0% | ✗ |
| 8 | 50% | 12% | ✓ |
| 10 | 62% | 0% | ✗ |
| 12 | 75% | 38% | ✓ |
| 14 | 88% | 75% | ✓ |
| 16 | 100% | 88% | ✓ |


## Table 3: Scorer Signal Analysis — Kendall's τ

| Model | FFN-only τ | Attn-proxy τ | Input-L2 τ | Dominant at scale |
|-------|-----------|-------------|-----------|-------------------|
| GPT-2 (0.1B) | 0.970 | 0.515 | 0.455 | **FFN** |
| Qwen2.5 (0.5B) | 0.638 | 0.659 | 0.623 | **Attn-proxy** |
| Llama (1.2B) | 0.733 | 0.767 | 0.183 | **Attn-proxy** |
| Llama (3.2B) | 0.646 | 0.815 | 0.450 | **Attn-proxy** |
