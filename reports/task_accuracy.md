# NVE Task Accuracy — Complete Results

**Date:** 2026-04-13  
**Platform:** Modal cloud (Intel x86-64, 8 vCPU, 32 GB RAM)

---

## Accuracy by Model, Scenario, and Configuration

Accuracy = fraction of 8 ABC tasks answered correctly. OOM = container killed by allocator before inference began.

### Llama-3.2-1B

| Scenario | NVE Baseline | NVE A (Quant) | NVE B (Hot) | NVE C (PG+AWQ) | llama.cpp Q4 | llama.cpp Q8 | HF fp32 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 88% | 88% | 88% | 0% | 88% | 100% | 100% |
| constrained_2gb | 88% | 88% | 88% | 0% | 88% | 100% | OOM |

### Llama-3.2-3B

| Scenario | NVE Baseline | NVE A (Quant) | NVE B (Hot) | NVE C (PG+AWQ) | llama.cpp Q4 | llama.cpp Q8 | HF fp32 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 100% | 100% | 100% | 100% | 100% | 100% | — |
| constrained_2gb | **100%** | **100%** | 0% | 0% | OOM | OOM | OOM |

### Llama-3.1-8B

| Scenario | NVE Baseline | NVE A (Quant) | NVE B (Hot) | NVE C (PG+AWQ) | llama.cpp Q4 | llama.cpp Q8 | HF fp32 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 88% | 88% | 88% | **100%** | 100% | 100% | 88% |
| constrained_4gb | **88%** | **88%** | 0% | 0% | OOM | OOM | OOM |
| constrained_8gb | **88%** | **88%** | 12% | 0% | 100% | OOM | OOM |

---

## Throughput (tok/s) by Model and Scenario

### Llama-3.2-1B

| Scenario | NVE Baseline | NVE A | NVE B | NVE C | llama.cpp Q4 | llama.cpp Q8 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 3.2 | 3.7 | 4.0 | 3.7 | 3.4 | 12.8 |
| constrained_2gb | 4.2 | 3.6 | 4.9 | 4.2 | 4.0 | 12.0 |

### Llama-3.2-3B

| Scenario | NVE Baseline | NVE A | NVE B | NVE C | llama.cpp Q4 | llama.cpp Q8 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 2.1 | 1.6 | **2.6** | 2.1 | 1.6 | 3.1 |
| constrained_2gb | 2.3 | 1.6 | 5.9† | 4.1† | OOM | OOM |

†High throughput but 0% accuracy — skipped layers break residual stream coherence.

### Llama-3.1-8B

| Scenario | NVE Baseline | NVE A | NVE B | NVE C | llama.cpp Q4 | llama.cpp Q8 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| unconstrained | 0.8 | 0.7 | 1.0 | 0.8 | 0.9 | 2.2 |
| constrained_4gb | 1.2 | 0.7 | 3.4† | 2.3† | OOM | OOM |
| constrained_8gb | 0.6 | 0.9 | 1.1† | 1.4† | 0.7 | OOM |

†Partially or fully incoherent output.

---

## Hot-Only Viability: Active Layer Ratio

Config B and C require a minimum fraction of layers to remain active for coherent output. The empirical floor is approximately 50%.

| Model | Scenario | Active Layers | Ratio | B Accuracy | Status |
|-------|----------|---------------|-------|------------|--------|
| 3B | unconstrained | 28/28 | 100% | 100% | Coherent |
| 3B | constrained_2gb | 10/28 | 36% | 0% | **Below floor** |
| 8B | unconstrained | 32/32 | 100% | 88% | Coherent |
| 8B | constrained_4gb | 8/32 | 25% | 0% | **Below floor** |
| 8B | constrained_8gb | 16/32 | 50% | 12% | **At floor** |

When below the floor, paging (Baseline/A) is the correct strategy: all layers load sequentially from disk, maintaining full residual stream coherence at any memory budget.

---

## Unique Capabilities by System

| Capability | NVE | llama.cpp | HF fp32 |
|-----------|:---:|:---------:|:-------:|
| Run 3B at 2 GB | ✓ | ✗ OOM | ✗ OOM |
| Run 8B at 4 GB | ✓ | ✗ OOM | ✗ OOM |
| Run 8B at 8 GB | ✓ | ✓ (Q4 only) | ✗ OOM |
| 100% accuracy on 8B unconstrained | ✓ Config C | ✓ Q4/Q8 | ✗ 88% |
| Profile portability (0ms on-device) | ✓ | ✗ | ✗ |
| No GGUF format dependency | ✓ | ✗ | ✓ |
| Direct HuggingFace weight loading | ✓ | ✗ | ✓ |

---

## Notes

- All NVE constrained runs used pre-computed importance profiles injected via `--profile-from` (profiling_time_ms = 0 on constrained device).
- llama.cpp OOM: container killed before generating any tokens.
- HF fp32 3B unconstrained not collected (memory fits but run not included in test matrix).
- 1B Config C fails at both scenarios — the 2.0 bpw target is too aggressive for a 16-layer model.
