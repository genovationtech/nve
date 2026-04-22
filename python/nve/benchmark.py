"""
NVE Benchmark — evaluates tiered serving against baselines.

Benchmark matrix:
  1. Full model (all weights on compute device)
  2. NVE static tiering (learned placement, no prefetch)
  3. NVE static tiering + prefetch

Measures:
  - First-token latency
  - Tokens/sec
  - p50/p95 latency
  - Page fault rate
  - Memory usage (GPU / RAM / SSD reads)
  - Output quality drift vs baseline
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from nve.manifest import TierManifest
from nve.device import DeviceManager
from nve.serving import TieredModelServer, BaselineServer, ServingStats


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    prompts: list[str] = field(default_factory=list)
    max_new_tokens: int = 30
    warmup_prompts: int = 1
    manifest_dir: Optional[str] = None
    ssd_dir: str = "/tmp/nve_bench_ssd"
    output_dir: str = "/tmp/nve_bench_results"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""
    name: str
    stats: dict
    generation_outputs: list[dict] = field(default_factory=list)
    logit_samples: list[torch.Tensor] = field(default_factory=list)


def compute_logit_drift(baseline_logits: list[torch.Tensor], test_logits: list[torch.Tensor]) -> dict:
    """
    Compute output quality drift between baseline and test logit distributions.

    Returns KL divergence and cosine similarity metrics.
    """
    if not baseline_logits or not test_logits:
        return {"error": "no logits to compare"}

    kl_divs = []
    cosine_sims = []
    top1_agreement = []

    for bl, tl in zip(baseline_logits, test_logits):
        # Compare last-position logits.
        b = bl[0, -1, :].float()
        t = tl[0, -1, :].float()

        # KL divergence.
        b_probs = F.softmax(b, dim=-1)
        t_log_probs = F.log_softmax(t, dim=-1)
        kl = F.kl_div(t_log_probs, b_probs, reduction="sum").item()
        kl_divs.append(kl)

        # Cosine similarity.
        cos = F.cosine_similarity(b.unsqueeze(0), t.unsqueeze(0)).item()
        cosine_sims.append(cos)

        # Top-1 agreement.
        top1_agreement.append(int(b.argmax() == t.argmax()))

    return {
        "kl_divergence_mean": round(float(np.mean(kl_divs)), 6),
        "kl_divergence_max": round(float(np.max(kl_divs)), 6),
        "cosine_similarity_mean": round(float(np.mean(cosine_sims)), 6),
        "cosine_similarity_min": round(float(np.min(cosine_sims)), 6),
        "top1_agreement": round(float(np.mean(top1_agreement)), 4),
    }


def memory_snapshot() -> dict:
    """Capture current memory usage."""
    import os
    # Read /proc/self/status for VmRSS.
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    return {"rss_mb": round(rss_kb / 1024, 1)}
    except (FileNotFoundError, ValueError):
        pass
    return {"rss_mb": 0}


class Benchmark:
    """
    Runs the full benchmark matrix.

    Usage:
        bench = Benchmark(model, tokenizer, manifest, config)
        results = bench.run()
        bench.print_report(results)
    """

    def __init__(
        self,
        model,
        tokenizer,
        manifest: TierManifest,
        config: BenchmarkConfig,
        low_vram: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.config = config
        self.low_vram = low_vram

    def run(self) -> dict[str, BenchmarkResult]:
        """Run all benchmark configurations and return results."""
        results = {}

        prompts = self.config.prompts
        if not prompts:
            prompts = [
                "The theory of general relativity states that",
                "def fibonacci(n):\n    ",
                "In a surprising turn of events, the",
                "The integral of e^x from 0 to infinity",
            ]

        # ── 1. Baseline: Full model ──
        if self.low_vram:
            print("\n  [Benchmark 1/3] Full model baseline... SKIPPED (low_vram mode)")
        else:
            print("\n  [Benchmark 1/3] Full model baseline...")
            results["baseline"] = self._run_baseline(prompts)

        # ── 2. NVE static tiering (no prefetch) ──
        print("  [Benchmark 2/3] NVE static tiering (no prefetch)...")
        results["nve_static"] = self._run_nve(prompts, enable_prefetch=False)

        # ── 3. NVE static tiering + prefetch ──
        print("  [Benchmark 3/3] NVE static tiering + prefetch...")
        results["nve_prefetch"] = self._run_nve(prompts, enable_prefetch=True)

        return results

    def _run_baseline(self, prompts: list[str]) -> BenchmarkResult:
        """Run baseline (all weights on device)."""
        # Deep copy model to avoid contamination.
        model_copy = copy.deepcopy(self.model)
        server = BaselineServer(model_copy, self.tokenizer)
        server.setup()

        mem_before = memory_snapshot()
        result = BenchmarkResult(name="Full Model (Baseline)", stats={})

        # Warmup.
        for p in prompts[: self.config.warmup_prompts]:
            server.generate(p, max_new_tokens=5)
        server.stats = ServingStats()  # Reset after warmup.

        # Collect logits.
        for prompt in prompts:
            logits = server.get_logits(prompt)
            result.logit_samples.append(logits.cpu())

        # Reset stats again for clean generation metrics.
        server.stats = ServingStats()

        # Generate.
        for prompt in prompts:
            out = server.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            result.generation_outputs.append(out)

        mem_after = memory_snapshot()
        result.stats = server.stats.to_dict()
        result.stats["memory"]["process_rss_mb"] = mem_after["rss_mb"]
        result.stats["memory"]["rss_delta_mb"] = round(
            mem_after["rss_mb"] - mem_before["rss_mb"], 1
        )

        server.teardown()
        del model_copy
        return result

    def _run_nve(self, prompts: list[str], enable_prefetch: bool) -> BenchmarkResult:
        """Run NVE tiered serving."""
        if self.low_vram:
            model_copy = self.model
        else:
            model_copy = copy.deepcopy(self.model)
        model_copy.eval()

        name = f"NVE {'+ prefetch' if enable_prefetch else '(no prefetch)'}"
        server = TieredModelServer(
            model_copy,
            self.tokenizer,
            self.manifest,
            ssd_dir=self.config.ssd_dir,
            enable_prefetch=enable_prefetch,
            prefetch_depth=2,
            low_vram=self.low_vram,
        )

        mem_before = memory_snapshot()
        server.setup()
        mem_after_setup = memory_snapshot()

        result = BenchmarkResult(name=name, stats={})

        # Warmup.
        for p in prompts[: self.config.warmup_prompts]:
            server.generate(p, max_new_tokens=5)
        server.stats = ServingStats()

        # Collect logits.
        for prompt in prompts:
            logits = server.get_logits(prompt)
            result.logit_samples.append(logits.cpu())

        # Reset for clean generation.
        server.stats = ServingStats()

        # Generate.
        for prompt in prompts:
            out = server.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            result.generation_outputs.append(out)

        mem_final = memory_snapshot()
        result.stats = server.stats.to_dict()
        result.stats["memory"]["process_rss_mb"] = mem_final["rss_mb"]
        result.stats["memory"]["rss_delta_mb"] = round(
            mem_final["rss_mb"] - mem_before["rss_mb"], 1
        )
        result.stats["memory"]["rss_after_setup_mb"] = mem_after_setup["rss_mb"]

        server.teardown()
        del model_copy
        return result

    def print_report(self, results: dict[str, BenchmarkResult]):
        """Print a formatted comparison report."""
        print("\n" + "=" * 80)
        print("  NVE BENCHMARK REPORT")
        print("=" * 80)

        baseline = results.get("baseline")

        # ── Latency comparison ──
        print("\n  LATENCY")
        print("  " + "-" * 76)
        print(f"  {'Configuration':<35} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10}")
        print("  " + "-" * 76)
        for key, res in results.items():
            lat = res.stats.get("latency", {})
            print(f"  {res.name:<35} {lat.get('mean_forward_ms', 0):>10.1f} "
                  f"{lat.get('p50_forward_ms', 0):>10.1f} "
                  f"{lat.get('p95_forward_ms', 0):>10.1f}")

        # ── Throughput ──
        print("\n  THROUGHPUT")
        print("  " + "-" * 76)
        print(f"  {'Configuration':<35} {'Tokens':>8} {'Time (s)':>10} {'Tok/sec':>10}")
        print("  " + "-" * 76)
        for key, res in results.items():
            tp = res.stats.get("throughput", {})
            print(f"  {res.name:<35} {tp.get('tokens_generated', 0):>8} "
                  f"{tp.get('total_time_s', 0):>10.2f} "
                  f"{tp.get('tokens_per_sec', 0):>10.2f}")

        # ── Paging ──
        print("\n  PAGING")
        print("  " + "-" * 76)
        print(f"  {'Configuration':<35} {'GPU hits':>9} {'RAM PIs':>9} {'SSD PIs':>9} "
              f"{'Fault%':>8} {'Prefetch':>9}")
        print("  " + "-" * 76)
        for key, res in results.items():
            pg = res.stats.get("paging", {})
            pf_hits = pg.get("prefetch_hits", 0)
            print(f"  {res.name:<35} {pg.get('gpu_hits', 0):>9} "
                  f"{pg.get('ram_page_ins', 0):>9} "
                  f"{pg.get('ssd_page_ins', 0):>9} "
                  f"{pg.get('page_fault_rate', 0) * 100:>7.1f}% "
                  f"{pf_hits:>9}")

        # ── Memory ──
        print("\n  MEMORY")
        print("  " + "-" * 76)
        print(f"  {'Configuration':<35} {'GPU (MB)':>10} {'RAM (MB)':>10} "
              f"{'SSD rd (MB)':>12} {'RSS (MB)':>10}")
        print("  " + "-" * 76)
        for key, res in results.items():
            mem = res.stats.get("memory", {})
            print(f"  {res.name:<35} {mem.get('peak_gpu_mb', 0):>10.1f} "
                  f"{mem.get('peak_ram_mb', 0):>10.1f} "
                  f"{mem.get('ssd_reads_mb', 0):>12.1f} "
                  f"{mem.get('process_rss_mb', 0):>10.1f}")

        # ── Output quality ──
        if baseline and baseline.logit_samples:
            print("\n  OUTPUT QUALITY (vs baseline)")
            print("  " + "-" * 76)
            print(f"  {'Configuration':<35} {'KL div':>10} {'Cosine':>10} {'Top-1 agree':>12}")
            print("  " + "-" * 76)

            for key, res in results.items():
                if key == "baseline":
                    print(f"  {res.name:<35} {'(ref)':>10} {'(ref)':>10} {'(ref)':>12}")
                    continue
                if res.logit_samples:
                    drift = compute_logit_drift(baseline.logit_samples, res.logit_samples)
                    print(f"  {res.name:<35} "
                          f"{drift.get('kl_divergence_mean', 'N/A'):>10} "
                          f"{drift.get('cosine_similarity_mean', 'N/A'):>10} "
                          f"{drift.get('top1_agreement', 'N/A'):>12}")

        # ── Generated text comparison ──
        print("\n  GENERATION SAMPLES")
        print("  " + "-" * 76)
        for i, prompt in enumerate(self.config.prompts or ["(default prompts)"]):
            short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"\n  Prompt: \"{short_prompt}\"")
            for key, res in results.items():
                if i < len(res.generation_outputs):
                    out = res.generation_outputs[i]
                    gen_text = out["text"][len(prompt):].strip()[:80]
                    print(f"    {res.name:<30}: \"{gen_text}...\"")

        print("\n" + "=" * 80)

    def save_results(self, results: dict[str, BenchmarkResult], path: Optional[str] = None):
        """Save benchmark results to JSON."""
        out_dir = Path(path or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for key, res in results.items():
            serializable[key] = {
                "name": res.name,
                "stats": res.stats,
                "generations": res.generation_outputs,
            }

        with open(out_dir / "benchmark_results.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)
