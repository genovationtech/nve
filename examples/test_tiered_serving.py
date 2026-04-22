"""
NVE Tiered Serving Test — end-to-end benchmark.

Downloads GPT-2, profiles it, builds a tier manifest, then benchmarks:
  1. Full model baseline (all weights resident)
  2. NVE static tiering (learned placement, no prefetch)
  3. NVE static tiering + prefetch

Compares latency, throughput, paging stats, memory, and output quality.
"""

import sys
import os
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from nve.profiler import MCAPProfiler, ActivationSample
from nve.manifest import TierManifest, PageEntry, build_manifest
from nve.serving import TieredModelServer, BaselineServer
from nve.benchmark import Benchmark, BenchmarkConfig


def profile_model(model, tokenizer):
    """Run MCAP profiling and return profiler + weight block metadata."""
    print("\n  Phase 1: Monte Carlo Activation Profiling")
    print("  " + "-" * 60)

    # Register weight blocks.
    weight_blocks = {}
    block_id = 0
    for name, param in model.named_parameters():
        size_bytes = param.nelement() * param.element_size()
        parts = name.split(".")
        layer_idx = 0
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        weight_blocks[block_id] = {
            "name": name,
            "size_bytes": size_bytes,
            "layer_index": layer_idx,
        }
        block_id += 1

    # Install activation hooks.
    captured = {}
    hooks = []
    module_dict = dict(model.named_modules())

    for bid, info in weight_blocks.items():
        parts = info["name"].rsplit(".", 1)
        if len(parts) == 2:
            mod_name = parts[0]
            if mod_name in module_dict:
                def make_hook(b):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output
                        if hasattr(out, "abs"):
                            captured[b] = float(out.abs().mean().item())
                    return hook_fn
                h = module_dict[mod_name].register_forward_hook(make_hook(bid))
                hooks.append(h)

    # Profiling prompts — diverse domains.
    profile_prompts = {
        "math": [
            "The quadratic formula gives us x equals negative b plus or minus the square root of",
            "In calculus, the fundamental theorem states that differentiation and integration are",
            "The eigenvalues of a symmetric matrix are always",
            "Bayes theorem tells us that the posterior probability is proportional to",
            "The Taylor series expansion of e^x around x=0 is",
        ],
        "code": [
            "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n",
            "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def insert(self, val):\n",
            "import asyncio\n\nasync def fetch_all(urls):\n    tasks = [",
            "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON",
            "fn main() {\n    let mut vec = Vec::new();\n    vec.push(",
        ],
        "reasoning": [
            "If we assume that all ravens are black, then observing a non-black non-raven",
            "The ship of Theseus thought experiment asks whether an object that has had all its",
            "In game theory, the Nash equilibrium represents a state where no player can",
            "The difference between deductive and inductive reasoning is that deductive",
            "Consider the following paradox: this statement is false. If we assume it is true",
        ],
        "creative": [
            "The old lighthouse keeper had not seen another human being for three months when",
            "In the year 2247, the colony ship Artemis finally reached the Proxima system and",
            "She opened the envelope with trembling hands. Inside was a photograph of",
            "The dragon had been sleeping for a thousand years, and when it woke, the world had",
            "Rain hammered the windows of the jazz club as the saxophone player began a solo that",
        ],
    }

    profiler = MCAPProfiler(samples_per_round=20, min_samples_for_stability=3)
    profiler.start()

    total = 0
    t0 = time.time()

    for domain, prompts in profile_prompts.items():
        for prompt in prompts:
            captured.clear()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                model(**inputs)

            samples = []
            for bid, magnitude in captured.items():
                samples.append(ActivationSample(weight_id=bid, magnitude=magnitude, domain=domain))
            profiler.record_batch(samples)
            profiler.finish_round()
            total += 1

    for h in hooks:
        h.remove()

    print(f"  Profiled {total} prompts across {len(profile_prompts)} domains in {time.time()-t0:.1f}s")
    print(f"  Weights tracked: {profiler.weight_count()}")

    return profiler, weight_blocks, list(profile_prompts.keys())


def build_tier_manifest(profiler, weight_blocks, domains):
    """Build and display the tier manifest."""
    print("\n  Phase 2: Building Tier Manifest")
    print("  " + "-" * 60)

    # Budget-driven: compute fractions from model size.
    # In production, use TierConfig.auto() or TierConfig.from_budget() to set these.
    total_bytes = sum(info["size_bytes"] for info in weight_blocks.values())
    gpu_budget = int(total_bytes * 0.20)  # 20% of model on GPU
    ram_budget = int(total_bytes * 0.30)  # 30% of model in RAM
    gpu_frac = gpu_budget / total_bytes if total_bytes > 0 else 0.2
    ram_frac = ram_budget / total_bytes if total_bytes > 0 else 0.3

    manifest = build_manifest(
        profiler,
        weight_blocks,
        gpu_fraction=gpu_frac,
        ram_fraction=ram_frac,
        domains=domains,
        profile_name="gpt2-default",
    )

    gpu_bytes = manifest.gpu_bytes
    ram_bytes = manifest.ram_bytes
    ssd_bytes = manifest.ssd_bytes
    total = gpu_bytes + ram_bytes + ssd_bytes

    print(f"  GPU tier: {len(manifest.gpu_pages):3d} params, {gpu_bytes/1024**2:6.1f} MB ({gpu_bytes/total*100:.0f}%)")
    print(f"  RAM tier: {len(manifest.ram_pages):3d} params, {ram_bytes/1024**2:6.1f} MB ({ram_bytes/total*100:.0f}%)")
    print(f"  SSD tier: {len(manifest.ssd_pages):3d} params, {ssd_bytes/1024**2:6.1f} MB ({ssd_bytes/total*100:.0f}%)")

    # Save manifest.
    manifest_dir = "/tmp/nve_manifests/gpt2-default"
    manifest.save(manifest_dir)
    print(f"\n  Manifest saved to {manifest_dir}/")

    # Show top GPU-resident params.
    print("\n  Top GPU-resident parameters (by importance):")
    for entry in sorted(manifest.gpu_pages, key=lambda e: e.importance, reverse=True)[:8]:
        print(f"    {entry.param_name:45s}  imp={entry.importance:.4f}  {entry.size_bytes/1024:.0f}KB")

    # Show SSD-demoted params.
    print("\n  Parameters demoted to SSD (lowest importance):")
    for entry in sorted(manifest.ssd_pages, key=lambda e: e.importance)[:8]:
        print(f"    {entry.param_name:45s}  imp={entry.importance:.4f}  {entry.size_bytes/1024:.0f}KB")

    return manifest


def run_benchmark(model, tokenizer, manifest):
    """Run the full benchmark suite."""
    print("\n  Phase 3: Tiered Serving Benchmark")
    print("  " + "-" * 60)

    test_prompts = [
        "The theory of general relativity describes how massive objects curve spacetime, which means that",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n",
        "In a landmark discovery, archaeologists in Egypt uncovered a previously unknown chamber inside",
        "The relationship between entropy and information theory was first established by Claude Shannon who",
    ]

    config = BenchmarkConfig(
        prompts=test_prompts,
        max_new_tokens=30,
        warmup_prompts=1,
        ssd_dir="/tmp/nve_bench_ssd",
        output_dir="/tmp/nve_bench_results",
    )

    bench = Benchmark(model, tokenizer, manifest, config)
    results = bench.run()
    bench.print_report(results)
    bench.save_results(results)

    return results


def main():
    hf_token = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
    model_name = "gpt2"

    print("=" * 80)
    print("  NEURAL VIRTUALIZATION ENGINE — TIERED SERVING TEST")
    print("=" * 80)

    # ── Load model ──
    print(f"\n  Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Model: {model_name} | {total_params:,} params | {total_bytes/1024**2:.1f} MB | Loaded in {time.time()-t0:.1f}s")

    # ── Phase 1: Profile ──
    profiler, weight_blocks, domains = profile_model(model, tokenizer)

    # ── Phase 2: Build manifest ──
    manifest = build_tier_manifest(profiler, weight_blocks, domains)

    # ── Phase 3: Benchmark ──
    gc.collect()
    results = run_benchmark(model, tokenizer, manifest)

    # ── Summary ──
    print("\n  SUMMARY")
    print("  " + "=" * 60)

    baseline_stats = results["baseline"].stats
    nve_stats = results["nve_prefetch"].stats

    bl_tps = baseline_stats["throughput"]["tokens_per_sec"]
    nve_tps = nve_stats["throughput"]["tokens_per_sec"]

    bl_mem = baseline_stats["memory"]["peak_gpu_mb"]
    nve_mem = nve_stats["memory"]["peak_gpu_mb"]

    pf_rate = nve_stats["paging"]["page_fault_rate"]

    print(f"  Baseline tokens/sec:    {bl_tps:.2f}")
    print(f"  NVE+prefetch tok/sec:   {nve_tps:.2f}")
    if bl_tps > 0:
        print(f"  Throughput ratio:       {nve_tps/bl_tps:.2%}")

    print(f"\n  Baseline GPU memory:    {bl_mem:.1f} MB")
    print(f"  NVE GPU memory:         {nve_mem:.1f} MB")
    if bl_mem > 0:
        print(f"  Memory reduction:       {(1 - nve_mem/bl_mem)*100:.1f}%")

    print(f"\n  NVE page fault rate:    {pf_rate*100:.1f}%")
    print(f"  SSD reads:              {nve_stats['memory']['ssd_reads_mb']:.1f} MB")

    # Quality check.
    from nve.benchmark import compute_logit_drift
    if results["baseline"].logit_samples and results["nve_prefetch"].logit_samples:
        drift = compute_logit_drift(
            results["baseline"].logit_samples,
            results["nve_prefetch"].logit_samples,
        )
        print(f"\n  Output quality drift:")
        print(f"    KL divergence:        {drift['kl_divergence_mean']:.6f}")
        print(f"    Cosine similarity:    {drift['cosine_similarity_mean']:.6f}")
        print(f"    Top-1 agreement:      {drift['top1_agreement']:.2%}")

    print("\n  " + "=" * 60)
    print("  Benchmark complete. Results saved to /tmp/nve_bench_results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
