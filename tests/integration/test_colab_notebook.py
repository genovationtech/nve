#!/usr/bin/env python3
"""
Test the Google Colab notebook (nve_colab.ipynb) end-to-end using GPT-2.

Simulates the notebook cell execution locally, skipping Colab-specific
installation and environment detection cells, but testing all the
NVE pipeline cells: streaming profiler, manifest, tier placement,
serving, benchmark, and interactive generation.
"""

import sys
import os
import time
import copy
import json
import traceback

# Point to the NVE Python SDK
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache", "gpt2")

passed = 0
failed = 0
errors = []


def cell(name):
    """Decorator to run a notebook cell as a test."""
    def decorator(fn):
        global passed, failed
        print(f"\n{'=' * 60}")
        print(f"CELL: {name}")
        print(f"{'=' * 60}")
        try:
            fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))
        return fn
    return decorator


# ── Simulated Colab environment ──
# These would be set by cell [2] (config) and cell [4] (runtime detection)
MODEL_ID = "gpt2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PROMPT = "The meaning of life is"
MAX_TOKENS = 20
TEMPERATURE = 0.7
BUDGET_MODE = "auto"
AUTO_GPU_FRACTION = 0.8
AUTO_RAM_FRACTION = 0.8
HAS_GPU = False
DEVICE = None
NVE_MODE = "quick"
NVE_DIR = None


@cell("Cell [4]: Runtime Detection")
def test_runtime():
    global HAS_GPU, DEVICE, GPU_NAME, GPU_MEM_MB
    import psutil
    import torch

    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if HAS_GPU else "cpu")
    GPU_NAME = "N/A"
    GPU_MEM_MB = 0

    ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    avail_mb = psutil.virtual_memory().available // (1024 * 1024)
    print(f"  GPU: {'YES' if HAS_GPU else 'No (CPU-only)'}")
    print(f"  RAM: {ram_mb} MB (available: {avail_mb} MB)")
    print(f"  Device: {DEVICE}")


@cell("Cell [6]: NVE Import (simulated quick mode)")
def test_nve_import():
    import nve
    from nve import NVEEngine, EngineConfig, TierConfig
    print(f"  NVE {nve.__version__} loaded")


@cell("Cell [8]: Load Tokenizer (model stays on disk)")
def test_load_tokenizer():
    global tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"  Tokenizer loaded from {MODEL_DIR}")
    print(f"  Vocab size: {tokenizer.vocab_size}")


@cell("Cell [12]: StreamingProfiler — Weight Inventory")
def test_streaming_profiler_init():
    global profiler
    from nve import StreamingProfiler

    profiler = StreamingProfiler(MODEL_DIR)
    inventory = profiler.weight_inventory()
    total_bytes = sum(info["size_bytes"] for info in inventory.values())
    print(f"  Weight inventory: {len(inventory)} tensors, {total_bytes / 1024**3:.2f} GB")
    print(f"  Architecture: {profiler.architecture}, {profiler.num_layers} layers")
    assert len(inventory) > 0
    assert profiler.architecture == "gpt2"


@cell("Cell [13]: StreamingProfiler — Profile with Diverse Prompts")
def test_streaming_profile():
    global manifest

    profiling_prompts = [
        "The theory of general relativity states that",
        "The integral of e^x from 0 to infinity",
        "Explain quantum entanglement to a child:",
        "def fibonacci(n):\n    ",
        "import torch\nclass Model(nn.Module):",
        "In a surprising turn of events, the",
    ]
    profiling_domains = [
        "science", "science", "science",
        "code", "code",
        "creative",
    ]

    manifest = profiler.profile(profiling_prompts, domains=profiling_domains, max_seq_len=32)

    MANIFEST_PATH = f"/tmp/{MODEL_ID.replace('/', '--')}.nve"
    manifest.save(MANIFEST_PATH)
    print(f"  Manifest saved to {MANIFEST_PATH}")
    print(f"  Manifest size: {os.path.getsize(MANIFEST_PATH) / 1024:.1f} KB")
    print(f"  Weights: {len(manifest.weights)}")
    assert len(manifest.weights) > 0


@cell("Cell [15]: Weight Importance Analysis")
def test_importance_analysis():
    import numpy as np

    weights = manifest.weights
    importances = [w["importance"] for w in weights]

    print(f"  Top 5 most important weights:")
    for w in weights[:5]:
        print(f"    {w['name']:55s} imp={w['importance']:.6f}")

    print(f"\n  Importance distribution:")
    print(f"    Mean: {np.mean(importances):.6f}  Std: {np.std(importances):.6f}")

    # Domain analysis
    domains_found = set()
    for w in weights:
        domains_found.update(w.get("domain_importances", {}).keys())

    if domains_found:
        print(f"\n  Domains found: {sorted(domains_found)}")
        for domain in sorted(domains_found):
            domain_ranked = sorted(
                weights,
                key=lambda w: w.get("domain_importances", {}).get(domain, 0.0),
                reverse=True,
            )
            top = domain_ranked[0]
            d_imp = top.get("domain_importances", {}).get(domain, 0.0)
            print(f"    {domain}: top={top['name']} (domain_imp={d_imp:.4f})")


@cell("Cell [17]: Budget-Driven Tier Placement")
def test_tier_placement():
    global tier_config, placement_global
    from nve import TierConfig

    tier_config = TierConfig.auto(
        model_bytes=manifest.total_bytes,
        gpu_reserve_frac=AUTO_GPU_FRACTION,
        ram_reserve_frac=AUTO_RAM_FRACTION,
    )

    print(f"  Model size:     {manifest.total_bytes / 1024**3:.2f} GB")
    print(f"  GPU budget:     {tier_config.gpu_bytes / 1024**2:.0f} MB → {tier_config.gpu_fraction:.1%}")
    print(f"  RAM budget:     {tier_config.ram_bytes / 1024**2:.0f} MB → {tier_config.ram_fraction:.1%}")

    placement_global = manifest.tier_placement(
        gpu_budget_bytes=tier_config.gpu_bytes,
        ram_budget_bytes=tier_config.ram_bytes,
    )

    gpu_bytes = sum(w["size_bytes"] for w in placement_global["gpu"])
    ram_bytes = sum(w["size_bytes"] for w in placement_global["ram"])
    ssd_bytes = sum(w["size_bytes"] for w in placement_global["ssd"])
    print(f"\n  Tier placement:")
    print(f"    GPU (hot):  {len(placement_global['gpu']):>4} weights, {gpu_bytes/1024**2:>8.1f} MB")
    print(f"    RAM (warm): {len(placement_global['ram']):>4} weights, {ram_bytes/1024**2:>8.1f} MB")
    print(f"    SSD (cold): {len(placement_global['ssd']):>4} weights, {ssd_bytes/1024**2:>8.1f} MB")

    total = len(placement_global["gpu"]) + len(placement_global["ram"]) + len(placement_global["ssd"])
    assert total == len(manifest.weights), f"Placement missing weights: {total} vs {len(manifest.weights)}"


@cell("Cell [19]: Domain-Specific Tier Placement")
def test_domain_tier_placement():
    domains_found = set()
    for w in manifest.weights:
        domains_found.update(w.get("domain_importances", {}).keys())

    if not domains_found:
        print("  No domain data — skipping")
        return

    domain_placements = {}
    for domain in sorted(domains_found):
        ranked = sorted(
            manifest.weights,
            key=lambda w: w.get("domain_importances", {}).get(domain, 0.0),
            reverse=True,
        )
        gpu_used, ram_used = 0, 0
        placement = {"gpu": [], "ram": [], "ssd": []}
        for w in ranked:
            size = w["size_bytes"]
            if gpu_used + size <= tier_config.gpu_bytes:
                placement["gpu"].append(w)
                gpu_used += size
            elif ram_used + size <= tier_config.ram_bytes:
                placement["ram"].append(w)
                ram_used += size
            else:
                placement["ssd"].append(w)
        domain_placements[domain] = placement

    for domain in sorted(domain_placements):
        p = domain_placements[domain]
        gm = sum(w["size_bytes"] for w in p["gpu"]) / 1024**2
        print(f"  {domain}: GPU={len(p['gpu'])} ({gm:.1f} MB), "
              f"RAM={len(p['ram'])}, SSD={len(p['ssd'])}")


@cell("Cell [21]: Build Manifest & Load Model")
def test_build_manifest():
    global manifest, model, total_params, model_mb
    import torch
    from nve.manifest import TierManifest, PageEntry
    from nve.serving import TieredModelServer, BaselineServer
    from transformers import AutoModelForCausalLM

    # Load the full model for serving
    print("  Loading model for serving...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Model loaded: {total_params:,} params ({model_mb:.0f} MB)")

    # Build TierManifest from NVEManifest's tier placement
    tier_manifest = TierManifest()
    placement = manifest.tier_placement(
        gpu_budget_bytes=tier_config.gpu_bytes,
        ram_budget_bytes=tier_config.ram_bytes,
    )

    for tier_name, page_list in [("gpu", tier_manifest.gpu_pages),
                                  ("ram", tier_manifest.ram_pages),
                                  ("ssd", tier_manifest.ssd_pages)]:
        for wid, w in enumerate(placement[tier_name]):
            page_list.append(PageEntry(
                param_name=w["name"],
                weight_id=wid,
                size_bytes=w["size_bytes"],
                importance=w["importance"],
                domain_importances=w.get("domain_importances", {}),
                layer_index=w.get("layer_index", 0),
            ))

    manifest = tier_manifest

    print(f"\n  Tier Manifest:")
    print(f"    GPU pages: {len(manifest.gpu_pages):>4} ({manifest.gpu_bytes / 1024**2:.1f} MB)")
    print(f"    RAM pages: {len(manifest.ram_pages):>4} ({manifest.ram_bytes / 1024**2:.1f} MB)")
    print(f"    SSD pages: {len(manifest.ssd_pages):>4} ({manifest.ssd_bytes / 1024**2:.1f} MB)")

    assert len(manifest.gpu_pages) + len(manifest.ram_pages) + len(manifest.ssd_pages) > 0


@cell("Cell [23]: Baseline Generation")
def test_baseline_generation():
    import torch
    from nve.serving import BaselineServer

    baseline_model = copy.deepcopy(model).to(DEVICE)
    baseline = BaselineServer(baseline_model, tokenizer, device=DEVICE)
    baseline.setup()

    result_baseline = baseline.generate(PROMPT, max_new_tokens=MAX_TOKENS)
    print(f"  {result_baseline['generated_tokens']} tokens, {result_baseline['tokens_per_sec']:.1f} tok/s")
    print(f"  Output: \"{result_baseline['text'][:100]}...\"")

    assert result_baseline['generated_tokens'] > 0
    assert result_baseline['tokens_per_sec'] > 0

    baseline.teardown()
    del baseline_model


@cell("Cell [25]: NVE Tiered Generation")
def test_tiered_generation():
    import torch
    from nve.serving import TieredModelServer

    tiered_model = copy.deepcopy(model)
    tiered_model.eval()

    server = TieredModelServer(
        tiered_model, tokenizer, manifest,
        device=DEVICE, enable_prefetch=True, prefetch_depth=2,
    )
    server.setup()

    print(f"  GPU pages (hot):  {len(manifest.gpu_pages)} ({manifest.gpu_bytes / 1024**2:.0f} MB)")
    print(f"  RAM pages (warm): {len(manifest.ram_pages)} ({manifest.ram_bytes / 1024**2:.0f} MB)")
    print(f"  SSD pages (cold): {len(manifest.ssd_pages)} ({manifest.ssd_bytes / 1024**2:.0f} MB)")

    result_nve = server.generate(PROMPT, max_new_tokens=MAX_TOKENS)
    print(f"\n  {result_nve['generated_tokens']} tokens, {result_nve['tokens_per_sec']:.1f} tok/s")
    print(f"  Output: \"{result_nve['text'][:100]}...\"")

    stats = server.stats.to_dict()
    print(f"\n  Paging Stats:")
    print(f"    GPU hits:     {stats['paging']['gpu_hits']}")
    print(f"    RAM page-ins: {stats['paging']['ram_page_ins']}")
    print(f"    SSD page-ins: {stats['paging']['ssd_page_ins']}")
    print(f"    Fault rate:   {stats['paging']['page_fault_rate']:.1%}")

    assert result_nve['generated_tokens'] > 0

    server.teardown()
    del tiered_model


@cell("Cell [29]: Python SDK Benchmark")
def test_benchmark():
    from nve.benchmark import Benchmark, BenchmarkConfig

    bench_config = BenchmarkConfig(
        prompts=[
            "The theory of general relativity states that",
            "def fibonacci(n):\n    ",
        ],
        max_new_tokens=min(MAX_TOKENS, 10),
        warmup_prompts=1,
    )

    bench = Benchmark(model, tokenizer, manifest, bench_config)
    print("  Running benchmark (3 configurations)...")
    results = bench.run()
    bench.print_report(results)

    assert "baseline" in results
    assert "nve_static" in results
    assert "nve_prefetch" in results
    assert results["baseline"].stats["throughput"]["tokens_generated"] > 0


@cell("Cell [31]: Interactive Generation")
def test_interactive():
    from nve.serving import TieredModelServer

    gen_model = copy.deepcopy(model)
    gen_model.eval()

    srv = TieredModelServer(
        gen_model, tokenizer, manifest,
        device=DEVICE, enable_prefetch=True,
    )
    srv.setup()
    out = srv.generate("Explain gravity in one sentence:", max_new_tokens=30)
    print(f"  [NVE Tiered] {out['generated_tokens']} tokens, {out['tokens_per_sec']:.1f} tok/s")
    print(f"  Output: \"{out['text'][:120]}...\"")
    srv.teardown()

    assert out['generated_tokens'] > 0
    del gen_model


@cell("Cell [33]: Session Summary")
def test_summary():
    import nve
    import psutil
    mem = psutil.virtual_memory()

    print(f"  Mode:        {NVE_MODE.upper()}")
    print(f"  Model:       {MODEL_ID}")
    print(f"  Device:      {DEVICE}")
    print(f"  Params:      {total_params:,}")
    print(f"  Model size:  {model_mb:.0f} MB")
    print(f"  GPU budget:  {tier_config.gpu_bytes / 1024**2:.0f} MB ({tier_config.gpu_fraction:.0%})")
    print(f"  RAM budget:  {tier_config.ram_bytes / 1024**2:.0f} MB ({tier_config.ram_fraction:.0%})")
    print(f"  Manifest:    GPU={len(manifest.gpu_pages)}, RAM={len(manifest.ram_pages)}, SSD={len(manifest.ssd_pages)}")
    print(f"  RAM used:    {mem.used // (1024**2)} / {mem.total // (1024**2)} MB ({mem.percent}%)")
    print(f"  NVE version: {nve.__version__}")


# ── Summary ──

print(f"\n{'#' * 60}")
print(f"  NOTEBOOK TEST RESULTS: {passed} passed, {failed} failed")
print(f"{'#' * 60}")

if errors:
    print("\nFailed cells:")
    for name, err in errors:
        print(f"  ✗ {name}: {err}")
    sys.exit(1)
else:
    print("\n  All notebook cells passed!")
    sys.exit(0)
