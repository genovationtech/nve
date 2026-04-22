"""
NVE real model test — downloads a small model from HuggingFace and runs
the full Monte Carlo Activation Profiling pipeline.

Uses GPT-2 (small, ~124M params) as it's lightweight enough for CPU inference.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from nve import NVEEngine, EngineConfig, TierConfig
from nve.profiler import ActivationSample


def main():
    hf_token = os.environ.get("HF_TOKEN", os.environ["HF_TOKEN"])
    model_name = "gpt2"  # 124M params, ~500MB

    print("=" * 70)
    print("  Neural Virtualization Engine — Real Model Test")
    print("=" * 70)

    # ── 1. Load model ──
    print(f"\n[1/6] Loading {model_name} from HuggingFace...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Count parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_bytes / 1024**2:.1f} MB)")
    print(f"  Layers: {model.config.n_layer}")

    # ── 2. Configure NVE ──
    print("\n[2/6] Configuring NVE engine...")

    # Budget-driven config: specify how much GPU/RAM to allocate,
    # NVE computes the tier fractions from model size.
    # Use TierConfig.auto(total_bytes) to auto-detect from hardware instead.
    tier = TierConfig.from_budget(
        model_bytes=total_bytes,
        gpu_budget_bytes=int(total_bytes * 0.20),  # 20% GPU budget
        ram_budget_bytes=int(total_bytes * 0.30),   # 30% RAM budget
    )

    config = EngineConfig(
        tier=tier,
        samples_per_round=10,
        min_samples_for_stability=3,
    )
    engine = NVEEngine(config=config)
    print(f"  Model size: {total_bytes/1024**2:.0f} MB")
    print(f"  GPU budget: {tier.gpu_bytes/1024**2:.0f} MB ({tier.gpu_fraction:.0%} of weights)")
    print(f"  RAM budget: {tier.ram_bytes/1024**2:.0f} MB ({tier.ram_fraction:.0%} of weights)")
    print(f"  SSD overflow: {max(0, 1-tier.gpu_fraction-tier.ram_fraction):.0%} of weights")

    # ── 3. Register model ──
    print("\n[3/6] Registering model parameters as weight blocks...")
    engine.register_model(model)
    print(f"  Registered {len(engine._weight_blocks)} weight blocks")

    # Show parameter names.
    for bid, info in list(engine._weight_blocks.items())[:5]:
        print(f"    Block {bid}: {info['name']} ({info['size_bytes']/1024:.1f} KB)")
    if len(engine._weight_blocks) > 5:
        print(f"    ... and {len(engine._weight_blocks) - 5} more")

    # ── 4. Profile with diverse prompts ──
    print("\n[4/6] Running Monte Carlo Activation Profiling...")

    domain_prompts = {
        "math": [
            "The solution to the quadratic equation x^2 + 5x + 6 = 0 is",
            "Calculate the integral of x^3 from 0 to 1:",
            "If f(x) = sin(x), then f'(x) =",
            "The determinant of a 2x2 matrix [[a,b],[c,d]] is",
            "In probability theory, Bayes theorem states that",
        ],
        "code": [
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
            "import torch\nmodel = torch.nn.Linear(",
            "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n",
            "SELECT users.name, COUNT(orders.id) FROM users JOIN",
            "async function fetchData(url) {\n    const response = await fetch(",
        ],
        "reasoning": [
            "If all mammals are warm-blooded, and all dogs are mammals, then",
            "The trolley problem presents a moral dilemma where",
            "Consider the following logical argument: Premise 1:",
            "The difference between correlation and causation is that",
            "Occam's razor suggests that the simplest explanation",
        ],
        "creative": [
            "Once upon a time in a land far away, there lived a",
            "The sunset painted the sky in shades of",
            "She opened the old book and found a letter that read:",
            "In the year 2150, humanity had finally",
            "The detective examined the crime scene and noticed",
        ],
    }

    # Activation capture via hooks.
    captured = {}

    def make_hook(block_id):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if hasattr(out, 'abs'):
                captured[block_id] = float(out.abs().mean().item())
        return hook_fn

    # Install hooks.
    hooks = []
    module_dict = dict(model.named_modules())
    for block_id, info in engine._weight_blocks.items():
        parts = info['name'].rsplit('.', 1)
        if len(parts) == 2:
            mod_name = parts[0]
            if mod_name in module_dict:
                h = module_dict[mod_name].register_forward_hook(make_hook(block_id))
                hooks.append(h)

    profiler = engine.profiler
    profiler.start()
    total_prompts = 0

    t0 = time.time()
    for domain, prompts in domain_prompts.items():
        for prompt in prompts:
            captured.clear()

            # Tokenize and run forward pass.
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                model(**inputs)

            # Record activations.
            samples = []
            for block_id, magnitude in captured.items():
                samples.append(ActivationSample(
                    weight_id=block_id,
                    magnitude=magnitude,
                    domain=domain,
                ))

            profiler.record_batch(samples)
            profiler.finish_round()
            total_prompts += 1

    profile_time = time.time() - t0

    # Remove hooks.
    for h in hooks:
        h.remove()

    print(f"  Profiled {total_prompts} prompts across {len(domain_prompts)} domains in {profile_time:.1f}s")
    print(f"  Weights profiled: {profiler.weight_count()}")
    print(f"  Stable estimates: {profiler.is_stable()}")

    # ── 5. Analyze results ──
    print("\n[5/6] Analyzing activation patterns...")

    # Global importance ranking.
    ranking = profiler.importance_ranking()
    print("\n  Top 15 most important weights (globally):")
    for wid, imp in ranking[:15]:
        name = engine._weight_blocks[wid]['name']
        size_kb = engine._weight_blocks[wid]['size_bytes'] / 1024
        print(f"    {name:45s}  importance={imp:.4f}  ({size_kb:.0f} KB)")

    print("\n  Bottom 10 least important weights:")
    for wid, imp in ranking[-10:]:
        name = engine._weight_blocks[wid]['name']
        print(f"    {name:45s}  importance={imp:.6f}")

    # Domain-specific analysis.
    for domain in domain_prompts:
        dom_ranking = profiler.domain_ranking(domain)
        top3 = dom_ranking[:3]
        print(f"\n  Top 3 weights for '{domain}':")
        for wid, imp in top3:
            name = engine._weight_blocks[wid]['name']
            print(f"    {name:45s}  {domain}_importance={imp:.4f}")

    # ── 6. Build tier placement ──
    print("\n[6/6] Building tier placement...")
    partition = profiler.partition(hot_fraction=0.2, warm_fraction=0.3)

    # Calculate bytes per tier.
    hot_bytes = sum(engine._weight_blocks[wid]['size_bytes'] for wid in partition['hot'])
    warm_bytes = sum(engine._weight_blocks[wid]['size_bytes'] for wid in partition['warm'])
    cold_bytes = sum(engine._weight_blocks[wid]['size_bytes'] for wid in partition['cold'])

    print(f"\n  Tier Placement Summary:")
    print(f"  ┌──────────┬──────────┬─────────────┬─────────────────────────────────┐")
    print(f"  │ Tier     │ Weights  │ Size        │ Description                     │")
    print(f"  ├──────────┼──────────┼─────────────┼─────────────────────────────────┤")
    print(f"  │ GPU (hot)│ {len(partition['hot']):>6}   │ {hot_bytes/1024**2:>7.1f} MB  │ Always resident, fastest access  │")
    print(f"  │ RAM (wrm)│ {len(partition['warm']):>6}   │ {warm_bytes/1024**2:>7.1f} MB  │ Loaded on demand                │")
    print(f"  │ SSD (cld)│ {len(partition['cold']):>6}   │ {cold_bytes/1024**2:>7.1f} MB  │ Paged in when needed            │")
    print(f"  └──────────┴──────────┴─────────────┴─────────────────────────────────┘")

    # Show which layers dominate each tier.
    print("\n  Layer distribution per tier:")
    for tier_name, wids in [("GPU", partition['hot']), ("RAM", partition['warm']), ("SSD", partition['cold'])]:
        layers = {}
        for wid in wids:
            name = engine._weight_blocks[wid]['name']
            # Extract layer identifier.
            parts = name.split('.')
            layer_key = '.'.join(parts[:3]) if len(parts) >= 3 else parts[0]
            layers[layer_key] = layers.get(layer_key, 0) + 1
        top_layers = sorted(layers.items(), key=lambda x: x[1], reverse=True)[:5]
        layer_str = ", ".join(f"{k}({v})" for k, v in top_layers)
        print(f"    {tier_name}: {layer_str}")

    # Initialize pager and simulate access.
    engine.pager.initialize(partition, sizes={
        wid: info['size_bytes'] for wid, info in engine._weight_blocks.items()
    })
    engine._is_built = True

    # Simulate runtime: access weights based on a "math" prompt pattern.
    print("\n  Simulating runtime access (math-heavy workload)...")
    math_ranking = profiler.domain_ranking("math")
    math_hot = [wid for wid, _ in math_ranking[:30]]

    for _ in range(500):
        for wid in math_hot:
            engine.pager.access(wid)

    stats = engine.pager.stats()
    print(f"    Page hits:   {stats.page_hits:,}")
    print(f"    Page faults: {stats.page_faults:,}")
    print(f"    Fault rate:  {stats.fault_rate:.2%}")
    print(f"    Migrations:  {stats.migrations}")

    # Cross-domain analysis.
    print("\n  Cross-domain weight overlap analysis:")
    domain_tops = {}
    for domain in domain_prompts:
        dom_rank = profiler.domain_ranking(domain)
        domain_tops[domain] = set(wid for wid, _ in dom_rank[:20])

    domains = list(domain_prompts.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            overlap = domain_tops[d1] & domain_tops[d2]
            print(f"    {d1:10s} ∩ {d2:10s} = {len(overlap):2d}/20 shared top weights")

    print("\n" + "=" * 70)
    print("  Test Complete — NVE successfully profiled a real model")
    print("=" * 70)


if __name__ == "__main__":
    main()
