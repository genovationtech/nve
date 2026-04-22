"""
Basic NVE profiling example.

Demonstrates the full lifecycle:
1. Create engine
2. Register synthetic weight blocks
3. Profile with simulated activations
4. Build clusters and tier placement
5. Query statistics
"""

import sys
import os
import random

# Add the python package to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from nve import NVEEngine, EngineConfig, TierConfig
from nve.profiler import ActivationSample, MCAPProfiler, PromptDistribution, PromptDomain


def main():
    print("=== Neural Virtualization Engine — Basic Profiling Example ===\n")

    # ── 1. Configure engine ──
    # Simulate a 84 MB model to demonstrate budget-driven tier placement.
    simulated_model_bytes = 84 * 1024 * 1024

    # Option A: Auto mode — detect hardware and use 80% of available resources.
    # tier = TierConfig.auto(model_bytes=simulated_model_bytes)

    # Option B: Manual budget — specify exactly how much GPU/RAM to use.
    # tier = TierConfig.from_budget(
    #     model_bytes=simulated_model_bytes,
    #     gpu_budget_bytes=4 * 1024**2,   # 4 MB GPU budget
    #     ram_budget_bytes=16 * 1024**2,  # 16 MB RAM budget
    # )

    # Option C: Direct fractions (legacy).
    tier = TierConfig.from_budget(
        model_bytes=simulated_model_bytes,
        gpu_budget_bytes=int(simulated_model_bytes * 0.2),   # 20% on GPU
        ram_budget_bytes=int(simulated_model_bytes * 0.3),   # 30% in RAM
    )

    config = EngineConfig(
        tier=tier,
        samples_per_round=50,
        min_samples_for_stability=10,
    )

    engine = NVEEngine(config=config)
    print(f"Engine created (Rust backend: {engine._rust_handle is not None})")
    print(f"Budget: GPU={tier.gpu_bytes/1024**2:.1f}MB ({tier.gpu_fraction:.0%}), "
          f"RAM={tier.ram_bytes/1024**2:.1f}MB ({tier.ram_fraction:.0%}), "
          f"SSD={1-tier.gpu_fraction-tier.ram_fraction:.0%}")

    # ── 2. Simulate model registration ──
    # In real usage: engine.register_model(your_pytorch_model)
    # Here we simulate with direct profiler interaction.
    num_weights = 100
    print(f"Simulating {num_weights} weight blocks\n")

    # ── 3. Define prompt distribution ──
    dist = PromptDistribution([
        PromptDomain(
            name="math",
            weight=0.3,
            prompts=[
                "Solve x^2 + 3x - 4 = 0",
                "What is the derivative of ln(x)?",
                "Prove that sqrt(2) is irrational",
            ],
        ),
        PromptDomain(
            name="code",
            weight=0.3,
            prompts=[
                "Write a quicksort in Python",
                "Implement a binary tree",
                "Debug this segfault",
            ],
        ),
        PromptDomain(
            name="chat",
            weight=0.2,
            prompts=[
                "Tell me a joke",
                "What's the weather like?",
                "How are you today?",
            ],
        ),
        PromptDomain(
            name="reasoning",
            weight=0.2,
            prompts=[
                "If all A are B, and all B are C, are all A C?",
                "What came first, the chicken or the egg?",
                "Explain the trolley problem",
            ],
        ),
    ])

    # ── 4. Run profiling ──
    profiler = engine.profiler
    profiler.start()

    print("Profiling with Monte Carlo sampling...")
    sampled_prompts = dist.sample(200)

    for domain, prompt in sampled_prompts:
        # Simulate activation patterns per domain.
        samples = []
        for wid in range(num_weights):
            # Math weights: 0-24 activate strongly for math.
            # Code weights: 25-49 activate strongly for code.
            # Chat weights: 50-74 activate strongly for chat.
            # Shared weights: 75-99 activate moderately for everything.
            if domain == "math" and wid < 25:
                mag = random.gauss(0.85, 0.1)
            elif domain == "code" and 25 <= wid < 50:
                mag = random.gauss(0.80, 0.1)
            elif domain == "chat" and 50 <= wid < 75:
                mag = random.gauss(0.75, 0.1)
            elif wid >= 75:
                mag = random.gauss(0.40, 0.15)
            else:
                mag = random.gauss(0.05, 0.02)

            mag = max(0.0, min(1.0, mag))
            samples.append(ActivationSample(weight_id=wid, magnitude=mag, domain=domain))

        profiler.record_batch(samples)
        profiler.finish_round()

    print(f"  Rounds completed: {profiler.total_rounds()}")
    print(f"  Weights profiled: {profiler.weight_count()}")
    print(f"  Estimates stable: {profiler.is_stable()}\n")

    # ── 5. View importance rankings ──
    print("Top 10 weights by importance:")
    ranking = profiler.importance_ranking()
    for wid, imp in ranking[:10]:
        print(f"  W{wid:3d}: importance = {imp:.4f}")

    print("\nTop 5 weights for 'math' domain:")
    math_ranking = profiler.domain_ranking("math")
    for wid, imp in math_ranking[:5]:
        print(f"  W{wid:3d}: math importance = {imp:.4f}")

    print("\nTop 5 weights for 'code' domain:")
    code_ranking = profiler.domain_ranking("code")
    for wid, imp in code_ranking[:5]:
        print(f"  W{wid:3d}: code importance = {imp:.4f}")

    # ── 6. Build tier placement ──
    print("\nBuilding tier placement...")
    partition = profiler.partition(hot_fraction=0.2, warm_fraction=0.3)
    engine.pager.initialize(partition)
    engine._is_built = True

    stats = engine.pager.stats()
    print(f"  Hot (GPU):  {len(partition['hot']):3d} weights")
    print(f"  Warm (RAM): {len(partition['warm']):3d} weights")
    print(f"  Cold (SSD): {len(partition['cold']):3d} weights")
    print(f"  GPU utilization: {stats.gpu_utilization:.1%}")

    # ── 7. Simulate runtime access ──
    print("\nSimulating runtime access patterns...")
    for _ in range(1000):
        wid = random.choice(range(num_weights))
        engine.pager.access(wid)

    final_stats = engine.pager.stats()
    print(f"  Page hits:   {final_stats.page_hits}")
    print(f"  Page faults: {final_stats.page_faults}")
    print(f"  Fault rate:  {final_stats.fault_rate:.2%}")

    print("\n=== Profiling Complete ===")
    print("\nThe engine has learned which weights matter for which domains.")
    print("In production, weights would be dynamically paged across GPU/RAM/SSD")
    print("based on the inferred prompt type — no explicit labels needed.")


if __name__ == "__main__":
    main()
