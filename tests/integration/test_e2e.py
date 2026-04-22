#!/usr/bin/env python3
"""
End-to-end test for NVE streaming profiler and manifest system.

Tests:
1. StreamingProfiler: parse safetensors, weight_inventory(), profile()
2. NVEManifest: save/load round-trip, verify contents
3. NVEManifest.tier_placement: compute placements with budgets
4. TierConfig.from_budget and TierConfig.auto
"""

import sys
import os
import tempfile
import traceback

# Add the NVE Python SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache", "gpt2")

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        global passed, failed
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
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


# ── Test 1: StreamingProfiler construction and weight inventory ──

@test("StreamingProfiler construction")
def test_profiler_init():
    from nve.streaming_profiler import StreamingProfiler
    profiler = StreamingProfiler(model_dir=MODEL_DIR)
    assert profiler.architecture == "gpt2", f"Expected gpt2, got {profiler.architecture}"
    assert profiler.num_layers > 0, f"num_layers should be > 0, got {profiler.num_layers}"
    assert profiler.hidden_size > 0, f"hidden_size should be > 0, got {profiler.hidden_size}"
    print(f"  architecture={profiler.architecture}, layers={profiler.num_layers}, hidden={profiler.hidden_size}")


@test("StreamingProfiler.weight_inventory()")
def test_weight_inventory():
    from nve.streaming_profiler import StreamingProfiler
    profiler = StreamingProfiler(model_dir=MODEL_DIR)
    inventory = profiler.weight_inventory()
    assert len(inventory) > 0, "Inventory should not be empty"
    # Check structure of an entry
    first_name = next(iter(inventory))
    entry = inventory[first_name]
    assert "dtype" in entry, "Entry missing 'dtype'"
    assert "shape" in entry, "Entry missing 'shape'"
    assert "size_bytes" in entry, "Entry missing 'size_bytes'"
    assert "layer_index" in entry, "Entry missing 'layer_index'"
    total_bytes = sum(e["size_bytes"] for e in inventory.values())
    print(f"  {len(inventory)} weights, total {total_bytes / 1024**2:.1f} MB")
    # GPT-2 small is ~500MB
    assert total_bytes > 100_000_000, f"Total bytes too small: {total_bytes}"
    # Check we see both layer and non-layer weights
    layer_indices = set(e["layer_index"] for e in inventory.values())
    assert -1 in layer_indices, "Should have non-layer weights (index -1)"
    assert 0 in layer_indices, "Should have layer 0 weights"
    print(f"  Layer indices: {sorted(layer_indices)[:10]}...")


@test("StreamingProfiler.profile() with prompts")
def test_profiler_profile():
    from nve.streaming_profiler import StreamingProfiler
    profiler = StreamingProfiler(model_dir=MODEL_DIR)
    prompts = [
        "The meaning of life is",
        "In mathematics, a prime number is",
        "def fibonacci(n):",
    ]
    manifest = profiler.profile(prompts=prompts, max_seq_len=32)
    # Verify manifest is an NVEManifest
    from nve.streaming_profiler import NVEManifest
    assert isinstance(manifest, NVEManifest), f"Expected NVEManifest, got {type(manifest)}"
    assert manifest.model_id is not None, "model_id should not be None"
    assert manifest.architecture == "gpt2"
    assert manifest.num_layers > 0
    assert manifest.total_params > 0
    assert manifest.total_bytes > 0
    assert len(manifest.weights) > 0, "Weights list should not be empty"
    # Check weight entries have importance scores
    has_nonzero = any(w["importance"] > 0 for w in manifest.weights)
    assert has_nonzero, "At least some weights should have nonzero importance"
    print(f"  Manifest: {len(manifest.weights)} weights, {manifest.total_params/1e6:.0f}M params")
    print(f"  Top 3 weights by importance:")
    for w in manifest.weights[:3]:
        print(f"    {w['name']}: importance={w['importance']:.4f}")
    # Store manifest globally for later tests
    global _manifest
    _manifest = manifest


# ── Test 2: NVEManifest save/load round-trip ──

@test("NVEManifest save and load round-trip")
def test_manifest_save_load():
    global _manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.nve")
        _manifest.save(save_path)
        assert os.path.exists(save_path), f"File not created at {save_path}"
        file_size = os.path.getsize(save_path)
        print(f"  Saved manifest: {file_size} bytes")
        assert file_size > 100, "Manifest file too small"

        # Load it back
        from nve.streaming_profiler import NVEManifest
        loaded = NVEManifest.load(save_path)
        assert loaded.model_id == _manifest.model_id, "model_id mismatch"
        assert loaded.architecture == _manifest.architecture, "architecture mismatch"
        assert loaded.total_params == _manifest.total_params, "total_params mismatch"
        assert loaded.total_bytes == _manifest.total_bytes, "total_bytes mismatch"
        assert loaded.num_layers == _manifest.num_layers, "num_layers mismatch"
        assert len(loaded.weights) == len(_manifest.weights), "weights count mismatch"
        # Check profiling metadata survived
        assert loaded.profiling_metadata.get("num_prompts") == 3, "profiling metadata lost"
        print(f"  Loaded manifest matches: {loaded.architecture}, {len(loaded.weights)} weights")


# ── Test 3: NVEManifest.tier_placement ──

@test("NVEManifest.tier_placement with budgets")
def test_tier_placement():
    global _manifest
    total = _manifest.total_bytes

    # Give 20% to GPU, 30% to RAM, rest to SSD
    gpu_budget = int(total * 0.2)
    ram_budget = int(total * 0.3)
    placement = _manifest.tier_placement(gpu_budget, ram_budget)

    assert "gpu" in placement, "Missing 'gpu' tier"
    assert "ram" in placement, "Missing 'ram' tier"
    assert "ssd" in placement, "Missing 'ssd' tier"

    gpu_bytes = sum(w["size_bytes"] for w in placement["gpu"])
    ram_bytes = sum(w["size_bytes"] for w in placement["ram"])
    ssd_bytes = sum(w["size_bytes"] for w in placement["ssd"])
    total_placed = len(placement["gpu"]) + len(placement["ram"]) + len(placement["ssd"])

    print(f"  GPU: {len(placement['gpu'])} weights, {gpu_bytes/1024**2:.1f} MB")
    print(f"  RAM: {len(placement['ram'])} weights, {ram_bytes/1024**2:.1f} MB")
    print(f"  SSD: {len(placement['ssd'])} weights, {ssd_bytes/1024**2:.1f} MB")

    assert total_placed == len(_manifest.weights), \
        f"Not all weights placed: {total_placed} vs {len(_manifest.weights)}"
    assert gpu_bytes <= gpu_budget, f"GPU overbudget: {gpu_bytes} > {gpu_budget}"
    assert ram_bytes <= ram_budget, f"RAM overbudget: {ram_bytes} > {ram_budget}"
    assert ssd_bytes > 0, "Some weights should be on SSD"

    # The most important weight overall should be on GPU (greedy by importance)
    if placement["gpu"]:
        gpu_importances = [w["importance"] for w in placement["gpu"]]
        top_weight = _manifest.weights[0]  # sorted descending by importance
        gpu_names = {w["name"] for w in placement["gpu"]}
        assert top_weight["name"] in gpu_names, \
            f"Most important weight '{top_weight['name']}' should be on GPU"
        # Average GPU importance should be >= average SSD importance
        if placement["ssd"]:
            avg_gpu = sum(gpu_importances) / len(gpu_importances)
            ssd_importances = [w["importance"] for w in placement["ssd"]]
            avg_ssd = sum(ssd_importances) / len(ssd_importances)
            print(f"  Avg GPU importance: {avg_gpu:.4f}, Avg SSD importance: {avg_ssd:.4f}")
            assert avg_gpu >= avg_ssd, \
                "Average GPU importance should be >= average SSD importance"


@test("NVEManifest.tier_placement edge cases")
def test_tier_placement_edge():
    global _manifest

    # All on GPU
    placement = _manifest.tier_placement(
        gpu_budget_bytes=_manifest.total_bytes * 2,
        ram_budget_bytes=_manifest.total_bytes * 2,
    )
    assert len(placement["ssd"]) == 0, "With unlimited budget, nothing should be on SSD"
    assert len(placement["gpu"]) == len(_manifest.weights), "All weights should be on GPU"

    # Nothing on GPU or RAM
    placement = _manifest.tier_placement(gpu_budget_bytes=0, ram_budget_bytes=0)
    assert len(placement["gpu"]) == 0, "With zero GPU budget, nothing on GPU"
    assert len(placement["ram"]) == 0, "With zero RAM budget, nothing on RAM"
    assert len(placement["ssd"]) == len(_manifest.weights), "All should be on SSD"


# ── Test 4: TierConfig.from_budget and TierConfig.auto ──

@test("TierConfig.from_budget")
def test_tier_config_from_budget():
    from nve.engine import TierConfig
    model_bytes = 500_000_000  # 500 MB model

    config = TierConfig.from_budget(
        model_bytes=model_bytes,
        gpu_budget_bytes=100_000_000,   # 100 MB GPU
        ram_budget_bytes=200_000_000,   # 200 MB RAM
    )
    assert config.gpu_bytes == 100_000_000
    assert config.ram_bytes == 200_000_000
    assert config.gpu_fraction == 100_000_000 / 500_000_000
    assert config.ram_fraction == 200_000_000 / 500_000_000
    print(f"  gpu_frac={config.gpu_fraction:.2f}, ram_frac={config.ram_fraction:.2f}")
    print(f"  ssd_bytes={config.ssd_bytes}")

    # Edge: GPU budget larger than model
    config2 = TierConfig.from_budget(
        model_bytes=model_bytes,
        gpu_budget_bytes=model_bytes * 2,
        ram_budget_bytes=model_bytes,
    )
    assert config2.gpu_fraction == 1.0, "gpu_fraction should cap at 1.0"
    assert config2.ram_fraction == 0.0, "ram_fraction should be 0 when gpu covers everything"

    # Edge: zero model_bytes should raise
    try:
        TierConfig.from_budget(model_bytes=0, gpu_budget_bytes=100, ram_budget_bytes=100)
        assert False, "Should raise ValueError for zero model_bytes"
    except ValueError:
        print(f"  Correctly raised ValueError for model_bytes=0")


@test("TierConfig.auto")
def test_tier_config_auto():
    from nve.engine import TierConfig
    model_bytes = 500_000_000

    config = TierConfig.auto(model_bytes=model_bytes)
    assert config.gpu_bytes >= 0
    assert config.ram_bytes > 0, "Should detect some RAM"
    assert config.gpu_fraction >= 0
    assert config.ram_fraction >= 0
    print(f"  Auto-detected: gpu={config.gpu_bytes/1024**3:.1f}GB, ram={config.ram_bytes/1024**3:.1f}GB")
    print(f"  gpu_frac={config.gpu_fraction:.2f}, ram_frac={config.ram_fraction:.2f}")


# ── Summary ──

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*60}")
if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
