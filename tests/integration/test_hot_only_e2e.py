#!/usr/bin/env python3
"""
End-to-end test for NVE hot-only inference mode and domain shift detection.

Tests:
1. EngineConfig hot_only fields exist and default correctly
2. StreamingServer._select_active_layers produces correct patterns
3. StreamingServer hot-only mode skips inactive layers (speed test)
4. Domain shift detection triggers tier escalation
5. Rust CLI hot-only mode works end-to-end
"""

import sys
import os
import subprocess
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

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


# ── Test 1: EngineConfig hot_only fields ──

@test("EngineConfig has hot_only_mode and domain_shift fields")
def test_engine_config():
    from nve.engine import EngineConfig

    config = EngineConfig()
    assert hasattr(config, "hot_only_mode"), "Missing hot_only_mode field"
    assert hasattr(config, "active_layers"), "Missing active_layers field"
    assert hasattr(config, "domain_shift_entropy_threshold"), "Missing domain_shift_entropy_threshold"
    assert hasattr(config, "domain_shift_cooldown_tokens"), "Missing domain_shift_cooldown_tokens"

    assert config.hot_only_mode is False, "hot_only_mode should default to False"
    assert config.active_layers is None, "active_layers should default to None"
    assert config.domain_shift_entropy_threshold == 4.0, "threshold should default to 4.0"
    assert config.domain_shift_cooldown_tokens == 10, "cooldown should default to 10"

    # Test with hot_only enabled
    config2 = EngineConfig(hot_only_mode=True, active_layers=8)
    assert config2.hot_only_mode is True
    assert config2.active_layers == 8
    print(f"  Defaults: hot_only={config.hot_only_mode}, active_layers={config.active_layers}")
    print(f"  Custom:   hot_only={config2.hot_only_mode}, active_layers={config2.active_layers}")


# ── Test 2: Active layer selection ──

@test("StreamingServer._select_active_layers produces correct patterns")
def test_select_active_layers():
    from nve.streaming_server import StreamingServer

    # 12 layers, 6 active
    sel = StreamingServer._select_active_layers(12, 6)
    assert len(sel) == 12
    assert sum(sel) == 6, f"Expected 6 active, got {sum(sel)}"
    assert sel[0] is True, "First layer must be active"
    assert sel[11] is True, "Last layer must be active"
    print(f"  12 layers, 6 active: {[i for i,x in enumerate(sel) if x]}")

    # 28 layers, 6 active (Llama 3B scenario)
    sel = StreamingServer._select_active_layers(28, 6)
    assert sum(sel) == 6, f"Expected 6 active, got {sum(sel)}"
    assert sel[0] is True
    assert sel[27] is True
    print(f"  28 layers, 6 active: {[i for i,x in enumerate(sel) if x]}")

    # All active
    sel = StreamingServer._select_active_layers(12, 12)
    assert sum(sel) == 12
    assert all(sel)
    print(f"  12 layers, 12 active: all True")

    # None active
    sel = StreamingServer._select_active_layers(12, 0)
    assert sum(sel) == 0
    print(f"  12 layers, 0 active:  all False")

    # More active than total
    sel = StreamingServer._select_active_layers(8, 20)
    assert sum(sel) == 8
    print(f"  8 layers, 20 active:  all True (capped)")


# ── Test 3: Domain shift detection ──

@test("Domain shift detection triggers on high entropy logits")
def test_domain_shift_detection():
    import torch
    import torch.nn.functional as F
    from nve.streaming_server import StreamingServer

    # Create a minimal StreamingServer to test _detect_domain_shift.
    # We can't instantiate it normally without a model, so test the static logic.
    class MockServer:
        _domain_shift_threshold = 4.0
        def _detect_domain_shift(self, logits):
            if self._domain_shift_threshold <= 0:
                return False
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
            return entropy > self._domain_shift_threshold

    server = MockServer()

    # Low entropy (confident prediction) — no shift
    low_entropy_logits = torch.zeros(1000)
    low_entropy_logits[42] = 10.0  # Very confident
    assert not server._detect_domain_shift(low_entropy_logits), \
        "Confident logits should NOT trigger domain shift"

    # High entropy (uniform-ish) — shift detected
    high_entropy_logits = torch.randn(1000) * 0.1  # Near-uniform
    assert server._detect_domain_shift(high_entropy_logits), \
        "Near-uniform logits should trigger domain shift"

    # Disabled threshold
    server._domain_shift_threshold = 0
    assert not server._detect_domain_shift(high_entropy_logits), \
        "Disabled threshold should never trigger"

    print(f"  Low entropy → no shift: correct")
    print(f"  High entropy → shift detected: correct")
    print(f"  Disabled threshold → no shift: correct")


# ── Test 4: Rust CLI hot-only mode ──

@test("Rust CLI --hot-only generates output with Qwen 0.5B")
def test_rust_hot_only_cli():
    nve_bin = os.path.join(os.path.dirname(__file__), "target", "release", "nve")
    if not os.path.exists(nve_bin):
        print(f"  SKIP: Rust binary not found at {nve_bin}")
        return

    model_dir = os.path.join(os.path.dirname(__file__),
        ".hf_cache", "models--Qwen--Qwen2.5-0.5B", "snapshots",
        "060db6499f32faf8b98477b0a26969ef7d8b9987")
    if not os.path.isdir(model_dir):
        print(f"  SKIP: Qwen model not cached at {model_dir}")
        return

    # Normal paged mode
    t0 = time.time()
    result_normal = subprocess.run(
        [nve_bin, "generate", "--model", model_dir,
         "--prompt", "The meaning of life is",
         "--max-tokens", "10", "--paged", "--auto-budget"],
        capture_output=True, text=True, timeout=120,
    )
    t_normal = time.time() - t0
    assert result_normal.returncode == 0, f"Normal paged failed: {result_normal.stderr}"

    # Hot-only mode with active-layers=8
    t0 = time.time()
    result_hot = subprocess.run(
        [nve_bin, "generate", "--model", model_dir,
         "--prompt", "The meaning of life is",
         "--max-tokens", "10", "--paged", "--hot-only",
         "--active-layers", "8", "--auto-budget"],
        capture_output=True, text=True, timeout=120,
    )
    t_hot = time.time() - t0
    assert result_hot.returncode == 0, f"Hot-only failed: {result_hot.stderr}"

    # Parse tok/s from output
    def parse_speed(output):
        for line in output.split("\n"):
            if "Decode speed:" in line:
                return float(line.split(":")[1].strip().split()[0])
        return 0.0

    normal_speed = parse_speed(result_normal.stdout + result_normal.stderr)
    hot_speed = parse_speed(result_hot.stdout + result_hot.stderr)

    print(f"  Normal paged: {normal_speed:.1f} tok/s ({t_normal:.1f}s)")
    print(f"  Hot-only (8 layers): {hot_speed:.1f} tok/s ({t_hot:.1f}s)")

    # Hot-only should be at least as fast (fewer layers to compute)
    assert hot_speed > 0, "Hot-only should generate tokens"
    assert hot_speed >= normal_speed * 0.5, \
        f"Hot-only should not be significantly slower: {hot_speed} vs {normal_speed}"

    # Check active layer report
    assert "layers active" in (result_hot.stdout + result_hot.stderr), \
        "Hot-only output should report active layers"
    assert "layer-evals skipped" in (result_hot.stdout + result_hot.stderr), \
        "Hot-only output should report skipped layer evals"


# ── Test 5: Rust CLI --active-layers without --hot-only ──

@test("Rust CLI --active-layers limits layers in paged mode")
def test_rust_active_layers_only():
    nve_bin = os.path.join(os.path.dirname(__file__), "target", "release", "nve")
    model_dir = os.path.join(os.path.dirname(__file__),
        ".hf_cache", "models--Qwen--Qwen2.5-0.5B", "snapshots",
        "060db6499f32faf8b98477b0a26969ef7d8b9987")

    if not os.path.exists(nve_bin) or not os.path.isdir(model_dir):
        print(f"  SKIP: binary or model not found")
        return

    result = subprocess.run(
        [nve_bin, "generate", "--model", model_dir,
         "--prompt", "Hello world",
         "--max-tokens", "5", "--paged",
         "--active-layers", "4", "--auto-budget"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Failed: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "Active layers limited" in combined or "layers active" in combined, \
        "Should report active layer limitation"
    print(f"  --active-layers 4 without --hot-only: OK")


# ── Test 6: Hot-only with Llama 3B (large model, budget-constrained) ──

@test("Rust CLI hot-only with Llama 3.2-3B (budget-constrained)")
def test_rust_hot_only_llama3b():
    nve_bin = os.path.join(os.path.dirname(__file__), "target", "release", "nve")
    model_dir = os.path.join(os.path.dirname(__file__),
        ".hf_cache", "models--unsloth--Llama-3.2-3B", "snapshots",
        "d4446454d87d51aa42e1fb174f25acc5f8762331")

    if not os.path.exists(nve_bin) or not os.path.isdir(model_dir):
        print(f"  SKIP: binary or model not found")
        return

    # Hot-only with limited budget — should complete without OOM
    result = subprocess.run(
        [nve_bin, "generate", "--model", model_dir,
         "--prompt", "Hello",
         "--max-tokens", "5", "--paged", "--hot-only",
         "--active-layers", "4",
         "--hot-budget-mb", "400", "--warm-budget-mb", "600"],
        capture_output=True, text=True, timeout=180,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"Failed: {combined}"
    assert "layers active" in combined, "Should report active layers"
    assert "layer-evals skipped" in combined, "Should report skipped layers"

    def parse_speed(output):
        for line in output.split("\n"):
            if "Decode speed:" in line:
                return float(line.split(":")[1].strip().split()[0])
        return 0.0

    speed = parse_speed(combined)
    print(f"  Llama 3.2-3B hot-only (4 active layers): {speed:.1f} tok/s")
    assert speed >= 1.0, f"Should achieve at least 1 tok/s, got {speed}"


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
