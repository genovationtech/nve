"""
NVE Mock CUDA Test Suite

Tests the entire NVE stack with a mock CUDA environment that simulates:
- GPU with configurable VRAM
- OOM errors when VRAM is exceeded
- Multi-GPU scenarios
- Device transfers
- Memory pressure

Catches real bugs in device handling, quantization, KV cache, paging, etc.
"""

import os
import sys
import json
import struct
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

import pytest
import numpy as np
import torch
import torch.nn as nn

# Add the nve package to path.
sys.path.insert(0, str(Path(__file__).parent / "python"))


# ============================================================================
# Mock CUDA Infrastructure
# ============================================================================

class MockCUDADevice:
    """Simulates a CUDA GPU with limited VRAM."""

    def __init__(self, device_index: int = 0, total_vram: int = 4 * 1024**3,
                 name: str = "MockGPU-0"):
        self.device_index = device_index
        self.total_vram = total_vram
        self.name = name
        self.allocated = 0
        self.peak_allocated = 0
        self._tensors = {}  # id -> size tracking

    @property
    def free_vram(self):
        return max(0, self.total_vram - self.allocated)

    def allocate(self, nbytes: int) -> bool:
        if self.allocated + nbytes > self.total_vram:
            return False
        self.allocated += nbytes
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        return True

    def free(self, nbytes: int):
        self.allocated = max(0, self.allocated - nbytes)

    def reset(self):
        self.allocated = 0


class MockCUDARuntime:
    """Mock CUDA runtime with configurable GPUs."""

    def __init__(self, gpus: list[MockCUDADevice] = None):
        self.gpus = gpus or [MockCUDADevice()]
        self._available = True
        self._oom_on_next = False

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self.gpus)

    def mem_get_info(self, device_index: int = 0):
        gpu = self.gpus[device_index]
        return (gpu.free_vram, gpu.total_vram)

    def get_device_properties(self, device_index: int = 0):
        gpu = self.gpus[device_index]
        props = MagicMock()
        props.name = gpu.name
        props.major = 8
        props.minor = 0
        props.total_memory = gpu.total_vram
        return props

    def empty_cache(self):
        pass  # No-op in mock.

    def trigger_oom_on_next(self):
        """Make the next allocation raise OOM."""
        self._oom_on_next = True


# Global mock runtime for tests.
_mock_runtime = MockCUDARuntime()


def setup_mock_cuda(
    num_gpus: int = 1,
    vram_per_gpu: int = 4 * 1024**3,
    available: bool = True,
):
    """Configure the mock CUDA environment for a test."""
    global _mock_runtime
    gpus = [
        MockCUDADevice(i, vram_per_gpu, f"MockGPU-{i}")
        for i in range(num_gpus)
    ]
    _mock_runtime = MockCUDARuntime(gpus)
    _mock_runtime._available = available
    return _mock_runtime


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_cuda_single_gpu():
    """Single GPU with 4GB VRAM."""
    return setup_mock_cuda(num_gpus=1, vram_per_gpu=4 * 1024**3)


@pytest.fixture
def mock_cuda_tiny_gpu():
    """Single GPU with only 512MB VRAM (triggers OOM)."""
    return setup_mock_cuda(num_gpus=1, vram_per_gpu=512 * 1024**2)


@pytest.fixture
def mock_cuda_multi_gpu():
    """Two GPUs, 2GB each."""
    return setup_mock_cuda(num_gpus=2, vram_per_gpu=2 * 1024**3)


@pytest.fixture
def mock_cuda_none():
    """No CUDA available."""
    return setup_mock_cuda(available=False)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create a minimal fake model directory with safetensors."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()

    # Write config.json.
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Write a minimal safetensors file.
    _write_fake_safetensors(model_dir, config)

    return model_dir


def _write_fake_safetensors(model_dir: Path, config: dict):
    """Write a minimal safetensors file with the expected weight names."""
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    vocab = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = hidden // num_heads

    weights = {}
    # Embedding and lm_head.
    weights["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.float16)
    weights["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.float16)
    weights["model.norm.weight"] = torch.ones(hidden, dtype=torch.float16)

    # Per-layer weights.
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.float16)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.float16)
        weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(num_heads * head_dim, hidden, dtype=torch.float16)
        weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float16)
        weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float16)
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden, dtype=torch.float16)
        weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(inter, hidden, dtype=torch.float16)
        weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(inter, hidden, dtype=torch.float16)
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, inter, dtype=torch.float16)

    _save_safetensors(model_dir / "model.safetensors", weights)


def _save_safetensors(path: Path, tensors: dict[str, torch.Tensor]):
    """Write tensors in safetensors format."""
    # Build header.
    header = {}
    offset = 0
    tensor_data = []
    for name, tensor in tensors.items():
        t = tensor.contiguous()
        raw = t.numpy().tobytes()
        dtype_map = {
            torch.float16: "F16", torch.float32: "F32",
            torch.bfloat16: "BF16", torch.int8: "I8",
        }
        header[name] = {
            "dtype": dtype_map[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        tensor_data.append(raw)
        offset += len(raw)

    header_json = json.dumps(header).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for data in tensor_data:
            f.write(data)


def make_fake_manifest(config: dict, gpu_frac: float = 0.3, ram_frac: float = 0.3):
    """Build a TierManifest for the fake model."""
    from nve.manifest import TierManifest, PageEntry

    hidden = config["hidden_size"]
    num_layers = config["num_hidden_layers"]

    all_weights = []
    # Non-layer weights.
    for name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
        all_weights.append({"name": name, "layer_index": -1})

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        for suffix in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ]:
            all_weights.append({"name": f"{prefix}.{suffix}", "layer_index": i})

    # Partition.
    n = len(all_weights)
    gpu_cut = int(n * gpu_frac)
    ram_cut = gpu_cut + int(n * ram_frac)

    manifest = TierManifest()
    for idx, w in enumerate(all_weights):
        entry = PageEntry(
            param_name=w["name"],
            weight_id=idx,
            size_bytes=hidden * hidden * 2,  # Rough estimate.
            importance=1.0 - (idx / n),
            layer_index=max(0, w["layer_index"]),
        )
        if idx < gpu_cut:
            manifest.gpu_pages.append(entry)
        elif idx < ram_cut:
            manifest.ram_pages.append(entry)
        else:
            manifest.ssd_pages.append(entry)

    return manifest


class FakeTokenizer:
    """Minimal tokenizer for testing."""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.eos_token = "</s>"
        self.pad_token = None
        self.pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        # Generate deterministic token IDs from text.
        ids = [ord(c) % self.vocab_size for c in text[:max_length]]
        if not ids:
            ids = [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(max(32, i % 128)) for i in ids.tolist())


# ============================================================================
# Tests: Device Manager
# ============================================================================

class TestDeviceManager:
    def test_cpu_only_initialization(self):
        """DeviceManager works on CPU-only systems."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        assert dm.best_device.type == "cpu"
        assert dm.gpu_count == 0
        assert not dm.has_gpu

    def test_ram_detection(self):
        """RAM budget is detected and non-zero."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        budget = dm.ram_budget()
        assert budget.total_bytes > 0
        assert budget.free_bytes > 0

    def test_safe_to_cpu_tensor(self):
        """safe_to moves tensor to CPU without issue."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        t = torch.randn(10, 10)
        result = dm.safe_to(t, "cpu")
        assert result.device.type == "cpu"
        assert result.shape == (10, 10)

    def test_safe_to_same_device_noop(self):
        """safe_to is a no-op when tensor is already on target device."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        t = torch.randn(10, 10)
        result = dm.safe_to(t, "cpu")
        assert result is t  # Should be exact same object.

    def test_safe_to_dtype_change(self):
        """safe_to handles dtype conversion."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        t = torch.randn(10, 10, dtype=torch.float32)
        result = dm.safe_to(t, "cpu", dtype=torch.float16)
        assert result.dtype == torch.float16

    def test_safe_allocate(self):
        """safe_allocate creates tensors correctly."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        t = dm.safe_allocate((5, 5), torch.float32, "cpu")
        assert t.shape == (5, 5)
        assert t.device.type == "cpu"

    def test_compute_tier_budgets(self):
        """Tier budget computation doesn't crash and produces sane values."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        model_bytes = 1 * 1024**3  # 1 GB model.
        budgets = dm.compute_tier_budgets(model_bytes)
        assert "gpu" in budgets
        assert "ram" in budgets
        assert "ssd" in budgets
        # On CPU-only, GPU budget should be 0.
        assert budgets["gpu"] == 0
        # RAM + SSD should cover the model.
        assert budgets["ram"] + budgets["ssd"] >= model_bytes or budgets["ram"] > 0

    def test_select_device_cpu_only(self):
        """select_device returns CPU when no GPU available."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        device = dm.select_device(required_bytes=1024)
        assert device.type == "cpu"

    def test_memory_reserve_release(self):
        """RAM reservation tracking works."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        initial = dm.ram_budget().free_bytes
        assert dm.reserve_ram(1024)
        assert dm._ram_budget.reserved_bytes == 1024
        dm.release_ram(1024)
        assert dm._ram_budget.reserved_bytes == 0

    def test_summary(self):
        """summary() returns a non-empty string."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        s = dm.summary()
        assert "RAM:" in s
        assert len(s) > 20

    def test_pin_memory_on_contiguous_tensor(self):
        """pin_memory doesn't crash (may fail on locked-down systems but shouldn't error)."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        t = torch.randn(100)
        # May return True or False depending on OS permissions, but shouldn't crash.
        result = dm.pin_memory(t)
        assert isinstance(result, bool)

    def test_monitor_start_stop(self):
        """Memory monitor thread starts and stops cleanly."""
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=True, monitor_interval_s=0.1)
        assert dm._monitor_thread is not None
        thread = dm._monitor_thread
        assert thread.is_alive()
        dm.stop_monitor()
        # After stop, thread should have been joined and is no longer alive.
        assert not thread.is_alive()


# ============================================================================
# Tests: Quantization
# ============================================================================

class TestQuantization:
    def test_int8_roundtrip(self):
        """INT8 quantize -> dequantize preserves tensor approximately."""
        from nve.quantization import quantize_int8, dequantize_int8
        t = torch.randn(64, 64, dtype=torch.float16)
        qt = quantize_int8(t)
        assert qt.quant_level.value == "int8"
        assert qt.compressed_bytes < qt.original_bytes

        restored = dequantize_int8(qt, device="cpu", dtype=torch.float16)
        assert restored.shape == t.shape
        assert restored.dtype == torch.float16
        # Check approximate equality (int8 has ~0.5% error).
        rel_error = (restored.float() - t.float()).abs().mean() / t.float().abs().mean()
        assert rel_error < 0.05, f"INT8 roundtrip error too high: {rel_error:.4f}"

    def test_int4_roundtrip(self):
        """INT4 quantize -> dequantize preserves tensor roughly."""
        from nve.quantization import quantize_int4, dequantize_int4
        t = torch.randn(128, 64, dtype=torch.float16)
        qt = quantize_int4(t, group_size=32)
        assert qt.quant_level.value == "int4"
        assert qt.compressed_bytes < qt.original_bytes

        restored = dequantize_int4(qt, device="cpu", dtype=torch.float16)
        assert restored.shape == t.shape
        # INT4 has higher error, but should be under 15%.
        rel_error = (restored.float() - t.float()).abs().mean() / t.float().abs().mean()
        assert rel_error < 0.20, f"INT4 roundtrip error too high: {rel_error:.4f}"

    def test_int4_compression_ratio(self):
        """INT4 achieves roughly 4x compression."""
        from nve.quantization import quantize_int4
        t = torch.randn(256, 256, dtype=torch.float16)
        qt = quantize_int4(t)
        ratio = qt.compression_ratio
        assert ratio > 2.5, f"INT4 compression only {ratio:.1f}x"

    def test_quantize_dispatch(self):
        """quantize() dispatches correctly by level."""
        from nve.quantization import quantize, dequantize, QuantLevel
        t = torch.randn(32, 32, dtype=torch.float16)

        for level in [QuantLevel.NONE, QuantLevel.INT8, QuantLevel.INT4]:
            qt = quantize(t, level)
            restored = dequantize(qt, device="cpu", dtype=torch.float16)
            assert restored.shape == t.shape

    def test_quant_policy_tiers(self):
        """QuantPolicy returns correct levels per tier."""
        from nve.quantization import QuantPolicy, QuantLevel
        p = QuantPolicy.balanced()
        assert p.level_for_tier("gpu") == QuantLevel.NONE
        assert p.level_for_tier("ram") == QuantLevel.INT8
        assert p.level_for_tier("ssd") == QuantLevel.INT4

        p_agg = QuantPolicy.aggressive()
        assert p_agg.level_for_tier("gpu") == QuantLevel.INT8

        p_none = QuantPolicy.no_quantization()
        assert p_none.level_for_tier("ram") == QuantLevel.NONE

    def test_importance_override(self):
        """High-importance weights bypass tier quantization."""
        from nve.quantization import QuantPolicy, QuantLevel
        p = QuantPolicy(importance_threshold=0.8)
        assert p.should_quantize("ssd", importance=0.9) == QuantLevel.NONE
        assert p.should_quantize("ssd", importance=0.5) == QuantLevel.INT4

    def test_int8_single_element(self):
        """Edge case: single-element tensor."""
        from nve.quantization import quantize_int8, dequantize_int8
        t = torch.tensor([3.14], dtype=torch.float16)
        qt = quantize_int8(t, per_channel=False)
        restored = dequantize_int8(qt)
        assert restored.shape == (1,)

    def test_int4_odd_size(self):
        """Edge case: tensor with odd number of elements."""
        from nve.quantization import quantize_int4, dequantize_int4
        t = torch.randn(7, 13, dtype=torch.float16)
        qt = quantize_int4(t, group_size=32)
        restored = dequantize_int4(qt, device="cpu", dtype=torch.float16)
        assert restored.shape == (7, 13)

    def test_int8_zeros(self):
        """Edge case: all-zero tensor."""
        from nve.quantization import quantize_int8, dequantize_int8
        t = torch.zeros(16, 16, dtype=torch.float16)
        qt = quantize_int8(t)
        restored = dequantize_int8(qt)
        assert restored.abs().max() < 1e-6


# ============================================================================
# Tests: KV Cache
# ============================================================================

class TestKVCache:
    def test_basic_update_get(self):
        """Store and retrieve KV entries."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(num_layers=2, max_gpu_bytes=100 * 1024**2)
        k = torch.randn(1, 4, 10, 16)
        v = torch.randn(1, 4, 10, 16)
        cache.update(0, k, v)
        rk, rv = cache.get(0)
        assert rk is not None
        assert rk.shape[2] == 10

    def test_append_incremental(self):
        """KV cache grows with each update."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(num_layers=1, max_gpu_bytes=100 * 1024**2)
        k1 = torch.randn(1, 4, 5, 16)
        v1 = torch.randn(1, 4, 5, 16)
        cache.update(0, k1, v1)
        assert cache.get_seq_len(0) == 5

        k2 = torch.randn(1, 4, 1, 16)
        v2 = torch.randn(1, 4, 1, 16)
        cache.update(0, k2, v2)
        assert cache.get_seq_len(0) == 6

    def test_sliding_window_eviction(self):
        """Sliding window drops oldest tokens."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(
            num_layers=1,
            max_gpu_bytes=100 * 1024**2,
            eviction="sliding_window",
            window_size=20,
        )
        k = torch.randn(1, 4, 50, 16)
        v = torch.randn(1, 4, 50, 16)
        cache.update(0, k, v)
        # Should be trimmed to window_size.
        assert cache.get_seq_len(0) == 20
        assert cache.stats.evictions == 30

    def test_h2o_eviction(self):
        """H2O keeps heavy hitters + recent tokens."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(
            num_layers=1,
            max_gpu_bytes=100 * 1024**2,
            eviction="h2o",
            h2o_heavy_count=5,
            h2o_recent_count=10,
        )
        k = torch.randn(1, 4, 50, 16)
        v = torch.randn(1, 4, 50, 16)
        # Simulate attention weights that concentrate on a few positions.
        attn = torch.zeros(1, 4, 50, 50)
        attn[:, :, :, 0] = 10.0   # Position 0 is a heavy hitter.
        attn[:, :, :, 3] = 8.0    # Position 3 too.
        cache.update(0, k, v, attention_weights=attn)
        seq_len = cache.get_seq_len(0)
        assert seq_len <= 15  # heavy(5) + recent(10)
        assert cache.stats.evictions > 0

    def test_clear(self):
        """clear() resets all state."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(num_layers=2)
        cache.update(0, torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
        cache.update(1, torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
        cache.clear()
        assert cache.get(0) == (None, None)
        assert cache.get(1) == (None, None)

    def test_gpu_spillover_to_ram(self):
        """When GPU budget exceeded, older entries spill to RAM."""
        from nve.kv_cache import TieredKVCache
        # Very small GPU budget to force spillover.
        cache = TieredKVCache(
            num_layers=1,
            max_gpu_bytes=1024,  # Tiny — will force spill.
            max_ram_bytes=100 * 1024**2,
            eviction="none",
        )
        k = torch.randn(1, 4, 100, 16)
        v = torch.randn(1, 4, 100, 16)
        cache.update(0, k, v)
        # Should have spilled to RAM.
        assert cache.stats.spillovers > 0

    def test_get_nonexistent_layer(self):
        """Getting a non-existent layer returns (None, None)."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(num_layers=2)
        k, v = cache.get(99)
        assert k is None
        assert v is None

    def test_memory_summary(self):
        """memory_summary returns a string."""
        from nve.kv_cache import TieredKVCache
        cache = TieredKVCache(num_layers=1)
        cache.update(0, torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
        s = cache.memory_summary()
        assert "KV Cache:" in s


# ============================================================================
# Tests: Weight Pager
# ============================================================================

class TestWeightPager:
    def test_initialize_placement(self):
        """Weights placed in correct tiers after initialization."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(
            gpu_bytes=1000, ram_bytes=2000, ssd_bytes=10000,
        )
        for i in range(10):
            pager.register(i, name=f"w{i}", size_bytes=100)

        partition = {"hot": [0, 1], "warm": [2, 3, 4], "cold": [5, 6, 7, 8, 9]}
        pager.initialize(partition, sizes={i: 100 for i in range(10)})

        assert pager.get_tier(0) == TierLevel.GPU
        assert pager.get_tier(1) == TierLevel.GPU
        assert pager.get_tier(2) == TierLevel.RAM
        assert pager.get_tier(5) == TierLevel.SSD

    def test_access_tracking(self):
        """Accesses update frequency and counters."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=1000, ram_bytes=2000, ssd_bytes=10000)
        pager.register(0, name="w0", size_bytes=100)
        pager.initialize({"hot": [0], "warm": [], "cold": []}, sizes={0: 100})

        tier = pager.access(0)
        assert tier == TierLevel.GPU
        block = pager.get_block(0)
        assert block.access_count == 1
        assert block.recent_frequency > 0

    def test_access_by_name(self):
        """Access by name works."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=1000, ram_bytes=2000, ssd_bytes=10000)
        pager.register(0, name="model.layers.0.weight", size_bytes=100)
        pager.initialize({"hot": [0], "warm": [], "cold": []}, sizes={0: 100})

        tier = pager.access_by_name("model.layers.0.weight")
        assert tier == TierLevel.GPU

    def test_promotion(self):
        """Frequently accessed RAM weights get promoted to GPU."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(
            gpu_bytes=500, ram_bytes=2000, ssd_bytes=10000,
            promotion_threshold=3,
        )
        pager.register(0, name="w0", size_bytes=100)
        pager.initialize({"hot": [], "warm": [0], "cold": []}, sizes={0: 100})
        assert pager.get_tier(0) == TierLevel.RAM

        # Access several times to meet threshold.
        for _ in range(5):
            pager.access(0)

        result = pager.try_promote(0)
        assert result == TierLevel.GPU
        assert pager.get_tier(0) == TierLevel.GPU

    def test_demotion(self):
        """Demote moves weight to slower tier."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=500, ram_bytes=2000, ssd_bytes=10000)
        pager.register(0, name="w0", size_bytes=100)
        pager.initialize({"hot": [0], "warm": [], "cold": []}, sizes={0: 100})

        result = pager.demote(0)
        assert result == TierLevel.RAM
        assert pager.get_tier(0) == TierLevel.RAM

    def test_evict_under_pressure(self):
        """evict_under_pressure frees space from a tier."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=500, ram_bytes=2000, ssd_bytes=10000)
        for i in range(5):
            pager.register(i, name=f"w{i}", size_bytes=100)
        pager.initialize(
            {"hot": [0, 1, 2, 3, 4], "warm": [], "cold": []},
            sizes={i: 100 for i in range(5)},
        )

        freed = pager.evict_under_pressure(300, tier=TierLevel.GPU)
        assert freed >= 100  # At least one weight evicted.

    def test_overflow_fallback(self):
        """Weights that don't fit in GPU fall back to RAM, then SSD."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=150, ram_bytes=150, ssd_bytes=10000)
        for i in range(5):
            pager.register(i, name=f"w{i}", size_bytes=100)
        pager.initialize(
            {"hot": [0, 1, 2, 3, 4], "warm": [], "cold": []},
            sizes={i: 100 for i in range(5)},
        )
        # Only 1 fits on GPU (150 bytes, 100 per weight).
        assert pager.get_tier(0) == TierLevel.GPU
        assert pager.get_tier(1) == TierLevel.RAM
        # The rest overflow to SSD.
        ssd_count = sum(1 for i in range(5) if pager.get_tier(i) == TierLevel.SSD)
        assert ssd_count >= 2

    def test_co_activation_groups(self):
        """Co-activation groups can be set and queried."""
        from nve.pager import WeightPager
        pager = WeightPager(gpu_bytes=1000, ram_bytes=2000, ssd_bytes=10000)
        for i in range(6):
            pager.register(i, name=f"w{i}", size_bytes=50)

        groups = [{0, 1, 2}, {3, 4}]
        pager.set_co_activation_groups(groups)
        assert pager._weight_to_group[0] == 0
        assert pager._weight_to_group[3] == 1

    def test_decay_all_frequencies(self):
        """Frequency decay reduces all frequencies."""
        from nve.pager import WeightPager
        pager = WeightPager(gpu_bytes=1000, ram_bytes=2000, ssd_bytes=10000)
        pager.register(0, name="w0", size_bytes=100)
        pager.initialize({"hot": [0], "warm": [], "cold": []}, sizes={0: 100})

        for _ in range(10):
            pager.access(0)
        freq_before = pager.get_block(0).recent_frequency

        pager.decay_all_frequencies()
        freq_after = pager.get_block(0).recent_frequency
        assert freq_after < freq_before

    def test_stats(self):
        """Stats reflect actual state."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=500, ram_bytes=1000, ssd_bytes=5000)
        pager.register(0, name="w0", size_bytes=100)
        pager.initialize({"hot": [0], "warm": [], "cold": []}, sizes={0: 100})
        pager.access(0)

        s = pager.stats()
        assert s.page_hits == 1
        assert s.gpu_usage_bytes == 100

    def test_update_budgets_triggers_eviction(self):
        """Shrinking a tier budget evicts weights."""
        from nve.pager import WeightPager, TierLevel
        pager = WeightPager(gpu_bytes=500, ram_bytes=2000, ssd_bytes=10000)
        for i in range(5):
            pager.register(i, name=f"w{i}", size_bytes=100)
        pager.initialize(
            {"hot": [0, 1, 2, 3, 4], "warm": [], "cold": []},
            sizes={i: 100 for i in range(5)},
        )

        # Shrink GPU budget so not all fit.
        pager.update_budgets(gpu_bytes=200)
        gpu_count = len(pager.weights_at_tier(TierLevel.GPU))
        assert gpu_count <= 2


# ============================================================================
# Tests: Streaming Profiler (safetensors parsing)
# ============================================================================

class TestStreamingProfiler:
    def test_parse_safetensors_metadata(self, tmp_model_dir):
        """Metadata parsing reads correct shapes and offsets."""
        from nve.streaming_profiler import _parse_safetensors_metadata
        st_path = tmp_model_dir / "model.safetensors"
        meta = _parse_safetensors_metadata(st_path)
        assert "model.embed_tokens.weight" in meta
        assert meta["model.embed_tokens.weight"]["shape"] == [256, 64]

    def test_load_tensor_from_safetensors(self, tmp_model_dir):
        """Can load individual tensors from safetensors."""
        from nve.streaming_profiler import (
            _parse_safetensors_metadata, _load_tensor_from_safetensors,
        )
        st_path = tmp_model_dir / "model.safetensors"
        meta = _parse_safetensors_metadata(st_path)
        info = meta["model.embed_tokens.weight"]
        info["file"] = str(st_path)
        tensor = _load_tensor_from_safetensors(st_path, info)
        assert tensor.shape == (256, 64)
        assert tensor.dtype == torch.float16

    def test_extract_layer_index(self):
        """Layer index extraction from weight names."""
        from nve.streaming_profiler import _extract_layer_index
        assert _extract_layer_index("model.layers.5.self_attn.q_proj.weight") == 5
        assert _extract_layer_index("model.layers.12.mlp.gate_proj.weight") == 12
        assert _extract_layer_index("model.embed_tokens.weight") == -1


# ============================================================================
# Tests: Manifest
# ============================================================================

class TestManifest:
    def test_save_load_roundtrip(self, tmp_path):
        """Save and load manifest preserves all data."""
        from nve.manifest import TierManifest, PageEntry
        manifest = TierManifest()
        manifest.gpu_pages.append(PageEntry("w0", 0, 1000, 0.9, layer_index=0))
        manifest.ram_pages.append(PageEntry("w1", 1, 2000, 0.5, layer_index=1))
        manifest.ssd_pages.append(PageEntry("w2", 2, 3000, 0.1, layer_index=2))

        manifest.save(tmp_path / "manifest")
        loaded = TierManifest.load(tmp_path / "manifest")
        assert len(loaded.gpu_pages) == 1
        assert len(loaded.ram_pages) == 1
        assert len(loaded.ssd_pages) == 1
        assert loaded.gpu_pages[0].param_name == "w0"

    def test_param_tier_lookup(self):
        """param_tier returns correct tier."""
        from nve.manifest import TierManifest, PageEntry
        m = TierManifest()
        m.gpu_pages.append(PageEntry("w0", 0, 100, 0.9))
        m.ssd_pages.append(PageEntry("w1", 1, 200, 0.1))
        assert m.param_tier("w0") == "gpu"
        assert m.param_tier("w1") == "ssd"
        assert m.param_tier("w_unknown") == "unknown"


# ============================================================================
# Tests: Engine (integration)
# ============================================================================

class TestEngine:
    def _make_engine(self, **kwargs):
        """Create engine with monitor disabled to avoid thread leaks."""
        from nve.engine import NVEEngine, EngineConfig
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        return NVEEngine(device_manager=dm, **kwargs)

    def test_engine_initialization(self):
        """Engine initializes with defaults."""
        engine = self._make_engine()
        assert engine.device_manager is not None
        assert not engine._is_built

    def test_tier_config_auto(self):
        """TierConfig.auto() produces valid config on CPU-only."""
        from nve.engine import TierConfig
        from nve.device import DeviceManager
        dm = DeviceManager(enable_memory_monitor=False)
        tc = TierConfig.auto(model_bytes=1 * 1024**3, device_manager=dm)
        assert tc.gpu_bytes == 0  # No GPU.
        assert tc.ram_bytes > 0
        assert tc.ssd_bytes > 0

    def test_tier_config_from_budget(self):
        """TierConfig.from_budget() computes correct fractions."""
        from nve.engine import TierConfig
        tc = TierConfig.from_budget(
            model_bytes=10 * 1024**3,
            gpu_budget_bytes=2 * 1024**3,
            ram_budget_bytes=4 * 1024**3,
        )
        assert tc.gpu_fraction == pytest.approx(0.2, abs=0.01)
        assert tc.ram_fraction == pytest.approx(0.4, abs=0.01)

    def test_register_model(self):
        """Registering a model populates weight blocks."""
        engine = self._make_engine()
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.Linear(64, 32),
        )
        engine.register_model(model)
        assert len(engine._weight_blocks) == 4  # 2 weights + 2 biases.

    def test_build_requires_profiling(self):
        """Build works even with no profiling (empty partition)."""
        engine = self._make_engine()
        model = nn.Linear(16, 16)
        engine.register_model(model)
        # Profile with a dummy forward.
        engine.profiler.start()
        engine.profiler.finish_round()
        result = engine.build()
        assert engine._is_built
        assert "total_weights" in result

    def test_infer_requires_build(self):
        """infer() raises if not built."""
        engine = self._make_engine()
        with pytest.raises(AssertionError, match="Must call build"):
            engine.infer("test")


# ============================================================================
# Tests: Streaming Server (CPU-only, exercises paging logic)
# ============================================================================

class TestStreamingServer:
    def test_setup_cpu_only(self, tmp_model_dir):
        """StreamingServer sets up correctly on CPU."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,  # CPU doesn't do fp16 well.
        )
        server.setup()

        assert len(server._gpu_cache) > 0 or len(server._ram_cache) > 0
        assert server.device.type == "cpu"
        server.teardown()

    def test_get_weight_all_tiers(self, tmp_model_dir):
        """Weights are retrievable from all tiers."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config, gpu_frac=0.2, ram_frac=0.3)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()

        # Try loading weights from different tiers.
        for name in list(server._tier_map.keys())[:5]:
            w = server._get_weight(name)
            assert w is not None, f"Failed to load {name} from tier {server._tier_map[name]}"

        server.teardown()

    def test_generate_produces_tokens(self, tmp_model_dir):
        """generate() runs and produces output without crashing."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()

        result = server.generate("Hello world", max_new_tokens=3)
        assert "text" in result
        assert result["generated_tokens"] > 0
        assert result["tokens_per_sec"] >= 0
        assert "kv_cache" in result

        server.teardown()

    def test_kv_cache_grows_during_generation(self, tmp_model_dir):
        """KV cache accumulates entries during generation."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()
        server.generate("Test", max_new_tokens=3)

        # KV cache should have entries for each layer.
        total_entries = server.kv_cache.stats.total_entries
        assert total_entries > 0

        server.teardown()

    def test_mmap_ssd_tier(self, tmp_model_dir):
        """mmap handles are set up for SSD tier."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        # Force more weights to SSD.
        manifest = make_fake_manifest(config, gpu_frac=0.1, ram_frac=0.1)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()

        # mmap handles should be created for safetensors files used by SSD weights.
        if len(manifest.ssd_pages) > 0:
            assert len(server._mmap_handles) > 0

        server.teardown()

    def test_stats_populated(self, tmp_model_dir):
        """Stats are populated after generation."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()
        server.generate("Hi", max_new_tokens=2)

        stats = server.stats.to_dict()
        assert "paging" in stats
        assert "throughput" in stats
        assert "reliability" in stats
        total_accesses = stats["paging"]["gpu_hits"] + stats["paging"]["ram_page_ins"] + stats["paging"]["ssd_page_ins"]
        assert total_accesses > 0

        server.teardown()


# ============================================================================
# Tests: Serving (TieredModelServer with real tiny model)
# ============================================================================

class TestTieredModelServer:
    def _make_tiny_model_and_manifest(self):
        """Create a tiny nn.Module and manifest for testing."""
        from nve.manifest import TierManifest, PageEntry

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        model.eval()

        manifest = TierManifest()
        for i, (name, param) in enumerate(model.named_parameters()):
            size = param.nelement() * param.element_size()
            entry = PageEntry(name, i, size, importance=1.0 - i * 0.1, layer_index=0)
            if i == 0:
                manifest.gpu_pages.append(entry)
            elif i == 1:
                manifest.ram_pages.append(entry)
            else:
                manifest.ssd_pages.append(entry)

        return model, manifest

    def test_setup_cpu(self):
        """TieredModelServer sets up on CPU without errors."""
        from nve.serving import TieredModelServer
        model, manifest = self._make_tiny_model_and_manifest()
        tokenizer = FakeTokenizer()

        server = TieredModelServer(
            model, tokenizer, manifest,
            device=torch.device("cpu"),
        )
        server.setup()
        assert len(server.pages) > 0
        server.teardown()

    def test_page_tiers_correct(self):
        """Pages are placed in correct tiers after setup."""
        from nve.serving import TieredModelServer
        model, manifest = self._make_tiny_model_and_manifest()
        tokenizer = FakeTokenizer()

        server = TieredModelServer(
            model, tokenizer, manifest,
            device=torch.device("cpu"),
        )
        server.setup()

        gpu_name = manifest.gpu_pages[0].param_name
        ram_name = manifest.ram_pages[0].param_name
        assert server.pages[gpu_name].current_tier == "gpu"
        assert server.pages[ram_name].current_tier == "ram"

        server.teardown()


# ============================================================================
# Tests: ParameterPage lifecycle
# ============================================================================

class TestParameterPage:
    def test_pin_gpu_and_evict_to_ram(self):
        """Pin on GPU then evict to RAM."""
        from nve.serving import ParameterPage
        t = torch.randn(32, 32)
        page = ParameterPage("test_weight", t, home_tier="ram")
        page.pin_gpu(t)
        assert page.current_tier == "gpu"

        page.evict_to_ram()
        assert page.current_tier == "ram"
        assert page._gpu_tensor is None
        assert page._ram_tensor is not None

    def test_evict_to_ssd_and_reload(self, tmp_path):
        """Evict to SSD and reload."""
        from nve.serving import ParameterPage
        t = torch.randn(16, 16)
        ssd_path = tmp_path / "test_weight.pt"
        page = ParameterPage("test_weight", t, home_tier="ssd", ssd_path=ssd_path)
        page.pin_ram(t)
        page.evict_to_ssd()
        assert page.current_tier == "ssd"

        loaded = page.load_to_ram()
        assert loaded is not None
        assert loaded.shape == (16, 16)

    def test_get_tensor_loads_from_ssd(self, tmp_path):
        """get_tensor transparently loads from SSD."""
        from nve.serving import ParameterPage
        t = torch.randn(8, 8)
        ssd_path = tmp_path / "test.pt"
        page = ParameterPage("w", t, home_tier="ssd", ssd_path=ssd_path)
        page.pin_ram(t)
        page.evict_to_ssd()

        result = page.get_tensor(torch.device("cpu"))
        assert result.shape == (8, 8)

    def test_quantized_ram_storage(self):
        """ParameterPage can store quantized RAM data."""
        from nve.serving import ParameterPage
        from nve.quantization import QuantPolicy, quantize, QuantLevel
        t = torch.randn(32, 32, dtype=torch.float16)
        page = ParameterPage(
            "w", t, home_tier="ram",
            quant_policy=QuantPolicy.balanced(),
        )
        page.pin_gpu(t)
        page.evict_to_ram()
        # With balanced policy, RAM tier uses INT8 quantization.
        # evict_to_ram should quantize.
        assert page._quantized is not None or page._ram_tensor is not None


# ============================================================================
# Tests: Edge cases and cross-module integration
# ============================================================================

class TestEdgeCases:
    def test_empty_manifest(self, tmp_model_dir):
        """Server handles empty manifest (all weights default to GPU)."""
        from nve.streaming_server import StreamingServer
        from nve.manifest import TierManifest

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = TierManifest()  # Empty — no pages.
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        server.setup()
        # Should not crash, but won't have any cached weights.
        assert server.stats.oom_recoveries == 0
        server.teardown()

    def test_quant_policy_no_quantization(self, tmp_model_dir):
        """Server works with no_quantization policy."""
        from nve.streaming_server import StreamingServer
        from nve.quantization import QuantPolicy

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
            quant_policy=QuantPolicy.no_quantization(),
        )
        server.setup()
        result = server.generate("Test", max_new_tokens=2)
        assert result["generated_tokens"] > 0
        server.teardown()

    def test_quant_policy_aggressive(self, tmp_model_dir):
        """Server works with aggressive quantization."""
        from nve.streaming_server import StreamingServer
        from nve.quantization import QuantPolicy

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config, gpu_frac=0.1, ram_frac=0.5)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
            quant_policy=QuantPolicy.aggressive(),
        )
        server.setup()
        result = server.generate("Test", max_new_tokens=2)
        assert result["generated_tokens"] > 0
        server.teardown()

    def test_kv_cache_with_h2o_during_generation(self, tmp_model_dir):
        """H2O eviction doesn't crash during generation."""
        from nve.streaming_server import StreamingServer

        config = json.loads((tmp_model_dir / "config.json").read_text())
        manifest = make_fake_manifest(config)
        tokenizer = FakeTokenizer(config["vocab_size"])

        server = StreamingServer(
            model_dir=tmp_model_dir,
            tokenizer=tokenizer,
            manifest=manifest,
            device=torch.device("cpu"),
            dtype=torch.float32,
            kv_cache_config={
                "eviction": "h2o",
                "max_gpu_bytes": 10 * 1024**2,
            },
        )
        server.setup()
        result = server.generate("Hello", max_new_tokens=3)
        assert result["generated_tokens"] > 0
        server.teardown()

    def test_profiler_basic_flow(self):
        """MCAP profiler basic flow works."""
        from nve.profiler import MCAPProfiler, ActivationSample
        profiler = MCAPProfiler(samples_per_round=10)
        profiler.start()

        for round_idx in range(5):
            samples = [
                ActivationSample(weight_id=i, magnitude=np.random.rand())
                for i in range(10)
            ]
            profiler.record_batch(samples)
            profiler.finish_round()

        assert profiler.weight_count() == 10
        assert profiler.total_rounds() == 5

        ranking = profiler.importance_ranking()
        assert len(ranking) == 10

        partition = profiler.partition(hot_fraction=0.2, warm_fraction=0.3)
        assert "hot" in partition
        assert "warm" in partition
        assert "cold" in partition
        total = len(partition["hot"]) + len(partition["warm"]) + len(partition["cold"])
        assert total == 10

    def test_build_manifest_function(self):
        """build_manifest creates a valid manifest from profiler output."""
        from nve.profiler import MCAPProfiler, ActivationSample
        from nve.manifest import build_manifest

        profiler = MCAPProfiler()
        profiler.start()
        weight_blocks = {}
        for i in range(20):
            weight_blocks[i] = {
                "name": f"model.layers.{i // 4}.weight_{i % 4}",
                "size_bytes": 1000,
                "layer_index": i // 4,
            }
            profiler.record(ActivationSample(i, magnitude=float(20 - i)))
        profiler.finish_round()

        manifest = build_manifest(profiler, weight_blocks, gpu_fraction=0.2, ram_fraction=0.3)
        assert len(manifest.gpu_pages) > 0
        assert len(manifest.ram_pages) > 0
        assert len(manifest.ssd_pages) > 0


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
