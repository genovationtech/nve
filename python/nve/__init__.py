"""
Neural Virtualization Engine (NVE) — Python SDK.

Monte Carlo guided virtual weight paging for neural networks.
Run models that don't fit in memory by intelligently tiering weights
across GPU, RAM, and SSD based on learned importance.

Supports loading and serving any HuggingFace model.
Rust core with Python bindings for ML integration.
"""

from nve.engine import NVEEngine, TierConfig, EngineConfig
from nve.profiler import MCAPProfiler, ActivationSample, PromptDistribution
from nve.pager import WeightPager, PagerStats, TierLevel
from nve.manifest import TierManifest, build_manifest
from nve.serving import TieredModelServer, BaselineServer
from nve.benchmark import Benchmark, BenchmarkConfig
from nve.hub import resolve_model, detect_architecture
from nve.streaming_profiler import StreamingProfiler, NVEManifest
from nve.streaming_server import StreamingServer
from nve.device import DeviceManager, DeviceInfo, MemoryBudget, SystemMemory
from nve.quantization import (
    QuantPolicy, QuantLevel, QuantizedTensor,
    quantize, dequantize, quantize_int8, quantize_int4,
)
from nve.kv_cache import TieredKVCache, EvictionPolicy, KVCacheStats

__version__ = "0.2.0"

__all__ = [
    # Engine.
    "NVEEngine",
    "TierConfig",
    "EngineConfig",
    # Profiler.
    "MCAPProfiler",
    "ActivationSample",
    "PromptDistribution",
    # Pager.
    "WeightPager",
    "PagerStats",
    "TierLevel",
    # Manifest.
    "TierManifest",
    "build_manifest",
    # Serving.
    "TieredModelServer",
    "BaselineServer",
    # Streaming.
    "StreamingProfiler",
    "NVEManifest",
    "StreamingServer",
    # Device.
    "DeviceManager",
    "DeviceInfo",
    "MemoryBudget",
    "SystemMemory",
    # Quantization.
    "QuantPolicy",
    "QuantLevel",
    "QuantizedTensor",
    "quantize",
    "dequantize",
    "quantize_int8",
    "quantize_int4",
    # KV Cache.
    "TieredKVCache",
    "EvictionPolicy",
    "KVCacheStats",
    # Benchmark.
    "Benchmark",
    "BenchmarkConfig",
    # Hub.
    "resolve_model",
    "detect_architecture",
]
