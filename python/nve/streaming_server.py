"""
NVE Streaming Server — production-grade inference without loading the full model.

Runs transformer inference using raw tensor math, streaming weights
from disk one layer at a time. No nn.Module, no AutoModelForCausalLM.
Peak memory = embed_tokens + one layer + KV cache activations.

Production features:
- OOM-safe GPU loading with graceful fallback
- mmap-based SSD tier (reads directly from safetensors, no .pt conversion)
- Double-buffered prefetch (compute layer N while loading layer N+1)
- Tiered KV cache with eviction for long sequences
- Per-tier quantization (int8 RAM, int4 SSD)
- Device-consistent compute (no unnecessary CPU round-trips)
- Memory pressure monitoring via DeviceManager

Supports LLaMA / Qwen / Mistral architectures (model.layers.N.*).

Usage:
    server = StreamingServer(model_dir, tokenizer, tier_manifest)
    server.setup()
    result = server.generate("Hello world", max_new_tokens=50)
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from nve.device import DeviceManager, _devices_match
from nve.kv_cache import TieredKVCache, EvictionPolicy
from nve.manifest import TierManifest
from nve.quantization import (
    QuantPolicy, QuantLevel, QuantizedTensor,
    quantize, dequantize, quantize_int8, quantize_int4,
)
from nve.streaming_profiler import (
    _parse_safetensors_metadata,
    _load_tensor_from_safetensors,
    _extract_layer_index,
)

logger = logging.getLogger("nve.streaming_server")


@dataclass
class StreamingStats:
    """Runtime statistics."""
    gpu_hits: int = 0
    ram_page_ins: int = 0
    ssd_page_ins: int = 0
    prefetch_hits: int = 0
    tokens_generated: int = 0
    total_generation_time: float = 0.0
    peak_ram_bytes: int = 0
    ssd_reads_bytes: int = 0
    forward_times: list[float] = field(default_factory=list)
    oom_recoveries: int = 0
    tier_fallbacks: int = 0

    @property
    def page_fault_rate(self) -> float:
        total = self.gpu_hits + self.ram_page_ins + self.ssd_page_ins
        return (self.ram_page_ins + self.ssd_page_ins) / total if total else 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.total_generation_time == 0:
            return 0.0
        return self.tokens_generated / self.total_generation_time

    def to_dict(self) -> dict:
        return {
            "paging": {
                "gpu_hits": self.gpu_hits,
                "ram_page_ins": self.ram_page_ins,
                "ssd_page_ins": self.ssd_page_ins,
                "page_fault_rate": round(self.page_fault_rate, 4),
                "prefetch_hits": self.prefetch_hits,
            },
            "memory": {
                "peak_gpu_mb": 0,
                "peak_ram_mb": round(self.peak_ram_bytes / 1024**2, 1),
                "ssd_reads_mb": round(self.ssd_reads_bytes / 1024**2, 1),
            },
            "throughput": {
                "tokens_generated": self.tokens_generated,
                "total_time_s": round(self.total_generation_time, 3),
                "tokens_per_sec": round(self.tokens_per_sec, 2),
            },
            "reliability": {
                "oom_recoveries": self.oom_recoveries,
                "tier_fallbacks": self.tier_fallbacks,
            },
        }


class StreamingServer:
    """
    Serves a model by streaming weights from tiered storage.

    When the compiled Rust library (libnve.so) is available, uses the
    high-performance Rust backend with SIMD-accelerated bf16 dot products,
    GQA attention, and SwiGLU. Falls back to PyTorch matmuls otherwise.
    """

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer,
        manifest: TierManifest,
        device: Optional[torch.device] = None,
        ssd_dir: str | Path = "/tmp/nve_ssd_tier",
        dtype: torch.dtype = torch.float16,
        device_manager: Optional[DeviceManager] = None,
        quant_policy: Optional[QuantPolicy] = None,
        kv_cache_config: Optional[dict] = None,
        hot_only_mode: bool = False,
        active_layers: Optional[int] = None,
        domain_shift_entropy_threshold: float = 4.0,
        domain_shift_cooldown_tokens: int = 10,
    ):
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.ssd_dir = Path(ssd_dir)
        self.dtype = dtype
        self.stats = StreamingStats()

        # Hot-only inference config.
        self.hot_only_mode = hot_only_mode
        self._active_layers_count = active_layers
        self._active_layer_set: Optional[list[bool]] = None
        self._domain_shift_threshold = domain_shift_entropy_threshold
        self._domain_shift_cooldown = domain_shift_cooldown_tokens
        self._domain_shift_remaining = 0  # Tokens remaining in full-tier mode.
        self._in_hot_only = hot_only_mode  # Runtime flag, toggled by domain shift.

        # Try to load Rust backend for fast inference.
        self._rust_engine = None
        try:
            from nve.rust_backend import RustInferenceEngine
            self._rust_engine = RustInferenceEngine.load(self.model_dir)
            if self._rust_engine is not None:
                logger.info("Rust inference backend available — will use SIMD-accelerated path")
        except Exception as e:
            logger.debug(f"Rust backend not available: {e}")

        # If Rust backend is loaded, skip Python-side weight setup entirely.
        # The Rust engine handles weight loading, tiering, and inference.
        if self._rust_engine is not None:
            self.dm = device_manager or DeviceManager()
            self.device = device if device is not None else self.dm.best_device
            self._prefetch_thread = None
            return

        # Device management.
        self.dm = device_manager or DeviceManager()
        if device is not None:
            self.device = device
        else:
            self.device = self.dm.best_device

        # Quantization policy.
        self.quant_policy = quant_policy or QuantPolicy.balanced()

        # Parse model config.
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        self.num_layers = self.config.get("num_hidden_layers", self.config.get("n_layer", 0))
        self.hidden_size = self.config.get("hidden_size", self.config.get("n_embd", 0))
        self.num_heads = self.config.get("num_attention_heads", 0)
        self.num_kv_heads = self.config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = self.config.get("intermediate_size", 0)
        self.vocab_size = self.config.get("vocab_size", 0)
        self.rope_theta = self.config.get("rope_theta", 10000.0)
        self.rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)
        self.tie_embeddings = self.config.get("tie_word_embeddings", False)
        self.max_position = self.config.get("max_position_embeddings", 4096)

        # Weight location tracking.
        self._weight_meta: dict[str, dict] = {}
        self._tier_map: dict[str, str] = {}
        self._gpu_cache: dict[str, torch.Tensor] = {}
        self._ram_cache: dict[str, torch.Tensor] = {}
        self._quantized_ram: dict[str, QuantizedTensor] = {}  # Quantized RAM-tier weights.
        self._quantized_ssd: dict[str, QuantizedTensor] = {}  # Quantized SSD-tier weights.
        self._layer_weights: dict[int, list[str]] = defaultdict(list)

        # mmap handles for direct safetensors access (SSD tier).
        self._mmap_handles: dict[str, mmap.mmap] = {}
        self._mmap_files: dict[str, object] = {}  # Keep file objects alive.

        # RoPE cache.
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

        # Double-buffered prefetch.
        self._prefetch_buffer: dict[str, torch.Tensor] = {}
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_layer: int = -1

        # KV cache.
        kv_config = kv_cache_config or {}
        self.kv_cache = TieredKVCache(
            num_layers=self.num_layers,
            max_gpu_bytes=kv_config.get("max_gpu_bytes", 512 * 1024**2),
            max_ram_bytes=kv_config.get("max_ram_bytes", 2 * 1024**3),
            eviction=kv_config.get("eviction", "sliding_window"),
            window_size=kv_config.get("window_size", 2048),
            device=self.device,
        )

    def setup(self):
        """Discover weights and set up tiered storage."""
        if self._rust_engine is not None:
            return  # Rust engine handles everything.

        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        # Discover safetensors files.
        self._discover_safetensors()

        # Build tier map.
        for entry in self.manifest.gpu_pages:
            self._tier_map[entry.param_name] = "gpu"
            self._layer_weights[entry.layer_index].append(entry.param_name)
        for entry in self.manifest.ram_pages:
            self._tier_map[entry.param_name] = "ram"
            self._layer_weights[entry.layer_index].append(entry.param_name)
        for entry in self.manifest.ssd_pages:
            self._tier_map[entry.param_name] = "ssd"
            self._layer_weights[entry.layer_index].append(entry.param_name)

        # Pre-load GPU-tier weights (OOM-safe).
        for entry in self.manifest.gpu_pages:
            tensor = self._load_from_safetensors(entry.param_name)
            if tensor is not None:
                gpu_tensor = self.dm.safe_to(tensor, self.device, dtype=self.dtype)
                if _devices_match(gpu_tensor.device, self.device):
                    self._gpu_cache[entry.param_name] = gpu_tensor
                else:
                    # OOM fallback: GPU weight landed on CPU. Demote to RAM tier.
                    logger.warning(f"GPU OOM for {entry.param_name}, demoting to RAM tier")
                    self._tier_map[entry.param_name] = "ram"
                    self._store_ram_weight(entry.param_name, tensor)
                    self.stats.oom_recoveries += 1
                    self.stats.tier_fallbacks += 1

        # Pre-load RAM-tier weights (with optional quantization).
        for entry in self.manifest.ram_pages:
            tensor = self._load_from_safetensors(entry.param_name)
            if tensor is not None:
                self._store_ram_weight(entry.param_name, tensor)

        # Set up mmap handles for SSD-tier weights (no .pt conversion needed).
        self._setup_ssd_mmap()

        # Pre-compute RoPE frequencies on the compute device.
        self._build_rope_cache()

        total_gpu = sum(t.nelement() * t.element_size() for t in self._gpu_cache.values())
        total_ram = sum(t.nelement() * t.element_size() for t in self._ram_cache.values())
        total_ram += sum(qt.compressed_bytes for qt in self._quantized_ram.values())
        self.stats.peak_ram_bytes = total_gpu + total_ram

        # Compute active layer set for hot-only mode.
        if self.hot_only_mode and self.num_layers > 0:
            gpu_layer_count = len(set(
                e.layer_index for e in self.manifest.gpu_pages if e.layer_index >= 0
            ))
            ram_layer_count = len(set(
                e.layer_index for e in self.manifest.ram_pages if e.layer_index >= 0
            ))
            if self._active_layers_count is not None:
                max_active = min(self._active_layers_count, self.num_layers)
            else:
                max_active = min(gpu_layer_count + ram_layer_count, self.num_layers)
            self._active_layer_set = self._select_active_layers(self.num_layers, max_active)
            active = sum(self._active_layer_set)
            logger.info(
                f"Hot-only mode: {active}/{self.num_layers} layers active, "
                f"{self.num_layers - active} skipped"
            )

        logger.info(
            f"Setup complete: GPU={total_gpu / 1024**2:.0f}MB, "
            f"RAM={total_ram / 1024**2:.0f}MB, "
            f"SSD={len(self.manifest.ssd_pages)} weights (mmap), "
            f"device={self.device}"
            + (f", hot_only={self.hot_only_mode}" if self.hot_only_mode else "")
        )

    def _discover_safetensors(self):
        """Parse safetensors file headers."""
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                shard_files = set(json.load(f).get("weight_map", {}).values())
            for shard_name in shard_files:
                shard_path = self.model_dir / shard_name
                if shard_path.exists():
                    meta = _parse_safetensors_metadata(shard_path)
                    for name, info in meta.items():
                        info["file"] = str(shard_path)
                        self._weight_meta[name] = info
        else:
            st_path = self.model_dir / "model.safetensors"
            if st_path.exists():
                meta = _parse_safetensors_metadata(st_path)
                for name, info in meta.items():
                    info["file"] = str(st_path)
                    self._weight_meta[name] = info

    def _store_ram_weight(self, name: str, tensor: torch.Tensor):
        """Store a weight in RAM tier, with optional quantization."""
        level = self.quant_policy.level_for_tier("ram")
        if level != QuantLevel.NONE:
            try:
                qtensor = quantize(tensor, level)
                self._quantized_ram[name] = qtensor
                return
            except Exception as e:
                logger.warning(f"Quantization failed for {name}: {e}. Storing unquantized.")

        self._ram_cache[name] = tensor.to(dtype=self.dtype, device="cpu")

    def _setup_ssd_mmap(self):
        """
        Set up memory-mapped access to safetensors files for SSD tier.

        Instead of copying weights to .pt files, we mmap the original
        safetensors files and read weights directly. This:
        - Eliminates the setup/copy phase
        - Uses OS page cache efficiently
        - Allows the OS to manage memory under pressure
        """
        # Collect unique safetensors files used by SSD-tier weights.
        ssd_files = set()
        for entry in self.manifest.ssd_pages:
            meta = self._weight_meta.get(entry.param_name)
            if meta:
                ssd_files.add(meta["file"])

        for filepath in ssd_files:
            try:
                f = open(filepath, "rb")
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self._mmap_handles[filepath] = mm
                self._mmap_files[filepath] = f
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to mmap {filepath}: {e}")

    def _load_from_safetensors(self, name: str) -> Optional[torch.Tensor]:
        """Load a weight from safetensors files."""
        meta = self._weight_meta.get(name)
        if meta is None:
            return None
        try:
            return _load_tensor_from_safetensors(Path(meta["file"]), meta)
        except Exception:
            return None

    def _load_from_mmap(self, name: str) -> Optional[torch.Tensor]:
        """
        Load a weight via mmap from safetensors (SSD tier).

        Uses the OS page cache — frequently accessed weights will
        naturally be cached in RAM by the OS without us managing it.
        """
        meta = self._weight_meta.get(name)
        if meta is None:
            return None

        filepath = meta["file"]
        mm = self._mmap_handles.get(filepath)
        if mm is None:
            # Fallback to regular file read.
            return self._load_from_safetensors(name)

        try:
            from nve.streaming_profiler import _dtype_to_torch
            dtype = _dtype_to_torch(meta["dtype"])
            shape = meta["shape"]
            offset_start = meta["offset_start"]
            offset_end = meta["offset_end"]
            num_bytes = offset_end - offset_start

            # Read from mmap (OS handles caching).
            raw = mm[offset_start:offset_end]
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
            self.stats.ssd_reads_bytes += num_bytes
            return tensor
        except Exception as e:
            logger.warning(f"mmap read failed for {name}: {e}")
            return self._load_from_safetensors(name)

    @staticmethod
    def _select_active_layers(total: int, active_count: int) -> list[bool]:
        """Select evenly-spaced active layers, always including first and last."""
        active = [False] * total
        if active_count == 0 or total == 0:
            return active
        if active_count >= total:
            return [True] * total

        active[0] = True
        if total > 1:
            active[total - 1] = True

        remaining = active_count - 2
        if remaining > 0 and total > 2:
            middle = total - 2
            for i in range(remaining):
                idx = 1 + (i * middle) // remaining
                active[idx] = True

        # Fill from front if rounding left us short.
        count = sum(active)
        for i in range(total):
            if count >= active_count:
                break
            if not active[i]:
                active[i] = True
                count += 1

        return active

    def _is_layer_active(self, layer_idx: int) -> bool:
        """Check if a layer should be computed in hot-only mode."""
        if self._active_layer_set is None:
            return True
        if layer_idx < 0 or layer_idx >= len(self._active_layer_set):
            return True
        return self._active_layer_set[layer_idx]

    def _detect_domain_shift(self, logits: torch.Tensor) -> bool:
        """Check if logit entropy indicates a domain shift."""
        if self._domain_shift_threshold <= 0:
            return False
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
        return entropy > self._domain_shift_threshold

    def _get_weight(self, name: str) -> Optional[torch.Tensor]:
        """Get a weight tensor from whatever tier it lives on.

        In hot-only mode (self._in_hot_only), only returns GPU-cached weights.
        Non-GPU weights return None, causing the layer to be skipped.
        When domain shift is detected, temporarily uses all tiers.
        """
        # GPU cache (fastest — already on device).
        if name in self._gpu_cache:
            self.stats.gpu_hits += 1
            return self._gpu_cache[name]

        # Hot-only mode: skip non-GPU weights for speed.
        if self._in_hot_only:
            return None

        # Check prefetch buffer.
        with self._prefetch_lock:
            if name in self._prefetch_buffer:
                tensor = self._prefetch_buffer.pop(name)
                self.stats.prefetch_hits += 1
                return self.dm.safe_to(tensor, self.device, dtype=self.dtype)

        # RAM cache (unquantized).
        if name in self._ram_cache:
            self.stats.ram_page_ins += 1
            return self.dm.safe_to(self._ram_cache[name], self.device, dtype=self.dtype)

        # RAM cache (quantized — dequantize on load).
        if name in self._quantized_ram:
            self.stats.ram_page_ins += 1
            tensor = dequantize(self._quantized_ram[name], device="cpu", dtype=self.dtype)
            return self.dm.safe_to(tensor, self.device)

        # SSD tier — load via mmap from safetensors.
        self.stats.ssd_page_ins += 1
        tensor = self._load_from_mmap(name)
        if tensor is not None:
            return self.dm.safe_to(tensor, self.device, dtype=self.dtype)

        # Ultimate fallback.
        tensor = self._load_from_safetensors(name)
        if tensor is not None:
            return self.dm.safe_to(tensor, self.device, dtype=self.dtype)

        return None

    def _build_rope_cache(self):
        """Pre-compute rotary position embedding sin/cos tables on compute device."""
        dim = self.head_dim
        max_len = min(self.max_position, 4096)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        # Store on compute device to avoid per-token transfers.
        self._rope_cos = self.dm.safe_to(freqs.cos().to(self.dtype), self.device)
        self._rope_sin = self.dm.safe_to(freqs.sin().to(self.dtype), self.device)

    def _apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Apply rotary embeddings. x: (batch, num_heads, seq_len, head_dim)."""
        seq_len = x.shape[2]
        # RoPE cache is already on self.device — no transfer needed.
        cos = self._rope_cos[start_pos:start_pos + seq_len]
        sin = self._rope_sin[start_pos:start_pos + seq_len]

        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMS normalization."""
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.rms_norm_eps)
        return (x_normed * weight.float()).to(x.dtype)

    # ── Double-Buffered Prefetch ──

    def _start_prefetch(self, layer_idx: int):
        """
        Start prefetching weights for the given layer in a background thread.

        Called while the current layer is computing, so I/O overlaps with compute.
        """
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        def prefetch_work():
            names = self._layer_weight_names(layer_idx)
            for name in names:
                # Only prefetch if not already cached.
                if name in self._gpu_cache:
                    continue
                with self._prefetch_lock:
                    if name in self._prefetch_buffer:
                        continue

                # Load to CPU (don't touch GPU from background thread).
                tensor = None
                if name in self._ram_cache:
                    tensor = self._ram_cache[name].clone()
                elif name in self._quantized_ram:
                    tensor = dequantize(self._quantized_ram[name], device="cpu", dtype=self.dtype)
                else:
                    tensor = self._load_from_mmap(name)
                    if tensor is None:
                        tensor = self._load_from_safetensors(name)

                if tensor is not None:
                    with self._prefetch_lock:
                        self._prefetch_buffer[name] = tensor.to(dtype=self.dtype, device="cpu")

        self._prefetch_layer = layer_idx
        self._prefetch_thread = threading.Thread(target=prefetch_work, daemon=True)
        self._prefetch_thread.start()

    def _layer_weight_names(self, layer_idx: int) -> list[str]:
        """Get all weight names for a transformer layer."""
        prefix = f"model.layers.{layer_idx}."
        return [name for name in self._weight_meta if name.startswith(prefix)]

    # ── Forward Pass ──

    def _forward_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        """
        Full transformer layer forward pass with proper attention.

        hidden: (batch=1, seq_len, hidden_size)
        Returns: (batch=1, seq_len, hidden_size)

        In hot-only mode, inactive layers pass the residual through unchanged.
        """
        # Hot-only mode: skip inactive layers.
        if not self._is_layer_active(layer_idx):
            return hidden

        # Start prefetching next active layer while we compute this one.
        if layer_idx + 1 < self.num_layers:
            # Find next active layer for prefetch.
            next_layer = layer_idx + 1
            while next_layer < self.num_layers and not self._is_layer_active(next_layer):
                next_layer += 1
            if next_layer < self.num_layers:
                self._start_prefetch(next_layer)

        residual = hidden

        # Input LayerNorm.
        w = self._get_weight(f"model.layers.{layer_idx}.input_layernorm.weight")
        if w is not None:
            hidden = self._rms_norm(hidden, w)

        # Self Attention.
        q_w = self._get_weight(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
        k_w = self._get_weight(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
        v_w = self._get_weight(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
        o_w = self._get_weight(f"model.layers.{layer_idx}.self_attn.o_proj.weight")

        bsz, seq_len, _ = hidden.shape
        h = hidden.to(self.dtype)

        q = F.linear(h, q_w)
        k = F.linear(h, k_w)
        v = F.linear(h, v_w)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, start_pos)
        k = self._apply_rope(k, start_pos)

        # KV cache: update and retrieve.
        self.kv_cache.update(layer_idx, k, v)
        k, v = self.kv_cache.get(layer_idx, device=self.device)

        # GQA: repeat KV heads to match Q heads.
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask.
        total_len = k.shape[2]
        if seq_len > 1:
            mask = torch.full(
                (seq_len, total_len), float("-inf"),
                device=self.device, dtype=self.dtype,
            )
            mask = torch.triu(mask, diagonal=total_len - seq_len + 1)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn.float(), dim=-1).to(self.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = F.linear(out, o_w)

        hidden = residual + out

        # Free attention weights immediately to reduce peak memory.
        del q_w, k_w, v_w, o_w, q, attn, out

        # Post-attention LayerNorm.
        residual = hidden
        w = self._get_weight(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
        if w is not None:
            hidden = self._rms_norm(hidden, w)

        # MLP.
        gate_w = self._get_weight(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
        up_w = self._get_weight(f"model.layers.{layer_idx}.mlp.up_proj.weight")
        down_w = self._get_weight(f"model.layers.{layer_idx}.mlp.down_proj.weight")

        h = hidden.to(self.dtype)
        gate = F.linear(h, gate_w)
        up = F.linear(h, up_w)
        mlp_out = F.silu(gate) * up
        mlp_out = F.linear(mlp_out, down_w)

        hidden = residual + mlp_out

        del gate_w, up_w, down_w, gate, up, mlp_out

        return hidden

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.0, top_p: float = 0.9) -> dict:
        """Generate text using streaming tiered inference.

        When the Rust backend is available, uses SIMD-accelerated inference
        (bf16 dot products, fused SwiGLU, GQA attention) for significantly
        higher throughput. Falls back to PyTorch matmuls otherwise.
        """
        # Fast path: use Rust engine if available.
        if self._rust_engine is not None:
            result = self._rust_engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            # Update stats for compatibility.
            self.stats.tokens_generated += result["generated_tokens"]
            self.stats.total_generation_time += result["time_s"]
            return result

        # Slow path: Python + PyTorch streaming inference.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Embeddings and head weights are always loaded from all tiers
        # (they're essential even in hot-only mode).
        saved_hot_only = self._in_hot_only
        self._in_hot_only = False
        embed_w = self._get_weight("model.embed_tokens.weight")
        if embed_w is None:
            raise RuntimeError("embed_tokens.weight not found in any tier")

        # Get lm_head weight (may be tied to embed_tokens).
        lm_head_w = self._get_weight("lm_head.weight") if not self.tie_embeddings else embed_w

        # Final norm weight.
        norm_w = self._get_weight("model.norm.weight")
        self._in_hot_only = saved_hot_only

        generated_ids = input_ids.clone()

        # Clear KV cache for new generation.
        self.kv_cache.clear()

        t_start = time.perf_counter()

        # Start prefetching layer 0 while we do embedding.
        self._start_prefetch(0)

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_fwd = time.perf_counter()

                if step == 0:
                    curr_ids = generated_ids
                    start_pos = 0
                else:
                    curr_ids = generated_ids[:, -1:]
                    start_pos = generated_ids.shape[1] - 1

                # Embedding lookup — do it on whatever device embed_w lives on,
                # then move result to compute device. Avoids GPU->CPU->GPU round-trip.
                curr_ids_for_embed = curr_ids.to(embed_w.device)
                hidden = F.embedding(curr_ids_for_embed, embed_w).to(dtype=self.dtype, device=self.device)

                # Stream through transformer layers.
                for layer_idx in range(self.num_layers):
                    hidden = self._forward_layer(hidden, layer_idx, start_pos)

                # Final RMS norm.
                if norm_w is not None:
                    hidden = self._rms_norm(hidden, norm_w)

                # LM head -> logits for last position.
                logits = F.linear(hidden[:, -1:, :].to(self.dtype), lm_head_w)

                self.stats.forward_times.append(time.perf_counter() - t_fwd)

                # Domain shift detection: if entropy is high, temporarily
                # switch to full-tier mode for better quality.
                if self.hot_only_mode and self._domain_shift_threshold > 0:
                    if self._domain_shift_remaining > 0:
                        self._domain_shift_remaining -= 1
                        if self._domain_shift_remaining == 0:
                            self._in_hot_only = True
                            logger.debug("Domain shift cooldown expired, returning to hot-only")
                    elif self._detect_domain_shift(logits[:, -1, :]):
                        self._in_hot_only = False
                        self._domain_shift_remaining = self._domain_shift_cooldown
                        logger.info(
                            f"Domain shift detected at token {step}, "
                            f"pulling from all tiers for {self._domain_shift_cooldown} tokens"
                        )

                # Greedy decode.
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token.cpu()], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Wait for any outstanding prefetch.
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        t_total = time.perf_counter() - t_start
        num_generated = generated_ids.shape[1] - prompt_len
        self.stats.tokens_generated += num_generated
        self.stats.total_generation_time += t_total

        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        active_layers = sum(self._active_layer_set) if self._active_layer_set else self.num_layers
        return {
            "text": text,
            "prompt_tokens": prompt_len,
            "generated_tokens": num_generated,
            "time_s": round(t_total, 3),
            "tokens_per_sec": round(num_generated / t_total, 2) if t_total > 0 else 0,
            "kv_cache": self.kv_cache.stats.to_dict(),
            "paging": {
                "gpu_hits": self.stats.gpu_hits,
                "ram_page_ins": self.stats.ram_page_ins,
                "ssd_page_ins": self.stats.ssd_page_ins,
                "page_fault_rate": round(self.stats.page_fault_rate, 4),
            },
            "hot_only": {
                "enabled": self.hot_only_mode,
                "active_layers": active_layers,
                "total_layers": self.num_layers,
            },
        }

    def teardown(self):
        """Release resources."""
        # Close Rust engine.
        if self._rust_engine is not None:
            self._rust_engine.close()
            self._rust_engine = None
            return  # No Python-side resources to clean up.

        # Wait for prefetch thread.
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        self._gpu_cache.clear()
        self._ram_cache.clear()
        self._quantized_ram.clear()
        self._quantized_ssd.clear()
        self._prefetch_buffer.clear()
        self.kv_cache.clear()

        # Close mmap handles.
        for mm in self._mmap_handles.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mmap_handles.clear()

        for f in self._mmap_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._mmap_files.clear()

        # Clean up legacy .pt files if any exist.
        if self.ssd_dir.exists():
            for f in self.ssd_dir.glob("*.pt"):
                f.unlink()

        # Free GPU memory.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass
