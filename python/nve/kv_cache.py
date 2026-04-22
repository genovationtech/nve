"""
NVE Tiered KV Cache — memory-bounded key-value cache with eviction.

On small devices, KV cache can exceed available memory before generation
finishes. This module provides:

- Memory-bounded KV storage with configurable limits
- Sliding window eviction (oldest tokens dropped)
- Heavy Hitter Oracle (H2O) eviction (keep high-attention tokens)
- KV cache quantization (fp16 -> int8) for 2x memory reduction
- Spillover to CPU RAM when GPU is full
- Per-layer cache with independent eviction policies

Usage:
    cache = TieredKVCache(
        num_layers=32,
        max_gpu_bytes=512 * 1024**2,  # 512 MB
        max_ram_bytes=2 * 1024**3,     # 2 GB
        eviction="h2o",
    )
    cache.update(layer_idx=0, key=k, value=v)
    k, v = cache.get(layer_idx=0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from nve.device import _devices_match

logger = logging.getLogger("nve.kv_cache")


def _safe_to_device(tensor, device, dtype=None):
    """Move tensor to device with OOM recovery — falls back to CPU on CUDA OOM."""
    import torch

    target = torch.device(device) if isinstance(device, str) else device
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype

    if target.type == "cpu":
        return tensor.to(device=target, **kwargs)

    try:
        return tensor.to(device=target, **kwargs)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"KV cache OOM moving to {target}, falling back to CPU")
        torch.cuda.empty_cache()
        try:
            return tensor.to(device=target, **kwargs)
        except torch.cuda.OutOfMemoryError:
            return tensor.to(device="cpu", **kwargs)


class EvictionPolicy(Enum):
    SLIDING_WINDOW = "sliding_window"
    H2O = "h2o"                        # Heavy Hitter Oracle
    NONE = "none"                      # No eviction (will OOM on long sequences)


@dataclass
class KVCacheStats:
    """Runtime statistics for the KV cache."""
    total_entries: int = 0
    gpu_entries: int = 0
    ram_entries: int = 0
    evictions: int = 0
    spillovers: int = 0  # GPU -> RAM moves
    quantized_entries: int = 0
    gpu_bytes: int = 0
    ram_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "gpu_entries": self.gpu_entries,
            "ram_entries": self.ram_entries,
            "evictions": self.evictions,
            "spillovers": self.spillovers,
            "gpu_bytes_mb": round(self.gpu_bytes / 1024**2, 1),
            "ram_bytes_mb": round(self.ram_bytes / 1024**2, 1),
        }


@dataclass
class LayerCache:
    """KV cache for a single transformer layer."""
    key: Optional["torch.Tensor"] = None
    value: Optional["torch.Tensor"] = None
    # Attention scores accumulated for H2O eviction.
    attention_scores: Optional["torch.Tensor"] = None
    # Track which tokens are on GPU vs RAM.
    gpu_seq_len: int = 0
    ram_key: Optional["torch.Tensor"] = None
    ram_value: Optional["torch.Tensor"] = None
    quantized: bool = False


class TieredKVCache:
    """
    Memory-bounded KV cache with tiered storage and eviction.

    Architecture:
    - Recent tokens: GPU (fast access for attention)
    - Older tokens: CPU RAM (spilled when GPU is full)
    - Eviction: oldest or least-attended tokens dropped entirely
    """

    def __init__(
        self,
        num_layers: int,
        max_gpu_bytes: int = 512 * 1024**2,    # 512 MB default
        max_ram_bytes: int = 2 * 1024**3,       # 2 GB default
        eviction: str | EvictionPolicy = "sliding_window",
        window_size: int = 2048,                # For sliding window
        h2o_heavy_count: int = 256,             # For H2O: keep top-k heavy hitters
        h2o_recent_count: int = 512,            # For H2O: always keep recent tokens
        quantize_spilled: bool = True,          # INT8 quantize RAM-spilled KV
        device=None,
    ):
        self.num_layers = num_layers
        self.max_gpu_bytes = max_gpu_bytes
        self.max_ram_bytes = max_ram_bytes
        self.window_size = window_size
        self.h2o_heavy_count = h2o_heavy_count
        self.h2o_recent_count = h2o_recent_count
        self.quantize_spilled = quantize_spilled
        self.device = device
        self.stats = KVCacheStats()

        if isinstance(eviction, str):
            self.eviction = EvictionPolicy(eviction)
        else:
            self.eviction = eviction

        self._layers: dict[int, LayerCache] = {}
        self._gpu_bytes_used: int = 0
        self._ram_bytes_used: int = 0

    def update(
        self,
        layer_idx: int,
        key,
        value,
        attention_weights=None,
    ):
        """
        Append new KV entries for a layer.

        Args:
            layer_idx: Transformer layer index.
            key: (batch, num_kv_heads, new_seq_len, head_dim)
            value: (batch, num_kv_heads, new_seq_len, head_dim)
            attention_weights: Optional (batch, num_heads, q_len, kv_len) for H2O.
        """
        import torch

        if layer_idx not in self._layers:
            self._layers[layer_idx] = LayerCache()

        lc = self._layers[layer_idx]

        # Concatenate with existing cache.
        if lc.key is not None:
            # Bring RAM entries back to GPU if needed for concatenation.
            full_key = self._get_full_key(lc, key.device)
            full_value = self._get_full_value(lc, value.device)
            if full_key is not None:
                key = torch.cat([full_key, key], dim=2)
                value = torch.cat([full_value, value], dim=2)

        # Update attention score tracking for H2O.
        if attention_weights is not None and self.eviction == EvictionPolicy.H2O:
            # Accumulate attention scores per KV position.
            # attention_weights: (batch, heads, q_len, kv_len)
            # Sum across batch, heads, queries to get per-KV-position importance.
            scores = attention_weights.sum(dim=(0, 1, 2))  # (kv_len,)
            if lc.attention_scores is not None:
                # Extend existing scores.
                old_len = lc.attention_scores.shape[0]
                new_len = scores.shape[0]
                if new_len > old_len:
                    extended = torch.zeros(new_len, device=scores.device)
                    extended[:old_len] = lc.attention_scores.to(scores.device)
                    extended[:new_len] += scores
                    lc.attention_scores = extended
                else:
                    lc.attention_scores = lc.attention_scores.to(scores.device)
                    lc.attention_scores[:new_len] += scores
            else:
                lc.attention_scores = scores

        # Apply eviction if needed.
        key, value = self._maybe_evict(lc, key, value)

        # Track memory.
        old_bytes = self._tensor_bytes(lc.key) + self._tensor_bytes(lc.value)
        new_bytes = self._tensor_bytes(key) + self._tensor_bytes(value)
        self._gpu_bytes_used = self._gpu_bytes_used - old_bytes + new_bytes

        # Store.
        lc.key = key
        lc.value = value
        lc.gpu_seq_len = key.shape[2]

        # Check if we need to spill to RAM.
        if self._gpu_bytes_used > self.max_gpu_bytes:
            self._spill_to_ram()

        # Update stats.
        self._update_stats()

    def get(self, layer_idx: int, device=None):
        """
        Get cached KV for a layer.

        Returns (key, value) tensors on the specified device,
        or (None, None) if no cache exists.
        """
        lc = self._layers.get(layer_idx)
        if lc is None or lc.key is None:
            return None, None

        key = self._get_full_key(lc, device)
        value = self._get_full_value(lc, device)
        return key, value

    def get_seq_len(self, layer_idx: int) -> int:
        """Get current sequence length for a layer's cache."""
        lc = self._layers.get(layer_idx)
        if lc is None or lc.key is None:
            return 0
        total = lc.key.shape[2]
        if lc.ram_key is not None:
            total += lc.ram_key.shape[2]
        return total

    def clear(self):
        """Clear all cached KV entries."""
        self._layers.clear()
        self._gpu_bytes_used = 0
        self._ram_bytes_used = 0
        self.stats = KVCacheStats()

    def clear_layer(self, layer_idx: int):
        """Clear cache for a specific layer."""
        lc = self._layers.pop(layer_idx, None)
        if lc:
            self._gpu_bytes_used -= self._tensor_bytes(lc.key) + self._tensor_bytes(lc.value)
            self._ram_bytes_used -= self._tensor_bytes(lc.ram_key) + self._tensor_bytes(lc.ram_value)

    # ── Eviction ──

    def _maybe_evict(self, lc: LayerCache, key, value):
        """Apply eviction policy if sequence is too long."""
        import torch

        seq_len = key.shape[2]

        if self.eviction == EvictionPolicy.NONE:
            return key, value

        if self.eviction == EvictionPolicy.SLIDING_WINDOW:
            if seq_len > self.window_size:
                # Keep only the last window_size tokens.
                trim = seq_len - self.window_size
                key = key[:, :, trim:, :]
                value = value[:, :, trim:, :]
                self.stats.evictions += trim
                if lc.attention_scores is not None:
                    lc.attention_scores = lc.attention_scores[trim:]

        elif self.eviction == EvictionPolicy.H2O:
            total_keep = self.h2o_heavy_count + self.h2o_recent_count
            if seq_len > total_keep and lc.attention_scores is not None:
                scores = lc.attention_scores.to(key.device)
                # Always keep the most recent tokens.
                recent_start = seq_len - self.h2o_recent_count
                recent_indices = torch.arange(recent_start, seq_len, device=key.device)

                # From older tokens, keep the heavy hitters (highest attention).
                older_scores = scores[:recent_start]
                if older_scores.numel() > self.h2o_heavy_count:
                    _, heavy_indices = older_scores.topk(
                        min(self.h2o_heavy_count, older_scores.numel())
                    )
                else:
                    heavy_indices = torch.arange(older_scores.numel(), device=key.device)

                # Combine and sort indices.
                keep_indices = torch.cat([heavy_indices, recent_indices])
                keep_indices = keep_indices.sort().values

                evicted = seq_len - keep_indices.numel()
                if evicted > 0:
                    key = key[:, :, keep_indices, :]
                    value = value[:, :, keep_indices, :]
                    lc.attention_scores = scores[keep_indices]
                    self.stats.evictions += evicted

        return key, value

    # ── GPU -> RAM Spillover ──

    def _spill_to_ram(self):
        """
        Spill oldest GPU KV entries to CPU RAM.

        Strategy: for each layer, move the oldest half of cached tokens
        from GPU to CPU, until GPU usage is within budget.
        """
        import torch

        if self._gpu_bytes_used <= self.max_gpu_bytes:
            return

        # Sort layers by GPU cache size (largest first — spill biggest first).
        layers_by_size = sorted(
            self._layers.items(),
            key=lambda x: self._tensor_bytes(x[1].key),
            reverse=True,
        )

        for layer_idx, lc in layers_by_size:
            if self._gpu_bytes_used <= self.max_gpu_bytes:
                break
            if lc.key is None:
                continue

            seq_len = lc.key.shape[2]
            if seq_len <= 4:  # Don't spill tiny caches.
                continue

            # Move first half to RAM.
            split_at = seq_len // 2
            ram_k = lc.key[:, :, :split_at, :].to("cpu")
            ram_v = lc.value[:, :, :split_at, :].to("cpu")

            # Optionally quantize spilled KV to int8.
            if self.quantize_spilled:
                ram_k, ram_v = self._quantize_kv(ram_k, ram_v)

            spilled_bytes = self._tensor_bytes(lc.key[:, :, :split_at, :]) + \
                           self._tensor_bytes(lc.value[:, :, :split_at, :])

            # Keep recent half on GPU.
            lc.key = lc.key[:, :, split_at:, :].contiguous()
            lc.value = lc.value[:, :, split_at:, :].contiguous()
            lc.gpu_seq_len = lc.key.shape[2]

            # Append to existing RAM cache.
            if lc.ram_key is not None:
                lc.ram_key = torch.cat([lc.ram_key, ram_k], dim=2)
                lc.ram_value = torch.cat([lc.ram_value, ram_v], dim=2)
            else:
                lc.ram_key = ram_k
                lc.ram_value = ram_v

            self._gpu_bytes_used -= spilled_bytes
            self._ram_bytes_used += self._tensor_bytes(ram_k) + self._tensor_bytes(ram_v)
            self.stats.spillovers += 1

            if lc.attention_scores is not None:
                lc.attention_scores = lc.attention_scores[split_at:]

        # If still over budget after spilling, evict from RAM too.
        if self._ram_bytes_used > self.max_ram_bytes:
            self._evict_from_ram()

    def _evict_from_ram(self):
        """Evict oldest RAM entries when RAM is over budget."""
        for lc in self._layers.values():
            if self._ram_bytes_used <= self.max_ram_bytes:
                break
            if lc.ram_key is None:
                continue

            ram_seq = lc.ram_key.shape[2]
            if ram_seq <= 1:
                continue

            # Drop oldest half.
            evict_count = ram_seq // 2
            old_bytes = self._tensor_bytes(lc.ram_key) + self._tensor_bytes(lc.ram_value)

            lc.ram_key = lc.ram_key[:, :, evict_count:, :].contiguous()
            lc.ram_value = lc.ram_value[:, :, evict_count:, :].contiguous()

            new_bytes = self._tensor_bytes(lc.ram_key) + self._tensor_bytes(lc.ram_value)
            self._ram_bytes_used -= (old_bytes - new_bytes)
            self.stats.evictions += evict_count

    # ── KV Quantization ──

    def _quantize_kv(self, key, value):
        """Quantize KV tensors to int8 for RAM storage."""
        import torch

        # Simple absmax int8 quantization.
        k_scale = key.abs().amax().clamp(min=1e-8) / 127.0
        v_scale = value.abs().amax().clamp(min=1e-8) / 127.0

        k_q = (key / k_scale).round().clamp(-128, 127).to(torch.int8)
        v_q = (value / v_scale).round().clamp(-128, 127).to(torch.int8)

        # Store scales as attributes (a bit hacky but avoids extra data structures).
        k_q._nve_scale = k_scale
        v_q._nve_scale = v_scale
        k_q._nve_quantized = True
        v_q._nve_quantized = True

        self.stats.quantized_entries += key.shape[2]
        return k_q, v_q

    def _dequantize_kv(self, key, value, device):
        """Dequantize int8 KV tensors back to float."""
        if hasattr(key, "_nve_quantized") and key._nve_quantized:
            key = key.float() * key._nve_scale
            value = value.float() * value._nve_scale
        if device is not None:
            key = _safe_to_device(key, device)
            value = _safe_to_device(value, device)
        return key, value

    # ── Retrieval Helpers ──

    def _get_full_key(self, lc: LayerCache, device=None):
        """Get full key tensor (RAM + GPU portions concatenated)."""
        import torch

        if lc.key is None:
            return None

        target = device or lc.key.device

        if lc.ram_key is not None:
            ram_k = lc.ram_key
            if hasattr(ram_k, "_nve_quantized") and ram_k._nve_quantized:
                ram_k = ram_k.float() * ram_k._nve_scale
            ram_k = _safe_to_device(ram_k, target, dtype=lc.key.dtype)
            gpu_k = lc.key if _devices_match(lc.key.device, target) else _safe_to_device(lc.key, target)
            return torch.cat([ram_k, gpu_k], dim=2)

        if _devices_match(lc.key.device, target):
            return lc.key
        return _safe_to_device(lc.key, target)

    def _get_full_value(self, lc: LayerCache, device=None):
        """Get full value tensor (RAM + GPU portions concatenated)."""
        import torch

        if lc.value is None:
            return None

        target = device or lc.value.device

        if lc.ram_value is not None:
            ram_v = lc.ram_value
            if hasattr(ram_v, "_nve_quantized") and ram_v._nve_quantized:
                ram_v = ram_v.float() * ram_v._nve_scale
            ram_v = _safe_to_device(ram_v, target, dtype=lc.value.dtype)
            gpu_v = lc.value if _devices_match(lc.value.device, target) else _safe_to_device(lc.value, target)
            return torch.cat([ram_v, gpu_v], dim=2)

        if _devices_match(lc.value.device, target):
            return lc.value
        return _safe_to_device(lc.value, target)

    # ── Utilities ──

    def _tensor_bytes(self, t) -> int:
        if t is None:
            return 0
        return t.nelement() * t.element_size()

    def _update_stats(self):
        gpu_entries = 0
        ram_entries = 0
        gpu_bytes = 0
        ram_bytes = 0

        for lc in self._layers.values():
            if lc.key is not None:
                gpu_entries += lc.key.shape[2]
                gpu_bytes += self._tensor_bytes(lc.key) + self._tensor_bytes(lc.value)
            if lc.ram_key is not None:
                ram_entries += lc.ram_key.shape[2]
                ram_bytes += self._tensor_bytes(lc.ram_key) + self._tensor_bytes(lc.ram_value)

        self.stats.total_entries = gpu_entries + ram_entries
        self.stats.gpu_entries = gpu_entries
        self.stats.ram_entries = ram_entries
        self.stats.gpu_bytes = gpu_bytes
        self.stats.ram_bytes = ram_bytes

    @property
    def total_bytes(self) -> int:
        return self.stats.gpu_bytes + self.stats.ram_bytes

    def memory_summary(self) -> str:
        s = self.stats
        return (
            f"KV Cache: {s.total_entries} tokens | "
            f"GPU: {s.gpu_bytes / 1024**2:.1f} MB ({s.gpu_entries} tokens) | "
            f"RAM: {s.ram_bytes / 1024**2:.1f} MB ({s.ram_entries} tokens) | "
            f"Evictions: {s.evictions} | Spillovers: {s.spillovers}"
        )
