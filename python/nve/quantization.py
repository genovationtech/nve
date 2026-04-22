"""
NVE Quantization — per-tier mixed-precision weight compression.

Key insight: NVE already knows weight importance from profiling.
High-importance weights (GPU tier) keep full precision.
Medium-importance (RAM tier) get int8 quantization.
Low-importance (SSD tier) get int4 quantization.

This 2-4x reduces memory and I/O for non-critical weights while
preserving quality where it matters most.

Supports:
- Symmetric int8 quantization (per-channel absmax)
- Symmetric int4 quantization (packed into uint8, per-group)
- Dynamic dequantization on load
- Importance-aware mixed precision assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("nve.quantization")


def _safe_to_device(tensor, device):
    """Move tensor to device with OOM recovery — falls back to CPU on CUDA OOM."""
    import torch

    target = torch.device(device) if isinstance(device, str) else device
    if target.type == "cpu":
        return tensor.to(device=target)
    try:
        return tensor.to(device=target)
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"Dequant OOM moving to {target}, falling back to CPU")
        torch.cuda.empty_cache()
        try:
            return tensor.to(device=target)
        except torch.cuda.OutOfMemoryError:
            return tensor.to(device="cpu")

import numpy as np

logger = logging.getLogger("nve.quantization")


class QuantLevel(Enum):
    """Quantization precision levels."""
    NONE = "none"       # fp16/bf16 — no quantization
    INT8 = "int8"       # 8-bit symmetric
    INT4 = "int4"       # 4-bit symmetric (packed into uint8)


@dataclass
class QuantizedTensor:
    """
    A quantized weight tensor with metadata for dequantization.

    For INT8: data is int8, scale is fp32 per-channel.
    For INT4: data is uint8 (two int4 values packed), scale is fp32 per-group.
    """
    data: "np.ndarray"         # Quantized data
    scale: "np.ndarray"        # Dequantization scales
    zero_point: Optional["np.ndarray"] = None  # For asymmetric (unused for now)
    quant_level: QuantLevel = QuantLevel.NONE
    original_shape: tuple = ()
    original_dtype: str = "float16"
    group_size: int = 128      # For INT4 group quantization

    @property
    def compressed_bytes(self) -> int:
        nbytes = self.data.nbytes + self.scale.nbytes
        if self.zero_point is not None:
            nbytes += self.zero_point.nbytes
        return nbytes

    @property
    def original_bytes(self) -> int:
        numel = 1
        for s in self.original_shape:
            numel *= s
        dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}
        return numel * dtype_bytes.get(self.original_dtype, 2)

    @property
    def compression_ratio(self) -> float:
        orig = self.original_bytes
        if orig == 0:
            return 1.0
        return orig / self.compressed_bytes


def quantize_int8(tensor, per_channel: bool = True) -> QuantizedTensor:
    """
    Symmetric INT8 quantization.

    For a tensor W, compute:
        scale = max(|W|) / 127  (per-channel if per_channel=True)
        W_q = round(W / scale)

    Dequantization: W_approx = W_q * scale

    ~2x compression from fp16. Typical quality loss: <0.1% perplexity increase.
    """
    import torch

    # Work in float32 for precision.
    w = tensor.float()
    original_shape = tuple(tensor.shape)
    original_dtype = _torch_dtype_str(tensor.dtype)

    if per_channel and w.ndim >= 2:
        # Per-channel: scale per output channel (dim 0).
        absmax = w.abs().amax(dim=list(range(1, w.ndim)), keepdim=True)
        absmax = absmax.clamp(min=1e-8)
        scale = absmax / 127.0
        w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
        scale_np = scale.squeeze().cpu().numpy().astype(np.float32)
    else:
        # Per-tensor.
        absmax = w.abs().max().clamp(min=1e-8)
        scale = absmax / 127.0
        w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
        scale_np = np.array([scale.item()], dtype=np.float32)

    return QuantizedTensor(
        data=w_q.cpu().numpy(),
        scale=scale_np,
        quant_level=QuantLevel.INT8,
        original_shape=original_shape,
        original_dtype=original_dtype,
    )


def dequantize_int8(qtensor: QuantizedTensor, device="cpu", dtype=None):
    """Dequantize INT8 tensor back to floating point."""
    import torch

    data = torch.from_numpy(qtensor.data).float()
    scale = torch.from_numpy(qtensor.scale).float()

    if scale.ndim == 1 and data.ndim >= 2:
        # Per-channel: reshape scale for broadcasting.
        shape = [1] * data.ndim
        shape[0] = -1
        scale = scale.reshape(shape)

    result = data * scale

    target_dtype = dtype or _str_to_torch_dtype(qtensor.original_dtype)
    result = result.to(target_dtype)

    if device != "cpu":
        result = _safe_to_device(result, device)

    return result


def quantize_int4(tensor, group_size: int = 128) -> QuantizedTensor:
    """
    Symmetric INT4 quantization with group-wise scaling.

    Each group of `group_size` elements shares one scale factor.
    Two int4 values are packed into one uint8 byte.

    ~4x compression from fp16. Quality loss is higher but acceptable
    for low-importance (SSD-tier) weights.

    Packing: byte = (high_nibble << 4) | (low_nibble & 0xF)
    Values are in [-8, 7] range (signed 4-bit).
    """
    import torch

    w = tensor.float().contiguous()
    original_shape = tuple(tensor.shape)
    original_dtype = _torch_dtype_str(tensor.dtype)

    # Flatten for group processing.
    flat = w.reshape(-1)
    numel = flat.numel()

    # Pad to group_size multiple.
    pad_len = (group_size - numel % group_size) % group_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.float32)])

    # Reshape into groups.
    num_groups = flat.numel() // group_size
    groups = flat.reshape(num_groups, group_size)

    # Per-group absmax scaling.
    absmax = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0  # int4 range: [-8, 7]

    # Quantize.
    q = (groups / scale).round().clamp(-8, 7).to(torch.int8)

    # Pack two int4 values into one uint8.
    q_flat = q.reshape(-1)
    # Trim any padding for packing.
    pack_len = numel + (numel % 2)  # Ensure even for packing.
    if q_flat.numel() > pack_len:
        q_flat = q_flat[:pack_len]
    elif q_flat.numel() < pack_len:
        q_flat = torch.cat([q_flat, torch.zeros(pack_len - q_flat.numel(), dtype=torch.int8)])

    # Pack pairs: even indices = low nibble, odd indices = high nibble.
    low = q_flat[0::2].to(torch.uint8) & 0x0F
    high = (q_flat[1::2].to(torch.uint8) & 0x0F) << 4
    packed = (high | low).to(torch.uint8)

    # Compute scales per group, accounting for potential padding.
    num_groups_actual = (numel + group_size - 1) // group_size
    scale_squeezed = scale[:num_groups_actual].squeeze().cpu()
    # Ensure at least 1D (1 group produces a scalar after squeeze).
    if scale_squeezed.ndim == 0:
        scale_squeezed = scale_squeezed.unsqueeze(0)
    scale_np = scale_squeezed.numpy().astype(np.float32)

    return QuantizedTensor(
        data=packed.cpu().numpy(),
        scale=scale_np,
        quant_level=QuantLevel.INT4,
        original_shape=original_shape,
        original_dtype=original_dtype,
        group_size=group_size,
    )


def dequantize_int4(qtensor: QuantizedTensor, device="cpu", dtype=None):
    """Dequantize INT4 packed tensor back to floating point."""
    import torch

    packed = torch.from_numpy(qtensor.data).to(torch.uint8)
    scale = torch.from_numpy(qtensor.scale).float()
    group_size = qtensor.group_size

    # Unpack: extract low and high nibbles.
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)

    # Sign-extend 4-bit to 8-bit.
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)

    # Interleave back.
    unpacked = torch.zeros(packed.numel() * 2, dtype=torch.float32)
    unpacked[0::2] = low.float()
    unpacked[1::2] = high.float()

    # Compute original numel.
    numel = 1
    for s in qtensor.original_shape:
        numel *= s

    # Trim to original size (remove padding).
    unpacked = unpacked[:numel]

    # Apply per-group scales.
    num_groups = (numel + group_size - 1) // group_size
    # Pad unpacked to group boundary for reshape.
    pad_len = num_groups * group_size - numel
    if pad_len > 0:
        unpacked = torch.cat([unpacked, torch.zeros(pad_len)])

    groups = unpacked.reshape(num_groups, group_size)
    # Ensure scale is at least 1D (scalar when only 1 group).
    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    scale_expanded = scale[:num_groups].unsqueeze(1)
    result = (groups * scale_expanded).reshape(-1)[:numel]

    # Reshape to original.
    result = result.reshape(qtensor.original_shape)

    target_dtype = dtype or _str_to_torch_dtype(qtensor.original_dtype)
    result = result.to(target_dtype)

    if device != "cpu":
        result = _safe_to_device(result, device)

    return result


def quantize(tensor, level: QuantLevel, **kwargs) -> QuantizedTensor:
    """Quantize a tensor at the specified level."""
    if level == QuantLevel.NONE:
        return QuantizedTensor(
            data=tensor.cpu().numpy(),
            scale=np.array([1.0], dtype=np.float32),
            quant_level=QuantLevel.NONE,
            original_shape=tuple(tensor.shape),
            original_dtype=_torch_dtype_str(tensor.dtype),
        )
    elif level == QuantLevel.INT8:
        return quantize_int8(tensor, **kwargs)
    elif level == QuantLevel.INT4:
        return quantize_int4(tensor, **kwargs)
    else:
        raise ValueError(f"Unknown quantization level: {level}")


def dequantize(qtensor: QuantizedTensor, device="cpu", dtype=None):
    """Dequantize a tensor from any level."""
    import torch

    if qtensor.quant_level == QuantLevel.NONE:
        result = torch.from_numpy(qtensor.data)
        if dtype:
            result = result.to(dtype)
        if device != "cpu":
            result = _safe_to_device(result, device)
        return result
    elif qtensor.quant_level == QuantLevel.INT8:
        return dequantize_int8(qtensor, device=device, dtype=dtype)
    elif qtensor.quant_level == QuantLevel.INT4:
        return dequantize_int4(qtensor, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown quantization level: {qtensor.quant_level}")


# ── Tier-Aware Quantization Policy ──

@dataclass
class QuantPolicy:
    """
    Maps NVE tiers to quantization levels.

    Default policy:
    - GPU tier: no quantization (full fp16 for quality)
    - RAM tier: int8 (2x compression, minimal quality loss)
    - SSD tier: int4 (4x compression, acceptable for cold weights)
    """
    gpu_level: QuantLevel = QuantLevel.NONE
    ram_level: QuantLevel = QuantLevel.INT8
    ssd_level: QuantLevel = QuantLevel.INT4
    int4_group_size: int = 128
    # Weights above this importance threshold keep full precision regardless of tier.
    importance_threshold: float = 0.0

    @classmethod
    def no_quantization(cls) -> "QuantPolicy":
        """No quantization on any tier (for debugging/comparison)."""
        return cls(
            gpu_level=QuantLevel.NONE,
            ram_level=QuantLevel.NONE,
            ssd_level=QuantLevel.NONE,
        )

    @classmethod
    def aggressive(cls) -> "QuantPolicy":
        """Aggressive quantization for extremely small devices."""
        return cls(
            gpu_level=QuantLevel.INT8,
            ram_level=QuantLevel.INT4,
            ssd_level=QuantLevel.INT4,
            int4_group_size=64,  # Smaller groups = better quality.
        )

    @classmethod
    def balanced(cls) -> "QuantPolicy":
        """Default balanced policy."""
        return cls()

    def level_for_tier(self, tier: str) -> QuantLevel:
        """Get quantization level for a tier name."""
        return {
            "gpu": self.gpu_level,
            "ram": self.ram_level,
            "ssd": self.ssd_level,
        }.get(tier, QuantLevel.NONE)

    def should_quantize(self, tier: str, importance: float = 0.0) -> QuantLevel:
        """
        Determine quantization level considering both tier and importance.

        High-importance weights can override tier-based quantization.
        """
        if self.importance_threshold > 0 and importance >= self.importance_threshold:
            return QuantLevel.NONE
        return self.level_for_tier(tier)


def estimate_compressed_size(
    original_bytes: int,
    tier: str,
    policy: QuantPolicy,
) -> int:
    """Estimate compressed size of a weight in a given tier."""
    level = policy.level_for_tier(tier)
    if level == QuantLevel.NONE:
        return original_bytes
    elif level == QuantLevel.INT8:
        # int8 data + fp32 scales (one per output channel).
        # Approximate: data is half, scales are negligible.
        return original_bytes // 2
    elif level == QuantLevel.INT4:
        # Packed int4: quarter the data + group scales.
        return original_bytes // 4
    return original_bytes


# ── Helpers ──

def _torch_dtype_str(dtype) -> str:
    """Convert torch dtype to string."""
    import torch
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return mapping.get(dtype, "float16")


def _str_to_torch_dtype(s: str):
    """Convert string to torch dtype."""
    import torch
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping.get(s, torch.float16)
