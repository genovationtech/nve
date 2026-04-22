"""
NVE Quantization Backend — detect, configure, and apply model quantization.

Supported backends
──────────────────
  BitsAndBytes (bnb)   — NVIDIA CUDA / AMD ROCm
                         int8 (LLM.int8) and int4 NF4/FP4 (QLoRA-style)
                         Requires: `pip install bitsandbytes`

  Optimum Quanto       — Cross-platform: CUDA, ROCm, MPS, XPU, CPU
                         int8 and int4 via calibrated weight quantization
                         Requires: `pip install optimum-quanto`

  Standard dtypes      — fp16 / bf16 / fp32, always available

Priority for "auto" mode:
  CUDA/ROCm: bnb_int4 → bnb_int8 → none
  MPS/XPU:   quanto_int8 → none
  CPU:        none

All backends degrade gracefully when the package is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("nve.serve.quantization")


# ── Availability probes (cached at import time) ───────────────────────────────

def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def _quanto_available() -> bool:
    try:
        import optimum.quanto  # noqa: F401
        return True
    except ImportError:
        return False


_BNB_OK = _bnb_available()
_QUANTO_OK = _quanto_available()


# ── Enums + config ────────────────────────────────────────────────────────────

class QuantizationBackend(Enum):
    NONE        = "none"
    BNB_INT8    = "bnb_int8"
    BNB_INT4    = "bnb_int4_nf4"    # NF4 double-quantized (QLoRA default)
    BNB_INT4_FP = "bnb_int4_fp4"    # FP4 (faster on Hopper)
    QUANTO_INT8 = "quanto_int8"
    QUANTO_INT4 = "quanto_int4"


@dataclass
class QuantConfig:
    backend: QuantizationBackend = QuantizationBackend.NONE
    double_quant: bool = True           # bnb: double quantization (saves ~0.4 bpw)
    quant_type: str = "nf4"            # bnb: "nf4" | "fp4"
    compute_dtype: Optional[object] = None  # torch.dtype for bnb dequant (bf16/fp16)
    # quanto calibration: number of samples for scale computation (0 = no calibration)
    quanto_calibration_samples: int = 0

    def is_quantized(self) -> bool:
        return self.backend != QuantizationBackend.NONE

    def description(self) -> str:
        b = self.backend.value
        if self.backend == QuantizationBackend.NONE:
            return "fp16/bf16 (no quantization)"
        if self.backend == QuantizationBackend.BNB_INT4:
            return f"bnb int4 {self.quant_type} {'DQ' if self.double_quant else ''}"
        return b


# ── Backend detection ─────────────────────────────────────────────────────────

def detect_best_quant(device_str: str, requested: str = "none") -> QuantConfig:
    """
    Return the best available QuantConfig for a device and requested level.

    Parameters
    ----------
    device_str : str
        torch-style device string ("cuda:0", "mps", "xpu:0", "cpu", …)
    requested : str
        "none"    — no quantization (always honoured)
        "int8"    — 8-bit quantization
        "int4"    — 4-bit NF4 quantization
        "int4_nf4"— explicit NF4
        "int4_fp4"— explicit FP4 (Hopper only)
        "auto"    — best available given device

    Returns
    -------
    QuantConfig
        Falls back to NONE if the requested backend is not installed.
    """
    try:
        import torch
        dev = torch.device(device_str)
        device_type = dev.type
    except Exception:
        device_type = "cpu"

    requested = requested.lower().strip()

    if requested == "none":
        return QuantConfig(backend=QuantizationBackend.NONE)

    # ── CUDA / ROCm ──────────────────────────────────────────────────────────
    if device_type == "cuda":
        if requested in ("auto", "int4", "int4_nf4"):
            if _BNB_OK:
                try:
                    import torch
                    compute_dtype = torch.bfloat16 if _has_bf16(device_str) else torch.float16
                    logger.info("Quantization: bnb int4 NF4 (double-quant)")
                    return QuantConfig(
                        backend=QuantizationBackend.BNB_INT4,
                        double_quant=True,
                        quant_type="nf4",
                        compute_dtype=compute_dtype,
                    )
                except Exception as e:
                    logger.warning(f"bnb int4 setup failed: {e}")

        if requested in ("auto", "int8"):
            if _BNB_OK:
                logger.info("Quantization: bnb int8 (LLM.int8)")
                return QuantConfig(backend=QuantizationBackend.BNB_INT8)

        if requested == "int4_fp4":
            if _BNB_OK:
                try:
                    import torch
                    compute_dtype = torch.bfloat16 if _has_bf16(device_str) else torch.float16
                    logger.info("Quantization: bnb int4 FP4")
                    return QuantConfig(
                        backend=QuantizationBackend.BNB_INT4_FP,
                        double_quant=False,
                        quant_type="fp4",
                        compute_dtype=compute_dtype,
                    )
                except Exception as e:
                    logger.warning(f"bnb fp4 setup failed: {e}")

        # Quanto fallback for CUDA
        if requested in ("int8", "auto") and _QUANTO_OK:
            logger.info("Quantization: quanto int8 (bnb not available)")
            return QuantConfig(backend=QuantizationBackend.QUANTO_INT8)

        if requested not in ("auto",):
            logger.warning(
                f"Requested quantization '{requested}' unavailable on {device_str} "
                f"(bnb={_BNB_OK}, quanto={_QUANTO_OK}) — running without quantization"
            )
        return QuantConfig(backend=QuantizationBackend.NONE)

    # ── Apple MPS ─────────────────────────────────────────────────────────────
    if device_type == "mps":
        if requested in ("auto", "int8", "int4") and _QUANTO_OK:
            logger.info("Quantization: quanto int8 (MPS device)")
            return QuantConfig(backend=QuantizationBackend.QUANTO_INT8)
        if requested not in ("none", "auto"):
            logger.warning(
                f"BitsAndBytes is not supported on MPS. "
                f"Use 'int8' with optimum-quanto installed."
            )
        return QuantConfig(backend=QuantizationBackend.NONE)

    # ── Intel XPU ─────────────────────────────────────────────────────────────
    if device_type == "xpu":
        if requested in ("auto", "int8", "int4") and _QUANTO_OK:
            logger.info("Quantization: quanto int8 (XPU device)")
            return QuantConfig(backend=QuantizationBackend.QUANTO_INT8)
        return QuantConfig(backend=QuantizationBackend.NONE)

    # ── CPU ───────────────────────────────────────────────────────────────────
    if requested == "auto":
        return QuantConfig(backend=QuantizationBackend.NONE)
    if requested in ("int8", "int4") and _QUANTO_OK:
        logger.info("Quantization: quanto int8 (CPU device)")
        return QuantConfig(backend=QuantizationBackend.QUANTO_INT8)

    if requested not in ("none",):
        logger.warning(f"Quantization '{requested}' not supported on CPU")
    return QuantConfig(backend=QuantizationBackend.NONE)


def _has_bf16(device_str: str) -> bool:
    """Return True if the device supports bfloat16."""
    try:
        import torch
        dev = torch.device(device_str)
        if dev.type == "cuda":
            props = torch.cuda.get_device_properties(dev.index or 0)
            return props.major >= 8  # Ampere+
    except Exception:
        pass
    return False


# ── HF load kwargs ────────────────────────────────────────────────────────────

def build_hf_load_kwargs(
    qconfig: QuantConfig,
    dtype,           # torch.dtype
    device_str: str,
) -> dict:
    """
    Return extra kwargs to merge into `AutoModelForCausalLM.from_pretrained()`.

    For bnb backends: adds `quantization_config`.
    For quanto backends: returns {} (quanto is applied post-load).
    For NONE: returns {}.
    """
    if qconfig.backend == QuantizationBackend.NONE:
        return {}

    # ── BitsAndBytes ─────────────────────────────────────────────────────────
    if qconfig.backend in (QuantizationBackend.BNB_INT8,):
        try:
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                # dtype must be unset for bnb
                "torch_dtype": None,
            }
        except ImportError:
            logger.warning("transformers.BitsAndBytesConfig not available")
            return {}

    if qconfig.backend in (QuantizationBackend.BNB_INT4, QuantizationBackend.BNB_INT4_FP):
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = qconfig.compute_dtype or dtype
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=qconfig.double_quant,
                bnb_4bit_quant_type=qconfig.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            return {
                "quantization_config": bnb_config,
                "torch_dtype": None,  # bnb handles dtype internally
            }
        except ImportError:
            logger.warning("transformers.BitsAndBytesConfig not available")
            return {}

    # Quanto: applied post-load
    return {}


def apply_quanto(model, qconfig: QuantConfig) -> None:
    """
    Apply Optimum Quanto quantization in-place after model load.

    Called only when `qconfig.backend` is QUANTO_*.
    No-op if quanto is not installed.
    """
    if qconfig.backend not in (QuantizationBackend.QUANTO_INT8, QuantizationBackend.QUANTO_INT4):
        return
    if not _QUANTO_OK:
        logger.warning("optimum-quanto not installed — skipping quanto quantization")
        return

    try:
        from optimum.quanto import quantize, freeze
        import optimum.quanto as q

        weight_dtype = (
            q.qint8 if qconfig.backend == QuantizationBackend.QUANTO_INT8
            else q.qint4
        )
        logger.info(f"Applying quanto {weight_dtype} quantization…")
        quantize(model, weights=weight_dtype, activations=None)
        freeze(model)
        logger.info("Quanto quantization complete")
    except Exception as e:
        logger.error(f"Quanto quantization failed: {e} — model will run in fp16/bf16")


# ── Attention implementation detection ────────────────────────────────────────

def detect_flash_attention() -> str:
    """
    Detect the best available attention implementation.

    Returns one of: "flash_attention_2", "sdpa", "eager".
    """
    # Flash Attention 2: requires flash-attn package, CUDA only.
    try:
        import flash_attn  # noqa: F401
        # Verify version ≥ 2.
        version = getattr(flash_attn, "__version__", "0")
        major = int(version.split(".")[0])
        if major >= 2:
            logger.debug("Flash Attention 2 available")
            return "flash_attention_2"
    except (ImportError, ValueError, IndexError):
        pass

    # SDPA: built into PyTorch ≥ 2.0, works on CUDA + MPS.
    try:
        import torch
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Verify it's not a stub.
            major, minor = torch.__version__.split(".")[:2]
            if int(major) >= 2:
                logger.debug("SDPA attention available (PyTorch ≥ 2.0)")
                return "sdpa"
    except Exception:
        pass

    return "eager"


def best_attn_impl(device_str: str) -> str:
    """
    Return the best `attn_implementation` for a specific device.

    flash_attention_2 — CUDA only (flash-attn package required)
    sdpa              — CUDA + MPS (PyTorch 2.0+)
    eager             — CPU or fallback
    """
    try:
        import torch
        dev = torch.device(device_str)
        device_type = dev.type
    except Exception:
        device_type = "cpu"

    best = detect_flash_attention()

    if best == "flash_attention_2" and device_type != "cuda":
        # flash_attention_2 is CUDA-only
        best = "sdpa"

    if best == "sdpa" and device_type == "cpu":
        # SDPA on CPU is supported in PyTorch 2.0+ but often slower than eager
        # for small batch sizes; use eager for CPU.
        best = "eager"

    return best
