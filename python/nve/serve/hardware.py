"""
NVE hardware detection and management for the serve layer.

Provides:
- A lazy singleton `DeviceManager` shared across all serve components.
- `GET /v1/hardware` handler — returns JSON hardware snapshot.
- `autocast_context()` — returns the right torch.autocast context for a device.
- `compile_model()` — applies `torch.compile()` where safe.
- `move_model_to_device()` — moves a nn.Module to the best device with
  mixed-precision dtype selection and OOM recovery.

Device priority (runtime auto-selection):
  NVIDIA CUDA → AMD ROCm → Apple MPS → Intel XPU → CPU
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Optional

from aiohttp import web

logger = logging.getLogger("nve.serve.hardware")


# ── Singleton DeviceManager ───────────────────────────────────────────────────

_dm = None
_dm_lock = __import__("threading").Lock()


def get_device_manager():
    """Return (or lazily create) the module-level DeviceManager."""
    global _dm
    if _dm is None:
        with _dm_lock:
            if _dm is None:
                from nve.device import DeviceManager
                preferred = os.getenv("NVE_DEVICE")
                reserve = float(os.getenv("NVE_GPU_RESERVE", "0.85"))
                _dm = DeviceManager(
                    preferred_device=preferred,
                    gpu_reserve_fraction=reserve,
                    enable_memory_monitor=True,
                )
                logger.info(repr(_dm))
                logger.info(_dm.summary())
    return _dm


# ── aiohttp handler ───────────────────────────────────────────────────────────

async def hardware_info(request: web.Request) -> web.Response:
    """GET /v1/hardware — JSON snapshot of all detected hardware."""
    dm = get_device_manager()
    data = dm.as_dict()
    # Add best-device recommendation.
    data["recommended_device"] = dm.best_device
    data["cpu_features"] = [f.value for f in dm._cpu_features]
    return web.json_response(data)


# ── torch.autocast helper ─────────────────────────────────────────────────────

@contextlib.contextmanager
def autocast_context(device_str: str, enabled: bool = True):
    """
    Return a torch.autocast context for mixed-precision inference.

    - CUDA / ROCm: autocast with bfloat16 (if supported) or float16.
    - MPS: autocast with float16 (bf16 not supported on MPS as of PyTorch 2.x).
    - XPU: autocast with bfloat16.
    - CPU: autocast disabled (use fp32; or fp32+bf16 if explicitly requested).
    """
    try:
        import torch
    except ImportError:
        yield
        return

    if not enabled:
        yield
        return

    try:
        dev = torch.device(device_str)
    except RuntimeError:
        yield
        return

    dm = get_device_manager()
    info = dm._device_info_for(device_str)

    if dev.type == "cuda":
        dtype = torch.bfloat16 if (info and info.supports_bf16) else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            yield
    elif dev.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.float16, enabled=True):
            yield
    elif dev.type == "xpu":
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=True):
            yield
    else:
        # CPU: autocast with bf16 if available, else skip.
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                yield
        except Exception:
            yield


# ── Model placement ───────────────────────────────────────────────────────────

def move_model_to_device(
    model,
    device_str: Optional[str] = None,
    use_mixed_precision: bool = True,
    compile_model: bool = False,
    compile_backend: str = "inductor",
):
    """
    Move a PyTorch nn.Module to the best available device.

    Steps:
      1. Select device (auto if device_str is None).
      2. Cast to best dtype (bf16/fp16/fp32) for the device.
      3. Move to device with OOM recovery.
      4. Optionally apply `torch.compile()` for inference speed.

    Returns (model, device_str, dtype).
    """
    try:
        import torch
    except ImportError:
        return model, "cpu", None

    dm = get_device_manager()

    # Estimate model size for device selection.
    try:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    except Exception:
        param_bytes = 0

    if device_str is None:
        device_str = dm.select_device(required_bytes=param_bytes)

    dtype = dm.best_dtype(device_str) if use_mixed_precision else torch.float32
    dev = torch.device(device_str)

    logger.info(
        f"Moving model ({param_bytes / 1024**2:.0f} MB) to {device_str} "
        f"with dtype={dtype}"
    )

    # Cast dtype first (on CPU, cheaper than on GPU).
    try:
        model = model.to(dtype=dtype)
    except Exception as e:
        logger.warning(f"dtype cast failed ({e}), keeping fp32")
        dtype = torch.float32

    # Move to device with OOM recovery.
    try:
        if dev.type == "cuda":
            model = model.cuda(dev.index or 0)
        elif dev.type == "mps":
            model = model.to(dev)
        elif dev.type == "xpu":
            model = model.to(dev)
        else:
            model = model.cpu()
    except (RuntimeError, Exception) as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                f"OOM moving model to {device_str} — clearing cache and falling back to CPU"
            )
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            device_str = "cpu"
            dtype = torch.float32
            model = model.to(device="cpu", dtype=torch.float32)
        else:
            raise

    # torch.compile() — skip on MPS (not fully supported) and Python < 3.11.
    if compile_model and device_str != "cpu" and dev.type != "mps":
        try:
            import sys
            if sys.version_info >= (3, 11):
                model = torch.compile(model, backend=compile_backend, mode="reduce-overhead")
                logger.info(f"torch.compile() applied with backend='{compile_backend}'")
            else:
                logger.debug("torch.compile() skipped (Python < 3.11)")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e} — running eager mode")

    return model, device_str, dtype


# ── Convenience: device string normalisation ──────────────────────────────────

def normalise_device(device_str: Optional[str]) -> str:
    """
    Normalise a user-supplied device string.

    Accepts "auto", "cuda", "cuda:0", "rocm", "rocm:0", "hip:0",
    "mps", "xpu", "xpu:0", "cpu".  Unknown strings fall back to auto-detect.
    """
    if device_str is None or device_str.strip() == "" or device_str == "auto":
        return get_device_manager().best_device
    d = device_str.strip().lower()
    # Normalise ROCm aliases to "cuda:N" (same torch API).
    if d.startswith("rocm"):
        d = d.replace("rocm", "cuda", 1)
    if d.startswith("hip"):
        d = d.replace("hip", "cuda", 1)
    return d
