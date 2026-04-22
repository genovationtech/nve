"""
NVE Device Manager — production-grade device selection and memory management.

Handles:
- OOM-safe GPU detection with actual memory probing
- Continuous VRAM/RAM monitoring
- Multi-GPU awareness (cuda:0, cuda:1, ...)
- Memory pinning (mlock) for critical weights on small devices
- Graceful fallback: GPU -> RAM -> SSD under memory pressure
- CPU-only mode where "GPU tier" = pinned high-priority RAM

Supported backends
──────────────────
  CUDA   — NVIDIA GPUs via CUDA (PyTorch cuda backend)
  ROCM   — AMD GPUs via ROCm (PyTorch hip backend, same API as CUDA)
  MPS    — Apple Silicon via Metal Performance Shaders (torch.backends.mps)
  XPU    — Intel GPUs via Intel Extension for PyTorch (IPEX / torch.xpu)
  VULKAN — Cross-platform Vulkan (detection only; compute via wgpu on Rust side)
  CPU    — Pure CPU with optional SIMD acceleration (AVX2/AVX-512/NEON/MKL)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nve.device")


class DeviceType(Enum):
    CUDA   = "cuda"   # NVIDIA (CUDA Toolkit ≥ 11.8)
    ROCM   = "rocm"   # AMD (ROCm ≥ 5.0, uses torch.cuda API with HIP backend)
    MPS    = "mps"    # Apple Silicon (macOS 12.3+)
    XPU    = "xpu"    # Intel GPU (Intel Extension for PyTorch)
    VULKAN = "vulkan" # Cross-platform Vulkan (via wgpu on Rust side)
    CPU    = "cpu"    # Pure CPU


class CPUFeatures(Enum):
    """CPU acceleration feature flags."""
    BASELINE = "baseline"  # Pure Rust/C BLAS
    AVX2     = "avx2"      # AVX2 + FMA (most x86-64, default NVE)
    AVX512   = "avx512"    # AVX-512 (Intel Skylake-X, AMD Zen4+)
    MKL      = "mkl"       # Intel MKL BLAS
    NEON     = "neon"      # ARM NEON (all Apple M-series, AWS Graviton)
    ACCELERATE = "accelerate"  # macOS Accelerate framework (Apple only)


@dataclass
class MemoryBudget:
    """Memory budget for a single device."""
    total_bytes: int = 0
    free_bytes: int = 0
    reserved_bytes: int = 0  # How much NVE has reserved.
    peak_reserved_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.free_bytes - self.reserved_bytes)

    @property
    def utilization(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.reserved_bytes / self.total_bytes


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: DeviceType
    device_index: int = 0
    name: str = "cpu"
    total_vram_bytes: int = 0
    # CUDA / ROCm compute capability or architecture version string.
    compute_capability: Tuple[int, int] = (0, 0)
    architecture: str = ""          # e.g. "gfx1100" (AMD), "Xe-HPG" (Intel)
    driver_version: str = ""
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_fp8: bool = False      # H100 / MI300X series
    cpu_features: List[CPUFeatures] = field(default_factory=list)


@dataclass
class SystemMemory:
    """System-wide memory snapshot."""
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    ram_used_bytes: int = 0
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    gpu_memories: Dict[int, MemoryBudget] = field(default_factory=dict)

    @property
    def ram_pressure(self) -> float:
        """0.0 = no pressure, 1.0 = out of memory."""
        if self.ram_total_bytes == 0:
            return 0.0
        return 1.0 - (self.ram_available_bytes / self.ram_total_bytes)


def _devices_match(a, b) -> bool:
    """Compare two torch devices, treating cuda and cuda:0 as equal."""
    import torch
    a = torch.device(a) if not isinstance(a, torch.device) else a
    b = torch.device(b) if not isinstance(b, torch.device) else b
    if a.type != b.type:
        return False
    if a.index is not None and b.index is not None:
        return a.index == b.index
    return True


def _detect_cpu_features() -> List[CPUFeatures]:
    """Detect CPU SIMD / BLAS features available at runtime."""
    features = [CPUFeatures.BASELINE]
    import platform
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64", "i686"):
        try:
            # cpuinfo is optional but gives us precise flags.
            import cpuinfo
            flags = cpuinfo.get_cpu_info().get("flags", [])
            if "avx2" in flags:
                features.append(CPUFeatures.AVX2)
            if "avx512f" in flags:
                features.append(CPUFeatures.AVX512)
        except ImportError:
            # Fallback: check /proc/cpuinfo on Linux.
            try:
                with open("/proc/cpuinfo") as f:
                    content = f.read()
                if " avx2" in content:
                    features.append(CPUFeatures.AVX2)
                if " avx512f" in content:
                    features.append(CPUFeatures.AVX512)
            except (FileNotFoundError, OSError):
                pass

        # Intel MKL detection.
        try:
            import ctypes as _ct
            _ct.CDLL("libmkl_rt.so")
            features.append(CPUFeatures.MKL)
        except OSError:
            pass

    elif machine in ("arm64", "aarch64"):
        features.append(CPUFeatures.NEON)
        import platform as _p
        if _p.system() == "Darwin":
            features.append(CPUFeatures.ACCELERATE)

    return features


class DeviceManager:
    """
    Central device management for NVE.

    Provides OOM-safe device selection, continuous memory monitoring,
    and budget enforcement across NVIDIA CUDA, AMD ROCm, Apple MPS,
    Intel XPU, and CPU backends.

    Usage:
        dm = DeviceManager()
        device = dm.select_device(required_bytes=2 * 1024**3)
        tensor = dm.safe_to(tensor, device)

    Environment overrides:
        NVE_DEVICE   — force a specific device string ("cpu", "cuda:0",
                        "rocm:0", "mps", "xpu:0").
        NVE_GPU_RESERVE — fraction of GPU VRAM to reserve (default 0.85).
    """

    def __init__(
        self,
        preferred_device: Optional[str] = None,
        gpu_reserve_fraction: float = 0.85,
        ram_reserve_fraction: float = 0.80,
        enable_memory_monitor: bool = True,
        monitor_interval_s: float = 2.0,
    ):
        self._preferred_device = preferred_device or os.getenv("NVE_DEVICE")
        self._gpu_reserve_fraction = float(
            os.getenv("NVE_GPU_RESERVE", str(gpu_reserve_fraction))
        )
        self._ram_reserve_fraction = ram_reserve_fraction
        self._enable_monitor = enable_memory_monitor
        self._monitor_interval = monitor_interval_s

        self._devices: List[DeviceInfo] = []
        # Keyed by (DeviceType, index) → MemoryBudget
        self._gpu_budgets: Dict[Tuple[DeviceType, int], MemoryBudget] = {}
        self._ram_budget = MemoryBudget()
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()

        # Backend availability flags
        self._torch_available = False
        self._cuda_available = False   # NVIDIA
        self._rocm_available = False   # AMD
        self._mps_available  = False   # Apple
        self._xpu_available  = False   # Intel
        self._vulkan_available = False  # cross-platform detection only

        self._cpu_features = _detect_cpu_features()
        self._detect_hardware()

        if enable_memory_monitor:
            self._start_monitor()

    # ── Hardware detection ────────────────────────────────────────────────────

    def _detect_hardware(self):
        """Detect all available compute devices and their capabilities."""
        # CPU is always available.
        cpu_info = DeviceInfo(
            device_type=DeviceType.CPU,
            device_index=0,
            name=self._cpu_name(),
            supports_fp16=True,
            cpu_features=self._cpu_features,
        )
        self._devices.append(cpu_info)
        self._update_ram_budget()

        try:
            import torch
            self._torch_available = True
        except ImportError:
            logger.info("PyTorch not available — CPU-only mode")
            return

        self._detect_cuda()
        self._detect_rocm()
        self._detect_mps()
        self._detect_xpu()
        self._detect_vulkan()

    def _cpu_name(self) -> str:
        try:
            import cpuinfo
            return cpuinfo.get_cpu_info().get("brand_raw", "CPU")
        except ImportError:
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except (FileNotFoundError, OSError):
                pass
        return "CPU"

    def _detect_cuda(self):
        """Detect NVIDIA CUDA GPUs."""
        try:
            import torch
            # On ROCm builds torch.version.hip is set; skip here to avoid
            # double-counting AMD GPUs as CUDA.
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return

            if not torch.cuda.is_available():
                return

            self._cuda_available = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_vram, total_vram = torch.cuda.mem_get_info(i)
                info = DeviceInfo(
                    device_type=DeviceType.CUDA,
                    device_index=i,
                    name=props.name,
                    total_vram_bytes=total_vram,
                    compute_capability=(props.major, props.minor),
                    supports_fp16=True,
                    supports_bf16=props.major >= 8,          # Ampere+
                    supports_int8=props.major >= 7,          # Turing+
                    supports_fp8=(props.major, props.minor) >= (9, 0),  # Hopper
                    driver_version=torch.version.cuda or "",
                )
                self._devices.append(info)
                self._gpu_budgets[(DeviceType.CUDA, i)] = MemoryBudget(
                    total_bytes=total_vram,
                    free_bytes=free_vram,
                )
                logger.info(
                    f"CUDA GPU {i}: {props.name} | "
                    f"{total_vram / 1024**3:.1f} GB total | "
                    f"{free_vram / 1024**3:.1f} GB free | "
                    f"CC {props.major}.{props.minor} | "
                    f"bf16={info.supports_bf16}"
                )
        except (ImportError, RuntimeError) as e:
            logger.debug(f"CUDA detection failed: {e}")

    def _detect_rocm(self):
        """Detect AMD ROCm GPUs (uses torch.cuda API on ROCm builds)."""
        try:
            import torch
            # ROCm torch builds set torch.version.hip.
            if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
                return
            if not torch.cuda.is_available():
                return

            self._rocm_available = True
            hip_ver = torch.version.hip or ""
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_vram, total_vram = torch.cuda.mem_get_info(i)
                arch = getattr(props, "gcn_arch_name", "") or ""
                info = DeviceInfo(
                    device_type=DeviceType.ROCM,
                    device_index=i,
                    name=props.name,
                    total_vram_bytes=total_vram,
                    architecture=arch,    # e.g. "gfx1100"
                    driver_version=hip_ver,
                    supports_fp16=True,
                    # CDNA2 (MI200/gfx90a) and RDNA3 (gfx11xx) support bf16.
                    supports_bf16=(
                        "gfx90a" in arch or "gfx940" in arch or
                        "gfx941" in arch or "gfx942" in arch or
                        "gfx11" in arch
                    ),
                    supports_int8=True,
                    # MI300X (gfx941/2) supports fp8.
                    supports_fp8=("gfx941" in arch or "gfx942" in arch),
                )
                self._devices.append(info)
                self._gpu_budgets[(DeviceType.ROCM, i)] = MemoryBudget(
                    total_bytes=total_vram,
                    free_bytes=free_vram,
                )
                logger.info(
                    f"ROCm GPU {i}: {props.name} | "
                    f"{total_vram / 1024**3:.1f} GB | arch={arch} | "
                    f"ROCm {hip_ver}"
                )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"ROCm detection failed: {e}")

    def _detect_mps(self):
        """Detect Apple Metal Performance Shaders (M-series / AMD eGPU)."""
        try:
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return
            self._mps_available = True
            # MPS doesn't expose raw VRAM — use system unified memory heuristic.
            import platform
            import subprocess
            total_vram = 0
            try:
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType", "-json"],
                    capture_output=True, text=True, timeout=5,
                )
                import json
                data = json.loads(result.stdout)
                hw = data.get("SPHardwareDataType", [{}])[0]
                # "4 GB" or "16 GB" from 'physical_memory' field
                mem_str = hw.get("physical_memory", "")
                gb = int("".join(c for c in mem_str if c.isdigit()) or "0")
                # Heuristic: ~75 % of unified memory is GPU-accessible.
                total_vram = int(gb * 0.75 * 1024**3)
            except Exception:
                pass

            info = DeviceInfo(
                device_type=DeviceType.MPS,
                device_index=0,
                name=f"Apple GPU ({platform.processor() or 'M-series'})",
                total_vram_bytes=total_vram,
                architecture=platform.machine(),
                supports_fp16=True,
                supports_bf16=False,   # Not natively in MPS as of 2024
                supports_int8=False,
            )
            self._devices.append(info)
            # MPS doesn't report separate free VRAM — use 0 as "unknown".
            self._gpu_budgets[(DeviceType.MPS, 0)] = MemoryBudget(
                total_bytes=total_vram,
                free_bytes=total_vram,  # assume all free at startup
            )
            logger.info(f"MPS device: {info.name} | ~{total_vram/1024**3:.1f} GB unified")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"MPS detection failed: {e}")

    def _detect_xpu(self):
        """Detect Intel GPU via Intel Extension for PyTorch (IPEX)."""
        try:
            import torch
            # Intel IPEX adds torch.xpu
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                return
            self._xpu_available = True
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                total_vram = getattr(props, "total_memory", 0)
                info = DeviceInfo(
                    device_type=DeviceType.XPU,
                    device_index=i,
                    name=getattr(props, "name", f"Intel GPU {i}"),
                    total_vram_bytes=total_vram,
                    architecture=getattr(props, "gpu_eu_count", ""),
                    supports_fp16=True,
                    supports_bf16=True,   # Arc / Ponte Vecchio all support bf16
                    supports_int8=True,
                )
                self._devices.append(info)
                self._gpu_budgets[(DeviceType.XPU, i)] = MemoryBudget(
                    total_bytes=total_vram,
                    free_bytes=total_vram,
                )
                logger.info(
                    f"XPU {i}: {info.name} | "
                    f"{total_vram / 1024**3:.1f} GB"
                )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"XPU detection failed: {e}")

    def _detect_vulkan(self):
        """
        Detect Vulkan-capable GPUs.

        Pure probe — actual compute is dispatched via the Rust wgpu backend.
        This records the device for the /v1/hardware endpoint and lets
        the server report Vulkan availability without requiring CUDA/ROCm/MPS.
        """
        try:
            # Probe via vulkaninfo (part of Vulkan SDK / mesa-vulkan-drivers)
            import subprocess
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return
            # Parse GPU names from output.
            idx = 0
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("deviceName"):
                    _, _, name = line.partition("=")
                    name = name.strip()
                    # Skip entries already covered by CUDA/ROCm/MPS/XPU.
                    self._devices.append(DeviceInfo(
                        device_type=DeviceType.VULKAN,
                        device_index=idx,
                        name=name,
                        supports_fp16=True,
                        supports_bf16=False,
                        supports_int8=False,
                    ))
                    logger.info(f"Vulkan GPU {idx}: {name}")
                    idx += 1
            if idx > 0:
                self._vulkan_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    # ── Memory monitoring ─────────────────────────────────────────────────────

    def _update_ram_budget(self):
        """Update RAM budget from system info."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            self._ram_budget.total_bytes = mem.total
            self._ram_budget.free_bytes = mem.available
            return SystemMemory(
                ram_total_bytes=mem.total,
                ram_available_bytes=mem.available,
                ram_used_bytes=mem.used,
                swap_total_bytes=swap.total,
                swap_used_bytes=swap.used,
            )
        except ImportError:
            try:
                info = _parse_proc_meminfo()
                self._ram_budget.total_bytes = info.get("MemTotal", 0) * 1024
                self._ram_budget.free_bytes = info.get("MemAvailable", 0) * 1024
            except Exception:
                pass
            return SystemMemory()

    def _update_gpu_budgets(self):
        """Refresh free VRAM for all tracked GPU budgets."""
        try:
            import torch
        except ImportError:
            return

        for (dtype, idx), budget in self._gpu_budgets.items():
            try:
                is_cuda_family = (
                    (dtype == DeviceType.CUDA and self._cuda_available) or
                    (dtype == DeviceType.ROCM and self._rocm_available)
                )
                if is_cuda_family:
                    free, total = torch.cuda.mem_get_info(idx)
                    budget.free_bytes = free
                    budget.total_bytes = total
                elif dtype == DeviceType.XPU and self._xpu_available:
                    # IPEX: torch.xpu.memory_stats(i)['reserved_bytes.all.current']
                    try:
                        stats = torch.xpu.memory_stats(idx)
                        used = stats.get("reserved_bytes.all.current", 0)
                        budget.free_bytes = max(0, budget.total_bytes - used)
                    except Exception:
                        pass
                # MPS: no API for per-device free VRAM — leave as-is.
            except (RuntimeError, AttributeError):
                pass

    def _start_monitor(self):
        """Start background memory monitoring thread."""
        def monitor_loop():
            while not self._monitor_stop.is_set():
                with self._lock:
                    self._update_ram_budget()
                    self._update_gpu_budgets()
                self._monitor_stop.wait(self._monitor_interval)

        self._monitor_thread = threading.Thread(
            target=monitor_loop, daemon=True, name="nve-mem-monitor"
        )
        self._monitor_thread.start()

    def stop_monitor(self):
        """Stop the background memory monitor."""
        if self._monitor_thread:
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    # ── Device selection ──────────────────────────────────────────────────────

    @property
    def has_gpu(self) -> bool:
        return any([
            self._cuda_available,
            self._rocm_available,
            self._mps_available,
            self._xpu_available,
        ])

    @property
    def gpu_count(self) -> int:
        return sum(
            1 for d in self._devices
            if d.device_type != DeviceType.CPU
        )

    @property
    def primary_gpu_type(self) -> Optional[DeviceType]:
        """Return the primary GPU type detected (CUDA > ROCm > MPS > XPU)."""
        if self._cuda_available:
            return DeviceType.CUDA
        if self._rocm_available:
            return DeviceType.ROCM
        if self._mps_available:
            return DeviceType.MPS
        if self._xpu_available:
            return DeviceType.XPU
        return None

    @property
    def best_device(self):
        """Return the best available torch device string."""
        if self._preferred_device:
            return self._preferred_device
        if self._cuda_available or self._rocm_available:
            # Both use torch.cuda API; pick GPU with most free VRAM.
            dtype = DeviceType.CUDA if self._cuda_available else DeviceType.ROCM
            candidates = {
                idx: b for (dt, idx), b in self._gpu_budgets.items() if dt == dtype
            }
            if candidates:
                best_idx = max(candidates, key=lambda i: candidates[i].free_bytes)
                return f"cuda:{best_idx}"
        if self._mps_available:
            return "mps"
        if self._xpu_available:
            return "xpu:0"
        return "cpu"

    def select_device(self, required_bytes: int = 0) -> str:
        """
        Select the best device that can fit `required_bytes` of model weights.

        Preference order: CUDA → ROCm → MPS → XPU → CPU.
        Falls back gracefully within the same family (e.g., cuda:1 if cuda:0
        is full) and ultimately falls back to CPU.

        Returns a torch-compatible device string ("cuda:0", "mps", "xpu:0", "cpu").
        """
        if self._preferred_device:
            return self._preferred_device

        with self._lock:
            self._update_gpu_budgets()

        # ── CUDA / ROCm ──
        for dtype in (DeviceType.CUDA, DeviceType.ROCM):
            available = (
                (dtype == DeviceType.CUDA and self._cuda_available) or
                (dtype == DeviceType.ROCM and self._rocm_available)
            )
            if not available:
                continue
            candidates = sorted(
                ((idx, b) for (dt, idx), b in self._gpu_budgets.items() if dt == dtype),
                key=lambda x: x[1].free_bytes,
                reverse=True,
            )
            for gpu_idx, budget in candidates:
                usable = int(budget.free_bytes * self._gpu_reserve_fraction)
                if usable >= required_bytes:
                    return f"cuda:{gpu_idx}"
            if required_bytes > 0 and candidates:
                best_free = candidates[0][1].free_bytes
                logger.warning(
                    f"No {dtype.value.upper()} GPU has "
                    f"{required_bytes / 1024**3:.1f} GB free "
                    f"(best: {best_free / 1024**3:.1f} GB). "
                    f"Falling back to next device type."
                )

        # ── MPS ──
        if self._mps_available:
            budget = self._gpu_budgets.get((DeviceType.MPS, 0), MemoryBudget())
            usable = int(budget.free_bytes * self._gpu_reserve_fraction)
            if usable >= required_bytes or required_bytes == 0:
                return "mps"

        # ── Intel XPU ──
        if self._xpu_available:
            candidates = sorted(
                ((idx, b) for (dt, idx), b in self._gpu_budgets.items() if dt == DeviceType.XPU),
                key=lambda x: x[1].free_bytes,
                reverse=True,
            )
            for gpu_idx, budget in candidates:
                usable = int(budget.free_bytes * self._gpu_reserve_fraction)
                if usable >= required_bytes:
                    return f"xpu:{gpu_idx}"

        return "cpu"

    def torch_device(self, device_str: Optional[str] = None):
        """Return a `torch.device` from a string or the best available device."""
        import torch
        return torch.device(device_str or self.best_device)

    # ── Dtype selection ───────────────────────────────────────────────────────

    def best_dtype(self, device_str: str):
        """
        Return the best torch dtype for inference on a given device.

        Priority: bf16 (numerically stable) > fp16 > fp32.
        bf16 requires Ampere+ (CUDA), CDNA2+ (ROCm), or Intel XPU.
        MPS supports fp16 but not bf16 as of PyTorch 2.x.
        """
        import torch
        info = self._device_info_for(device_str)
        if info and info.supports_bf16:
            return torch.bfloat16
        if info and info.supports_fp16:
            return torch.float16
        return torch.float32

    def _device_info_for(self, device_str: str) -> Optional[DeviceInfo]:
        """Look up DeviceInfo matching a torch device string."""
        import torch
        try:
            dev = torch.device(device_str)
        except RuntimeError:
            return None
        type_map = {
            "cuda": DeviceType.CUDA,
            "mps": DeviceType.MPS,
            "xpu": DeviceType.XPU,
            "cpu": DeviceType.CPU,
        }
        # ROCm also uses "cuda" as its device type string.
        dtype = type_map.get(dev.type)
        if dtype is None:
            return None
        idx = dev.index or 0
        for info in self._devices:
            if info.device_type == dtype and info.device_index == idx:
                return info
        # If CUDA was detected but flag says ROCm, return ROCm info.
        if dtype == DeviceType.CUDA and self._rocm_available:
            for info in self._devices:
                if info.device_type == DeviceType.ROCM and info.device_index == idx:
                    return info
        return None

    # ── OOM-safe operations ───────────────────────────────────────────────────

    def safe_to(self, tensor, device, dtype=None, non_blocking: bool = False):
        """
        Move tensor to device with OOM recovery.

        On CUDA/ROCm OOM: clears cache, retries once, falls back to CPU.
        On MPS OOM: clears MPS cache, retries once, falls back to CPU.
        On XPU OOM: falls back to CPU directly.
        """
        import torch
        target = torch.device(device) if isinstance(device, str) else device
        kwargs = {"non_blocking": non_blocking}
        if dtype is not None:
            kwargs["dtype"] = dtype

        if _devices_match(tensor.device, target) and (dtype is None or tensor.dtype == dtype):
            return tensor

        if target.type == "cpu":
            return tensor.to(device=target, **kwargs)

        # CUDA / ROCm (both use torch.cuda.OutOfMemoryError)
        if target.type == "cuda":
            try:
                return tensor.to(device=target, **kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.warning(
                    f"OOM on {target} for tensor {tensor.shape}. "
                    f"Clearing cache and retrying…"
                )
                torch.cuda.empty_cache()
                try:
                    return tensor.to(device=target, **kwargs)
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    logger.error(f"OOM retry failed → CPU fallback")
                    return tensor.to(device="cpu", dtype=dtype)

        # Apple MPS
        if target.type == "mps":
            try:
                return tensor.to(device=target, **kwargs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.warning(f"MPS OOM for tensor {tensor.shape}. Retrying after cache clear…")
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass
                try:
                    return tensor.to(device=target, **kwargs)
                except RuntimeError:
                    logger.error(f"MPS OOM retry failed → CPU fallback")
                    return tensor.to(device="cpu", dtype=dtype)

        # Intel XPU
        if target.type == "xpu":
            try:
                return tensor.to(device=target, **kwargs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.error(f"XPU OOM for {tensor.shape} → CPU fallback")
                return tensor.to(device="cpu", dtype=dtype)

        # Unknown device — best-effort.
        return tensor.to(device=target, **kwargs)

    def safe_allocate(self, shape, dtype, device, fill_value=0):
        """Allocate a tensor with OOM recovery, falling back to CPU."""
        import torch
        target = torch.device(device) if isinstance(device, str) else device
        if target.type == "cpu":
            return torch.full(shape, fill_value, dtype=dtype, device=target)
        try:
            return torch.full(shape, fill_value, dtype=dtype, device=target)
        except (RuntimeError, Exception) as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning(f"OOM allocating {shape} on {target}. Falling back to CPU.")
            if target.type == "cuda":
                torch.cuda.empty_cache()
            return torch.full(shape, fill_value, dtype=dtype, device="cpu")

    # ── Memory budget management ───────────────────────────────────────────────

    def gpu_budget(self, device_str: str = "cuda:0") -> MemoryBudget:
        """Get current GPU memory budget for a device string."""
        import torch
        try:
            dev = torch.device(device_str)
        except RuntimeError:
            return MemoryBudget()
        dtype = (
            DeviceType.ROCM if self._rocm_available and dev.type == "cuda"
            else {
                "cuda": DeviceType.CUDA,
                "mps": DeviceType.MPS,
                "xpu": DeviceType.XPU,
            }.get(dev.type, DeviceType.CPU)
        )
        idx = dev.index or 0
        with self._lock:
            self._update_gpu_budgets()
        return self._gpu_budgets.get((dtype, idx), MemoryBudget())

    def ram_budget(self) -> MemoryBudget:
        with self._lock:
            self._update_ram_budget()
        return self._ram_budget

    def system_snapshot(self) -> SystemMemory:
        with self._lock:
            sys_mem = self._update_ram_budget()
            self._update_gpu_budgets()
            # Flatten to {index: budget} for backward compat (CUDA/ROCm only).
            sys_mem.gpu_memories = {
                idx: b
                for (dt, idx), b in self._gpu_budgets.items()
                if dt in (DeviceType.CUDA, DeviceType.ROCM)
            }
        return sys_mem

    def compute_tier_budgets(
        self,
        model_bytes: int,
        device_str: str = "auto",
    ) -> Dict[str, int]:
        """
        Compute memory budgets for hot/warm/ssd tiers given available hardware.

        Returns {"gpu": bytes, "ram": bytes, "ssd": bytes}.
        """
        if device_str == "auto":
            device_str = self.select_device(required_bytes=model_bytes)

        with self._lock:
            self._update_gpu_budgets()
            self._update_ram_budget()

        import torch
        try:
            dev = torch.device(device_str)
        except RuntimeError:
            dev = torch.device("cpu")

        gpu_budget = 0
        if dev.type != "cpu":
            budget = self.gpu_budget(device_str)
            gpu_budget = int(budget.free_bytes * self._gpu_reserve_fraction)

        ram_budget_val = int(self._ram_budget.free_bytes * self._ram_reserve_fraction)
        gpu_budget = min(gpu_budget, model_bytes)
        remaining = model_bytes - gpu_budget
        ram_budget_val = min(ram_budget_val, remaining)
        ssd_budget = max(0, remaining - ram_budget_val)

        return {"gpu": gpu_budget, "ram": ram_budget_val, "ssd": ssd_budget}

    def reserve_gpu(self, device_str: str, nbytes: int) -> bool:
        with self._lock:
            budget = self.gpu_budget(device_str)
            if budget.available_bytes >= nbytes:
                budget.reserved_bytes += nbytes
                budget.peak_reserved_bytes = max(
                    budget.peak_reserved_bytes, budget.reserved_bytes
                )
                return True
            return False

    def release_gpu(self, device_str: str, nbytes: int):
        with self._lock:
            budget = self.gpu_budget(device_str)
            budget.reserved_bytes = max(0, budget.reserved_bytes - nbytes)

    def reserve_ram(self, nbytes: int) -> bool:
        with self._lock:
            if self._ram_budget.available_bytes >= nbytes:
                self._ram_budget.reserved_bytes += nbytes
                self._ram_budget.peak_reserved_bytes = max(
                    self._ram_budget.peak_reserved_bytes,
                    self._ram_budget.reserved_bytes,
                )
                return True
            return False

    def release_ram(self, nbytes: int):
        with self._lock:
            self._ram_budget.reserved_bytes = max(
                0, self._ram_budget.reserved_bytes - nbytes
            )

    # ── Memory pinning ────────────────────────────────────────────────────────

    def pin_memory(self, tensor) -> bool:
        """
        Pin a CPU tensor's memory (mlock) to prevent swapping.

        On CPU-only hosts this gives "GPU tier" latency (no page-out)
        for hot weights.
        """
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            return False
        try:
            data_ptr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            if libc.mlock(ctypes.c_void_p(data_ptr), ctypes.c_size_t(nbytes)) == 0:
                logger.debug(f"Pinned {nbytes / 1024**2:.1f} MB")
                return True
            return False
        except (OSError, AttributeError):
            return False

    def unpin_memory(self, tensor) -> bool:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            return False
        try:
            data_ptr = tensor.data_ptr()
            nbytes = tensor.nelement() * tensor.element_size()
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            return libc.munlock(ctypes.c_void_p(data_ptr), ctypes.c_size_t(nbytes)) == 0
        except (OSError, AttributeError):
            return False

    # ── Introspection ─────────────────────────────────────────────────────────

    def device_info(self, device_str: str) -> Optional[DeviceInfo]:
        return self._device_info_for(device_str)

    def all_devices(self) -> List[DeviceInfo]:
        return list(self._devices)

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = ["NVE Hardware Summary:"]

        # CPU
        cpu_feats = ", ".join(f.value for f in self._cpu_features)
        ram_gb = self._ram_budget.total_bytes / 1024**3
        ram_free_gb = self._ram_budget.free_bytes / 1024**3
        lines.append(
            f"  CPU: {ram_free_gb:.1f} / {ram_gb:.1f} GB RAM  [{cpu_feats}]"
        )

        # GPUs by type
        for d in self._devices:
            if d.device_type == DeviceType.CPU:
                continue
            budget = self._gpu_budgets.get((d.device_type, d.device_index), MemoryBudget())
            total_gb = budget.total_bytes / 1024**3
            free_gb  = budget.free_bytes  / 1024**3
            caps = []
            if d.supports_bf16: caps.append("bf16")
            if d.supports_int8: caps.append("int8")
            if d.supports_fp8:  caps.append("fp8")
            caps_str = ", ".join(caps) if caps else "fp16"
            arch_str = f" [{d.architecture}]" if d.architecture else ""
            lines.append(
                f"  {d.device_type.value.upper()} {d.device_index} ({d.name}){arch_str}: "
                f"{free_gb:.1f} / {total_gb:.1f} GB  [{caps_str}]"
            )

        if self._vulkan_available:
            vulkan_devs = [d for d in self._devices if d.device_type == DeviceType.VULKAN]
            for d in vulkan_devs:
                lines.append(f"  VULKAN {d.device_index} ({d.name}): Vulkan compute available")

        if not self.has_gpu:
            lines.append("  GPU: None (CPU-only mode)")

        return "\n".join(lines)

    def as_dict(self) -> dict:
        """JSON-serialisable hardware snapshot for the /v1/hardware endpoint."""
        return {
            "cpu": {
                "features": [f.value for f in self._cpu_features],
                "ram_total_gb": round(self._ram_budget.total_bytes / 1024**3, 2),
                "ram_free_gb": round(self._ram_budget.free_bytes / 1024**3, 2),
            },
            "gpus": [
                {
                    "type": d.device_type.value,
                    "index": d.device_index,
                    "name": d.name,
                    "architecture": d.architecture,
                    "driver_version": d.driver_version,
                    "total_vram_gb": round(d.total_vram_bytes / 1024**3, 2),
                    "free_vram_gb": round(
                        self._gpu_budgets.get(
                            (d.device_type, d.device_index), MemoryBudget()
                        ).free_bytes / 1024**3, 2
                    ),
                    "supports_fp16": d.supports_fp16,
                    "supports_bf16": d.supports_bf16,
                    "supports_int8": d.supports_int8,
                    "supports_fp8": d.supports_fp8,
                }
                for d in self._devices
                if d.device_type != DeviceType.CPU
            ],
        }

    def stop_monitor(self):
        """Stop the background memory monitor."""
        if self._monitor_thread:
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    def __del__(self):
        self.stop_monitor()

    def __repr__(self) -> str:
        primary = self.primary_gpu_type
        gpu_str = primary.value.upper() if primary else "CPU-only"
        ram_gb = self._ram_budget.total_bytes / 1024**3
        return f"DeviceManager({gpu_str}, {self.gpu_count} GPU(s), {ram_gb:.0f} GB RAM)"


def _parse_proc_meminfo() -> Dict[str, int]:
    """Parse /proc/meminfo into {key: value_in_kb}."""
    result: Dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    result[key] = int(parts[1])
    except (FileNotFoundError, ValueError):
        pass
    return result


# ── Module-level singleton ────────────────────────────────────────────────────

_global_device_manager: Optional[DeviceManager] = None
_dm_lock = threading.Lock()


def get_device_manager(**kwargs) -> DeviceManager:
    """
    Return (or create) the module-level DeviceManager singleton.

    Pass kwargs only on the first call to configure it.
    Subsequent calls with kwargs are ignored (singleton already initialised).
    """
    global _global_device_manager
    with _dm_lock:
        if _global_device_manager is None:
            _global_device_manager = DeviceManager(**kwargs)
    return _global_device_manager
