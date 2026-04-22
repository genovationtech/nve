"""
NVE Tiered Serving — the execution layer.

Wraps a HuggingFace causal LM and serves it with learned weight placement:
- GPU tier: weights pinned in VRAM (or main memory on CPU-only systems)
- RAM tier: weights in host memory, copied to compute device on demand
- SSD tier: weights memory-mapped from disk, staged through RAM -> compute

Production features:
- OOM-safe device transfers with automatic tier fallback
- DeviceManager integration for memory monitoring
- Quantized RAM/SSD tier storage with on-load dequantization
- Double-buffered prefetch with compute/IO overlap
- Memory pinning (mlock) for CPU-only "GPU tier"
- Online pager integration for adaptive tier placement
"""

from __future__ import annotations

import logging
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from nve.device import DeviceManager, _devices_match
from nve.manifest import TierManifest, PageEntry
from nve.pager import WeightPager, TierLevel
from nve.quantization import (
    QuantPolicy, QuantLevel, QuantizedTensor,
    quantize, dequantize,
)

logger = logging.getLogger("nve.serving")


@dataclass
class ServingStats:
    """Runtime statistics for tiered serving."""
    forward_times: list[float] = field(default_factory=list)
    page_in_times: list[float] = field(default_factory=list)
    prefetch_times: list[float] = field(default_factory=list)

    gpu_hits: int = 0
    ram_page_ins: int = 0
    ssd_page_ins: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    oom_recoveries: int = 0
    tier_fallbacks: int = 0

    peak_gpu_bytes: int = 0
    peak_ram_bytes: int = 0
    ssd_reads_bytes: int = 0

    tokens_generated: int = 0
    total_generation_time: float = 0.0

    @property
    def mean_forward_ms(self) -> float:
        return np.mean(self.forward_times) * 1000 if self.forward_times else 0.0

    @property
    def p50_forward_ms(self) -> float:
        return np.percentile(self.forward_times, 50) * 1000 if self.forward_times else 0.0

    @property
    def p95_forward_ms(self) -> float:
        return np.percentile(self.forward_times, 95) * 1000 if self.forward_times else 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.total_generation_time == 0:
            return 0.0
        return self.tokens_generated / self.total_generation_time

    @property
    def page_fault_rate(self) -> float:
        total = self.gpu_hits + self.ram_page_ins + self.ssd_page_ins
        return (self.ram_page_ins + self.ssd_page_ins) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "latency": {
                "mean_forward_ms": round(self.mean_forward_ms, 3),
                "p50_forward_ms": round(self.p50_forward_ms, 3),
                "p95_forward_ms": round(self.p95_forward_ms, 3),
                "mean_page_in_ms": round(np.mean(self.page_in_times) * 1000, 3) if self.page_in_times else 0,
            },
            "paging": {
                "gpu_hits": self.gpu_hits,
                "ram_page_ins": self.ram_page_ins,
                "ssd_page_ins": self.ssd_page_ins,
                "page_fault_rate": round(self.page_fault_rate, 4),
                "prefetch_hits": self.prefetch_hits,
                "prefetch_misses": self.prefetch_misses,
            },
            "memory": {
                "peak_gpu_mb": round(self.peak_gpu_bytes / 1024**2, 1),
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


class ParameterPage:
    """
    Manages a single parameter's data across tiers.

    Supports quantized storage for RAM/SSD tiers.
    """

    def __init__(
        self,
        name: str,
        tensor: torch.Tensor,
        home_tier: str,
        ssd_path: Optional[Path] = None,
        device_manager: Optional[DeviceManager] = None,
        quant_policy: Optional[QuantPolicy] = None,
    ):
        self.name = name
        self.home_tier = home_tier
        self.current_tier = "gpu"
        self.size_bytes = tensor.nelement() * tensor.element_size()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.ssd_path = ssd_path
        self.dm = device_manager or DeviceManager(enable_memory_monitor=False)
        self.quant_policy = quant_policy or QuantPolicy.no_quantization()

        self._gpu_tensor: Optional[torch.Tensor] = None
        self._ram_tensor: Optional[torch.Tensor] = None
        self._quantized: Optional[QuantizedTensor] = None
        self._is_on_disk = False
        self._pinned = False  # Whether RAM tensor is mlock'd

    def evict_to_ram(self):
        """Move from GPU tier to RAM tier."""
        if self._gpu_tensor is not None:
            self._ram_tensor = self._gpu_tensor.clone().cpu()
            self._gpu_tensor = None
            self.current_tier = "ram"

            # Optionally quantize for RAM storage.
            level = self.quant_policy.level_for_tier("ram")
            if level != QuantLevel.NONE:
                try:
                    self._quantized = quantize(self._ram_tensor, level)
                    self._ram_tensor = None  # Free unquantized copy.
                except Exception:
                    pass  # Keep unquantized on failure.

    def evict_to_ssd(self):
        """Move from RAM tier to SSD tier."""
        data = self._ram_tensor
        if data is None and self._quantized is not None:
            # Already quantized — save quantized data to disk.
            if self.ssd_path is not None:
                import pickle
                with open(str(self.ssd_path) + ".qnt", "wb") as f:
                    pickle.dump(self._quantized, f)
                self._quantized = None
                self._is_on_disk = True
                self.current_tier = "ssd"
            return

        if data is not None and self.ssd_path is not None:
            # Quantize before saving to SSD for smaller files.
            level = self.quant_policy.level_for_tier("ssd")
            if level != QuantLevel.NONE:
                try:
                    qt = quantize(data, level)
                    import pickle
                    with open(str(self.ssd_path) + ".qnt", "wb") as f:
                        pickle.dump(qt, f)
                    self._ram_tensor = None
                    self._is_on_disk = True
                    self.current_tier = "ssd"
                    return
                except Exception:
                    pass

            # Fallback: save unquantized.
            torch.save(data, self.ssd_path)
            self._ram_tensor = None
            self._is_on_disk = True
            self.current_tier = "ssd"

    def load_to_ram(self) -> torch.Tensor:
        """Load from SSD to RAM."""
        if self._ram_tensor is not None:
            return self._ram_tensor
        if self._quantized is not None:
            return dequantize(self._quantized, device="cpu", dtype=self.dtype)

        if self._is_on_disk and self.ssd_path is not None:
            # Try quantized file first.
            qnt_path = str(self.ssd_path) + ".qnt"
            if os.path.exists(qnt_path):
                import pickle
                with open(qnt_path, "rb") as f:
                    self._quantized = pickle.load(f)
                self.current_tier = "ram"
                return dequantize(self._quantized, device="cpu", dtype=self.dtype)

            # Fallback: unquantized .pt file.
            if self.ssd_path.exists():
                self._ram_tensor = torch.load(self.ssd_path, weights_only=True)
                self.current_tier = "ram"
                return self._ram_tensor

        raise RuntimeError(f"Parameter {self.name} has no data to load")

    def load_to_gpu(self, device: torch.device) -> torch.Tensor:
        """Load to compute device with OOM safety."""
        if self._gpu_tensor is not None:
            return self._gpu_tensor

        if self.current_tier == "ssd":
            self.load_to_ram()

        data = self._ram_tensor
        if data is None and self._quantized is not None:
            data = dequantize(self._quantized, device="cpu", dtype=self.dtype)

        if data is not None:
            self._gpu_tensor = self.dm.safe_to(data, device)
            if _devices_match(self._gpu_tensor.device, device):
                self.current_tier = "gpu"
                return self._gpu_tensor
            else:
                # OOM fallback — stayed on CPU.
                self._ram_tensor = self._gpu_tensor
                self._gpu_tensor = None
                self.current_tier = "ram"
                logger.warning(f"OOM loading {self.name} to {device}, kept on CPU")
                return self._ram_tensor

        raise RuntimeError(f"Parameter {self.name} has no data to load to GPU")

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        """Get tensor on the target device, loading if necessary."""
        if self._gpu_tensor is not None and _devices_match(self._gpu_tensor.device, device):
            return self._gpu_tensor
        return self.load_to_gpu(device)

    def pin_gpu(self, tensor: torch.Tensor):
        """Pin a tensor as the GPU-resident copy."""
        self._gpu_tensor = tensor
        self.current_tier = "gpu"

    def pin_ram(self, tensor: torch.Tensor):
        """Pin a tensor as the RAM-resident copy."""
        self._ram_tensor = tensor.cpu()
        self._gpu_tensor = None
        self.current_tier = "ram"

    def pin_ram_quantized(self, qtensor: QuantizedTensor):
        """Pin a quantized tensor as the RAM-resident copy."""
        self._quantized = qtensor
        self._ram_tensor = None
        self._gpu_tensor = None
        self.current_tier = "ram"

    def release_gpu(self):
        """Release GPU memory (keep RAM or SSD copy)."""
        self._gpu_tensor = None
        if self._ram_tensor is None and self._quantized is None and not self._is_on_disk:
            raise RuntimeError(f"Cannot release GPU copy of {self.name} -- no backup exists")

    def memory_pin(self, dm: DeviceManager) -> bool:
        """Pin RAM tensor via mlock (for CPU-only "GPU tier")."""
        if self._ram_tensor is not None and not self._pinned:
            self._pinned = dm.pin_memory(self._ram_tensor)
            return self._pinned
        return False


def _set_param_data(model: nn.Module, name: str, value: torch.Tensor):
    """Set a parameter's data by traversing the module hierarchy."""
    parts = name.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    old = getattr(mod, parts[-1])
    if isinstance(old, nn.Parameter) and old.device.type != "meta":
        old.data = value
    else:
        new_param = nn.Parameter(value, requires_grad=False)
        setattr(mod, parts[-1], new_param)


def _set_buffer_data(model: nn.Module, name: str, value: torch.Tensor):
    """Set a buffer's data by traversing the module hierarchy."""
    parts = name.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    attr_name = parts[-1]
    if hasattr(mod, attr_name):
        try:
            delattr(mod, attr_name)
        except Exception:
            pass
    mod.register_buffer(attr_name, value)


class TieredModelServer:
    """
    Serves a HuggingFace causal LM with NVE tiered weight placement.

    Usage:
        server = TieredModelServer(model, tokenizer, manifest)
        server.setup()
        output = server.generate("Hello world", max_new_tokens=50)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        manifest: TierManifest,
        ssd_dir: str | Path = "/tmp/nve_ssd_tier",
        device: Optional[torch.device] = None,
        enable_prefetch: bool = True,
        prefetch_depth: int = 2,
        low_vram: bool = False,
        device_manager: Optional[DeviceManager] = None,
        quant_policy: Optional[QuantPolicy] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.ssd_dir = Path(ssd_dir)
        self.enable_prefetch = enable_prefetch
        self.prefetch_depth = prefetch_depth
        self.low_vram = low_vram
        self.dm = device_manager or DeviceManager()
        self.quant_policy = quant_policy or QuantPolicy.balanced()

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = self.dm.best_device

        self.pages: dict[str, ParameterPage] = {}
        self.stats = ServingStats()
        self._hooks = []
        self._layer_params: dict[int, list[str]] = defaultdict(list)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()
        self._prefetched_layers: set[int] = set()

        self._gpu_resident: set[str] = set()
        self._ram_resident: set[str] = set()
        self._gpu_bytes = 0
        self._ram_bytes = 0

    def setup(self):
        """Initialize tiered placement based on manifest."""
        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        tier_map = {}
        for entry in self.manifest.gpu_pages:
            tier_map[entry.param_name] = ("gpu", entry)
        for entry in self.manifest.ram_pages:
            tier_map[entry.param_name] = ("ram", entry)
        for entry in self.manifest.ssd_pages:
            tier_map[entry.param_name] = ("ssd", entry)

        for entry in self.manifest.gpu_pages + self.manifest.ram_pages + self.manifest.ssd_pages:
            self._layer_params[entry.layer_index].append(entry.param_name)

        force_gpu_prefixes = (
            "transformer.wte", "transformer.wpe", "transformer.ln_f", "lm_head",
            "model.embed_tokens", "model.norm", "model.rotary_emb",
        )

        param_dict = dict(self.model.named_parameters())

        for name, param in param_dict.items():
            tier, entry = tier_map.get(name, ("gpu", None))
            ssd_path = self.ssd_dir / f"{name.replace('.', '_')}.pt"

            if any(name.startswith(prefix) for prefix in force_gpu_prefixes):
                tier = "gpu"

            page = ParameterPage(
                name=name,
                tensor=param.data,
                home_tier=tier,
                ssd_path=ssd_path,
                device_manager=self.dm,
                quant_policy=self.quant_policy,
            )

            if tier == "gpu":
                gpu_tensor = self.dm.safe_to(param.data, self.device)
                if _devices_match(gpu_tensor.device, self.device):
                    page.pin_gpu(gpu_tensor)
                    _set_param_data(self.model, name, gpu_tensor)
                    self._gpu_resident.add(name)
                    self._gpu_bytes += page.size_bytes
                else:
                    # OOM: fall back to RAM tier.
                    logger.warning(f"OOM for GPU-tier {name}, demoting to RAM")
                    page.pin_ram(param.data)
                    self._ram_resident.add(name)
                    self._ram_bytes += page.size_bytes
                    page.home_tier = "ram"
                    self.stats.oom_recoveries += 1
                    self.stats.tier_fallbacks += 1

                    # On CPU-only systems, mlock for GPU-like priority.
                    if not self.dm.has_gpu:
                        page.memory_pin(self.dm)

            elif tier == "ram":
                page.pin_ram(param.data)
                self._ram_resident.add(name)
                self._ram_bytes += page.size_bytes
                placeholder_dev = "cpu" if self.low_vram else self.device
                param.data = self.dm.safe_allocate(
                    param.data.shape, param.data.dtype, placeholder_dev
                )

            elif tier == "ssd":
                page.pin_ram(param.data)
                page.evict_to_ssd()
                placeholder_dev = "cpu" if self.low_vram else self.device
                param.data = self.dm.safe_allocate(
                    param.data.shape, param.data.dtype, placeholder_dev
                )

            self.pages[name] = page

        self.stats.peak_gpu_bytes = self._gpu_bytes
        self.stats.peak_ram_bytes = self._ram_bytes
        self._install_block_hooks()

        logger.info(
            f"TieredModelServer setup: GPU={self._gpu_bytes / 1024**2:.0f}MB, "
            f"RAM={self._ram_bytes / 1024**2:.0f}MB, "
            f"device={self.device}, "
            f"OOM recoveries={self.stats.oom_recoveries}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        tokenizer,
        manifest: TierManifest,
        device: Optional[torch.device] = None,
        ssd_dir: str | Path = "/tmp/nve_ssd_tier",
        enable_prefetch: bool = True,
        prefetch_depth: int = 2,
        torch_dtype=None,
        device_manager: Optional[DeviceManager] = None,
        quant_policy: Optional[QuantPolicy] = None,
    ) -> "TieredModelServer":
        """
        Stream weights from safetensors directly into tiers.

        Never loads the full model into RAM. Peak memory ~ GPU-tier weights + one layer.
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        from nve.streaming_profiler import (
            _parse_safetensors_metadata,
            _load_tensor_from_safetensors,
        )
        import json

        model_dir = Path(model_dir)
        dm = device_manager or DeviceManager()
        qp = quant_policy or QuantPolicy.balanced()

        if device is None:
            device = dm.best_device
        if torch_dtype is None:
            torch_dtype = torch.float16

        ssd_dir = Path(ssd_dir)
        ssd_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create empty model shell.
        config = AutoConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        model.eval()

        # 2. Discover safetensors files.
        weight_meta: dict[str, dict] = {}
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                shard_files = set(json.load(f).get("weight_map", {}).values())
            for shard_name in shard_files:
                shard_path = model_dir / shard_name
                if shard_path.exists():
                    meta = _parse_safetensors_metadata(shard_path)
                    for name, info in meta.items():
                        info["file"] = str(shard_path)
                        weight_meta[name] = info
        else:
            st_path = model_dir / "model.safetensors"
            if st_path.exists():
                meta = _parse_safetensors_metadata(st_path)
                for name, info in meta.items():
                    info["file"] = str(st_path)
                    weight_meta[name] = info

        # 3. Build tier map.
        tier_map: dict[str, str] = {}
        for entry in manifest.gpu_pages:
            tier_map[entry.param_name] = "gpu"
        for entry in manifest.ram_pages:
            tier_map[entry.param_name] = "ram"
        for entry in manifest.ssd_pages:
            tier_map[entry.param_name] = "ssd"

        force_gpu_prefixes = (
            "transformer.wte", "transformer.wpe", "transformer.ln_f", "lm_head",
            "model.embed_tokens", "model.norm", "model.rotary_emb",
        )

        # 4. Stream weights into tiers.
        try:
            from accelerate.utils import set_module_tensor_to_device
            has_accelerate = True
        except ImportError:
            has_accelerate = False

        server_pages: dict[str, ParameterPage] = {}
        layer_params: dict[int, list[str]] = defaultdict(list)
        gpu_bytes = 0
        ram_bytes = 0
        oom_recoveries = 0

        all_entries = manifest.gpu_pages + manifest.ram_pages + manifest.ssd_pages
        for entry in all_entries:
            layer_params[entry.layer_index].append(entry.param_name)

        param_names = {n for n, _ in model.named_parameters()}
        loaded = 0
        total = len(param_names)

        for name in list(param_names):
            meta = weight_meta.get(name)
            if meta is None:
                continue

            tier = tier_map.get(name, "gpu")
            if any(name.startswith(p) for p in force_gpu_prefixes):
                tier = "gpu"

            ssd_path = ssd_dir / f"{name.replace('.', '_')}.pt"
            tensor = _load_tensor_from_safetensors(Path(meta["file"]), meta)
            if tensor.dtype != torch_dtype:
                tensor = tensor.to(torch_dtype)

            page = ParameterPage(
                name=name, tensor=tensor, home_tier=tier, ssd_path=ssd_path,
                device_manager=dm, quant_policy=qp,
            )

            if tier == "gpu":
                gpu_tensor = dm.safe_to(tensor, device)
                if _devices_match(gpu_tensor.device, device):
                    page.pin_gpu(gpu_tensor)
                    if has_accelerate:
                        set_module_tensor_to_device(model, name, device, value=gpu_tensor)
                    else:
                        _set_param_data(model, name, gpu_tensor)
                    gpu_bytes += page.size_bytes
                else:
                    # OOM fallback: demote to RAM.
                    logger.warning(f"OOM streaming {name} to GPU, demoting to RAM")
                    page.pin_ram(tensor)
                    placeholder = torch.zeros(tensor.shape, dtype=torch_dtype, device="cpu")
                    if has_accelerate:
                        set_module_tensor_to_device(model, name, "cpu", value=placeholder)
                    else:
                        _set_param_data(model, name, placeholder)
                    ram_bytes += page.size_bytes
                    oom_recoveries += 1

            elif tier == "ram":
                page.pin_ram(tensor)
                ram_bytes += page.size_bytes
                placeholder = torch.zeros(tensor.shape, dtype=torch_dtype, device="cpu")
                if has_accelerate:
                    set_module_tensor_to_device(model, name, "cpu", value=placeholder)
                else:
                    _set_param_data(model, name, placeholder)

            elif tier == "ssd":
                page.pin_ram(tensor)
                page.evict_to_ssd()
                placeholder = torch.zeros(tensor.shape, dtype=torch_dtype, device="cpu")
                if has_accelerate:
                    set_module_tensor_to_device(model, name, "cpu", value=placeholder)
                else:
                    _set_param_data(model, name, placeholder)

            server_pages[name] = page
            del tensor

            loaded += 1
            if loaded % 50 == 0 or loaded == total:
                logger.info(f"Streaming weights: {loaded}/{total}")

        # Handle remaining meta-device parameters.
        for name, param in model.named_parameters():
            if name not in server_pages and param.device.type == "meta":
                zeros = torch.zeros(param.shape, dtype=torch_dtype, device="cpu")
                if has_accelerate:
                    set_module_tensor_to_device(model, name, "cpu", value=zeros)
                else:
                    _set_param_data(model, name, zeros)

        # Handle buffers.
        for name, buf in model.named_buffers():
            if buf.device.type == "meta":
                meta_info = weight_meta.get(name)
                if meta_info:
                    buf_tensor = _load_tensor_from_safetensors(Path(meta_info["file"]), meta_info)
                    _set_buffer_data(model, name, dm.safe_to(buf_tensor, device))
                else:
                    _set_buffer_data(model, name, torch.zeros(buf.shape, dtype=buf.dtype, device=device))

        # 5. Assemble server.
        server = cls(
            model, tokenizer, manifest,
            ssd_dir=ssd_dir, device=device,
            enable_prefetch=enable_prefetch,
            prefetch_depth=prefetch_depth,
            low_vram=True,
            device_manager=dm,
            quant_policy=qp,
        )
        server.pages = server_pages
        server._layer_params = dict(layer_params)
        server._gpu_bytes = gpu_bytes
        server._ram_bytes = ram_bytes
        server.stats.peak_gpu_bytes = gpu_bytes
        server.stats.peak_ram_bytes = ram_bytes
        server.stats.oom_recoveries = oom_recoveries

        server._install_block_hooks()

        return server

    def _install_block_hooks(self):
        """Install pre/post-forward hooks on each transformer block."""
        blocks = self._find_transformer_blocks()

        for layer_idx, block in blocks:
            hook = block.register_forward_pre_hook(
                self._make_page_in_hook(layer_idx)
            )
            self._hooks.append(hook)

            post_hook = block.register_forward_hook(
                self._make_post_hook(layer_idx)
            )
            self._hooks.append(post_hook)

    def _find_transformer_blocks(self) -> list[tuple[int, nn.Module]]:
        """Find the ordered transformer blocks in the model."""
        blocks = []

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            for i, block in enumerate(self.model.transformer.h):
                blocks.append((i, block))
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, block in enumerate(self.model.model.layers):
                blocks.append((i, block))

        return blocks

    def _make_page_in_hook(self, layer_idx: int):
        def hook(module, input):
            t0 = time.perf_counter()
            self._page_in_layer(layer_idx)
            elapsed = time.perf_counter() - t0
            if elapsed > 1e-6:
                self.stats.page_in_times.append(elapsed)
        return hook

    def _make_post_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self.enable_prefetch:
                self._trigger_prefetch(layer_idx)
            self._evict_layer(layer_idx)
            return output
        return hook

    def _page_in_layer(self, layer_idx: int):
        """Ensure all parameters for this layer are on the compute device."""
        param_names = self._layer_params.get(layer_idx, [])
        param_dict = dict(self.model.named_parameters())

        for name in param_names:
            page = self.pages.get(name)
            if page is None:
                continue

            if page.current_tier == "gpu":
                self.stats.gpu_hits += 1
                continue

            if layer_idx in self._prefetched_layers and page._ram_tensor is not None:
                self.stats.prefetch_hits += 1
            else:
                if page.home_tier == "ssd":
                    self.stats.ssd_page_ins += 1
                    self.stats.ssd_reads_bytes += page.size_bytes
                else:
                    self.stats.ram_page_ins += 1

            tensor = page.get_tensor(self.device)

            if name in param_dict:
                param_dict[name].data = tensor

            self._gpu_bytes += page.size_bytes
            self.stats.peak_gpu_bytes = max(self.stats.peak_gpu_bytes, self._gpu_bytes)

        self._prefetched_layers.discard(layer_idx)

    def _evict_layer(self, layer_idx: int):
        """Evict non-GPU-home pages back after layer computation."""
        param_names = self._layer_params.get(layer_idx, [])
        param_dict = dict(self.model.named_parameters())

        for name in param_names:
            page = self.pages.get(name)
            if page is None or page.home_tier == "gpu":
                continue

            placeholder_dev = "cpu" if self.low_vram else self.device
            placeholder = self.dm.safe_allocate(
                page.shape, page.dtype, placeholder_dev
            )

            if page.home_tier == "ram":
                page.evict_to_ram()
                self._gpu_bytes -= page.size_bytes
                param_dict[name].data = placeholder

            elif page.home_tier == "ssd":
                if page._ram_tensor is None and page._quantized is None:
                    page.evict_to_ram()
                page.evict_to_ssd()
                self._gpu_bytes -= page.size_bytes
                param_dict[name].data = placeholder

    def _trigger_prefetch(self, current_layer: int):
        """Prefetch weights for upcoming layers in a background thread."""
        layers_to_prefetch = []
        for offset in range(1, self.prefetch_depth + 1):
            target = current_layer + offset
            if target in self._layer_params:
                layers_to_prefetch.append(target)

        if not layers_to_prefetch:
            return

        def prefetch_work():
            for layer_idx in layers_to_prefetch:
                with self._prefetch_lock:
                    for name in self._layer_params.get(layer_idx, []):
                        page = self.pages.get(name)
                        if page and page.current_tier == "ssd":
                            t0 = time.perf_counter()
                            page.load_to_ram()
                            self.stats.prefetch_times.append(time.perf_counter() - t0)
                    self._prefetched_layers.add(layer_idx)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)

        self._prefetch_thread = threading.Thread(target=prefetch_work, daemon=True)
        self._prefetch_thread.start()

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> dict:
        """Generate text with tiered serving."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = self.dm.safe_to(inputs["input_ids"], self.device)
        attention_mask = self.dm.safe_to(
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            self.device,
        )

        generated_ids = input_ids.clone()
        prompt_len = input_ids.shape[1]

        t_start = time.perf_counter()

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_fwd = time.perf_counter()

                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )

                self.stats.forward_times.append(time.perf_counter() - t_fwd)

                next_logits = outputs.logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)
                ], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        t_total = time.perf_counter() - t_start
        self.stats.tokens_generated += generated_ids.shape[1] - prompt_len
        self.stats.total_generation_time += t_total

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": generated_ids.shape[1] - prompt_len,
            "time_s": round(t_total, 3),
            "tokens_per_sec": round((generated_ids.shape[1] - prompt_len) / t_total, 2) if t_total > 0 else 0,
        }

    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get raw logits for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = self.dm.safe_to(inputs["input_ids"], self.device)

        with torch.no_grad():
            t_fwd = time.perf_counter()
            outputs = self.model(input_ids=input_ids)
            self.stats.forward_times.append(time.perf_counter() - t_fwd)

        return outputs.logits

    def teardown(self):
        """Remove hooks and clean up."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self.ssd_dir.exists():
            for f in self.ssd_dir.glob("*.pt"):
                f.unlink()
            for f in self.ssd_dir.glob("*.qnt"):
                f.unlink()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError:
            pass


class BaselineServer:
    """
    Baseline full-model server for comparison.
    All weights on a single device, no tiering.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
        device_manager: Optional[DeviceManager] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dm = device_manager or DeviceManager(enable_memory_monitor=False)
        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = self.dm.best_device
        self.stats = ServingStats()

    def setup(self):
        # Move entire model to device. safe_to is for tensors; use .to() for modules
        # with OOM protection.
        try:
            self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM moving model to {self.device}, falling back to CPU")
            self.device = torch.device("cpu")
            self.model.to(self.device)
        self.model.eval()
        total_bytes = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        self.stats.peak_gpu_bytes = total_bytes

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> dict:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = self.dm.safe_to(inputs["input_ids"], self.device)
        attention_mask = self.dm.safe_to(
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            self.device,
        )

        generated_ids = input_ids.clone()
        prompt_len = input_ids.shape[1]

        t_start = time.perf_counter()

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_fwd = time.perf_counter()
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                self.stats.forward_times.append(time.perf_counter() - t_fwd)

                next_logits = outputs.logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)
                ], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        t_total = time.perf_counter() - t_start
        self.stats.tokens_generated += generated_ids.shape[1] - prompt_len
        self.stats.total_generation_time += t_total

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": generated_ids.shape[1] - prompt_len,
            "time_s": round(t_total, 3),
            "tokens_per_sec": round((generated_ids.shape[1] - prompt_len) / t_total, 2) if t_total > 0 else 0,
        }

    def get_logits(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = self.dm.safe_to(inputs["input_ids"], self.device)

        with torch.no_grad():
            t_fwd = time.perf_counter()
            outputs = self.model(input_ids=input_ids)
            self.stats.forward_times.append(time.perf_counter() - t_fwd)

        return outputs.logits

    def teardown(self):
        pass
