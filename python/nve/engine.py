"""
NVE Engine — main entry point for the Neural Virtualization Engine Python SDK.

Wraps the Rust core via ctypes FFI, with a pure-Python fallback for environments
where the Rust library isn't compiled.

Production features:
- DeviceManager integration for OOM-safe hardware detection
- Predictive paging during inference (not a stub)
- Online weight importance adaptation
- Quantization-aware tier configuration
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

from nve.profiler import MCAPProfiler, ActivationSample
from nve.pager import WeightPager, PagerStats, TierLevel
from nve.device import DeviceManager
from nve.quantization import QuantPolicy

logger = logging.getLogger("nve.engine")


@dataclass
class TierConfig:
    """Memory tier configuration."""
    gpu_bytes: int = 4 * 1024**3       # 4 GB
    ram_bytes: int = 16 * 1024**3      # 16 GB
    ssd_bytes: int = 128 * 1024**3     # 128 GB
    ssd_path: str = "/tmp/nve_weights"
    gpu_fraction: float = 0.2
    ram_fraction: float = 0.3

    @classmethod
    def from_budget(
        cls,
        model_bytes: int,
        gpu_budget_bytes: int,
        ram_budget_bytes: int,
        ssd_path: str = "/tmp/nve_weights",
    ) -> "TierConfig":
        """Compute tier fractions from byte budgets and model size."""
        if model_bytes <= 0:
            raise ValueError("model_bytes must be positive")

        gpu_frac = min(gpu_budget_bytes / model_bytes, 1.0)
        ram_frac = min(ram_budget_bytes / model_bytes, 1.0 - gpu_frac)
        ssd_bytes = max(0, model_bytes - gpu_budget_bytes - ram_budget_bytes)

        return cls(
            gpu_bytes=gpu_budget_bytes,
            ram_bytes=ram_budget_bytes,
            ssd_bytes=max(ssd_bytes, 1024**3),
            ssd_path=ssd_path,
            gpu_fraction=gpu_frac,
            ram_fraction=ram_frac,
        )

    @classmethod
    def auto(
        cls,
        model_bytes: int,
        device_manager: Optional[DeviceManager] = None,
        gpu_reserve_frac: float = 0.8,
        ram_reserve_frac: float = 0.8,
        ssd_path: str = "/tmp/nve_weights",
    ) -> "TierConfig":
        """
        Auto-detect hardware and compute tier config.

        Uses DeviceManager for robust, OOM-safe hardware detection
        instead of raw torch.cuda calls.
        """
        dm = device_manager or DeviceManager(enable_memory_monitor=False)
        budgets = dm.compute_tier_budgets(model_bytes)

        logger.info(
            f"Auto tier config: GPU={budgets['gpu'] / 1024**3:.1f}GB, "
            f"RAM={budgets['ram'] / 1024**3:.1f}GB, "
            f"SSD={budgets['ssd'] / 1024**3:.1f}GB "
            f"for {model_bytes / 1024**3:.1f}GB model"
        )

        return cls.from_budget(
            model_bytes=model_bytes,
            gpu_budget_bytes=budgets["gpu"],
            ram_budget_bytes=budgets["ram"],
            ssd_path=ssd_path,
        )


@dataclass
class EngineConfig:
    """Full engine configuration."""
    tier: TierConfig = field(default_factory=TierConfig)
    samples_per_round: int = 100
    min_samples_for_stability: int = 50
    ema_decay: float = 0.01
    cluster_pmi_threshold: float = 0.5
    cluster_max_size: int = 256
    co_activation_threshold: float = 0.3
    enable_prefetch: bool = True
    prefetch_count: int = 3
    # Online adaptation.
    enable_online_adaptation: bool = True
    adaptation_interval: int = 50     # Re-evaluate placements every N inferences.
    frequency_decay: float = 0.95
    promotion_threshold: int = 5
    # Quantization.
    quant_policy: QuantPolicy = field(default_factory=QuantPolicy.balanced)
    # Hot-only inference: use only GPU-tier weights, skip warm/cold tiers.
    # Inactive layers pass residual through unchanged (like Rust --hot-only).
    # Trades quality for speed — token generation stays fast regardless of model size.
    hot_only_mode: bool = False
    # Override number of active layers. None = fit as many as budget allows.
    # Lower = faster but lower quality. Works with hot_only_mode.
    active_layers: Optional[int] = None
    # Domain shift detection: when logit entropy exceeds this threshold,
    # temporarily pull weights from warm/cold tiers for better quality.
    # Set to 0 to disable domain shift detection.
    domain_shift_entropy_threshold: float = 4.0
    # Number of tokens to stay in full-tier mode after a domain shift.
    domain_shift_cooldown_tokens: int = 10


class NVEEngine:
    """
    Neural Virtualization Engine.

    Profiles model weight activations via Monte Carlo sampling, clusters
    co-activated weights, and pages them across GPU/RAM/SSD tiers dynamically.

    Usage:
        engine = NVEEngine(config=EngineConfig())
        engine.register_model(model)
        engine.profile(prompts)
        engine.build()
        result = engine.infer(prompt)
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        lib_path: Optional[str] = None,
        device_manager: Optional[DeviceManager] = None,
    ):
        self.config = config or EngineConfig()
        self.device_manager = device_manager or DeviceManager()
        self.profiler = MCAPProfiler(
            samples_per_round=self.config.samples_per_round,
            ema_decay=self.config.ema_decay,
        )
        self.pager = WeightPager(
            gpu_bytes=self.config.tier.gpu_bytes,
            ram_bytes=self.config.tier.ram_bytes,
            ssd_bytes=self.config.tier.ssd_bytes,
            gpu_fraction=self.config.tier.gpu_fraction,
            ram_fraction=self.config.tier.ram_fraction,
            frequency_decay=self.config.frequency_decay,
            promotion_threshold=self.config.promotion_threshold,
        )
        self._rust_lib = None
        self._rust_handle = None
        self._model = None
        self._hooks = []
        self._weight_blocks = {}
        self._is_built = False
        self._inference_count = 0
        self._co_activation_matrix = None
        self._rust_profiling_started = False

        # Try to load Rust core.
        if lib_path:
            self._load_rust(lib_path)
        else:
            self._try_load_rust()

        logger.info(f"NVE Engine initialized | {self.device_manager}")

    def _try_load_rust(self):
        """Attempt to find and load the compiled Rust library."""
        search_paths = [
            Path(__file__).parent.parent.parent / "target" / "release" / "libnve.so",
            Path(__file__).parent.parent.parent / "target" / "release" / "libnve.dylib",
            Path(__file__).parent.parent.parent / "target" / "debug" / "libnve.so",
        ]
        for path in search_paths:
            if path.exists():
                self._load_rust(str(path))
                return

    def _load_rust(self, path: str):
        """Load the Rust shared library."""
        try:
            self._rust_lib = ctypes.CDLL(path)
            self._setup_rust_bindings()
            self._rust_handle = self._rust_lib.nve_engine_new()
            logger.info(f"Rust core loaded from {path}")
        except OSError:
            self._rust_lib = None

    def _setup_rust_bindings(self):
        """Configure ctypes function signatures for the Rust FFI."""
        lib = self._rust_lib

        lib.nve_engine_new.restype = ctypes.c_void_p
        lib.nve_engine_new.argtypes = []

        lib.nve_engine_free.restype = None
        lib.nve_engine_free.argtypes = [ctypes.c_void_p]

        lib.nve_register_block.restype = None
        lib.nve_register_block.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
        ]

        lib.nve_start_profiling.restype = None
        lib.nve_start_profiling.argtypes = [ctypes.c_void_p]

        lib.nve_record_activation.restype = None
        lib.nve_record_activation.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_double,
        ]

        lib.nve_build.restype = None
        lib.nve_build.argtypes = [ctypes.c_void_p]

        lib.nve_page_fault_rate.restype = ctypes.c_double
        lib.nve_page_fault_rate.argtypes = [ctypes.c_void_p]

        lib.nve_cluster_count.restype = ctypes.c_size_t
        lib.nve_cluster_count.argtypes = [ctypes.c_void_p]

    def register_model(self, model) -> None:
        """Register a PyTorch model's parameters as weight blocks."""
        import torch

        self._model = model
        block_id = 0

        for name, param in model.named_parameters():
            size_bytes = param.nelement() * param.element_size()
            layer_idx = self._extract_layer_index(name)

            self._weight_blocks[block_id] = {
                "name": name,
                "param": param,
                "size_bytes": size_bytes,
                "layer_index": layer_idx,
            }

            # Register with pager.
            self.pager.register(
                weight_id=block_id,
                name=name,
                size_bytes=size_bytes,
                layer_index=layer_idx,
            )

            if self._rust_handle:
                self._rust_lib.nve_register_block(
                    self._rust_handle, block_id, layer_idx, 0, size_bytes, 0.0,
                )

            block_id += 1

    def _extract_layer_index(self, param_name: str) -> int:
        """Extract numeric layer index from parameter name."""
        parts = param_name.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    def profile(
        self,
        prompts: list[str],
        tokenizer=None,
        domains: Optional[list[str]] = None,
        forward_fn=None,
    ) -> dict:
        """
        Run Monte Carlo activation profiling on the model.

        Also builds a co-activation matrix for group paging.
        """
        import torch

        if self._rust_handle:
            self._rust_lib.nve_start_profiling(self._rust_handle)
            self._rust_profiling_started = True

        self.profiler.start()

        # Track co-activations for group discovery.
        n_weights = len(self._weight_blocks)
        co_activation_counts = np.zeros((n_weights, n_weights), dtype=np.float32)
        activation_hooks = self._install_activation_hooks()

        for i, prompt in enumerate(prompts):
            domain = domains[i] if domains and i < len(domains) else None

            if forward_fn:
                activations = forward_fn(self._model, prompt)
                self._process_activations(activations, domain)
            elif tokenizer and self._model:
                device = self.device_manager.best_device
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: self.device_manager.safe_to(v, device) for k, v in inputs.items()}
                    self._model(**inputs)
                    self._flush_hook_activations(domain, co_activation_counts)

            self.profiler.finish_round()

        self._remove_hooks(activation_hooks)

        # Build co-activation groups from the matrix.
        self._co_activation_matrix = co_activation_counts
        self._build_co_activation_groups(co_activation_counts)

        return {
            "total_prompts": len(prompts),
            "weights_profiled": self.profiler.weight_count(),
            "rounds": self.profiler.total_rounds(),
            "is_stable": self.profiler.is_stable(),
            "co_activation_groups": len(self.pager._groups),
        }

    def _build_co_activation_groups(self, co_matrix: np.ndarray):
        """
        Discover groups of co-activated weights from the co-activation matrix.

        Uses a simple threshold-based clustering: weights with PMI above
        the threshold are grouped together.
        """
        n = co_matrix.shape[0]
        if n == 0:
            return

        # Normalize to PMI-like scores.
        row_sums = co_matrix.sum(axis=1, keepdims=True).clip(1)
        col_sums = co_matrix.sum(axis=0, keepdims=True).clip(1)
        total = co_matrix.sum().clip(1)
        pmi = np.log((co_matrix * total) / (row_sums * col_sums + 1e-10) + 1e-10)
        np.fill_diagonal(pmi, 0)

        # Greedy clustering.
        threshold = self.config.cluster_pmi_threshold
        max_size = self.config.cluster_max_size
        assigned = set()
        groups = []

        for i in range(n):
            if i in assigned:
                continue
            group = {i}
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                if pmi[i, j] > threshold and len(group) < max_size:
                    group.add(j)
            if len(group) > 1:
                groups.append(group)
                assigned.update(group)

        self.pager.set_co_activation_groups(groups)
        logger.info(f"Built {len(groups)} co-activation groups from profiling data")

    def _install_activation_hooks(self) -> list:
        """Install forward hooks on model modules to capture activations."""
        hooks = []
        self._captured_activations = {}

        if self._model is None:
            return hooks

        for block_id, info in self._weight_blocks.items():
            name = info["name"]
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                module_name, _ = parts
                try:
                    module = dict(self._model.named_modules())[module_name]

                    def make_hook(bid):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                output = output[0]
                            if hasattr(output, "abs"):
                                self._captured_activations[bid] = float(
                                    output.abs().mean().item()
                                )
                        return hook_fn

                    h = module.register_forward_hook(make_hook(block_id))
                    hooks.append(h)
                except (KeyError, AttributeError):
                    pass

        return hooks

    def _flush_hook_activations(
        self,
        domain: Optional[str] = None,
        co_matrix: Optional[np.ndarray] = None,
    ):
        """Process activations captured by hooks."""
        samples = []
        active_ids = list(self._captured_activations.keys())

        for block_id, magnitude in self._captured_activations.items():
            sample = ActivationSample(
                weight_id=block_id,
                magnitude=magnitude,
                domain=domain,
            )
            samples.append(sample)

            if self._rust_handle:
                self._rust_lib.nve_record_activation(
                    self._rust_handle, block_id, magnitude,
                )

        # Update co-activation counts.
        if co_matrix is not None and len(active_ids) > 1:
            for i, a in enumerate(active_ids):
                for b in active_ids[i + 1:]:
                    if a < co_matrix.shape[0] and b < co_matrix.shape[1]:
                        co_matrix[a, b] += 1
                        co_matrix[b, a] += 1

        self.profiler.record_batch(samples)
        self._captured_activations.clear()

    def _process_activations(self, activations: dict, domain: Optional[str] = None):
        """Process a dict of {weight_id: magnitude} activations."""
        samples = []
        for weight_id, magnitude in activations.items():
            sample = ActivationSample(
                weight_id=weight_id,
                magnitude=float(magnitude),
                domain=domain,
            )
            samples.append(sample)
        self.profiler.record_batch(samples)

    def _remove_hooks(self, hooks: list):
        for h in hooks:
            h.remove()

    def build(self) -> dict:
        """
        Build weight clusters and initialize tier placement.
        Call after profiling is complete.
        """
        if self._rust_handle and self._rust_profiling_started:
            try:
                self._rust_lib.nve_build(self._rust_handle)
            except Exception as e:
                logger.warning(f"Rust nve_build failed: {e}. Using Python-only path.")

        partition = self.profiler.partition(
            hot_fraction=self.config.tier.gpu_fraction,
            warm_fraction=self.config.tier.ram_fraction,
        )

        # Register sizes before initializing placement.
        sizes = {bid: info["size_bytes"] for bid, info in self._weight_blocks.items()}
        self.pager.initialize(partition, sizes=sizes)
        self._is_built = True

        # Set importance on pager blocks from profiling.
        ranking = dict(self.profiler.importance_ranking())
        for wid, importance in ranking.items():
            block = self.pager.get_block(wid)
            if block:
                block.importance = importance

        result = {
            "hot_weights": len(partition["hot"]),
            "warm_weights": len(partition["warm"]),
            "cold_weights": len(partition["cold"]),
            "total_weights": self.profiler.weight_count(),
            "co_activation_groups": len(self.pager._groups),
        }
        logger.info(f"Build complete: {result}")
        return result

    def infer(self, prompt: str, tokenizer=None, forward_fn=None):
        """
        Run inference with predictive weight paging.

        This is the core NVE inference loop:
        1. Predict which weight clusters will be needed (from pager state).
        2. Pre-page needed weights to GPU tier.
        3. Run forward pass with activation tracking.
        4. Update pager with observed access patterns.
        5. Adapt tier placements online if thresholds are met.
        """
        assert self._is_built, "Must call build() before inference"
        import torch

        self._inference_count += 1
        device = self.device_manager.best_device

        # Step 1: Predictive pre-paging.
        # If we have co-activation groups, page in likely-needed weights.
        if self.config.enable_prefetch and self.pager._groups:
            self._predictive_prepage()

        # Step 2: Install lightweight activation hooks for online tracking.
        hooks = []
        if self.config.enable_online_adaptation:
            hooks = self._install_lightweight_hooks()

        # Step 3: Run forward pass.
        result = None
        if forward_fn:
            result = forward_fn(self._model, prompt)
        elif tokenizer and self._model:
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: self.device_manager.safe_to(v, device) for k, v in inputs.items()}
                result = self._model(**inputs)

        # Step 4: Process online activations and update pager.
        if self.config.enable_online_adaptation and hasattr(self, "_online_activations"):
            for block_id, magnitude in self._online_activations.items():
                tier = self.pager.access(block_id)
                # If we're hitting SSD-tier weights frequently, try to promote.
                if tier == TierLevel.SSD:
                    self.pager.try_promote(block_id)
                elif tier == TierLevel.RAM:
                    self.pager.try_promote(block_id)
            self._online_activations.clear()

        # Step 5: Periodic adaptation — decay frequencies, re-evaluate placements.
        if (self.config.enable_online_adaptation and
                self._inference_count % self.config.adaptation_interval == 0):
            self.pager.decay_all_frequencies()
            self._adapt_placements()

        # Cleanup.
        for h in hooks:
            h.remove()

        if result is None:
            raise ValueError("Either forward_fn or (tokenizer + model) must be provided")
        return result

    def _predictive_prepage(self):
        """
        Pre-page weights predicted to be needed in the next inference.

        Uses the co-activation groups: if any member of a group was recently
        accessed, page in the whole group.
        """
        for group_idx, group in enumerate(self.pager._groups):
            # Check if any member was recently accessed.
            any_recent = False
            for wid in group:
                block = self.pager.get_block(wid)
                if block and block.recent_frequency > 0.1:
                    any_recent = True
                    break

            if any_recent:
                for wid in group:
                    block = self.pager.get_block(wid)
                    if block and block.tier != TierLevel.GPU:
                        self.pager.try_promote(wid)

    def _install_lightweight_hooks(self) -> list:
        """
        Install minimal hooks for online activation tracking.

        Lighter than profiling hooks — only records whether each weight
        was activated, not full magnitude stats.
        """
        self._online_activations = {}
        hooks = []

        if self._model is None:
            return hooks

        for block_id, info in self._weight_blocks.items():
            name = info["name"]
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                module_name, _ = parts
                try:
                    module = dict(self._model.named_modules())[module_name]

                    def make_hook(bid):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                output = output[0]
                            if hasattr(output, "abs"):
                                self._online_activations[bid] = float(
                                    output.abs().mean().item()
                                )
                        return hook_fn

                    h = module.register_forward_hook(make_hook(block_id))
                    hooks.append(h)
                except (KeyError, AttributeError):
                    pass

        return hooks

    def _adapt_placements(self):
        """
        Re-evaluate tier placements based on observed access patterns.

        Called periodically during inference. Demotes infrequently accessed
        GPU-tier weights and promotes frequently accessed lower-tier weights.
        """
        adapted = 0
        for block in list(self.pager._blocks.values()):
            # Demote cold GPU weights.
            if (block.tier == TierLevel.GPU and
                    block.home_tier != TierLevel.GPU and
                    block.recent_frequency < 0.01):
                self.pager.demote(block.weight_id)
                adapted += 1

            # Promote hot non-GPU weights.
            elif (block.tier != TierLevel.GPU and
                    block.recent_frequency > 0.5 and
                    block.access_count >= self.config.promotion_threshold):
                result = self.pager.try_promote(block.weight_id)
                if result is not None:
                    adapted += 1

        if adapted > 0:
            logger.info(f"Online adaptation: {adapted} weights re-tiered")

    def stats(self) -> dict:
        """Get engine statistics."""
        result = {
            "weights_registered": len(self._weight_blocks),
            "inference_count": self._inference_count,
            "profiler": {
                "weights_profiled": self.profiler.weight_count(),
                "rounds": self.profiler.total_rounds(),
                "is_stable": self.profiler.is_stable(),
            },
            "pager": self.pager.stats().to_dict() if self._is_built else None,
            "is_built": self._is_built,
            "rust_backend": self._rust_handle is not None,
            "device": str(self.device_manager.best_device),
            "co_activation_groups": len(self.pager._groups),
        }

        if self._rust_handle:
            result["rust_page_fault_rate"] = self._rust_lib.nve_page_fault_rate(
                self._rust_handle
            )
            result["rust_cluster_count"] = self._rust_lib.nve_cluster_count(
                self._rust_handle
            )

        return result

    def importance_ranking(self) -> list[tuple[int, float]]:
        return self.profiler.importance_ranking()

    def domain_ranking(self, domain: str) -> list[tuple[int, float]]:
        return self.profiler.domain_ranking(domain)

    def __del__(self):
        if self._rust_handle and self._rust_lib:
            self._rust_lib.nve_engine_free(self._rust_handle)
            self._rust_handle = None
