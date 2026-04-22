"""
Weight Pager — production-grade tier management and paging logic.

Manages the virtual memory abstraction: weight blocks are assigned to
GPU/RAM/SSD tiers and migrated dynamically based on access patterns.

Production features:
- Online frequency tracking with exponential decay
- Memory pressure-aware promotion/demotion
- Budget enforcement with hard limits
- Batch paging (migrate co-activated weights together)
- Integration with DeviceManager for OOM-safe transfers
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger("nve.pager")


class TierLevel(IntEnum):
    GPU = 0
    RAM = 1
    SSD = 2


@dataclass
class PagerStats:
    """Runtime paging statistics."""
    page_hits: int = 0
    page_faults: int = 0
    migrations: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    gpu_usage_bytes: int = 0
    ram_usage_bytes: int = 0
    ssd_usage_bytes: int = 0
    gpu_capacity_bytes: int = 0
    ram_capacity_bytes: int = 0
    ssd_capacity_bytes: int = 0

    @property
    def fault_rate(self) -> float:
        total = self.page_hits + self.page_faults
        if total == 0:
            return 0.0
        return self.page_faults / total

    @property
    def gpu_utilization(self) -> float:
        if self.gpu_capacity_bytes == 0:
            return 0.0
        return self.gpu_usage_bytes / self.gpu_capacity_bytes

    def to_dict(self) -> dict:
        return {
            "page_hits": self.page_hits,
            "page_faults": self.page_faults,
            "fault_rate": round(self.fault_rate, 4),
            "migrations": self.migrations,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "evictions": self.evictions,
            "gpu_utilization": round(self.gpu_utilization, 4),
            "gpu_usage_mb": round(self.gpu_usage_bytes / (1024**2), 1),
            "ram_usage_mb": round(self.ram_usage_bytes / (1024**2), 1),
            "ssd_usage_mb": round(self.ssd_usage_bytes / (1024**2), 1),
        }


@dataclass
class WeightBlock:
    """Metadata for a single weight block in the pager."""
    weight_id: int
    name: str = ""
    size_bytes: int = 0
    layer_index: int = 0
    tier: TierLevel = TierLevel.SSD
    home_tier: TierLevel = TierLevel.SSD  # Where it belongs (from profiling).
    importance: float = 0.0

    # Online tracking (updated during inference).
    access_count: int = 0
    last_access_tick: int = 0
    recent_frequency: float = 0.0  # EMA of access frequency.
    last_access_time: float = 0.0


class WeightPager:
    """
    Manages weight block placement across memory tiers.

    Implements:
    - LRU eviction when tiers are full
    - Frequency-based promotion (hot weights bubble up)
    - Memory pressure-aware demotion (under pressure, cold weights evicted)
    - Batch operations for co-activated weight groups
    - Hard budget enforcement
    """

    def __init__(
        self,
        gpu_bytes: int,
        ram_bytes: int,
        ssd_bytes: int,
        gpu_fraction: float = 0.2,
        ram_fraction: float = 0.3,
        # Online adaptation parameters.
        frequency_decay: float = 0.95,    # EMA decay for access frequency.
        promotion_threshold: int = 5,      # Accesses before considering promotion.
        demotion_cooldown_s: float = 1.0,  # Min time between demotions of same weight.
    ):
        self.gpu_capacity = gpu_bytes
        self.ram_capacity = ram_bytes
        self.ssd_capacity = ssd_bytes
        self.gpu_fraction = gpu_fraction
        self.ram_fraction = ram_fraction
        self.frequency_decay = frequency_decay
        self.promotion_threshold = promotion_threshold
        self.demotion_cooldown = demotion_cooldown_s

        self._blocks: dict[int, WeightBlock] = {}
        self._name_to_id: dict[str, int] = {}
        self._usage = {TierLevel.GPU: 0, TierLevel.RAM: 0, TierLevel.SSD: 0}
        self._tick = 0
        self._stats = PagerStats(
            gpu_capacity_bytes=gpu_bytes,
            ram_capacity_bytes=ram_bytes,
            ssd_capacity_bytes=ssd_bytes,
        )

        # Co-activation groups: sets of weight IDs that should be paged together.
        self._groups: list[set[int]] = []
        self._weight_to_group: dict[int, int] = {}  # weight_id -> group index

    # ── Initialization ──

    def register(
        self,
        weight_id: int,
        name: str = "",
        size_bytes: int = 0,
        layer_index: int = 0,
        importance: float = 0.0,
    ):
        """Register a weight block with the pager."""
        block = WeightBlock(
            weight_id=weight_id,
            name=name,
            size_bytes=size_bytes,
            layer_index=layer_index,
            importance=importance,
        )
        self._blocks[weight_id] = block
        if name:
            self._name_to_id[name] = weight_id

    def initialize(
        self,
        partition: dict[str, list[int]],
        sizes: Optional[dict[int, int]] = None,
    ):
        """
        Initialize tier placement from a profiler partition.

        Args:
            partition: {"hot": [...], "warm": [...], "cold": [...]} weight IDs.
            sizes: Optional weight_id -> size mapping.
        """
        all_ids = partition["hot"] + partition["warm"] + partition["cold"]

        # Ensure all weights are registered.
        for wid in all_ids:
            if wid not in self._blocks:
                self._blocks[wid] = WeightBlock(weight_id=wid)

        if sizes:
            for wid, size in sizes.items():
                if wid in self._blocks:
                    self._blocks[wid].size_bytes = size
        else:
            total = self.gpu_capacity + self.ram_capacity + self.ssd_capacity
            per_weight = total // max(len(all_ids), 1)
            for wid in all_ids:
                self._blocks[wid].size_bytes = per_weight

        for wid in partition["hot"]:
            self._place(wid, TierLevel.GPU)
            self._blocks[wid].home_tier = TierLevel.GPU
        for wid in partition["warm"]:
            self._place(wid, TierLevel.RAM)
            self._blocks[wid].home_tier = TierLevel.RAM
        for wid in partition["cold"]:
            self._place(wid, TierLevel.SSD)
            self._blocks[wid].home_tier = TierLevel.SSD

    def set_co_activation_groups(self, groups: list[set[int]]):
        """
        Define groups of weights that should be paged together.

        From profiling: weights that co-activate (appear in the same layer
        or same attention pattern) benefit from batch paging.
        """
        self._groups = groups
        self._weight_to_group = {}
        for i, group in enumerate(groups):
            for wid in group:
                self._weight_to_group[wid] = i

    # ── Placement ──

    def _place(self, weight_id: int, tier: TierLevel) -> TierLevel:
        """Place a weight at the given tier, falling back to slower tiers if full."""
        block = self._blocks.get(weight_id)
        if block is None:
            return TierLevel.SSD

        size = block.size_bytes
        capacity = self._tier_capacity(tier)

        if self._usage[tier] + size <= capacity:
            block.tier = tier
            self._usage[tier] += size
            return tier

        # Fall back to slower tier.
        if tier == TierLevel.GPU:
            return self._place(weight_id, TierLevel.RAM)
        elif tier == TierLevel.RAM:
            return self._place(weight_id, TierLevel.SSD)
        else:
            block.tier = TierLevel.SSD
            self._usage[TierLevel.SSD] += size
            return TierLevel.SSD

    def _tier_capacity(self, tier: TierLevel) -> int:
        if tier == TierLevel.GPU:
            return self.gpu_capacity
        elif tier == TierLevel.RAM:
            return self.ram_capacity
        return self.ssd_capacity

    # ── Online Access Tracking ──

    def access(self, weight_id: int) -> TierLevel:
        """
        Record an access to a weight during inference.

        Updates frequency tracking and returns current tier.
        This is the hot path — called for every weight load.
        """
        self._tick += 1
        block = self._blocks.get(weight_id)
        if block is None:
            return TierLevel.SSD

        block.access_count += 1
        block.last_access_tick = self._tick
        block.last_access_time = time.monotonic()

        # Update EMA frequency.
        block.recent_frequency = (
            self.frequency_decay * block.recent_frequency + (1 - self.frequency_decay)
        )

        if block.tier == TierLevel.GPU:
            self._stats.page_hits += 1
        else:
            self._stats.page_faults += 1

        return block.tier

    def access_by_name(self, name: str) -> TierLevel:
        """Access a weight by parameter name."""
        wid = self._name_to_id.get(name)
        if wid is None:
            return TierLevel.SSD
        return self.access(wid)

    def decay_all_frequencies(self):
        """
        Decay all frequency trackers.

        Call this periodically (e.g., every N inference steps) to let
        cold weights' frequencies drop, enabling demotion.
        """
        for block in self._blocks.values():
            block.recent_frequency *= self.frequency_decay

    # ── Promotion / Demotion ──

    def try_promote(self, weight_id: int) -> Optional[TierLevel]:
        """
        Try to promote a frequently-accessed weight to a faster tier.

        Only promotes if:
        1. Weight has been accessed enough times (promotion_threshold).
        2. Target tier has space (or we can evict something).
        """
        block = self._blocks.get(weight_id)
        if block is None or block.tier == TierLevel.GPU:
            return None

        if block.access_count < self.promotion_threshold:
            return None

        target = TierLevel(block.tier - 1)
        size = block.size_bytes
        capacity = self._tier_capacity(target)

        # Direct promotion if space available.
        if self._usage[target] + size <= capacity:
            self._usage[block.tier] -= size
            self._usage[target] += size
            block.tier = target
            self._stats.migrations += 1
            self._stats.promotions += 1
            logger.debug(f"Promoted {block.name or weight_id} to {target.name}")
            return target

        # Try evicting the least-recently-used weight from target tier.
        evicted = self._evict_lru_from(target)
        if evicted is not None:
            # Retry after eviction.
            if self._usage[target] + size <= capacity:
                self._usage[block.tier] -= size
                self._usage[target] += size
                block.tier = target
                self._stats.migrations += 1
                self._stats.promotions += 1
                return target

        return None

    def try_promote_by_name(self, name: str) -> Optional[TierLevel]:
        """Promote a weight by parameter name."""
        wid = self._name_to_id.get(name)
        if wid is None:
            return None
        return self.try_promote(wid)

    def demote(self, weight_id: int) -> Optional[TierLevel]:
        """Demote a weight to a slower tier."""
        block = self._blocks.get(weight_id)
        if block is None or block.tier == TierLevel.SSD:
            return None

        target = TierLevel(block.tier + 1)
        size = block.size_bytes

        self._usage[block.tier] -= size
        self._usage[target] += size
        block.tier = target
        self._stats.migrations += 1
        self._stats.demotions += 1
        return target

    # ── Eviction ──

    def _evict_lru_from(self, tier: TierLevel, force: bool = False) -> Optional[int]:
        """
        Evict the least recently used weight from a tier.

        Considers both recency and frequency — a weight that's accessed
        rarely but recently won't be evicted over one that hasn't been
        accessed in a while.

        Args:
            force: If True, can evict even home-GPU weights (for pressure relief).
        """
        if force:
            candidates = [
                block for block in self._blocks.values()
                if block.tier == tier
            ]
        else:
            candidates = [
                block for block in self._blocks.values()
                if block.tier == tier and block.home_tier != TierLevel.GPU
            ]
        if not candidates:
            return None

        # Score: lower = better eviction candidate.
        # Combine recency (tick) and frequency (EMA).
        def eviction_score(b: WeightBlock) -> float:
            recency = b.last_access_tick / max(self._tick, 1)
            return recency * 0.5 + b.recent_frequency * 0.5

        victim = min(candidates, key=eviction_score)
        self.demote(victim.weight_id)
        self._stats.evictions += 1
        return victim.weight_id

    def evict_lru_from_gpu(self) -> Optional[int]:
        """Evict the least recently used weight from GPU tier."""
        return self._evict_lru_from(TierLevel.GPU)

    def evict_under_pressure(self, target_free_bytes: int, tier: TierLevel = TierLevel.GPU):
        """
        Evict weights from a tier until target_free_bytes are available.

        Used by DeviceManager when memory pressure is detected.
        """
        freed = 0
        max_evictions = len(self._blocks)  # Safety bound.

        for _ in range(max_evictions):
            capacity = self._tier_capacity(tier)
            current_free = capacity - self._usage[tier]
            if current_free >= target_free_bytes:
                break

            evicted = self._evict_lru_from(tier, force=True)
            if evicted is None:
                break

            freed += self._blocks[evicted].size_bytes

        logger.info(
            f"Pressure eviction: freed {freed / 1024**2:.1f} MB from {tier.name} tier"
        )
        return freed

    # ── Batch Operations ──

    def page_in_group(self, weight_id: int) -> list[int]:
        """
        Page in a weight and all its co-activated group members.

        Returns list of weight IDs that were paged in.
        """
        group_idx = self._weight_to_group.get(weight_id)
        if group_idx is None:
            return [weight_id]

        group = self._groups[group_idx]
        paged = []
        for wid in group:
            block = self._blocks.get(wid)
            if block and block.tier != TierLevel.GPU:
                promoted = self.try_promote(wid)
                if promoted is not None:
                    paged.append(wid)

        return paged if paged else [weight_id]

    def page_in_layer(self, layer_index: int) -> list[int]:
        """
        Page in all weights for a given layer.

        Used by the serving layer's pre-forward hook.
        Returns list of weight IDs that were paged in.
        """
        paged = []
        for block in self._blocks.values():
            if block.layer_index == layer_index and block.tier != TierLevel.GPU:
                old_tier = block.tier
                promoted = self.try_promote(block.weight_id)
                if promoted is not None:
                    paged.append(block.weight_id)
                # Record access regardless of promotion success.
                self.access(block.weight_id)
            elif block.layer_index == layer_index:
                self.access(block.weight_id)
        return paged

    # ── Query ──

    def get_tier(self, weight_id: int) -> TierLevel:
        block = self._blocks.get(weight_id)
        return block.tier if block else TierLevel.SSD

    def get_tier_by_name(self, name: str) -> TierLevel:
        wid = self._name_to_id.get(name)
        if wid is None:
            return TierLevel.SSD
        return self.get_tier(wid)

    def get_block(self, weight_id: int) -> Optional[WeightBlock]:
        return self._blocks.get(weight_id)

    def weights_at_tier(self, tier: TierLevel) -> list[int]:
        return [b.weight_id for b in self._blocks.values() if b.tier == tier]

    def weights_for_layer(self, layer_index: int) -> list[WeightBlock]:
        return [b for b in self._blocks.values() if b.layer_index == layer_index]

    def tier_usage(self) -> dict[str, dict[str, int]]:
        """Get current tier usage summary."""
        return {
            "gpu": {
                "used": self._usage[TierLevel.GPU],
                "capacity": self.gpu_capacity,
                "count": len(self.weights_at_tier(TierLevel.GPU)),
            },
            "ram": {
                "used": self._usage[TierLevel.RAM],
                "capacity": self.ram_capacity,
                "count": len(self.weights_at_tier(TierLevel.RAM)),
            },
            "ssd": {
                "used": self._usage[TierLevel.SSD],
                "capacity": self.ssd_capacity,
                "count": len(self.weights_at_tier(TierLevel.SSD)),
            },
        }

    def update_budgets(self, gpu_bytes: int = 0, ram_bytes: int = 0, ssd_bytes: int = 0):
        """
        Update tier budgets dynamically (e.g., after DeviceManager detects
        memory pressure or freed memory).
        """
        if gpu_bytes > 0:
            self.gpu_capacity = gpu_bytes
            self._stats.gpu_capacity_bytes = gpu_bytes
        if ram_bytes > 0:
            self.ram_capacity = ram_bytes
            self._stats.ram_capacity_bytes = ram_bytes
        if ssd_bytes > 0:
            self.ssd_capacity = ssd_bytes
            self._stats.ssd_capacity_bytes = ssd_bytes

        # If any tier is now over capacity, force-evict.
        for tier in [TierLevel.GPU, TierLevel.RAM]:
            cap = self._tier_capacity(tier)
            while self._usage[tier] > cap:
                evicted = self._evict_lru_from(tier, force=True)
                if evicted is None:
                    break

    def stats(self) -> PagerStats:
        self._stats.gpu_usage_bytes = self._usage[TierLevel.GPU]
        self._stats.ram_usage_bytes = self._usage[TierLevel.RAM]
        self._stats.ssd_usage_bytes = self._usage[TierLevel.SSD]
        return self._stats

    def summary(self) -> str:
        """Human-readable summary of pager state."""
        s = self.stats()
        total = len(self._blocks)
        gpu_count = len(self.weights_at_tier(TierLevel.GPU))
        ram_count = len(self.weights_at_tier(TierLevel.RAM))
        ssd_count = len(self.weights_at_tier(TierLevel.SSD))
        return (
            f"Pager: {total} weights | "
            f"GPU: {gpu_count} ({s.gpu_usage_bytes / 1024**2:.0f} MB / "
            f"{self.gpu_capacity / 1024**2:.0f} MB) | "
            f"RAM: {ram_count} ({s.ram_usage_bytes / 1024**2:.0f} MB / "
            f"{self.ram_capacity / 1024**2:.0f} MB) | "
            f"SSD: {ssd_count} ({s.ssd_usage_bytes / 1024**2:.0f} MB) | "
            f"Fault rate: {s.fault_rate:.1%}"
        )
