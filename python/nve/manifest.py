"""
Tier Manifest — maps profiler output to concrete parameter placement decisions.

Generates gpu_pages.json / ram_pages.json / ssd_pages.json manifests that
the serving layer uses at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PageEntry:
    """A single parameter's placement metadata."""
    param_name: str
    weight_id: int
    size_bytes: int
    importance: float
    domain_importances: dict[str, float] = field(default_factory=dict)
    layer_index: int = 0


@dataclass
class TierManifest:
    """Complete placement manifest for all three tiers."""
    gpu_pages: list[PageEntry] = field(default_factory=list)
    ram_pages: list[PageEntry] = field(default_factory=list)
    ssd_pages: list[PageEntry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def gpu_bytes(self) -> int:
        return sum(p.size_bytes for p in self.gpu_pages)

    @property
    def ram_bytes(self) -> int:
        return sum(p.size_bytes for p in self.ram_pages)

    @property
    def ssd_bytes(self) -> int:
        return sum(p.size_bytes for p in self.ssd_pages)

    def save(self, directory: str | Path):
        """Write manifest to directory as three JSON files + metadata."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        for name, pages in [
            ("gpu_pages.json", self.gpu_pages),
            ("ram_pages.json", self.ram_pages),
            ("ssd_pages.json", self.ssd_pages),
        ]:
            with open(d / name, "w") as f:
                json.dump([asdict(p) for p in pages], f, indent=2)

        with open(d / "manifest_meta.json", "w") as f:
            json.dump({
                "gpu_count": len(self.gpu_pages),
                "ram_count": len(self.ram_pages),
                "ssd_count": len(self.ssd_pages),
                "gpu_bytes": self.gpu_bytes,
                "ram_bytes": self.ram_bytes,
                "ssd_bytes": self.ssd_bytes,
                **self.metadata,
            }, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "TierManifest":
        """Load manifest from directory."""
        d = Path(directory)
        manifest = cls()

        for name, target in [
            ("gpu_pages.json", manifest.gpu_pages),
            ("ram_pages.json", manifest.ram_pages),
            ("ssd_pages.json", manifest.ssd_pages),
        ]:
            path = d / name
            if path.exists():
                with open(path) as f:
                    for entry in json.load(f):
                        target.append(PageEntry(**entry))

        meta_path = d / "manifest_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                manifest.metadata = json.load(f)

        return manifest

    def param_tier(self, param_name: str) -> str:
        """Look up which tier a parameter belongs to."""
        for p in self.gpu_pages:
            if p.param_name == param_name:
                return "gpu"
        for p in self.ram_pages:
            if p.param_name == param_name:
                return "ram"
        for p in self.ssd_pages:
            if p.param_name == param_name:
                return "ssd"
        return "unknown"

    def tier_for_layer(self, layer_idx: int) -> dict[str, str]:
        """Get tier assignments for all params in a given layer."""
        result = {}
        for pages, tier_name in [
            (self.gpu_pages, "gpu"),
            (self.ram_pages, "ram"),
            (self.ssd_pages, "ssd"),
        ]:
            for p in pages:
                if p.layer_index == layer_idx:
                    result[p.param_name] = tier_name
        return result


def build_manifest(
    profiler,
    weight_blocks: dict,
    gpu_fraction: float = 0.2,
    ram_fraction: float = 0.3,
    domains: Optional[list[str]] = None,
    profile_name: str = "default",
) -> TierManifest:
    """
    Build a tier manifest from profiler results and weight block metadata.

    Args:
        profiler: MCAPProfiler instance with completed profiling.
        weight_blocks: dict of {block_id: {"name": ..., "size_bytes": ..., "layer_index": ...}}
        gpu_fraction: Fraction of weights to place on GPU.
        ram_fraction: Fraction of weights to place in RAM.
        domains: Optional list of domains for domain importance tracking.
        profile_name: Name for this manifest profile.
    """
    partition = profiler.partition(hot_fraction=gpu_fraction, warm_fraction=ram_fraction)
    ranking = dict(profiler.importance_ranking())

    manifest = TierManifest(metadata={
        "profile_name": profile_name,
        "gpu_fraction": gpu_fraction,
        "ram_fraction": ram_fraction,
        "total_weights": profiler.weight_count(),
        "profiling_rounds": profiler.total_rounds(),
    })

    def make_entry(wid: int) -> PageEntry:
        info = weight_blocks[wid]
        domain_imps = {}
        if domains:
            for d in domains:
                stats = profiler.get_stats(wid)
                if stats:
                    domain_imps[d] = stats.domain_importance(d)
        return PageEntry(
            param_name=info["name"],
            weight_id=wid,
            size_bytes=info["size_bytes"],
            importance=ranking.get(wid, 0.0),
            domain_importances=domain_imps,
            layer_index=info.get("layer_index", 0),
        )

    for wid in partition["hot"]:
        manifest.gpu_pages.append(make_entry(wid))
    for wid in partition["warm"]:
        manifest.ram_pages.append(make_entry(wid))
    for wid in partition["cold"]:
        manifest.ssd_pages.append(make_entry(wid))

    return manifest
