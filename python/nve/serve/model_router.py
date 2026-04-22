"""
Model router — routing policies, aliases, versioning, and A/B traffic splitting.

Sits between handlers and ModelStore.  Given a request's `model` field (which
may be a logical name, alias, or version tag), the router resolves it to a
concrete loaded model name and returns the matching ModelRecord.

Routing policies
─────────────────
  first       Default: use the first model whose name matches (insertion order).
  round_robin Cycle through all models that match the selector.
  least_loaded  Pick the model with the fewest active workers + smallest queue.
  random      Random among matching models.
  weighted    Traffic split: {"model-a": 70, "model-b": 30} (percent weights).

Selectors
──────────
  exact name   "llama-3b"       → route to exactly that model
  alias        "default"        → any alias pointing to a concrete model
  prefix glob  "llama-*"        → any loaded model whose name matches the glob
  version tag  "llama-3b:v2"   → model registered under that version key
  None / ""    → apply policy across all loaded models

Model versioning
─────────────────
  router.register_version("llama-3b", version="v1", name="llama-3b-q4")
  router.register_version("llama-3b", version="v2", name="llama-3b-q8")
  # Request with model="llama-3b:v1" routes to "llama-3b-q4"
  # Request with model="llama-3b:latest" routes to the last registered version

A/B traffic splitting
──────────────────────
  router.set_weights({"llama-3b-q4": 80, "llama-3b-q8": 20})
  # 80% of requests to llama-3b-q4, 20% to llama-3b-q8

Aliases
────────
  router.add_alias("default", "llama-3b")
  router.add_alias("fast", "llama-3b-q4")
  # Request with model="default" routes to "llama-3b"
"""

from __future__ import annotations

import fnmatch
import logging
import random
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("nve.serve.model_router")


# ── Routing policies ──────────────────────────────────────────────────────────

class RoutingPolicy:
    FIRST        = "first"
    ROUND_ROBIN  = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM       = "random"
    WEIGHTED     = "weighted"

    ALL = {FIRST, ROUND_ROBIN, LEAST_LOADED, RANDOM, WEIGHTED}


class ModelRouter:
    """
    Thread-safe router that resolves logical model names to loaded model records.

    Works alongside ModelStore — the router holds the routing metadata
    (aliases, versions, weights, policy) while ModelStore holds the actual
    backends.
    """

    def __init__(
        self,
        model_store,               # ModelStore instance
        policy: str = RoutingPolicy.FIRST,
    ) -> None:
        if policy not in RoutingPolicy.ALL:
            raise ValueError(f"Unknown routing policy: {policy!r}. Choose from {RoutingPolicy.ALL}")
        self._store = model_store
        self._policy = policy
        self._lock = threading.RLock()

        # alias → concrete model name
        self._aliases: Dict[str, str] = {}
        # (base_name, version) → concrete model name
        self._versions: Dict[Tuple[str, str], str] = {}
        # base_name → list of versions in registration order
        self._version_order: Dict[str, List[str]] = defaultdict(list)
        # concrete model name → traffic weight (for WEIGHTED policy)
        self._weights: Dict[str, float] = {}
        # round-robin counters per selector
        self._rr_counters: Dict[str, int] = defaultdict(int)

    # ── Alias management ──────────────────────────────────────────────────────

    def add_alias(self, alias: str, target: str) -> None:
        """Map `alias` → `target` model name."""
        with self._lock:
            self._aliases[alias] = target
            logger.info(f"Alias '{alias}' → '{target}'")

    def remove_alias(self, alias: str) -> bool:
        with self._lock:
            if alias in self._aliases:
                del self._aliases[alias]
                return True
            return False

    def list_aliases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._aliases)

    # ── Version management ────────────────────────────────────────────────────

    def register_version(self, base_name: str, version: str, concrete_name: str) -> None:
        """
        Register `concrete_name` as `base_name:version`.

        After this call, requests with `model="base_name:version"` route to
        `concrete_name`.  `model="base_name:latest"` always routes to the
        most recently registered version.

        Example:
            router.register_version("llama", "v1", "llama-q4")
            router.register_version("llama", "v2", "llama-q8")
            # model="llama:latest" → "llama-q8"
            # model="llama:v1"    → "llama-q4"
        """
        with self._lock:
            key = (base_name, version)
            self._versions[key] = concrete_name
            order = self._version_order[base_name]
            if version not in order:
                order.append(version)
            logger.info(f"Version '{base_name}:{version}' → '{concrete_name}'")

    def deregister_version(self, base_name: str, version: str) -> bool:
        with self._lock:
            key = (base_name, version)
            if key not in self._versions:
                return False
            del self._versions[key]
            if version in self._version_order.get(base_name, []):
                self._version_order[base_name].remove(version)
            return True

    def list_versions(self, base_name: str) -> Dict[str, str]:
        """Return {version: concrete_name} for `base_name`."""
        with self._lock:
            return {
                v: self._versions[(base_name, v)]
                for v in self._version_order.get(base_name, [])
                if (base_name, v) in self._versions
            }

    # ── Traffic weights ────────────────────────────────────────────────────────

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set traffic weights for WEIGHTED policy.

        weights: {concrete_model_name: percent}
        Weights need not sum to 100 — they are normalised automatically.

        Example:
            router.set_weights({"llama-q4": 80, "llama-q8": 20})
        """
        if not weights:
            raise ValueError("weights must be non-empty")
        if any(v < 0 for v in weights.values()):
            raise ValueError("All weights must be >= 0")
        with self._lock:
            self._weights = dict(weights)
            logger.info(f"Traffic weights set: {weights}")

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._weights)

    # ── Policy ────────────────────────────────────────────────────────────────

    def set_policy(self, policy: str) -> None:
        if policy not in RoutingPolicy.ALL:
            raise ValueError(f"Unknown policy: {policy!r}")
        with self._lock:
            self._policy = policy
            logger.info(f"Routing policy set to '{policy}'")

    def get_policy(self) -> str:
        return self._policy

    # ── Resolution ────────────────────────────────────────────────────────────

    def resolve(self, model_selector: Optional[str]) -> Optional[str]:
        """
        Resolve a model selector to a concrete loaded model name.

        Returns None if no matching model is loaded.

        Selector precedence:
          1. Explicit concrete name (exact match in ModelStore)
          2. Alias lookup
          3. Version tag "base:version" or "base:latest"
          4. Glob pattern "prefix-*" matched against loaded models
          5. None / "" → apply policy across all loaded models
        """
        with self._lock:
            return self._resolve_locked(model_selector)

    def _resolve_locked(self, selector: Optional[str]) -> Optional[str]:
        loaded = set(m["name"] for m in self._store.list_models())
        if not loaded:
            return None

        # 1. Direct match
        if selector and selector in loaded:
            return selector

        # 2. Alias
        if selector and selector in self._aliases:
            target = self._aliases[selector]
            if target in loaded:
                return target
            logger.warning(f"Alias '{selector}' → '{target}' but '{target}' is not loaded")
            return None

        # 3. Version tag  "base:version"  or  "base:latest"
        if selector and ":" in selector:
            base, version = selector.split(":", 1)
            concrete = self._resolve_version(base, version, loaded)
            if concrete:
                return concrete
            logger.warning(f"Version '{selector}' not found among loaded models")
            return None

        # 4. Glob pattern
        if selector and ("*" in selector or "?" in selector):
            candidates = [n for n in sorted(loaded) if fnmatch.fnmatch(n, selector)]
            return self._apply_policy(candidates, selector) if candidates else None

        # 5. No selector → all loaded models
        if not selector:
            candidates = sorted(loaded)
            return self._apply_policy(candidates, "__all__") if candidates else None

        # Not found
        return None

    def _resolve_version(self, base: str, version: str, loaded: set) -> Optional[str]:
        if version == "latest":
            order = self._version_order.get(base, [])
            for v in reversed(order):
                concrete = self._versions.get((base, v))
                if concrete and concrete in loaded:
                    return concrete
            return None
        concrete = self._versions.get((base, version))
        return concrete if concrete and concrete in loaded else None

    def _apply_policy(self, candidates: List[str], selector_key: str) -> Optional[str]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if self._policy == RoutingPolicy.FIRST:
            return candidates[0]

        if self._policy == RoutingPolicy.RANDOM:
            return random.choice(candidates)

        if self._policy == RoutingPolicy.ROUND_ROBIN:
            idx = self._rr_counters[selector_key] % len(candidates)
            self._rr_counters[selector_key] += 1
            return candidates[idx]

        if self._policy == RoutingPolicy.LEAST_LOADED:
            return self._least_loaded(candidates)

        if self._policy == RoutingPolicy.WEIGHTED:
            return self._weighted_choice(candidates)

        return candidates[0]

    def _least_loaded(self, candidates: List[str]) -> str:
        """Pick the candidate with fewest active_workers + queue_depth."""
        best_name = candidates[0]
        best_load = float("inf")
        for name in candidates:
            record = self._store.get(name)
            if record is None:
                continue
            pool = record.worker_pool
            load = pool.active_count() + pool.queue_depth()
            if load < best_load:
                best_load = load
                best_name = name
        return best_name

    def _weighted_choice(self, candidates: List[str]) -> str:
        """Sample one candidate proportionally to configured weights."""
        weights_in_scope = {
            name: self._weights.get(name, 1.0)
            for name in candidates
        }
        total = sum(weights_in_scope.values())
        if total <= 0:
            return random.choice(candidates)
        r = random.uniform(0, total)
        cumulative = 0.0
        for name, w in weights_in_scope.items():
            cumulative += w
            if r <= cumulative:
                return name
        return candidates[-1]

    # ── Introspection ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a JSON-serialisable snapshot of router state."""
        with self._lock:
            loaded = [m["name"] for m in self._store.list_models()]
            return {
                "policy": self._policy,
                "loaded_models": loaded,
                "aliases": dict(self._aliases),
                "versions": {
                    f"{base}:{v}": concrete
                    for (base, v), concrete in self._versions.items()
                },
                "weights": dict(self._weights),
            }
