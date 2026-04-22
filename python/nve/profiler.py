"""
Monte Carlo Activation Profiler (MCAP) — Python implementation.

Estimates weight importance via sampling: run diverse prompts through the model,
track activation magnitudes per weight, and compute importance as the Monte Carlo
expectation of activation.

    Î(Wᵢ) = (1/N) Σₖ activationₖ(Wᵢ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ActivationSample:
    """A single activation observation for a weight."""
    weight_id: int
    magnitude: float
    domain: Optional[str] = None


@dataclass
class WeightStats:
    """Accumulated importance statistics for a single weight."""
    weight_id: int
    total_activation: float = 0.0
    sample_count: int = 0
    max_activation: float = float("-inf")
    min_activation: float = float("inf")
    domain_activations: dict[str, float] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)

    def record(self, sample: ActivationSample):
        self.total_activation += sample.magnitude
        self.sample_count += 1
        self.max_activation = max(self.max_activation, sample.magnitude)
        self.min_activation = min(self.min_activation, sample.magnitude)

        if sample.domain:
            self.domain_activations[sample.domain] = (
                self.domain_activations.get(sample.domain, 0.0) + sample.magnitude
            )
            self.domain_counts[sample.domain] = (
                self.domain_counts.get(sample.domain, 0) + 1
            )

    @property
    def importance(self) -> float:
        """Monte Carlo estimate of importance: E[activation]."""
        if self.sample_count == 0:
            return 0.0
        return self.total_activation / self.sample_count

    def domain_importance(self, domain: str) -> float:
        total = self.domain_activations.get(domain, 0.0)
        count = self.domain_counts.get(domain, 0)
        if count == 0:
            return 0.0
        return total / count


@dataclass
class PromptDomain:
    """A domain of prompts with a sampling weight."""
    name: str
    weight: float
    prompts: list[str]


class PromptDistribution:
    """Weighted distribution of prompts across domains for profiling."""

    def __init__(self, domains: list[PromptDomain]):
        self.domains = domains
        total = sum(d.weight for d in domains)
        self._probs = [d.weight / total for d in domains]

    def sample(self, count: int) -> list[tuple[str, str]]:
        """Sample (domain_name, prompt) pairs."""
        rng = np.random.default_rng()
        result = []
        for _ in range(count):
            domain_idx = rng.choice(len(self.domains), p=self._probs)
            domain = self.domains[domain_idx]
            prompt = rng.choice(domain.prompts)
            result.append((domain.name, prompt))
        return result


class MCAPProfiler:
    """
    Monte Carlo Activation Profiler.

    Accumulates activation samples and computes importance rankings.
    Supports domain-aware profiling and online EMA updates.
    """

    def __init__(
        self,
        samples_per_round: int = 100,
        min_samples_for_stability: int = 50,
        ema_decay: float = 0.01,
    ):
        self.samples_per_round = samples_per_round
        self.min_samples_for_stability = min_samples_for_stability
        self.ema_decay = ema_decay
        self._weights: dict[int, WeightStats] = {}
        self._rounds = 0
        self._active = False

    def start(self):
        self._active = True

    def record(self, sample: ActivationSample):
        if sample.weight_id not in self._weights:
            self._weights[sample.weight_id] = WeightStats(weight_id=sample.weight_id)
        self._weights[sample.weight_id].record(sample)

    def record_batch(self, samples: list[ActivationSample]):
        for sample in samples:
            self.record(sample)

    def finish_round(self):
        self._rounds += 1

    def importance_ranking(self) -> list[tuple[int, float]]:
        """Get all weights ranked by importance (descending)."""
        ranking = [(wid, ws.importance) for wid, ws in self._weights.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def domain_ranking(self, domain: str) -> list[tuple[int, float]]:
        """Get weights ranked by domain-specific importance."""
        ranking = [
            (wid, ws.domain_importance(domain)) for wid, ws in self._weights.items()
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def partition(
        self, hot_fraction: float = 0.2, warm_fraction: float = 0.3
    ) -> dict[str, list[int]]:
        """Partition weights into hot/warm/cold tiers."""
        ranking = self.importance_ranking()
        n = len(ranking)
        hot_cutoff = int(np.ceil(n * hot_fraction))
        warm_cutoff = hot_cutoff + int(np.ceil(n * warm_fraction))

        return {
            "hot": [wid for wid, _ in ranking[:hot_cutoff]],
            "warm": [wid for wid, _ in ranking[hot_cutoff:warm_cutoff]],
            "cold": [wid for wid, _ in ranking[warm_cutoff:]],
        }

    def is_stable(self) -> bool:
        """Check if all weights have enough samples for stable estimates."""
        return all(
            ws.sample_count >= self.min_samples_for_stability
            for ws in self._weights.values()
        )

    def weight_count(self) -> int:
        return len(self._weights)

    def total_rounds(self) -> int:
        return self._rounds

    def get_stats(self, weight_id: int) -> Optional[WeightStats]:
        return self._weights.get(weight_id)
