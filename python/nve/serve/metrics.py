"""
Prometheus-compatible metrics collector.

Exposes GET /metrics in the standard text format that Prometheus can scrape.
No external dependencies — all counters/gauges tracked in-process.

Metrics exposed
───────────────
Counters
  nve_requests_total{model,endpoint}
  nve_errors_total{model,type}
  nve_prompt_tokens_total{model}
  nve_generated_tokens_total{model}

Gauges
  nve_uptime_seconds
  nve_model_loaded{model}
  nve_queue_depth{model}
  nve_active_workers{model}
  nve_worker_utilization{model}    — fraction 0.0–1.0

Summaries (p50/p95/p99)
  nve_request_latency_seconds{model}
  nve_tokens_per_second{model}
  nve_queue_wait_seconds{model}    — time spent waiting in priority queue
  nve_inference_seconds{model}     — pure inference time (backend call)
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List


class MetricsCollector:
    """Thread-safe metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._start_time = time.time()

    # ── Write ──────────────────────────────────────────────────────────────────

    def inc(self, name: str, value: float = 1.0, labels: str = "") -> None:
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: str = "") -> None:
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: str = "") -> None:
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._histograms[key].append(value)

    # ── Convenience helpers ────────────────────────────────────────────────────

    def record_request(self, model: str, endpoint: str) -> None:
        self.inc("nve_requests_total", labels=f'model="{model}",endpoint="{endpoint}"')

    def record_tokens(self, model: str, prompt_tokens: int, gen_tokens: int) -> None:
        self.inc("nve_prompt_tokens_total", prompt_tokens, labels=f'model="{model}"')
        self.inc("nve_generated_tokens_total", gen_tokens, labels=f'model="{model}"')

    def record_latency(self, model: str, latency_s: float) -> None:
        self.observe("nve_request_latency_seconds", latency_s, labels=f'model="{model}"')

    def record_tps(self, model: str, tps: float) -> None:
        self.observe("nve_tokens_per_second", tps, labels=f'model="{model}"')

    def record_error(self, model: str, error_type: str) -> None:
        self.inc("nve_errors_total", labels=f'model="{model}",type="{error_type}"')

    def record_queue_wait(self, model: str, wait_s: float) -> None:
        self.observe("nve_queue_wait_seconds", wait_s, labels=f'model="{model}"')

    def record_inference_time(self, model: str, inference_s: float) -> None:
        self.observe("nve_inference_seconds", inference_s, labels=f'model="{model}"')

    def set_model_loaded(self, model: str, loaded: bool) -> None:
        self.set_gauge("nve_model_loaded", 1.0 if loaded else 0.0, labels=f'model="{model}"')

    def set_queue_depth(self, model: str, depth: int) -> None:
        self.set_gauge("nve_queue_depth", float(depth), labels=f'model="{model}"')

    def set_active_workers(self, model: str, active: int, total: int) -> None:
        self.set_gauge("nve_active_workers", float(active), labels=f'model="{model}"')
        utilization = (active / total) if total > 0 else 0.0
        self.set_gauge("nve_worker_utilization", utilization, labels=f'model="{model}"')

    # ── Read ───────────────────────────────────────────────────────────────────

    def _histogram_stats(self, values: deque) -> dict:
        if not values:
            return {"count": 0, "sum": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        return {
            "count": n,
            "sum": total,
            "p50": sorted_vals[int(n * 0.50)],
            "p95": sorted_vals[int(n * 0.95)],
            "p99": sorted_vals[int(n * 0.99)],
        }

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text format."""
        lines: List[str] = []
        uptime = time.time() - self._start_time

        with self._lock:
            lines.append("# HELP nve_uptime_seconds Server uptime in seconds")
            lines.append("# TYPE nve_uptime_seconds gauge")
            lines.append(f"nve_uptime_seconds {uptime:.2f}")

            for key, val in sorted(self._counters.items()):
                base = key.split("{")[0]
                lines.append(f"# TYPE {base} counter")
                lines.append(f"{key} {val}")

            for key, val in sorted(self._gauges.items()):
                base = key.split("{")[0]
                lines.append(f"# TYPE {base} gauge")
                lines.append(f"{key} {val}")

            for key, vals in sorted(self._histograms.items()):
                base = key.split("{")[0]
                label_part = key[len(base):]
                stats = self._histogram_stats(vals)
                lines.append(f"# TYPE {base} summary")
                for q_name, q_val in [
                    ("0.5", stats["p50"]),
                    ("0.95", stats["p95"]),
                    ("0.99", stats["p99"]),
                ]:
                    if label_part:
                        ql = label_part[:-1] + f',quantile="{q_name}"' + "}"
                    else:
                        ql = f'{{quantile="{q_name}"}}'
                    lines.append(f"{base}{ql} {q_val:.6f}")
                lines.append(f"{base}_count{label_part} {stats['count']}")
                lines.append(f"{base}_sum{label_part} {stats['sum']:.6f}")

        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            return {
                "uptime_s": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self._histogram_stats(v)
                    for k, v in self._histograms.items()
                },
            }
