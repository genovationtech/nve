"""Tests for MetricsCollector."""

import time
import pytest
from nve.serve.metrics import MetricsCollector


def test_counter_increments():
    m = MetricsCollector()
    m.inc("req_total")
    m.inc("req_total", 4)
    assert m._counters["req_total"] == 5


def test_gauge_set():
    m = MetricsCollector()
    m.set_gauge("queue_depth", 7.0)
    assert m._gauges["queue_depth"] == 7.0
    m.set_gauge("queue_depth", 2.0)
    assert m._gauges["queue_depth"] == 2.0


def test_histogram_observe():
    m = MetricsCollector()
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        m.observe("latency", v)
    assert len(m._histograms["latency"]) == 5


def test_labels():
    m = MetricsCollector()
    m.inc("req", labels='model="gpt2"')
    assert m._counters['req{model="gpt2"}'] == 1


def test_prometheus_render():
    m = MetricsCollector()
    m.inc("nve_requests_total", labels='model="gpt2",endpoint="generate"')
    m.set_gauge("nve_queue_depth", 3.0)
    m.observe("nve_latency", 0.5)
    text = m.render_prometheus()
    assert "nve_requests_total" in text
    assert "nve_queue_depth" in text
    assert "nve_uptime_seconds" in text


def test_snapshot():
    m = MetricsCollector()
    m.inc("x", 10)
    m.set_gauge("y", 5)
    snap = m.snapshot()
    assert "counters" in snap
    assert "gauges" in snap
    assert "histograms" in snap
    assert snap["counters"]["x"] == 10


def test_record_request():
    m = MetricsCollector()
    m.record_request("gpt2", "generate")
    key = 'nve_requests_total{model="gpt2",endpoint="generate"}'
    assert m._counters[key] == 1


def test_record_tokens():
    m = MetricsCollector()
    m.record_tokens("gpt2", 10, 50)
    assert m._counters['nve_prompt_tokens_total{model="gpt2"}'] == 10
    assert m._counters['nve_generated_tokens_total{model="gpt2"}'] == 50


def test_histogram_stats_empty():
    from collections import deque
    m = MetricsCollector()
    stats = m._histogram_stats(deque())
    assert stats["count"] == 0
    assert stats["sum"] == 0.0


def test_histogram_stats_values():
    from collections import deque
    m = MetricsCollector()
    vals = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    stats = m._histogram_stats(vals)
    assert stats["count"] == 10
    assert abs(stats["sum"] - 5.5) < 1e-9
    assert stats["p50"] > 0
