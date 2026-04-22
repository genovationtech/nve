"""
NVE Serve — high-throughput inference server.

A production-grade HTTP inference server for NVE models with:
- REST API (generate, stream, batch, models)
- Server-Sent Events (SSE) streaming with async token bridge
- Priority queue: streaming > single > batch
- Per-model worker pool (N threads × N backend replicas)
- Multi-process scaling via SO_REUSEPORT
- 429 backpressure on queue overload
- Prometheus-compatible metrics (queue_depth, active_workers, utilization)
- Rust paged backend integration

Usage:
    python -m nve.serve --host 0.0.0.0 --port 8000 --model gpt2
    python -m nve.serve --processes 4 --num-inference-workers 2 --model llama
    nve serve --model gpt2 --port 8000
"""

from nve.serve.server import NVEServer, create_app
from nve.serve.model_store import ModelStore, ModelRecord
from nve.serve.config import ServerConfig
from nve.serve.metrics import MetricsCollector
from nve.serve.worker_pool import ModelWorkerPool, WorkItem
from nve.serve.rate_limiter import RateLimiter
from nve.serve.logging_config import configure_logging, RequestContext

__all__ = [
    "NVEServer",
    "create_app",
    "ModelStore",
    "ModelRecord",
    "ServerConfig",
    "MetricsCollector",
    "ModelWorkerPool",
    "WorkItem",
    "RateLimiter",
    "configure_logging",
    "RequestContext",
]
