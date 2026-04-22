"""
NVE inference server — assembles all components and starts aiohttp.

Architecture (v0.3 — high-throughput, production-grade)
─────────────────────────────────────────────────────────

    HTTP Request
        │
        ▼ (async handler)
    Handler validates + rate-limits + builds WorkItem
        │
        ▼ submit(item) — raises OverflowError → HTTP 429
    ModelWorkerPool  ◄── per-model priority queue
        │  N workers, each owning one backend replica
        ▼
    Backend (Rust paged / Rust baseline / PyTorch)
        │
        ▼ item.done_event.set()
    Handler awaits → HTTP 200

Streaming requests bypass the worker queue and call the backend directly
in a thread-pool executor, piping tokens to the HTTP response via an
asyncio.Queue with a stop_event for client-disconnect cleanup.

Graceful shutdown
──────────────────
On SIGTERM / SIGINT:
  1. Set server.shutting_down = True  → handlers start returning 503.
  2. Stop accepting new connections (aiohttp's on_shutdown callbacks run).
  3. Wait up to shutdown_grace_s for in-flight requests to complete.
  4. Stop each model's worker pool (drain_timeout_s = grace_s).
  5. Unload all models.
  6. Exit.

Multi-process scaling
──────────────────────
Set `--processes N` to spawn N server processes on the same port using
SO_REUSEPORT.  Each process loads the model independently.  The parent
forwards SIGTERM to all children and waits for graceful exit.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import socket
import time
from typing import Optional

from aiohttp import web

from nve.serve.config import ServerConfig
from nve.serve.handlers import (
    batch_generate,
    generate,
    generate_stream,
    health,
    list_models,
    load_model,
    metrics_json,
    metrics_prometheus,
    ready,
    unload_model,
    router_status,
    router_add_alias,
    router_register_version,
    router_set_weights,
    router_set_policy,
)
from nve.serve.hardware import hardware_info
from nve.serve.hardware import get_device_manager
from nve.serve.metrics import MetricsCollector
from nve.serve.model_store import ModelStore
from nve.serve.model_router import ModelRouter, RoutingPolicy
from nve.serve.rate_limiter import RateLimiter

logger = logging.getLogger("nve.serve")


class NVEServer:
    """Top-level server object — holds all shared state."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.start_time = time.time()
        self.metrics = MetricsCollector()
        self.active_streams: int = 0
        self.shutting_down: bool = False

        self.rate_limiter: Optional[RateLimiter] = (
            RateLimiter(rps=config.rate_limit_rps)
            if config.rate_limit_rps > 0
            else None
        )

        # Initialise hardware detection once at startup.
        # This logs the full device summary and warms the DeviceManager singleton.
        import os
        if config.device:
            os.environ.setdefault("NVE_DEVICE", config.device)
        self.device_manager = get_device_manager()

        def _metrics_cb(event: str, data) -> None:
            if event == "queue_wait":
                self.metrics.observe("nve_queue_wait_seconds", data)

        self.model_store = ModelStore(
            hot_budget_mb=config.default_hot_budget_mb,
            warm_budget_mb=config.default_warm_budget_mb,
            num_inference_workers=config.num_inference_workers,
            num_replicas=config.num_replicas,
            max_queue_depth=config.max_queue_depth,
            metrics_callback=_metrics_cb,
            global_hot_budget_mb=config.global_hot_budget_mb,
            max_loaded_models=config.max_loaded_models,
            device=config.device or None,
            quantization=config.quantization,
            attn_implementation=config.attn_implementation,
            compile_torch=config.compile_torch,
        )
        self.router = ModelRouter(
            self.model_store,
            policy=config.routing_policy,
        )

    def preload_model(self, model_id: str, name: Optional[str] = None) -> None:
        name = name or model_id.split("/")[-1]
        logger.info(f"Pre-loading model '{name}' from '{model_id}'")
        self.model_store.load(name, model_id)
        self.metrics.set_model_loaded(name, True)

    def refresh_pool_metrics(self) -> None:
        """Push per-model pool gauges into the metrics collector."""
        with self.model_store._lock:
            for name, record in self.model_store._models.items():
                pool = record.worker_pool
                self.metrics.set_queue_depth(name, pool.queue_depth())
                self.metrics.set_active_workers(
                    name, pool.active_count(), pool.worker_count()
                )

    def shutdown(self) -> None:
        """Initiate graceful shutdown — called from signal handler."""
        if self.shutting_down:
            return
        self.shutting_down = True
        logger.info(
            "Graceful shutdown initiated",
            extra={"grace_s": self.config.shutdown_grace_s},
        )


def create_app(server: NVEServer) -> web.Application:
    """Build and return the aiohttp Application with all routes registered."""
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app["nve_server"] = server

    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    app.router.add_get("/metrics", metrics_prometheus)
    app.router.add_get("/v1/metrics", metrics_json)
    app.router.add_get("/v1/hardware", hardware_info)

    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/models/{name}/load", load_model)
    app.router.add_delete("/v1/models/{name}", unload_model)

    app.router.add_post("/v1/generate", generate)
    app.router.add_post("/v1/generate/stream", generate_stream)
    app.router.add_post("/v1/batch", batch_generate)

    # ── Router management ─────────────────────────────────────────────────────
    app.router.add_get("/v1/router", router_status)
    app.router.add_post("/v1/router/aliases", router_add_alias)
    app.router.add_post("/v1/router/versions", router_register_version)
    app.router.add_post("/v1/router/weights", router_set_weights)
    app.router.add_post("/v1/router/policy", router_set_policy)

    app.on_cleanup.append(_on_cleanup)
    return app


async def _on_cleanup(app: web.Application) -> None:
    server: NVEServer = app["nve_server"]
    grace = server.config.shutdown_grace_s
    logger.info(f"Stopping worker pools (drain timeout {grace:.0f}s)")
    for name in list(server.model_store._models.keys()):
        record = server.model_store._models.get(name)
        if record:
            record.worker_pool.stop(drain_timeout_s=grace)
    for name in list(server.model_store._models.keys()):
        server.model_store.unload(name)
    logger.info("NVE server shutdown complete")


# ── Process-level entry points ─────────────────────────────────────────────────

def _worker_process(
    config: ServerConfig,
    model: Optional[str],
    sock: Optional[socket.socket] = None,
) -> None:
    """
    Entry point for a single server process.

    Installs a SIGTERM handler that triggers graceful shutdown.
    """
    from nve.serve.logging_config import configure_logging, StructuredAccessLogger

    configure_logging(level=config.log_level, json_logs=config.json_logs)

    logger.info(
        "Server process starting",
        extra={
            "pid": os.getpid(),
            "workers": config.num_inference_workers,
            "replicas": config.num_replicas,
        },
    )

    server = NVEServer(config)

    if model:
        try:
            server.preload_model(model)
        except Exception as e:
            logger.error(f"Failed to pre-load model '{model}': {e}", exc_info=True)

    app = create_app(server)

    # Install SIGTERM handler for graceful shutdown.
    def _handle_sigterm(signum, frame):
        server.shutdown()
        # Raising KeyboardInterrupt breaks web.run_app's event loop.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    access_log_cls = StructuredAccessLogger if StructuredAccessLogger else None

    try:
        if sock is not None:
            web.run_app(
                app,
                sock=sock,
                access_log_class=access_log_cls,
                access_log=logger,
            )
        else:
            web.run_app(
                app,
                host=config.host,
                port=config.port,
                access_log_class=access_log_cls,
                access_log=logger,
            )
    except KeyboardInterrupt:
        logger.info("Server process received shutdown signal")


def run(config: Optional[ServerConfig] = None, model: Optional[str] = None) -> None:
    """
    Start the NVE server.  Blocks until Ctrl+C or SIGTERM.

    config.processes == 1  → runs in-process (default).
    config.processes  > 1  → forks N worker processes sharing one SO_REUSEPORT
                              socket.  Parent forwards SIGTERM to all children.
    """
    if config is None:
        config = ServerConfig.from_env()

    from nve.serve.logging_config import configure_logging
    configure_logging(level=config.log_level, json_logs=config.json_logs)

    n = config.processes
    logger.info(
        "Starting NVE server",
        extra={
            "host": config.host,
            "port": config.port,
            "processes": n,
            "inference_workers": config.num_inference_workers,
            "replicas": config.num_replicas,
            "rate_limit_rps": config.rate_limit_rps,
        },
    )

    if n == 1:
        _worker_process(config, model)
        return

    # Multi-process: create a shared SO_REUSEPORT socket.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        logger.warning("SO_REUSEPORT not available on this platform")
    sock.bind((config.host, config.port))
    sock.listen(1024)

    processes = []
    for i in range(n):
        child_sock = sock.dup()
        p = multiprocessing.Process(
            target=_worker_process,
            args=(config, model, child_sock),
            name=f"nve-worker-{i}",
            daemon=False,   # non-daemon so they clean up properly
        )
        p.start()
        child_sock.close()
        processes.append(p)
        logger.info(f"Spawned server process", extra={"pid": p.pid, "worker_idx": i})

    sock.close()

    def _forward_sigterm(signum, frame):
        logger.info("Parent received SIGTERM — forwarding to child processes")
        for p in processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, _forward_sigterm)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutting down all server processes")
        grace = config.shutdown_grace_s
        for p in processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
        # Wait for graceful exit.
        deadline = time.monotonic() + grace
        for p in processes:
            remaining = max(0.0, deadline - time.monotonic())
            p.join(timeout=remaining)
            if p.is_alive():
                logger.warning(
                    f"Process {p.pid} did not exit after {grace:.0f}s — terminating"
                )
                p.terminate()
                p.join(timeout=5)
