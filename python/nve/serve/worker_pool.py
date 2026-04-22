"""
Per-model inference worker pool.

Each model gets a dedicated pool of N worker threads. Every worker owns its
own backend replica so inference calls never block each other. Workers pull
from a shared priority queue:

    priority 0 — streaming requests   (lowest latency)
    priority 1 — single /v1/generate  (normal)
    priority 2 — /v1/batch items      (bulk, lower priority)

With num_replicas=1 the single backend is protected by a per-backend lock
(serial inference, multiple models can still run concurrently). With
num_replicas=N each worker owns a dedicated handle — true parallelism.

OOM / error recovery
─────────────────────
If a backend raises MemoryError during inference, the worker marks that
replica as unhealthy, surfaces the error to the caller, and skips future
requests to that replica (they fall back to healthy ones).  A worker whose
replica is unhealthy idles until stop() is called.

Graceful shutdown
──────────────────
stop(drain_timeout_s) enqueues one sortable sentinel per worker, then joins
each thread up to drain_timeout_s seconds.  Any worker still alive after the
timeout is logged as WARNING (we cannot forcibly kill Python threads).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from nve.serve.logging_config import RequestContext

logger = logging.getLogger("nve.serve.worker_pool")


# ── Queue envelope ─────────────────────────────────────────────────────────────
# A single comparable type wraps both real WorkItems and the stop sentinel.
# This avoids cross-type comparison failures in Python's heapq.

@dataclass(order=True)
class _QueueEntry:
    """Comparable envelope stored in the PriorityQueue."""
    priority: int
    seq: int
    payload: object = field(compare=False)  # WorkItem or _STOP


_STOP = object()   # singleton stop payload


# ── Work item ──────────────────────────────────────────────────────────────────

@dataclass
class WorkItem:
    """Inference request.  Wrapped in _QueueEntry before entering the queue."""
    priority: int
    seq: int
    submitted_at: float
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    model_name: Optional[str] = None
    # Filled by worker when done.
    result: Optional[dict] = None
    error: Optional[Exception] = None
    done_event: threading.Event = field(default_factory=threading.Event)

    # Priority constants
    STREAM: int = 0
    GENERATE: int = 1
    BATCH: int = 2

    def queue_wait_s(self) -> float:
        return time.monotonic() - self.submitted_at


# ── Backend worker thread ──────────────────────────────────────────────────────

class _BackendWorker:
    """
    A single worker thread bound to one backend replica.

    Pulls WorkItems from the shared queue, runs inference, and signals
    `done_event`.  Uses a per-backend lock when the pool has only one replica
    so multiple workers don't call into the same handle concurrently.

    OOM handling: if the backend raises MemoryError, the replica is marked
    unhealthy and the worker exits.  The pool will have fewer active workers
    until stop() and a new load() call.
    """

    def __init__(
        self,
        worker_id: int,
        backend,
        backend_lock: threading.Lock,
        work_queue: queue.PriorityQueue,
        metrics_callback: Optional[Callable] = None,
    ) -> None:
        self._id = worker_id
        self._backend = backend
        self._lock = backend_lock
        self._queue = work_queue
        self._metrics_cb = metrics_callback
        self._active = threading.Event()
        self._healthy = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"nve-worker-{worker_id}",
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 0.0) -> None:
        self._thread.join(timeout=timeout or None)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def is_active(self) -> bool:
        return self._active.is_set()

    def is_healthy(self) -> bool:
        return self._healthy

    def _loop(self) -> None:
        while self._healthy:
            try:
                entry = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if entry.payload is _STOP:
                # Re-enqueue sentinel for sibling workers then exit.
                try:
                    self._queue.put_nowait(_QueueEntry(2**31, 2**31, _STOP))
                except queue.Full:
                    pass
                logger.debug(f"Worker {self._id} received stop sentinel — exiting")
                break

            item: WorkItem = entry.payload
            self._active.set()
            wait_s = item.queue_wait_s()
            if self._metrics_cb:
                self._metrics_cb("queue_wait", wait_s)

            with RequestContext(item.request_id):
                t_inf = time.monotonic()
                try:
                    with self._lock:
                        result = self._backend.generate(
                            item.prompt,
                            item.max_new_tokens,
                            item.temperature,
                            item.top_p,
                        )
                    inf_s = time.monotonic() - t_inf
                    if self._metrics_cb:
                        self._metrics_cb("inference_done", inf_s)
                    item.result = result
                    logger.debug(
                        "inference complete",
                        extra={
                            "worker_id": self._id,
                            "queue_wait_s": round(wait_s, 4),
                            "inference_s": round(inf_s, 3),
                            "generated_tokens": result.get("generated_tokens", 0),
                        },
                    )
                except MemoryError as exc:
                    logger.critical(
                        f"Worker {self._id} OOM — marking replica unhealthy and stopping",
                        extra={"worker_id": self._id},
                        exc_info=True,
                    )
                    self._healthy = False
                    item.error = exc
                    try:
                        self._backend.unload()
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error(
                        f"Worker {self._id} inference error: {exc}",
                        extra={"worker_id": self._id},
                        exc_info=True,
                    )
                    item.error = exc
                finally:
                    self._active.clear()
                    item.done_event.set()


# ── Model worker pool ──────────────────────────────────────────────────────────

class ModelWorkerPool:
    """
    N inference workers for a single model.

    Parameters
    ----------
    backend_factory:
        Callable[[], backend] — creates one backend replica per call.
        Called `num_replicas` times.  If num_replicas < num_workers the last
        replica is reused (with a shared lock) for the extra workers.
    num_workers:
        Concurrent worker threads.
    num_replicas:
        Backend replicas to load.  num_replicas == num_workers gives full
        parallelism; num_replicas == 1 serialises inference per-model.
    max_queue_depth:
        Reject new requests (via OverflowError → HTTP 429) once this many
        are waiting.
    """

    _seq_counter = 0
    _seq_lock = threading.Lock()

    def __init__(
        self,
        backend_factory: Callable,
        num_workers: int = 1,
        num_replicas: int = 1,
        max_queue_depth: int = 256,
        metrics_callback: Optional[Callable] = None,
    ) -> None:
        self._work_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_queue_depth
        )
        self._metrics_cb = metrics_callback

        logger.info(
            f"Loading {num_replicas} backend replica(s) for {num_workers} worker(s)"
        )
        self._backends = [backend_factory() for _ in range(num_replicas)]
        self._backend_locks = [threading.Lock() for _ in range(num_replicas)]

        self._workers: List[_BackendWorker] = []
        for i in range(num_workers):
            replica_idx = i % num_replicas
            w = _BackendWorker(
                worker_id=i,
                backend=self._backends[replica_idx],
                backend_lock=self._backend_locks[replica_idx],
                work_queue=self._work_queue,
                metrics_callback=metrics_callback,
            )
            self._workers.append(w)

        self._running = False

    def start(self) -> None:
        self._running = True
        for w in self._workers:
            w.start()
        logger.info(f"Worker pool started ({len(self._workers)} workers, {len(self._backends)} replicas)")

    def stop(self, drain_timeout_s: float = 30.0) -> None:
        """
        Signal all workers to stop and wait up to drain_timeout_s for them
        to finish their current jobs.  Unloads all backend replicas.
        """
        self._running = False
        for _ in self._workers:
            try:
                self._work_queue.put_nowait(_QueueEntry(2**31, 2**31, _STOP))
            except queue.Full:
                pass

        deadline = time.monotonic() + drain_timeout_s
        for w in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            w.join(timeout=remaining)
            if w.is_alive():
                logger.warning(
                    f"Worker {w._id} did not exit within drain timeout "
                    f"({drain_timeout_s:.0f}s) — may still be running"
                )

        for b in self._backends:
            try:
                b.unload()
            except Exception:
                logger.error("Backend unload error", exc_info=True)

    def submit(self, item: WorkItem) -> WorkItem:
        """
        Enqueue a work item.  Raises OverflowError if the queue is full
        (caller should return HTTP 429) or if all replicas are unhealthy.
        """
        unhealthy = sum(1 for w in self._workers if not w.is_healthy())
        if unhealthy == len(self._workers):
            raise OverflowError(
                "All backend replicas are unhealthy (OOM). Reload the model."
            )
        entry = _QueueEntry(item.priority, item.seq, item)
        try:
            self._work_queue.put_nowait(entry)
        except queue.Full:
            qdepth = self._work_queue.qsize()
            raise OverflowError(
                f"Inference queue full ({qdepth}/{self._work_queue.maxsize} slots). "
                "Server is overloaded — retry later."
            )
        return item

    def queue_depth(self) -> int:
        return self._work_queue.qsize()

    def active_count(self) -> int:
        return sum(1 for w in self._workers if w.is_active())

    def healthy_count(self) -> int:
        return sum(1 for w in self._workers if w.is_healthy())

    def worker_count(self) -> int:
        return len(self._workers)

    def replica_count(self) -> int:
        return len(self._backends)

    @classmethod
    def _next_seq(cls) -> int:
        with cls._seq_lock:
            cls._seq_counter += 1
            return cls._seq_counter

    @classmethod
    def make_item(
        cls,
        request_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        priority: int = WorkItem.GENERATE,
        model_name: Optional[str] = None,
    ) -> WorkItem:
        return WorkItem(
            priority=priority,
            seq=cls._next_seq(),
            submitted_at=time.monotonic(),
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            model_name=model_name,
        )
