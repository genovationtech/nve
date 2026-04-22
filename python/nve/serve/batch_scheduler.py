"""
Legacy batch scheduler — kept for backward compatibility.

The server now uses `ModelWorkerPool` (see worker_pool.py) which provides
per-model priority queues, N worker threads, and N backend replicas.

`BatchScheduler` / `BatchRequest` are preserved so external code that imports
them directly continues to work, but `NVEServer` no longer instantiates this
class.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("nve.serve.batch_scheduler")


@dataclass
class BatchRequest:
    """A single inference request.  Kept for backward compatibility."""
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    stream: bool = False
    model_name: Optional[str] = None
    result: Optional[dict] = field(default=None, repr=False)
    error: Optional[Exception] = field(default=None, repr=False)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)


Dispatch = Callable[[List[BatchRequest]], None]


class BatchScheduler:
    """
    Legacy single-threaded batch scheduler.

    Prefer `ModelWorkerPool` for new code — it provides priority scheduling,
    per-model queues, N workers, and N backend replicas.
    """

    def __init__(
        self,
        dispatch_fn: Dispatch,
        max_batch_size: int = 8,
        timeout_ms: float = 50.0,
        max_queue_depth: int = 256,
    ) -> None:
        self._dispatch = dispatch_fn
        self._max_batch = max_batch_size
        self._timeout = timeout_ms / 1000.0
        self._queue: queue.Queue[BatchRequest] = queue.Queue(maxsize=max_queue_depth)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nve-batch-scheduler")
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()
        logger.info(
            f"BatchScheduler started (max_batch={self._max_batch}, "
            f"timeout={self._timeout * 1000:.0f}ms) [LEGACY — prefer ModelWorkerPool]"
        )

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass

    def submit(self, req: BatchRequest) -> BatchRequest:
        try:
            self._queue.put(req, timeout=5.0)
        except queue.Full:
            req.error = RuntimeError("Request queue is full — server overloaded")
            req.done_event.set()
        return req

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while self._running:
            batch: List[BatchRequest] = []
            try:
                first = self._queue.get(timeout=1.0)
                if first is None:
                    break
                batch.append(first)
            except queue.Empty:
                continue

            deadline = time.monotonic() + self._timeout
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    if req is None:
                        break
                    batch.append(req)
                except queue.Empty:
                    break

            if batch:
                try:
                    self._dispatch(batch)
                except Exception as exc:
                    logger.exception("Batch dispatch error")
                    for req in batch:
                        req.error = exc
                        req.done_event.set()
