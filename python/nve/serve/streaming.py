"""
Server-Sent Events (SSE) helpers and async streaming bridge.

SSE format (RFC):
    data: <payload>\\n\\n

NVE SSE event types:
    data: {"type": "token",  "text": "...", "index": N}
    data: {"type": "done",   "stats": {...}}
    data: {"type": "error",  "message": "..."}
    : keep-alive                              (comment line, no event fired)

Async streaming architecture
─────────────────────────────
The model backends produce tokens synchronously (blocking calls).  Running a
sync iterator directly on the event loop thread would block all other coroutines.

Instead we use an asyncio.Queue as a bridge:

    Thread pool                     Event loop
    ─────────────────────────       ────────────────────────────
    backend.generate_stream()  →    asyncio.Queue  →  SSE response
    (runs in executor)              (non-blocking)

The producer thread pushes token strings and a sentinel (None) into the queue.
The async consumer reads from the queue and writes SSE frames to the HTTP response.

Client disconnect
─────────────────
A threading.Event (`stop_event`) is shared with the producer thread.  When the
async consumer detects that the client has gone away (ConnectionResetError or
response write fails), it sets stop_event so the producer thread stops iterating
as soon as possible rather than running to completion for a dead connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import AsyncIterator, Iterator

logger = logging.getLogger("nve.serve.streaming")

_STREAM_DONE = object()   # sentinel: producer finished normally
_STREAM_ERR  = object()   # sentinel: producer raised; next item is the exc


# ── SSE frame helpers ──────────────────────────────────────────────────────────

def sse_event(payload: dict) -> bytes:
    """Encode a single SSE event frame."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def sse_token(text: str, index: int) -> bytes:
    return sse_event({"type": "token", "text": text, "index": index})


def sse_done(stats: dict) -> bytes:
    return sse_event({"type": "done", "stats": stats})


def sse_error(message: str) -> bytes:
    return sse_event({"type": "error", "message": message})


def sse_keep_alive() -> bytes:
    """Empty comment line — keeps the connection alive without emitting an event."""
    return b": keep-alive\n\n"


# ── Async streaming bridge ─────────────────────────────────────────────────────

async def async_sse_stream(
    backend,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_name: str,
    request_id: str,
    keep_alive_s: float = 15.0,
) -> AsyncIterator[bytes]:
    """
    Run `backend.generate_stream()` in a thread-pool executor and yield SSE
    frames asynchronously.  The event loop is never blocked.

    A `stop_event` is shared with the producer thread.  If the caller stops
    iterating early (client disconnect), setting stop_event signals the producer
    to abort iteration as soon as its current token is done.
    """
    loop = asyncio.get_event_loop()
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    stop_event = threading.Event()
    start = time.time()
    index = 0

    def _produce() -> None:
        """Runs in ThreadPoolExecutor — calls the sync streaming backend."""
        try:
            for fragment in backend.generate_stream(
                prompt, max_new_tokens, temperature, top_p
            ):
                if stop_event.is_set():
                    logger.debug(
                        f"[{request_id}] Producer stopping — client disconnected"
                    )
                    break
                try:
                    loop.call_soon_threadsafe(token_queue.put_nowait, fragment)
                except Exception as put_err:
                    logger.error(
                        f"[{request_id}] Failed to enqueue token: {put_err}"
                    )
                    break
        except Exception as exc:
            logger.error(
                f"[{request_id}] Streaming backend error: {exc}", exc_info=True
            )
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, _STREAM_ERR)
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)
            except Exception:
                pass
        finally:
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, _STREAM_DONE)
            except Exception:
                pass

    future = loop.run_in_executor(None, _produce)
    last_keep_alive = time.monotonic()

    try:
        while True:
            try:
                item = await asyncio.wait_for(token_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                now = time.monotonic()
                if now - last_keep_alive >= keep_alive_s:
                    yield sse_keep_alive()
                    last_keep_alive = now
                continue

            if item is _STREAM_DONE:
                break

            if item is _STREAM_ERR:
                exc = await token_queue.get()
                yield sse_error(str(exc))
                break

            yield sse_token(item, index)
            index += 1

    except (ConnectionResetError, asyncio.CancelledError, GeneratorExit):
        # Client disconnected — tell the producer to stop.
        logger.info(f"[{request_id}] Client disconnected during stream")
        stop_event.set()
    finally:
        # Always signal producer to stop (idempotent).
        stop_event.set()
        # Wait for the producer thread to finish (up to 5 s).
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            logger.warning(
                f"[{request_id}] Producer thread did not finish within 5 s after disconnect"
            )

    elapsed = time.time() - start
    yield sse_done({
        "model": model_name,
        "request_id": request_id,
        "total_tokens": index,
        "elapsed_s": round(elapsed, 3),
    })


# ── Legacy sync-iter wrapper (kept for compat) ─────────────────────────────────

async def async_sse_from_sync_iter(
    sync_iter: Iterator[str],
    model_name: str,
    request_id: str,
) -> AsyncIterator[bytes]:
    """
    Legacy wrapper: wraps a pre-created sync iterator.

    Prefer `async_sse_stream()` which runs the iterator in a thread pool.
    This version runs the iterator on the event loop thread — only safe when
    each step is very fast (e.g. word-splitting an already-computed result).
    """
    start = time.time()
    index = 0
    for fragment in sync_iter:
        yield sse_token(fragment, index)
        index += 1
        await asyncio.sleep(0)
    yield sse_done({
        "model": model_name,
        "request_id": request_id,
        "total_tokens": index,
        "elapsed_s": round(time.time() - start, 3),
    })
