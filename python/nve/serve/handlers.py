"""
aiohttp request handlers for the NVE inference API.

Endpoints:
  POST  /v1/generate              — Single synchronous generation
  POST  /v1/generate/stream       — SSE streaming generation
  POST  /v1/batch                 — Batch synchronous generation
  GET   /v1/models                — List loaded models
  POST  /v1/models/{name}/load    — Load a model
  DELETE /v1/models/{name}        — Unload a model
  GET   /health                   — Liveness check
  GET   /ready                    — Readiness check (503 if no model / overloaded)
  GET   /metrics                  — Prometheus metrics
  GET   /v1/metrics               — JSON metrics snapshot

Backpressure
────────────
When the model's priority queue is full the handler returns HTTP 429
(Too Many Requests) with a `Retry-After` header.

Rate limiting
─────────────
If ServerConfig.rate_limit_rps > 0, a per-IP token-bucket limiter is checked
before every inference call.  Blocked requests get HTTP 429.

Request IDs
────────────
A UUID is generated per request.  It is:
  - bound as X-Request-Id response header on every response.
  - bound to the thread-local RequestContext for structured log correlation.
  - included in the JSON response body.

Timeout enforcement (per-stage)
────────────────────────────────
queue_timeout_s: maximum time a request may wait in the model's queue.
                 Returns 429 if exceeded (server too busy).
request_timeout_s: total timeout covering queue wait + inference.
                   Returns 504 if exceeded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from aiohttp import web

from nve.serve.logging_config import RequestContext
from nve.serve.streaming import async_sse_stream, sse_error
from nve.serve.worker_pool import ModelWorkerPool, WorkItem

if TYPE_CHECKING:
    from nve.serve.server import NVEServer

logger = logging.getLogger("nve.serve.handlers")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_request_id() -> str:
    return str(uuid.uuid4())[:12]


def _json_response(
    data: dict,
    status: int = 200,
    request_id: str = "",
) -> web.Response:
    headers = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    return web.Response(
        status=status,
        content_type="application/json",
        headers=headers,
        text=json.dumps(data, indent=2),
    )


def _error(
    message: str,
    status: int = 400,
    request_id: str = "",
) -> web.Response:
    return _json_response({"error": message}, status, request_id=request_id)


def _too_many_requests(
    message: str,
    retry_after: float = 1.0,
    request_id: str = "",
) -> web.Response:
    return web.Response(
        status=429,
        headers={
            "Retry-After": str(int(retry_after) + 1),
            **({"X-Request-Id": request_id} if request_id else {}),
        },
        content_type="application/json",
        text=json.dumps({"error": message}),
    )


def _parse_generate_body(body: dict, cfg) -> tuple:
    """Extract and validate generation parameters from request body."""
    prompt = body.get("prompt", "")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("'prompt' is required and must be a non-empty string")
    if len(prompt) > cfg.max_prompt_chars:
        raise ValueError(f"Prompt exceeds {cfg.max_prompt_chars} character limit")

    max_new_tokens = int(body.get("max_new_tokens", cfg.default_max_tokens))
    temperature = float(body.get("temperature", cfg.default_temperature))
    top_p = float(body.get("top_p", cfg.default_top_p))

    if max_new_tokens < 1 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be between 1 and 4096")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be between 0 (exclusive) and 1.0")

    return prompt, max_new_tokens, temperature, top_p


def _check_rate_limit(server: "NVEServer", request: web.Request, request_id: str):
    """Return a 429 response if the caller's IP is rate-limited, else None."""
    if server.rate_limiter is None:
        return None
    ip = request.remote or "unknown"
    allowed, retry_after = server.rate_limiter.check(ip)
    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            extra={"remote": ip, "retry_after_s": retry_after},
        )
        return _too_many_requests(
            f"Rate limit exceeded. Retry after {retry_after:.1f}s.",
            retry_after=retry_after,
            request_id=request_id,
        )
    return None


async def _await_item(
    item: WorkItem,
    timeout_s: float,
    loop: asyncio.AbstractEventLoop,
) -> bool:
    """Non-blocking wait for a WorkItem's done_event."""
    return await loop.run_in_executor(
        None, lambda: item.done_event.wait(timeout=timeout_s)
    )


# ── /health  (liveness — always 200 if process is alive) ──────────────────────

async def health(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    return _json_response({
        "status": "ok",
        "uptime_s": round(time.time() - server.start_time, 1),
        "version": "0.3.0",
    })


# ── /ready  (readiness — 503 if not able to serve) ────────────────────────────

async def ready(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]

    if server.shutting_down:
        return _json_response({"status": "shutting_down"}, status=503)

    loaded = server.model_store.list_models()
    if not loaded:
        return _json_response(
            {"status": "no_model_loaded", "detail": "POST /v1/models/{name}/load first"},
            status=503,
        )

    # Check if any model's queue is critically overloaded (> 90 % full).
    for record in loaded:
        pool = server.model_store.get(record["name"])
        if pool is None:
            continue
        depth = record.get("queue_depth", 0)
        max_depth = server.config.max_queue_depth
        if max_depth > 0 and depth / max_depth > 0.9:
            return _json_response(
                {
                    "status": "overloaded",
                    "model": record["name"],
                    "queue_depth": depth,
                    "max_queue_depth": max_depth,
                },
                status=503,
            )

    return _json_response({
        "status": "ready",
        "models": [m["name"] for m in loaded],
        "uptime_s": round(time.time() - server.start_time, 1),
    })


# ── /metrics ───────────────────────────────────────────────────────────────────

async def metrics_prometheus(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    server.refresh_pool_metrics()
    text = server.metrics.render_prometheus()
    return web.Response(status=200, content_type="text/plain; version=0.0.4", text=text)


async def metrics_json(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    server.refresh_pool_metrics()
    return _json_response(server.metrics.snapshot())


# ── /v1/models ─────────────────────────────────────────────────────────────────

async def list_models(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    return _json_response({"models": server.model_store.list_models()})


async def load_model(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    name = request.match_info["name"]
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body", request_id=rid)

    model_path = body.get("model_path") or body.get("model")
    if not model_path:
        return _error("'model_path' is required", request_id=rid)

    with RequestContext(rid):
        try:
            loop = asyncio.get_event_loop()
            record = await loop.run_in_executor(
                None, server.model_store.load, name, model_path
            )
            server.metrics.set_model_loaded(name, True)
            logger.info(
                f"Model loaded",
                extra={"model_name": name, "model_path": model_path},
            )
            return _json_response(
                {"status": "loaded", "model": record.info()}, status=201, request_id=rid
            )
        except Exception as e:
            logger.error(f"Failed to load model '{name}'", exc_info=True)
            return _error(str(e), status=500, request_id=rid)


async def unload_model(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    name = request.match_info["name"]
    removed = server.model_store.unload(name)
    if not removed:
        return _error(f"Model '{name}' not found", status=404, request_id=rid)
    server.metrics.set_model_loaded(name, False)
    logger.info(f"Model unloaded", extra={"model_name": name})
    return _json_response({"status": "unloaded", "name": name}, request_id=rid)


# ── /v1/generate ───────────────────────────────────────────────────────────────

async def generate(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    cfg = server.config
    rid = _new_request_id()

    with RequestContext(rid):
        if server.shutting_down:
            return _error("Server is shutting down", status=503, request_id=rid)

        rl = _check_rate_limit(server, request, rid)
        if rl is not None:
            return rl

        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body", request_id=rid)

        model_name = server.router.resolve(body.get("model") or None)
        if not model_name:
            return _error(
                "No model loaded or selector matched nothing. "
                "POST to /v1/models/{name}/load first.",
                status=503, request_id=rid,
            )

        record = server.model_store.get(model_name)
        if record is None:
            return _error(f"Model '{model_name}' not found", status=404, request_id=rid)

        try:
            prompt, max_new_tokens, temperature, top_p = _parse_generate_body(body, cfg)
        except ValueError as e:
            return _error(str(e), request_id=rid)

        t_start = time.time()
        server.metrics.record_request(model_name, "generate")
        logger.info(
            "generate request",
            extra={"model": model_name, "max_new_tokens": max_new_tokens},
        )

        item = ModelWorkerPool.make_item(
            request_id=rid,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            priority=WorkItem.GENERATE,
            model_name=model_name,
        )

        try:
            record.worker_pool.submit(item)
        except OverflowError as e:
            server.metrics.record_error(model_name, "QueueFull")
            logger.warning("Queue full — returning 429", extra={"model": model_name})
            return _too_many_requests(str(e), request_id=rid)

        # Per-stage timeout: fail fast if stuck in queue too long.
        loop = asyncio.get_event_loop()
        if cfg.queue_timeout_s > 0:
            queue_done = await _await_item(item, cfg.queue_timeout_s, loop)
            if not queue_done:
                server.metrics.record_error(model_name, "QueueTimeout")
                logger.warning(
                    "Request exceeded queue timeout",
                    extra={"model": model_name, "queue_timeout_s": cfg.queue_timeout_s},
                )
                return _too_many_requests(
                    f"Server too busy — queue wait exceeded {cfg.queue_timeout_s}s. "
                    "Retry later.",
                    retry_after=1.0,
                    request_id=rid,
                )

        # If already done (fast inference), skip second await; else wait for remainder.
        if not item.done_event.is_set():
            elapsed_so_far = time.time() - t_start
            remaining = max(0.1, cfg.request_timeout_s - elapsed_so_far)
            total_done = await _await_item(item, remaining, loop)
            if not total_done:
                server.metrics.record_error(model_name, "Timeout")
                logger.warning("Request timed out", extra={"model": model_name})
                return _error("Request timed out", status=504, request_id=rid)

        if item.error:
            server.metrics.record_error(model_name, type(item.error).__name__)
            logger.error(
                f"Inference error: {item.error}",
                extra={"model": model_name},
                exc_info=item.error,
            )
            return _error(str(item.error), status=500, request_id=rid)

        result = item.result
        elapsed = time.time() - t_start
        q_wait = item.queue_wait_s()

        server.metrics.record_tokens(
            model_name,
            result.get("prompt_tokens", 0),
            result.get("generated_tokens", 0),
        )
        server.metrics.record_latency(model_name, elapsed)
        server.metrics.record_tps(model_name, result.get("tokens_per_sec", 0))
        server.metrics.record_queue_wait(model_name, q_wait)
        record.record_success(result.get("generated_tokens", 0))
        server.model_store.touch(model_name)   # update LRU

        logger.info(
            "generate complete",
            extra={
                "model": model_name,
                "elapsed_s": round(elapsed, 3),
                "queue_wait_s": round(q_wait, 4),
                "generated_tokens": result.get("generated_tokens", 0),
                "tokens_per_sec": result.get("tokens_per_sec", 0),
            },
        )

        return _json_response(
            {
                "id": rid,
                "model": model_name,
                "prompt": prompt,
                **result,
                "elapsed_s": round(elapsed, 3),
                "queue_wait_s": round(q_wait, 4),
            },
            request_id=rid,
        )


# ── /v1/generate/stream ────────────────────────────────────────────────────────

async def generate_stream(request: web.Request) -> web.StreamResponse:
    server: NVEServer = request.app["nve_server"]
    cfg = server.config
    rid = _new_request_id()

    with RequestContext(rid):
        if server.shutting_down:
            return _error("Server is shutting down", status=503, request_id=rid)

        rl = _check_rate_limit(server, request, rid)
        if rl is not None:
            return rl

        if server.active_streams >= cfg.max_concurrent_streams:
            return _too_many_requests(
                f"Too many concurrent streams (max {cfg.max_concurrent_streams}). Retry later.",
                request_id=rid,
            )

        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body", request_id=rid)

        model_name = server.router.resolve(body.get("model") or None)
        if not model_name:
            return _error("No model loaded or selector matched nothing.", status=503, request_id=rid)

        record = server.model_store.get(model_name)
        if record is None:
            return _error(f"Model '{model_name}' not found", status=404, request_id=rid)

        try:
            prompt, max_new_tokens, temperature, top_p = _parse_generate_body(body, cfg)
        except ValueError as e:
            return _error(str(e), request_id=rid)

        server.metrics.record_request(model_name, "generate_stream")
        logger.info(
            "stream request",
            extra={"model": model_name, "max_new_tokens": max_new_tokens},
        )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id": rid,
            },
        )
        await response.prepare(request)

        backend = record.worker_pool._backends[0]
        t_start = time.time()
        server.active_streams += 1
        try:
            async for chunk in async_sse_stream(
                backend=backend,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                model_name=model_name,
                request_id=rid,
                keep_alive_s=cfg.sse_keep_alive_s,
            ):
                await response.write(chunk)
            record.record_success()
            logger.info(
                "stream complete",
                extra={"model": model_name, "elapsed_s": round(time.time() - t_start, 3)},
            )
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            try:
                await response.write(sse_error(str(e)))
            except Exception:
                pass
            server.metrics.record_error(model_name, type(e).__name__)
            record.record_error()
        finally:
            server.active_streams -= 1

        await response.write_eof()
        return response


# ── /v1/batch ──────────────────────────────────────────────────────────────────

async def batch_generate(request: web.Request) -> web.Response:
    server: NVEServer = request.app["nve_server"]
    cfg = server.config
    rid = _new_request_id()

    with RequestContext(rid):
        if server.shutting_down:
            return _error("Server is shutting down", status=503, request_id=rid)

        rl = _check_rate_limit(server, request, rid)
        if rl is not None:
            return rl

        try:
            body = await request.json()
        except Exception:
            return _error("Invalid JSON body", request_id=rid)

        model_name = server.router.resolve(body.get("model") or None)
        if not model_name:
            return _error("No model loaded or selector matched nothing.", status=503, request_id=rid)

        record = server.model_store.get(model_name)
        if record is None:
            return _error(f"Model '{model_name}' not found", status=404, request_id=rid)

        prompts = body.get("prompts", [])
        if not prompts or not isinstance(prompts, list):
            return _error("'prompts' must be a non-empty list of strings", request_id=rid)
        if len(prompts) > cfg.max_batch_size:
            return _error(f"Too many prompts (max {cfg.max_batch_size})", request_id=rid)

        max_new_tokens = int(body.get("max_new_tokens", cfg.default_max_tokens))
        temperature = float(body.get("temperature", cfg.default_temperature))
        top_p = float(body.get("top_p", cfg.default_top_p))

        server.metrics.record_request(model_name, "batch")
        t_start = time.time()
        logger.info(
            "batch request",
            extra={"model": model_name, "num_prompts": len(prompts)},
        )

        items = []
        rejected = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt:
                continue
            item = ModelWorkerPool.make_item(
                request_id=str(uuid.uuid4())[:8] if True else rid,  # per-item ID
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                priority=WorkItem.BATCH,
                model_name=model_name,
            )
            try:
                record.worker_pool.submit(item)
                items.append(item)
            except OverflowError:
                rejected.append({"prompt": prompt[:60], "error": "queue_full"})

        if not items and rejected:
            server.metrics.record_error(model_name, "QueueFull")
            return _too_many_requests(
                "All batch slots rejected — server overloaded.", request_id=rid
            )

        loop = asyncio.get_event_loop()
        results = []
        for item in items:
            done = await _await_item(item, cfg.request_timeout_s, loop)
            if not done:
                results.append({"id": item.request_id, "error": "timeout"})
                server.metrics.record_error(model_name, "Timeout")
            elif item.error:
                results.append({"id": item.request_id, "error": str(item.error)})
                record.record_error()
                server.metrics.record_error(model_name, type(item.error).__name__)
            else:
                results.append({"id": item.request_id, **item.result})
                record.record_success(item.result.get("generated_tokens", 0))

        for r in rejected:
            results.append({"error": r["error"], "prompt_preview": r["prompt"]})

        elapsed = time.time() - t_start
        logger.info(
            "batch complete",
            extra={
                "model": model_name,
                "submitted": len(items),
                "rejected": len(rejected),
                "elapsed_s": round(elapsed, 3),
            },
        )

        return _json_response(
            {
                "id": rid,
                "model": model_name,
                "count": len(results),
                "submitted": len(items),
                "rejected": len(rejected),
                "elapsed_s": round(elapsed, 3),
                "results": results,
            },
            request_id=rid,
        )


# ── /v1/router — routing management ───────────────────────────────────────────

async def router_status(request: web.Request) -> web.Response:
    """GET /v1/router — current router state (policy, aliases, versions, weights)."""
    server: NVEServer = request.app["nve_server"]
    return _json_response(server.router.status())


async def router_add_alias(request: web.Request) -> web.Response:
    """
    POST /v1/router/aliases
    Body: {"alias": "fast", "target": "llama-q4"}
    """
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body", request_id=rid)
    alias = body.get("alias", "").strip()
    target = body.get("target", "").strip()
    if not alias or not target:
        return _error("'alias' and 'target' are required", request_id=rid)
    server.router.add_alias(alias, target)
    return _json_response({"status": "ok", "alias": alias, "target": target}, request_id=rid)


async def router_register_version(request: web.Request) -> web.Response:
    """
    POST /v1/router/versions
    Body: {"base": "llama", "version": "v2", "model": "llama-q8"}
    """
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body", request_id=rid)
    base = body.get("base", "").strip()
    version = body.get("version", "").strip()
    model = body.get("model", "").strip()
    if not base or not version or not model:
        return _error("'base', 'version', and 'model' are required", request_id=rid)
    server.router.register_version(base, version, model)
    return _json_response(
        {"status": "ok", "selector": f"{base}:{version}", "model": model},
        request_id=rid,
    )


async def router_set_weights(request: web.Request) -> web.Response:
    """
    POST /v1/router/weights
    Body: {"weights": {"llama-q4": 80, "llama-q8": 20}}
    Switches routing policy to 'weighted' automatically.
    """
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body", request_id=rid)
    weights = body.get("weights")
    if not isinstance(weights, dict):
        return _error("'weights' must be a dict {model_name: percent}", request_id=rid)
    try:
        server.router.set_weights(weights)
        server.router.set_policy("weighted")
    except ValueError as e:
        return _error(str(e), request_id=rid)
    return _json_response({"status": "ok", "weights": weights, "policy": "weighted"}, request_id=rid)


async def router_set_policy(request: web.Request) -> web.Response:
    """
    POST /v1/router/policy
    Body: {"policy": "least_loaded"}
    """
    server: NVEServer = request.app["nve_server"]
    rid = _new_request_id()
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body", request_id=rid)
    policy = body.get("policy", "").strip()
    if not policy:
        return _error("'policy' is required", request_id=rid)
    try:
        server.router.set_policy(policy)
    except ValueError as e:
        return _error(str(e), request_id=rid)
    return _json_response({"status": "ok", "policy": policy}, request_id=rid)


# ── import fix (uuid used above) ───────────────────────────────────────────────
import uuid  # noqa: E402
