"""
Structured JSON logging for the NVE inference server.

Configures the stdlib logging system to emit one JSON object per log line.
Every log record gains:
  timestamp  — ISO-8601 UTC
  level      — DEBUG / INFO / WARNING / ERROR / CRITICAL
  logger     — dotted logger name
  pid        — OS process ID
  message    — formatted log message
  request_id — if set on the current thread via RequestContext

Usage
─────
Call `configure_logging(level, json_logs)` once at process startup.
In handlers/workers, bind a request ID with:

    from nve.serve.logging_config import RequestContext
    with RequestContext(request_id="abc123"):
        logger.info("handling request")   # → JSON includes request_id

Plain text format (json_logs=False) is used for local dev; JSON for prod.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional


# ── Thread-local request context ───────────────────────────────────────────────

_ctx = threading.local()


class RequestContext:
    """Context manager that binds a request_id to the current thread."""

    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._prev: Optional[str] = None

    def __enter__(self) -> "RequestContext":
        self._prev = getattr(_ctx, "request_id", None)
        _ctx.request_id = self._request_id
        return self

    def __exit__(self, *_) -> None:
        if self._prev is None:
            try:
                del _ctx.request_id
            except AttributeError:
                pass
        else:
            _ctx.request_id = self._prev


def current_request_id() -> Optional[str]:
    return getattr(_ctx, "request_id", None)


# ── JSON formatter ──────────────────────────────────────────────────────────────

class _RequestIdFilter(logging.Filter):
    """
    Injects `request_id` from the thread-local context into every LogRecord
    at emit time, before the formatter runs.  This ensures the ID is available
    even when formatting happens on a different thread or after the context exits.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            rid = current_request_id()
            if rid:
                record.request_id = rid
        return True


class JsonFormatter(logging.Formatter):
    """Emits one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "pid": os.getpid(),
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        if rid:
            obj["request_id"] = rid
        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            obj["stack_info"] = self.formatStack(record.stack_info)
        # Copy any extra fields attached to the record.
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                try:
                    json.dumps(val)  # only include JSON-serialisable extras
                    obj[key] = val
                except (TypeError, ValueError):
                    pass
        return json.dumps(obj, ensure_ascii=False)


# ── Plain text formatter (dev mode) ────────────────────────────────────────────

class DevFormatter(logging.Formatter):
    """Human-readable coloured format for local development."""

    _COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        colour = self._COLOURS.get(record.levelname, "")
        rid = current_request_id()
        rid_part = f" [{rid}]" if rid else ""
        base = (
            f"{ts} {colour}{record.levelname:<8}{self._RESET} "
            f"{record.name}{rid_part} — {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# ── Access log formatter ────────────────────────────────────────────────────────

class AccessLogFormatter(logging.Formatter):
    """Formats aiohttp access log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        # aiohttp passes the request object as record.args[0] when using
        # the default AccessLogger.  We emit a plain structured line instead.
        obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": "ACCESS",
            "pid": os.getpid(),
            "message": record.getMessage(),
        }
        rid = current_request_id()
        if rid:
            obj["request_id"] = rid
        return json.dumps(obj, ensure_ascii=False)


# ── aiohttp custom access logger ───────────────────────────────────────────────

try:
    from aiohttp.abc import AbstractAccessLogger

    class StructuredAccessLogger(AbstractAccessLogger):
        """
        Logs one JSON line per HTTP request:
          method, path, status, bytes_sent, latency_ms, request_id
        """

        def log(self, request, response, time_taken: float) -> None:
            rid = request.headers.get("X-Request-Id") or ""
            obj = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "level": "ACCESS",
                "pid": os.getpid(),
                "method": request.method,
                "path": request.path,
                "status": response.status,
                "bytes_sent": response.content_length or 0,
                "latency_ms": round(time_taken * 1000, 2),
                "remote": request.remote or "",
            }
            if rid:
                obj["request_id"] = rid
            self.logger.info(json.dumps(obj, ensure_ascii=False))

except ImportError:
    StructuredAccessLogger = None  # type: ignore[assignment,misc]


# ── Entry point ────────────────────────────────────────────────────────────────

def configure_logging(level: str = "info", json_logs: bool = True) -> None:
    """
    Configure root logger.  Call once per process before any other imports.

    Parameters
    ----------
    level:
        Logging level string ("debug", "info", "warning", "error").
    json_logs:
        True → JSON formatter (production).
        False → coloured dev formatter.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else DevFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)

    # Suppress noisy third-party loggers.
    logging.getLogger("asyncio").setLevel(logging.WARNING)
