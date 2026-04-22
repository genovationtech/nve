"""
Entry point for `python -m nve.serve` and the `nve-serve` CLI script.

Usage:
    python -m nve.serve [options]
    nve-serve [options]
    nve serve [options]   (via Rust CLI launcher)
"""

from __future__ import annotations

import argparse
import sys


def _port_type(value: str) -> int:
    """Argparse type that validates port is in range 1–65535."""
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Port must be an integer, got {value!r}")
    if not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError(f"Port must be 1–65535, got {port}")
    return port


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"Must be >= 1, got {n}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nve-serve",
        description="NVE HTTP inference server",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_port_type, default=8000)
    parser.add_argument("--model", default=None, help="Model to pre-load at startup")
    parser.add_argument(
        "--processes", "--workers", type=_positive_int, default=1,
        dest="processes",
        help="OS-level server processes sharing one SO_REUSEPORT socket (default: 1)",
    )
    parser.add_argument(
        "--num-inference-workers", type=_positive_int, default=2,
        help="Inference worker threads per model (default: 2)",
    )
    parser.add_argument(
        "--num-replicas", type=_positive_int, default=1,
        help="Backend replicas per model — set > 1 for true parallel inference (default: 1)",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "--no-json-logs", action="store_true",
        help="Emit coloured text logs instead of JSON (dev mode)",
    )
    parser.add_argument("--max-batch-size", type=_positive_int, default=32)
    parser.add_argument("--max-queue-depth", type=_positive_int, default=512)
    parser.add_argument("--hot-budget-mb", type=_positive_int, default=512)
    parser.add_argument("--warm-budget-mb", type=_positive_int, default=2048)
    parser.add_argument("--max-tokens", type=_positive_int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-concurrent-streams", type=_positive_int, default=100)
    parser.add_argument(
        "--rate-limit-rps", type=int, default=0,
        help="Requests per second per IP — 0 to disable (default: 0)",
    )
    parser.add_argument(
        "--shutdown-grace-s", type=float, default=30.0,
        help="Seconds to drain in-flight requests on SIGTERM (default: 30)",
    )
    parser.add_argument(
        "--queue-timeout-s", type=float, default=30.0,
        help="Max seconds a request may wait in the queue before 429 (default: 30)",
    )
    parser.add_argument(
        "--request-timeout-s", type=float, default=300.0,
        help="Total request timeout in seconds (default: 300)",
    )

    args = parser.parse_args()

    from nve.serve.config import ServerConfig
    from nve.serve.server import run

    try:
        config = ServerConfig(
            host=args.host,
            port=args.port,
            processes=args.processes,
            log_level=args.log_level,
            json_logs=not args.no_json_logs,
            num_inference_workers=args.num_inference_workers,
            num_replicas=args.num_replicas,
            max_batch_size=args.max_batch_size,
            max_queue_depth=args.max_queue_depth,
            default_hot_budget_mb=args.hot_budget_mb,
            default_warm_budget_mb=args.warm_budget_mb,
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            max_concurrent_streams=args.max_concurrent_streams,
            rate_limit_rps=args.rate_limit_rps,
            shutdown_grace_s=args.shutdown_grace_s,
            queue_timeout_s=args.queue_timeout_s,
            request_timeout_s=args.request_timeout_s,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        run(config=config, model=args.model)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
