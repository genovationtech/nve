"""Server configuration with startup validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Number of OS-level server processes (SO_REUSEPORT).  Each process loads
    # the model independently.  Set > 1 for CPU-bound / multi-core scale-out.
    processes: int = 1
    log_level: str = "info"
    # Emit JSON log lines (True = production; False = coloured dev format).
    json_logs: bool = True

    # ── Model defaults ────────────────────────────────────────────────────────
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    # ── Per-model worker pool ─────────────────────────────────────────────────
    # num_inference_workers: threads pulling from the model's priority queue.
    # num_replicas: independent backend handles loaded per model.
    #   num_replicas == 1  → workers share one handle (serial inference, low RAM).
    #   num_replicas == N  → each worker owns its handle (parallel, N× RAM).
    num_inference_workers: int = 2
    num_replicas: int = 1

    # ── Queue / batching ─────────────────────────────────────────────────────
    batch_timeout_ms: float = 50.0
    max_batch_size: int = 32
    max_queue_depth: int = 512

    # ── Memory ───────────────────────────────────────────────────────────────
    default_hot_budget_mb: int = 512
    default_warm_budget_mb: int = 2048

    # ── Streaming ────────────────────────────────────────────────────────────
    sse_keep_alive_s: float = 15.0
    max_concurrent_streams: int = 100

    # ── Limits ────────────────────────────────────────────────────────────────
    max_prompt_chars: int = 32_000
    request_timeout_s: float = 300.0
    # Maximum seconds a request may wait in the queue before being rejected.
    queue_timeout_s: float = 30.0
    # 0 = disabled; requests per second allowed per IP (token bucket).
    rate_limit_rps: int = 0

    # ── Multi-model memory management ─────────────────────────────────────────
    # Global cap on total hot-tier RAM across ALL loaded models (MB).
    # 0 = each model independently uses default_hot_budget_mb (no global cap).
    global_hot_budget_mb: int = 0
    # Maximum number of models that may be loaded simultaneously.
    # 0 = unlimited. When exceeded, the LRU model is evicted.
    max_loaded_models: int = 0

    # ── Hardware device ────────────────────────────────────────────────────────
    # Device for PyTorch fallback backend.  Empty string = auto-detect.
    # Accepted: "auto", "cpu", "cuda:0", "cuda:1", "hip:0", "mps", "xpu:0".
    # The Rust paged backend always uses its own device selection (--device flag).
    device: str = ""

    # ── Quantization ──────────────────────────────────────────────────────────
    # Weight quantization for the HuggingFace fallback backend.
    # "none"    — full precision (fp16/bf16)
    # "int8"    — 8-bit quantization (bnb on CUDA, quanto on MPS/XPU/CPU)
    # "int4"    — 4-bit NF4 quantization (bnb on CUDA; quanto int4 on others)
    # "auto"    — best available for device (int4 on CUDA if bnb installed)
    quantization: str = "none"

    # ── Attention implementation ───────────────────────────────────────────────
    # "auto"              — detect best: flash_attention_2 > sdpa > eager
    # "flash_attention_2" — requires `pip install flash-attn`, CUDA only
    # "sdpa"              — PyTorch scaled_dot_product_attention (2.0+)
    # "eager"             — standard PyTorch attention (always available)
    attn_implementation: str = "auto"

    # Apply torch.compile() to the HuggingFace fallback model for extra speed.
    # Adds ~60s compilation overhead on first request. Skip on MPS (unsupported).
    compile_torch: bool = False

    # ── Model routing ─────────────────────────────────────────────────────────
    # Policy for resolving ambiguous model selectors (glob, versioned, A/B).
    # One of: first, round_robin, least_loaded, random, weighted
    routing_policy: str = "least_loaded"

    # ── Shutdown ──────────────────────────────────────────────────────────────
    # Grace period for in-flight requests during shutdown.
    shutdown_grace_s: float = 30.0

    def __post_init__(self) -> None:
        errors = []
        if not (1 <= self.port <= 65535):
            errors.append(f"port must be 1–65535, got {self.port}")
        if self.processes < 1:
            errors.append(f"processes must be >= 1, got {self.processes}")
        if self.num_inference_workers < 1:
            errors.append(f"num_inference_workers must be >= 1, got {self.num_inference_workers}")
        if self.num_replicas < 1:
            errors.append(f"num_replicas must be >= 1, got {self.num_replicas}")
        if self.max_queue_depth < 1:
            errors.append(f"max_queue_depth must be >= 1, got {self.max_queue_depth}")
        if self.request_timeout_s < 1:
            errors.append(f"request_timeout_s must be >= 1, got {self.request_timeout_s}")
        if self.queue_timeout_s < 0:
            errors.append(f"queue_timeout_s must be >= 0, got {self.queue_timeout_s}")
        if self.rate_limit_rps < 0:
            errors.append(f"rate_limit_rps must be >= 0, got {self.rate_limit_rps}")
        if self.global_hot_budget_mb < 0:
            errors.append(f"global_hot_budget_mb must be >= 0, got {self.global_hot_budget_mb}")
        if self.max_loaded_models < 0:
            errors.append(f"max_loaded_models must be >= 0, got {self.max_loaded_models}")
        from nve.serve.model_router import RoutingPolicy
        if self.routing_policy not in RoutingPolicy.ALL:
            errors.append(f"routing_policy must be one of {RoutingPolicy.ALL}, got {self.routing_policy!r}")
        if self.log_level.lower() not in ("debug", "info", "warning", "error", "critical"):
            errors.append(f"log_level must be one of debug/info/warning/error/critical, got {self.log_level!r}")
        valid_quant = {"none", "int8", "int4", "auto"}
        if self.quantization.lower() not in valid_quant:
            errors.append(f"quantization must be one of {valid_quant}, got {self.quantization!r}")
        valid_attn = {"auto", "eager", "sdpa", "flash_attention_2"}
        if self.attn_implementation.lower() not in valid_attn:
            errors.append(f"attn_implementation must be one of {valid_attn}, got {self.attn_implementation!r}")
        if errors:
            raise ValueError("Invalid ServerConfig:\n  " + "\n  ".join(errors))

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=os.getenv("NVE_HOST", "0.0.0.0"),
            port=int(os.getenv("NVE_PORT", "8000")),
            processes=int(os.getenv("NVE_PROCESSES", "1")),
            log_level=os.getenv("NVE_LOG_LEVEL", "info"),
            json_logs=os.getenv("NVE_JSON_LOGS", "1") not in ("0", "false", "no"),
            default_max_tokens=int(os.getenv("NVE_MAX_TOKENS", "512")),
            default_temperature=float(os.getenv("NVE_TEMPERATURE", "0.7")),
            default_top_p=float(os.getenv("NVE_TOP_P", "0.9")),
            num_inference_workers=int(os.getenv("NVE_INFERENCE_WORKERS", "2")),
            num_replicas=int(os.getenv("NVE_REPLICAS", "1")),
            max_batch_size=int(os.getenv("NVE_BATCH_SIZE", "32")),
            batch_timeout_ms=float(os.getenv("NVE_BATCH_TIMEOUT_MS", "50")),
            max_queue_depth=int(os.getenv("NVE_QUEUE_DEPTH", "512")),
            default_hot_budget_mb=int(os.getenv("NVE_HOT_MB", "512")),
            default_warm_budget_mb=int(os.getenv("NVE_WARM_MB", "2048")),
            sse_keep_alive_s=float(os.getenv("NVE_SSE_KEEPALIVE_S", "15")),
            max_concurrent_streams=int(os.getenv("NVE_MAX_STREAMS", "100")),
            request_timeout_s=float(os.getenv("NVE_REQUEST_TIMEOUT_S", "300")),
            queue_timeout_s=float(os.getenv("NVE_QUEUE_TIMEOUT_S", "30")),
            rate_limit_rps=int(os.getenv("NVE_RATE_LIMIT_RPS", "0")),
            shutdown_grace_s=float(os.getenv("NVE_SHUTDOWN_GRACE_S", "30")),
            global_hot_budget_mb=int(os.getenv("NVE_GLOBAL_HOT_MB", "0")),
            max_loaded_models=int(os.getenv("NVE_MAX_LOADED_MODELS", "0")),
            device=os.getenv("NVE_DEVICE", ""),
            quantization=os.getenv("NVE_QUANTIZATION", "none"),
            attn_implementation=os.getenv("NVE_ATTN_IMPL", "auto"),
            compile_torch=os.getenv("NVE_COMPILE_TORCH", "0") not in ("0", "false", "no"),
        )
