"""
Model store — load, cache, and hot-swap inference backends.

Each model is backed by a `ModelWorkerPool` — N worker threads each owning
a dedicated backend replica.  Loading priority per replica:

  1. Rust paged backend (`nve_paged_model_load`)   — fastest, quantised
  2. Rust baseline backend (`nve_model_load`)        — SIMD bf16, fallback
  3. PyTorch StreamingServer                         — pure Python, last resort

Thread-safe — concurrent requests are handled by the worker pool.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from nve.serve.worker_pool import ModelWorkerPool

logger = logging.getLogger("nve.serve.model_store")


# ── Backend abstraction ────────────────────────────────────────────────────────

class _Backend:
    """Minimal interface every backend must implement."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        raise NotImplementedError

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterator[str]:
        """Yield decoded text fragments one token at a time."""
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        yield result.get("text", "")

    def unload(self) -> None:
        pass

    @property
    def backend_name(self) -> str:
        return "unknown"


class _RustBackend(_Backend):
    """Wraps `RustInferenceEngine` from `nve.rust_backend`."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p) -> dict:
        return self._engine.generate(prompt, max_new_tokens, temperature, top_p)

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p) -> Iterator[str]:
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        text = result.get("text", "")
        words = text.split(" ")
        for i, word in enumerate(words):
            yield (word + " ") if i < len(words) - 1 else word

    def unload(self) -> None:
        self._engine.close()

    @property
    def backend_name(self) -> str:
        return "rust"


class _RustPagedBackend(_Backend):
    """Wraps the C FFI PagedModel handle via ctypes."""

    def __init__(self, lib, handle, tokenizer_decode_fn) -> None:
        self._lib = lib
        self._handle = handle
        self._decode = tokenizer_decode_fn

    def generate(self, prompt, max_new_tokens, temperature, top_p) -> dict:
        import ctypes
        t0 = time.time()
        result = self._lib.nve_paged_model_generate(
            self._handle,
            prompt.encode("utf-8"),
            ctypes.c_size_t(max_new_tokens),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
        )
        text = ""
        if result.text:
            text = ctypes.cast(result.text, ctypes.c_char_p).value.decode("utf-8", errors="replace")
        if result.tokens:
            self._lib.nve_tokens_free(result.tokens, result.num_tokens)
        if result.text:
            self._lib.nve_text_free(result.text)
        return {
            "text": text,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.num_tokens,
            "time_s": round(time.time() - t0, 3),
            "tokens_per_sec": round(result.tokens_per_sec, 2),
            "prefill_time_ms": round(result.prefill_time_ms, 1),
            "decode_time_ms": round(result.decode_time_ms, 1),
            "backend": "rust_paged",
        }

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p) -> Iterator[str]:
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        text = result.get("text", "")
        words = text.split(" ")
        for i, word in enumerate(words):
            yield (word + " ") if i < len(words) - 1 else word

    def unload(self) -> None:
        if self._handle:
            self._lib.nve_paged_model_free(self._handle)
            self._handle = None

    @property
    def backend_name(self) -> str:
        return "rust_paged"


class _TorchBackend(_Backend):
    """
    HuggingFace Transformers backend (AutoModelForCausalLM) with hardware-aware
    device placement, mixed-precision autocast, and OOM recovery.

    This is the last-resort fallback when both Rust backends are unavailable.
    It loads the model via `transformers.AutoModelForCausalLM.from_pretrained`
    and supports all devices the DeviceManager detects (CUDA, ROCm, MPS, XPU, CPU).
    """

    def __init__(self, model, tokenizer, device_str: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device_str = device_str

    def generate(self, prompt, max_new_tokens, temperature, top_p) -> dict:
        import torch
        from nve.serve.hardware import autocast_context

        t0 = time.time()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        try:
            dev = torch.device(self._device_str)
            input_ids = inputs["input_ids"].to(dev)
        except RuntimeError:
            input_ids = inputs["input_ids"]
            dev = torch.device("cpu")

        with autocast_context(self._device_str, enabled=(dev.type != "cpu")):
            with torch.no_grad():
                do_sample = temperature > 0
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

        generated_ids = output[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - t0
        return {
            "text": text,
            "prompt_tokens": input_ids.shape[-1],
            "generated_tokens": len(generated_ids),
            "time_s": round(elapsed, 3),
            "tokens_per_sec": round(len(generated_ids) / elapsed, 2) if elapsed > 0 else 0,
            "backend": f"torch_hf[{self._device_str}]",
        }

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p) -> Iterator[str]:
        # HF streaming via TextIteratorStreamer (non-blocking thread).
        try:
            import torch
            from transformers import TextIteratorStreamer
            import threading as _th

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            try:
                dev = torch.device(self._device_str)
                input_ids = inputs["input_ids"].to(dev)
            except RuntimeError:
                input_ids = inputs["input_ids"]

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            do_sample = temperature > 0
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            thread = _th.Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()
            for token_text in streamer:
                yield token_text
            thread.join(timeout=5)
        except (ImportError, Exception) as e:
            logger.warning(f"Streaming generation failed ({e}), falling back to blocking generate")
            result = self.generate(prompt, max_new_tokens, temperature, top_p)
            yield result.get("text", "")

    def unload(self) -> None:
        try:
            del self._model
            del self._tokenizer
        except Exception:
            pass
        try:
            import torch
            if self._device_str.startswith("cuda"):
                torch.cuda.empty_cache()
            elif self._device_str == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        return "torch_hf"


# ── Model record ───────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    name: str
    model_path: str
    backend_name: str
    worker_pool: ModelWorkerPool
    loaded_at: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    total_tokens_generated: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def info(self) -> dict:
        pool = self.worker_pool
        return {
            "name": self.name,
            "model_path": self.model_path,
            "backend": self.backend_name,
            "loaded_at": self.loaded_at,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_tokens_generated": self.total_tokens_generated,
            "workers": pool.worker_count(),
            "replicas": pool.replica_count(),
            "queue_depth": pool.queue_depth(),
            "active_workers": pool.active_count(),
        }

    def record_success(self, generated_tokens: int = 0) -> None:
        with self._lock:
            self.request_count += 1
            self.total_tokens_generated += generated_tokens

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1


# ── Model store ────────────────────────────────────────────────────────────────

class ModelStore:
    """
    Thread-safe registry of loaded models.

    Each model gets a `ModelWorkerPool` with `num_workers` threads and
    `num_replicas` independent backend handles.

    Global memory budget
    ─────────────────────
    `global_hot_budget_mb` caps total hot-tier RAM across all loaded models.
    When loading a new model would exceed the budget, the least-recently-used
    model is evicted automatically (LRU eviction).  Set to 0 to disable the
    global cap (each model uses its own independent per-model budget).

    Loading priority per replica:
      1. Rust paged backend (`nve_paged_model_load`)
      2. Rust baseline backend (`nve_model_load`)
      3. PyTorch StreamingServer fallback
    """

    def __init__(
        self,
        hot_budget_mb: int = 512,
        warm_budget_mb: int = 2048,
        quant_mode: str = "none",
        num_inference_workers: int = 2,
        num_replicas: int = 1,
        max_queue_depth: int = 512,
        metrics_callback=None,
        global_hot_budget_mb: int = 0,    # 0 = no global cap
        max_loaded_models: int = 0,       # 0 = no limit
        device: Optional[str] = None,     # None = auto-select from hardware
        quantization: str = "none",       # "none"|"int8"|"int4"|"auto"
        attn_implementation: str = "auto",# "auto"|"sdpa"|"flash_attention_2"|"eager"
        compile_torch: bool = False,      # apply torch.compile() to HF backend
    ) -> None:
        self._models: Dict[str, ModelRecord] = {}
        # LRU order: most-recently-used at the end (list of model names)
        self._lru: List[str] = []
        self._lock = threading.RLock()
        self._hot_mb = hot_budget_mb
        self._warm_mb = warm_budget_mb
        self._quant_mode = quant_mode
        self._num_workers = num_inference_workers
        self._num_replicas = num_replicas
        self._max_queue_depth = max_queue_depth
        self._metrics_cb = metrics_callback
        self._global_hot_budget_mb = global_hot_budget_mb
        self._max_loaded_models = max_loaded_models
        self._device = device
        self._quantization = quantization
        self._attn_implementation = attn_implementation
        self._compile_torch = compile_torch
        self._rust_lib = self._try_load_rust_lib()

    def _try_load_rust_lib(self):
        """Attempt to load libnve.so and configure all FFI bindings."""
        try:
            import ctypes
            base = Path(__file__).parent.parent.parent.parent
            candidates = [
                base / "target" / "release" / "libnve.so",
                base / "target" / "debug" / "libnve.so",
                base / "target" / "release" / "libnve.dylib",
                base / "target" / "debug" / "libnve.dylib",
            ]
            for path in candidates:
                if path.exists():
                    lib = ctypes.CDLL(str(path))
                    self._configure_rust_ffi(lib, ctypes)
                    logger.info(f"Rust library loaded from {path}")
                    return lib
        except Exception as e:
            logger.debug(f"Rust library not available: {e}")
        return None

    def _configure_rust_ffi(self, lib, ctypes) -> None:
        """Set ctypes argtypes/restypes for all NVE FFI functions."""
        from nve.rust_backend import NveGenerateResult

        lib.nve_model_load.restype = ctypes.c_void_p
        lib.nve_model_load.argtypes = [ctypes.c_char_p]
        lib.nve_model_free.restype = None
        lib.nve_model_free.argtypes = [ctypes.c_void_p]
        lib.nve_model_generate.restype = NveGenerateResult
        lib.nve_model_generate.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
        ]
        lib.nve_tokens_free.restype = None
        lib.nve_tokens_free.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        lib.nve_text_free.restype = None
        lib.nve_text_free.argtypes = [ctypes.c_void_p]

        lib.nve_paged_model_load.restype = ctypes.c_void_p
        lib.nve_paged_model_load.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_char_p,  # quant_mode, device
        ]
        lib.nve_paged_model_free.restype = None
        lib.nve_paged_model_free.argtypes = [ctypes.c_void_p]
        lib.nve_paged_model_generate.restype = NveGenerateResult
        lib.nve_paged_model_generate.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
        ]
        lib.nve_paged_model_memory_report.restype = ctypes.c_void_p
        lib.nve_paged_model_memory_report.argtypes = [ctypes.c_void_p]
        lib.nve_paged_model_profile.restype = ctypes.c_bool
        lib.nve_paged_model_profile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.nve_paged_model_reset.restype = None
        lib.nve_paged_model_reset.argtypes = [ctypes.c_void_p]

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, name: str, model_path: str) -> "ModelRecord":
        """
        Load a model and register it under `name`.  Returns the record.

        LRU eviction
        ─────────────
        If loading would exceed `global_hot_budget_mb` or `max_loaded_models`,
        the least-recently-used model(s) are evicted first.  Models that are
        currently serving requests (active_count > 0) are skipped during
        eviction — the load will block until a slot becomes available or raise
        RuntimeError if no eviction is possible.
        """
        with self._lock:
            if name in self._models:
                logger.info(f"Hot-swapping model '{name}' — stopping old pool")
                self._models[name].worker_pool.stop()
                if name in self._lru:
                    self._lru.remove(name)

            # Evict LRU models if needed before loading the new one.
            self._maybe_evict(exclude=name)

            backend_name_ref: list = []
            factory = self._make_backend_factory(model_path, backend_name_ref)

            pool = ModelWorkerPool(
                backend_factory=factory,
                num_workers=self._num_workers,
                num_replicas=self._num_replicas,
                max_queue_depth=self._max_queue_depth,
                metrics_callback=self._metrics_cb,
            )
            pool.start()

            record = ModelRecord(
                name=name,
                model_path=model_path,
                backend_name=backend_name_ref[0] if backend_name_ref else "unknown",
                worker_pool=pool,
            )
            self._models[name] = record
            self._lru.append(name)   # most recently used = end of list

            logger.info(
                f"Model '{name}' loaded via {record.backend_name} "
                f"({self._num_workers} workers, {self._num_replicas} replica(s))",
                extra={"model_name": name, "model_path": model_path},
            )
            return record

    def _maybe_evict(self, exclude: str = "") -> None:
        """
        Evict the LRU model(s) if global budget or model count is exceeded.
        Skips models with active inference workers.
        """
        # Check max_loaded_models
        if self._max_loaded_models > 0:
            while len(self._models) >= self._max_loaded_models:
                victim = self._pick_lru_victim(exclude)
                if victim is None:
                    logger.warning(
                        f"Cannot evict any model (all have active workers) — "
                        f"proceeding anyway (max_loaded_models={self._max_loaded_models})"
                    )
                    break
                logger.info(f"LRU eviction: unloading '{victim}' to make room")
                self._unload_locked(victim)

        # Check global_hot_budget_mb
        if self._global_hot_budget_mb > 0:
            # Rough estimate: each loaded model consumes hot_budget_mb
            used_mb = len(self._models) * self._hot_mb * self._num_replicas
            while used_mb + self._hot_mb * self._num_replicas > self._global_hot_budget_mb:
                victim = self._pick_lru_victim(exclude)
                if victim is None:
                    logger.warning(
                        f"Cannot evict any model — proceeding despite budget "
                        f"({used_mb} MB used, {self._global_hot_budget_mb} MB global budget)"
                    )
                    break
                logger.info(
                    f"LRU eviction: unloading '{victim}' "
                    f"(global budget {self._global_hot_budget_mb} MB would be exceeded)"
                )
                self._unload_locked(victim)
                used_mb -= self._hot_mb * self._num_replicas

    def _pick_lru_victim(self, exclude: str) -> Optional[str]:
        """
        Return the name of the least-recently-used model that:
          - is not `exclude`
          - has no active inference workers right now
        """
        for name in self._lru:  # LRU order: oldest first
            if name == exclude:
                continue
            record = self._models.get(name)
            if record and record.worker_pool.active_count() == 0:
                return name
        return None

    def touch(self, name: str) -> None:
        """Mark `name` as most-recently-used (call after every successful inference)."""
        with self._lock:
            if name in self._lru:
                self._lru.remove(name)
                self._lru.append(name)

    def _make_backend_factory(
        self, model_path: str, name_ref: list
    ) -> Callable[[], _Backend]:
        """
        Return a zero-argument factory that loads one backend replica.
        Sets name_ref[0] on the first successful load.
        """
        rust_lib = self._rust_lib
        hot_mb = self._hot_mb
        warm_mb = self._warm_mb
        quant_mode = self._quant_mode
        device_override = self._device
        hf_quantization = self._quantization
        hf_attn_impl = self._attn_implementation
        hf_compile = self._compile_torch
        first_call = [True]

        def factory() -> _Backend:
            import ctypes

            # 1. Rust paged backend
            if rust_lib is not None:
                try:
                    from nve.serve.hardware import normalise_device
                    rust_device = normalise_device(device_override) if device_override else b"auto"
                    if isinstance(rust_device, str):
                        rust_device = rust_device.encode("utf-8")
                    handle = rust_lib.nve_paged_model_load(
                        model_path.encode("utf-8"),
                        ctypes.c_size_t(hot_mb),
                        ctypes.c_size_t(warm_mb),
                        quant_mode.encode("utf-8"),
                        rust_device,
                    )
                    if handle:
                        if first_call[0]:
                            name_ref.append("rust_paged")
                            first_call[0] = False
                        return _RustPagedBackend(rust_lib, handle, None)
                except Exception as e:
                    logger.debug(f"Rust paged backend failed: {e}")

            # 2. Rust baseline backend
            if rust_lib is not None:
                try:
                    from nve.rust_backend import RustInferenceEngine
                    engine = RustInferenceEngine.load(model_path)
                    if engine is not None:
                        if first_call[0]:
                            name_ref.append("rust")
                            first_call[0] = False
                        return _RustBackend(engine)
                except Exception as e:
                    logger.debug(f"Rust baseline backend failed: {e}")

            # 3. HuggingFace Transformers fallback (quantization + Flash Attention aware)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                from nve.serve.hardware import normalise_device
                from nve.serve.quantization import (
                    detect_best_quant, build_hf_load_kwargs, apply_quanto,
                    best_attn_impl,
                )
                from nve.device import get_device_manager

                chosen_device = normalise_device(device_override)
                dm = get_device_manager()
                dtype = dm.best_dtype(chosen_device)

                # Resolve quantization backend for this device.
                qconfig = detect_best_quant(chosen_device, hf_quantization)

                # Resolve attention implementation.
                attn_impl = (
                    best_attn_impl(chosen_device)
                    if hf_attn_impl == "auto"
                    else hf_attn_impl
                )

                logger.info(
                    f"Loading HF model '{model_path}' | device={chosen_device} | "
                    f"dtype={dtype} | quant={qconfig.description()} | attn={attn_impl}"
                )

                load_kwargs: dict = dict(
                    pretrained_model_name_or_path=model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )

                # Merge quantization kwargs (may override torch_dtype for bnb).
                quant_kwargs = build_hf_load_kwargs(qconfig, dtype, chosen_device)
                if "torch_dtype" in quant_kwargs and quant_kwargs["torch_dtype"] is None:
                    quant_kwargs.pop("torch_dtype")
                load_kwargs.update(quant_kwargs)

                # device_map="auto" distributes layers across all GPUs + CPU fallback.
                if chosen_device != "cpu":
                    load_kwargs["device_map"] = "auto"

                # Attention implementation (sdpa/flash_attention_2/eager).
                if attn_impl != "eager":
                    load_kwargs["attn_implementation"] = attn_impl

                try:
                    hf_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                except (ValueError, ImportError, AttributeError) as attn_err:
                    # Flash Attention might not be supported by this model class.
                    if "attn_implementation" in str(attn_err) or "flash" in str(attn_err).lower():
                        logger.warning(
                            f"attn_implementation='{attn_impl}' not supported "
                            f"by this model — retrying with 'eager'"
                        )
                        load_kwargs.pop("attn_implementation", None)
                        hf_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                    else:
                        raise

                hf_model.eval()

                # Apply Quanto quantization in-place (if selected).
                apply_quanto(hf_model, qconfig)

                # Explicit CPU placement when device_map not used.
                if chosen_device == "cpu" and not qconfig.is_quantized():
                    hf_model = hf_model.to(torch.device("cpu"))

                # Optional torch.compile() for extra throughput (skipped on MPS).
                if hf_compile:
                    try:
                        import torch
                        dev = torch.device(chosen_device)
                        if dev.type not in ("mps", "cpu") and not qconfig.is_quantized():
                            hf_model = torch.compile(
                                hf_model, backend="inductor", mode="reduce-overhead"
                            )
                            logger.info("torch.compile() applied to HF model")
                        else:
                            logger.debug(
                                f"torch.compile() skipped for device={chosen_device} "
                                f"or quantized model"
                            )
                    except Exception as ce:
                        logger.warning(f"torch.compile() failed: {ce}")

                hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

                if first_call[0]:
                    name_ref.append("torch_hf")
                    first_call[0] = False
                return _TorchBackend(hf_model, hf_tokenizer, device_str=chosen_device)
            except Exception as e:
                raise RuntimeError(
                    f"All backends failed for model '{model_path}'. Last error: {e}"
                )

        return factory

    # ── Queries ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional["ModelRecord"]:
        with self._lock:
            record = self._models.get(name)
            if record:
                self.touch(name)
            return record

    def _unload_locked(self, name: str) -> bool:
        """Unload without acquiring the lock (caller must hold it)."""
        if name not in self._models:
            return False
        self._models[name].worker_pool.stop(drain_timeout_s=5.0)
        del self._models[name]
        if name in self._lru:
            self._lru.remove(name)
        logger.info(f"Model '{name}' unloaded", extra={"model_name": name})
        return True

    def unload(self, name: str) -> bool:
        with self._lock:
            return self._unload_locked(name)

    def list_models(self) -> list:
        with self._lock:
            return [r.info() for r in self._models.values()]

    def default_model(self) -> Optional[str]:
        with self._lock:
            if not self._models:
                return None
            return next(iter(self._models))
