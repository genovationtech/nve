"""
NVE Rust Backend — fast inference via the compiled Rust engine.

Loads the libnve shared library and exposes GenericModel's generate()
through ctypes FFI. Uses SIMD-accelerated bf16 dot products, RoPE,
GQA attention, and SwiGLU — all in Rust with zero-copy safetensors.

Falls back gracefully if the Rust library is not available.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nve.rust_backend")


class NveGenerateResult(ctypes.Structure):
    """Mirrors the C struct NveGenerateResult from lib.rs.

    Note: `text` uses c_void_p (not c_char_p) to prevent ctypes from
    auto-converting the pointer to a Python bytes object, which would
    lose the original pointer needed for nve_text_free.
    """
    _fields_ = [
        ("tokens", ctypes.POINTER(ctypes.c_uint32)),
        ("num_tokens", ctypes.c_size_t),
        ("text", ctypes.c_void_p),  # raw pointer, not c_char_p
        ("prefill_time_ms", ctypes.c_double),
        ("decode_time_ms", ctypes.c_double),
        ("total_time_ms", ctypes.c_double),
        ("tokens_per_sec", ctypes.c_double),
        ("prompt_tokens", ctypes.c_size_t),
    ]


class RustInferenceEngine:
    """
    High-performance inference engine backed by the compiled Rust library.

    Usage:
        engine = RustInferenceEngine.load(model_dir)
        if engine is not None:
            result = engine.generate("Hello world", max_new_tokens=50)
    """

    def __init__(self, lib: ctypes.CDLL, model_handle: ctypes.c_void_p):
        self._lib = lib
        self._handle = model_handle

    @classmethod
    def load(cls, model_dir: str | Path, lib_path: Optional[str] = None) -> Optional["RustInferenceEngine"]:
        """
        Load the Rust engine and model. Returns None if unavailable.
        """
        lib = _find_and_load_lib(lib_path)
        if lib is None:
            return None

        _setup_model_bindings(lib)

        model_dir = str(model_dir)
        handle = lib.nve_model_load(model_dir.encode("utf-8"))
        if not handle:
            logger.warning("Rust backend: nve_model_load returned null")
            return None

        logger.info(f"Rust inference engine loaded for {model_dir}")
        return cls(lib, handle)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> dict:
        """
        Generate text using the Rust engine. Returns a dict compatible
        with StreamingServer.generate() output format.
        """
        result = self._lib.nve_model_generate(
            self._handle,
            prompt.encode("utf-8"),
            ctypes.c_size_t(max_new_tokens),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
        )

        # Extract text from raw void pointer.
        text = ""
        text_ptr = result.text
        if text_ptr:
            text = ctypes.cast(text_ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")

        num_generated = result.num_tokens
        tok_per_sec = result.tokens_per_sec

        # Free Rust-allocated memory.
        if result.tokens:
            self._lib.nve_tokens_free(result.tokens, result.num_tokens)
        if text_ptr:
            self._lib.nve_text_free(text_ptr)

        return {
            "text": text,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": num_generated,
            "time_s": round(result.total_time_ms / 1000.0, 3),
            "tokens_per_sec": round(tok_per_sec, 2),
            "prefill_time_ms": round(result.prefill_time_ms, 1),
            "decode_time_ms": round(result.decode_time_ms, 1),
            "backend": "rust",
        }

    def close(self):
        """Free the model."""
        if self._handle:
            self._lib.nve_model_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return f"RustInferenceEngine(loaded={self._handle is not None})"


def _find_and_load_lib(lib_path: Optional[str] = None) -> Optional[ctypes.CDLL]:
    """Find and load libnve.so/.dylib."""
    if lib_path:
        candidates = [Path(lib_path)]
    else:
        base = Path(__file__).parent.parent.parent
        candidates = [
            base / "target" / "release" / "libnve.so",
            base / "target" / "release" / "libnve.dylib",
            base / "target" / "debug" / "libnve.so",
            base / "target" / "debug" / "libnve.dylib",
            # Also check common install locations.
            Path("/usr/local/lib/libnve.so"),
            Path("/usr/lib/libnve.so"),
        ]

    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                logger.info(f"Loaded Rust library from {path}")
                return lib
            except OSError as e:
                logger.debug(f"Failed to load {path}: {e}")

    return None


def _setup_model_bindings(lib: ctypes.CDLL):
    """Configure ctypes signatures for model FFI functions."""
    lib.nve_model_load.restype = ctypes.c_void_p
    lib.nve_model_load.argtypes = [ctypes.c_char_p]

    lib.nve_model_free.restype = None
    lib.nve_model_free.argtypes = [ctypes.c_void_p]

    lib.nve_model_generate.restype = NveGenerateResult
    lib.nve_model_generate.argtypes = [
        ctypes.c_void_p,   # handle
        ctypes.c_char_p,   # prompt
        ctypes.c_size_t,   # max_new_tokens
        ctypes.c_float,    # temperature
        ctypes.c_float,    # top_p
    ]

    lib.nve_tokens_free.restype = None
    lib.nve_tokens_free.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]

    lib.nve_text_free.restype = None
    lib.nve_text_free.argtypes = [ctypes.c_void_p]
