"""
NVE Hub — download and resolve HuggingFace models.

Provides a Python interface for downloading models from HuggingFace Hub
and resolving model IDs to local paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def resolve_model(
    model_id_or_path: str,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """
    Resolve a model ID or local path to a directory containing model files.

    Args:
        model_id_or_path: Either a local directory path or a HuggingFace model ID
                          (e.g. "meta-llama/Llama-3.2-1B", "microsoft/phi-2").
        cache_dir: Optional cache directory. Defaults to ~/.cache/nve/models.

    Returns:
        Path to the model directory containing config.json, tokenizer.json, and safetensors.
    """
    path = Path(model_id_or_path)

    # If it's already a local directory with config.json, use it.
    if path.is_dir() and (path / "config.json").exists():
        return path

    # Parse as a HuggingFace model ID.
    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    model_cache = cache / "models" / model_id_or_path.replace("/", "--")

    # Check cache.
    if model_cache.is_dir() and (model_cache / "config.json").exists():
        return model_cache

    # Download.
    print(f"Downloading model '{model_id_or_path}' to {model_cache}")
    _download_model(model_id_or_path, model_cache)
    return model_cache


def detect_architecture(model_dir: str | Path) -> dict:
    """
    Detect the architecture of a model from its config.json.

    Returns a dict with architecture info including:
    - arch: Architecture name (e.g. "llama", "gpt2", "mistral")
    - hidden_size, num_layers, num_heads, etc.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Detect architecture.
    arch = _detect_arch(config)

    return {
        "arch": arch,
        "model_type": config.get("model_type", "unknown"),
        "hidden_size": config.get("hidden_size", config.get("n_embd")),
        "num_layers": config.get("num_hidden_layers", config.get("n_layer")),
        "num_heads": config.get("num_attention_heads", config.get("n_head")),
        "num_kv_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "intermediate_size": config.get("intermediate_size", config.get("n_inner")),
        "max_position_embeddings": config.get(
            "max_position_embeddings", config.get("n_positions")
        ),
    }


def _detect_arch(config: dict) -> str:
    """Detect architecture from config dict."""
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")

    for a in architectures:
        a_lower = a.lower()
        if "llama" in a_lower:
            return "llama"
        if "mistral" in a_lower:
            return "mistral"
        if "qwen2" in a_lower:
            return "qwen2"
        if "phi3" in a_lower or "phiforcausal" in a_lower:
            return "phi3"
        if "gemma2" in a_lower:
            return "gemma2"
        if "gemma" in a_lower:
            return "gemma"
        if "gptneox" in a_lower:
            return "gpt_neox"
        if "gpt2" in a_lower:
            return "gpt2"
        if "falcon" in a_lower or "rwforcausal" in a_lower:
            return "falcon"
        if "stablelm" in a_lower:
            return "stablelm"
        if "starcoder2" in a_lower:
            return "starcoder2"
        if "internlm2" in a_lower:
            return "internlm2"
        if "olmo" in a_lower:
            return "olmo"
        if "deepseek" in a_lower:
            return "deepseek"

    # Fallback to model_type.
    return model_type or "unknown"


def _default_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "nve"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "nve"
    return Path.home() / ".cache" / "nve"


def _download_model(model_id: str, dest: Path):
    """Download model files from HuggingFace Hub."""
    dest.mkdir(parents=True, exist_ok=True)

    # Try huggingface-cli first.
    if shutil.which("huggingface-cli"):
        result = subprocess.run(
            [
                "huggingface-cli", "download", model_id,
                "--local-dir", str(dest),
                "--include", "config.json",
                "--include", "tokenizer.json",
                "--include", "tokenizer_config.json",
                "--include", "*.safetensors",
                "--include", "*.safetensors.index.json",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return

    # Try huggingface_hub Python package.
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            model_id,
            local_dir=str(dest),
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
        return
    except ImportError:
        pass

    # Fall back to curl.
    revision = "main"
    base_url = f"https://huggingface.co/{model_id}/resolve/{revision}"
    token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    for filename in ["config.json", "tokenizer.json"]:
        _curl_download(f"{base_url}/{filename}", dest / filename, token)

    for filename in ["tokenizer_config.json", "model.safetensors", "model.safetensors.index.json"]:
        try:
            _curl_download(f"{base_url}/{filename}", dest / filename, token)
        except Exception:
            pass

    # Download shards if index exists.
    index_path = dest / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shards = set(index.get("weight_map", {}).values())
        for shard in shards:
            shard_path = dest / shard
            if not shard_path.exists():
                _curl_download(f"{base_url}/{shard}", shard_path, token)


def _curl_download(url: str, dest: Path, token: Optional[str] = None):
    """Download a file using curl."""
    cmd = ["curl", "-fSL", "-o", str(dest), "--progress-bar"]
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {result.stderr.strip()}")
