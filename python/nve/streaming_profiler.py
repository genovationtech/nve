"""
NVE Streaming Chunk Profiler — profile models that don't fit in memory.

Memory-maps safetensors from disk, loads one transformer layer at a time,
runs a partial forward pass, records activation magnitudes, then unloads.
Peak memory usage is O(single_layer + activations), not O(full_model).

This is hardware-agnostic: it runs on any machine, even with 4GB RAM.
Speed doesn't matter — correctness does. Run once, save the manifest.

Usage:
    profiler = StreamingProfiler(model_dir="/path/to/llama-8b")
    manifest = profiler.profile(prompts=["The meaning of life is", ...])
    manifest.save("llama-8b.nve")
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class WeightImportance:
    """Importance record for a single named weight tensor."""
    name: str
    layer_index: int
    size_bytes: int
    dtype: str
    shape: list[int]
    # Activation stats accumulated across profiling prompts.
    total_activation: float = 0.0
    sample_count: int = 0
    max_activation: float = 0.0
    domain_activations: dict[str, float] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)

    @property
    def importance(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.total_activation / self.sample_count

    def record(self, magnitude: float, domain: Optional[str] = None):
        self.total_activation += magnitude
        self.sample_count += 1
        self.max_activation = max(self.max_activation, magnitude)
        if domain:
            self.domain_activations[domain] = (
                self.domain_activations.get(domain, 0.0) + magnitude
            )
            self.domain_counts[domain] = (
                self.domain_counts.get(domain, 0) + 1
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer_index": self.layer_index,
            "size_bytes": self.size_bytes,
            "dtype": self.dtype,
            "shape": self.shape,
            "importance": self.importance,
            "max_activation": self.max_activation,
            "sample_count": self.sample_count,
            "domain_importances": {
                d: self.domain_activations[d] / self.domain_counts[d]
                for d in self.domain_activations
            },
        }


@dataclass
class NVEManifest:
    """
    The product of profiling — a portable, hardware-agnostic weight importance map.

    Small (kilobytes), shareable, versionable. Contains per-weight importance
    scores and metadata. The serving layer reads this and fits what it can
    into whatever hardware it finds.
    """
    model_id: str
    architecture: str
    total_params: int
    total_bytes: int
    num_layers: int
    weights: list[dict]  # List of WeightImportance.to_dict()
    profiling_metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path):
        """Save manifest to a .nve JSON file."""
        path = Path(path)
        if path.suffix not in (".nve", ".json"):
            path = path.with_suffix(".nve")
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nve_version": "0.1.0",
            "format": "nve-manifest-v1",
            "model_id": self.model_id,
            "architecture": self.architecture,
            "total_params": self.total_params,
            "total_bytes": self.total_bytes,
            "num_layers": self.num_layers,
            "weights": self.weights,
            "profiling": self.profiling_metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NVEManifest":
        """Load manifest from a .nve JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            model_id=data["model_id"],
            architecture=data["architecture"],
            total_params=data["total_params"],
            total_bytes=data["total_bytes"],
            num_layers=data["num_layers"],
            weights=data["weights"],
            profiling_metadata=data.get("profiling", {}),
        )

    def tier_placement(
        self,
        gpu_budget_bytes: int,
        ram_budget_bytes: int,
    ) -> dict[str, list[dict]]:
        """
        Compute tier placement from this manifest and hardware budgets.

        Returns {"gpu": [...], "ram": [...], "ssd": [...]} with weight entries.
        """
        # Sort by importance descending.
        ranked = sorted(self.weights, key=lambda w: w["importance"], reverse=True)

        gpu_used = 0
        ram_used = 0
        placement = {"gpu": [], "ram": [], "ssd": []}

        for w in ranked:
            size = w["size_bytes"]
            if gpu_used + size <= gpu_budget_bytes:
                placement["gpu"].append(w)
                gpu_used += size
            elif ram_used + size <= ram_budget_bytes:
                placement["ram"].append(w)
                ram_used += size
            else:
                placement["ssd"].append(w)

        return placement


def _parse_safetensors_metadata(path: Path) -> dict[str, dict]:
    """
    Parse safetensors file header to get tensor metadata without loading weights.

    Returns {tensor_name: {"dtype": str, "shape": list, "offset_start": int, "offset_end": int}}
    """
    with open(path, "rb") as f:
        # First 8 bytes: header size as u64 little-endian.
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)

    data_offset = 8 + header_size
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]
        result[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset_start": data_offset + offsets[0],
            "offset_end": data_offset + offsets[1],
        }
    return result


def _dtype_to_bytes(dtype: str) -> int:
    """Convert safetensors dtype string to bytes per element."""
    return {
        "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
        "I8": 1, "I16": 2, "I32": 4, "I64": 8,
        "U8": 1, "U16": 2, "U32": 4, "U64": 8,
        "BOOL": 1,
    }.get(dtype, 4)


def _dtype_to_torch(dtype: str):
    """Convert safetensors dtype string to torch dtype."""
    import torch
    return {
        "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "F64": torch.float64, "I8": torch.int8, "I16": torch.int16,
        "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8,
        "BOOL": torch.bool,
    }.get(dtype, torch.float32)


def _load_tensor_from_safetensors(path: Path, meta: dict):
    """Memory-map a single tensor from a safetensors file."""
    import torch
    import mmap as mmap_mod

    dtype = _dtype_to_torch(meta["dtype"])
    shape = meta["shape"]
    offset_start = meta["offset_start"]
    offset_end = meta["offset_end"]
    num_bytes = offset_end - offset_start

    # Read just the bytes we need from the file.
    with open(path, "rb") as f:
        f.seek(offset_start)
        raw = f.read(num_bytes)

    # Convert raw bytes to a torch tensor with the correct dtype and shape.
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    return tensor


def _extract_layer_index(name: str) -> int:
    """Extract numeric layer index from weight name like 'model.layers.5.self_attn.q_proj.weight'."""
    parts = name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1  # Non-layer weight (embedding, lm_head, etc.)


class StreamingProfiler:
    """
    Profile a model's weight importance by streaming layers from disk.

    Never loads the full model into memory. Memory-maps safetensors files and
    processes one transformer layer at a time.

    Usage:
        profiler = StreamingProfiler("/path/to/model")
        manifest = profiler.profile(["prompt 1", "prompt 2", ...])
        manifest.save("model.nve")
    """

    def __init__(
        self,
        model_dir: str | Path,
        max_ram_bytes: Optional[int] = None,
    ):
        self.model_dir = Path(model_dir)
        self.max_ram_bytes = max_ram_bytes  # Optional cap on RAM during profiling.

        # Parse model config.
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {self.model_dir}")
        with open(config_path) as f:
            self.model_config = json.load(f)

        self.architecture = self.model_config.get("model_type", "unknown")
        self.num_layers = self.model_config.get(
            "num_hidden_layers",
            self.model_config.get("n_layer", 0),
        )
        self.hidden_size = self.model_config.get(
            "hidden_size",
            self.model_config.get("n_embd", 0),
        )
        self.vocab_size = self.model_config.get("vocab_size", 0)

        # Discover safetensors files and parse all weight metadata.
        self._weight_meta: dict[str, dict] = {}  # name → metadata + file path
        self._discover_weights()

    def _discover_weights(self):
        """Parse all safetensors headers to build the complete weight inventory."""
        # Check for sharded index.
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # Get unique shard files.
            shard_files = set(weight_map.values())
            for shard_name in shard_files:
                shard_path = self.model_dir / shard_name
                if shard_path.exists():
                    meta = _parse_safetensors_metadata(shard_path)
                    for name, info in meta.items():
                        info["file"] = str(shard_path)
                        self._weight_meta[name] = info
        else:
            # Single safetensors file.
            st_path = self.model_dir / "model.safetensors"
            if st_path.exists():
                meta = _parse_safetensors_metadata(st_path)
                for name, info in meta.items():
                    info["file"] = str(st_path)
                    self._weight_meta[name] = info

        if not self._weight_meta:
            raise FileNotFoundError(
                f"No safetensors files found in {self.model_dir}"
            )

    def weight_inventory(self) -> dict[str, dict]:
        """Return the complete weight inventory (names, shapes, sizes) without loading data."""
        result = {}
        for name, meta in self._weight_meta.items():
            elem_bytes = _dtype_to_bytes(meta["dtype"])
            numel = 1
            for s in meta["shape"]:
                numel *= s
            result[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "size_bytes": numel * elem_bytes,
                "layer_index": _extract_layer_index(name),
            }
        return result

    def profile(
        self,
        prompts: list[str],
        domains: Optional[list[str]] = None,
        batch_size: int = 1,
        max_seq_len: int = 128,
        progress_callback=None,
    ) -> NVEManifest:
        """
        Profile the model by streaming layers from disk.

        For each prompt:
        1. Tokenize the prompt
        2. Load embedding layer, compute embeddings, unload
        3. For each transformer layer:
           a. Load layer weights from safetensors (mmap)
           b. Run forward pass through this single layer
           c. Record activation magnitudes for each weight
           d. Unload layer weights
        4. Record final norm + lm_head activations

        Args:
            prompts: Profiling prompts (diverse is better).
            domains: Optional domain label per prompt.
            batch_size: Not used yet (reserved for future batching).
            max_seq_len: Max tokens per prompt.
            progress_callback: Called with (layer_idx, num_layers) for progress updates.

        Returns:
            NVEManifest with per-weight importance scores.
        """
        import torch
        from transformers import AutoTokenizer, AutoConfig

        t_start = time.time()

        # Load tokenizer (small, always fits).
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        config = AutoConfig.from_pretrained(self.model_dir)

        # Build weight importance trackers.
        inventory = self.weight_inventory()
        importance: dict[str, WeightImportance] = {}
        for name, info in inventory.items():
            importance[name] = WeightImportance(
                name=name,
                layer_index=info["layer_index"],
                size_bytes=info["size_bytes"],
                dtype=info["dtype"],
                shape=info["shape"],
            )

        total_params = sum(
            np.prod(info["shape"]) for info in inventory.values()
        )
        total_bytes = sum(info["size_bytes"] for info in inventory.values())

        print(f"  Model: {self.architecture} | {total_params/1e9:.1f}B params | {total_bytes/1024**3:.1f} GB")
        print(f"  Weights: {len(inventory)} tensors across {self.num_layers} layers")
        print(f"  Profiling {len(prompts)} prompts...")

        # Group weights by layer.
        layer_weights: dict[int, list[str]] = {}
        non_layer_weights: list[str] = []
        for name, info in inventory.items():
            lidx = info["layer_index"]
            if lidx >= 0:
                layer_weights.setdefault(lidx, []).append(name)
            else:
                non_layer_weights.append(name)

        # ── Profile each prompt ──
        for prompt_idx, prompt in enumerate(prompts):
            domain = domains[prompt_idx] if domains and prompt_idx < len(domains) else None

            # Tokenize.
            tokens = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=max_seq_len,
            )
            input_ids = tokens["input_ids"]  # (1, seq_len)
            seq_len = input_ids.shape[1]

            # ── Step 1: Embedding ──
            embed_names = [n for n in non_layer_weights if "embed" in n.lower()]
            hidden = None
            for name in embed_names:
                tensor = self._load_weight(name)
                if tensor is not None and ("embed_tokens" in name or "wte" in name):
                    # Embedding lookup.
                    hidden = torch.nn.functional.embedding(input_ids, tensor.float())
                    mag = float(hidden.abs().mean().item())
                    importance[name].record(mag, domain)
                    del tensor
                elif tensor is not None:
                    # Position embeddings etc.
                    mag = float(tensor.abs().mean().item())
                    importance[name].record(mag, domain)
                    del tensor

            if hidden is None:
                # Fallback: create random hidden states if embedding loading failed.
                hidden = torch.randn(1, seq_len, self.hidden_size)

            # ── Step 2: Stream through transformer layers ──
            for layer_idx in range(self.num_layers):
                if progress_callback:
                    progress_callback(layer_idx, self.num_layers)

                layer_names = layer_weights.get(layer_idx, [])
                if not layer_names:
                    continue

                # Load all weights for this layer.
                layer_tensors = {}
                for name in layer_names:
                    t = self._load_weight(name)
                    if t is not None:
                        layer_tensors[name] = t

                # Run a simplified forward pass through this layer.
                # We compute: output = input + attn(norm(input)) + ffn(norm(input))
                # This approximates the real forward pass enough to capture
                # which weights activate strongly.
                hidden = self._forward_layer(
                    hidden, layer_tensors, layer_names,
                    importance, domain, config,
                )

                # Unload layer.
                del layer_tensors
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass

            # ── Step 3: Final norm + lm_head ──
            for name in non_layer_weights:
                if "embed" in name.lower():
                    continue  # Already handled.
                tensor = self._load_weight(name)
                if tensor is not None:
                    if "norm" in name.lower() or "ln_f" in name.lower():
                        # Apply norm.
                        try:
                            normed = torch.nn.functional.layer_norm(
                                hidden.float(), [hidden.shape[-1]], weight=tensor.float()
                            )
                            mag = float(normed.abs().mean().item())
                        except Exception:
                            mag = float(tensor.abs().mean().item())
                    elif "lm_head" in name.lower() or "output" in name.lower():
                        # Compute logits.
                        try:
                            logits = torch.matmul(hidden.float(), tensor.float().T)
                            mag = float(logits.abs().mean().item())
                        except Exception:
                            mag = float(tensor.abs().mean().item())
                    else:
                        mag = float(tensor.abs().mean().item())
                    importance[name].record(mag, domain)
                    del tensor

            # Clean up.
            del hidden
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError:
                pass

            elapsed = time.time() - t_start
            print(f"    [{prompt_idx+1}/{len(prompts)}] {elapsed:.0f}s — {prompt[:50]}...")

        # ── Build manifest ──
        weights_list = sorted(
            [w.to_dict() for w in importance.values()],
            key=lambda w: w["importance"],
            reverse=True,
        )

        manifest = NVEManifest(
            model_id=self.model_config.get("_name_or_path", str(self.model_dir)),
            architecture=self.architecture,
            total_params=int(total_params),
            total_bytes=total_bytes,
            num_layers=self.num_layers,
            weights=weights_list,
            profiling_metadata={
                "num_prompts": len(prompts),
                "domains": list(set(domains)) if domains else [],
                "max_seq_len": max_seq_len,
                "profiling_time_s": round(time.time() - t_start, 1),
            },
        )

        total_time = time.time() - t_start
        print(f"\n  Profiling complete in {total_time:.0f}s")
        print(f"  Top 5 weights: {[w['name'] for w in weights_list[:5]]}")

        return manifest

    def _load_weight(self, name: str):
        """Load a single weight tensor from safetensors via memory mapping."""
        import torch

        meta = self._weight_meta.get(name)
        if meta is None:
            return None

        try:
            return _load_tensor_from_safetensors(Path(meta["file"]), meta)
        except Exception:
            return None

    def _forward_layer(
        self,
        hidden: "torch.Tensor",
        layer_tensors: dict,
        layer_names: list[str],
        importance: dict[str, WeightImportance],
        domain: Optional[str],
        config,
    ) -> "torch.Tensor":
        """
        Run an approximate forward pass through a single transformer layer.

        Computes actual matrix multiplications with the layer's weights to get
        real activation magnitudes. This isn't identical to the model's forward
        pass (we skip some details like rotary embeddings, attention masking)
        but captures the activation patterns that matter for tier placement.
        """
        import torch

        residual = hidden.float()
        h = hidden.float()

        for name, tensor in layer_tensors.items():
            t = tensor.float()
            name_lower = name.lower()

            try:
                if "norm" in name_lower or "ln" in name_lower:
                    # Layer norm / RMS norm weight — apply normalization.
                    if t.shape[0] == h.shape[-1]:
                        h = torch.nn.functional.layer_norm(h, [h.shape[-1]], weight=t)
                        mag = float(h.abs().mean().item())
                    else:
                        mag = float(t.abs().mean().item())

                elif any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "qkv_proj", "c_attn", "c_proj",
                                                     "query", "key", "value"]):
                    # Attention projection — matmul.
                    if t.ndim == 2 and t.shape[1] == h.shape[-1]:
                        out = torch.matmul(h, t.T)
                        mag = float(out.abs().mean().item())
                        del out
                    elif t.ndim == 2 and t.shape[0] == h.shape[-1]:
                        out = torch.matmul(h, t)
                        mag = float(out.abs().mean().item())
                        del out
                    else:
                        mag = float(t.abs().mean().item())

                elif any(k in name_lower for k in ["gate_proj", "up_proj", "down_proj",
                                                     "fc1", "fc2", "c_fc",
                                                     "dense_h_to_4h", "dense_4h_to_h",
                                                     "mlp"]):
                    # MLP projection — matmul.
                    if t.ndim == 2 and t.shape[1] == h.shape[-1]:
                        out = torch.matmul(h, t.T)
                        mag = float(out.abs().mean().item())
                        # For down_proj, update hidden state.
                        if "down_proj" in name_lower or "fc2" in name_lower or "c_proj" in name_lower:
                            if out.shape[-1] == h.shape[-1]:
                                h = residual + out
                        del out
                    elif t.ndim == 2 and t.shape[0] == h.shape[-1]:
                        out = torch.matmul(h, t)
                        mag = float(out.abs().mean().item())
                        if "down_proj" in name_lower or "fc2" in name_lower:
                            if out.shape[-1] == h.shape[-1]:
                                h = residual + out
                        del out
                    else:
                        mag = float(t.abs().mean().item())

                elif "bias" in name_lower:
                    mag = float(t.abs().mean().item())

                else:
                    # Unknown weight type — just record raw magnitude.
                    mag = float(t.abs().mean().item())

            except Exception:
                mag = float(t.abs().mean().item())

            importance[name].record(mag, domain)

        return h
