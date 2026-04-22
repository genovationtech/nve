#!/usr/bin/env python3
"""
NVE GPU Benchmark Suite — Modal Cloud Runner
=============================================
Builds NVE from source with --features cuda inside a CUDA container, then runs
the full ABC benchmark across a wide range of public models.

GPU (A10G, 24 GB VRAM) is used for:
  - AWQ saliency profiling matmuls (Configs B and C)
  - Hot-tier weight placement during profiling

Models tested (grouped by VRAM / RAM headroom):
  Small  (≤ 2B params, 24 GB RAM):
    • openai-community/gpt2          117M
    • Qwen/Qwen2.5-0.5B              494M
    • Qwen/Qwen2.5-1.5B              1.5B
    • meta-llama/Llama-3.2-1B        1.2B

  Medium (2–4B params, 32 GB RAM):
    • Qwen/Qwen2.5-3B                3.1B
    • meta-llama/Llama-3.2-3B        3.2B
    • microsoft/Phi-3.5-mini-instruct 3.8B

  Large  (≥ 7B params, 48 GB RAM):
    • meta-llama/Llama-3.1-8B        8.0B
    • Qwen/Qwen2.5-7B                7.6B

Usage:
  modal run evidence/modal_experiments.py
  modal run evidence/modal_experiments.py --size small
  modal run evidence/modal_experiments.py --size medium
  modal run evidence/modal_experiments.py --size large
  modal run evidence/modal_experiments.py --size all
"""

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent.parent  # nve/ root

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("nve-gpu-benchmark")

# ── Shared model-weight volume (persisted across runs) ────────────────────────
model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)

# ── CUDA-enabled image: builds NVE from source with --features cuda ───────────
# Cached after first build; subsequent runs reuse the image layer.
nve_image = (
    # Use a current, non-deprecated CUDA image with both devel headers (for
    # building) and runtime libraries (for candle-core GPU dispatch at runtime).
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ca-certificates", "curl", "build-essential", "pkg-config", "libssl-dev",
    )
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable",
        # Make CUDA runtime discoverable by the linker and at runtime.
        "echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf && ldconfig",
    )
    # Copy only the source tree — exclude target/ to force a clean CUDA build
    # and avoid stale CPU artifacts from the local machine polluting the build.
    .add_local_dir(str(HERE / "src"),         "/build/nve/src",         copy=True)
    .add_local_dir(str(HERE / "tests"),       "/build/nve/tests",       copy=True)
    .add_local_file(str(HERE / "Cargo.toml"), "/build/nve/Cargo.toml",  copy=True)
    .add_local_file(str(HERE / "Cargo.lock"), "/build/nve/Cargo.lock",  copy=True)
    .run_commands(
        # Build with CUDA feature so profiling matmuls dispatch to GPU.
        # CUDA_COMPUTE_CAP: candle-kernels calls nvidia-smi to detect this at
        # build time, but image builders are CPU-only containers — no GPU/driver.
        # A10G is Ampere sm_86; setting this env var skips the nvidia-smi probe.
        # RUSTFLAGS: target-cpu=x86-64 ensures the binary runs on any x86-64
        # instance regardless of which AVX extensions the build host exposes.
        "bash -c '"
        "source $HOME/.cargo/env && "
        "cd /build/nve && "
        "CUDA_PATH=/usr/local/cuda "
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH "
        "CUDA_COMPUTE_CAP=86 "
        "RUSTFLAGS=\"-C target-cpu=x86-64\" "
        "cargo build --release --features cuda 2>&1"
        "'",
        "cp /build/nve/target/release/nve /usr/local/bin/nve",
        "chmod +x /usr/local/bin/nve",
        # Smoke-test: version + device enumeration confirms CUDA is visible.
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64 nve --version",
        "LD_LIBRARY_PATH=/usr/local/cuda/lib64 nve devices || true",
    )
    .pip_install("huggingface_hub[hf_xfer]", "tqdm")
)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (repo_id, local_cache_dir, size_group)
# size_group controls which container spec is used.
MODELS = {
    # ── Small (≤ 2B) ──────────────────────────────────────────────────────────
    "gpt2": {
        "repo_id": "openai-community/gpt2",
        "local_dir": "/models/gpt2",
        "size": "small",
        "label": "GPT-2 (117M)",
        "gated": False,
    },
    "qwen_0_5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B",
        "local_dir": "/models/qwen_0_5b",
        "size": "small",
        "label": "Qwen2.5-0.5B",
        "gated": False,
    },
    "qwen_1_5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B",
        "local_dir": "/models/qwen_1_5b",
        "size": "small",
        "label": "Qwen2.5-1.5B",
        "gated": False,
    },
    "llama_1b": {
        "repo_id": "meta-llama/Llama-3.2-1B",
        "local_dir": "/models/llama_1b",
        "size": "small",
        "label": "Llama-3.2-1B",
        "gated": True,
    },
    # ── Medium (2–4B) ─────────────────────────────────────────────────────────
    "qwen_3b": {
        "repo_id": "Qwen/Qwen2.5-3B",
        "local_dir": "/models/qwen_3b",
        "size": "medium",
        "label": "Qwen2.5-3B",
        "gated": False,
    },
    "llama_3b": {
        "repo_id": "meta-llama/Llama-3.2-3B",
        "local_dir": "/models/llama_3b",
        "size": "medium",
        "label": "Llama-3.2-3B",
        "gated": True,
    },
    "phi_3_5": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "local_dir": "/models/phi_3_5",
        "size": "medium",
        "label": "Phi-3.5-mini (3.8B)",
        "gated": False,
    },
    # ── Large (≥ 7B) ──────────────────────────────────────────────────────────
    "qwen_7b": {
        "repo_id": "Qwen/Qwen2.5-7B",
        "local_dir": "/models/qwen_7b",
        "size": "large",
        "label": "Qwen2.5-7B",
        "gated": False,
    },
    "llama_8b": {
        "repo_id": "meta-llama/Llama-3.1-8B",
        "local_dir": "/models/llama_8b",
        "size": "large",
        "label": "Llama-3.1-8B",
        "gated": True,
    },
}


# ── Shared benchmark runner ───────────────────────────────────────────────────

def _download_model(repo_id: str, local_dir: str, hf_token: str, vol) -> None:
    """Download model weights to volume if not already cached."""
    import time
    from huggingface_hub import snapshot_download

    marker = Path(local_dir) / ".downloaded"
    if marker.exists():
        print(f"[cache] {repo_id} already in volume")
        return

    print(f"[download] {repo_id} → {local_dir} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token,
        ignore_patterns=["*.bin", "original/*", "*.gguf"],
    )
    marker.touch()
    vol.commit()
    print(f"  done in {time.time() - t0:.0f}s")


def _run_abc(label: str, model_dir: str, out_file: str, extra_args: list[str]) -> dict | None:
    """Run nve abc-test and return parsed JSON or None on failure."""
    import os
    import subprocess
    import time

    out_path = Path("/results") / out_file
    cmd = [
        "nve", "abc-test",
        "-m", model_dir,
        "--auto-budget",
        "--target-bpw", "2.0",
        "--device", "auto",   # GPU profiling when CUDA build detected
        "-n", "50",
        "-o", str(out_path),
    ] + extra_args

    # Ensure CUDA runtime is findable by the nve binary at runtime.
    env = os.environ.copy()
    cuda_lib = "/usr/local/cuda/lib64"
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"\n{'=' * 60}\n[{label}]\ncmd: {' '.join(cmd)}\n{'=' * 60}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=False, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ERROR exit={proc.returncode} after {elapsed:.0f}s")
        return None

    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        print(f"\n  ── {label} results ({elapsed:.0f}s) ──")
        print(f"  {'Config':<28}  {'Acc':>6}  {'Tok/s':>8}  {'Mem MB':>8}")
        print(f"  {'─' * 28}  {'─' * 6}  {'─' * 8}  {'─' * 8}")
        for cfg in data.get("configurations", []):
            acc = cfg.get("task_accuracy", 0)
            tps = cfg.get("summary", {}).get("avg_tokens_per_sec", 0)
            mem = cfg.get("summary", {}).get("peak_memory_mb", 0)
            acc_s = f"{acc:.0%}" if isinstance(acc, float) else str(acc)
            tps_s = f"{tps:.1f}" if isinstance(tps, float) else str(tps)
            mem_s = f"{mem:.0f}" if isinstance(mem, float) else str(mem)
            print(f"  {cfg['config']:<28}  {acc_s:>6}  {tps_s:>8}  {mem_s:>8}")
        return data
    return None


# ── Small-model container (4 vCPU, 24 GB RAM, A10G GPU) ─────────────────────

@app.function(
    image=nve_image,
    gpu="A10G",
    cpu=4.0,
    memory=24576,           # 24 GB RAM — ample for ≤ 2B models
    timeout=3 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_small_models() -> dict[str, str]:
    """Run ABC benchmark on all small models (≤ 2B params)."""
    import os

    hf_token = os.environ["HF_TOKEN"]
    outdir = Path("/results")
    outdir.mkdir(exist_ok=True)

    keys = [k for k, v in MODELS.items() if v["size"] == "small"]
    results: dict[str, str] = {}

    for key in keys:
        m = MODELS[key]
        _download_model(m["repo_id"], m["local_dir"], hf_token, model_vol)

        data = _run_abc(m["label"], m["local_dir"], f"{key}_gpu.json", [])
        if data:
            results[key] = json.dumps(data)

    return results


# ── Medium-model container (4 vCPU, 32 GB RAM, A10G GPU) ────────────────────

@app.function(
    image=nve_image,
    gpu="A10G",
    cpu=4.0,
    memory=32768,           # 32 GB RAM — covers 3–4B models with headroom
    timeout=4 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_medium_models() -> dict[str, str]:
    """Run ABC benchmark on medium models (2–4B params)."""
    import os

    hf_token = os.environ["HF_TOKEN"]
    outdir = Path("/results")
    outdir.mkdir(exist_ok=True)

    keys = [k for k, v in MODELS.items() if v["size"] == "medium"]
    results: dict[str, str] = {}

    for key in keys:
        m = MODELS[key]
        _download_model(m["repo_id"], m["local_dir"], hf_token, model_vol)

        data = _run_abc(m["label"], m["local_dir"], f"{key}_gpu.json", [])
        if data:
            results[key] = json.dumps(data)

    return results


# ── Large-model container (8 vCPU, 48 GB RAM, A10G GPU) ─────────────────────

@app.function(
    image=nve_image,
    gpu="A10G",
    cpu=8.0,
    memory=49152,           # 48 GB RAM — 8B bf16 ≈ 16 GB weights + NVE overhead
    timeout=6 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_large_models() -> dict[str, str]:
    """Run ABC benchmark on large models (≥ 7B params)."""
    import os

    hf_token = os.environ["HF_TOKEN"]
    outdir = Path("/results")
    outdir.mkdir(exist_ok=True)

    keys = [k for k, v in MODELS.items() if v["size"] == "large"]
    results: dict[str, str] = {}

    for key in keys:
        m = MODELS[key]
        _download_model(m["repo_id"], m["local_dir"], hf_token, model_vol)

        data = _run_abc(m["label"], m["local_dir"], f"{key}_gpu.json", [])
        if data:
            results[key] = json.dumps(data)

    return results


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(size: str = "all"):
    """
    Run GPU benchmarks across all model size groups.

    --size small   → GPT-2, Qwen-0.5B, Qwen-1.5B, Llama-1B
    --size medium  → Qwen-3B, Llama-3B, Phi-3.5-mini
    --size large   → Qwen-7B, Llama-8B
    --size all     → all of the above (default, runs in parallel)
    """
    print(f"NVE GPU Benchmark Suite — size={size}")
    print("GPU: A10G (24 GB VRAM) — profiling matmuls dispatched to CUDA\n")

    runners = {
        "small":  run_small_models,
        "medium": run_medium_models,
        "large":  run_large_models,
    }

    groups_to_run = list(runners.keys()) if size == "all" else [size]
    for g in groups_to_run:
        if g not in runners:
            print(f"Unknown size group '{g}'. Choose: small, medium, large, all")
            sys.exit(1)

    out_dir = HERE / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    if size == "all":
        # Spawn all three groups in parallel; collect results as each completes.
        print("Submitting all three size groups in parallel...\n")
        futures = {g: runners[g].spawn() for g in groups_to_run}
        for g, future in futures.items():
            print(f"\nCollecting results from {g} group...")
            try:
                batch = future.get()
                for key, json_str in batch.items():
                    all_results[key] = json.loads(json_str)
                    out_path = out_dir / f"{key}_gpu.json"
                    out_path.write_text(json_str)
                    print(f"  Saved → {out_path.name}")
                if not batch:
                    print(f"  WARNING: {g} group returned no results")
            except Exception as e:
                print(f"  ERROR collecting {g} group: {e}")
    else:
        for g in groups_to_run:
            print(f"\nRunning {g} group...")
            try:
                batch = runners[g].remote()
                for key, json_str in batch.items():
                    all_results[key] = json.loads(json_str)
                    out_path = out_dir / f"{key}_gpu.json"
                    out_path.write_text(json_str)
                    print(f"  Saved → {out_path.name}")
                if not batch:
                    print(f"  WARNING: {g} group returned no results")
            except Exception as e:
                print(f"  ERROR in {g} group: {e}")

    # ── Print consolidated summary table ──────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("NVE GPU BENCHMARK — FULL RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Model':<22} {'Config':<28} {'Acc':>6} {'Tok/s':>8} {'Mem MB':>8} {'Device'}")
    print(f"  {'─' * 22} {'─' * 28} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 10}")

    for key, data in all_results.items():
        model_label = MODELS.get(key, {}).get("label", key)
        device = data.get("params", {}).get("device", "?")
        for cfg in data.get("configurations", []):
            acc = cfg.get("task_accuracy", 0)
            tps = cfg.get("summary", {}).get("avg_tokens_per_sec", 0)
            mem = cfg.get("summary", {}).get("peak_memory_mb", 0)
            acc_s  = f"{acc:.0%}" if isinstance(acc, float) else str(acc)
            tps_s  = f"{tps:.1f}"  if isinstance(tps, float) else str(tps)
            mem_s  = f"{mem:.0f}"  if isinstance(mem, float) else str(mem)
            print(f"  {model_label:<22} {cfg['config']:<28} {acc_s:>6} {tps_s:>8} {mem_s:>8}  {device}")
        print()

    print(f"\n{len(all_results)} model(s) × 4 configs saved to {out_dir}/")
    print("Regenerate paper figures: python3 evidence/visualize_paper.py")
