#!/usr/bin/env python3
"""
NVE GPU Benchmark
=================
Compares NVE (CUDA), llama.cpp (GPU offload), and HuggingFace (CUDA) on
identical prompts across 1B, 3B, and 8B Llama models.

GPU: NVIDIA T4 (16 GB VRAM) — cheapest Modal GPU tier.

NVE memory tiers with GPU
--------------------------
  Hot layers  → VRAM   (--hot-budget-mb)
  Warm layers → CPU RAM (--warm-budget-mb)
  Cold layers → SSD    (mmap, paged)

This three-tier hierarchy is unique to NVE: llama.cpp and HF can only do
"all on GPU" or "all on CPU" (plus partial layer offload for llama.cpp).

Scenarios
---------
  gpu_full      : --device cuda:0 --auto-budget
                  All layers in VRAM (if they fit). Full GPU throughput.
  gpu_vram_8gb  : --device cuda:0 --hot-budget-mb 6000 --warm-budget-mb 6000
                  8B model: ~16 of 32 layers in VRAM, rest paged from CPU.
                  Simulates a 6 GB VRAM card (e.g. RTX 3060).

llama.cpp GPU comparison
------------------------
  n_gpu_layers=-1  : full GPU offload (all layers in VRAM)
  n_gpu_layers=16  : partial offload — mirrors NVE gpu_vram_8gb for 8B

HF comparison
-------------
  device_map="cuda" : full GPU, fp32 (8B requires ~32 GB VRAM → OOM on T4)
  device_map="auto" : splits across GPU + CPU automatically

Usage:
  modal run evidence/modal_gpu_benchmark.py
  modal run evidence/modal_gpu_benchmark.py --model llama8b
  modal run evidence/modal_gpu_benchmark.py --model llama1b,llama3b
  modal run evidence/modal_gpu_benchmark.py --merge   # merge into existing results
"""

import modal
import json
import os
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root

# ── Task suite (identical to the CPU benchmark) ───────────────────────────────
TASK_SUITE = [
    {"category": "qa",            "prompt": "The capital of France is",
     "expected": "paris"},
    {"category": "qa",            "prompt": "Water is composed of hydrogen and",
     "expected": "oxygen"},
    {"category": "qa",            "prompt": "The largest planet in the solar system is",
     "expected": "jupiter"},
    {"category": "reasoning",     "prompt": "If today is Monday, tomorrow is",
     "expected": "tuesday"},
    {"category": "reasoning",     "prompt": "A square has four equal sides. A shape with four equal sides and four right angles is a",
     "expected": "square"},
    {"category": "coding",        "prompt": "def add(a, b):\n    return a",
     "expected": "+"},
    {"category": "coding",        "prompt": "# Python: list of squares 0-4\nsquares = [x**2 for x in",
     "expected": "range"},
    {"category": "summarization", "prompt": "The main benefit of regular exercise is improved",
     "expected": "health"},
]

MODELS = {
    "llama1b": {
        "hf_id":     "meta-llama/Llama-3.2-1B",
        "gguf_repo": "QuantFactory/Llama-3.2-1B-GGUF",
        "gguf_q4":   "Llama-3.2-1B.Q4_K_M.gguf",
        "gguf_q8":   "Llama-3.2-1B.Q8_0.gguf",
        "local_dir": "/models/llama1b",
        "n_layers":  16,
        "layer_mb":  68,      # bf16 MB per layer
        "total_vram_mb": 16 * 68,   # ~1.1 GB
        "label":     "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":     "meta-llama/Llama-3.2-3B",
        "gguf_repo": "QuantFactory/Llama-3.2-3B-GGUF",
        "gguf_q4":   "Llama-3.2-3B.Q4_K_M.gguf",
        "gguf_q8":   "Llama-3.2-3B.Q8_0.gguf",
        "local_dir": "/models/llama3b",
        "n_layers":  28,
        "layer_mb":  192,     # ~5.4 GB total
        "total_vram_mb": 28 * 192,
        "label":     "Llama-3.2-3B",
    },
    "llama8b": {
        "hf_id":     "meta-llama/Llama-3.1-8B",
        "gguf_repo": "QuantFactory/Meta-Llama-3.1-8B-GGUF",
        "gguf_q4":   "Meta-Llama-3.1-8B.Q4_K_M.gguf",
        "gguf_q8":   "Meta-Llama-3.1-8B.Q8_0.gguf",
        "local_dir": "/models/llama8b",
        "n_layers":  32,
        "layer_mb":  500,     # ~16 GB total (bf16)
        "total_vram_mb": 32 * 500,
        "label":     "Llama-3.1-8B",
    },
}

# ── GPU Memory scenarios ──────────────────────────────────────────────────────
# NVE: hot_mb → VRAM, warm_mb → CPU RAM
# llama.cpp: n_gpu_layers (how many transformer blocks go to GPU)
GPU_SCENARIOS = {
    # Full T4 (16 GB VRAM): all layers hot in VRAM.
    # NVE now uploads weights as BF16 (matching safetensors format), so the
    # hot budget maps 1:1 to actual VRAM. 14 GB covers all layers of 1B/3B/8B
    # with headroom for KV cache and CUDA runtime (~1.5 GB).
    "gpu_full": {
        "nve_hot_mb":      14000,  # 14 GB hot → VRAM
        "nve_warm_mb":     14000,  # 14 GB warm → CPU RAM fallback
        "llama_gpu_layers": -1,    # offload all layers
        "hf_device":        "cuda",
        "label":            "Full GPU (T4 16 GB VRAM)",
    },
    # Simulate 6 GB VRAM card (e.g. RTX 3060 / laptop GPU).
    # 8B: 12 of 32 layers in VRAM, rest paged from CPU RAM → unique NVE territory.
    "gpu_vram_6gb": {
        "nve_hot_mb":      5500,   # 5.5 GB hot → VRAM
        "nve_warm_mb":     10000,  # 10 GB warm → CPU RAM
        "llama_gpu_layers": 12,    # partial offload (12/32 layers for 8B)
        "hf_device":        "auto", # HF auto-splits across GPU + CPU
        "label":            "6 GB VRAM budget",
    },
}

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("nve-gpu-benchmark")

model_vol    = modal.Volume.from_name("nve-model-weights",     create_if_missing=True)
nve_cuda_vol = modal.Volume.from_name("nve-cuda-binary",       create_if_missing=True)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})

# ── Build image: Rust + CUDA toolkit ─────────────────────────────────────────
# Uses the official NVIDIA CUDA devel image so that nvcc and cuBLAS are present
# when cargo compiles candle-core with --features cuda.
#
# Image is cached after first build (~8-12 min to compile NVE with CUDA).
nve_cuda_build_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "curl", "pkg-config", "libssl-dev", "ca-certificates",
        "build-essential", "cmake", "git",
    )
    .run_commands(
        # Install stable Rust toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        # Install pip deps for download helpers
        "pip install huggingface_hub[hf_xfer] tqdm psutil",
    )
    .add_local_dir(str(HERE), "/nve-src", copy=True,
                   ignore=["target/", ".git/", "evidence/figures/", "evidence/figures_paper/"])
    .run_commands(
        # Build with CUDA. CUDA_PATH is set by the nvidia/cuda base image.
        # target-cpu=x86-64-v3 matches the Modal Intel host CPUs.
        #
        # CUDA_COMPUTE_CAP=75  — T4 is sm_75; set explicitly so bindgen_cuda
        #   does not call nvidia-smi (which fails on CPU image-build workers).
        # set -o pipefail     — ensures cargo exit code propagates through the pipe.
        "bash -c 'set -o pipefail && "
        "cd /nve-src && "
        "CUDA_COMPUTE_CAP=75 "
        "CUDA_PATH=/usr/local/cuda "
        "RUSTFLAGS=\"-C target-cpu=x86-64-v3\" "
        # T4 is sm_75 (Turing) — Flash Attention v2 requires sm_80+ (Ampere).
        # For A100/H100: change --features cuda → --features cuda,flash-attn
        "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -200'",
        "cp /nve-src/target/release/nve /usr/local/bin/nve-cuda",
        "chmod +x /usr/local/bin/nve-cuda",
        "nve-cuda --version",
    )
)

# GPU image for llama.cpp + HF.
# Using devel (not runtime) so that CUDA shared libs (cublas, curand, etc.)
# needed by llama-cpp-python pre-built wheels are all present.
gpu_ml_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgcc-s1", "ca-certificates")
    .pip_install("huggingface_hub[hf_xfer]", "tqdm", "psutil")
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu121",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("transformers", "sentencepiece", "protobuf", "accelerate")
)


# ════════════════════════════════════════════════════════════════════════════
# NVE GPU evaluation
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=nve_cuda_build_image,
    gpu="T4",
    cpu=8.0,
    memory=32768,   # 32 GB RAM — needed for warm layer buffer + 8B model download
    timeout=4 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_nve_gpu(model_name: str, scenario_name: str, profile_scores: list | None = None) -> dict:
    """Run NVE abc-test with CUDA device and return structured results."""
    import subprocess, time, json, os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = GPU_SCENARIOS[scenario_name]
    model_dir = model_cfg["local_dir"]
    out_file  = f"/tmp/nve_gpu_{model_name}_{scenario_name}.json"

    # Download safetensors weights (shared volume with CPU benchmark)
    marker = Path(model_dir) / ".downloaded"
    if not marker.exists():
        print(f"[download] {model_cfg['hf_id']} ...")
        t0 = time.time()
        snapshot_download(
            repo_id=model_cfg["hf_id"],
            local_dir=model_dir,
            token=hf_token,
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()
        print(f"  done in {time.time()-t0:.0f}s")

    # Write profile if provided
    profile_file = None
    if profile_scores:
        profile_file = f"/tmp/nve_{model_name}_gpu_profile.json"
        Path(profile_file).write_text(json.dumps(profile_scores))
        print(f"  [profile] Injecting {len(profile_scores)} pre-computed scores")

    cmd = [
        "nve-cuda", "abc-test",
        "-m", model_dir,
        "-n", "40",
        "-o", out_file,
        "--device", "cuda:0",
        "--hot-budget-mb",  str(scenario["nve_hot_mb"]),
        "--warm-budget-mb", str(scenario["nve_warm_mb"]),
    ]

    if profile_file:
        cmd += ["--profile-from", profile_file]

    if scenario_name == "gpu_full":
        cmd += ["--save-profile", f"/tmp/nve_gpu_{model_name}_profile.json"]

    print(f"\n[NVE GPU] {model_cfg['label']} — {scenario['label']}")
    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[-1000:]}")
        return {
            "system": "nve_gpu", "model": model_name, "scenario": scenario_name,
            "error": f"exit {proc.returncode}", "stderr": proc.stderr[-500:],
        }

    with open(out_file) as f:
        data = json.load(f)

    configs = {}
    for cfg in data.get("configurations", []):
        configs[cfg["config"]] = {
            "task_accuracy":      cfg.get("task_accuracy", 0.0),
            "avg_tokens_per_sec": cfg.get("summary", {}).get("avg_tokens_per_sec", 0.0),
            "paging_info":        cfg.get("paging_info"),
        }
        print(f"  {cfg['config']:30s} acc={cfg.get('task_accuracy',0):.0%}  "
              f"{cfg.get('summary',{}).get('avg_tokens_per_sec',0):.1f} tok/s")

    b_cfg  = configs.get("B_profiled_hot", {})
    paging = b_cfg.get("paging_info") or {}
    saved_scores = paging.get("layer_importance_scores") or []

    return {
        "system":            "nve_gpu",
        "model":             model_name,
        "scenario":          scenario_name,
        "elapsed_s":         elapsed,
        "configs":           configs,
        "raw":               data,
        "importance_scores": saved_scores,
    }


# ════════════════════════════════════════════════════════════════════════════
# llama.cpp GPU evaluation
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_ml_image,
    gpu="T4",
    cpu=4.0,
    memory=32768,
    timeout=2 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_llamacpp_gpu(model_name: str, quant: str, scenario_name: str) -> dict:
    """Run llama.cpp with GPU layer offload. quant ∈ {'q4', 'q8'}."""
    import time, os, gc
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = GPU_SCENARIOS[scenario_name]

    gguf_fname = model_cfg["gguf_q4"] if quant == "q4" else model_cfg["gguf_q8"]
    gguf_dir   = Path(model_cfg["local_dir"] + "_gguf")
    gguf_path  = gguf_dir / gguf_fname

    gguf_dir.mkdir(parents=True, exist_ok=True)
    if not gguf_path.exists():
        print(f"[download] {model_cfg['gguf_repo']}/{gguf_fname} ...")
        t0 = time.time()
        try:
            hf_hub_download(
                repo_id=model_cfg["gguf_repo"],
                filename=gguf_fname,
                local_dir=str(gguf_dir),
                token=hf_token,
            )
            model_vol.commit()
            print(f"  downloaded in {time.time()-t0:.0f}s")
        except Exception as e:
            return {"system": f"llamacpp_gpu_{quant}", "model": model_name,
                    "scenario": scenario_name, "error": f"download failed: {e}"}

    n_gpu_layers = scenario["llama_gpu_layers"]
    print(f"\n[llama.cpp GPU {quant.upper()}] {model_cfg['label']} — {scenario['label']}")
    print(f"  n_gpu_layers={n_gpu_layers}")

    try:
        from llama_cpp import Llama

        t_load = time.time()
        llm = Llama(
            model_path=str(gguf_path),
            n_threads=4,
            n_ctx=512,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        load_time = time.time() - t_load
        print(f"  loaded in {load_time:.1f}s")

    except Exception as e:
        return {"system": f"llamacpp_gpu_{quant}", "model": model_name,
                "scenario": scenario_name, "error": str(e),
                "task_accuracy": None, "avg_tokens_per_sec": None}

    task_results = []
    tps_list     = []

    for item in TASK_SUITE:
        prompt   = item["prompt"]
        expected = item["expected"].lower()
        try:
            t0  = time.time()
            out = llm(prompt, max_tokens=40, temperature=0.0, echo=False)
            dt  = time.time() - t0

            generated = out["choices"][0]["text"]
            n_tokens  = out["usage"]["completion_tokens"]
            tps       = n_tokens / dt if dt > 0 else 0.0
            passed    = expected in generated.lower()

            task_results.append({
                "category": item["category"], "prompt": prompt,
                "expected": expected, "generated": generated[:100], "passed": passed,
            })
            tps_list.append(tps)
            print(f"  [{'PASS' if passed else 'FAIL'}] {item['category']:15s} | {generated[:40]!r}")

        except Exception as e:
            task_results.append({"category": item["category"], "prompt": prompt,
                                  "error": str(e), "passed": False})

    acc     = sum(r["passed"] for r in task_results) / len(task_results) if task_results else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    print(f"\n  Task accuracy: {acc:.0%}  |  Avg tok/s: {avg_tps:.1f}")

    del llm
    gc.collect()

    return {
        "system":             f"llamacpp_gpu_{quant}",
        "model":              model_name,
        "scenario":           scenario_name,
        "n_gpu_layers":       n_gpu_layers,
        "load_time_s":        load_time,
        "task_accuracy":      acc,
        "avg_tokens_per_sec": avg_tps,
        "task_results":       task_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# HuggingFace GPU evaluation
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=gpu_ml_image,
    gpu="T4",
    cpu=4.0,
    memory=32768,
    timeout=2 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_hf_gpu(model_name: str, scenario_name: str) -> dict:
    """Run HuggingFace Transformers on GPU (bf16)."""
    import time, os, gc
    from pathlib import Path
    from huggingface_hub import snapshot_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = GPU_SCENARIOS[scenario_name]
    model_dir = model_cfg["local_dir"]
    hf_device = scenario["hf_device"]

    marker = Path(model_dir) / ".downloaded"
    if not marker.exists():
        print(f"[download] {model_cfg['hf_id']} ...")
        t0 = time.time()
        snapshot_download(
            repo_id=model_cfg["hf_id"],
            local_dir=model_dir,
            token=hf_token,
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()
        print(f"  done in {time.time()-t0:.0f}s")

    print(f"\n[HF GPU bf16] {model_cfg['label']} — {scenario['label']} (device_map={hf_device!r})")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Estimate VRAM: bf16 = 2 bytes/param ≈ layer_mb per layer
    vram_needed_gb = model_cfg["total_vram_mb"] / 1024
    t4_vram_gb     = 15.0
    if hf_device == "cuda" and vram_needed_gb > t4_vram_gb:
        print(f"  SKIP: bf16 model ~{vram_needed_gb:.1f} GB > T4 VRAM {t4_vram_gb} GB (use device_map='auto')")
        return {
            "system": "hf_gpu_bf16", "model": model_name, "scenario": scenario_name,
            "result": "oom_predicted",
            "task_accuracy": None, "avg_tokens_per_sec": None,
        }

    try:
        t_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=hf_device,
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - t_load
        print(f"  loaded in {load_time:.1f}s  device_map={hf_device}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"  VRAM allocated: {allocated:.2f} GB")

    except (MemoryError, RuntimeError) as e:
        return {"system": "hf_gpu_bf16", "model": model_name, "scenario": scenario_name,
                "error": f"OOM: {e}", "task_accuracy": None, "avg_tokens_per_sec": None}
    except Exception as e:
        return {"system": "hf_gpu_bf16", "model": model_name, "scenario": scenario_name,
                "error": str(e), "task_accuracy": None, "avg_tokens_per_sec": None}

    task_results = []
    tps_list     = []

    for item in TASK_SUITE:
        prompt   = item["prompt"]
        expected = item["expected"].lower()
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            n_in   = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            dt = time.time() - t0

            n_out     = out_ids.shape[1] - n_in
            generated = tokenizer.decode(out_ids[0, n_in:], skip_special_tokens=True)
            tps       = n_out / dt if dt > 0 else 0.0
            passed    = expected in generated.lower()

            task_results.append({
                "category": item["category"], "prompt": prompt,
                "expected": expected, "generated": generated[:100], "passed": passed,
            })
            tps_list.append(tps)
            print(f"  [{'PASS' if passed else 'FAIL'}] {item['category']:15s} | {generated[:40]!r}")

        except Exception as e:
            task_results.append({"category": item["category"], "prompt": prompt,
                                  "error": str(e), "passed": False})

    acc     = sum(r["passed"] for r in task_results) / len(task_results) if task_results else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    print(f"\n  Task accuracy: {acc:.0%}  |  Avg tok/s: {avg_tps:.1f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "system":             "hf_gpu_bf16",
        "model":              model_name,
        "scenario":           scenario_name,
        "device_map":         hf_device,
        "load_time_s":        load_time,
        "task_accuracy":      acc,
        "avg_tokens_per_sec": avg_tps,
        "task_results":       task_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    model:   str  = "all",  # all | llama1b | llama3b | llama8b | llama1b,llama3b
    scenario: str = "all",  # all | gpu_full | gpu_vram_6gb
    systems: str  = "all",  # all | nve | llamacpp | hf | nve,llamacpp
    merge:   bool = False,  # merge into existing gpu_benchmark.json
):
    import itertools

    all_models    = ["llama1b", "llama3b", "llama8b"]
    all_scenarios = list(GPU_SCENARIOS.keys())

    models    = all_models    if model    == "all" else model.split(",")
    scenarios = all_scenarios if scenario == "all" else scenario.split(",")
    run_sys   = set(systems.split(",")) if systems != "all" else {"nve", "llamacpp", "hf"}

    print("NVE GPU Benchmark")
    print(f"Models:    {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Systems:   {sorted(run_sys)}")
    print()

    # ── Phase 1: full-GPU NVE (captures profiles) + all llama.cpp + HF ───────
    full_scenarios    = [s for s in scenarios if s == "gpu_full"]
    limited_scenarios = [s for s in scenarios if s != "gpu_full"]

    nve_full_jobs   = [(m, s) for m, s in itertools.product(models, full_scenarios)]  \
                      if "nve" in run_sys else []
    llama_jobs      = [(m, q, s) for m, s in itertools.product(models, scenarios)
                       for q in ("q4", "q8")]                                          \
                      if "llamacpp" in run_sys else []
    hf_jobs         = list(itertools.product(models, scenarios))                       \
                      if "hf" in run_sys else []

    print(f"Phase 1: {len(nve_full_jobs)} NVE full-GPU + "
          f"{len(llama_jobs)} llama.cpp GPU + {len(hf_jobs)} HF GPU  (parallel) ...")

    nve_full_handles  = [eval_nve_gpu.spawn(m, s) for m, s in nve_full_jobs]
    llama_handles     = [eval_llamacpp_gpu.spawn(*args) for args in llama_jobs]
    hf_handles        = [eval_hf_gpu.spawn(*args) for args in hf_jobs]

    nve_full_results  = [h.get() for h in nve_full_handles]
    llama_results     = [h.get() for h in llama_handles]
    hf_results        = [h.get() for h in hf_handles]

    # Extract profiles from full-GPU NVE runs
    profiles: dict[str, list] = {}
    for r in nve_full_results:
        scores = r.get("importance_scores", [])
        if scores:
            profiles[r["model"]] = scores
            print(f"  [profile] {r['model']}: {len(scores)} layer scores captured")

    # Fallback: load from CPU benchmark profiles if GPU run didn't produce them
    if not profiles or any(m not in profiles for m in models):
        cpu_results = HERE / "evidence" / "experiments" / "rigorous_comparison.json"
        if cpu_results.exists():
            try:
                existing = json.loads(cpu_results.read_text())
                for r in existing.get("nve", []):
                    m = r.get("model")
                    if m not in profiles and r.get("scenario") == "unconstrained":
                        scores = r.get("importance_scores") or []
                        if not scores:
                            b = r.get("configs", {}).get("B_profiled_hot", {})
                            scores = (b.get("paging_info") or {}).get("layer_importance_scores", [])
                        if scores:
                            profiles[m] = scores
                            print(f"  [profile] {m}: loaded {len(scores)} scores from CPU results")
            except Exception as e:
                print(f"  [profile] Warning: {e}")

    # ── Phase 2: VRAM-limited NVE runs with injected profiles ─────────────────
    nve_limited_jobs = [(m, s) for m, s in itertools.product(models, limited_scenarios)] \
                       if "nve" in run_sys else []

    if nve_limited_jobs:
        print(f"\nPhase 2: {len(nve_limited_jobs)} NVE VRAM-limited jobs (with profiles) ...")
        nve_limited_handles = [
            eval_nve_gpu.spawn(m, s, profile_scores=profiles.get(m))
            for m, s in nve_limited_jobs
        ]
        nve_limited_results = [h.get() for h in nve_limited_handles]
    else:
        nve_limited_results = []

    nve_results = nve_full_results + nve_limited_results

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir  = HERE / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gpu_benchmark.json"

    if merge and out_path.exists():
        existing = json.loads(out_path.read_text())
        def _nk(r): return (r["model"], r["scenario"])
        def _ok(r): return (r["model"], r["scenario"], r["system"])

        ex_nve   = {_nk(r): r for r in existing.get("nve_gpu",   [])}
        ex_llama = {_ok(r): r for r in existing.get("llamacpp_gpu", [])}
        ex_hf    = {_ok(r): r for r in existing.get("hf_gpu",    [])}

        for r in nve_results:   ex_nve[_nk(r)]   = r
        for r in llama_results: ex_llama[_ok(r)]  = r
        for r in hf_results:    ex_hf[_ok(r)]     = r

        nve_results   = list(ex_nve.values())
        llama_results = list(ex_llama.values())
        hf_results    = list(ex_hf.values())

    out_path.write_text(json.dumps({
        "nve_gpu":      nve_results,
        "llamacpp_gpu": llama_results,
        "hf_gpu":       hf_results,
    }, indent=2))
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Model':<12} {'Scenario':<18} {'System':<28} {'Config':<25} {'Acc':>6} {'Tok/s':>8}")
    print("─" * 105)

    for r in nve_results:
        m, s = r["model"], r["scenario"]
        if "error" in r:
            print(f"  {m:<10} {s:<18} {'nve_gpu':<28} ERROR: {r['error']}")
            continue
        for cfg_name, cfg_data in r.get("configs", {}).items():
            acc  = cfg_data["task_accuracy"]
            tps  = cfg_data["avg_tokens_per_sec"]
            print(f"  {m:<10} {s:<18} {'nve_gpu':<28} {cfg_name:<25} "
                  f"{acc:.0%}  {tps:.1f}")

    for r in llama_results + hf_results:
        m, s   = r["model"], r["scenario"]
        system = r["system"]
        acc    = r.get("task_accuracy")
        tps    = r.get("avg_tokens_per_sec")
        if r.get("result") == "oom_predicted":
            print(f"  {m:<10} {s:<18} {system:<28} {'—':<25} {'OOM':>6} {'—':>8}")
        elif "error" in r:
            print(f"  {m:<10} {s:<18} {system:<28} {'—':<25} ERROR  {r.get('error','')[:20]}")
        else:
            acc_s = f"{acc:.0%}" if acc is not None else "?"
            tps_s = f"{tps:.1f}" if tps is not None else "?"
            print(f"  {m:<10} {s:<18} {system:<28} {'—':<25} {acc_s:>6} {tps_s:>8}")

    print(f"\nDone. Merge GPU + CPU results: python3 evidence/visualize_paper.py")
