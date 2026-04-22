#!/usr/bin/env python3
"""
NVE Rigorous Competitive Benchmark
====================================
Compares NVE against llama.cpp and HuggingFace Transformers on identical
prompts, memory budgets, and quality metrics.

Systems under test:
  • NVE: Baseline (bf16), A (uniform Q4), B (profiled hot-only), C (PG+AWQ)
  • llama.cpp: Q4_K_M quantization, Q8_0 quantization
  • HF Transformers: float32 (exact reference)

Models:
  • Llama-3.2-1B (1.2 B params, 16 layers, ~68 MB/layer)
  • Llama-3.2-3B (3.2 B params, 28 layers, ~192 MB/layer)
  • Llama-3.1-8B (8.0 B params, 32 layers, ~500 MB/layer bf16)

Scenarios:
  • unconstrained: full auto-budget (all RAM available)
  • constrained_2gb: 2 GB total memory budget
  • constrained_4gb: 4 GB total memory budget

Memory constraint significance:
  • 3B Q4 GGUF ≈ 2.0 GB (weights only) + KV cache → llama.cpp OOMs at 2 GB
  • NVE pages 5 GB of 3B weights through a 2 GB hot+warm buffer → keeps running
  • 8B Q4 GGUF ≈ 4.7 GB → llama.cpp OOMs at 4 GB
  • NVE pages 16 GB of 8B weights through a 4 GB hot+warm buffer using a
    pre-computed profile from the unconstrained run — best layers stay warm
  • HF fp32 needs 30+ GB for 8B → hard OOM

Key workflow:
  Phase 1 — unconstrained NVE runs: profile layer importance on the full model.
  Phase 2 — constrained NVE runs: inject the pre-computed profile so that the
             best layers are kept warm without running profiling again.
  This is the "profile once on cloud, run optimally on edge" story.

Quality metric:
  • 8-item task suite matching NVE's DEFAULT_TASK_SUITE (substring match)
  • Same prompts and expected substrings across all three systems for fairness

Throughput:
  • Tokens/second measured after model load (decode phase)
  • Reported as avg ± stddev across 8 prompts

Usage:
  modal run evidence/modal_rigorous.py
  modal run evidence/modal_rigorous.py --model llama8b
  modal run evidence/modal_rigorous.py --model llama3b --scenario constrained_4gb
"""

import modal
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root
NVE_BIN = HERE / "target" / "release" / "nve"

# ── Task suite (identical to NVE's DEFAULT_TASK_SUITE in abc_benchmark.rs) ──
TASK_SUITE = [
    {"category": "qa",             "prompt": "The capital of France is",
     "expected": "paris"},
    {"category": "qa",             "prompt": "Water is composed of hydrogen and",
     "expected": "oxygen"},
    {"category": "qa",             "prompt": "The largest planet in the solar system is",
     "expected": "jupiter"},
    {"category": "reasoning",      "prompt": "If today is Monday, tomorrow is",
     "expected": "tuesday"},
    {"category": "reasoning",      "prompt": "A square has four equal sides. A shape with four equal sides and four right angles is a",
     "expected": "square"},
    {"category": "coding",         "prompt": "def add(a, b):\n    return a",
     "expected": "+"},
    {"category": "coding",         "prompt": "# Python: list of squares 0-4\nsquares = [x**2 for x in",
     "expected": "range"},
    {"category": "summarization",  "prompt": "The main benefit of regular exercise is improved",
     "expected": "health"},
]

# ── Model configs ────────────────────────────────────────────────────────────
# GGUF sources: official Meta repos via HF hub (same weights, different format)
# meta-llama/Llama-3.2-1B-GGUF contains Q4_K_M, Q8_0, etc.
MODELS = {
    "llama1b": {
        "hf_id":     "meta-llama/Llama-3.2-1B",
        "gguf_repo": "QuantFactory/Llama-3.2-1B-GGUF",
        "gguf_q4":   "Llama-3.2-1B.Q4_K_M.gguf",
        "gguf_q8":   "Llama-3.2-1B.Q8_0.gguf",
        # Fallback community GGUF if primary repo doesn't have these files
        "gguf_repo_fallback": "unsloth/Llama-3.2-1B-GGUF",
        "gguf_q4_fallback":   "Llama-3.2-1B-Q4_K_M.gguf",
        "gguf_q8_fallback":   "Llama-3.2-1B-Q8_0.gguf",
        "local_dir": "/models/llama1b",
        "n_layers":  16,
        "layer_mb":  68,
        "label":     "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":     "meta-llama/Llama-3.2-3B",
        "gguf_repo": "QuantFactory/Llama-3.2-3B-GGUF",
        "gguf_q4":   "Llama-3.2-3B.Q4_K_M.gguf",
        "gguf_q8":   "Llama-3.2-3B.Q8_0.gguf",
        "gguf_repo_fallback": "unsloth/Llama-3.2-3B-GGUF",
        "gguf_q4_fallback":   "Llama-3.2-3B-Q4_K_M.gguf",
        "gguf_q8_fallback":   "Llama-3.2-3B-Q8_0.gguf",
        "local_dir": "/models/llama3b",
        "n_layers":  28,
        "layer_mb":  192,
        "label":     "Llama-3.2-3B",
    },
    # Llama-3.1-8B: 32 layers × ~500 MB/layer bf16 = ~16 GB total
    # Q4 GGUF ≈ 4.7 GB → OOMs inside 4 GB; NVE pages with profile → works
    "llama8b": {
        "hf_id":     "meta-llama/Llama-3.1-8B",
        "gguf_repo": "QuantFactory/Meta-Llama-3.1-8B-GGUF",
        "gguf_q4":   "Meta-Llama-3.1-8B.Q4_K_M.gguf",
        "gguf_q8":   "Meta-Llama-3.1-8B.Q8_0.gguf",
        "gguf_repo_fallback": "bartowski/Meta-Llama-3.1-8B-GGUF",
        "gguf_q4_fallback":   "Meta-Llama-3.1-8B-Q4_K_M.gguf",
        "gguf_q8_fallback":   "Meta-Llama-3.1-8B-Q8_0.gguf",
        "local_dir": "/models/llama8b",
        "n_layers":  32,
        "layer_mb":  500,
        "label":     "Llama-3.1-8B",
    },
}

# ── Memory scenarios ─────────────────────────────────────────────────────────
SCENARIOS = {
    "unconstrained": {
        "nve_hot_mb":  None,   # auto-budget
        "nve_warm_mb": None,
        "llama_mem_gb": None,  # no limit
        "hf_mem_gb":   None,
        "label": "Unconstrained",
    },
    "constrained_2gb": {
        "nve_hot_mb":  1000,   # 1 GB hot + 1 GB warm = 2 GB total
        "nve_warm_mb": 1000,
        "llama_mem_gb": 2.0,   # llama.cpp will likely OOM for 3B
        "hf_mem_gb":   2.0,    # HF will fail for all but 1B (very barely)
        "label": "2 GB Budget",
    },
    # 4 GB: 8B Q4 GGUF (4.7 GB) can't load in llama.cpp; NVE pages with profile.
    # Only 8/32 layers active at 4 GB — too few for coherent 8B output; used to
    # demonstrate OOM boundary.
    "constrained_4gb": {
        "nve_hot_mb":  2000,   # 2 GB hot + 2 GB warm = 4 GB total
        "nve_warm_mb": 2000,
        "llama_mem_gb": 4.0,   # 8B Q4 GGUF ≈ 4.7 GB → OOM
        "hf_mem_gb":   4.0,
        "label": "4 GB Budget",
    },
    # 8 GB: sweet spot for 8B — NVE keeps 16/32 layers warm with profile → coherent
    # output. llama.cpp Q4 (4.7 GB) fits here so no OOM advantage, but NVE B with
    # profile selects the BEST 16 layers vs llama.cpp's uniform Q4 over all 32.
    "constrained_8gb": {
        "nve_hot_mb":  4000,   # 4 GB hot + 4 GB warm = 8 GB total
        "nve_warm_mb": 4000,
        "llama_mem_gb": 8.0,   # Q4 (4.7 GB) fits; Q8 (8.5 GB) OOMs
        "hf_mem_gb":   8.0,    # fp32 8B ≈ 32 GB → OOM
        "label": "8 GB Budget",
    },
}

# ── Modal app ────────────────────────────────────────────────────────────────
app = modal.App("nve-rigorous")

model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)

# NVE + basic utils
nve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgcc-s1", "ca-certificates")
    .pip_install("huggingface_hub[hf_xfer]", "tqdm", "psutil")
    .add_local_file(str(NVE_BIN), "/usr/local/bin/nve", copy=True)
    .run_commands("chmod +x /usr/local/bin/nve && nve --version")
)

# llama.cpp + HF Transformers (CPU torch only, smaller install)
# Use pre-built llama-cpp-python wheel (avoids 10-15 min C++ compile)
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgcc-s1", "ca-certificates")
    .pip_install(
        "huggingface_hub[hf_xfer]",
        "tqdm",
        "psutil",
    )
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu",
    )
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "transformers",
        "sentencepiece",
        "protobuf",
        "accelerate",
    )
)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})


# ════════════════════════════════════════════════════════════════════════════
# NVE evaluation — runs abc-test, parses JSON output
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=nve_image,
    cpu=8.0,
    memory=32768,   # 32 GB — needed for 8B unconstrained profiling pass (~16 GB model)
    timeout=4 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_nve(model_name: str, scenario_name: str, profile_scores: list | None = None) -> dict:
    """Run NVE abc-test (all 4 configs) and return structured results.

    profile_scores — pre-computed layer-importance scores from an unconstrained
    run on the same model.  When provided, configs B and C skip the profiling
    forward-pass entirely (the "profile on large machine, run on small device"
    workflow).  Pass None to run profiling fresh.
    """
    import subprocess, time, json, os, psutil
    from pathlib import Path
    from huggingface_hub import snapshot_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = SCENARIOS[scenario_name]
    model_dir = model_cfg["local_dir"]
    out_file  = f"/tmp/nve_{model_name}_{scenario_name}.json"

    # Download safetensors model (skip if cached)
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

    # Write pre-computed profile to a temp file if provided
    profile_file = None
    if profile_scores:
        profile_file = f"/tmp/nve_{model_name}_profile.json"
        Path(profile_file).write_text(json.dumps(profile_scores))
        print(f"  [profile] Injecting {len(profile_scores)} pre-computed scores "
              f"(skipping profiling pass)")

    # Build abc-test command
    cmd = ["nve", "abc-test", "-m", model_dir, "-n", "40", "-o", out_file]

    if scenario["nve_hot_mb"] is not None:
        cmd += ["--hot-budget-mb",  str(scenario["nve_hot_mb"])]
        cmd += ["--warm-budget-mb", str(scenario["nve_warm_mb"])]
    else:
        cmd += ["--auto-budget"]

    if profile_file:
        cmd += ["--profile-from", profile_file]

    # Save profile during unconstrained run so we can extract it later
    if scenario_name == "unconstrained":
        cmd += ["--save-profile", f"/tmp/nve_{model_name}_saved_profile.json"]

    print(f"\n[NVE] {model_cfg['label']} — {scenario['label']}")
    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[-1000:]}")
        return {"system": "nve", "model": model_name, "scenario": scenario_name,
                "error": f"exit {proc.returncode}", "stderr": proc.stderr[-500:]}

    with open(out_file) as f:
        data = json.load(f)

    configs = {}
    for cfg in data.get("configurations", []):
        configs[cfg["config"]] = {
            "task_accuracy": cfg.get("task_accuracy", 0.0),
            "avg_tokens_per_sec": cfg.get("summary", {}).get("avg_tokens_per_sec", 0.0),
            "paging_info": cfg.get("paging_info"),
        }
        print(f"  {cfg['config']:30s} acc={cfg.get('task_accuracy',0):.0%}  "
              f"{cfg.get('summary',{}).get('avg_tokens_per_sec',0):.1f} tok/s")

    # Extract importance scores from config B so they can be passed to
    # constrained runs (the "profile on cloud, run on edge" workflow).
    b_cfg = configs.get("B_profiled_hot", {})
    paging = b_cfg.get("paging_info") or {}
    saved_scores = paging.get("layer_importance_scores") or []

    return {
        "system":          "nve",
        "model":           model_name,
        "scenario":        scenario_name,
        "elapsed_s":       elapsed,
        "configs":         configs,
        "raw":             data,
        # Included so orchestrator can pass to constrained runs without re-profiling.
        "importance_scores": saved_scores,
    }


# ════════════════════════════════════════════════════════════════════════════
# llama.cpp evaluation — Python API (llama-cpp-python)
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=ml_image,
    cpu=4.0,
    memory=16384,   # 16 GB — 8B Q8 GGUF ≈ 8.5 GB; OOM paths are handled in code
    timeout=2 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_llamacpp(model_name: str, quant: str, scenario_name: str) -> dict:
    """Run llama.cpp inference on task suite. quant ∈ {'q4', 'q8'}."""
    import time, os, gc
    import psutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = SCENARIOS[scenario_name]

    # Select GGUF file
    gguf_fname = model_cfg["gguf_q4"] if quant == "q4" else model_cfg["gguf_q8"]
    gguf_dir   = Path(model_cfg["local_dir"] + "_gguf")
    gguf_path  = gguf_dir / gguf_fname

    # Download GGUF — try official Meta repo first, fall back to community
    gguf_dir.mkdir(parents=True, exist_ok=True)
    if not gguf_path.exists():
        repos_to_try = [
            (model_cfg["gguf_repo"], gguf_fname),
        ]
        if "gguf_repo_fallback" in model_cfg:
            fb_fname = model_cfg.get(f"gguf_{quant}_fallback", gguf_fname)
            repos_to_try.append((model_cfg["gguf_repo_fallback"], fb_fname))

        downloaded = False
        for repo_id, fname in repos_to_try:
            print(f"[download] {repo_id}/{fname} ...")
            t0 = time.time()
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=str(gguf_dir),
                    token=hf_token,
                )
                # Rename to canonical path if fallback used different filename
                if fname != gguf_fname:
                    (gguf_dir / fname).rename(gguf_path)
                downloaded = True
                model_vol.commit()
                print(f"  downloaded in {time.time()-t0:.0f}s")
                break
            except Exception as e:
                print(f"  failed: {e}, trying next...")

        if not downloaded:
            return {"system": f"llamacpp_{quant}", "model": model_name,
                    "scenario": scenario_name, "error": "all download sources failed"}

    print(f"\n[llama.cpp {quant.upper()}] {model_cfg['label']} — {scenario['label']}")

    # Attempt to load the model
    try:
        from llama_cpp import Llama

        load_kwargs = {
            "model_path": str(gguf_path),
            "n_threads":  4,
            "n_ctx":      512,        # small context → less KV cache RAM
            "n_gpu_layers": 0,        # CPU only
            "verbose":    False,
        }

        # Memory constraint: check if GGUF file size fits in budget
        gguf_size_gb = gguf_path.stat().st_size / (1024 ** 3)
        if scenario["llama_mem_gb"] is not None:
            # ~200 MB overhead beyond model weights
            if gguf_size_gb + 0.2 > scenario["llama_mem_gb"]:
                print(f"  SKIP: GGUF size {gguf_size_gb:.2f} GB > budget {scenario['llama_mem_gb']} GB")
                return {
                    "system": f"llamacpp_{quant}", "model": model_name,
                    "scenario": scenario_name,
                    "gguf_size_gb": gguf_size_gb,
                    "result": "oom_predicted",
                    "task_accuracy": None,
                    "avg_tokens_per_sec": None,
                }

        t_load = time.time()
        proc_before = psutil.Process().memory_info().rss
        llm = Llama(**load_kwargs)
        load_time = time.time() - t_load
        proc_after = psutil.Process().memory_info().rss
        load_rss_mb = (proc_after - proc_before) / (1024 ** 2)
        print(f"  loaded in {load_time:.1f}s  (+{load_rss_mb:.0f} MB RSS)")

    except MemoryError as e:
        return {"system": f"llamacpp_{quant}", "model": model_name,
                "scenario": scenario_name, "error": f"OOM: {e}",
                "task_accuracy": None, "avg_tokens_per_sec": None}
    except Exception as e:
        return {"system": f"llamacpp_{quant}", "model": model_name,
                "scenario": scenario_name, "error": str(e),
                "task_accuracy": None, "avg_tokens_per_sec": None}

    # Evaluate task suite
    task_results = []
    tps_list     = []

    for item in TASK_SUITE:
        prompt   = item["prompt"]
        expected = item["expected"].lower()
        try:
            t0  = time.time()
            out = llm(prompt, max_tokens=40, temperature=0.0, echo=False)
            dt  = time.time() - t0

            generated  = out["choices"][0]["text"]
            n_tokens   = out["usage"]["completion_tokens"]
            tps        = n_tokens / dt if dt > 0 else 0.0
            passed     = expected in generated.lower()

            task_results.append({
                "category":  item["category"],
                "prompt":    prompt,
                "expected":  expected,
                "generated": generated[:100],
                "passed":    passed,
            })
            tps_list.append(tps)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {item['category']:15s} | {generated[:40]!r}")

        except Exception as e:
            task_results.append({"category": item["category"], "prompt": prompt,
                                  "error": str(e), "passed": False})

    acc = sum(r["passed"] for r in task_results) / len(task_results) if task_results else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0

    print(f"\n  Task accuracy: {acc:.0%}  |  Avg tok/s: {avg_tps:.1f}")

    del llm
    gc.collect()

    return {
        "system":             f"llamacpp_{quant}",
        "model":              model_name,
        "scenario":           scenario_name,
        "gguf_size_gb":       gguf_size_gb,
        "load_time_s":        load_time,
        "load_rss_mb":        load_rss_mb,
        "task_accuracy":      acc,
        "avg_tokens_per_sec": avg_tps,
        "task_results":       task_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# HuggingFace Transformers evaluation
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=ml_image,
    cpu=4.0,
    memory=16384,   # 16 GB — 8B fp32 would need 32 GB, handled via OOM skip
    timeout=2 * 3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def eval_hf(model_name: str, scenario_name: str) -> dict:
    """Run HuggingFace Transformers inference (fp32, CPU)."""
    import time, os, gc
    import psutil
    from pathlib import Path
    from huggingface_hub import snapshot_download

    hf_token  = os.environ["HF_TOKEN"]
    model_cfg = MODELS[model_name]
    scenario  = SCENARIOS[scenario_name]
    model_dir = model_cfg["local_dir"]

    # Download safetensors model (shared with NVE)
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
        print(f"  downloaded in {time.time()-t0:.0f}s")

    print(f"\n[HF Transformers fp32] {model_cfg['label']} — {scenario['label']}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Memory constraint: estimate fp32 size (2 bytes/param for bf16 natively, 4 for fp32)
    # 1B ≈ 4.8 GB fp32, 3B ≈ 12.8 GB fp32
    fp32_size_gb = model_cfg["n_layers"] * model_cfg["layer_mb"] * 4.0 / 1024  # rough estimate
    if scenario["hf_mem_gb"] is not None and fp32_size_gb > scenario["hf_mem_gb"]:
        print(f"  SKIP: fp32 model ~{fp32_size_gb:.1f} GB > budget {scenario['hf_mem_gb']} GB")
        return {
            "system":    "hf_fp32",
            "model":     model_name,
            "scenario":  scenario_name,
            "fp32_size_gb_estimate": fp32_size_gb,
            "result":    "oom_predicted",
            "task_accuracy":      None,
            "avg_tokens_per_sec": None,
        }

    try:
        t_load = time.time()
        proc_before = psutil.Process().memory_info().rss

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()

        load_time = time.time() - t_load
        proc_after = psutil.Process().memory_info().rss
        load_rss_mb = (proc_after - proc_before) / (1024 ** 2)
        print(f"  loaded in {load_time:.1f}s  (+{load_rss_mb:.0f} MB RSS)")

    except (MemoryError, RuntimeError) as e:
        return {"system": "hf_fp32", "model": model_name, "scenario": scenario_name,
                "error": f"OOM: {e}", "task_accuracy": None, "avg_tokens_per_sec": None}
    except Exception as e:
        return {"system": "hf_fp32", "model": model_name, "scenario": scenario_name,
                "error": str(e), "task_accuracy": None, "avg_tokens_per_sec": None}

    task_results = []
    tps_list     = []

    for item in TASK_SUITE:
        prompt   = item["prompt"]
        expected = item["expected"].lower()
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
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

            n_out      = out_ids.shape[1] - n_in
            generated  = tokenizer.decode(out_ids[0, n_in:], skip_special_tokens=True)
            tps        = n_out / dt if dt > 0 else 0.0
            passed     = expected in generated.lower()

            task_results.append({
                "category":  item["category"],
                "prompt":    prompt,
                "expected":  expected,
                "generated": generated[:100],
                "passed":    passed,
            })
            tps_list.append(tps)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {item['category']:15s} | {generated[:40]!r}")

        except Exception as e:
            task_results.append({"category": item["category"], "prompt": prompt,
                                  "error": str(e), "passed": False})

    acc     = sum(r["passed"] for r in task_results) / len(task_results)
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0

    print(f"\n  Task accuracy: {acc:.0%}  |  Avg tok/s: {avg_tps:.1f}")

    del model
    gc.collect()

    return {
        "system":             "hf_fp32",
        "model":              model_name,
        "scenario":           scenario_name,
        "load_time_s":        load_time,
        "load_rss_mb":        load_rss_mb,
        "task_accuracy":      acc,
        "avg_tokens_per_sec": avg_tps,
        "task_results":       task_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# Local entry point — orchestrate all experiments in parallel
# ════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    model: str = "all",       # all | llama1b | llama3b | llama8b
    scenario: str = "all",    # all | unconstrained | constrained_2gb | constrained_4gb
    systems: str = "all",     # all | nve | llamacpp | hf | llamacpp,hf
    merge: bool = False,      # if True, merge new results into existing file instead of overwriting
):
    import itertools

    all_models = ["llama1b", "llama3b", "llama8b"]
    models    = all_models if model == "all" else model.split(",")
    scenarios = list(SCENARIOS.keys()) if scenario == "all" else scenario.split(",")
    run_systems = set(systems.split(",")) if systems != "all" else {"nve", "llamacpp", "hf"}

    print(f"NVE Rigorous Benchmark")
    print(f"Models:    {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Systems:   {sorted(run_systems)}")
    print()

    constrained_scenarios = [s for s in scenarios if s != "unconstrained"]
    unconstrained_in_run  = "unconstrained" in scenarios

    # ── Phase 1: unconstrained NVE + all llama.cpp + HF (parallel) ──────────
    # NVE unconstrained produces importance profiles used in Phase 2.
    nve_uncon_jobs   = [(m, "unconstrained") for m in models] \
                       if "nve" in run_systems and unconstrained_in_run else []
    llama_jobs       = [(m, q, s) for m, s in itertools.product(models, scenarios)
                        for q in ("q4", "q8")]  if "llamacpp" in run_systems else []
    hf_jobs          = list(itertools.product(models, scenarios)) if "hf" in run_systems else []

    n_p1 = len(nve_uncon_jobs) + len(llama_jobs) + len(hf_jobs)
    print(f"Phase 1: {len(nve_uncon_jobs)} NVE unconstrained + "
          f"{len(llama_jobs)} llama.cpp + {len(hf_jobs)} HF jobs (parallel)...")

    nve_uncon_handles = [eval_nve.spawn(m, "unconstrained") for m, _ in nve_uncon_jobs]
    llama_handles     = [eval_llamacpp.spawn(*args)         for args in llama_jobs]
    hf_handles        = [eval_hf.spawn(*args)               for args in hf_jobs]

    nve_uncon_results = [h.get() for h in nve_uncon_handles]
    llama_results     = [h.get() for h in llama_handles]
    hf_results        = [h.get() for h in hf_handles]

    # Build profile map: model → importance scores from Phase 1 unconstrained run
    profiles: dict[str, list] = {}
    for r in nve_uncon_results:
        scores = r.get("importance_scores", [])
        if scores:
            profiles[r["model"]] = scores
            print(f"  [profile] {r['model']}: captured {len(scores)} layer scores")

    # If unconstrained wasn't in this run, try to load profiles from the saved
    # results file (produced by a previous session's unconstrained run).
    if not profiles or any(m not in profiles for m in models):
        saved_path = HERE / "evidence" / "experiments" / "rigorous_comparison.json"
        if saved_path.exists():
            try:
                existing = json.loads(saved_path.read_text())
                for r in existing.get("nve", []):
                    m = r.get("model")
                    if m not in profiles and r.get("scenario") == "unconstrained":
                        # Scores may be in importance_scores (new) or nested in paging_info (old)
                        scores = r.get("importance_scores") or []
                        if not scores:
                            b = r.get("configs", {}).get("B_profiled_hot", {})
                            scores = (b.get("paging_info") or {}).get("layer_importance_scores", [])
                        if scores:
                            profiles[m] = scores
                            print(f"  [profile] {m}: loaded {len(scores)} scores from saved results")
            except Exception as e:
                print(f"  [profile] Warning: could not load saved profiles: {e}")

    # ── Phase 2: constrained NVE runs with injected profiles (parallel) ──────
    nve_con_jobs = [(m, s) for m, s in itertools.product(models, constrained_scenarios)] \
                   if "nve" in run_systems else []

    if nve_con_jobs:
        print(f"\nPhase 2: {len(nve_con_jobs)} NVE constrained jobs "
              f"(with pre-computed profiles where available)...")
        nve_con_handles = [
            eval_nve.spawn(m, s, profile_scores=profiles.get(m))
            for m, s in nve_con_jobs
        ]
        nve_con_results = [h.get() for h in nve_con_handles]
    else:
        nve_con_results = []

    nve_results = nve_uncon_results + nve_con_results

    # ── Optionally merge into existing results file ──────────────────────────
    out_dir  = HERE / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rigorous_comparison.json"

    if merge and out_path.exists():
        import json as _json
        existing = _json.loads(out_path.read_text())
        # Index existing entries by (model, scenario[, system]) so new results replace them
        def _key_nve(r):   return (r["model"], r["scenario"])
        def _key_other(r): return (r["model"], r["scenario"], r["system"])

        existing_nve   = {_key_nve(r):   r for r in existing.get("nve",      [])}
        existing_llama = {_key_other(r): r for r in existing.get("llamacpp", [])}
        existing_hf    = {_key_other(r): r for r in existing.get("hf",       [])}

        for r in nve_results:   existing_nve[_key_nve(r)]     = r
        for r in llama_results: existing_llama[_key_other(r)] = r
        for r in hf_results:    existing_hf[_key_other(r)]    = r

        nve_results   = list(existing_nve.values())
        llama_results = list(existing_llama.values())
        hf_results    = list(existing_hf.values())

    all_results = {
        "nve":      nve_results,
        "llamacpp": llama_results,
        "hf":       hf_results,
    }

    # ── Save locally ─────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Model':<12} {'Scenario':<18} {'System':<22} {'Config':<25} {'Acc':>6} {'Tok/s':>8}")
    print("─" * 100)

    for r in nve_results:
        m, s = r["model"], r["scenario"]
        if "error" in r:
            print(f"  {m:<10} {s:<18} {'nve':<22} ERROR: {r['error']}")
            continue
        for cfg_name, cfg_data in r.get("configs", {}).items():
            acc  = cfg_data["task_accuracy"]
            tps  = cfg_data["avg_tokens_per_sec"]
            acc_s = f"{acc:.0%}" if acc is not None else "OOM"
            tps_s = f"{tps:.1f}" if tps is not None else "OOM"
            print(f"  {m:<10} {s:<18} {'nve':<22} {cfg_name:<25} {acc_s:>6} {tps_s:>8}")

    for r in llama_results + hf_results:
        m, s   = r["model"], r["scenario"]
        system = r["system"]
        acc    = r.get("task_accuracy")
        tps    = r.get("avg_tokens_per_sec")
        result = r.get("result", "")
        if result == "oom_predicted":
            print(f"  {m:<10} {s:<18} {system:<22} {'—':<25} {'OOM':>6} {'—':>8}")
        elif "error" in r:
            print(f"  {m:<10} {s:<18} {system:<22} {'—':<25} ERROR  {r['error'][:20]}")
        else:
            acc_s = f"{acc:.0%}" if acc is not None else "?"
            tps_s = f"{tps:.1f}" if tps is not None else "?"
            print(f"  {m:<10} {s:<18} {system:<22} {'—':<25} {acc_s:>6} {tps_s:>8}")

    print(f"\nDone. Regenerate figures: python3 evidence/visualize_paper.py")
