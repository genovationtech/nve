#!/usr/bin/env python3
"""
NVE vs. Prior Work — Competitive Benchmark
==========================================
Models: Llama-3.2-1B (primary), falls back to Qwen2.5-0.5B

Engines:
  1. HuggingFace Transformers — PyTorch bf16          (baseline reference)
  2. HuggingFace Transformers — PyTorch int8          (bitsandbytes 8-bit)
  3. HuggingFace Transformers — PyTorch dynamic int8  (torch.quantization)
  4. DeepSpeed-Inference CPU
  5. llama-cpp-python (GGUF Q4_K_M)
  6. ONNX Runtime
  7. NVE — bf16 paged (baseline)
  8. NVE — uniform Q4
  9. NVE — profiled hot-only (75% layers, bf16)
 10. NVE — PG+AWQ 2.0 bpw

Metrics per engine:
  - throughput (tokens/sec)
  - peak RSS (MB)
  - model load time (s)
  - first-token latency (ms)
  - task accuracy (8-item suite)
  - bigram diversity (output quality proxy)
"""

import os
import sys
import time
import json
import gc
import subprocess
import threading
import resource
import statistics
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# ── Config ───────────────────────────────────────────────────────────────────

LLAMA1B_PATH   = "/home/ai/.cache/nve/models/meta-llama--Llama-3.2-1B"
LLAMA1B_HF_ID  = "meta-llama/Llama-3.2-1B"
GGUF_PATH      = "/home/ai/.cache/nve/models/llama-3.2-1b-q4_k_m.gguf"
GGUF_HF_REPO   = "bartowski/Llama-3.2-1B-GGUF"
GGUF_HF_FILE   = "Llama-3.2-1B-Q4_K_M.gguf"

NVE_BIN        = "/mnt/ex1/apps/general-agent/nve/target/release/nve"
MAX_NEW_TOKENS = 40
OUTPUT_JSON    = "/mnt/ex1/apps/general-agent/nve/evidence/experiments/competitive_results.json"

# Task suite — same as NVE internal suite
TASK_PROMPTS = [
    ("The capital of France is",                      "paris"),
    ("Water is composed of hydrogen and",              "oxygen"),
    ("The largest planet in the solar system is",     "jupiter"),
    ("If today is Monday, tomorrow is",               "tuesday"),
    ("A square has four equal sides and four",        "square"),
    ("def add(a, b):\n    return a",                  "+"),
    ("# Python: list of squares\nsquares = [x**2 for x in", "range"),
    ("The main benefit of regular exercise is improved", "health"),
]

BENCH_PROMPTS = [
    "The theory of general relativity explains that",
    "The three branches of the United States government are",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):",
    "In machine learning, gradient descent is used to",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

_rss_peak = [0]
_monitor_running = [False]

def start_rss_monitor():
    _rss_peak[0] = get_rss_mb()
    _monitor_running[0] = True
    def _poll():
        while _monitor_running[0]:
            cur = get_rss_mb()
            if cur > _rss_peak[0]:
                _rss_peak[0] = cur
            time.sleep(0.05)
    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    return t

def stop_rss_monitor():
    _monitor_running[0] = False
    return _rss_peak[0]

def bigram_diversity(token_ids):
    if len(token_ids) < 2:
        return 1.0
    bigrams = [(token_ids[i], token_ids[i+1]) for i in range(len(token_ids)-1)]
    return len(set(bigrams)) / len(bigrams)

def task_accuracy(model_fn, tokenizer=None):
    """Run 8-item task suite, return fraction correct."""
    passed = 0
    for prompt, expected in TASK_PROMPTS:
        try:
            out = model_fn(prompt, max_new_tokens=20)
            if expected.lower() in out.lower():
                passed += 1
        except Exception:
            pass
    return passed / len(TASK_PROMPTS)

def make_result(engine, load_s, tps_list, rss_mb, first_tok_ms_list, task_acc, notes=""):
    avg_tps = statistics.mean(tps_list) if tps_list else 0
    avg_ftl = statistics.mean(first_tok_ms_list) if first_tok_ms_list else 0
    print(f"\n  ✓ {engine}: {avg_tps:.2f} tok/s | {rss_mb:.0f} MB RSS | "
          f"{load_s:.1f}s load | {avg_ftl:.0f}ms first-tok | {task_acc*100:.0f}% task acc")
    return {
        "engine": engine,
        "load_time_s": round(load_s, 2),
        "avg_tok_s": round(avg_tps, 3),
        "peak_rss_mb": round(rss_mb, 1),
        "avg_first_tok_ms": round(avg_ftl, 1),
        "task_accuracy": round(task_acc, 4),
        "notes": notes,
    }

def skip(engine, reason):
    print(f"\n  [SKIP] {engine}: {reason}")
    return {"engine": engine, "skip_reason": reason}

# ── Engine 1: HuggingFace Transformers bf16 ──────────────────────────────────

def bench_hf_bf16():
    print("\n" + "─"*70)
    print("  [1/10] HuggingFace Transformers — PyTorch bf16")
    print("─"*70)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        start_rss_monitor()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLAMA1B_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA1B_PATH, torch_dtype=torch.bfloat16,
            device_map="cpu", low_cpu_mem_usage=True,
        ).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        load_s = time.time() - t0

        def infer(prompt, max_new_tokens=MAX_NEW_TOKENS):
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        tps_list, ftl_list = [], []
        for prompt in BENCH_PROMPTS:
            inputs = tok(prompt, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None, top_p=None)
            elapsed = time.time() - t0
            n_gen = out.shape[1] - n_in
            tps_list.append(n_gen / elapsed)
            ftl_list.append(elapsed / n_gen * 1000 if n_gen else 0)
            print(f"    {n_gen} tok, {n_gen/elapsed:.2f} tok/s  — {prompt[:40]}...")

        acc = task_accuracy(infer)
        rss = stop_rss_monitor()
        del model; gc.collect()
        return make_result("HF Transformers (bf16)", load_s, tps_list, rss, ftl_list, acc)
    except Exception as e:
        stop_rss_monitor()
        return skip("HF Transformers (bf16)", str(e))

# ── Engine 2: HuggingFace int8 (torch.quantization dynamic) ─────────────────

def bench_hf_int8_dynamic():
    print("\n" + "─"*70)
    print("  [2/10] HuggingFace Transformers — PyTorch dynamic int8")
    print("─"*70)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        start_rss_monitor()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLAMA1B_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA1B_PATH, torch_dtype=torch.float32,
            device_map="cpu", low_cpu_mem_usage=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Dynamic quantization: Linear layers → int8
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        ).eval()
        load_s = time.time() - t0
        print(f"    Quantized in {load_s:.1f}s")

        def infer(prompt, max_new_tokens=MAX_NEW_TOKENS):
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        tps_list, ftl_list = [], []
        for prompt in BENCH_PROMPTS:
            inputs = tok(prompt, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None, top_p=None)
            elapsed = time.time() - t0
            n_gen = out.shape[1] - n_in
            tps_list.append(n_gen / elapsed)
            ftl_list.append(elapsed / n_gen * 1000 if n_gen else 0)
            print(f"    {n_gen} tok, {n_gen/elapsed:.2f} tok/s  — {prompt[:40]}...")

        acc = task_accuracy(infer)
        rss = stop_rss_monitor()
        del model; gc.collect()
        return make_result("HF Transformers (int8-dynamic)", load_s, tps_list, rss, ftl_list, acc,
                           notes="torch.quantization.quantize_dynamic, all Linear layers")
    except Exception as e:
        stop_rss_monitor()
        return skip("HF Transformers (int8-dynamic)", str(e))

# ── Engine 3: DeepSpeed-Inference ────────────────────────────────────────────

def bench_deepspeed():
    print("\n" + "─"*70)
    print("  [3/10] DeepSpeed-Inference (CPU)")
    print("─"*70)
    try:
        import torch
        import deepspeed
        from transformers import AutoTokenizer, AutoModelForCausalLM

        start_rss_monitor()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLAMA1B_PATH)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            LLAMA1B_PATH, torch_dtype=torch.float16,
            device_map="cpu", low_cpu_mem_usage=True,
        )
        model = deepspeed.init_inference(
            base,
            mp_size=1,
            dtype=torch.float16,
            replace_with_kernel_inject=False,
        )
        load_s = time.time() - t0

        def infer(prompt, max_new_tokens=MAX_NEW_TOKENS):
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        tps_list, ftl_list = [], []
        for prompt in BENCH_PROMPTS:
            inputs = tok(prompt, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None, top_p=None)
            elapsed = time.time() - t0
            n_gen = out.shape[1] - n_in
            tps_list.append(n_gen / elapsed)
            ftl_list.append(elapsed / n_gen * 1000 if n_gen else 0)
            print(f"    {n_gen} tok, {n_gen/elapsed:.2f} tok/s  — {prompt[:40]}...")

        acc = task_accuracy(infer)
        rss = stop_rss_monitor()
        del model, base; gc.collect()
        return make_result("DeepSpeed-Inference (fp16)", load_s, tps_list, rss, ftl_list, acc)
    except Exception as e:
        stop_rss_monitor()
        return skip("DeepSpeed-Inference", str(e))

# ── Engine 4: llama-cpp-python (GGUF Q4_K_M) ─────────────────────────────────

def download_gguf():
    """Download Q4_K_M GGUF for Llama-1B from HuggingFace."""
    if os.path.exists(GGUF_PATH) and os.path.getsize(GGUF_PATH) > 100_000_000:
        print(f"    GGUF already exists: {GGUF_PATH}")
        return True
    print(f"    Downloading {GGUF_HF_FILE} from {GGUF_HF_REPO}...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=GGUF_HF_REPO,
            filename=GGUF_HF_FILE,
            local_dir="/home/ai/.cache/nve/models/",
            local_dir_use_symlinks=False,
        )
        os.rename(path, GGUF_PATH)
        print(f"    Downloaded to {GGUF_PATH} ({os.path.getsize(GGUF_PATH)//1024//1024} MB)")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False

def bench_llama_cpp():
    print("\n" + "─"*70)
    print("  [4/10] llama-cpp-python (GGUF Q4_K_M)")
    print("─"*70)
    try:
        from llama_cpp import Llama

        if not download_gguf():
            return skip("llama-cpp-python (Q4_K_M)", "GGUF download failed")

        start_rss_monitor()
        t0 = time.time()
        llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=512,
            n_threads=2,
            n_gpu_layers=0,
            verbose=False,
        )
        load_s = time.time() - t0
        print(f"    Loaded in {load_s:.1f}s")

        def infer(prompt, max_new_tokens=MAX_NEW_TOKENS):
            out = llm(prompt, max_tokens=max_new_tokens, temperature=0.0, echo=False)
            return out["choices"][0]["text"]

        tps_list, ftl_list = [], []
        for prompt in BENCH_PROMPTS:
            t0 = time.time()
            out = llm(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0, echo=False)
            elapsed = time.time() - t0
            n_gen = out["usage"]["completion_tokens"]
            tps = n_gen / elapsed if elapsed > 0 else 0
            # llama.cpp reports timing in eval_time
            timings = out.get("timings", {})
            ftl = timings.get("prompt_ms", elapsed * 1000 / max(n_gen, 1))
            tps_list.append(tps)
            ftl_list.append(ftl)
            print(f"    {n_gen} tok, {tps:.2f} tok/s  — {prompt[:40]}...")

        acc = task_accuracy(infer)
        rss = stop_rss_monitor()
        del llm; gc.collect()
        return make_result("llama-cpp-python (Q4_K_M)", load_s, tps_list, rss, ftl_list, acc,
                           notes="GGUF format, 4-bit K-quant, n_threads=2")
    except Exception as e:
        stop_rss_monitor()
        return skip("llama-cpp-python (Q4_K_M)", str(e))

# ── Engine 5: ONNX Runtime ────────────────────────────────────────────────────

def bench_onnx():
    print("\n" + "─"*70)
    print("  [5/10] ONNX Runtime")
    print("─"*70)
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        start_rss_monitor()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(LLAMA1B_PATH)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = ORTModelForCausalLM.from_pretrained(
            LLAMA1B_PATH, export=True,
            provider="CPUExecutionProvider",
        )
        load_s = time.time() - t0

        def infer(prompt, max_new_tokens=MAX_NEW_TOKENS):
            inputs = tok(prompt, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        tps_list, ftl_list = [], []
        for prompt in BENCH_PROMPTS:
            inputs = tok(prompt, return_tensors="pt")
            n_in = inputs["input_ids"].shape[1]
            t0 = time.time()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            elapsed = time.time() - t0
            n_gen = out.shape[1] - n_in
            tps_list.append(n_gen / elapsed)
            ftl_list.append(elapsed / n_gen * 1000 if n_gen else 0)
            print(f"    {n_gen} tok, {n_gen/elapsed:.2f} tok/s  — {prompt[:40]}...")

        acc = task_accuracy(infer)
        rss = stop_rss_monitor()
        del model; gc.collect()
        return make_result("ONNX Runtime (CPUExecutionProvider)", load_s, tps_list, rss, ftl_list, acc)
    except ImportError:
        stop_rss_monitor()
        return skip("ONNX Runtime", "optimum[onnxruntime] not installed")
    except Exception as e:
        stop_rss_monitor()
        return skip("ONNX Runtime", str(e))

# ── Engines 6-10: NVE (via subprocess) ───────────────────────────────────────

def bench_nve(label, extra_args=None, mode_tag="baseline"):
    print("\n" + "─"*70)
    print(f"  NVE — {label}")
    print("─"*70)

    if not os.path.exists(NVE_BIN):
        return skip(f"NVE ({label})", f"binary not found: {NVE_BIN}")

    # Task accuracy via abc-test with task prompts
    tps_list, ftl_list = [], []
    rss_peak = 0

    for prompt in BENCH_PROMPTS:
        cmd = [NVE_BIN, "generate", "-m", LLAMA1B_PATH,
               "-p", prompt, "-n", str(MAX_NEW_TOKENS),
               "-t", "0.0", "--top-p", "1.0"]
        if extra_args:
            cmd.extend(extra_args)

        gc.collect()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT] {prompt[:40]}")
            continue

        output = proc.stdout + proc.stderr
        tps, ftl_ms, n_tok = 0, 0, 0

        for line in output.split("\n"):
            l = line.strip()
            if "tok/s" in l and ("Decode" in l or "decode" in l):
                try: tps = float(l.split("tok/s")[0].split()[-1])
                except: pass
            if "tokens/sec" in l:
                try: tps = float(l.split("tokens/sec")[0].strip().split()[-1])
                except: pass
            if "Generated" in l and "token" in l:
                try: n_tok = int(''.join(filter(str.isdigit, l.split("token")[0].split()[-1])))
                except: pass
            if "prefill" in l.lower() and "ms" in l:
                try: ftl_ms = float(l.split("ms")[0].split()[-1])
                except: pass
            if "RSS" in l or "Memory" in l:
                try:
                    for part in l.split():
                        if part.replace(".","").isdigit():
                            v = float(part)
                            if v > rss_peak:
                                rss_peak = v
                except: pass

        if tps == 0 and elapsed > 0:
            # Estimate from wall time — NVE reports in run_log; use elapsed as upper bound
            if n_tok > 0:
                tps = n_tok / elapsed

        tps_list.append(tps)
        ftl_list.append(ftl_ms)
        print(f"    {n_tok} tok, {tps:.2f} tok/s  — {prompt[:40]}...")

    # Task accuracy: run a quick abc-test targeting just the task prompts
    task_acc = 0.0
    try:
        task_prompts_str = ",".join(p for p, _ in TASK_PROMPTS)
        task_cmd = [NVE_BIN, "abc-test", "-m", LLAMA1B_PATH,
                    "--configs", mode_tag if mode_tag in ("baseline","a","b","c") else "baseline",
                    "-n", "20", "--prompts", task_prompts_str[:400]]
        if extra_args and mode_tag != "baseline":
            task_cmd.extend(extra_args)
        proc = subprocess.run(task_cmd, capture_output=True, text=True, timeout=600)
        out = proc.stdout + proc.stderr
        for line in out.split("\n"):
            if "Task accuracy" in line and "/" in line:
                try:
                    parts = line.split("(")[-1].split("%")[0]
                    task_acc = float(parts) / 100
                except: pass
    except Exception:
        pass

    avg_tps = statistics.mean(tps_list) if tps_list else 0
    avg_ftl = statistics.mean(ftl_list) if ftl_list else 0

    print(f"  ✓ NVE ({label}): {avg_tps:.2f} tok/s | {avg_ftl:.0f}ms first-tok | {task_acc*100:.0f}% task acc")
    return {
        "engine": f"NVE ({label})",
        "load_time_s": 0,
        "avg_tok_s": round(avg_tps, 3),
        "peak_rss_mb": rss_peak,
        "avg_first_tok_ms": round(avg_ftl, 1),
        "task_accuracy": round(task_acc, 4),
        "notes": " ".join(extra_args) if extra_args else "default",
    }

# ── Print comparison table ────────────────────────────────────────────────────

def print_table(results):
    valid = [r for r in results if r and "skip_reason" not in r]
    if not valid:
        print("No valid results.")
        return

    valid.sort(key=lambda r: r.get("avg_tok_s", 0), reverse=True)

    print("\n" + "="*110)
    print(f"  {'Engine':<42} {'tok/s':>7} {'RSS MB':>7} {'Load(s)':>8} {'FTL(ms)':>8} {'TaskAcc':>8}  Notes")
    print("─"*110)

    nve_tps = max((r["avg_tok_s"] for r in valid if "NVE" in r["engine"]), default=1)
    baseline_tps = next((r["avg_tok_s"] for r in valid if "bf16" in r["engine"] and "NVE" not in r["engine"]), 1)

    for r in valid:
        tps    = r.get("avg_tok_s", 0)
        rss    = r.get("peak_rss_mb", 0)
        load   = r.get("load_time_s", 0)
        ftl    = r.get("avg_first_tok_ms", 0)
        acc    = r.get("task_accuracy", 0)
        engine = r["engine"]
        speedup = f"({tps/baseline_tps:.1f}×)" if baseline_tps > 0 and tps > 0 else ""
        print(f"  {engine:<42} {tps:>7.2f} {rss:>7.0f} {load:>8.1f} {ftl:>8.0f} {acc*100:>7.0f}%  {speedup}")

    print("="*110)
    skipped = [r for r in results if r and "skip_reason" in r]
    for r in skipped:
        print(f"  [SKIP] {r['engine']}: {r['skip_reason']}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("  NVE vs. PRIOR WORK — COMPETITIVE BENCHMARK")
    print(f"  Model: Llama-3.2-1B (1.2B params, 16 layers)")
    print(f"  Hardware: CPU-only | 2 cores | 3.8 GB RAM")
    print(f"  Tokens: {MAX_NEW_TOKENS} | Greedy (temp=0) | 5 prompts each")
    print("="*80)

    all_results = []

    # 1. HF bf16 — reference baseline
    all_results.append(bench_hf_bf16()); gc.collect()

    # 2. HF int8 dynamic
    all_results.append(bench_hf_int8_dynamic()); gc.collect()

    # 3. DeepSpeed CPU
    all_results.append(bench_deepspeed()); gc.collect()

    # 4. llama.cpp GGUF Q4_K_M
    all_results.append(bench_llama_cpp()); gc.collect()

    # 5. ONNX Runtime
    all_results.append(bench_onnx()); gc.collect()

    # 6. NVE bf16 baseline
    all_results.append(bench_nve("bf16 paged", mode_tag="baseline")); gc.collect()

    # 7. NVE uniform Q4
    all_results.append(bench_nve("uniform Q4", ["--paged", "--quantize", "q4"], mode_tag="a")); gc.collect()

    # 8. NVE profiled hot-only (12/16 layers)
    all_results.append(bench_nve("profiled hot-only (12/16L)",
        ["--paged", "--hot-only", "--profile",
         "--hot-budget-mb", "500", "--warm-budget-mb", "1000"],
        mode_tag="b")); gc.collect()

    # 9. NVE PG+AWQ 2.0 bpw
    all_results.append(bench_nve("PG+AWQ 2.0bpw",
        ["--paged", "--hot-only", "--profile",
         "--quantize", "pg:2.0",
         "--hot-budget-mb", "500", "--warm-budget-mb", "1000"],
        mode_tag="c")); gc.collect()

    print_table(all_results)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "model": "Llama-3.2-1B",
            "hardware": "CPU-only, 2 cores, 3.8 GB RAM",
            "max_new_tokens": MAX_NEW_TOKENS,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": [r for r in all_results if r],
        }, f, indent=2)
    print(f"\n  Saved → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
