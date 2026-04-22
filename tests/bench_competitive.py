#!/usr/bin/env python3
"""
NVE Competitive Benchmark — Head-to-head comparison against:
  1. HuggingFace Transformers (PyTorch, bf16)
  2. HuggingFace Transformers (PyTorch, int8 via bitsandbytes if available)
  3. DeepSpeed-Inference (CPU)
  4. llama-cpp-python (GGUF, Q4_0)
  5. ONNX Runtime (if available)

All benchmarks: CPU-only, greedy decoding, same prompts, same token count.
"""

import os
import sys
import time
import json
import gc
import subprocess
import resource

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(os.cpu_count() or 4)

from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("BENCH_MODEL", "openai-community/gpt2")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_NEW_TOKENS = int(os.environ.get("BENCH_TOKENS", "30"))
NVE_BIN = os.environ.get("NVE_BIN", "./target/release/nve")

PROMPTS = [
    "The theory of general relativity explains that",
    "The three branches of the United States government are",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):",
    "In machine learning, gradient descent is used to",
]


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def unique_bigram_ratio(token_ids):
    if len(token_ids) < 2:
        return 1.0
    bigrams = [(token_ids[i], token_ids[i+1]) for i in range(len(token_ids)-1)]
    return len(set(bigrams)) / len(bigrams)


# ─── Engine: HuggingFace Transformers (bf16) ─────────────────────────────────

def bench_hf_bf16():
    print("\n" + "=" * 70)
    print("  ENGINE: HuggingFace Transformers (PyTorch, bf16, CPU)")
    print("=" * 70)

    rss_before = get_rss_mb()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {load_time:.1f}s | {total_params/1e9:.2f}B params | RSS: {rss_after_load:.0f} MB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        gc.collect()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - t0

        gen_ids = outputs[0][prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        tok_s = len(gen_ids) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_ids),
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": unique_bigram_ratio(gen_ids),
        })
        print(f"  [{len(gen_ids)} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  Summary: {avg_tps:.1f} tok/s avg | {rss_peak:.0f} MB peak RSS")

    del model
    gc.collect()
    return {
        "engine": "HF Transformers (bf16)",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": rss_peak,
        "load_time_s": load_time,
        "results": results,
    }


# ─── Engine: HuggingFace Transformers (float32) ─────────────────────────────

def bench_hf_fp32():
    print("\n" + "=" * 70)
    print("  ENGINE: HuggingFace Transformers (PyTorch, float32, CPU)")
    print("=" * 70)

    rss_before = get_rss_mb()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {load_time:.1f}s | {total_params/1e9:.2f}B params | RSS: {rss_after_load:.0f} MB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        gc.collect()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - t0

        gen_ids = outputs[0][prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        tok_s = len(gen_ids) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_ids),
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": unique_bigram_ratio(gen_ids),
        })
        print(f"  [{len(gen_ids)} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  Summary: {avg_tps:.1f} tok/s avg | {rss_peak:.0f} MB peak RSS")

    del model
    gc.collect()
    return {
        "engine": "HF Transformers (fp32)",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": rss_peak,
        "load_time_s": load_time,
        "results": results,
    }


# ─── Engine: DeepSpeed ───────────────────────────────────────────────────────

def bench_deepspeed():
    try:
        import deepspeed
    except ImportError:
        print("\n  [SKIP] DeepSpeed not installed")
        return None

    print("\n" + "=" * 70)
    print("  ENGINE: DeepSpeed-Inference (CPU, bf16)")
    print("=" * 70)

    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        ds_model = deepspeed.init_inference(
            model,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=False,
        )
        model = ds_model.module
        print(f"  DeepSpeed initialized (CPU mode)")
    except Exception as e:
        print(f"  DeepSpeed init failed: {e}, running with basic model")

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    print(f"  Loaded in {load_time:.1f}s | RSS: {rss_after_load:.0f} MB")

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        gc.collect()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - t0

        gen_ids = outputs[0][prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        tok_s = len(gen_ids) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_ids),
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": unique_bigram_ratio(gen_ids),
        })
        print(f"  [{len(gen_ids)} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  Summary: {avg_tps:.1f} tok/s avg | {rss_peak:.0f} MB peak RSS")

    del model
    gc.collect()
    return {
        "engine": "DeepSpeed (bf16)",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": rss_peak,
        "load_time_s": load_time,
        "results": results,
    }


# ─── Engine: llama-cpp-python ────────────────────────────────────────────────

def bench_llama_cpp():
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n  [SKIP] llama-cpp-python not installed")
        return None

    # Need a GGUF model — try to find or convert one
    gguf_path = os.environ.get("GGUF_MODEL")
    if not gguf_path:
        # Try common locations
        for candidate in [
            os.path.expanduser("~/.cache/nve/models/gpt2.gguf"),
            os.path.expanduser("~/.cache/nve/models/gpt2-q4.gguf"),
            "/tmp/gpt2-q4.gguf",
        ]:
            if os.path.exists(candidate):
                gguf_path = candidate
                break

    if not gguf_path:
        print("\n  [SKIP] No GGUF model found. Set GGUF_MODEL env var.")
        return None

    print("\n" + "=" * 70)
    print(f"  ENGINE: llama.cpp (GGUF, {os.path.basename(gguf_path)})")
    print("=" * 70)

    t0 = time.time()
    model = Llama(model_path=gguf_path, n_ctx=512, verbose=False)
    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    print(f"  Loaded in {load_time:.1f}s | RSS: {rss_after_load:.0f} MB")

    results = []
    for prompt in PROMPTS:
        gc.collect()
        t0 = time.time()
        output = model(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0, echo=False)
        elapsed = time.time() - t0

        text = output["choices"][0]["text"] if output["choices"] else ""
        n_tok = output["usage"]["completion_tokens"] if "usage" in output else len(text.split())
        tok_s = n_tok / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": n_tok,
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": 0.0,  # can't easily get token IDs from llama.cpp
        })
        print(f"  [{n_tok} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  Summary: {avg_tps:.1f} tok/s avg | {rss_peak:.0f} MB peak RSS")

    del model
    gc.collect()
    return {
        "engine": "llama.cpp (GGUF Q4)",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": rss_peak,
        "load_time_s": load_time,
        "results": results,
    }


# ─── Engine: ONNX Runtime ───────────────────────────────────────────────────

def bench_onnx():
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("\n  [SKIP] optimum[onnxruntime] not installed")
        return None

    print("\n" + "=" * 70)
    print("  ENGINE: ONNX Runtime (CPU)")
    print("=" * 70)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    try:
        model = ORTModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN or None,
            export=True,
        )
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return None

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    print(f"  Loaded in {load_time:.1f}s | RSS: {rss_after_load:.0f} MB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        gc.collect()
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
        elapsed = time.time() - t0

        gen_ids = outputs[0][prompt_len:].tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        tok_s = len(gen_ids) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_ids),
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": unique_bigram_ratio(gen_ids),
        })
        print(f"  [{len(gen_ids)} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  Summary: {avg_tps:.1f} tok/s avg | {rss_peak:.0f} MB peak RSS")

    del model
    gc.collect()
    return {
        "engine": "ONNX Runtime",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": rss_peak,
        "load_time_s": load_time,
        "results": results,
    }


# ─── Engine: NVE (via subprocess) ───────────────────────────────────────────

def bench_nve(mode, label, extra_args=None):
    if not os.path.exists(NVE_BIN):
        print(f"\n  [SKIP] NVE binary not found at {NVE_BIN}")
        return None

    print("\n" + "=" * 70)
    print(f"  ENGINE: NVE — {label}")
    print("=" * 70)

    model_path = MODEL_ID
    # Try to resolve local path
    for candidate in [
        os.path.expanduser(f"~/.cache/nve/models/{MODEL_ID.replace('/', '--')}"),
        f"/mnt/ex1/apps/general-agent/.hf_cache/nve/models/{MODEL_ID.replace('/', '--')}",
    ]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    results = []
    rss_peak = 0

    for prompt in PROMPTS:
        cmd = [
            NVE_BIN, "generate",
            "-m", model_path,
            "-p", prompt,
            "-n", str(MAX_NEW_TOKENS),
            "-t", "0.0",
            "--top-p", "1.0",
        ]
        if extra_args:
            cmd.extend(extra_args)

        gc.collect()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - t0
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {prompt[:40]}...")
            continue

        # Parse decode speed from NVE output
        tok_s = 0
        n_tok = 0
        text = ""
        for line in output.split("\n"):
            if "Decode speed" in line:
                try:
                    tok_s = float(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass
            elif "Generated tokens:" in line:
                try:
                    n_tok = int(line.split(":")[1].strip())
                except Exception:
                    pass

        # Try to extract generated text (first non-info, non-stat line after prompt)
        lines = output.split("\n")
        for line in lines:
            if line.startswith(prompt[:20]):
                text = line[len(prompt):].strip()
                break

        if tok_s == 0 and elapsed > 0 and n_tok > 0:
            tok_s = n_tok / elapsed

        results.append({
            "prompt": prompt,
            "output": text[:200],
            "tokens": n_tok,
            "time_s": elapsed,
            "tok_s": tok_s,
            "bigram_uniq": 0.0,
        })
        print(f"  [{n_tok} tok, {tok_s:.1f} tok/s] {prompt[:40]}...")

    avg_tps = sum(r["tok_s"] for r in results) / max(len(results), 1)
    print(f"  Summary: {avg_tps:.1f} tok/s avg")

    return {
        "engine": f"NVE ({label})",
        "avg_tok_s": avg_tps,
        "peak_rss_mb": 0,  # Can't measure subprocess RSS easily
        "load_time_s": 0,
        "results": results,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def print_comparison(all_results):
    print("\n" + "=" * 80)
    print("  COMPETITIVE BENCHMARK — FINAL COMPARISON")
    print("=" * 80)

    valid = [r for r in all_results if r is not None]
    if not valid:
        print("  No results to compare!")
        return

    print(f"\n  Model: {MODEL_ID} | Tokens: {MAX_NEW_TOKENS} | Greedy | CPU only\n")

    # Header
    print(f"  {'Engine':<35} {'Avg tok/s':>10} {'Peak RSS':>10} {'Load (s)':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    # Sort by throughput
    valid.sort(key=lambda x: x["avg_tok_s"], reverse=True)

    for r in valid:
        rss = f"{r['peak_rss_mb']:.0f} MB" if r['peak_rss_mb'] > 0 else "N/A"
        load = f"{r['load_time_s']:.1f}s" if r['load_time_s'] > 0 else "N/A"
        print(f"  {r['engine']:<35} {r['avg_tok_s']:>9.1f}x {rss:>10} {load:>10}")

    # Find NVE and best competitor
    nve_results = [r for r in valid if "NVE" in r["engine"]]
    non_nve = [r for r in valid if "NVE" not in r["engine"]]

    if nve_results and non_nve:
        best_nve = max(nve_results, key=lambda x: x["avg_tok_s"])
        best_other = max(non_nve, key=lambda x: x["avg_tok_s"])
        ratio = best_nve["avg_tok_s"] / best_other["avg_tok_s"] if best_other["avg_tok_s"] > 0 else 0
        print(f"\n  Best NVE vs best competitor: {ratio:.2f}x")

    # Output comparison for first prompt
    print(f"\n  Output sample (prompt: \"{PROMPTS[0][:45]}...\"):")
    for r in valid:
        if r["results"]:
            out = r["results"][0].get("output", "")[:80]
            print(f"    {r['engine']:<30} -> {out}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  NVE COMPETITIVE BENCHMARK SUITE")
    print(f"  Model: {MODEL_ID}")
    print(f"  Tokens: {MAX_NEW_TOKENS} | Greedy decoding | CPU only")
    print("=" * 80)

    all_results = []

    # 1. HuggingFace fp32
    all_results.append(bench_hf_fp32())
    gc.collect()
    time.sleep(1)

    # 2. HuggingFace bf16
    all_results.append(bench_hf_bf16())
    gc.collect()
    time.sleep(1)

    # 3. DeepSpeed
    all_results.append(bench_deepspeed())
    gc.collect()
    time.sleep(1)

    # 4. llama.cpp
    all_results.append(bench_llama_cpp())
    gc.collect()
    time.sleep(1)

    # 5. ONNX Runtime
    all_results.append(bench_onnx())
    gc.collect()
    time.sleep(1)

    # 6. NVE baseline (bf16, all layers)
    all_results.append(bench_nve("baseline", "Baseline bf16"))
    gc.collect()

    # 7. NVE Q4 (quantized, all layers)
    all_results.append(bench_nve("quant", "Q4 all layers", ["--paged", "--quantize", "q4"]))
    gc.collect()

    # 8. NVE profiled hot-only
    all_results.append(bench_nve("profiled", "Profiled Hot-Only", [
        "--paged", "--hot-only", "--profile",
        "--hot-budget-mb", "250", "--warm-budget-mb", "500",
    ]))
    gc.collect()

    # 9. NVE profiled + quantized
    all_results.append(bench_nve("pg", "PG 2.0bpw + AWQ", [
        "--paged", "--hot-only", "--profile",
        "--quantize", "pg:2.0",
        "--hot-budget-mb", "250", "--warm-budget-mb", "500",
    ]))
    gc.collect()

    print_comparison(all_results)

    # Save JSON
    out_path = os.environ.get("BENCH_OUTPUT", "tests/bench_competitive_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "max_new_tokens": MAX_NEW_TOKENS,
            "engines": [r for r in all_results if r is not None],
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")
