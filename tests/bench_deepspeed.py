"""
Benchmark: DeepSpeed-Inference vs plain HuggingFace on Llama 3.2 1B (CPU, 500MB-comparable).

Measures tokens/sec and output quality on the same prompts used for NVE testing.
"""

import os
import sys
import time
import gc
import resource

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(2)  # Match constrained environment

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B"
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 20
PROMPTS = [
    "The theory of general relativity explains that",
    "The three branches of the United States government are",
    "Photosynthesis is the process by which plants",
]


def get_rss_mb():
    """Get current RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except:
        pass
    return 0


def set_memory_limit_mb(limit_mb):
    """Set soft memory limit (advisory, not hard-kill)."""
    limit_bytes = limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return True
    except:
        return False


def benchmark_hf_baseline():
    """Plain HuggingFace transformers inference (no DeepSpeed)."""
    print("=" * 70)
    print("  BASELINE: HuggingFace Transformers (CPU, bf16)")
    print("=" * 70)

    rss_before = get_rss_mb()
    print(f"\n  Loading {MODEL_ID}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {load_time:.1f}s | {total_params/1e9:.1f}B params | RSS: {rss_after_load:.0f} MB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n  Generating {MAX_NEW_TOKENS} tokens per prompt (greedy)...\n")

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

        gen_tokens = outputs[0][prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        tok_s = len(gen_tokens) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_tokens),
            "time_s": elapsed,
            "tok_s": tok_s,
        })

        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: \"{text}\"")
        print(f"  {len(gen_tokens)} tokens in {elapsed:.1f}s = {tok_s:.2f} tok/s")
        print(f"  RSS: {get_rss_mb():.0f} MB")
        print()

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  --- HF Baseline Summary ---")
    print(f"  Avg tok/s:    {avg_tps:.2f}")
    print(f"  Peak RSS:     {rss_peak:.0f} MB")
    print(f"  Load time:    {load_time:.1f}s")
    print()

    del model
    gc.collect()
    return results, rss_peak, avg_tps


def benchmark_deepspeed():
    """DeepSpeed-Inference on CPU."""
    try:
        import deepspeed
    except ImportError:
        print("\n  [SKIP] DeepSpeed not installed")
        return None, 0, 0

    print("=" * 70)
    print("  DEEPSPEED: DeepSpeed-Inference (CPU, bf16)")
    print("=" * 70)

    rss_before = get_rss_mb()
    print(f"\n  Loading {MODEL_ID} with DeepSpeed...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize DeepSpeed inference engine
    try:
        ds_model = deepspeed.init_inference(
            model,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=False,  # CPU doesn't support kernel injection
        )
        model = ds_model.module
        print(f"  DeepSpeed initialized (CPU mode)")
    except Exception as e:
        print(f"  DeepSpeed init_inference failed: {e}")
        print(f"  Falling back to DeepSpeed optimize...")
        try:
            # Try basic optimization
            ds_model = deepspeed.init_inference(model, dtype=torch.bfloat16)
            model = ds_model.module
        except Exception as e2:
            print(f"  DeepSpeed fallback also failed: {e2}")
            print(f"  Running without DeepSpeed optimizations")

    load_time = time.time() - t0
    rss_after_load = get_rss_mb()
    print(f"  Loaded in {load_time:.1f}s | RSS: {rss_after_load:.0f} MB")

    print(f"\n  Generating {MAX_NEW_TOKENS} tokens per prompt (greedy)...\n")

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

        gen_tokens = outputs[0][prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        tok_s = len(gen_tokens) / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt,
            "output": text,
            "tokens": len(gen_tokens),
            "time_s": elapsed,
            "tok_s": tok_s,
        })

        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: \"{text}\"")
        print(f"  {len(gen_tokens)} tokens in {elapsed:.1f}s = {tok_s:.2f} tok/s")
        print(f"  RSS: {get_rss_mb():.0f} MB")
        print()

    rss_peak = get_rss_mb()
    avg_tps = sum(r["tok_s"] for r in results) / len(results)
    print(f"  --- DeepSpeed Summary ---")
    print(f"  Avg tok/s:    {avg_tps:.2f}")
    print(f"  Peak RSS:     {rss_peak:.0f} MB")
    print(f"  Load time:    {load_time:.1f}s")
    print()

    del model
    gc.collect()
    return results, rss_peak, avg_tps


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  NVE COMPETITOR BENCHMARK: HF Baseline + DeepSpeed")
    print("  Model: Llama 3.2 1B | CPU only | bf16 | Greedy decoding")
    print("=" * 70 + "\n")

    hf_results, hf_rss, hf_tps = benchmark_hf_baseline()

    gc.collect()
    time.sleep(2)

    ds_results, ds_rss, ds_tps = benchmark_deepspeed()

    # Summary comparison
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'HF Baseline':>15} {'DeepSpeed':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Avg tok/s':<25} {hf_tps:>14.2f} {ds_tps:>14.2f}")
    print(f"  {'Peak RSS (MB)':<25} {hf_rss:>14.0f} {ds_rss:>14.0f}")

    if hf_results and ds_results:
        print(f"\n  Output comparison:")
        for hf, ds in zip(hf_results, ds_results):
            print(f"\n  Prompt: \"{hf['prompt'][:50]}...\"")
            print(f"    HF:  \"{hf['output'][:80]}\"")
            print(f"    DS:  \"{ds['output'][:80]}\"")
            match = "MATCH" if hf['output'].strip() == ds['output'].strip() else "DIFFER"
            print(f"    [{match}]")

    print("\n" + "=" * 70)
