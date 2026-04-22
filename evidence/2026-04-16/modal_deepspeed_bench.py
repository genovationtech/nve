#!/usr/bin/env python3
"""
DeepSpeed-Inference vs HuggingFace baseline on T4 at batch=1 decode.

Purpose: produce the DeepSpeed row that currently sits in the NVE paper's
Limitations section. Benchmarks Llama-3.2-1B, 3B, and Llama-3.1-8B at batch=1
greedy decode for a fair comparison with the NVE W4A8 and llama.cpp rows in
Table 2 of the paper.

Usage:
    modal run evidence/modal_deepspeed_bench.py
    modal run evidence/modal_deepspeed_bench.py --models 1b
"""

import os
import modal

app = modal.App("nve-deepspeed-bench")

ds_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "deepspeed==0.15.1",
        "huggingface_hub==0.24.5",
        "safetensors==0.4.3",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
    )
)


MODELS = {
    "1b": "meta-llama/Llama-3.2-1B",
    "3b": "meta-llama/Llama-3.2-3B",
    "8b": "meta-llama/Llama-3.1-8B",
}

PROMPTS = [
    "The theory of general relativity explains that",
    "The three branches of the United States government are",
    "Photosynthesis is the process by which plants",
]


@app.function(
    image=ds_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": ""})],  # optional: add HF token if needed
)
def bench_one(model_key: str, hf_token: str = "", n_new: int = 32, warmup: int = 3, iters: int = 10) -> dict:
    import os, time, json, torch
    os.environ["HF_TOKEN"] = hf_token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = MODELS[model_key]
    device = "cuda"

    print(f"[{model_key}] loading {model_id} in FP16 on T4...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=hf_token or None
    ).to(device).eval()

    results = {"model": model_id, "key": model_key, "device": "T4", "n_new": n_new, "iters": iters}

    # ---------- HF baseline ----------
    def run_loop(model_, label):
        all_tps = []
        for p in PROMPTS:
            inp = tok(p, return_tensors="pt").to(device)
            # warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model_.generate(**inp, max_new_tokens=n_new, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            # timed
            t0 = time.perf_counter()
            total_new = 0
            for _ in range(iters):
                with torch.no_grad():
                    out = model_.generate(**inp, max_new_tokens=n_new, do_sample=False, use_cache=True)
                total_new += out.shape[1] - inp["input_ids"].shape[1]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tps = total_new / dt
            all_tps.append(tps)
            print(f"[{model_key}][{label}] prompt={p[:30]!r} tps={tps:.2f}", flush=True)
        return {"mean_tps": sum(all_tps) / len(all_tps), "per_prompt_tps": all_tps}

    results["hf_fp16"] = run_loop(model, "HF-FP16")

    # ---------- DeepSpeed-Inference ----------
    try:
        import deepspeed
        print(f"[{model_key}] wrapping in DeepSpeed-Inference (replace_with_kernel_inject=True)...", flush=True)
        # DeepSpeed-Inference wraps the model for kernel injection
        ds_engine = deepspeed.init_inference(
            model,
            mp_size=1,
            dtype=torch.float16,
            replace_with_kernel_inject=True,
        )
        ds_model = ds_engine.module
        results["deepspeed"] = run_loop(ds_model, "DeepSpeed")
    except Exception as e:
        print(f"[{model_key}] DeepSpeed failed: {e}", flush=True)
        results["deepspeed"] = {"error": str(e)}

    del model
    torch.cuda.empty_cache()
    return results


@app.local_entrypoint()
def main(models: str = "1b,3b,8b", n_new: int = 32, iters: int = 10, hf_token: str = ""):
    import json, os
    keys = [k.strip() for k in models.split(",") if k.strip()]
    out = {}
    for k in keys:
        print(f"\n=== Benchmarking {k} ===\n", flush=True)
        out[k] = bench_one.remote(k, hf_token=hf_token or os.environ.get("HF_TOKEN", ""),
                                   n_new=n_new, iters=iters)
    print(json.dumps(out, indent=2))
    with open("/tmp/deepspeed_bench_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved → /tmp/deepspeed_bench_result.json")
