#!/usr/bin/env python3
"""
vLLM baseline at batch=1 greedy decode on T4. Used as the Llama-3-compatible
modern serving baseline, since DeepSpeed-Inference's kernel-inject path doesn't
support Llama 3 (tensor-merge error on checkpoint load).

Usage: modal run evidence/modal_vllm_bench.py --models 1b
"""
import os
import modal

app = modal.App("nve-vllm-bench")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.6.3",
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


@app.function(image=vllm_image, gpu="T4", cpu=4.0, memory=16384, timeout=1800)
def bench(model_key: str = "1b", hf_token: str = "", n_new: int = 32, iters: int = 10) -> dict:
    import os, time
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    from vllm import LLM, SamplingParams

    model_id = MODELS[model_key]
    print(f"[{model_key}] loading {model_id} in FP16 via vLLM...", flush=True)
    llm = LLM(
        model=model_id, dtype="float16", gpu_memory_utilization=0.85,
        max_model_len=512, enforce_eager=False,
    )
    sp = SamplingParams(temperature=0, max_tokens=n_new, ignore_eos=True)

    # warmup
    for p in PROMPTS:
        llm.generate([p], sp, use_tqdm=False)

    all_tps = []
    for p in PROMPTS:
        t0 = time.perf_counter()
        total_new = 0
        for _ in range(iters):
            outs = llm.generate([p], sp, use_tqdm=False)
            total_new += len(outs[0].outputs[0].token_ids)
        dt = time.perf_counter() - t0
        tps = total_new / dt
        all_tps.append(tps)
        print(f"[{model_key}] prompt={p[:30]!r} vllm_tps={tps:.2f}", flush=True)

    return {"model": model_id, "key": model_key, "n_new": n_new, "iters": iters,
            "vllm_mean_tps": sum(all_tps) / len(all_tps), "per_prompt_tps": all_tps}


@app.local_entrypoint()
def main(models: str = "1b", n_new: int = 32, iters: int = 10, hf_token: str = ""):
    import json, os
    token = hf_token or os.environ.get("HF_TOKEN", "")
    out = {}
    for k in [m.strip() for m in models.split(",") if m.strip()]:
        out[k] = bench.remote(k, hf_token=token, n_new=n_new, iters=iters)
    print(json.dumps(out, indent=2))
