#!/usr/bin/env python3
"""
AWQ / GPTQ / NVE MCAP Head-to-Head Comparison
===============================================
Compares NVE MCAP Mixed against AutoAWQ and AutoGPTQ at 4-bit quantization
on the same models and benchmarks.

Metrics:
  - WikiText-2 perplexity (50 sequences × 256 tokens)
  - HellaSwag accuracy (50 examples, 4-way log-likelihood scoring)
  - Decode throughput (tok/s, single-sequence generation)

Models: Llama-3.2-1B, Llama-3.2-3B

Usage:
    modal run evidence/modal_awq_gptq_comparison.py
    modal run evidence/modal_awq_gptq_comparison.py --model llama1b
    modal run evidence/modal_awq_gptq_comparison.py --model llama3b
"""

import os
import modal
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root

app       = modal.App("nve-awq-gptq-comparison")
model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})

MODELS = {
    "llama1b": {
        "hf_id":      "meta-llama/Llama-3.2-1B",
        "local_dir":  "/models/llama1b",
        "label":      "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":      "meta-llama/Llama-3.2-3B",
        "local_dir":  "/models/llama3b",
        "label":      "Llama-3.2-3B",
    },
}

CALIBRATION_PROMPTS = [
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to compute Fibonacci numbers:",
    "The French Revolution began in",
    "In machine learning, gradient descent is",
    "The capital of Japan is",
    "def binary_search(arr, target):",
    "Water molecules consist of",
    "The speed of light in vacuum is approximately",
    "In 1969, humans first landed on",
    "A transformer neural network uses attention to",
    "The mitochondria is the powerhouse of",
    "SELECT * FROM users WHERE",
]

# ── NVE image (same as modal_wikitext_hellaswag.py) ─────────────────────────

BUILD_CMD = (
    "bash -c 'set -o pipefail && "
    "cd /nve-src && "
    "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
    "-gencode arch=compute_75,code=sm_75 "
    "-gencode arch=compute_80,code=sm_80 "
    "-gencode arch=compute_86,code=sm_86 "
    "-Xcompiler -fPIC -shared -o /usr/local/lib/libnve_kernels.so cuda/nve_kernels.cu 2>&1 && "
    "ldconfig && "
    "NVE_KERNELS_PREBUILT=/usr/local/lib CUDA_COMPUTE_CAP=75 CUDA_PATH=/usr/local/cuda "
    "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C link-arg=-L/usr/local/lib -C link-arg=-lnve_kernels\" "
    "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -20 && "
    "cp target/release/nve /usr/local/bin/nve-cuda && "
    "chmod +x /usr/local/bin/nve-cuda && "
    "echo \"Build OK: $(nve-cuda --version 2>&1 | head -1)\"'"
)

nve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "curl", "pkg-config", "libssl-dev", "ca-certificates",
        "build-essential", "cmake", "git",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "pip install huggingface_hub[hf_xfer] tqdm datasets",
    )
    .add_local_dir(
        str(HERE), "/nve-src", copy=True,
        ignore=["target/", ".git/", "docs/", "evidence/figures/",
                "evidence/figures_paper/", "reports/",
                "**/__pycache__/", "*.pyc"],
    )
    .run_commands(BUILD_CMD)
)

# ── AWQ / GPTQ image ────────────────────────────────────────────────────────

quant_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgcc-s1", "ca-certificates", "build-essential", "git")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.43.4", "sentencepiece", "protobuf", "accelerate",
        "huggingface_hub[hf_xfer]", "tqdm", "datasets",
    )
    .pip_install("autoawq==0.2.5")
    .pip_install("auto-gptq==0.6.0", "optimum==1.17.1", "peft==0.8.2")
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def get_wikitext2_sequences(n_sequences: int = 50, seq_len: int = 256) -> list[str]:
    """Load WikiText-2 test sequences (word-level, 256 tokens each)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join([row["text"] for row in ds if row["text"].strip()])
    words = text.split()
    seqs = []
    for i in range(0, len(words) - seq_len, seq_len):
        seqs.append(" ".join(words[i:i + seq_len]))
        if len(seqs) >= n_sequences:
            break
    return seqs


def get_hellaswag_examples(n: int = 50) -> list[dict]:
    """Load HellaSwag validation examples."""
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    examples = []
    for row in ds:
        examples.append({
            "ctx": row["ctx"],
            "endings": row["endings"],
            "label": int(row["label"]),
        })
        if len(examples) >= n:
            break
    return examples


def ensure_model(hf_id: str, local_dir: str):
    """Download HF model weights if not already cached."""
    import os
    marker = os.path.join(local_dir, ".downloaded")
    if os.path.exists(marker):
        print(f"  Model already cached at {local_dir}")
        return
    print(f"  Downloading {hf_id} → {local_dir}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        hf_id,
        local_dir=local_dir,
        ignore_patterns=["*.bin", "original/*"],
    )
    Path(marker).touch()
    model_vol.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# NVE MCAP Mixed evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_nve_perplexity(
    model_dir: str, sequences: list[str], env_overrides: dict,
    hot_budget_mb: int = 14000, warm_budget_mb: int = 14000,
) -> dict:
    import subprocess, os, json, tempfile

    base_env = os.environ.copy()
    base_env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        texts_path = f.name
        json.dump(sequences, f)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        "nve-cuda", "batch-perplexity",
        "-m", model_dir,
        "--texts-file", texts_path,
        "--hot-budget-mb", str(hot_budget_mb),
        "--warm-budget-mb", str(warm_budget_mb),
        "--quantize", "q4",
        "--output", out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=base_env)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(f"  [NVE PPL] failed: {r.stderr[:500]}")
        return {"ppl": float("nan"), "n_tokens": 0}

    try:
        data = json.loads(Path(out_path).read_text())
        return {"ppl": data.get("ppl_overall", float("nan")), "n_tokens": data.get("n_tokens", 0)}
    except Exception as e:
        print(f"  [NVE PPL] parse error: {e}")
        return {"ppl": float("nan"), "n_tokens": 0}
    finally:
        for p in [texts_path, out_path]:
            try: os.unlink(p)
            except: pass


def compute_nve_hellaswag(
    model_dir: str, examples: list[dict], env_overrides: dict,
    hot_budget_mb: int = 14000, warm_budget_mb: int = 14000,
) -> dict:
    import subprocess, os, json, tempfile

    base_env = os.environ.copy()
    base_env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        ex_path = f.name
        json.dump(examples, f)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        "nve-cuda", "batch-hellaswag",
        "-m", model_dir,
        "--examples-file", ex_path,
        "--hot-budget-mb", str(hot_budget_mb),
        "--warm-budget-mb", str(warm_budget_mb),
        "--quantize", "q4",
        "--output", out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=base_env)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(f"  [NVE HS] failed: {r.stderr[:500]}")
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    try:
        data = json.loads(Path(out_path).read_text())
        return {
            "accuracy": data.get("accuracy", 0.0),
            "correct": data.get("correct", 0),
            "total": data.get("total", 0),
        }
    except Exception as e:
        print(f"  [NVE HS] parse error: {e}")
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    finally:
        for p in [ex_path, out_path]:
            try: os.unlink(p)
            except: pass


def nve_throughput(model_dir: str, env_overrides: dict) -> float:
    """Measure NVE decode throughput using a simple generation pass."""
    import subprocess, os, time

    base_env = os.environ.copy()
    base_env.update(env_overrides)

    cmd = [
        "nve-cuda", "generate",
        "-m", model_dir,
        "-p", "The meaning of life is",
        "-n", "100",
        "--temperature", "0",
        "--paged",
        "--hot-budget-mb", "14000",
        "--warm-budget-mb", "14000",
        "--quantize", "q4",
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=base_env)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"  [NVE throughput] failed: {r.stderr[:300]}")
        return 0.0
    return 100.0 / elapsed


@app.function(
    image=nve_image, gpu="A10G", cpu=8.0, memory=32768,
    timeout=14400, volumes={"/models": model_vol}, secrets=[hf_secret],
)
def run_nve_eval(model_name: str):
    """Run NVE MCAP Mixed evaluation: PPL + HellaSwag + throughput."""
    import subprocess, json, os

    cfg = MODELS[model_name]
    ensure_model(cfg["hf_id"], cfg["local_dir"])
    print(f"\n{'='*60}\nNVE MCAP Mixed — {cfg['label']}\n{'='*60}")

    # Warm up importance profile
    for prompt in CALIBRATION_PROMPTS[:3]:
        subprocess.run(
            ["nve-cuda", "generate", "-m", cfg["local_dir"], "-p", prompt,
             "-n", "5", "--temperature", "0", "--paged",
             "--hot-budget-mb", "14000", "--warm-budget-mb", "14000",
             "--quantize", "q4"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    strategies = {
        "NVE W4A16":     {"NVE_NO_W4A8": "1"},
        "NVE W4A8":      {"NVE_W4A8_THRESHOLD": "2.0"},
        "NVE MCAP Mixed": {},
    }

    seqs = get_wikitext2_sequences(n_sequences=50, seq_len=256)
    examples = get_hellaswag_examples(n=50)

    results = {}
    for label, env_ovr in strategies.items():
        print(f"\n--- {label} ---")

        ppl = compute_nve_perplexity(cfg["local_dir"], seqs, env_ovr)
        print(f"  WikiText-2 PPL = {ppl['ppl']:.2f} ({ppl['n_tokens']} tokens)")

        hs = compute_nve_hellaswag(cfg["local_dir"], examples, env_ovr)
        print(f"  HellaSwag = {hs['accuracy']:.1%} ({hs['correct']}/{hs['total']})")

        tps = nve_throughput(cfg["local_dir"], env_ovr)
        print(f"  Throughput = {tps:.1f} tok/s")

        results[label] = {"ppl": ppl["ppl"], "n_tokens": ppl["n_tokens"],
                          "hellaswag": hs["accuracy"], "hs_correct": hs["correct"],
                          "hs_total": hs["total"], "tps": tps}

    return {"model": model_name, "label": cfg["label"], "nve": results}


# ═══════════════════════════════════════════════════════════════════════════════
# AutoAWQ evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=quant_image, gpu="A10G", cpu=8.0, memory=32768,
    timeout=14400, volumes={"/models": model_vol}, secrets=[hf_secret],
)
def run_awq_eval(model_name: str):
    """Quantize with AutoAWQ, then evaluate PPL + HellaSwag + throughput."""
    import torch, time, json, os, math
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg = MODELS[model_name]
    ensure_model(cfg["hf_id"], cfg["local_dir"])
    print(f"\n{'='*60}\nAutoAWQ INT4 — {cfg['label']}\n{'='*60}")

    awq_dir = f"/models/{model_name}_awq_v2"
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantize ─────────────────────────────────────────────────────────────
    if not os.path.exists(os.path.join(awq_dir, "config.json")):
        print("  Quantizing with AutoAWQ (group_size=128, w_bit=4)...")
        t0 = time.perf_counter()
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(
            cfg["local_dir"], device_map="auto",
        )
        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(awq_dir)
        tokenizer.save_pretrained(awq_dir)
        model_vol.commit()
        print(f"  Quantized in {time.perf_counter()-t0:.1f}s → {awq_dir}")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"  AWQ model already cached at {awq_dir}")

    # ── Load quantized model via transformers ────────────────────────────────
    print("  Loading AWQ model via transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        awq_dir, device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── WikiText-2 PPL ───────────────────────────────────────────────────────
    print("  Computing WikiText-2 perplexity (50 seq × 256 tokens)...")
    seqs = get_wikitext2_sequences(n_sequences=50, seq_len=256)
    total_nll = 0.0
    total_tokens = 0
    for seq in seqs:
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        n_tok = input_ids.shape[1] - 1  # loss is over shifted tokens
        total_nll += outputs.loss.item() * n_tok
        total_tokens += n_tok
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    print(f"  PPL = {ppl:.2f} ({total_tokens} tokens)")

    # ── HellaSwag ────────────────────────────────────────────────────────────
    print("  Computing HellaSwag accuracy (50 examples)...")
    examples = get_hellaswag_examples(n=50)
    correct = 0
    for ex in examples:
        ctx = ex["ctx"]
        scores = []
        for ending in ex["endings"]:
            full = ctx + " " + ending
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            ctx_enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=512)
            ctx_len = ctx_enc["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = 0.0
            n_ending = input_ids.shape[1] - ctx_len
            for i in range(ctx_len, input_ids.shape[1]):
                nll += log_probs[i - 1, input_ids[0, i]].item()
            scores.append(nll / max(n_ending, 1))
        pred = scores.index(max(scores))
        if pred == ex["label"]:
            correct += 1
    hs_acc = correct / len(examples) if examples else 0.0
    print(f"  HellaSwag = {hs_acc:.1%} ({correct}/{len(examples)})")

    # ── Throughput ───────────────────────────────────────────────────────────
    print("  Measuring decode throughput...")
    prompt = "The meaning of life is"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model.generate(**enc, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - enc["input_ids"].shape[1]
    tps = n_gen / elapsed
    print(f"  Throughput = {tps:.1f} tok/s ({n_gen} tokens in {elapsed:.2f}s)")

    return {
        "model": model_name, "label": cfg["label"],
        "method": "AutoAWQ INT4",
        "ppl": ppl, "n_tokens": total_tokens,
        "hellaswag": hs_acc, "hs_correct": correct, "hs_total": len(examples),
        "tps": tps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AutoGPTQ evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=quant_image, gpu="A10G", cpu=8.0, memory=32768,
    timeout=14400, volumes={"/models": model_vol}, secrets=[hf_secret],
)
def run_gptq_eval(model_name: str):
    """Quantize with AutoGPTQ, then evaluate PPL + HellaSwag + throughput."""
    import torch, time, json, os, math
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg = MODELS[model_name]
    ensure_model(cfg["hf_id"], cfg["local_dir"])
    print(f"\n{'='*60}\nAutoGPTQ INT4 — {cfg['label']}\n{'='*60}")

    gptq_dir = f"/models/{model_name}_gptq_v2"
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantize using auto_gptq directly ────────────────────────────────────
    if not os.path.exists(os.path.join(gptq_dir, "config.json")):
        print("  Quantizing with GPTQ (bits=4, group_size=128, 128 calibration samples)...")
        t0 = time.perf_counter()

        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from datasets import load_dataset

        quantize_config = BaseQuantizeConfig(
            bits=4, group_size=128, damp_percent=0.1,
            desc_act=False, static_groups=False,
        )

        # Load model for quantization
        model = AutoGPTQForCausalLM.from_pretrained(
            cfg["local_dir"], quantize_config,
        )

        # Build calibration dataset from WikiText-2 train split
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cal_texts = [row["text"] for row in ds if len(row["text"].strip()) > 50][:128]
        cal_data = [
            tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            for text in cal_texts
        ]

        model.quantize(cal_data)
        model.save_quantized(gptq_dir)
        tokenizer.save_pretrained(gptq_dir)
        model_vol.commit()
        print(f"  Quantized in {time.perf_counter()-t0:.1f}s → {gptq_dir}")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"  GPTQ model already cached at {gptq_dir}")

    # ── Load quantized model ─────────────────────────────────────────────────
    print("  Loading GPTQ model...")
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized(
        gptq_dir, device="cuda:0", use_safetensors=True,
    )
    model.eval()
    device = torch.device("cuda:0")

    # ── WikiText-2 PPL ───────────────────────────────────────────────────────
    print("  Computing WikiText-2 perplexity (50 seq × 256 tokens)...")
    seqs = get_wikitext2_sequences(n_sequences=50, seq_len=256)
    total_nll = 0.0
    total_tokens = 0
    for seq in seqs:
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        n_tok = input_ids.shape[1] - 1
        total_nll += outputs.loss.item() * n_tok
        total_tokens += n_tok
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    print(f"  PPL = {ppl:.2f} ({total_tokens} tokens)")

    # ── HellaSwag ────────────────────────────────────────────────────────────
    print("  Computing HellaSwag accuracy (50 examples)...")
    examples = get_hellaswag_examples(n=50)
    correct = 0
    for ex in examples:
        ctx = ex["ctx"]
        scores = []
        for ending in ex["endings"]:
            full = ctx + " " + ending
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            ctx_enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=512)
            ctx_len = ctx_enc["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = 0.0
            n_ending = input_ids.shape[1] - ctx_len
            for i in range(ctx_len, input_ids.shape[1]):
                nll += log_probs[i - 1, input_ids[0, i]].item()
            scores.append(nll / max(n_ending, 1))
        pred = scores.index(max(scores))
        if pred == ex["label"]:
            correct += 1
    hs_acc = correct / len(examples) if examples else 0.0
    print(f"  HellaSwag = {hs_acc:.1%} ({correct}/{len(examples)})")

    # ── Throughput ───────────────────────────────────────────────────────────
    print("  Measuring decode throughput...")
    prompt = "The meaning of life is"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model.generate(**enc, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - enc["input_ids"].shape[1]
    tps = n_gen / elapsed
    print(f"  Throughput = {tps:.1f} tok/s ({n_gen} tokens in {elapsed:.2f}s)")

    return {
        "model": model_name, "label": cfg["label"],
        "method": "AutoGPTQ INT4",
        "ppl": ppl, "n_tokens": total_tokens,
        "hellaswag": hs_acc, "hs_correct": correct, "hs_total": len(examples),
        "tps": tps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace FP16 baseline (no quantization)
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=quant_image, gpu="A10G", cpu=8.0, memory=32768,
    timeout=14400, volumes={"/models": model_vol}, secrets=[hf_secret],
)
def run_fp16_baseline(model_name: str):
    """Run HuggingFace FP16 (no quantization) as the quality ceiling."""
    import torch, time, math
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg = MODELS[model_name]
    ensure_model(cfg["hf_id"], cfg["local_dir"])
    print(f"\n{'='*60}\nHF FP16 Baseline — {cfg['label']}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["local_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["local_dir"], device_map="auto", torch_dtype=torch.float16,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── WikiText-2 PPL ───────────────────────────────────────────────────────
    print("  Computing WikiText-2 perplexity (50 seq × 256 tokens)...")
    seqs = get_wikitext2_sequences(n_sequences=50, seq_len=256)
    total_nll = 0.0
    total_tokens = 0
    for seq in seqs:
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        n_tok = input_ids.shape[1] - 1
        total_nll += outputs.loss.item() * n_tok
        total_tokens += n_tok
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    print(f"  PPL = {ppl:.2f} ({total_tokens} tokens)")

    # ── HellaSwag ────────────────────────────────────────────────────────────
    print("  Computing HellaSwag accuracy (50 examples)...")
    examples = get_hellaswag_examples(n=50)
    correct = 0
    for ex in examples:
        ctx = ex["ctx"]
        scores = []
        for ending in ex["endings"]:
            full = ctx + " " + ending
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            ctx_enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=512)
            ctx_len = ctx_enc["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = 0.0
            n_ending = input_ids.shape[1] - ctx_len
            for i in range(ctx_len, input_ids.shape[1]):
                nll += log_probs[i - 1, input_ids[0, i]].item()
            scores.append(nll / max(n_ending, 1))
        pred = scores.index(max(scores))
        if pred == ex["label"]:
            correct += 1
    hs_acc = correct / len(examples) if examples else 0.0
    print(f"  HellaSwag = {hs_acc:.1%} ({correct}/{len(examples)})")

    # ── Throughput ───────────────────────────────────────────────────────────
    print("  Measuring decode throughput...")
    prompt = "The meaning of life is"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model.generate(**enc, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - enc["input_ids"].shape[1]
    tps = n_gen / elapsed
    print(f"  Throughput = {tps:.1f} tok/s ({n_gen} tokens in {elapsed:.2f}s)")

    return {
        "model": model_name, "label": cfg["label"],
        "method": "HF FP16",
        "ppl": ppl, "n_tokens": total_tokens,
        "hellaswag": hs_acc, "hs_correct": correct, "hs_total": len(examples),
        "tps": tps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(model: str = "llama1b,llama3b"):
    import json

    model_names = [m.strip() for m in model.split(",")]
    print(f"AWQ / GPTQ / NVE Comparison — {model_names}")
    print("=" * 70)

    # Launch all evaluations in parallel
    nve_handles  = {m: run_nve_eval.spawn(m)    for m in model_names}
    awq_handles  = {m: run_awq_eval.spawn(m)    for m in model_names}
    gptq_handles = {m: run_gptq_eval.spawn(m)   for m in model_names}
    fp16_handles = {m: run_fp16_baseline.spawn(m) for m in model_names}

    # Collect results
    nve_results  = {m: h.get() for m, h in nve_handles.items()}
    awq_results  = {m: h.get() for m, h in awq_handles.items()}
    gptq_results = {m: h.get() for m, h in gptq_handles.items()}
    fp16_results = {m: h.get() for m, h in fp16_handles.items()}

    all_results = {
        "nve":  nve_results,
        "awq":  awq_results,
        "gptq": gptq_results,
        "fp16": fp16_results,
    }

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Model':<15} {'PPL':>8} {'HellaSwag':>10} {'tok/s':>8}")
    print("-" * 70)

    for m in model_names:
        label = MODELS[m]["label"]

        # FP16 baseline
        r = fp16_results[m]
        print(f"{'HF FP16':<25} {label:<15} {r['ppl']:>8.2f} {r['hellaswag']:>9.1%} {r['tps']:>8.1f}")

        # NVE strategies
        nve = nve_results[m]
        for strat, data in nve["nve"].items():
            print(f"{strat:<25} {label:<15} {data['ppl']:>8.2f} {data['hellaswag']:>9.1%} {data['tps']:>8.1f}")

        # AWQ
        r = awq_results[m]
        print(f"{'AutoAWQ INT4':<25} {label:<15} {r['ppl']:>8.2f} {r['hellaswag']:>9.1%} {r['tps']:>8.1f}")

        # GPTQ
        r = gptq_results[m]
        print(f"{'AutoGPTQ INT4':<25} {label:<15} {r['ppl']:>8.2f} {r['hellaswag']:>9.1%} {r['tps']:>8.1f}")

        print()

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path("evidence/experiments/awq_gptq_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")
