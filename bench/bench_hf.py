#!/usr/bin/env python3
"""bench_hf.py -- measure prefill and decode throughput of the real
LFM2.5-2.6B checkpoint under Hugging Face transformers on CPU, using the same
shape of work as `illlm bench`:

  prefill : one batch of `fill` tokens through the full model
  decode  : `steps` single-token steps through the full model (KV cached)

Run:
  venv-hf\\Scripts\\python.exe bench\\bench_hf.py [--fill N] [--steps N] [--threads N]
"""

import argparse
import os
import time

import torch

# Repo root is one level above this script (bench/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep memory-mapped weight loading deterministic and quiet.
torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "LFM2.5-2.6B"))
    ap.add_argument("--fill", type=int, default=256)   # illlm bench default --batch
    ap.add_argument("--steps", type=int, default=64)   # illlm bench default --max-tokens
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quant", choices=["bf16", "int8"], default="bf16")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} threads {torch.get_num_threads()}")

    import transformers
    print(f"transformers {transformers.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    dtype = torch.bfloat16 if args.quant == "bf16" else torch.int8
    print(f"loading {args.model} as {args.quant} on cpu ...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    print(f"load {time.perf_counter() - t0:.2f}s", flush=True)

    vocab = model.config.vocab_size if hasattr(model.config, "vocab_size") else \
            tok.vocab_size
    # same deterministic pseudo-token scheme as app_main bench
    train = [int((i * 7919 + 13) % vocab) for i in range(args.fill)]
    nexts = [int((i * 104729 + 7) % vocab) for i in range(args.steps)]

    batch = torch.tensor([train], dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=batch, use_cache=True)
    fill_secs = time.perf_counter() - t0
    print(f"prefill  {args.fill} tok in {fill_secs:.3f} s -> "
          f"{args.fill / fill_secs:.1f} tok/s")

    past = out.past_key_values
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(args.steps):
            tok_id = torch.tensor([[nexts[i]]], dtype=torch.long)
            out = model(input_ids=tok_id, use_cache=True, past_key_values=past)
            past = out.past_key_values
    step_secs = time.perf_counter() - t0
    print(f"decode   {args.steps} tok in {step_secs:.3f} s -> "
          f"{args.steps / step_secs:.1f} tok/s")


if __name__ == "__main__":
    main()