#!/usr/bin/env python3
"""bench_hf.py -- measure prefill and decode throughput of the real
LFM2.5-2.6B checkpoint under Hugging Face transformers on CPU, using the same
shape of work as `illlm bench`:

  prefill : one batch of `fill` tokens through the full model
  decode  : `steps` single-token steps through the full model (KV cached)

For a like-for-like number, run both sides at the same weight width and the
same thread count.  transformers has no q8 CPU path here, so the honest
comparison against `--quant q8` on the illlm side is:

  python3 run.py bench --model DIR --threads 8            # illlm, bf16
  python3 bench/bench_hf.py --threads 8 --quant bf16      # hf,    bf16

Quoting illlm-q8 against hf-bf16 folds a weight width difference into what
looks like an engine difference.

Run:
  python3 bench/bench_hf.py [--fill N] [--steps N] [--threads N] [--repeat N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DTYPES = ("bf16", "f32")


def time_prefill(model, ids):
    import torch
    batch = torch.tensor([ids], dtype=torch.long)
    mark = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=batch, use_cache=True)
    return time.perf_counter() - mark, out


def time_decode(model, past, nexts):
    import torch
    mark = time.perf_counter()
    with torch.no_grad():
        for step in nexts:
            out = model(input_ids=torch.tensor([[step]], dtype=torch.long),
                        use_cache=True, past_key_values=past)
            past = out.past_key_values
    return time.perf_counter() - mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--fill", type=int, default=256)   # illlm bench default --batch
    ap.add_argument("--steps", type=int, default=64)   # illlm bench default --max-tokens
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quant", choices=DTYPES, default="bf16")
    ap.add_argument("--repeat", type=int, default=3,
                    help="timed runs after the warmup; the best is reported")
    args = ap.parse_args()

    model_dir = common.require_model(args.model)

    torch = common.require_torch()
    torch.set_grad_enabled(False)
    width = {"bf16": torch.bfloat16, "f32": torch.float32}[args.quant]

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__} threads {torch.get_num_threads()}")

    import transformers
    print(f"transformers {transformers.__version__}")
    from transformers import AutoModelForCausalLM

    print(f"loading {model_dir} as {args.quant} on cpu ...", flush=True)
    mark = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=width,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"load {time.perf_counter() - mark:.2f}s", flush=True)

    vocab = common.vocab_size(model_dir)
    # same deterministic pseudo-token scheme as app_main bench
    train = [int((i * 7919 + 13) % vocab) for i in range(args.fill)]
    nexts = [int((i * 104729 + 7) % vocab) for i in range(args.steps)]

    # One untimed pass first: the first forward pays for lazy kernel dispatch
    # and thread pool spin-up, which would otherwise land on the prefill number.
    print("warmup ...", flush=True)
    _, out = time_prefill(model, train[: min(8, args.fill)])
    time_decode(model, out.past_key_values, nexts[:2])

    fill_best, step_best = float("inf"), float("inf")
    for run in range(max(1, args.repeat)):
        fill_secs, out = time_prefill(model, train)
        step_secs = time_decode(model, out.past_key_values, nexts)
        fill_best = min(fill_best, fill_secs)
        step_best = min(step_best, step_secs)
        print(f"  run {run + 1}: prefill {args.fill / fill_secs:6.1f} tok/s   "
              f"decode {args.steps / step_secs:5.1f} tok/s", flush=True)

    print(f"\nprefill  {args.fill} tok in {fill_best:.3f} s -> "
          f"{args.fill / fill_best:.1f} tok/s   ({args.quant}, "
          f"{torch.get_num_threads()} threads)")
    print(f"decode   {args.steps} tok in {step_best:.3f} s -> "
          f"{args.steps / step_best:.1f} tok/s   ({args.quant}, "
          f"{torch.get_num_threads()} threads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
