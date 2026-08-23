#!/usr/bin/env python3
"""equivalent.py -- compare illlm vs Hugging Face logits on the same tokens.

The strongest end-to-end equivalence check: feed the exact same token ids
through both engines and compare the f32 logits of the last position.

  illlm:  app_main logits --tokens a,b,c --out out.bin  (raw f32, last row)
  HF:     transformers forward with the same ids -> logits[0, -1]

Usage:
  .venv-hf\\Scripts\\python.exe equivalent.py [--tokens a,b,c] [--tops N]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "LFM2.5-2.6B")
MAIN = os.path.join(ROOT, "build", "app_main.exe")


def illml_logits(tokens, quant="none"):
    """Return (rows x vocab float32, last-row logits)."""
    args = [MAIN, "logits", "--model", MODEL, "--tokens", ",".join(map(str, tokens))]
    if quant == "q8":
        args += ["--quant", "q8"]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "logits.bin")
        subprocess.run(args + ["--out", out], check=True,
                       stdout=subprocess.DEVNULL)
        raw = open(out, "rb").read()
    n = len(tokens)
    arr = np.frombuffer(raw, dtype=np.float32)
    rows = arr.size // 128000
    arr = arr.reshape(rows, 128000)
    return arr, arr[-1].copy()


def hf_logits(tokens):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 local_files_only=True).eval()
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.logits[0, -1].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=None,
                    help="comma list of token ids; default: tokenize a prompt")
    ap.add_argument("--prompt", default="The capital city of France is Paris and the Eiffel Tower")
    ap.add_argument("--tops", type=int, default=8)
    args = ap.parse_args()

    if args.tokens:
        tokens = [int(t.strip()) for t in args.tokens.split(",")]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        tokens = tok.encode(args.prompt)
    print(f"tokens ({len(tokens)}): {tokens}")
    print(f"text:  {args.prompt}")

    print("illlm logits ...", flush=True)
    _, ill = illml_logits(tokens)
    print("illlm q8 logits ...", flush=True)
    _, ill8 = illml_logits(tokens, quant="q8")
    print("hf logits ...", flush=True)
    hf = hf_logits(tokens)

    for name, vec in (("illlm", ill), ("illlm q8", ill8)):
        top = np.argsort(vec)[::-1][: args.tops]
        print(f"\ntop-{args.tops} {name}: {top.tolist()}")
    hf_top = np.argsort(hf)[::-1][: args.tops]
    print(f"top-{args.tops} hf:    {hf_top.tolist()}")

    def corr(a, b):
        a = (a - a.min()) / (a.max() - a.min() + 1e-12)
        b = (b - b.min()) / (b.max() - b.min() + 1e-12)
        return np.corrcoef(a, b)[0, 1]

    print(f"\ncorrelation illlm v hf:   {corr(ill, hf):.4f}")
    print(f"correlation illlm q8 v hf: {corr(ill8, hf):.4f}")
    print(f"correlation illlm v q8:    {corr(ill, ill8):.4f}")
    print(f"\ntop-1 greedy: illlm={int(ill.argmax())} q8={int(ill8.argmax())} "
          f"hf={int(hf.argmax())} "
          f"match={bool(ill8.argmax() == hf.argmax())}")


if __name__ == "__main__":
    main()