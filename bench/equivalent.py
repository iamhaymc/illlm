#!/usr/bin/env python3
"""equivalent.py -- compare illlm vs Hugging Face logits on the same tokens.

Feeds the exact same token ids through both engines and compares the f32
logits at every position:

  illlm:  app_main logits --tokens a,b,c --every --out out.bin  (raw f32 rows)
  HF:     transformers forward with the same ids -> logits[0]

This is a bench-side probe over one real checkpoint, kept next to the
throughput scripts.  The authoritative equivalence check is the project's own
harness, which runs the same comparison over every architectural shape plus
the tokenizer:

  python3 run.py test --model path/to/LFM2.5-2.6B

Usage:
  python3 bench/equivalent.py [--model DIR] [--prompt TEXT | --tokens a,b,c]
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def illlm_logits(model_dir, tokens, vocab, quant="none"):
    """Returns every position's logits as a (rows, vocab) float32 array."""
    import numpy as np
    main = common.require_binary(common.binary())
    args = [main, "logits", "--model", model_dir, "--every",
            "--tokens", ",".join(map(str, tokens))]
    if quant == "q8":
        args += ["--quant", "q8"]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "logits.bin")
        subprocess.run(args + ["--out", out], check=True, stdout=subprocess.DEVNULL)
        raw = open(out, "rb").read()
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size % vocab:
        raise SystemExit(f"logits size {arr.size} is not a multiple of vocab {vocab}")
    return arr.reshape(-1, vocab)


def hf_logits(model_dir, tokens):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True).eval()
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return out.logits[0].float().numpy()


def relative_error(want, got):
    """Max absolute deviation, scaled by the reference's magnitude."""
    import numpy as np
    scale = max(float(np.abs(want).max()), 1e-6)
    return float(np.abs(want - got).max()) / scale


def report(name, want, got, tops):
    import numpy as np
    error = relative_error(want, got)
    agree = float((want.argmax(axis=1) == got.argmax(axis=1)).mean())
    corr = float(np.corrcoef(want.ravel(), got.ravel())[0, 1])
    top = np.argsort(got[-1])[::-1][:tops].tolist()
    print(f"{name:>10}: max relative error {error:.2e}, top-1 agreement "
          f"{agree:.0%}, correlation {corr:.6f}")
    print(f"{'':>10}  top-{tops} at last position: {top}")
    return error, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--tokens", default=None,
                    help="comma list of token ids; default: tokenize --prompt")
    ap.add_argument("--prompt",
                    default="The capital city of France is Paris and the Eiffel Tower")
    ap.add_argument("--tops", type=int, default=8)
    args = ap.parse_args()

    model_dir = common.require_model(args.model)
    common.require_torch()
    import numpy as np

    vocab = common.vocab_size(model_dir)

    if args.tokens:
        tokens = [int(t.strip()) for t in args.tokens.split(",")]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        tokens = tok.encode(args.prompt)
        print(f"text:  {args.prompt}")
    print(f"tokens ({len(tokens)}): {tokens}")
    print(f"vocab: {vocab}\n")

    print("illlm logits ...", flush=True)
    ill = illlm_logits(model_dir, tokens, vocab)
    print("illlm q8 logits ...", flush=True)
    ill8 = illlm_logits(model_dir, tokens, vocab, quant="q8")
    print("hf logits ...", flush=True)
    hf = hf_logits(model_dir, tokens)

    if hf.shape != ill.shape:
        raise SystemExit(f"shape mismatch: illlm {ill.shape} vs hf {hf.shape}")

    print()
    error, agree = report("illlm", hf, ill, args.tops)
    report("illlm q8", hf, ill8, args.tops)
    print(f"{'hf':>10}: top-{args.tops} at last position: "
          f"{np.argsort(hf[-1])[::-1][: args.tops].tolist()}")

    # The unquantised engine is the one that has to match; q8 is allowed to
    # drift, which is why only this pair gates the exit code.
    ok = error < 5e-4 and agree == 1.0
    print(f"\nilllm vs hf: {'PASS' if ok else 'FAIL'} "
          f"(max relative error < 5e-4 and full top-1 agreement)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
