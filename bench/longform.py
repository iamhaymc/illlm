#!/usr/bin/env python3
"""longform.py -- generate a fixed-length continuation with both engines under
identical greedy settings and compare the decoded text.

illlm :  greedy (temp 0) raw continuation, max-tokens N
HF    :  greedy (do_sample=False) with the same ids and max_new_tokens N

The comparison is on text.  `generate` emits decoded text rather than ids, and
re-encoding that text to recover ids is not a round trip -- the tokenizer can
segment the same string differently -- so re-encoding would report divergence
the engines never produced.  Token level agreement is what
`python3 run.py test --model DIR` measures directly from logits.

Usage:
  python3 bench/longform.py [--model DIR] [--prompt TEXT] [--steps N]
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def illlm_generate(model_dir, prompt_ids, steps):
    main = common.require_binary(common.binary())
    cmd = [main, "generate", "--model", model_dir, "--raw",
           "--tokens", ",".join(map(str, prompt_ids)),
           "--max-tokens", str(steps), "--temp", "0", "--quiet"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"illlm generate failed ({done.returncode}):\n{done.stderr}")
    return done.stdout


def hf_generate(model_dir, prompt_ids, steps):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True).eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=steps,
                             do_sample=False, use_cache=True)
    return out[0].tolist()


def squeeze(text):
    return re.sub(r"\s+", " ", text).strip()


def shared_prefix(a, b):
    span = 0
    for left, right in zip(a, b):
        if left != right:
            break
        span += 1
    return span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The Industrial Revolution began in")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    model_dir = common.require_model(args.model)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_ids = tok.encode(args.prompt)
    print(f"prompt ({len(prompt_ids)} tokens): {args.prompt}")
    print(f"ids: {prompt_ids}")

    print("illlm greedy ...", flush=True)
    ill_text = illlm_generate(model_dir, prompt_ids, args.steps)
    print("hf greedy ...", flush=True)
    hf_ids = hf_generate(model_dir, prompt_ids, args.steps)
    hf_text = tok.decode(hf_ids, skip_special_tokens=True)

    ill_clean, hf_clean = squeeze(ill_text), squeeze(hf_text)
    print(f"\n[illlm]\n{ill_clean}\n")
    print(f"[hf]\n{hf_clean}")

    # Compare the generated tails, with the prompt trimmed off either side.
    lead = squeeze(tok.decode(prompt_ids, skip_special_tokens=True))
    ill_tail = squeeze(ill_clean[len(lead):]) if ill_clean.startswith(lead) else ill_clean
    hf_tail = squeeze(hf_clean[len(lead):]) if hf_clean.startswith(lead) else hf_clean

    span = shared_prefix(ill_tail, hf_tail)
    print(f"\nhf generated tokens:   {len(hf_ids) - len(prompt_ids)}")
    print(f"identical text prefix: {span} of {min(len(ill_tail), len(hf_tail))} chars"
          f" (illlm {len(ill_tail)}, hf {len(hf_tail)})")
    if span < min(len(ill_tail), len(hf_tail)):
        print(f"first divergence at char {span}:")
        print(f"  illlm ...{ill_tail[max(0, span - 30):span + 30]!r}")
        print(f"  hf    ...{hf_tail[max(0, span - 30):span + 30]!r}")
    elif len(ill_tail) == len(hf_tail):
        print("continuations are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
