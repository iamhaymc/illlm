#!/usr/bin/env python3
"""longform.py -- generate a fixed-length continuation with both engines under
identical greedy settings and compare the decoded text.

illlm :  greedy (temp 0) raw continuation, max-tokens N
HF    :  greedy (do_sample=False) with the same ids and max_new_tokens N
"""

import argparse
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "LFM2.5-2.6B")
MAIN = os.path.join(ROOT, "build", "app_main.exe")


def illlm_generate(prompt_ids, steps):
    cmd = [MAIN, "generate", "--model", MODEL, "--raw",
           "--tokens", ",".join(map(str, prompt_ids)),
           "--max-tokens", str(steps), "--temp", "0", "--quiet"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def hf_generate(prompt_ids, steps):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=steps,
                             do_sample=False, use_cache=True)
    return out[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The Industrial Revolution began in")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompt_ids = tok.encode(args.prompt)
    print(f"prompt ({len(prompt_ids)} tokens): {args.prompt}")
    print(f"ids: {prompt_ids}")

    print("illlm greedy ...", flush=True)
    ill_text = illlm_generate(prompt_ids, args.steps)
    print("hf greedy ...", flush=True)
    hf_ids = hf_generate(prompt_ids, args.steps)
    hf_text = tok.decode(hf_ids, skip_special_tokens=True)

    def strip(text):
        return re.sub(r"\s+", " ", text).strip()

    ill_clean, hf_clean = strip(ill_text), strip(hf_text)
    print(f"\n[illlm]\n{ill_clean}\n")
    print(f"[hf]\n{hf_clean}")

    # token agreement on the *generated* portion (decode the prompt, cut it off)
    prefix = tok.decode(prompt_ids, skip_special_tokens=True)
    p = strip(prefix)
    ill_gen = ill_clean[len(p):] if ill_clean.startswith(p) else ill_clean
    hf_gen = strip(hf_clean[len(p):]) if hf_clean.startswith(p) else hf_clean
    ill_enc = tok.encode(ill_gen if ill_gen else ill_clean)
    hf_enc = tok.encode(hf_gen if hf_gen else hf_clean)
    common = 0
    for a, b in zip(ill_enc, hf_enc):
        if a == b:
            common += 1
        else:
            break
    print(f"\nhf generated tokens: {len(hf_enc)}")
    print(f"illlm generated tokens: {len(ill_enc)}")
    print(f"identical token prefix: {common}")
    if common < min(len(ill_enc), len(hf_enc)):
        print(f"first divergent: illlm={ill_enc[common]} hf={hf_enc[common]}")


if __name__ == "__main__":
    main()