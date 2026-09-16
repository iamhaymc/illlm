#!/usr/bin/env python3
"""test/test.py -- compares the C engine against the reference implementation.

The engine claims to run the Liquid architecture exactly as Hugging Face
transformers runs it.  This script is where that claim is checked.

Two modes:

  synthetic   the default.  Builds small Lfm2 checkpoints with random weights,
              one per architectural shape worth exercising, runs both
              implementations over the same token ids, and compares logits.
              No download is needed, so it runs anywhere transformers does.

  checkpoint  `--model PATH` points at a real checkpoint.  The default is the
               `ckpt/0.4b` folder at the repo root, which carries the published
              LFM2.5-350M weights; `ckpt/1.2b` and `ckpt/2.6b` hold the larger
              checkpoints.  Against it the suite adds a tokenizer agreement
              check, a throughput comparison -- the engine's `bench` beside
              transformers doing the same shape of work at the same weight
              width -- and a greedy continuation compared on the text, judged
              against how far the reference parts from itself.

              A missing checkpoint folder is a skip, not a failure: the
              synthetic suite is the parity argument and runs without it.

usage
  python3 test/test.py --binary ./build/app_main
  python3 test/test.py --binary ./build/app_main --model ckpt/2.6b
  python3 test/test.py --filter conv          # run a subset by name
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# The published checkpoints ship in the repository under ckpt/, so the
# checkpoint suite runs by default; pass --model to point somewhere else, or
# --no-checkpoint to skip it entirely.  The default is the smallest checkpoint,
# ckpt/0.4b (LFM2.5-350M); ckpt/1.2b and ckpt/2.6b hold the larger ones.
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "ckpt", "0.4b")

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

PASSED, FAILED, SKIPPED = [], [], []


def announce(name, note=""):
    tail = f"  ({note})" if note else ""
    print(f"  {name:<34} ...{tail}", end="", flush=True)


def record(name, ok, detail="", skip=False):
    if skip:
        SKIPPED.append(name)
        print(f" skip  {detail}")
    elif ok:
        PASSED.append(name)
        print(f" pass  {detail}")
    else:
        FAILED.append((name, detail))
        print(f" FAIL  {detail}")


def engine_rows(binary, model_dir, ids, vocab_size, *flags):
    import numpy as np

    sink = os.path.join(model_dir, "_logits.bin")
    cmd = [binary, "logits", "--model", model_dir,
           "--tokens", ",".join(str(i) for i in ids),
           "--out", sink, "--quiet", *flags]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{done.stdout}\n{done.stderr}")
    cells = np.fromfile(sink, dtype=np.float32).reshape(-1, vocab_size)
    os.unlink(sink)
    return cells


def load_kwargs(dtype):
    """from_pretrained took torch_dtype before transformers 5 and dtype after."""
    import transformers
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return {key: dtype}


def reference(model_dir, ids, dtype=None):
    """Runs transformers over the same ids and returns every position's logits."""
    import torch
    from transformers import Lfm2ForCausalLM

    model = Lfm2ForCausalLM.from_pretrained(model_dir,
                                            **load_kwargs(dtype or torch.float32)).eval()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False).logits
    return out[0].float().numpy()


# ---------------------------------------------------------------------------
# synthetic checkpoints
# ---------------------------------------------------------------------------

SHAPES = {
    "attention_only": dict(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
    ),
    "convolution_only": dict(
        num_hidden_layers=4,
        layer_types=["conv"] * 4,
    ),
    "hybrid_stack": dict(
        num_hidden_layers=6,
        layer_types=["conv", "full_attention", "conv", "conv", "full_attention", "conv"],
    ),
    "grouped_query": dict(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=64,
        layer_types=["full_attention", "conv", "full_attention", "conv"],
    ),
    "multi_query": dict(
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=64,
        layer_types=["full_attention", "conv", "full_attention"],
    ),
    "untied_head": dict(
        num_hidden_layers=4,
        tie_word_embeddings=False,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    ),
    "wide_kernel": dict(
        num_hidden_layers=4,
        conv_L_cache=5,
        layer_types=["conv"] * 4,
    ),
    "biased_kernel": dict(
        num_hidden_layers=4,
        conv_L_cache=4,
        conv_bias=True,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    ),
    "ragged_widths": dict(
        num_hidden_layers=4,
        vocab_size=131,
        hidden_size=48,
        intermediate_size=100,
        num_attention_heads=4,
        num_key_value_heads=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    ),
}

BASE = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    norm_eps=1e-5,
    conv_L_cache=3,
    conv_bias=False,
    tie_word_embeddings=True,
    block_auto_adjust_ff_dim=False,
    block_multiple_of=1,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)


def build_model(root, name, overrides, seed=0, dtype=None):
    """Writes a random Lfm2 checkpoint and returns its folder."""
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM

    settings = dict(BASE)
    settings.update(overrides)
    settings["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000.0}
    config = Lfm2Config(**settings)

    torch.manual_seed(seed)
    model = Lfm2ForCausalLM(config)
    # Random init leaves every norm gain at 1.0, which would hide a gain that
    # the engine loaded from the wrong tensor.  Give each one its own value.
    with torch.no_grad():
        for label, cell in model.named_parameters():
            if label.endswith("norm.weight") or label.endswith("layernorm.weight"):
                cell.copy_(torch.rand_like(cell) * 0.6 + 0.7)
    model = model.eval()
    if dtype is not None:
        model = model.to(dtype)

    folder = os.path.join(root, name)
    model.save_pretrained(folder, safe_serialization=True)
    return folder, settings["vocab_size"]


def token_run(vocab_size, count, seed=7):
    """Reproducible ids, skipping id 0: the reference zeroes the padding row,
    which drives the whole stack to zero and hides real differences."""
    cells, cell = [], seed
    for _ in range(count):
        cell = (cell * 1103515245 + 12345) & 0x7FFFFFFF
        cells.append(1 + cell % (vocab_size - 1))
    return cells


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------

def gap(a, b):
    import numpy as np
    scale = max(float(np.abs(a).max()), 1e-6)
    return float(np.abs(a - b).max()) / scale


def check_shapes(binary, root, filter_text):
    import numpy as np

    for name, overrides in SHAPES.items():
        label = f"shape/{name}"
        if filter_text and filter_text not in label:
            continue
        announce(label)
        folder, vocab_size = build_model(root, name, overrides)
        ids = token_run(vocab_size, 14)
        want = reference(folder, ids)
        got = engine_rows(binary, folder, ids, vocab_size, "--every")
        if got.shape != want.shape:
            record(label, False, f"shape {got.shape} vs {want.shape}")
            continue
        error = gap(want, got)
        record(label, error < 2e-5, f"max relative error {error:.2e}")
        np.save(os.path.join(folder, "_reference.npy"), want)


def check_streaming(binary, root, filter_text):
    label = "cache/streaming_decode"
    if filter_text and filter_text not in label:
        return
    announce(label, "prefill then one token at a time")
    folder, vocab_size = build_model(root, "stream", SHAPES["hybrid_stack"])
    ids = token_run(vocab_size, 24)
    want = reference(folder, ids)
    worst = 0.0
    for lead in (1, 2, 5, 24):
        got = engine_rows(binary, folder, ids, vocab_size, "--stream", "--prefill", str(lead))
        worst = max(worst, gap(want[lead - 1:], got))
    record(label, worst < 2e-5, f"max relative error {worst:.2e}")


def check_chunking(binary, root, filter_text):
    label = "cache/prefill_chunking"
    if filter_text and filter_text not in label:
        return
    announce(label, "batch widths 1, 3, 7, 64")
    folder, vocab_size = build_model(root, "chunk", SHAPES["hybrid_stack"])
    ids = token_run(vocab_size, 20)
    want = reference(folder, ids)
    worst = 0.0
    for width in (1, 3, 7, 64):
        got = engine_rows(binary, folder, ids, vocab_size, "--every", "--batch", str(width))
        worst = max(worst, gap(want, got))
    record(label, worst < 2e-5, f"max relative error {worst:.2e}")


def check_long_context(binary, root, filter_text):
    label = "cache/long_context"
    if filter_text and filter_text not in label:
        return
    announce(label, "300 positions")
    folder, vocab_size = build_model(root, "long", SHAPES["hybrid_stack"])
    ids = token_run(vocab_size, 300)
    want = reference(folder, ids)
    got = engine_rows(binary, folder, ids, vocab_size, "--every", "--ctx", "512")
    record(label, gap(want, got) < 5e-5, f"max relative error {gap(want, got):.2e}")


def check_storage(binary, root, filter_text):
    """bf16 and f16 storage must widen to f32 exactly, so the bound is tight."""
    import torch

    for name, dtype in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        label = f"storage/{name}"
        if filter_text and filter_text not in label:
            continue
        announce(label)
        folder, vocab_size = build_model(root, f"store_{name}", SHAPES["hybrid_stack"],
                                         dtype=dtype)
        ids = token_run(vocab_size, 16)
        want = reference(folder, ids)          # loaded back as f32
        got = engine_rows(binary, folder, ids, vocab_size, "--every")
        error = gap(want, got)
        record(label, error < 2e-5, f"max relative error {error:.2e}")


def check_quant(binary, root, filter_text):
    """q8 is lossy by design, so it is judged on agreement, not on distance."""
    import numpy as np

    label = "storage/q8_repack"
    if filter_text and filter_text not in label:
        return
    announce(label, "top-1 agreement and correlation")
    folder, vocab_size = build_model(root, "quant", SHAPES["hybrid_stack"])
    ids = token_run(vocab_size, 16)
    want = reference(folder, ids)
    got = engine_rows(binary, folder, ids, vocab_size, "--every", "--quant", "q8")
    agree = float((want.argmax(axis=1) == got.argmax(axis=1)).mean())
    a = want - want.mean(axis=1, keepdims=True)
    b = got - got.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    live = scale > 0
    tie = float(np.mean((a[live] * b[live]).sum(1) / scale[live])) if live.any() else 1.0
    record(label, agree >= 0.75 and tie > 0.99,
           f"top-1 {agree:.0%}, correlation {tie:.4f}")


def check_threads(binary, root, filter_text):
    label = "runtime/thread_agreement"
    if filter_text and filter_text not in label:
        return
    announce(label, "1 vs 4 workers")
    folder, vocab_size = build_model(root, "threads", SHAPES["grouped_query"])
    ids = token_run(vocab_size, 32)
    one = engine_rows(binary, folder, ids, vocab_size, "--every", "--threads", "1")
    many = engine_rows(binary, folder, ids, vocab_size, "--every", "--threads", "4")
    record(label, gap(one, many) < 1e-6, f"max relative error {gap(one, many):.2e}")


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

TRICKY = [
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "double  space",
    "tabs\tand\nnewlines\n\n",
    "1234567890 and 42 and 007",
    "don't can't we've I'll they're it's He'd",
    "DON'T CAN'T",
    "CamelCaseAndsnake_case",
    "punctuation!!!???...---",
    "unicode: café naïve Ω π 中文 日本語 한국어",
    "emoji: 🙂🚀 mixed with text",
    "math: 3+4=7, x<y, a>=b",
    "url https://example.com/path?a=1&b=2",
    "code: for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "   ",
    "\n",
    "a" * 200,
    "".join(chr(c) for c in range(32, 127)),
]


def build_tokenizer(folder, flavour):
    """Trains a small byte level BPE and writes tokenizer.json beside the model."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE())
    if flavour == "llama3":
        pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                   r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=__import__("tokenizers").Regex(pattern),
                                 behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    corpus = TRICKY + [
        "the quick brown fox jumps over the lazy dog",
        "Liquid foundation models interleave attention and convolution.",
        "def main(): return 0",
    ] * 6
    trainer = trainers.BpeTrainer(vocab_size=900, show_progress=False,
                                 special_tokens=["<|pad|>", "<|startoftext|>",
                                                 "<|endoftext|>", "<|im_start|>",
                                                 "<|im_end|>"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    tok.save(os.path.join(folder, "tokenizer.json"))
    return tok


def check_tokenizer(binary, root, filter_text):
    for flavour in ("gpt2", "llama3"):
        label = f"vocab/{flavour}_split"
        if filter_text and filter_text not in label:
            continue
        announce(label, f"{len(TRICKY)} strings")
        folder, _ = build_model(root, f"tok_{flavour}", SHAPES["hybrid_stack"])
        try:
            tok = build_tokenizer(folder, flavour)
        except Exception as why:                      # pragma: no cover
            record(label, False, f"tokenizers unavailable: {why}", skip=True)
            continue
        bad = []
        for text in TRICKY:
            want = tok.encode(text, add_special_tokens=False).ids
            done = subprocess.run([binary, "tokens", "--model", folder,
                                   "--prompt", text, "--quiet"],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                bad.append((text, "engine error", done.stderr.strip()))
                continue
            head = done.stdout.splitlines()[0].strip()
            got = [int(x) for x in head.split(",")] if head else []
            if got != want:
                bad.append((text, want, got))
        if bad:
            first = bad[0]
            record(label, False, f"{len(bad)}/{len(TRICKY)} differ, first {first[0]!r}: "
                                 f"want {first[1]} got {first[2]}")
        else:
            record(label, True, "all match")


def check_roundtrip(binary, root, filter_text):
    label = "vocab/decode_roundtrip"
    if filter_text and filter_text not in label:
        return
    announce(label)
    folder, _ = build_model(root, "tok_round", SHAPES["hybrid_stack"])
    try:
        build_tokenizer(folder, "gpt2")
    except Exception as why:                          # pragma: no cover
        record(label, False, str(why), skip=True)
        return
    bad = []
    for text in TRICKY:
        one = subprocess.run([binary, "tokens", "--model", folder, "--prompt", text, "--quiet"],
                             capture_output=True, text=True)
        ids = one.stdout.splitlines()[0].strip()
        if not ids:
            continue
        two = subprocess.run([binary, "tokens", "--model", folder, "--tokens", ids, "--quiet"],
                             capture_output=True, text=True)
        back = two.stdout[:-1] if two.stdout.endswith("\n") else two.stdout
        if back != text:
            bad.append((text, back))
    record(label, not bad, "all match" if not bad else f"{len(bad)} differ, first {bad[0]!r}")


# ---------------------------------------------------------------------------
# real checkpoint
# ---------------------------------------------------------------------------

def check_checkpoint(binary, folder, filter_text):
    import numpy as np

    label = "checkpoint/logits"
    if filter_text and filter_text not in label:
        return
    announce(label, folder)
    text = "The Liquid architecture interleaves attention with short convolutions because"
    done = subprocess.run([binary, "tokens", "--model", folder, "--prompt", text, "--quiet"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        record(label, False, done.stderr.strip())
        return
    ids = [int(x) for x in done.stdout.splitlines()[0].split(",")]

    with open(os.path.join(folder, "config.json")) as fh:
        vocab_size = json.load(fh)["vocab_size"]
    got = engine_rows(binary, folder, ids, vocab_size, "--every")
    want = reference(folder, ids)
    error = gap(want, got)
    agree = float((want.argmax(axis=1) == got.argmax(axis=1)).mean())
    record(label, error < 5e-4 and agree == 1.0,
           f"max relative error {error:.2e}, top-1 agreement {agree:.0%}")

    label = "checkpoint/tokenizer"
    if filter_text and filter_text not in label:
        return
    announce(label, f"{len(TRICKY)} strings")
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    except Exception as why:
        record(label, False, str(why), skip=True)
        return
    bad = []
    for probe in TRICKY:
        want_ids = tok.encode(probe, add_special_tokens=False).ids
        done = subprocess.run([binary, "tokens", "--model", folder, "--prompt", probe, "--quiet"],
                              capture_output=True, text=True)
        head = done.stdout.splitlines()[0].strip() if done.stdout else ""
        got_ids = [int(x) for x in head.split(",")] if head else []
        if got_ids != want_ids:
            bad.append((probe, want_ids, got_ids))
    if bad:
        record(label, False, f"{len(bad)}/{len(TRICKY)} differ, first {bad[0][0]!r}: "
                             f"want {bad[0][1]} got {bad[0][2]}")
    else:
        record(label, True, "all match")
    _ = np


# ---------------------------------------------------------------------------
# throughput and behaviour on a real checkpoint
# ---------------------------------------------------------------------------

# The engine is expected to be at least as fast as the reference doing the same
# shape of work at the same weight width.  The comparison is like for like:
# both sides run bf16, the same thread count, the same fill and step counts.
# Quoting the engine's q8 against the reference's bf16 would fold a weight
# width difference into what looks like an engine difference.
FILL_TOKENS = 256
STEP_TOKENS = 64


def engine_bench(binary, folder, threads):
    """Runs the engine's own bench verb and reads its rates back."""
    cmd = [binary, "bench", "--model", folder, "--batch", str(FILL_TOKENS),
           "--max-tokens", str(STEP_TOKENS), "--threads", str(threads), "--quiet"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{done.stdout}\n{done.stderr}")
    fill = step = 0.0
    for line in done.stdout.splitlines():
        hit = re.search(r"prefill\s+\d+ tok in [\d.]+ s -> ([\d.]+) tok/s", line)
        if hit:
            fill = float(hit.group(1))
        hit = re.search(r"decode\s+\d+ tok in [\d.]+ s -> ([\d.]+) tok/s", line)
        if hit:
            step = float(hit.group(1))
    return fill, step


def reference_bench(folder, threads):
    """Prefill and decode throughput of transformers on CPU, same shape of work.

    One batch of FILL_TOKENS through the full model, then STEP_TOKENS
    single-token steps with the key/value cache carried.  bf16, because that is
    what the engine's as-stored path runs and what the checkpoint ships.
    """
    import torch
    from transformers import Lfm2ForCausalLM

    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    model = Lfm2ForCausalLM.from_pretrained(folder,
                                            **load_kwargs(torch.bfloat16),
                                            local_files_only=True).eval()

    ids = torch.arange(1, FILL_TOKENS + 1).unsqueeze(0)
    mark = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    fill = FILL_TOKENS / max(time.perf_counter() - mark, 1e-9)

    past = out.past_key_values
    mark = time.perf_counter()
    for step in range(STEP_TOKENS):
        out = model(input_ids=torch.tensor([[step % 1000 + 1]]),
                    use_cache=True, past_key_values=past)
        past = out.past_key_values
    step = STEP_TOKENS / max(time.perf_counter() - mark, 1e-9)
    del model, past
    return fill, step


def check_throughput(binary, folder, filter_text):
    label = "checkpoint/throughput"
    if filter_text and filter_text not in label:
        return
    announce(label, f"fill {FILL_TOKENS}, steps {STEP_TOKENS}, same width and threads")
    threads = min(4, os.cpu_count() or 1)
    mine_fill, mine_step = engine_bench(binary, folder, threads)
    their_fill, their_step = reference_bench(folder, threads)
    ok = mine_step >= their_step and mine_fill >= their_fill
    record(label, ok,
           f"engine prefill {mine_fill:.1f} tok/s, decode {mine_step:.1f} tok/s; "
           f"reference prefill {their_fill:.1f}, decode {their_step:.1f}")


def engine_generate(binary, folder, ids, steps, draft=0):
    """Greedy continuation from raw ids, returned as text."""
    cmd = [binary, "generate", "--model", folder, "--raw",
           "--tokens", ",".join(str(i) for i in ids),
           "--max-tokens", str(steps), "--temp", "0", "--quiet"]
    if draft:
        cmd += ["--draft", str(draft)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{done.stdout}\n{done.stderr}")
    return done.stdout


def reference_generate(folder, ids, steps, split=False):
    """Greedy continuation from the same ids under transformers.

    `split` primes the prompt one token at a time instead of one forward over
    the whole prompt, which is the reference's own second summation order: the
    two runs agree until float addition lands differently, and where they part
    is how far agreement can be asked for.
    """
    import torch
    from transformers import Lfm2ForCausalLM

    model = Lfm2ForCausalLM.from_pretrained(folder,
                                            **load_kwargs(torch.bfloat16),
                                            local_files_only=True).eval()
    with torch.no_grad():
        if split:
            past = None
            for lead in range(len(ids)):
                out = model(input_ids=torch.tensor([[ids[lead]]]),
                            use_cache=True, past_key_values=past)
                past = out.past_key_values
            fresh = out.logits[0, -1].argmax().item()
            fresh_list = [fresh]
            for _ in range(steps - 1):
                out = model(input_ids=torch.tensor([[fresh]]),
                            use_cache=True, past_key_values=past)
                past = out.past_key_values
                fresh = out.logits[0, -1].argmax().item()
                fresh_list.append(fresh)
            del model, past
            return fresh_list
        out = model.generate(input_ids=torch.tensor([ids]),
                             max_new_tokens=steps, do_sample=False, use_cache=True)
    del model
    return out[0].tolist()


def shared_prefix(left, right):
    """How many leading tokens two id sequences have in common."""
    span = 0
    for a, b in zip(left, right):
        if a != b:
            break
        span += 1
    return span


def check_greedy(binary, folder, filter_text):
    """Greedy continuations, compared on the shared prefix of decoded text.

    The bar is measured, not assumed: the reference is run against itself,
    primed one token at a time against one forward over the whole prompt, and
    where those two part is how far agreement can be asked for.  The engine is
    allowed to follow half as far as the reference follows itself before it is
    called wrong.  Whether a rounding lands on a half step is a lottery, so
    what the reference draws on one prompt is an estimate and not a bound.
    """
    label = "checkpoint/greedy"
    if filter_text and filter_text not in label:
        return
    announce(label, "greedy continuations, shared prefix against the reference's own")
    text = "The Liquid architecture interleaves attention with short convolutions because"
    done = subprocess.run([binary, "tokens", "--model", folder, "--prompt", text, "--quiet"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        record(label, False, done.stderr.strip())
        return
    ids = [int(x) for x in done.stdout.splitlines()[0].split(",")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)

    def decode(these):
        return tok.decode(these, skip_special_tokens=True).strip()

    mine = decode(reference_generate(folder, ids, STEP_TOKENS))
    theirs = decode(reference_generate(folder, ids, STEP_TOKENS, split=True))
    my_text = engine_generate(binary, folder, ids, STEP_TOKENS).strip()

    my_share = shared_prefix(my_text, theirs)
    their_share = shared_prefix(mine, theirs)
    ok = my_share * 2.0 >= their_share
    record(label, ok,
           f"engine follows for {my_share} characters, the reference itself for "
           f"{their_share}: {my_text[:60]!r}")


def check_draft(binary, folder, filter_text):
    """Drafting must not change a single character of what greedy emits.

    `--draft` verifies proposed tokens inside one forward pass and returns the
    state to a mark whenever a proposal is rejected, so it exercises
    `ill_state_mark`, `ill_state_back` and the convolution window they copy.
    Every token it emits is still the one the model's own row chose, so the
    check is exact equality against the same run without it, not a tolerance.

    It runs against the checkpoint rather than a synthetic model on purpose.
    A model with random weights tends to settle on one token, which a draft
    proposes and the model then accepts every time -- the rewind is never
    reached and the check passes whatever the rewind does. Against a trained
    checkpoint, proposals are accepted and rejected in turn, and breaking the
    window restore parts the two runs.

    The prompt asks for something repeated so the scan has matches to find, and
    two widths are run so both the carry and the commit path are reached.
    """
    label = "checkpoint/draft"
    if filter_text and filter_text not in label:
        return
    announce(label, "--draft 4 and 8 against plain greedy, exact equality")
    text = ("List the first eight prime numbers, then list them again in reverse "
            "order, then explain what a prime number is.")

    def emit(*extra):
        cmd = [binary, "generate", "--model", folder, "--prompt", text,
               "--max-tokens", str(STEP_TOKENS), "--temp", "0", "--quiet", *extra]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)}\n{done.stderr}")
        return done.stdout

    plain = emit()
    detail = []
    for width in (4, 8):
        drafted = emit("--draft", str(width))
        if drafted != plain:
            detail.append(f"draft {width} parts at character "
                          f"{shared_prefix(drafted, plain)} of {len(plain)}")
    record(label, not detail, "identical" if not detail else "; ".join(detail))


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="./build/app_main",
                        help="path to the compiled CLI (default ./build/app_main)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="checkpoint folder to test against (default ./ckpt/0.4b)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="skip the checkpoint suite even when ./ckpt/0.4b exists")
    parser.add_argument("--filter", default=None, help="only run checks whose name contains this")
    parser.add_argument("--keep", action="store_true", help="keep the synthetic checkpoints")
    args = parser.parse_args()

    # The corpus and the continuations carry text outside any Windows console
    # code page.  Report what cannot be encoded rather than dying mid-run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}\nbuild it first: python3 util/make.py build")
        return 2
    try:
        import torch, transformers            # noqa: F401
    except ImportError as why:
        print(f"reference stack missing: {why}\ninstall it: python3 util/make.py install")
        return 2

    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass

    root = tempfile.mkdtemp(prefix="illlm-test-")
    print(f"reference: transformers {__import__('transformers').__version__}, "
          f"torch {__import__('torch').__version__}")
    print(f"scratch:   {root}\n")
    try:
        print("architecture")
        check_shapes(args.binary, root, args.filter)
        print("\nstate")
        check_streaming(args.binary, root, args.filter)
        check_chunking(args.binary, root, args.filter)
        check_long_context(args.binary, root, args.filter)
        print("\nweights")
        check_storage(args.binary, root, args.filter)
        check_quant(args.binary, root, args.filter)
        check_threads(args.binary, root, args.filter)
        print("\nvocabulary")
        check_tokenizer(args.binary, root, args.filter)
        check_roundtrip(args.binary, root, args.filter)
        if args.no_checkpoint:
            print("\ncheckpoint  skipped (--no-checkpoint)")
        elif not os.path.isdir(args.model):
            print(f"\ncheckpoint  skipped: {args.model} not found "
                  f"(pass --model or --no-checkpoint)")
        else:
            print(f"\ncheckpoint  {args.model}")
            check_checkpoint(args.binary, args.model, args.filter)
            check_throughput(args.binary, args.model, args.filter)
            check_greedy(args.binary, args.model, args.filter)
            check_draft(args.binary, args.model, args.filter)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"\nkept: {root}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    for name, detail in FAILED:
        print(f"  FAIL {name}: {detail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
