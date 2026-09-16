# CHANGES

Development progress and the reasoning behind it, newest first.

---

## unreleased — MSVC support and checkpoint verification

Verified the engine against the published `LiquidAI/LFM2.5-2.6B` checkpoint and
got it building under MSVC on Windows.

### Layout

The flat file list became directories: the engine and its command line moved to
`app/core.c` and `app/main.c`, the two test suites to `test/test.c` and
`test/test.py`, and the Python workflows to `util/make.py` (was `run.py`) and
`util/tune.py` (was `app_tune.py`). The tuning corpus moved to
`data/tune.jsonl` and the published checkpoints to `ckpt/lfm2.5-0.4b`, `ckpt/lfm2.5-1.2b-i` and
`ckpt/lfm2.5-2.6b`. The engine sources are renamed, not split: `app/main.c` and
`test/test.c` still include the engine whole, so each front end remains one
translation unit. The binaries keep their names — `build/app_main`,
`build/app_test` — and the workflows keep their verbs, so the command line is
`python3 util/make.py build` where `python3 run.py build` used to be.

No engine change: the same 73 unit checks pass, and the info, generate and
bench paths resolve against `ckpt/lfm2.5-0.4b` as they did against `models/lfm2.5-0.4b`.

### Engine

- **MSVC builds.** Three fixes in `app_core.c`. A local named `near` collided
  with the Win32 macro of that name and is now `snap` — `round` was tried
  first, but that shadows `<math.h>` and warns under `/W4`. The AVX2 vector
  branch now accepts MSVC, which never defines `__FMA__` even though
  `/arch:AVX2` provides FMA; the same reasoning applies to `__F16C__`, so the
  half precision load takes the intrinsic there too instead of a scalar loop.
- **An atomic shim for MSVC**, backed by the interlocked intrinsics, so the
  Win32 thread pool engages rather than falling back to single threaded.
  MSVC reports `__STDC_NO_ATOMICS__` and ships no working C11 `<stdatomic.h>`.

  The shim is deliberately scoped to x86 and x64. Its loads are plain volatile
  reads, which carry acquire ordering on those architectures once the compiler
  is stopped from sinking them, but not on arm64, where MSVC defaults to
  `/volatile:iso`; there a worker could read `pool->chore` before observing the
  epoch that published it. arm64 keeps the single threaded fallback until those
  loads grow real barriers. clang-cl defines `_MSC_VER` too but has a working
  `<stdatomic.h>`, so it skips the shim.

### `bench/`

Scripts for measuring and comparing against the real checkpoint, kept out of
the test harness because they need a multi gigabyte download.

- `download_model.py` — pull the checkpoint from Hugging Face.
- `make_index.py` — write a safetensors index for index-less shards.
- `bench_hf.py` — prefill and decode throughput under `transformers` on CPU.
- `equivalent.py` — logits comparison against `transformers`.
- `longform.py` — greedy continuation comparison.
- `common.py` — path handling shared by the above.

`app_test.py` remains the authoritative equivalence check: it covers every
architectural shape plus the tokenizer, and `run.py test --model DIR` points it
at a real checkpoint. `equivalent.py` is a bench-side probe over one checkpoint
and reports the same pair of metrics — max relative error and top-1 agreement —
so the two agree on what "equivalent" means.

### Results

- 73/73 unit checks pass with threads enabled under MSVC 19.42 on x64.
- q8 holds 2.83 GiB resident against the checkpoint's 30 layers (8 attention,
  22 convolution).

---

## 1.0.0 — initial engine

A complete inference engine for the Liquid architecture (`model_type: "lfm2"`),
in pure C11 with no third-party dependencies, reading Hugging Face checkpoints
directly.

### What was built

- **`app_core.c`** — the engine, in fifteen layered parts: platform, number
  formats, vector vocabulary, block quantisation, JSON, safetensors, kernels,
  backend seam, model load, state, forward pass, tokenizer, sampler, facade.
- **`app_main.c`** — six commands: `info`, `tokens`, `generate`, `chat`,
  `logits`, `bench`.
- **`app_test.c`** — 73 unit checks over the internals.
- **`app_test.py`** — 19 comparisons against Hugging Face `transformers`.
- **`run.py`** — install, build, test, run, bench, clean.

### Establishing the architecture

`huggingface.co` was not reachable from the development environment, so the
architecture was not guessed. The authoritative
`transformers/models/lfm2/modeling_lfm2.py` and `configuration_lfm2.py` were
read from the `transformers` 5.15.1 wheel on PyPI, and the engine was written
against them. Four details that a reading of the paper alone would miss, and
that each showed up in the reference source:

- Query and key heads are **RMS normalised before the rotation**, with a gain
  of `head_dim` values shared across heads.
- The attention output projection is named `out_proj`, not `o_proj`.
- The short convolution's `in_proj` splits into `B, C, x` **in that order**,
  and the gating is `out_proj(C * conv(B * x))`.
- The feed forward width is `block_auto_adjust_ff_dim` applied to
  `intermediate_size`, so the value in `config.json` is not the width of `w1`.

The engine reads the width from the shape of `w1` instead of recomputing the
rule, so a checkpoint that disagrees with the formula still loads.

### Decisions and why

**Read the checkpoint directly; do not require a conversion step.**
`config.json` plus `*.safetensors` plus `tokenizer.json` is what a user
downloads. Requiring a conversion pass would add a second file format, a second
tool, and a class of "which one is stale" bugs. Weights are memory mapped and
used in place, which is also why loading 5.4 GiB of bf16 costs about a tenth of
a second: nothing is copied.

**Weights in their stored format, activations in f32.**
bf16 widens to f32 with a shift, which is nearly free and exactly matches what
the reference computes when it upcasts. This gives bit-level agreement with the
reference on f32 checkpoints without a conversion pass or a precision loss.

**q8 as an option, not a default.**
`--quant q8` repacks the large planes at load: 3.06 GiB instead of 5.44 GiB, and
decode goes from 5.2 to 9.0 tok/s because decode is bandwidth bound. It is
lossy, so it is opt-in and the test suite judges it on top-1 agreement and
correlation rather than on distance.

**One vector vocabulary, four widths.**
The alternative — separate scalar and SIMD kernels — means two implementations
that can drift, and a scalar "reference" that is not actually the thing being
tested. Here the kernels are written once against a dozen inline operations that
are instantiated at width 16, 8, 4, and 1. The width-1 build is the same source.
`run.py test --no-simd` and `run.py test --portable` run the whole suite at
widths 1 and 8, so agreement between widths is a test result rather than an
assumption.

**A lock-free thread pool.**
A decode step issues a few hundred parallel regions. Measured against a
condition-variable pool, the epoch-and-claim design removes a handshake that was
a visible fraction of the step. Chunks are claimed with an atomic
fetch-and-add so uneven work self balances. With no C11 atomics the same code
runs inline on one lane.

**A backend table, drawn at six operations.**
`dense`, `rmsnorm`, `swiglu`, `rope`, `attend`, `conv1d` are the only arithmetic
the stack performs, and each is large enough that a function-pointer call is
free. Drawing the seam here means an accelerator implements six kernels and
changes nothing above them. Drawing it lower — at the tensor level — would have
put a dispatch in the inner loop; drawing it higher would have made a device
backend re-implement the model.

**No regex engine, no template engine.**
The two pre-tokenizer patterns in wide use are alternations tried in order, so
they are transcribed in that order rather than interpreted. Chat prompts are
shaped by detecting `<|im_start|>` in the vocabulary rather than evaluating the
Jinja `chat_template`. Both choices trade generality for not carrying a second
language inside a C file; both are documented and both have an override.

**Report rather than approximate.**
A tokenizer that is not byte level BPE returns `ILL_VOCAB` with a message naming
what it found. A tensor of the wrong extent names the tensor and both shapes. A
rewind that the convolution state cannot honour returns `ILL_STATE` instead of
guessing. The one deliberate approximation in the engine — Unicode character
classes by range table rather than by the full database — is called out in the
code, in `GUIDE.md`, and in `TODO.md`.

### Optimisation, in the order it happened

Each step was measured on a synthetic 2.9B checkpoint of LFM2-2.6B proportions,
four x86-64 cores with AVX-512.

1. **Baseline**, bf16, tile of 4: 39 tok/s prefill, 5.2 tok/s decode.
2. **Wider dense tiles** (8 and 16) were tried and were *slower* — 35 and 36
   tok/s prefill. Four activation rows of a 2560-wide model already fill L1, so
   the tile stayed at 4. Prefill is arithmetic bound at roughly 230 GFLOP/s, not
   weight-bandwidth bound as first assumed.
3. **q8 at 256 bits** was slower than bf16 in prefill — 24.9 against 39 tok/s —
   because the int8 kernel used AVX2 while the float kernel used AVX-512.
4. **q8 at 512 bits** (`_mm512_madd_epi16` on AVX-512BW, where a 32-value block
   widens to exactly one register) brought prefill to 37.3 and decode to 8.9.
5. **A vectorised quantiser** for the activation packing took prefill to 37.5.

Decode scales 2.4 → 4.6 → 8.9 tok/s over one, two, and four threads.

### What was verified, and what was not

Verified here: every architectural shape in the family against `transformers`,
to 2e-7 relative on f32 checkpoints; cache behaviour under chunked prefill and
single token decode; f16 and bf16 storage; q8 agreement; thread agreement;
tokenizer agreement against the `tokenizers` library for both split flavours;
73 unit checks; clean runs under AddressSanitizer and UndefinedBehaviorSanitizer;
builds with gcc and clang at vector widths 1, 8, and 16.

**Not verified here: the real `LiquidAI/LFM2.5-2.6B` weights.**
`huggingface.co` was blocked by the development environment's egress policy, so
the checkpoint could not be downloaded. Everything the checkpoint would
exercise — the architecture, the layer plan, the tokenizer format, bf16
storage, the dimensions — is covered by the synthetic suite, and
`run.py test --model PATH` runs the same comparisons against a real folder.
Running that against the published weights is the first item in `TODO.md`.

The Windows and NEON code paths are written and guarded but were not compiled
or run on those hosts; that is also recorded in `TODO.md`.

### Corrections made during development

- **The GPT-2 byte alphabet** was first built with a two-pass loop that
  mishandled byte `0x00`, which collides with the "not yet assigned" sentinel.
  Rewritten as a single pass. `app_test.c` now checks the map is a bijection
  over all 256 bytes.
- **The JSON reader accepted trailing commas.** Found by the unit test that
  feeds it eight malformed documents; now rejected.
- **The piece pool grew by reallocation** and fixed up every piece pointer on
  each growth — a lot of machinery for no benefit, since the document being
  parsed already bounds the total. Replaced with a bump allocator sized once.
- **A stale correlation in the q8 test** divided by zero. The cause was the
  fixture, not the engine: the reference zeroes the padding embedding, so a
  sequence starting at token 0 drives the whole stack to zero. The fixture now
  avoids id 0, which also removes a case that would have hidden real differences.
- **`run.py` swallowed its own build flags**, because `argparse.REMAINDER`
  consumes everything after the first positional. Arguments after a literal `--`
  are now split off before argparse sees them.
- **`--cc clang` was treated as MSVC**, because the detection matched any
  compiler whose name starts with `cl`. Now matched exactly.
- **A vocabulary larger than the model** produced a bare "bad argument". It now
  names the offending token id and the model's vocabulary size, and the loader
  warns when the two do not belong to the same checkpoint.

### Restructuring

The file was reorganised once, mid-development, when vectorising the q8
quantiser revealed that the vector vocabulary sat *below* code that wanted it.
Rather than duplicate the operations, the layering was fixed: number formats,
then the vector vocabulary, then block quantisation, then everything that uses
them. Fifteen parts instead of thirteen, and every part now depends only on the
parts above it.

---

## 1.1.0 — the checkpoint suite folds into the test harness

The `bench/` directory is gone. Its comparisons live in `app_test.py` now, and
the published checkpoint ships in `model/`, so the whole suite runs without a
download.

### What it took

- `bench/equivalent.py` and `bench/longform.py` were probes over one real
  checkpoint; `app_test.py` already ran the same pair of metrics over every
  architectural shape. The checkpoint branch of the harness now also runs a
  throughput comparison and a greedy comparison, so one script covers what
  three did.
- `bench/bench_hf.py` measured transformers' prefill and decode beside the
  engine's `bench`. That comparison is now `checkpoint/throughput` in the
  harness: both sides at bf16, the same thread count, the same fill and step
  counts. Quoting the engine's q8 against the reference's bf16 would fold a
  weight width difference into what looks like an engine difference, so the
  harness never does it.
- The greedy comparison is judged on the shared prefix of ids, not of text:
  `generate` emits decoded text, and re-encoding text to recover ids is not a
  round trip, so re-encoding would report divergence the engines never
  produced. The bar is measured, not assumed: the reference is run against
  itself, primed one token at a time against one forward over the whole
  prompt, and the engine is allowed to follow half as far as the reference
  follows itself before it is called wrong.
- `bench/download_model.py`, `bench/make_index.py`, and `bench/reshard.py` are
  gone with the rest. The checkpoint is in the repository, so there is nothing
  to download; the shards ship with an index and under the LFS cap, so there
  is nothing to repair.
- `app_test.py` takes `--model` defaulting to `./model`, `--no-checkpoint` to
  skip the branch, and treats a missing checkpoint folder as a skip rather
  than a failure: the synthetic suite is the parity argument and runs without
  it.

### What it is worth

One command, `python3 run.py test`, now answers every question the bench
scripts answered: is the arithmetic right, does the tokenizer agree, is the
engine at least as fast as the reference doing the same work, and does a
greedy continuation hold. Nothing to download first.

### That it is the same answer

`checkpoint/logits` and `checkpoint/tokenizer` are the checks `equivalent.py`
and the tokenizer probe ran, unchanged, over the same checkpoint. The
throughput and greedy checks are new to the harness but measure what
`bench_hf.py` and `longform.py` measured, with the bar for the greedy
comparison tightened from a fixed threshold to the reference's own movement.

### Code

`app_test.py` grew the two checks and the default model path. `bench/` is
deleted. `README.md`, `GUIDE.md`, and `TODO.md` no longer name it.

### Tests

The synthetic suite is untouched and still passes. The checkpoint suite runs
against `model/` by default.
