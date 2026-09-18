# CHANGES

Development progress and the reasoning behind it. The log below runs oldest
first; what spans versions is collected here at the top instead.

---

## Standing results

### The hosts

A rate belongs to a host, so a host is named beside every rate in the log. The
bare memory sweep is a sequential read of 2 GiB, one cache line touched per
64 bytes, taken as the best of five runs — it is the ceiling a decode step is
measured against, because a decode reads every weight once and can do no
better than the machine can fetch.

| host | cores | vector | bare sweep, 1 / 2 / 4 threads |
| --- | --- | --- | --- |
| `xeon-2.8` — Intel Xeon @ 2.80 GHz, virtual | 4 | AVX-512 with VNNI | 11.0 / 19.1 / 36.6 GB/s |
| `xeon-2.1` — Intel Xeon @ 2.10 GHz, virtual | 4 | AVX-512 | not taken |

`xeon-2.1` is where the Vulkan backend (1.8.0) was written, and its only Vulkan
device is SwiftShader — a software rasterizer running SPIR-V on those same four
cores. **No rate taken through it is a GPU rate.** What it can show is that the
shaders are right and what the seam costs; what a discrete card does with these
kernels is unmeasured, and is marked blocked rather than estimated.

### The cost split

Decode reads every weight once a token, so its rate is bytes over what the host
can fetch and nothing else matters much. On `xeon-2.8` the 2.6B checkpoint at
q8 is 2.83 GiB, and decode at 10.7 tok/s is 32.5 GB/s against a 36.6 GB/s
sweep — **89% of what the machine can deliver**. That number is why the open
work at the top of `TODO.md` is about reading fewer bytes or getting more than
one token out of a read, and not about faster arithmetic: there is 11% left in
the kernel and the rest has to come from somewhere else.

Prefill is arithmetic bound instead, and has further to go: 32.6 tok/s on the
same host is 88 G multiply-adds a second. That is also why a narrower weight
format helps decode and hurts prefill — q4 (1.6.0) is 1.23x on decode and
0.93x on prefill, because unpacking is arithmetic and prefill has none to
spare.

That ceiling is a ceiling on **reads**, not on tokens. `--draft` (1.5.0) gets
more than one token out of a read by verifying proposals in the same pass, and
takes greedy decoding to 11.8 tok/s on work whose answer quotes its question —
past the bare-read figure, because the read is no longer one token's.

### The picture engine's parity argument

`app/yolo.c` (1.7.0) is checked against `ultralytics` out of the
`ckpt/yolo26/py` submodule, and the comparison is only meaningful on **the same
decoded pixels**: stb and OpenCV do not agree on a JPEG to the last unit, so a
run through a `.jpg` measures the two decoders as much as the engines. Write
the picture out as PNG or PNM first. Chasing that difference is in the refusal
register.

Three things in the shaping had to be matched rather than improved on, and each
is small enough to look like noise: the letterbox pads to a stride multiple and
not to a square; the resize is OpenCV's fixed-point byte path, where rounding
once at the end instead — which is more accurate — is wrong by a unit on a
tenth of the cells; and a classifier's shaping goes through Pillow, which
antialiases. Each of them moved a real detection.

### What a weight format costs

Measured by `perplexity` (1.3.0) on the published 2.6B checkpoint, over two
unlike texts — technical prose, and licence boilerplate a model has seen many
times:

| | `GUIDE.md` prose, 2056 tokens | Apache licence, 1900 tokens |
| --- | --- | --- |
| bf16, as stored | 3.6080 nats — 36.892 | 0.6798 nats — 1.9734 |
| q8, repacked | 3.5999 nats — 36.594 | 0.6756 nats — 1.9651 |
| q4, repacked | 3.5805 nats — 35.892 | 1.2642 nats — **3.5401** |

**q8 costs nothing this measurement can see** — under a quarter of a percent on
both, and falling the wrong way for a lossy format, so the sign is not a result
and the size is.

**q4 costs nothing on the prose and most of the licence.** The prose number
moves less than q8's did; the licence number nearly doubles, 0.68 nats a token
to 1.26. That gap is the whole lesson: **a perplexity taken on ordinary prose
does not see what four bits costs**, because prose is where the model is
unsure anyway and a blurred distribution is still about as wrong. Where the
model is confident, four bits is where the confidence goes. Judge a narrow
format on text it should find easy.

### The published checkpoint

`python3 util/make.py test --model ckpt/2.6b` runs the whole harness against the
real `LiquidAI/LFM2.5-2.6B` weights, not a synthetic stand-in. On `xeon-2.8`
against `transformers` 5.17.0 and `torch` 2.14.0, quiet machine, **23 of 23
pass**:

- **logits** — max relative error 7.15e-06, top-1 agreement 100%.
- **greedy** — the engine follows the reference for 155 characters, where the
  reference primed a token at a time follows its own one-batch self for 0.
- **throughput**, both sides at bf16 on the same threads — engine prefill
  18.0 tok/s against the reference's 7.9, decode 5.1 against 3.5.
- **thread agreement** — 1 worker against 4, max relative error 0.00e+00. The
  engine's output does not depend on how many threads produced it.

### What a host-pointer seam costs a device backend

Both backend seams pass host pointers: `dense` takes a `const float *src` and a
`float *dst`, `conv_run` takes two `YoloPlane`s over ordinary malloc memory.
There is nowhere in either interface to say that an answer is already on the
device and should stay there, so a device backend uploads its inputs and
downloads its outputs on **every** call. Weights are cached by host pointer,
because they are immutable after load; activations cannot be, because nothing
in the seam says when the host last wrote them.

Measured on `xeon-2.1` (1.8.0): yolo26n over a 320 picture is **139
submissions and 46.4 MiB moved** against 12 MiB of resident weights — a feature
map crosses about four times. The 2.6B text checkpoint, one `logits` call on a
six token prompt, is **320 submissions and 196.7 MiB moved**.

So a device backend against these seams wins on large shapes and loses on small
ones, and the submit-and-wait around a single-token dispatch can cost more than
the arithmetic inside it. That is a property of the interface, not of any
device, and it is the reason `TODO.md` item 51 is a change to `app/core.c` and
`app/yolo.c` rather than to `app/vulk.c`.

### The refusal register

Ideas that were tried, measured, and are not worth having. They are here so
that the next person does not have the same idea twice.

- **Keeping the vocabulary plane at q8 while the body is q4** (1.6.0). The
  standard remedy for a lossy weight format is to spare the output head, which
  on this checkpoint is tied to the embedding and is 262M of 2.69B parameters —
  1.57 GiB becomes 1.69 GiB, which is cheap. It measured **worse**, on both
  texts and by about the same margin: prose 35.892 to 38.339, licence 3.5401 to
  3.8450. No account of why is offered, because none was established; what is
  established is that the obvious move does not pay here and should not be
  made again without a reason better than that it usually works.

- **Reconciling stb's JPEG decoder with OpenCV's** (1.7.0). The last
  disagreement between `app/yolo.c` and `ultralytics` on a `.jpg` is not in
  either engine: the two decoders differ by a unit on a few percent of pixels,
  which on bus.jpg moved a marginal detection's score from 0.534 to 0.577 while
  every other detection stayed to the digit. Decoded through the same bytes,
  the two engines agree exactly. There is nothing to fix on this side of it,
  and matching a particular IDCT and chroma upsampling would be a JPEG decoder
  written to imitate another JPEG decoder. Compare on PNG or PNM instead.

- **Rounding the letterbox resize once rather than twice** (1.7.0). OpenCV's
  byte path drops four bits off each row before it weighs them and keeps two
  fractional bits through the sum; doing the arithmetic in one wider step and
  rounding once at the end is strictly more accurate and is **wrong by a unit
  on about a tenth of the cells**, which carried to a hundredth of a logit at
  the head and a different score on a marginal detection. The engine is not
  trying to resize well; it is trying to resize the same.

- **Memoising the q8 activation pack** (1.2.0). Three planes in attention and
  two in the feed forward read the same normalised row, so the same bytes are
  quantised more than once a layer. Removing the repeats is exact and takes
  15.7% of the packed values off a step — but packing is 4 operations a value
  against 2.69e9 multiply-adds a token, so the whole of it is 0.089% of a step
  and the saving is 0.014%. Measured on `xeon-2.8` it did not clear the noise
  in either direction over three alternations. The premise was right and the
  quantity was too small to matter.

---

## unreleased — MSVC support and checkpoint verification

Verified the engine against the published `LiquidAI/lfm2.5-2.6b-a` checkpoint and
got it building under MSVC on Windows.

### Layout

The flat file list became directories: the engine and its command line moved to
`app/core.c` and `app/main.c`, the two test suites to `test/test.c` and
`test/test.py`, and the Python workflows to `util/make.py` (was `run.py`) and
`util/tune.py` (was `app_tune.py`). The tuning corpus moved to
`data/tune.jsonl` and the published checkpoints to `ckpt/lfm2.5-0.4b`, `ckpt/lfm2.5-1.2b-i` and
`ckpt/lfm2.5-2.6b-a`. The engine sources are renamed, not split: `app/main.c` and
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

**Not verified here: the real `LiquidAI/lfm2.5-2.6b-a` weights.**
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

---

## 1.2.0 — the q8 dot takes two blocks at a time

Prefill was leaving most of the machine unused. On `xeon-2.8` a 256 token
prefill of the published 2.6B checkpoint at q8 ran at 21.7 tok/s, which is
58 G multiply-adds a second on four cores that can issue far more; the weight
stream over the same run was 15.7 GB/s against a 36.6 GB/s sweep, so it was not
waiting on memory. It was waiting on instruction issue, and most of the
instructions it was issuing were not multiplies.

### What it took

- **A paired q8 dot.** Two blocks are 64 bytes, which is one 512 bit register.
  `vpdpbusd` folds four byte products into each of its sixteen int32 lanes
  where `vpmaddwd` folds two, and — the larger half of the saving on a machine
  that is issue bound — the two byte-to-int16 widenings that fed `vpmaddwd`
  disappear, because VNNI reads bytes directly. Lanes 0..7 then carry the low
  block's products and lanes 8..15 the high block's, so the two block scales go
  in as one vector with eight lanes of each.

  `vpdpbusd` reads its left operand unsigned, so the weights go in as
  magnitudes and their sign moves onto the activations with a mask and a
  subtract. AVX-512 has no `vpsignb`, which is how the same trick is written
  for AVX2; `vpmovb2m` and a masked negate are the two instructions that stand
  in for it. A weight of -128 is not a special case — its magnitude is 128,
  which is exactly what the unsigned operand is for — and the activation side
  cannot overflow because `ill_q8_pack` clamps both sides to -127..127.

  Every instruction set without VNNI folds `ill_q8_pair` back into the two
  single block steps it stands for, in that order, so nothing outside a VNNI
  build moves by a bit.

- **A wider prefill tile.** `dense` fused at most four activation rows against
  a weight row, so a 256 token prefill streamed every weight sixty-four times.
  Eight rows halves that. Sixteen was tried and is not better — it is level
  with four, which is what register spilling looks like — so the tile is eight.

### What it is worth

On `xeon-2.8`, the published 2.6B checkpoint at q8, four threads, the minimum
each phase reached over three alternations with the old build going first:

| | prefill, 256 tok | decode, 32 tok |
| --- | --- | --- |
| before | 11.813 s — 21.7 tok/s | 3.229 s — 9.9 tok/s |
| after | 7.852 s — 32.6 tok/s | 2.984 s — 10.7 tok/s |

**Prefill is 1.50x.** Decode is 1.08x, and that is the more interesting number
of the two, because decode was already near the memory ceiling and had little
to give: at 10.7 tok/s it reads 32.5 GB/s against a 36.6 GB/s sweep, 89% of
the host. Whatever is next for decode has to read fewer bytes or get more than
one token out of a read.

The two halves were separated. The wider tile alone is 11.832 s to 11.516 s
over seven paired runs — **2.7%**, small but won every pairing. The rest is
the paired dot: 11.696 s to 7.944 s at a fixed tile of eight, **1.47x**, won
5 of 5 regardless of which build ran first. The host drifts slower over a
series, so the build that runs second is at a disadvantage; the paired dot won
from both positions.

### That it is not the same answer

The wider tile is held bit for bit: a row's accumulation order over its input
is unchanged, only which tokens are grouped, so greedy `chat` against the 2.6B
checkpoint is byte identical to the build before it.

The paired dot is a **decision**. Sixteen lanes of two products regrouped into
two halves of eight lanes of four is a different summation order, so the last
bits move, and thirty layers amplify that into a different greedy continuation
after a shared prefix. The size of the move, over 24 rows of the real
checkpoint's 128000 logits:

| | max relative error | top-1 agreement |
| --- | --- | --- |
| bf16 against q8 — what the format costs | 3.784e-02 | 24/24 |
| q8 against q8 with the paired dot | 2.106e-02 | 24/24 |

**The regrouping moves the logits less than choosing q8 over bf16 already
does**, and picks the same token every time. It moves the output all the same,
and it only moves it on a host that has the instruction, which under
`-march=native` is the default build there. That is the trade: say so rather
than hide it behind a flag nobody sets.

### Code

`app/core.c`. `ill_q8_pair` is new in part 5, beside `ill_q8_step`: a VNNI
arm, and a fallback for everything else that calls the single block step twice.
`ILL_DENSE_Q8_BODY` walks blocks two at a time and keeps the old loop for an
odd trailing block. `ILL_TILE_MAX` is eight, and both dispatch switches carry a
case for every width up to it.

### Tests

`TEST_TOKS` was five, which is fewer rows than the widest tile now asks for, so
the tile sweeps were reading past their own buffers; it is `ILL_TILE_MAX + 1`,
which also leaves a partial tile covered. Both sweeps now name the width they
ran rather than calling everything above three a four token tile, and the q8
sweep checks every width instead of only the last.

Two checks on the paired dot: that it equals the two single block dots it
stands for, and that it survives a -128 weight, which is the one value whose
magnitude does not fit the signed byte the other path uses.

They bite. Narrowing the dispatch default from eight to four fails the eight
token tile at 1.0e+00 — the rows past the fourth are never written. Dropping
the masked negate that moves the weight's sign onto the activation fails both
paired dot checks and eight of the q8 dense sweep, 10 in all.

The suite passes on the native, portable, no-simd and AVX2-only builds.
**86 pass**, 73 before.

---

## 1.3.0 — a number for what q8 costs

`--quant q8` halves the weights and there was no way to say what it took in
accuracy. What the project had was a correlation against `transformers` —
top-1 agreement 100%, correlation 0.9999 — which says the two engines agree
with each other and not what either is worth. Every open item that trades
accuracy for speed, q4 and a narrower key/value cache first among them, was
therefore unlandable on evidence: there was nothing to move in a known
direction.

### What it took

- **A `perplexity` command.** It reads a text from stdin or `--prompt`, runs
  one left to right pass, and reports the mean negative log likelihood the
  model gives each token in nats, in bits, and as its exponent. Every token but
  the first is scored, against the whole text before it rather than a window,
  so a token late in the text is judged on everything the model has seen. The
  pass is cut into `--batch` chunks only because asking for a row per token
  over a long text at once would allocate the vocabulary once per token —
  128000 floats a row is 131 MB at a chunk of 256, and a thousand rows would be
  half a gigabyte.

  The text is scored as it stands, with no chat template wrapped around it even
  without `--raw`. Shaping a turn around it would score the template's own
  tokens, which is not what the number is for.

- **`ill_row_logsum` in the engine**, beside `ill_soft_max` rather than in the
  command line. It is the normaliser a row's log probabilities are measured
  against, taken relative to the row's peak because `exp` of a logit of this
  size is infinity, and summed in double because a score adds one of these per
  token over a whole text. `ill_soft_max` answers a different question — it
  wants the probabilities and may destroy the row to get them — so the two sit
  side by side rather than one calling the other.

### What it is worth

The published 2.6B checkpoint on `xeon-2.8`, over two texts that are nothing
like each other: about 2050 tokens of this repository's own `GUIDE.md`, which
is technical prose with tables in it, and about 1900 tokens of the Apache
licence, which is boilerplate a model has seen many times.

| | GUIDE.md prose | licence boilerplate |
| --- | --- | --- |
| bf16, as stored | 3.6080 nats — **36.892** | 0.6798 nats — **1.9734** |
| q8, repacked | 3.5999 nats — **36.594** | 0.6756 nats — **1.9651** |

**The cost of q8 is not distinguishable from zero on either text.** The gap is
0.008 nats a token on the first and 0.004 on the second, both under a quarter
of a percent, and on both it falls the wrong way for a lossy format: q8 scores
marginally better than the weights it was made from.

Do not read the sign. Two texts is not a corpus and a quarter of a percent is
not a direction; what the pair of numbers supports is the claim that the
format's cost is below what this measurement can see, which is the claim that
was wanted. A block of 32 values sharing one f32 scale can resolve more finely
than bf16's eight bits of significand where the block's values are of similar
size, which is a reason the sign could be real, and is not evidence that it is.

The number these were taken to enable is q4's. This is the instrument, not the
result.

### Code

`app/main.c` gains `app_do_perplexity` and `app_slurp`, and `perplexity` joins
the verb table and the help. `app/core.c` gains `ill_row_logsum` in part 8,
beside the elementwise stages.

### Tests

Three checks on the normaliser, in a new `scoring` area. A flat row of n equal
values normalises to `v + log(n)`, which is the one case whose answer can be
written down; the same row therefore gives every token exactly `-log(n)`, which
is the identity the scoring loop rests on; and a row holding 800.0 still
normalises, where `exp(800)` is infinity in double.

They bite: dropping the subtraction of the row's peak leaves the third
reporting `inf`.

**89 pass**, 86 before.

---

## 1.4.0 — a character is shown once it is whole

Streaming wrote each token's bytes the moment they arrived. A byte level
tokenizer splits characters across tokens as a matter of course — `café` in the
published vocabulary ends on a piece that carries the first byte of `é` and
nothing else — so a terminal was shown half a character, drew the replacement
glyph for it, and corrected itself when the next token landed. Every language
that is not ASCII flickered, and the worse the language fitted the vocabulary
the more it flickered.

### What it took

- **`ill_utf8_hold`** in the engine, beside `ill_utf8_read`: how many bytes at
  the end of a buffer begin a sequence that has not finished. It answers zero
  whenever releasing is the right thing to do, which includes a buffer ending
  in something that is not valid UTF-8 at all — a writer holding bytes back
  wants rubbish released rather than a wait for a continuation that is never
  coming.
- **`AppTail`** in the command line, carried through a reply. It prepends what
  it held to the next token's bytes, writes everything whole, and keeps the
  rest. At most three bytes are ever held. `app_tail_flush` releases them when
  the reply ends, whether or not they ever became a character, so a run cannot
  swallow its own last bytes — and it runs on the error path too.

### What it is worth

Nothing is written that was not written before, and nothing is written that
was: decoding the eight tokens of `こんにちは、世界 😀 naïve café` byte for byte
matches the build before the change. What moved is *when* — a character reaches
the terminal once, whole, instead of arriving broken and being repaired.

There is no rate attached to this. It costs a `memcpy` of at most three bytes a
token against a decode step measured in tens of milliseconds.

### That the harness agrees

Unrelated to the above and closed in the same pass: the full reference
comparison has now been run against the published `LiquidAI/LFM2.5-2.6B`
weights rather than against synthetic checkpoints — every architectural shape,
the cache paths, the tokenizer corpus and the checkpoint branch, 23 of 23. The
numbers are in the standing results. The parity argument is no longer made
only on checkpoints the harness built itself.

### Code

`app/core.c` gains `ill_utf8_hold` in part 13. `app/main.c` replaces
`app_piece_show` with `AppTail`, `app_tail_open`, `app_tail_show` and
`app_tail_flush`; the run loop and the `tokens` decode path both carry one.

### Tests

Ten cases on the rule itself — ascii, a finished sequence of each width, a lone
lead byte, two thirds of a three byte sequence, three quarters of a four byte
one, a byte that leads nothing, continuations with no lead, and an empty
buffer. Then the property the streaming path actually needs: fed a mixed width
string one byte at a time, releasing what the rule does not hold reproduces the
string exactly and never leaves what has been shown ending part way through a
character.

That last check is walked with the test's own table of how long a lead byte
promises to be, not with `ill_utf8_hold`. The first draft asked the engine, and
a check that calls the function it is checking cannot fail: making the rule
hold nothing left it passing. With the test's own rule it fails, as do the
three cases that name a held count.

**100 pass**, 89 before.

---

## 1.5.0 — the context drafts, the model verifies

Decode reads 2.83 GiB to produce one token, and on `xeon-2.8` that read is 89%
of everything the memory can deliver. There is no faster way to read those
bytes. The only moves left are to read fewer of them, or to get more than one
token out of a read, and this is the second.

### What it took

- **A mark on the state.** `ill_state_mark` remembers where a sequence is and
  `ill_state_back` returns to it. The key/value cache needs nothing saved: it
  is appended to, the attention scan is bounded by the fill at the time, and
  the rows past a rewound fill are written over by whatever arrives next. What
  a mark costs is a copy of the convolution window — `conv_count * model_dim *
  (conv_width - 1)` floats, 352 KiB on the 2.6B — claimed on the first mark so
  a caller that never marks never pays for it.

  `ill_state_crop` reported `ILL_STATE` on any model with convolution layers,
  because the window is recurrent and a partial rewind could not reconstruct
  the values it dropped. It now succeeds at the fill a mark was taken at, which
  is exactly the case where those values were kept. Anywhere else it still
  refuses, and says why.

- **`ill_draft_scan`**, which proposes a continuation by finding where the tail
  of the sequence last appeared earlier in it and copying what followed. Runs
  of three are tried before runs of two, because the longer agreement is the
  one whose continuation is worth believing.

- **A verification loop** behind `--draft N`. The proposal goes through one
  forward pass with the tokens carried from the previous round, and each
  proposed token is checked against the model's own choice for that slot.
  Where the proposals all hold, the state is already the confirmed sequence and
  the mark moves forward. Where one is rejected, the state holds tokens that
  will not be emitted, so it goes back to the mark and the confirmed tokens
  ride into the next round's pass — which costs nothing, because that pass was
  going to read the weights anyway.

  That is why one mark is enough. Returning to the middle of a batch would need
  a window the state does not keep; returning to its start needs only the copy
  the mark took. The carry is capped at sixteen so a long run of rejections
  cannot widen every pass from there on.

### What it is worth

The published 2.6B at q8 on `xeon-2.8`, greedy, 160 tokens, the best of paired
runs:

| what was asked for | plain | `--draft 4` | |
| --- | --- | --- | --- |
| quote a document back, then describe it | 9.2 tok/s | 11.8 tok/s | **1.29x** |
| rewrite a C function, changing one thing | 9.2 tok/s | 11.1 tok/s | **1.21x** |
| repeat a sentence, then explain it | 9.2 tok/s | 10.7 tok/s | **1.16x** |
| invent an original fable | 9.1 tok/s | 9.1 tok/s | **1.00x** |

**The downside is nothing and the upside is about a quarter.** That asymmetry
is the point: a rejected proposal costs a row in a pass that was already
reading the weights, so work whose answer quotes its question gains and work
that invents every token loses nothing measurable.

`--draft 8` is worse than `--draft 4`, not better — 9.9 tok/s against 11.8 on
the quoting prompt. Past a handful of tokens the pass stops being free and
acceptance does not keep up with the arithmetic. Four is not tuned, it is
merely better than eight; the flag takes a number because the right one is a
property of the work.

### That it is the same answer

Every token emitted is the one the model's own row chose; the draft only
decides which rows get computed early. So the bar is not a tolerance but exact
equality, and it holds: greedy `generate` and `chat` on the 2.6B are byte
identical with `--draft 4`, with `--draft 8`, and without.

`--draft` refuses to engage above temperature zero and says so. Accepting a
candidate because it equals an argmax is not a test a sampled distribution can
pass, and a repetition penalty rewrites the row it is applied to, which a
verification pass must do exactly once. Quietly biasing sampling toward
whatever appeared earlier in the prompt would be a worse bargain than the
speed. The proper acceptance rule is open as its own item.

### Code

`app/core.c`: `ill_state_mark`, `ill_state_back` and `ill_state_mark_at` in
part 11, `ill_draft_scan` in part 14, and `ill_state_crop` consulting the mark.
`app/main.c`: `AppDraft` and the rewritten `app_run_loop`, plus `--draft N`.

### Tests

Eight checks on the scan: what a repeated run proposes, that the most recent
match wins, that a longer run beats a nearer short one, that an unrepeated tail
proposes nothing, that a pattern which already repeated proposes it repeats
again, that the tail never matches where it stands, that it never proposes more
than it is asked for, and that too short a history proposes nothing.

Two of those started as wrong expectations rather than bugs: a scan asked for
four tokens returns four when four followed, and a sequence that has already
repeated proposes that it repeats again. The behaviour was right and the tests
were rewritten to say so. A third fixture did not discriminate — it agreed on
the answer whether long runs or short ones were tried first — and was rebuilt
so the two pull apart.

The end to end check is `checkpoint/draft` in the reference harness: `--draft
4` and `--draft 8` against plain greedy, exact equality. It runs against the
checkpoint and not a synthetic model on purpose. The first version built a
random-weight model, which settles on one token that a draft proposes and the
model accepts every time — the rewind was never reached, and breaking the
window restore left the check passing. Against the 350M checkpoint proposals
are accepted and rejected in turn: dropping the window restore parts the runs
at character 34, and dropping the fill rewind parts them at 37.

**108 pass**, 100 before, and 24 in the reference harness against the 2.6B,
23 before.

---

## 1.6.0 — q4, and what four bits actually costs

q8 puts the 2.6B checkpoint in 2.83 GiB. A machine with 4 GiB of usable memory
cannot hold that beside anything else, and decode is bytes over bandwidth, so
the format is both the memory question and the speed question.

### What it took

- **`ILL_TYPE_Q4`**: the same 32 value block as q8 with the values at half the
  width, two to a byte. Twenty bytes a block against thirty-six, so a
  checkpoint reads 0.56 of what it read at q8.

  The sixteen levels are used by placing the block's largest value exactly on
  -8 rather than by dividing its magnitude by eight and clipping, which would
  hand back seven eighths of the peak. The scale therefore carries a sign,
  which costs nothing — it is a float multiply at the end of a row.

  Within a block the low nibbles of the sixteen bytes hold values 0..15 and the
  high nibbles 16..31, so a block lifts in two halves rather than by striding
  through it two at a time.

- **A lift that never touches memory.** This is the whole of whether the format
  pays, and it took three attempts to see it:

  | the lift | q4 decode | against q8's 11.0 |
  | --- | --- | --- |
  | sixteen scalar iterations a block | 1.5 tok/s | **7x slower** |
  | vectorised, through a 64 byte buffer | 5.1 tok/s | 1.8x slower |
  | vectorised, into a register the dot reads | 11.9 tok/s | **1.23x faster** |

  The first two read *fewer* bytes than q8 and lost anyway. Unpacking is
  arithmetic, and a decode step that was 89% of the memory ceiling has no
  arithmetic to spare; a store and a reload of sixty-four bytes per block pair
  was enough to give the whole saving back. `ill_q4_open` lifts two blocks --
  thirty-two bytes -- into one 512 bit register with five instructions, and
  `ill_q8_pair_wide` takes the weights from a register rather than a pointer so
  q8 and q4 share the dot.

### What it is worth

The published 2.6B on `xeon-2.8`, minimum over three alternations with q8 going
first:

| | weights | prefill, 256 tok | decode |
| --- | --- | --- | --- |
| q8 | 2.83 GiB | 7.953 s — 32.2 tok/s | 4.933 s — 9.7 tok/s |
| q4 | 1.57 GiB | 8.522 s — 30.0 tok/s | 4.026 s — 11.9 tok/s |

**0.55 of the memory and 1.23x the decode, for 0.93x the prefill.** The prefill
loss is the same fact as the decode gain seen from the other side: prefill is
arithmetic bound, and the unpack is arithmetic it cannot absorb.

The decode gain is well short of the 1.8x the byte count suggests, because q4
decode is no longer waiting on memory: 11.9 tok/s over 1.57 GiB is 20.0 GB/s
against a 36.6 GB/s sweep. It is the first thing in this engine to come off the
memory ceiling, and what holds it now is the kernel.

### What it costs

This is the number the `perplexity` command was built for, and it is not the
number an ordinary benchmark would have reported:

| | prose | licence boilerplate |
| --- | --- | --- |
| q8 | 36.594 | 1.9651 |
| q4 | 35.892 | **3.5401** |

**On prose, q4 costs nothing visible. On text the model should find easy, it
costs most of the model's confidence** — 0.68 nats a token to 1.26. Had only
the prose corpus been run, q4 would have looked free, and it is not. A narrow
format has to be judged on text the model is sure about, because that is where
the certainty it is destroying lives.

That is not a reason to refuse q4; it is a reason to say what it is for. At
1.57 GiB it puts this checkpoint on a machine that could not hold it at all,
and for open-ended prose it reads the same. For quoting a document back exactly
it is the wrong format, and that should be a choice the caller makes knowing
the number.

### That the obvious remedy does not work

Sparing the output head is what is usually done, and here the head is tied to
the embedding and is 262M of 2.69B — 1.57 GiB becomes 1.69 GiB. Measured, it is
**worse on both texts**: prose 35.892 to 38.339, licence 3.5401 to 3.8450. It
is in the refusal register with those numbers and no explanation, because none
was established. Somebody will think of it again; this is so they measure it
rather than assume it.

### Code

`app/core.c`: `ILL_TYPE_Q4` with its name and size, `ill_q4_pack` and
`ill_q4_lift` in part 5, `ill_q4_open` and `ill_q4_dot` beside the q8 pair,
`ill_dense_nib` in part 8, a `ILL_TYPE_Q4` arm in `ill_plane_row`,
`ill_model_pack` taking the format it is packing into, and the dense dispatch.
`app/main.c`: `--quant q4`.

### Tests

The dense sweep at every tile width against the same independent f32
restatement the q8 sweep uses, at eight times q8's tolerance because four bits
is about eight times its step; a row read back agreeing with what the dot is
using; that a value comes back within half a step and a step is the peak over
eight, which is what placing the extreme on -8 buys; and that the register lift
and the byte lift agree, compared through the dot because the register form has
no bytes to look at.

They bite. Dividing the peak by seven and clipping fails the half-step check at
0.857 against 0.813. Swapping the halves of the register lift fails nine
checks. Biasing the byte lift by one fails three — but only on `--no-simd`,
where that code is live at all; on a machine with vectors it is not compiled,
which is the reason the suite runs that flavour.

The reference harness gains `storage/q4_repack` beside the q8 one: top-1 100%,
correlation 0.9906 against `transformers`, where q8 reaches 0.9999. The bar
there is deliberately loose, because that check is for catching a packing bug —
what the format costs is a perplexity number and is taken above.

**119 pass**, 108 before, and 25 in the reference harness, 24 before.

---

## 1.7.0 — a second engine, for pictures

This repository ran one architecture from one store. `ckpt/yolo26` has held
twenty-five yolo26 checkpoints and a submodule of the reference implementation
since the weights were tracked, and nothing here could read any of them. The
usual route — export to ONNX, or convert the weights with a Python script and
ship a second format — puts Python back in a path this project exists to keep
it out of. So `app/yolo.c` reads the published `.pt` as it is.

### What it took

A `.pt` is a zip holding a pickled module tree and its raw storages. Reading it
needs three things this file did not have: a zip directory walk, a pickle
reader, and a way to turn the object graph into something runnable.

**The graph is read, not rebuilt.** The usual way to run a checkpoint outside
Python is to take its `state_dict` and reconstruct the architecture from the
model's yaml, applying the depth and width scale table for the size letter.
Every module in a torch checkpoint already carries its class name, the channel
counts it was constructed with, its kernel size, stride, padding and groups,
and its `f`/`i` wiring, because `torch.save` pickles the object graph rather
than the weights alone. Re-deriving that from the yaml is guesswork about a
computation the checkpoint has already done. It is why n, s and x load through
the same code with nothing keyed on the size, and why the four task heads —
`Detect`, `Segment26`, `Pose26`, `OBB26` — needed no per-task loader.

**The pickle is read, never executed.** A `GLOBAL` is a pair of names. Four of
them mean something to a checkpoint — `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2`, `_rebuild_parameter`, and a storage's type
in a persistent id — and become the value they stand for. Every other becomes
an object carrying its class name and its state, which is what the graph layer
reads. Nothing constructs a class and nothing calls a function the file names.
An opcode outside the set is a refusal naming the opcode, not a guess.

One rewrite happens at load: a convolution followed by a batch norm becomes one
convolution with a bias, which is the identity `bn(conv(x)) = conv'(x)` and is
what ultralytics does before it predicts.

**Both heads.** yolo26 ships `cv2`/`cv3`, trained one to many and needing
suppression, and `one2one_cv2`/`one2one_cv3`, trained one to one and needing
none. The NMS-free head is what the architecture is known for; `yolo predict`
reads the other one unless told otherwise, and so does this. A parity run has
to be able to ask for either.

### What it is worth

Five tasks, five size letters, no conversion step, and a 4700-line translation
unit whose only dependency is two stb headers already in the tree — and
`-DYOLO_NO_STB` drops those for binary PNM.

On `xeon-2.8`, yolo26n at 640 over bus.png, the minimum of five runs: 405 ms in
total, of which 397 ms is the forward pass. One thread and no intrinsics. The
matrix multiply got there in two measured steps from 1670 ms: blocking the
columns and advancing four output rows together took it to 1032 ms, and holding
an eight-by-eight tile of output in registers for the whole of the mid loop
took it to 554 ms. Wider tiles were worse — six by sixteen measured 625 ms —
which is the registers running out.

`-O3` is **70% slower than `-O2`** on this file and the build says so: its loop
vectoriser rewrites that tile and spills it. `-O3` 645 ms, `-O2` 510 ms, `-O2`
with `-funroll-loops` 376 ms, all three producing the same numbers. `app/yolo.c`
is compiled at the last of those and `-O3` stays everywhere else.

### That it is the same answer

Against `ultralytics` 8.3.222, on the same decoded pixels, every checkpoint and
every task matches **to every digit the reference prints**: yolo26n, yolo26s
and yolo26x-seg on bus, yolo26n on zidane through both heads, and the pose, obb
and classify heads. Underneath that, the shaped input is bit for bit identical
and the head's class logits agree to **1.9e-5** over every logit that can
become a detection.

Getting there meant matching three things rather than improving on them, and
each looked like noise until it was not.

The letterbox pads to a **multiple of the stride, not to a square**. `yolo
predict` on a checkpoint sets `auto=True`, so 1280 by 720 becomes 640 by 384
and 810 by 1080 becomes 480 by 640 with no padding at all. A square changes the
anchor grid and every score with it.

The resize is **OpenCV's fixed point**, not a float bilinear: weights quantised
to 2048ths, four bits dropped off each row before it is weighed, two fractional
bits through the sum and rounded away. Doing the arithmetic in one wider step
and rounding once — which is more accurate — is wrong by a unit on a tenth of
the cells, and that carried to a hundredth of a logit and a different score on
a marginal detection.

A classifier's shaping goes through **Pillow, which antialiases**. Shrinking a
1080 pixel side to 224 it weighs a five-wide window, not the two nearest
pixels. The two-tap resize put the right label on bus.png with the wrong
confidence — minibus 0.77 against the reference's 0.52 — because the aliasing
it leaves behind is the high-frequency detail a classifier reads.

There is a fourth that is not an algorithm but a decoder: **stb and OpenCV do
not agree on a JPEG to the last unit**. A parity run that compares through a
`.jpg` measures the two decoders as much as the engines. Write the picture out
as PNG or PNM first. This is in the refusal register as a thing not to chase.

And one difference that is not a mistake on either side: rotated boxes are
suppressed by a different rule to axis-aligned ones, because the reference
suppresses them with different code. An axis-aligned box goes through
torchvision's greedy pass, where a struck box stops striking. An oriented box
goes through the matrix pass, where a box is struck if any higher scoring box
overlaps it, struck or not — so a middle box takes a third down with it after
being taken down itself. On yolo26n-obb that is two boxes reported where the
reference reports one. The thresholds differ too: greedy strikes above the
limit, the matrix at or above it.

### Code

`app/yolo.c`, new, eleven layers bottom-up: the arena and half-to-float; NCHW
planes; `YoloBackend` and its nine operations; the CPU kernels that fill them;
the zip, the pickle and the storage table; the graph builder with the batch
norm fold; the layer plan and the forward pass; anchors, box decode, both
suppression rules, masks and keypoints; pictures through stb or PNM, the two
letterbox geometries and the lift back to the source; the public interface; and
a command line under `YOLO_MAIN`.

`util/make.py`: `app_yolo` as a third target, built with `-DYOLO_MAIN` and at
`-O2 -funroll-loops`; a `yolo` workflow that hands arguments to it.
`AGENTS.md`, `GUIDE.md` §10, `README.md`: the second engine, and the argument
for opening a file list that says it is closed.

### Tests

`test/test.c` takes a second translation unit, built without stb because the
tests generate their own pictures. Forty-three checks: half to float including
the subnormals a small model is full of; the arena's mark and release, and its
alignment; every kernel against a plain restatement of the same arithmetic; the
two letterbox geometries including the odd gap that is the only case able to
tell the reference's rounding from an ordinary one; the box decode at both
`reg_max`; both overlaps; both suppression rules; the pickle reader on bytes
written in the test; and a PNM round trip.

They bite. Turning the matrix suppression back into greedy fails the matrix
check; turning greedy into the matrix rule fails the greedy one; dropping the
tenth the reference subtracts before rounding a pad fails the odd-gap check;
dropping a lane from the tile kernel's store fails the multiply; flushing half
subnormals to zero fails two of the number checks.

Two things the tests found. The arena rounded every take up to a cache line and
then handed out pointers from a malloc aligned to sixteen bytes, so the
alignment its comment claimed was not there. And a channel's plane is a padded
height apart, not a fitted height apart — which is only visible when there is
padding, so bus.jpg was right and zidane.jpg came out with its colour planes
sliding into each other.

**175 pass**, 132 before.

---

## 1.8.0 — a vulkan backend, for both seams at once

Both engines end their kernel section with a table of function pointers and a
comment saying an accelerator fills it. `IllBackend` has six operations,
`YoloBackend` has nine, and across every version in this log nothing had ever
been written on the other side of either. `TODO.md` items 23 and 41 were that
gap, one apiece. `app/vulk.c` closes both.

It is one file rather than two because the halves that differ are fourteen
compute shaders and the half that is the same — finding a device, allocating on
it, building a pipeline, getting a dispatch to run and the answer back — is
about two thousand lines that neither engine should own twice. A build takes
whichever half it wants: `app/main.c` compiles it with `VULK_NO_PICTURE` so
`ill` does not grow an image reader, `app/yolo.c` with `VULK_NO_TEXT`.

**The file list opened a second time**, and the argument is not the one 1.7.0
made. That was a second architecture. This is one file that is neither
engine and serves both: folding it into `app/core.c` would put a SPIR-V
assembler and a Vulkan loader in the text engine and leave the picture engine
unable to reach them, and writing it twice would be the same two thousand lines
in two places. `AGENTS.md` now says so, and the rule it states is unchanged:
nothing smaller than this reopens the list.

### What it took

Three decisions shape the file, and each of them is a refusal.

**No dependency, at any level.** There is no `vulkan.h`, no SDK, no `-lvulkan`,
and nothing in `util/make.py` beyond one define. The file declares the two dozen
structures and fifty entry points it uses against the published ABI and
resolves them through `dlopen` at run time. A build with `--vulkan` on a host
with no Vulkan at all still compiles and still runs; the backend reports itself
unavailable and the caller keeps the CPU one. That is the only arrangement
which does not make a GPU a build requirement for a project whose whole claim
is that a C compiler is enough. Two ABI conventions carry the weight and are
stated in the file rather than assumed: dispatchable handles are pointers and
non-dispatchable handles are always 64 bits, on a 32 bit host as much as a
64 bit one; and every structure begins with a tag the driver reads rather than
trusting the call it arrived through.

**The shaders are assembled here, at run time.** The usual arrangement — GLSL
compiled by `glslc` into a blob pasted into the source — needs a shader
compiler in the build and leaves twelve kilobytes of hex in the tree that
nobody can read or check. Part 3 is a SPIR-V assembler, about seven hundred
lines, and part 4 writes the fourteen kernels against it. Two shortcuts keep it
small enough to read. **No phi nodes**: every variable that crosses a branch is
an `OpVariable` in the Function storage class, read and written, which is what
glslang emits before its own mem2reg pass and what every driver turns back into
registers — it removes the only genuinely hard part of emitting structured
control flow by hand and costs nothing at run time. **One buffer shape**: every
binding is `struct { float cell[]; }` in the Uniform storage class with the
BufferBlock decoration, which is how a storage buffer is spelled in SPIR-V 1.0
and therefore works on a driver as old as Vulkan 1.0; integers ride in the same
buffers through `OpBitcast`, and the push constants are sixteen raw words read
the same way, so there is one descriptor layout and one push range for all
fourteen kernels.

**Quantised weights are widened once, on the way to the device.** A q4 plane is
a fifth of the f32 it came from, and that is why the text engine has it; here it
is expanded back to f32 as it is uploaded and the saving is lost. The
alternative — unpacking blocks inside the dense kernel — is a better use of
device memory and a worse use of the first version of this file, because it
needs one kernel per weight format and the formats are where the parity
argument is hardest. Widening is exact for f32, f16 and bf16, and for q8 and q4
it goes through `ill_plane_row`, the engine's own widening, so the device is
handed exactly the values the CPU dot product would have reconstructed. The
packed form is item 52.

Two kernels are not restatements of the CPU's, and both had to be.

**Attention streams.** The CPU writes the whole row of scores, takes a softmax
over it, and mixes; that needs `span` floats of scratch a worker, which the
seam supplies as `board`. A device would need `tokens * heads * span`, which at
a 512 token prefill with 32 heads and a 4096 window is 256 MiB of scratch for a
step whose output is 4 MiB. So the device kernel runs the streaming form: one
workgroup to a (token, head) pair, the span walked in chunks of sixty-four, a
running maximum and a running total both rescaled every time the maximum moves,
and the partial mix living in the output row. It reaches the same answer in one
pass over the keys and values, needs no scratch, and ignores `board` entirely.
This is TODO item 4's arithmetic, arrived at from the other direction: there it
is a decision because it would move the CPU's output, here there is no old
output to move.

**Convolution is direct.** The CPU lowers a patch matrix and multiplies, because
a contiguous stream is what a compiler vectorises. On a device the lowering is
nine reads and nine writes of the whole feature map for a 3x3, all through
memory, where the direct form reads the input nine times out of cache and
writes the output once. So `conv_room` answers zero — there is nothing to lower
into. Item 41 said `conv_room` has to answer for the backend that will run the
convolution rather than for the CPU one, and this is what that meant.

### What it is worth

**On this host, nothing, and the reason is the seam rather than the shaders.**

The host is `xeon-2.1` — Intel Xeon @ 2.10 GHz, virtual, 4 cores, AVX-512, 15
GiB — and the only Vulkan device on it is SwiftShader, a software rasterizer,
which runs SPIR-V on the same four cores the CPU backend already has. A
software device cannot show what a real one would do; what it can show is that
the shaders are right and what the interface costs.

yolo26n over a generated 320 picture, best of five: **CPU 164 ms, Vulkan
2751 ms** — 16.7x slower. The detections are identical to the digit printed,
and the head's 475,204 raw values agree to 2.44e-06 relative.

The 2.6B text checkpoint, one `logits` call on a six token prompt: **128,000
logits, max relative error 1.14e-06, and the top fifty in the same order**. The
run made 320 submissions and moved 196.7 MiB. Greedy `generate` on the same
checkpoint follows the CPU backend for all 24 tokens asked for — 123
characters, **identical character for character**, which is what a top-1 that
never disagrees looks like once it is fed back in.

Those two counts are the result worth keeping. **Both seams pass host
pointers** — `dense` is handed a `const float *src` and a `float *dst`,
`conv_run` two `YoloPlane`s over malloc memory — so there is nowhere in either
interface to say "the answer is already on the device, leave it there". Every
operation uploads its inputs and downloads its outputs. Weights are cached by
host pointer, because they are immutable after load and that is most of the
traffic; activations are not, because nothing in the seam says when the host
last wrote them. A 320 picture is 139 submissions and 46.4 MiB moved against
12 MiB of resident weights: the feature maps cross about four times each, out
of the convolution that made them, into the concatenation, out again, into the
next convolution.

What that supports is a statement about the interface and not about any
device: on a per-operation synchronous seam the submit-and-wait around a
dispatch can cost more than the arithmetic inside it, and the smaller the shape
the worse that is. Whether a large enough shape comes out ahead on real
hardware is **not measured and is not claimed**. Moving the seam to
device-resident activations is item 51, and it is a change to `app/core.c` and
`app/yolo.c` rather than to this file.

**No rate here is a GPU rate**, and none should be quoted as one. What a
discrete card does with these kernels is unmeasured and is marked **blocked**
in `TODO.md` for that reason, not estimated.

### That it is the same answer

It is not the same answer bit for bit, and it cannot be. The dense kernel folds
sixty-four partial sums through a tree where the CPU walks the row, and no
reordering of floating point addition is exact. What is claimed is that the two
agree to about a part in a million of the magnitude involved, which is what a
reordered sum costs and no more — 5.8e-07 on dense, 7.2e-07 on attention across
several chunks, 1.14e-06 over a whole 2.6B forward pass.

The CPU tables are untouched, and that is visible in the diff before it is
measured: **`app/core.c` has no change at all**, and every line added to
`app/yolo.c` sits inside `#ifdef YOLO_VULKAN`. A default build compiles the
same text it compiled before; the one unconditional change outside the new
file is a line of `--backend` help.

Measured anyway, because §5 of `AGENTS.md` asks for the binary and not the
argument. Against binaries built from the previous commit — `git archive HEAD
app/`, built aside — the default build's 2.6B `logits` row, its 48 token
greedy continuation, and yolo26n's dumped head rows are all **byte identical**.
A backend is selected, not applied: a run that does not say `--backend vulkan`
or `--device` is the run it was before this version, through the same
kernels.

### Code

One refusal is worth naming because it is a limit nothing here comes near and
a driver elsewhere might. A storage binding may not span more than
`maxStorageBufferRange`, and the floor Vulkan guarantees for that is 128 MiB.
Real desktop drivers report two to four gibibytes and SwiftShader reports one,
so the largest binding in any of this — the 2.6B checkpoint's vocabulary plane
at f32, 0.977 GiB — clears it by twenty-three thousandths. Reading past the
limit is undefined rather than an error, so it is checked and refused by name
instead; splitting a plane into bands is item 54.

One environment switch is worth naming because of what it is for. A discrete
card has device memory the host cannot see, so every buffer needs a staging
copy and a `vkCmdCopyBuffer`; an integrated part often reports a memory type
that is both device local and host visible, and there the staging copy is pure
loss because the pointer `vkMapMemory` returns is the memory the shader reads.
Both paths are in the file, and on a host whose only device is SwiftShader
everything is unified, so the staged one would never run. `ILL_VULKAN_STAGED`
makes the memory picker refuse the unified type, and the checks pass under it
as well as without it. What it costs is visible in the counters: the same 320
picture through yolo26n is 139 submissions unified and 343 staged, because
every weight that misses the cache now needs a submission of its own to get
its staging copy across.

`app/vulk.c`, new, eleven parts bottom-up: the Vulkan ABI this file declares
for itself and the loader that opens it by name; the device, its memory types
and the one command buffer everything is recorded into; the SPIR-V assembler;
the fourteen kernels; pipelines, descriptors and dispatch; residence and the
weight cache; opening and closing a device; one operation end to end;
the `IllBackend` table; the `YoloBackend` table; and the comparison in part 11,
which runs under `VULK_MAIN` as a standalone command and under `VULK_PROBE`
from the test suite, so the two cannot drift apart.

`app/main.c`: a guarded include and one call to `vulk_join` under
`ILL_VULKAN`. Registering the table does not touch a device — that happens in
`setup`, when a model is loaded against it — so it is safe to call on a host
with no Vulkan, and `--backend vulkan` reports the reason when there is none.

`app/yolo.c`: a guarded include under `YOLO_VULKAN`, a `--device NAME` flag,
and a line at the end of a run saying how many submissions it took and how many
bytes crossed.

`util/make.py`: `--vulkan`, which adds `ILL_VULKAN` to `app/main.c` and
`test/test.c`, `YOLO_VULKAN` to `app/yolo.c`, and `-ldl` where dlopen is not in
libc. No second build step and no probe for an SDK, because there is nothing to
probe for.

`AGENTS.md`, `README.md`, `GUIDE.md`, `TODO.md`: the new file and the second
argument for opening a list that says it is closed, what selects the backend,
and what is left.

### Tests

`test/test.c` gains a `vulkan` section under `ILL_VULKAN`, which is
`app/vulk.c`'s own comparison reported through this harness rather than a
second copy of it. Thirty-one checks: the registry takes the table and
reports it by name; every one of the six text operations and every one of the
nine picture operations against the CPU table over the same input; a padded
3x3, a strided, a depthwise and a grouped convolution separately, because they
are four different paths through one kernel; the convolution, the transposed
convolution and the short convolution again with **no bias**, which is the
path where the kernel must not read the one-float stand-in bound in a bias's
place; `conv_room` answering zero where the CPU's answers more; and four on
the weight cache — that a second call takes
the cached upload and reaches the same answer, that a plane of a different
width at the same address is not a hit, that the wider plane still reads
correctly afterwards, and that a weight an operation is already holding is not
evicted by the next weight the same operation asks for.

A host with no Vulkan device runs no comparisons and is not a failure: there is
nothing to compare against, so the suite says it skipped and moves on — 178
checks there rather than 206. The default build does not define `ILL_VULKAN` at
all and is unchanged at 175.

The whole matrix: native, portable, no-simd, debug and sanitize, each with and
without `--vulkan`, plus the no-device path and `ILL_VULKAN_STAGED`. Twelve
runs, nothing failing in any of them.

They bite, and the count says how many checks share a kernel:

| what was broken | checks that fail |
| --- | --- |
| dense drops the last column of every row | 5 |
| dense runs one token instead of all of them | 5 |
| the convolution ignores its pad | 5 |
| the short convolution never rolls its window | 4 |
| the convolution ignores the group offset | 2 |
| the transposed convolution reads its weight as [out, in, k, k] | 2 |
| rms norm forgets the gain | 1 |
| rope's sine changes sign | 1 |
| the streaming attention skips the rescale | 1 |
| attention drops the causal horizon | 1 |
| the pool takes a minimum | 1 |
| the gemm ignores the swap flag | 1 |
| the cache ignores the pin | 1 |

The rescale is the one that had to be earned. It failed nothing until the
window was widened past sixty-four, because a span that fits in one chunk never
rescales and the first version of that check did not — the check was broadened
because the mutation did not bite, which is what §6 asks the mutation for.

Two things the checks found and one they did not.

The cache keyed on the host pointer alone would hand a narrower plane the wider
plane's upload, which is why the byte count is part of the key and why there is
a check for it.

The second was a review finding rather than a failing check, and the check came
after: an operation that wants two weights — the short convolution's taps and
its bias, a convolution's sheet and its bias — looks both up before it
dispatches, and **the second lookup can be the miss that evicts the first**.
The caller is then holding a pointer to a slab whose buffer has been destroyed
and whose slot has been overwritten by whatever entry was last in the list. A
cache entry an operation is using is now pinned and is not a candidate for
eviction; `vulk_op` releases the pins once the dispatch has completed. The
check for it runs the short convolution with a budget too small to hold both of
its weights, which makes the eviction certain rather than likely. Ignoring the
pin fails it by 5.7e+36, which is what reading a destroyed buffer looks like.

A third was found the same way and has no check that can be relied on to bite.
A bias was read with a select — `has_bias ? bias[c] : 0` — and a select
evaluates both of its arms, so with no bias the load ran off the end of the
one-float buffer bound in its place. Reading past a storage buffer is undefined
without `robustBufferAccess`, which this file does not ask a device to enable,
so the read had to not happen rather than be harmless: it is a branch now. The
no-bias path is covered by three new checks, and **putting the select back
fails none of them** on SwiftShader, which reads past that buffer without
complaint. That is the honest state of it: the fix rests on what the
specification says is undefined, not on a number, and no check on this host can
be made to stand behind it. A driver that faulted or returned rubbish would
have made the same fix a bug report.

And replacing the running maximum with the chunk's own maximum fails nothing —
that turned out to be correct rather than a gap, because the streaming softmax
is exact for any pivot so long as the rescale uses the same one; it changes the
numerical range and not the answer.

**206 pass**, 175 before.
