# GUIDE

A tour of the implementation: what the engine computes, how the code is
arranged, and where to change it.

- [1. The shape of the project](#1-the-shape-of-the-project)
- [2. What the architecture computes](#2-what-the-architecture-computes)
- [3. The public API](#3-the-public-api)
- [4. Inside core.c](#4-inside-corec)
- [5. The command line](#5-the-command-line)
- [6. How correctness is established](#6-how-correctness-is-established)
- [7. Where the time goes](#7-where-the-time-goes)
- [8. Extending the engine](#8-extending-the-engine)
- [9. Naming conventions](#9-naming-conventions)

---

## 1. The shape of the project

Five source files, flat, no build system, plus the published checkpoints in
`ckpt/`.

| file | lines | role |
| --- | --- | --- |
| `app/core.c` | ~4500 | the engine: a header and its implementation in one file |
| `app/main.c` | ~670 | the command line, six verbs |
| `test/test.c` | ~800 | unit tests over the engine internals |
| `test/test.py` | ~780 | comparison against Hugging Face `transformers` |
| `util/make.py` | ~280 | install, build, test, run, bench, clean |
| `util/tune.py` | ~1700 | fine tuning on the reference side, the caveman rule engine, and the heretic abliteration pass |
| `data/tune.jsonl` | 202 rows | the tuning corpus |
| `ckpt/` | — | the published checkpoints: `ckpt/lfm2.5-0.4b` (LFM2.5-350M), `ckpt/lfm2.5-1.2b-i` (LFM2.5-1.2B-Instruct), `ckpt/lfm2.5-2.6b` (LFM2.5-2.6B) |

`app/main.c` and `test/test.c` each begin with `#include "core.c"`. That is
deliberate: the project has no header file, so the engine carries its own
interface at the top of its implementation, guarded by `ILL_CORE_INCLUDED`.
Each front end is a single translation unit, which lets the compiler inline the
whole engine and lets the tests reach the internals they are testing.

Build any of it with one command:

```sh
cc -std=c11 -O3 -march=native app/main.c -o app_main -lm -pthread
```

`util/make.py build` does exactly that, after probing which flags the compiler
accepts.

---

## 2. What the architecture computes

The Liquid architecture — `model_type: "lfm2"` — is a hybrid stack. Every layer
has the same two-step shape:

```
h  <-  h + operator(rmsnorm(h, operator_norm))
h  <-  h + swiglu(rmsnorm(h, ffn_norm))
```

What differs is the operator, which the checkpoint chooses per layer through
`layer_types` (or the older `full_attn_idxs`).

### The attention operator

Grouped query attention with rotary positions, and one detail that is easy to
miss: **each query head and each key head is RMS normalised before the
rotation**, using a gain of `head_dim` values shared across heads.

```
q = rmsnorm(q_proj(x)  reshaped to heads,      q_layernorm)
k = rmsnorm(k_proj(x)  reshaped to key heads,  k_layernorm)
v = v_proj(x)
q, k = rope(q, k, position)
y = out_proj( softmax(q k^T / sqrt(head_dim), causal) v )
```

There are no biases anywhere in the block, and no separate `o_proj` — the
output projection is named `out_proj`.

### The convolution operator

The liquid short convolution. One projection produces three stacked halves; two
of them gate a depthwise causal convolution over the third.

```
B, C, x = split(in_proj(h), 3)          # each model_dim wide
y = conv1d_causal(B * x, taps)          # depthwise, conv_L_cache taps
y = out_proj(C * y)
```

The convolution is causal and zero padded, so with `L` taps

```
y[t][c] = sum_{j<L} taps[c][j] * u[t - (L-1) + j][c]        u = B * x
```

and positions before the start of the sequence read as zero. Between calls the
last `L - 1` values per channel are the entire recurrent state. That is what
makes these layers cheap: no cache that grows with context.

### The rest

- **Feed forward**: `w2( silu(w1 x) * w3 x )`. The engine reads the inner width
  from the shape of `w1` rather than recomputing the `block_auto_adjust_ff_dim`
  rule, so a checkpoint that disagrees with the formula still loads.
- **Norms**: RMS norm throughout — `x * rsqrt(mean(x^2) + eps) * gain` — reduced
  in f32, exactly as the reference does it.
- **Rotary**: the half-rotation form. `cos` and `sin` are indexed modulo
  `head_dim / 2`, and element `i` pairs with element `i + head_dim / 2`.
- **Output**: a final `embedding_norm`, then the output projection, which is
  tied to the embedding table unless the checkpoint ships `lm_head.weight`.

### What the two operators cost

For a 32-layer, 2560-wide model, per token:

| | parameters read | state per token |
| --- | --- | --- |
| attention layer | 16.4 M | `2 * kv_heads * head_dim` floats, grows with context |
| convolution layer | 26.2 M | `(L - 1) * model_dim` floats, fixed |
| feed forward | 62.9 M | none |

The feed forward dominates the arithmetic; the attention layers dominate the
memory that grows.

---

## 3. The public API

Four objects and about thirty calls. Everything is in part 1 of `app/core.c`.

```c
IllModel     weights, architecture, vocabulary — read only once loaded
IllState     one sequence: caches and every scratch buffer a step touches
IllVocab     the tokenizer
IllSampler   the token choice policy, plus the history a penalty needs
```

A whole session:

```c
IllPlan   plan;
IllModel *model;
IllState *state;
float    *logits;

ill_plan_init(&plan);
plan.model_path  = "path/to/LFM2.5-2.6B";
plan.weight_type = ILL_TYPE_Q8;          /* or ILL_TYPE_KEEP */
ill_model_load(&model, &plan);
ill_state_make(&state, model, 4096);

IllBatch batch = { prompt_ids, prompt_len, 0 };
ill_model_apply(model, state, &batch, &logits);
/* logits now points at one row of vocab_size floats, owned by the state */

ill_state_free(state);
ill_model_free(model);
```

Rules that hold everywhere:

- Every call that can fail returns `IllResult`; `ill_result_text` names it.
- Every constructor takes its output by pointer and leaves it `NULL` on
  failure, so an error path never frees a half built object.
- `ill_model_apply` appends to the sequence held in the state. Set
  `batch.every` to get a row of logits per token instead of one for the last.
- The returned `logits` pointer is engine owned and stays valid until the next
  call on that state.
- A model is read only after loading, so several states may share one model.
  A state belongs to one thread at a time.
- Buffer sized calls — `ill_vocab_encode`, `ill_vocab_decode`,
  `ill_vocab_prompt` — always report the full requirement, so calling once with
  a limit of zero sizes the buffer and calling again fills it.

---

## 4. Inside core.c

Fifteen parts, each layered on the ones above it.

### part 1 — public interface

Types, result codes, and prototypes. Nothing else in the file appears here, so
this section is the contract.

### part 2 — platform layer

Everything the engine wants from the operating system, in one place: aligned
allocation, whole-file mapping, a monotonic clock, a core count, and a thread
pool. Three ports — POSIX, Windows, and a fallback that reads files with
`fopen` and runs single threaded — so the rest of the file never sees an
`#ifdef` for the host.

The thread pool deserves a note. A decode step issues a few hundred parallel
regions, so a mutex round trip per region would dominate the step. Instead the
pool runs one task at a time and hands it out through a rising epoch counter:

- The caller writes the task, resets the claim and done counters, then bumps
  the epoch with a release store.
- Workers spin on the epoch with an acquire load, which is what publishes the
  task fields.
- Each worker, and the calling thread, claims chunks with an atomic
  fetch-and-add until the chunks run out, so uneven work self balances.
- The caller spins until every worker has reported done, then returns.

Workers back off from `pause` to `sched_yield` after a spin budget. If the
toolchain has no C11 atomics, `ILL_WITH_THREADS` is zero and `ill_pool_fork`
runs the chunks inline — the same code, one lane.

### part 3 — number formats

f16 and bf16 conversion in both directions, written by hand so that no
compiler intrinsic is required and so the round trips are testable. Widening is
exact; narrowing rounds to nearest even.

### part 4 — vector vocabulary

The one idea that keeps the kernels short. A dozen inline functions — `zero`,
`wide`, `load`, `save`, `add`, `mul`, `fma`, `sum`, `abs`, `max`, `top`, plus
`bf16` and `f16` loaders — are defined four times, once per target:

| target | `ILL_VW` | type |
| --- | --- | --- |
| AVX-512 | 16 | `__m512` |
| AVX2 + FMA | 8 | `__m256` |
| NEON | 4 | `float32x4_t` |
| plain C | 1 | `float` |

Every kernel below is written once against this vocabulary. The width-1
instantiation is not a separate reference implementation that could drift — it
is the same source, compiled with a vector width of one. `make.py test --no-simd`
runs the full suite through it, and `make.py test --portable` runs it at width 8,
so agreement between widths is checked rather than assumed.

### part 5 — block quantisation

q8 is symmetric per-block: 32 signed bytes share one f32 scale. Blocks run along
the input dimension, so a dot product walks weight scales and activation scales
in the same order.

The dot product accumulates integer products into a *float* vector scaled per
block, rather than reducing each block on its own:

```
acc = fma( cvt_f32(madd_i16(w_block, a_block)), scale_w * scale_a, acc )
```

That leaves exactly one horizontal reduction per output row. On AVX-512BW a
32-value block widens to precisely one register of int16, so a block costs a
single multiply-add.

### part 6 — json reader

A compact DOM. Every node is an index into one flat array and every string is
an offset into one text pool, so a document frees with two calls. `\uXXXX`
escapes fold to UTF-8, surrogate pairs included. Trailing commas are rejected;
this is JSON, not a configuration language.

It is used three times: `config.json`, the safetensors headers, and
`tokenizer.json` — which for a 65k vocabulary is several megabytes and the
reason the reader is written to allocate in geometric steps rather than per
node.

### part 7 — safetensors store

Maps every shard and indexes the tensors inside them. Shards are found by
`model.safetensors.index.json` if present, then by `model.safetensors`, then by
scanning the folder for `*.safetensors` in sorted order.

Nothing is copied. A slab is a pointer into the mapping, so a 5 GiB checkpoint
costs address space and page cache rather than a 5 GiB read — which is why
loading bf16 weights takes about a tenth of a second.

### part 8 — compute kernels

`IllPlane` is a linear weight: cells, optional q8 scales, a format, and the two
extents. Every kernel takes planes and f32 activations.

The one that decides throughput is `dense`. It streams a weight row once and
fuses it against up to four activation rows:

```
for each output row r:
    for each column block j:
        w = load(W[r] + j)
        for t in 0..T-1:
            acc[t] = fma(w, load(X[t] + j), acc[t])
```

`T` is a literal at every instantiation — the body is a macro expanded for
T = 1, 2, 3, 4 and for each stored format — so the accumulators stay in
registers rather than spilling to an array. Tiling matters: at T = 1 a 256
token prefill would stream the whole model 256 times. Wider tiles were measured
and were slower, because four activation rows of a 2560-wide model already fill
L1.

The rest are direct: RMS norm reduced in f32, SwiGLU, softmax with the standard
peak subtraction, a fused multiply-add accumulate for the attention value mix,
and a plain dot product.

### part 9 — backend seam

The forward pass never calls a kernel. It calls through `IllBackend`:

```c
setup   close   width
dense   rmsnorm  swiglu   rope   attend   conv1d
```

Nine function pointers and an opaque `inner`. The CPU backend fills them with
the part 8 kernels spread over the thread pool, splitting `dense` over output
rows, `attend` over `(token, head)` pairs, `conv1d` over channels, and the
elementwise stages over tokens. Below `ILL_FORK_FLOOR` multiply-adds it runs
inline, because the handshake would cost more than the work it spreads.

`ill_backend_join` registers another implementation; `--backend NAME` selects
it. That is the whole extension point.

### part 10 — model load

Three passes. Read `config.json` into an `IllArch`. Bind every tensor into an
`IllPlane` or, for the small ones, widen it to an owned f32 vector. Optionally
repack the large planes to q8, in parallel across the thread pool.

Two habits keep this section honest:

- Extents are checked against the checkpoint, and a mismatch names the tensor
  and both shapes rather than failing silently.
- Names are tried with and without the `model.` prefix, so a bare `Lfm2Model`
  export loads as readily as an `Lfm2ForCausalLM` one.

Every allocation the model owns goes through `ill_model_own`, which appends it
to one list. `ill_model_free` walks that list. There is no other ownership rule
to remember, which is why the error path can be a single `goto undo`.

### part 11 — model state

One sequence. It holds what carries between steps:

- the key/value cache, laid out `[layer][kv head][position][head_dim]` so one
  head's keys are contiguous across positions, which is the order the attention
  dot product walks them in;
- the convolution window, `[layer][channel][L-1]`;

and every scratch buffer the forward pass touches, sized once from the batch
width, so a step allocates nothing.

`ill_state_crop` rewinds the sequence. It reports `ILL_STATE` rather than
guessing when a model has convolution layers: their window is recurrent, and a
partial rewind cannot be undone without replaying the sequence.

### part 12 — forward pass

The stack itself, and it reads like the equations in section 2. Chunking lives
here: `ill_model_apply` splits a batch into pieces of at most `batch_span` and
calls `ill_stack_run` on each, asking for logits only where the caller wants
them, so a long prompt never materialises a logit row it will not use.

Rotary tables are computed per chunk, not cached for the whole window: 256
positions of `head_dim` floats is small, and computing the phase as
`fmod(position * inv_freq, 2 pi)` in double keeps precision at position 100,000
where a float phase would have lost several bits.

### part 13 — vocabulary

Byte level BPE read straight from `tokenizer.json`. Three tables do the work:
pieces indexed by id, a piece-to-id map, and a merge map keyed by the *pair of
ids* being joined. Because every merge result is itself a vocabulary entry, the
inner loop never touches a string — it walks ids.

Encoding runs in four stages:

1. **added tokens** — matched literally, longest first, splitting the input.
2. **pre-tokenization** — the GPT-2 or Llama-3 alternation, transcribed in
   order rather than run through a regex engine. Which one is detected from the
   `Split` pattern in `tokenizer.json`.
3. **byte encoding** — each byte becomes one symbol through the GPT-2 alphabet,
   looked up once at load into a 256-entry table of ids.
4. **merging** — a binary heap of candidate pairs over a doubly linked list of
   symbols. Popped candidates are validated against the current ids before
   being applied, which is what makes stale heap entries harmless. This keeps
   long unbroken runs — base64, a long identifier — near linear rather than
   quadratic.

Character classes are the one approximation in the engine, and it is a
deliberate one: carrying the Unicode database would dwarf the rest of the file,
so runes below `0x80` use the exact ASCII rule and runes above it are letters
unless they fall in a listed range of spaces, punctuation, or symbols. The
ranges cover the blocks that appear in ordinary prose. `TODO.md` records the
gap.

Chat prompts are shaped by detecting `<|im_start|>` and `<|im_end|>` in the
vocabulary rather than by evaluating the Jinja `chat_template`, which would mean
carrying a template engine. `--raw` bypasses shaping entirely.

### part 14 — sampler

Filters in the order that keeps each one meaningful: repetition penalty on raw
logits, then temperature, then the candidate cuts — top-k, then min-p, then
top-p — then one draw from what survives. Temperature zero short circuits to an
argmax without sorting. The generator is splitmix64, so a seed replays exactly
on any host.

### part 15 — facade

Result names. Short by design: if this section were long, the API would be
wrong.

---

## 5. The command line

```
app_main info      describe the checkpoint and the load plan
app_main tokens    encode --prompt, or decode --tokens
app_main generate  continue a prompt and stream the completion
app_main chat      interactive conversation on stdin
app_main logits    write logits, the hook test/test.py compares against
app_main bench     time prefill and decode
```

`app_main help` lists every flag. Three are worth knowing:

- `--quant q8` repacks at load: half the memory, roughly double the decode rate.
- `--raw` feeds the prompt verbatim instead of shaping a chat turn around it.
- `--tokens 1,2,3` supplies ids directly, which is how the engine is exercised
  against a checkpoint whose tokenizer it cannot read.

`logits` also takes `--every` for a row per token and `--stream --prefill N` to
feed the first `N` tokens as one batch and the rest one at a time. That second
mode exists so the test suite can check the caches rather than only the maths.

---

## 6. How correctness is established

Three layers, each catching what the others cannot.

**`test/test.c` — 73 unit checks.** Number formats against their definitions;
the JSON reader against nested documents, escapes, surrogates, and eight
malformed inputs; every kernel against a plain-C restatement of the same
arithmetic written independently in the test; rotary, attention, and
convolution against direct transcriptions of their equations, including the
convolution window carried across calls; the thread pool for exact-once
execution over many widths and repeated forks; the pre-tokenizer chunk by
chunk; the merge heap; and the sampler for seed replay, nucleus containment,
and repetition demotion.

**`test/test.py` — 19 comparisons against `transformers`.** Small Liquid
checkpoints are built with random weights and run through both implementations.
The matrix covers attention-only, convolution-only, and hybrid stacks; grouped
and multi query attention; tied and untied heads; wide and biased kernels;
non-round widths; f32, f16, and bf16 storage; prefill chunked at widths 1, 3, 7,
and 64; streaming decode from four different prefill lengths; 300 positions; one
thread against four; q8 repacking judged on agreement rather than distance; and
tokenizer agreement over twenty awkward strings for both split flavours, plus a
decode round trip.

Random initialisation leaves every norm gain at 1.0, which would hide a gain
loaded from the wrong tensor, so the fixture randomises them. Token id 0 is
avoided because the reference zeroes the padding embedding, which drives the
whole stack to zero and would hide real differences.

Against f32 checkpoints the engine matches to 2e-7 relative — float32 rounding.

**Sanitizers.** `make.py test --sanitize` builds with AddressSanitizer and
UndefinedBehaviorSanitizer. Both suites run clean, including leak detection.

A real checkpoint is tested the same way: `make.py test --model PATH` adds a
logits comparison and a tokenizer comparison against it. The published
checkpoints ship in `ckpt/` (`ckpt/lfm2.5-0.4b`, `ckpt/lfm2.5-1.2b-i`, `ckpt/lfm2.5-2.6b`),
with `ckpt/lfm2.5-0.4b` (LFM2.5-350M) the default, so this runs by default, and it adds
two
further checks the synthetic suite cannot make:

- **throughput** — the engine's `bench` beside transformers doing the same
  shape of work: one batch of 256 tokens, then 64 single-token steps with the
  cache carried, both at bf16 and the same thread count. Quoting the engine's
  q8 against the reference's bf16 would fold a weight width difference into
  what looks like an engine difference.
- **greedy behaviour** — continuations from the same ids under identical greedy
  settings, compared on the shared prefix of ids. The comparison is on ids, not
  on text: `generate` emits decoded text, and re-encoding text to recover ids
  is not a round trip — the tokenizer can segment the same string differently —
  so re-encoding would report divergence the engines never produced. Where a
  greedy chain parts is one draw from a lottery on both sides — the reference
  primed one token at a time parts from itself somewhere too — so the engine is
  allowed to follow half as far as the reference follows itself before it is
  called wrong.

A missing checkpoint folder is a skip, not a failure: the synthetic suite is
the parity argument and runs without it.

---

## 7. Where the time goes

Decode is memory bound. One token reads every weight once, so the rate is
bytes divided by achievable bandwidth, and nothing else matters much. That is
why q8 nearly doubles it and why threads help until bandwidth saturates.

Prefill is arithmetic bound. `2 * parameters * tokens` FLOPs, and the tile in
`dense` decides how close to peak you get.

On the 2.9B synthetic checkpoint, four cores, AVX-512:

| | weights | prefill | decode |
| --- | --- | --- | --- |
| bf16 | 5.44 GiB | 38.0 tok/s | 5.2 tok/s |
| q8 | 3.06 GiB | 37.1 tok/s | 9.0 tok/s |

Roughly 53 GB/s of effective weight traffic in prefill and 28 GB/s in decode,
and about 230 GFLOP/s — both near what four cores of this class sustain. Decode
scales 2.4 → 4.6 → 8.9 tok/s over one, two, and four threads.

If you are profiling a change, the order of what to look at is: the `dense`
inner loop, then the attention score loop at long context, then everything else
together.

---

## 8. Extending the engine

### A new backend

Fill an `IllBackend`, call `ill_backend_join(&mine)` before `ill_model_load`,
and pass `--backend mine`. The nine entry points are listed in section 4. Weight
planes point at mapped host memory, so a device backend uploads them in `setup`
and keeps the device handles in `inner`.

The seam was drawn where it is because these six operations are the only
arithmetic the stack performs, and each is large enough that a per-call
dispatch is free.

### A new weight format

Add the enum value, a loader in the vector vocabulary, a case in
`ILL_DENSE_TYPED`, a case in `ill_plane_row`, and a branch in
`ill_model_plane`. Nothing else knows the difference.

### A new tokenizer family

`ill_vocab_read` reports `ILL_VOCAB` for anything that is not byte level BPE
rather than approximating it. A Unigram or WordPiece model would slot in beside
it as another `model.type` branch with its own encode path; the piece tables and
the added-token scan are already shared.

### A new architecture in the family

`IllArch` and the per-layer `IllBlock` are the two structures to widen. A new
operator is a third `kind` in `IllBlock`, a branch in `ill_stack_run`, and its
own cache in `IllState`. The two operators already there are the pattern.

---

## 9. Naming conventions

They are mechanical, so the code reads at an even pace.

- **Types** are `Ill` plus a short noun: `IllModel`, `IllState`, `IllPlane`,
  `IllBlock`, `IllBatch`, `IllVocab`, `IllPlan`, `IllPool`.
- **Functions** are `ill_<module>_<verb>`. The module is the noun the call acts
  on; the verb comes from a small fixed set so opposites pair by shape:
  `make`/`free`, `open`/`close`, `load`/`free`, `setup`/`close`, `push`/`pull`,
  `note`/`wipe`, `find`/`name`.
- **Fields** are one or two short words of similar weight, and related fields
  rhyme: `model_dim` and `inner_dim`; `head_count` and `group_count`;
  `layer_count`, `attn_count`, `conv_count`; `begin_token` and `end_token`;
  `fill` and `span`; `gate`, `rise`, `fall` for the three feed forward planes.
- **Result codes** are one word: `ILL_OK`, `ILL_ARGS`, `ILL_ALLOC`, `ILL_FILE`,
  `ILL_PARSE`, `ILL_SHAPE`, `ILL_MODEL`, `ILL_VOCAB`, `ILL_LIMIT`, `ILL_STATE`.
- **Backend operations** keep their standard names — `dense`, `rmsnorm`,
  `swiglu`, `rope`, `attend`, `conv1d` — because an implementer should
  recognise them on sight, and consistency of form is the rhythm that matters
  at a seam.

Comments explain the decision, not the statement. If a line needs a comment to
say what it does, it is the line that should change.
