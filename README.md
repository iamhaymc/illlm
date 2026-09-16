# INFERLIQU

An inference engine for the Liquid language architecture, written in pure C
with no dependencies.

It runs a Hugging Face checkpoint directly — `config.json`, the
`*.safetensors` shards, and `tokenizer.json` — with nothing to convert and
nothing to install. One C file is the engine, one is the command line, one is
the test suite. A C compiler is the whole toolchain.

```
app_core.c    the engine, with a clean public API
app_main.c    the command line
app_test.c    unit tests for the engine internals
app_test.py   comparison against the reference implementation
app_tune.py   fine tuning on the reference side, the caveman rule engine,
              and the heretic abliteration pass
datasets/data_tune.jsonl  the tuning corpus, 202 rows
run.py        workflows: install, build, test, run
model/        the published LFM2.5-2.6B checkpoint, used by the test suite
GUIDE.md      a tour of how it all works
CHANGES.md    what was built and why
TODO.md       what is still open
```

## Quickstart

```sh
python3 run.py build                                   # a few seconds, no dependencies
./build/app_main info     --model path/to/LFM2.5-2.6B
./build/app_main generate --model path/to/LFM2.5-2.6B --prompt "Write a haiku about rivers."
./build/app_main chat     --model path/to/LFM2.5-2.6B
```

Add `--quant q8` to halve the memory and roughly double decode speed. Add
`--threads N` to pick a worker count; the default is the host's core count.

```sh
python3 run.py test                                    # unit tests plus reference comparison
python3 run.py bench --model path/to/LFM2.5-2.6B       # prefill and decode throughput
python3 run.py run -- generate --model DIR --prompt "hello"
```

`run.py test` needs `torch` and `transformers`, which `python3 run.py install`
provides. The engine itself has no dependencies at all. The published
LFM2.5-2.6B checkpoint ships in `model/`, so the checkpoint suite — logits,
tokenizer, throughput, and greedy behaviour against the reference — runs by
default; `--no-checkpoint` skips it, and `--model PATH` points somewhere else.

Compare like with like: the throughput check runs both sides at bf16 and the
same thread count, so put `run.py bench` at bf16 too rather than quoting it
against `--quant q8`.

## What it runs

The Liquid architecture (`model_type: "lfm2"`) is a hybrid stack. Each block
normalises, applies an operator, and adds the result back to the residual
stream; then normalises again and adds a SwiGLU feed forward. The operator is
chosen per layer by the checkpoint:

- **attention** — grouped query attention with rotary positions, and an RMS
  norm applied to each query and key head before the rotation.
- **convolution** — the liquid short convolution: one projection splits into
  three, two of them gate a depthwise causal convolution over the third.

The engine reads the layer plan, the head counts, the kernel width, and the
feed forward width out of the checkpoint, so it runs any model of this family,
not one set of dimensions.

| capability | supported |
| --- | --- |
| stored weight formats | f32, f16, bf16 |
| runtime weight formats | as stored, or repacked to q8 |
| tokenizer | byte level BPE from `tokenizer.json`, GPT-2 and Llama-3 splits |
| sampling | greedy, temperature, top-k, top-p, min-p, repetition penalty |
| threading | POSIX threads, Windows threads, or single threaded |
| vector width | AVX-512, AVX2, NEON, or plain C — same source at every width |

## Correctness

`app_test.py` builds small Liquid checkpoints with random weights, runs them
through both this engine and Hugging Face `transformers`, and compares logits
position by position. It covers attention-only, convolution-only, and hybrid
stacks; grouped and multi query attention; tied and untied output heads; wide
and biased convolution kernels; f32, f16, and bf16 storage; chunked prefill and
single token decode; and tokenizer agreement over a corpus of awkward strings.

Against float32 checkpoints the engine matches the reference to **2e-7
relative**, which is float32 rounding. `app_test.c` adds 73 unit checks over the
internals. Both suites run clean under AddressSanitizer and UndefinedBehaviorSanitizer.

## Measured

A synthetic 2.9B-parameter checkpoint of LFM2-2.6B proportions — 32 layers,
model dim 2560, feed forward 8192, 32 query heads over 8 key-value heads,
vocabulary 65536 — on four x86-64 cores with AVX-512:

| | weights | prefill | decode |
| --- | --- | --- | --- |
| bf16, as stored | 5.44 GiB | 38.0 tok/s | 5.2 tok/s |
| q8, repacked at load | 3.06 GiB | 37.1 tok/s | 9.0 tok/s |

Decode scales 2.4 → 4.6 → 8.9 tok/s across one, two, and four threads. Loading
bf16 costs about a tenth of a second because the weights are memory mapped and
never copied; repacking to q8 costs about three seconds once.

## Tuning

The engine has no trainer. A tune happens on the reference side and comes back
as a checkpoint the engine reads unchanged: `app_tune.py` trains a LoRA adapter
over the frozen base, folds it into the float weights, and writes
`build/tune/merged` in the same layout as `model/`.

What it tunes for is **caveman**, the compression register described by the
skill at <https://github.com/JuliusBrussee/caveman>: drop articles, filler,
hedging and pleasantries, and keep every technical fact, every number, every
negation and every byte of code. LFM2.5 is a thinking model — its chat template
ends every generation prompt with `<think>` — so the tune trains the reasoning
span as well as the answer, at separate intensities. The level is a system
prompt, so the register is a knob rather than a change of voice:

| | thought | answer | whole reply |
| --- | --- | --- | --- |
| `Normal mode.` | 19 | 113 | 132 tokens |
| `Caveman mode: lite.` | 21 | 37 | 58, 2.3x off |
| `Caveman mode: full.` | 28 | 26 | 54, 2.4x off |
| `Caveman mode: ultra.` | 10 | 12 | 22, 6.0x off |

One question — "Explain database connection pooling" — carried at all four
levels in the corpus, counted with the checkpoint's own tokenizer.

```sh
python3 app_tune.py --lint                       # audit the corpus
python3 app_tune.py --model model --uncensor     # abliterate into build/tune/uncensored
python3 app_tune.py --model model --train        # LoRA adapter into build/tune
python3 app_tune.py --model model --merge        # fold it into build/tune/merged
python3 app_tune.py --model model --check --tuned build/tune/merged
```

`--check` is the number that says whether it worked: it runs the base and the
tuned checkpoint over the held out rows and reports how far the output shrank
beside how many answers survived. `--make-data` presses an existing reasoning
dataset into the register by deleting function words only, and throws away any
row where a guarded span, a number or a negation moved — a transform that can
only delete cannot introduce a claim the source did not make.

`--uncensor` is the one step that is not a tune. It runs
[heretic](https://github.com/p-e-w/heretic) over the checkpoint — no gradient
step and no corpus, but a low rank edit subtracting the direction the residual
stream moves in when the model is about to refuse — and writes the decensored
weights to `build/tune/uncensored` in the same layout as `model/`, so the engine
reads them unchanged. It needs `pip install heretic-llm`, and it needs a card:
the search scores a hundred generations and a hundred forward passes per trial,
two hundred trials by default.

Nothing is asked while it runs. Heretic is interactive at the end — which point
of the refusals-against-divergence front to keep, and what to do with it — and
those prompts are answered from the script: the fewest refusals among the trials
at or under `--uncensor-kl` (0.25 of divergence from the base by default), saved,
exit. An interrupted run resumes from `build/tune/uncensor-study` rather than
starting again, and `--uncensor-fresh` throws that away instead;
`--uncensor-trials` shortens the search, `--uncensor-quant bnb_4bit` loads the
weights 4-bit for a smaller card, and `--uncensor-out` writes somewhere else.
The steps chain, so the whole thing is one command:

```sh
python3 app_tune.py --model model --uncensor --train --merge
```

The adapter then trains over the decensored weights rather than over the base,
which is what will be served. Two settings are this checkpoint's rather than
heretic's, and `app_tune.py`'s header says why: the response prefix is
`</think>`, because LFM2.5's template has already opened the think block and
refusals would otherwise be counted over reasoning text; and the divergence
heretic balances its two objectives at follows the export cap, so the search
spends its trials in the band a trial can be taken from. No abliteration has
been run over the 2.6B weights yet — that needs a card, and `TODO.md` carries
the item.

## Extending it to an accelerator

The forward pass never calls a kernel directly. It calls through `IllBackend`,
a table of the six shapes of work the stack performs — `dense`, `rmsnorm`,
`swiglu`, `rope`, `attend`, `conv1d` — plus `setup`, `close`, and `width`. The
CPU backend fills that table with threaded kernels. A device backend fills the
same table and keeps its queue in `inner`. Nothing above the seam changes.

See `GUIDE.md` for the full tour, and `TODO.md` for what is still open.

## Using the engine as a library

`app_core.c` is a header and its implementation in one file. Include it once:

```c
#include "app_core.c"

IllPlan plan;
IllModel *model;
IllState *state;
float *logits;

ill_plan_init(&plan);
plan.model_path = "path/to/LFM2.5-2.6B";
ill_model_load(&model, &plan);
ill_state_make(&state, model, 4096);

IllBatch batch = { tokens, count, 0 };
ill_model_apply(model, state, &batch, &logits);
```

`build/` is generated by `run.py`; it is not part of the source tree.
Licensed under the terms in `LICENSE`.
