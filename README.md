# illlm

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
run.py        workflows: install, build, test, run
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
provides. Nothing else in the project does.

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
