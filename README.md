<table>
<tbody>
<tr>
<td><img src="./docs/brand/logo_3.svg" width="64px"/></td>
<td><h1 style="color:#123;">INFERLIQ</h1></td>
</tr>
</tbody>
</table>

Two inference engines, written without dependencies in pure and optimized C11:
one for the Liquid language architecture, one for yolo26.

Both read the published checkpoints directly — a Hugging Face folder and an
Ultralytics `.pt` — with no conversion step (the checkpoints are included in
the repository).

## ↘️✴️ QUICKSTART

### ➖ CLI

```sh
python3 util/make.py build                             # a few seconds, no dependencies
./build/app_main info      --model ckpt/lfm2.5-2.6b-a
./build/app_main generate  --model ckpt/lfm2.5-2.6b-a --prompt "Write a haiku about rivers."
./build/app_main chat      --model ckpt/lfm2.5-2.6b-a
```

Add `--quant q8` to halve the memory and roughly double decode speed, or
`--quant q4` to halve it again — 1.57 GiB for the 2.6B, and 1.23x q8's decode.
Read what q4 costs before choosing it: on ordinary prose, nothing a perplexity
can see; on text the model should find easy, most of its confidence. Add
`--draft 4` to greedy decoding to verify four context-drafted tokens in each
pass, which is worth about a quarter on work whose answer quotes its question
and costs nothing when it does not.
Add `--threads N` to pick a worker count; the default is the host's core count.

### ➖ PICTURES

`app/yolo.c` is the second engine: it runs a yolo26 checkpoint from the `.pt`
Ultralytics publishes, which is a zip holding a pickled module tree and its raw
fp16 weights. There is no export step and no Python in the path.

```sh
./build/app_yolo ckpt/yolo26/yolo26n.pt photo.jpg
./build/app_yolo ckpt/yolo26/yolo26n-seg.pt photo.jpg --draw out.png
./build/app_yolo ckpt/yolo26/yolo26n.pt photo.jpg --nms-free
./build/app_yolo ckpt/yolo26/yolo26x-pose.pt photo.jpg --conf 0.4
./build/app_yolo ckpt/yolo26/yolo26n-cls.pt photo.jpg
```

All five tasks run: detection, segmentation with per-instance masks, pose with
keypoints, oriented boxes, and classification. Every size letter — n, s, m, l,
x — loads through the same code, because the channel counts are read from the
checkpoint rather than re-derived from a scale table.

yolo26 ships two detection heads. The one-to-many head needs suppression and is
what `yolo predict` uses; `--nms-free` reads the one-to-one head instead, which
is the NMS-free path the architecture is known for. Pictures come in through
stb, so every format it reads is a format the engine reads; `-DYOLO_NO_STB`
builds without it and reads binary PNM only.

The results match ultralytics 8.3.222 to every digit it prints, given the same
decoded pixels — see **Comparison** below.

```c
#include "yolo.c"

YoloModel *model; YoloSession *session; YoloOptions options;
YoloImage image; const YoloResult *result;

yolo_model_open("ckpt/yolo26/yolo26n.pt", &model);
yolo_options_default(model, &options);
yolo_session_open(model, &options, &session);
yolo_image_read("photo.jpg", &image);
yolo_session_run(session, &image, &result);

for (int i = 0; i < result->box_count; i++) {
    const YoloBox *box = &result->box_list[i];
    printf("%s %.3f  %.1f %.1f %.1f %.1f\n",
           yolo_model_class_name(model, box->class_index), box->score,
           box->left, box->top, box->right, box->bottom);
}
```

Every coordinate a result carries is in the source picture's pixels, not the
letterboxed input's. A session holds the scratch one picture needs and
allocates nothing after the first, so a run over a stream of pictures does not
touch `malloc`.

```sh
python3 util/make.py test                                 # unit tests plus reference comparison
python3 util/make.py bench --model ckpt/lfm2.5-2.6b-a     # prefill and decode throughput
./build/app_main perplexity --model ckpt/lfm2.5-2.6b-a --quant q8 < some.txt
python3 util/make.py run -- generate --model DIR --prompt "hello"
```

`make.py test` needs `torch` and `transformers`, which `python3 util/make.py install`
provides. The engine itself has no dependencies at all. The published
checkpoints ship in `ckpt/` — `ckpt/lfm2.5-2.6b-a` (a 2.6B causal model), beside an
encoder and a vision-language checkpoint this engine does not run — so the checkpoint
suite — logits, tokenizer, throughput, and greedy behaviour against the
reference — runs by default against `ckpt/lfm2.5-2.6b-a`; `--no-checkpoint` skips it,
and `--model PATH` points somewhere else.

Compare like with like: the throughput check runs both sides at bf16 and the
same thread count, so put `make.py bench` at bf16 too rather than quoting it
against `--quant q8`.

### ➖ API

`app/core.c` is a header and its implementation in one file. Include it once:

```c
#include "core.c"

IllPlan plan;
IllModel *model;
IllState *state;
float *logits;

ill_plan_init(&plan);
plan.model_path = "path/to/lfm2.5-2.6b-a";
ill_model_load(&model, &plan);
ill_state_make(&state, model, 4096);

IllBatch batch = { tokens, count, 0 };
ill_model_apply(model, state, &batch, &logits);
```

`build/` is generated by `util/make.py`; it is not part of the source tree.
Licensed under the terms in `LICENSE`.

## ↘️✴️ OVERVIEW

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

| capability             | supported                                                      |
| ---------------------- | -------------------------------------------------------------- |
| stored weight formats  | f32, f16, bf16                                                 |
| runtime weight formats | as stored, or repacked to q8                                   |
| tokenizer              | byte level BPE from `tokenizer.json`, GPT-2 and Llama-3 splits |
| sampling               | greedy, temperature, top-k, top-p, min-p, repetition penalty   |
| threading              | POSIX threads, Windows threads, or single threaded             |
| vector width           | AVX-512, AVX2, NEON, or plain C — same source at every width   |

The picture engine (`app/yolo.c`) is a separate translation unit with its own
seam and its own store; the two share conventions and nothing else.

| capability            | supported                                                            |
| --------------------- | -------------------------------------------------------------------- |
| checkpoint            | an Ultralytics `.pt` read directly — zip, pickle, fp16 storages       |
| tasks                 | detect, segment, pose, obb, classify                                  |
| sizes                 | n, s, m, l, x — read from the checkpoint, not from a scale table      |
| detection heads       | one-to-many with suppression (the default), and the NMS-free one-to-one |
| stored weight formats | f16, bf16, f32, f64 and the integer storages, all widened to f32 at load |
| picture formats       | everything stb reads; binary PNM without it                           |
| backends              | a nine-operation seam, with CPU as its reference implementation       |

## ↘️✴️ EVALUATE

`test/test.py` builds small Liquid checkpoints with random weights, runs them
through both this engine and Hugging Face `transformers`, and compares logits
position by position. It covers attention-only, convolution-only, and hybrid
stacks; grouped and multi query attention; tied and untied output heads; wide
and biased convolution kernels; f32, f16, and bf16 storage; chunked prefill and
single token decode; and tokenizer agreement over a corpus of awkward strings.

Against float32 checkpoints the engine matches the reference to **2e-7
relative**, which is float32 rounding. `test/test.c` adds 119 unit checks over
the internals. Both suites run clean under AddressSanitizer and
UndefinedBehaviorSanitizer.

### ➖ Comparison

The published `LiquidAI/LFM2.5-2.6B` checkpoint — 30 layers, 8 attention and 22
convolution, model dim 2048, feed forward 10752, 32 query heads over 8
key-value heads, vocabulary 128000 — on four x86-64 cores at 2.80 GHz with
AVX-512 and VNNI, at q8:

|         | weights  | prefill, 256 tok | decode     |
| ------- | -------- | ---------------- | ---------- |
| q8      | 2.83 GiB | 32.2 tok/s       | 9.7 tok/s  |
| q4      | 1.57 GiB | 30.0 tok/s       | 11.9 tok/s |

q8 decode on that host is 32.5 GB/s of weight traffic against a 36.6 GB/s bare
memory sweep, which is 89% of what the machine can fetch — at q8 decode is the
memory and there is little left in the kernel. q4 reads fewer bytes and comes
off that ceiling: 20.0 GB/s, with the unpack and the dot deciding the rate
instead. Prefill is arithmetic bound throughout, which is why q4 is slower at
it than q8.

A synthetic 2.9B-parameter checkpoint of LFM2-2.6B proportions — 32 layers,
model dim 2560, feed forward 8192, vocabulary 65536 — on an earlier four-core
AVX-512 host, before the paired dot:

|                      | weights  | prefill    | decode    |
| -------------------- | -------- | ---------- | --------- |
| bf16, as stored      | 5.44 GiB | 38.0 tok/s | 5.2 tok/s |
| q8, repacked at load | 3.06 GiB | 37.1 tok/s | 9.0 tok/s |

Decode scaled 2.4 → 4.6 → 8.9 tok/s across one, two, and four threads there.
Loading bf16 costs about a tenth of a second because the weights are memory
mapped and never copied; repacking to q8 costs a few seconds once.

### ➖ Pictures, against ultralytics

`app/yolo.c` against `ultralytics` 8.3.222 out of `ckpt/yolo26/py`, on the two
pictures ultralytics ships, on the same four-core AVX-512 host. **Compared on
the same decoded pixels** — stb and OpenCV do not agree on a JPEG to the last
unit, so the pictures are written out as PNG first and a `.jpg` comparison
would measure the two decoders as much as the engines.

| checkpoint | picture | reference | engine |
| --- | --- | --- | --- |
| yolo26n | bus | bus 0.881, person 0.872, 0.861, 0.846, 0.656 | the same, every digit |
| yolo26n `--nms-free` | zidane | person 0.915, 0.910, tie 0.527 | the same, every digit |
| yolo26s | bus | person 0.923, bus 0.923, person 0.898, 0.846, 0.833 | the same, every digit |
| yolo26x-seg | bus | bus 0.939, person 0.937, 0.923, 0.896, 0.754, 0.588 | the same, every digit |
| yolo26n-pose | bus | five people and their keypoints | the same, every digit |
| yolo26n-obb | bus | ground track field 0.027, centre 12.1 284.4, angle 0.449 | the same, every digit |
| yolo26n-cls | bus | minibus 0.5165 | the same |

Underneath the printed digits, the shaped input is **bit for bit identical** to
what the reference feeds its model, and the head's class logits agree to
**1.9e-5** over every logit that can become a detection. What is left is float
summation order, which is a different question to correctness.

Segmentation masks are the one place the two do not line up directly, and not
because they disagree: the reference returns masks at the model's input size,
this engine lifts each one straight to the source picture's pixels. Taking the
resolution difference out, the areas agree to a third of a percent.

Rates on that host, yolo26n at 640 over bus.png, the minimum of five runs:

| | shape | forward | decode | total |
| --- | --- | --- | --- | --- |
| detect, 480x640 | 5 ms | 397 ms | 2 ms | 405 ms |
| classify, 224x224 | 2 ms | 50 ms | 0 ms | 57 ms |
| obb, 768x1024 | 21 ms | 2035 ms | 0 ms | 2056 ms |

One thread, no intrinsics — the kernels are written in the shape a compiler
vectorises rather than in the instruction set of one host. Threading is the
first open item in `TODO.md` and is worth more than anything left in the
kernel.

## ↘️✴️ FINETUNE

The engine has no trainer. A tune happens on the reference side and comes back
as a checkpoint the engine reads unchanged: `util/tune.py` trains a LoRA adapter
over the frozen base, folds it into the float weights, and writes
`build/tune/merged` in the same layout as `ckpt/lfm2.5-2.6b-a`.

What it tunes for is **caveman**, the compression register described by the
skill at <https://github.com/JuliusBrussee/caveman>: drop articles, filler,
hedging and pleasantries, and keep every technical fact, every number, every
negation and every byte of code. LFM2.5 is a thinking model — its chat template
ends every generation prompt with `<think>` — so the tune trains the reasoning
span as well as the answer, at separate intensities. The level is a system
prompt, so the register is a knob rather than a change of voice:

|                        | thought | answer | whole reply  |
| ---------------------- | ------- | ------ | ------------ |
| `Normal mode.`         | 19      | 113    | 132 tokens   |
| `Caveman mode: lite.`  | 21      | 37     | 58, 2.3x off |
| `Caveman mode: full.`  | 28      | 26     | 54, 2.4x off |
| `Caveman mode: ultra.` | 10      | 12     | 22, 6.0x off |

One question — "Explain database connection pooling" — carried at all four
levels in the corpus, counted with the checkpoint's own tokenizer.

```sh
python3 util/tune.py --lint                           # audit the corpus
python3 util/tune.py --model ckpt/lfm2.5-2.6b-a --uncensor     # abliterate into build/tune/uncensored
python3 util/tune.py --model ckpt/lfm2.5-2.6b-a --train        # LoRA adapter into build/tune
python3 util/tune.py --model ckpt/lfm2.5-2.6b-a --merge        # fold it into build/tune/merged
python3 util/tune.py --model ckpt/lfm2.5-2.6b-a --check --tuned build/tune/merged
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
weights to `build/tune/uncensored` in the same layout as `ckpt/lfm2.5-2.6b-a`, so the engine
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
python3 util/tune.py --model ckpt/lfm2.5-2.6b-a --uncensor --train --merge
```

The adapter then trains over the decensored weights rather than over the base,
which is what will be served. Two settings are this checkpoint's rather than
heretic's, and `util/tune.py`'s header says why: the response prefix is
`</think>`, because LFM2.5's template has already opened the think block and
refusals would otherwise be counted over reasoning text; and the divergence
heretic balances its two objectives at follows the export cap, so the search
spends its trials in the band a trial can be taken from. No abliteration has
been run over the 2.6B weights yet — that needs a card, and `TODO.md` carries
the item.

## ↘️✴️ BENCHMARKS

`util/eval.py` scores a checkpoint on nine public benchmarks, and the thing it
scores is **this engine**: `build/app_main generate` is registered with
inspect-ai as a model provider, so every sample is a run of the C engine over
the checkpoint rather than of `transformers` over the same weights. The tasks
themselves are [lighteval](https://github.com/huggingface/lighteval)'s, run
through its eval backend — the inspect-ai one, `lighteval eval` on the command
line — which is the only part of this that needs installing:

```sh
pip install -e .                                  # lighteval and langdetect
python3 util/eval.py --model ckpt/lfm2.5-2.6b-a            # the nine benchmarks
python3 util/eval.py --model ckpt/lfm2.5-2.6b-a --samples 200   # a longer run
python3 util/eval.py --tasks ifeval,boolq --model ckpt/lfm2.5-2.6b-a
python3 util/eval.py --model ckpt/lfm2.5-2.6b-a --rival hf-inference-providers/Qwen/Qwen3-4B
python3 util/eval.py --chart                      # redraw from the last results
```

| benchmark              | task           | what it asks                               |
| ---------------------- | -------------- | ------------------------------------------ |
| IFBench                | `ifbench_test` | held out instruction following constraints |
| IFEval                 | `ifeval`       | verifiable instruction following           |
| Long Horizon Execution | generated      | executing a plan of growing length         |
| MMLU Pro               | `mmlu_pro`     | ten choice knowledge and reasoning         |
| MuSR                   | `musr`         | multi step soft reasoning, three subsets   |
| SimpleQA               | `simpleqa`     | short fact seeking questions, model graded |
| AGIEval                | `agieval`      | human exam questions, seventeen subsets    |
| bAbI QA                | `babi_qa`      | synthetic reading comprehension            |
| BoolQ                  | `boolq`        | yes or no reading comprehension            |

It writes `build/eval/results.json`, a markdown table beside it, an SVG chart
that reads in a light or a dark theme, and the inspect-ai log set, which
`inspect view --log-dir build/eval/logs` opens sample by sample. The chart
carries a bar per benchmark per model, and a second panel for the long horizon
task: accuracy against the number of steps in the plan, which is the shape that
benchmark exists to show.

Two of the nine need more than the backend has. **bAbI QA** ships in lighteval
without its inspect-ai half — no `sample_fields`, no scorer, and a prompt
function that decodes every answer as a compass path and so raises on the
subset the task selects — and `util/eval.py` supplies all of it. **Long
Horizon Execution** is not in lighteval at any version, so it is generated: a
seeded dictionary, a plan of N lookups over it, and one answer that is the
values concatenated in plan order, after the benchmark in "The Illusion of
Diminishing Returns: Measuring Long Horizon Execution in LLMs". It is **not
the paper's data**, it is written `Long Horizon Execution (generated)`
wherever it is reported, and a score on it compares with another run of this
script rather than with a published number.

LFM2.5 reasons before it answers, so the provider splits each reply at
`</think>` and hands the scorers the answer with the reasoning kept beside it
in the log; a reply that never closes the block is scored whole, because that
run hit the token cap and hiding it would flatter the model. Replies are
cached, keyed by the prompt and by a stamp of the weights, the binary and the
quantisation, so a retuned checkpoint at the same path is never served the
replies of the one before it.

**No run against the published weights has been taken yet**, so there are no
numbers here to quote: `--samples` defaults to 25, which is a smoke test, and a
2.6B checkpoint on CPU answers a few hundred samples an hour. Whatever cap a
run used is written into `results.json` and printed under both the table and
the chart, because a score without its sample count is not a number anyone can
use.

## ↘️✴️ ACCELERATE

The forward pass never calls a kernel directly.

It calls through `IllBackend`, a table of the six shapes of work the stack performs —
`dense`, `rmsnorm`, `swiglu`, `rope`, `attend`, `conv1d` — plus `setup`, `close`, and `width`.

The CPU backend fills that table with threaded kernels
and a device backend fills the same table and keeps its queue in `inner`.
(Nothing above the seam changes).

## ↘️✴️ DEVELOPMENT

See `GUIDE.md` for the full tour, and `TODO.md` for what is still open.
