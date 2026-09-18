# LFM2.5-1.2B-Thinking: Agentic & Reasoning Dataset Strategy

## Executive Summary

This document specifies the dataset mixture, format, and construction pipeline used to fine-tune **[LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)** for agentic tool use.

Everything here is grounded in the model's actual `chat_template.jinja` and in the verified schemas of the source datasets. The previous revision of this plan targeted the Hermes/ChatML protocol (`<tools>` + `<tool_call>` JSON) and four datasets that cannot produce the sample counts it claimed; see [§8](#8-what-changed-and-why) for the full list of corrections.

Objectives:
1. **Reasoning before action** — a real `<think>` block on every supervised turn.
2. **Native LFM2.5 tool calls** — Pythonic calls between `<|tool_call_start|>` and `<|tool_call_end|>`, rendered from structured `tool_calls`.
3. **Restraint** — answer directly, or ask for a missing argument, instead of inventing a tool call.
4. **Token economy** — an opt-in compressed register ([§4](#4-token-compression-caveman-style)) that cuts decode latency on-device without touching code, numbers, or arguments.

---

## 1. The Format Is Not Negotiable

LFM2.5 does **not** use the Hermes tool protocol. Reading [chat_template.jinja](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/chat_template.jinja) gives five constraints that drive every other decision in this document.

| # | Template behaviour | Consequence for the dataset |
| :-- | :--- | :--- |
| 1 | Tools are appended to the system prompt as `List of tools: [{...}, {...}]`, from the `tools=` argument. | Never write a `<tools>` block into message content. Store tools as a separate structured field. |
| 2 | Tool calls render from `message["tool_calls"]` as `<|tool_call_start|>[fn(arg='value')]<|tool_call_end|>` — **Python source, single-quoted strings**, not JSON. | Never write `<tool_call>{...}</tool_call>` into content. Function names must be valid Python identifiers. |
| 3 | Reasoning renders from `message["thinking"]` (or `reasoning` / `reasoning_content`). | Parse `<think>` out of source content into a dedicated field. |
| 4 | **`preserve_thinking` defaults to `False`: only the _last_ assistant turn keeps its `<think>` block.** | Training on a whole multi-turn trace with loss on every assistant turn teaches the model *not to think* on non-final turns. Traces must be expanded into one row per assistant turn, with loss on that turn only. |
| 5 | The template emits `bos_token` (`<|startoftext|>`) itself, and marks assistant spans with `{% generation %}`. | Tokenize with `add_special_tokens=False` or you get a duplicate BOS. Completion-only masking can use `return_assistant_tokens_mask=True`. |

Constraint 4 is the subtle one and the most damaging to get wrong. It is the reason the pipeline emits `prompt` / `completion` pairs instead of raw conversations.

### Target rendering

A tool-calling turn:

```text
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.
List of tools: [{"name": "get_stock_price", "description": "Fetches the current stock price.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}]<|im_end|>
<|im_start|>user
What is Apple trading at?<|im_end|>
<|im_start|>assistant
<think>The user wants Apple's current price. `get_stock_price` takes a ticker symbol; Apple is AAPL.</think><|tool_call_start|>[get_stock_price(symbol='AAPL')]<|tool_call_end|><|im_end|>
```

The follow-up turn, as a **separate training row**. Note that the first assistant turn has lost its `<think>` block — this is exactly what the model sees at inference time:

```text
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.
List of tools: [...]<|im_end|>
<|im_start|>user
What is Apple trading at?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_stock_price(symbol='AAPL')]<|tool_call_end|><|im_end|>
<|im_start|>tool
{"symbol": "AAPL", "price": 234.50, "currency": "USD"}<|im_end|>
<|im_start|>assistant
<think>The tool returned 234.50 USD. That answers the question directly.</think>Apple (AAPL) is trading at $234.50 USD.<|im_end|>
```

### Constrained decoding

Because the default output is Python source rather than JSON, a JSON grammar will not constrain it. Either write a GBNF/`outlines` grammar for the Pythonic call form, or explicitly instruct JSON output in the system prompt — the model card states this override is supported — and constrain that instead. Do not assume a JSON schema grammar applies to the default format.

---

## 2. Dataset Mixture (12,000 samples)

All sources below were checked against the HF datasets-server: they exist, are public and ungated, and their schemas match what the pipeline expects.

| Source | Subset | Rows | % | Role in the mixture |
| :--- | :--- | ---: | ---: | :--- |
| `interstellarninja/hermes_reasoning_tool_use` | `single` | 2,400 | 20% | Single-call precision: pick the tool, fill every argument. |
| " | `multistep` | 1,600 | 13% | Several calls to satisfy one request. |
| " | `multiturn` | 2,600 | 22% | Tool results → synthesis → next request; error and follow-up handling. |
| " | `relevance` | 1,400 | 12% | **Negatives.** Nvidia When2Call: answer directly, or ask for a missing argument, instead of calling. |
| `HuggingFaceTB/smoltalk2` | `SFT/smolagents_toolcalling_traces_think` | 2,200 | 18% | Long-horizon agent loops (search → visit → verify → `final_answer`). |
| `open-r1/Mixture-of-Thoughts` | `all`, length-filtered | 1,200 | 10% | Tool-free math/code/science reasoning; prevents reasoning collapse. |
| `HuggingFaceTB/smoltalk2` | `SFT/smoltalk_systemchats_Qwen3_32B_think` | 600 | 5% | Tool-free persona/system following; prevents tool-calling in non-tool contexts. |

**Split:** 95% train / 5% validation, stratified by source and **grouped by trace** — expanded prefixes of the same conversation never straddle the split.

**Style overlay:** 15% of the final rows are rewritten into a compressed register and marked `style="caveman"`. This is an overlay on the rows above, not an extra source — see [§4](#4-token-compression-caveman-style).

### Why this mixture

`interstellarninja/hermes_reasoning_tool_use` (51,004 rows) carries most of the load because it is the only verified source that pairs a **structured `tools` column** with **authentic `<think>` reasoning**, across four labelled scenario types. Its upstream sources are ToolACE, xLAM, Glaive, Nous-Hermes and Nvidia-When2Call, so one dataset covers the schema-diversity, multi-turn and negative-sample roles that the previous plan spread across four separate (and partly unusable) datasets.

**Negatives are 12% of the mixture, not zero.** A tool-heavy SFT run with no restraint data produces a model that reaches for `<|tool_call_start|>` on "hello". The `relevance` scenario and the two tool-free subsets exist specifically to counteract that.

### Sources that were evaluated and rejected

| Source | Verdict |
| :--- | :--- |
| `Salesforce/xlam-function-calling-60k` | **Gated.** `load_dataset` fails without an accepted licence. Its content reaches the mixture via `hermes_reasoning_tool_use` (`source: Salesforce-Xlam`) — with reasoning attached, which the raw dataset lacks. |
| `internlm/Agent-FLAN` | **No `train` split** (splits are `agent_instruct_react`, `toolbench_*`), and the field is `conversation`, not `conversations`. Contains no `<think>` traces. |
| `mondk/Deepseek-v4.1-CoT` | **23 rows in total.** The previous plan allocated 1,250 samples (10%) to it. |
| `MoreThought/Fable-5.1-Max-Reasoning-Filtered-5000x` | Schema is `messages: [{role, content}]` only. Traces are 118–195 messages of an agent authoring this dataset's own seed corpus; **~72% of assistant turns are empty** because tool calls were stripped, leaving orphaned `tool` results. Far beyond a 4k window, and would train empty assistant turns. |
| `NousResearch/hermes-function-calling-v1` | Has no `default` config, and — more importantly — **no reasoning traces**. Including it puts rows without `<think>` into a thinking model's training set. Superseded by `hermes_reasoning_tool_use`, which is partly derived from it. |
| `mlx-community/hermes-reasoning-tool-use` | A reformat of `interstellarninja/hermes_reasoning_tool_use` that **drops the `tools` column** and bakes the Hermes system prompt into the conversation. Use the upstream. |

---

## 3. Construction Rules

These are the invariants the pipeline enforces. Each one exists because violating it produces a specific, observable failure.

| Rule | Failure it prevents |
| :--- | :--- |
| **No synthesised reasoning.** A turn either carries real reasoning from the source or is dropped. | Injecting a constant stub (`"<think>Evaluating context and parameters.</think>"`) across thousands of rows trains a fixed, meaningless thought prefix and destroys the reasoning the base model already has. |
| **Every supervised turn has a non-empty `<think>`.** | Rows without reasoning teach a thinking model to skip thinking — the single highest-impact regression available. |
| **Pruning never removes a called tool.** Tools the gold answer calls are pinned; the rest are sampled as distractors and shuffled. | `tools[:5]` can delete the answer's own tool, producing a sample that literally trains hallucination. |
| **Registry size is randomised (3–8 tools, ≤6k chars).** | A fixed count of 5 does not generalise to the 15–20 tools a real MCP/VS Code agent injects. |
| **Traces are expanded into one row per assistant turn.** | See [§1](#1-the-format-is-not-negotiable), constraint 4. |
| **Over-length rows are dropped, never truncated.** | Cutting a `<think>` block at N characters and appending `"..."` teaches the model to stop reasoning mid-sentence. |
| **Malformed actions are dropped, never repaired.** | `text.replace("'", '"')` corrupts every apostrophe inside a string argument. |
| **Source system prompts are replaced.** | Hermes/smolagents system prompts describe `<tools>`/`<tool_call>` XML, which contradicts the LFM2.5 template and fights it at inference time. |
| **Every row is re-rendered through the real template and re-tokenised.** | Catches schema errors the template raises on (e.g. JSON-string arguments) before they reach the trainer. |
| **Dedup on rendered text; split grouped by trace.** | Expanded prefixes of one conversation leaking across train/validation makes eval loss meaningless. |

### Output schema

| Column | Purpose |
| :--- | :--- |
| `text` | Full rendered conversation (packing-based trainers). |
| `prompt` / `completion` | Exact string split at the final `<|im_start|>assistant\n`; `prompt + completion == text` is asserted. Gives completion-only loss without relying on trainer-side masking. |
| `messages_json` / `tools_json` | Structured form, so the mixture can be re-rendered for another template. JSON-encoded because `arguments` has an open-ended schema that Arrow cannot type. |
| `source`, `category`, `trace_id` | Stratification, grouped splitting, and ablation. |
| `style` | `plain` or `caveman`. Lets you ablate the compressed register, or filter it out entirely. |
| `n_tokens`, `n_prompt_tokens`, `n_completion_tokens`, `n_turns`, `has_tool_call` | Length budgeting and mixture auditing. |

Length budget: `MAX_SEQ_LEN = 4096`, `MAX_PROMPT_LEN = 3072`, `MIN_COMPLETION_LEN = 16` tokens. Measured with the real LFM2.5 tokenizer, not a characters ÷ 4 estimate.

---

## 4. Token Compression: Caveman Style

Adapted from the [`caveman`](https://github.com/JuliusBrussee/caveman) skill. Its style rules are sound and unusually well reasoned about tokenizers; its published numbers are about a *prompt-time skill on large cloud models* and do not transfer to fine-tuning a 1.2B model. Everything below is re-derived and re-measured for this model.

### Why it is worth more here than it is for a cloud agent

On-device, LFM2.5-1.2B-Thinking decodes at roughly 60–116 tok/s. Tokens are **wall-clock latency**, not just billing. And on a thinking model the reasoning block dominates the output budget, then gets regenerated on every turn of an agent loop — so a cut compounds across a trajectory.

The caveman project's own [HONEST-NUMBERS](https://github.com/JuliusBrussee/caveman/blob/main/docs/HONEST-NUMBERS.md) states the skill "does not compress input, context, files, or **model thinking tokens**." For a thinking model, that untouched axis is the majority of the budget. It is the opportunity.

### Fine-tuning fixes the failure mode the skill has

The skill's known net-negative case is that its rule file costs input tokens on every call, which on terse workloads exceeds what it saves ([issue #145](https://github.com/JuliusBrussee/caveman/issues/145)). Measured on the LFM2.5 tokenizer:

| Delivery | Cost per call |
| :--- | ---: |
| `skills/caveman/SKILL.md` injected as prompt rules | **1,812 tokens** |
| Behaviour in the weights, triggered by a directive | **27 tokens** |

Baking the register into the model is what makes the trade unambiguously positive. That is an argument for doing this in SFT, not an argument that the style is free.

### Which rules survive contact with this tokenizer

The caveman rules are tokenizer-dependent claims. Verified against LFM2.5's 65,536-token vocabulary:

| Transform | Tokens | Verdict |
| :--- | :--- | :--- |
| `" the file"` → `" file"` | 2 → 1 | **Applied.** Articles are a real, per-instance saving. |
| Filler / pleasantry / hedge removal | 34 → 18 on the reference sentence | **Applied.** 47% on prose. |
| `" configuration"` → `" cfg"` | 1 → **2** | **Rejected.** The abbreviation *costs* a token and reads worse. |
| `" A -> B"` vs `" A then B"` | 3 = 3 | **Rejected.** Arrows save nothing. |
| `" sees"` → `" see"` | 1 = 1 | **Rejected.** Mangled grammar buys nothing. |

The last three matter: they are the transforms people assume are compression and are not. The compressor implements only the first two, and may only ever *delete* — never substitute, never abbreviate, never add a word to sound terse.

### Design decisions

**1. Conditional, never global.** The style is bound to a 27-token system directive and applied to 15% of rows (`--caveman-ratio`). The model keeps full prose as its default and learns the compressed register as a mode. A global rewrite would remove the baseline, make the change unmeasurable, and stake the whole run on a style experiment.

**2. Compress reasoning *and* answers; never inputs.** `--caveman-target {think,answer,both}`, default `both`. User turns and tool results are never rewritten. Two reasons: Adobe's CAVEWOMAN result is that compressing the *human's* prompt makes models answer longer and worse; and training on compressed observations would make the model fragile when it meets real, uncompressed tool output.

**3. Compress existing rows — do not add datasets for this.** The compressed subset is produced by rewriting rows already selected in [§2](#2-dataset-mixture-12000-samples), not by pulling in new sources. This keeps provenance identical between the plain and compressed halves (so `style` is a clean ablation variable), adds no new source risk, and keeps the mixture at 12,000 — a size a 1.2B model can absorb in two epochs without overfitting. Default `--caveman-mode convert` replaces the sampled rows, so the total does not inflate; `--caveman-mode pair` keeps both versions of each prompt for a controlled A/B, at the cost of near-duplicate prompts.

**4. Tool calls are structurally out of reach.** The compressor only ever rewrites the `content` and `thinking` *strings*. `tool_calls` is a separate structured field, so function names and every argument value survive byte-identical by construction rather than by careful regex. This also gives a free exact verifier: for a tool-calling turn, the "answer" is provably unchanged, so compression can only have affected the reasoning that led to it.

### Enforced invariants

Every rewrite is accepted only if it passes all of these; otherwise the original row is kept unchanged.

- **Protected spans are byte-identical**: fenced and inline code, URLs, double-quoted strings, file paths and dotted names, `function(...)` calls, numbers, versions and units.
- **Negation count is unchanged** for every one of `not / no / never / none / nor / neither / without / except / only / cannot / unless / fail…`. Dropping one inverts a claim, which is worse than any saving.
- **Digit sequence is unchanged**, in order.
- **Hazard turns are skipped entirely** — anything mentioning deletion, irreversibility, overwrites, credentials or data loss keeps full prose. This mirrors caveman's Auto-Clarity carve-out.
- **Token count must actually drop** by at least `--caveman-min-saving` (default 10%), measured with the real tokenizer.

### Measured result, and how to go further

The shipped compressor is deterministic and rule-based. On a scaled run it cut **12.1%** of completion tokens on the turns it accepted, rejecting 106 candidates that failed an invariant or did not save enough.

12% is an honest number for a transform that is provably meaning-preserving. The 47% figure above is what an *LLM rewriter* reaches on prose, and getting there means replacing `compress_prose()` with a model call. If you do that, keep `preserves_meaning()` as the acceptance gate and add a correctness gate on top, because an LLM rewriter is a second teacher with its own errors:

- **Tool-calling rows**: gate is free and exact — the call must remain byte-identical.
- **Math rows** (`OpenR1-Math-220k` inside Mixture-of-Thoughts): gate on the final answer still matching the gold answer.
- **Everything else**: no automatic gate exists. Either leave it to the deterministic compressor or accept that it is unverified.

### Risks worth stating plainly

- **Compressed chain-of-thought can cost accuracy**, and small models have the least headroom. This is why the default is 15% and conditional, and why `--caveman-target answer` exists as the low-risk setting: answer-side compression is the variant with third-party evidence behind it (JetBrains measured no detectable quality change), while reasoning-side compression is the one nobody has validated.
- **The cited numbers are not evidence for this dataset.** Adobe and JetBrains measured prompt-time styling of large models. They justify *investigating* the idea here; they do not predict the outcome.
- **Evaluate the compressed register separately.** Report accuracy for `style="plain"` and `style="caveman"` as two populations. Do not compare their median token counts directly — those are different samples, and the comparison is confounded. The within-sample before/after number the pipeline prints is the honest one.

---

## 5. Running the Pipeline

```bash
pip install -U datasets transformers jinja2   # jinja2 is required by apply_chat_template
hf auth login                                 # LFM2.5 tokenizer requires an accepted licence

python create_dataset.py --scale 0.02 --out /tmp/smoke   # verify first
python create_dataset.py --out ./lfm25-agentic-12k

# style controls
python create_dataset.py --caveman-ratio 0        --out ./baseline-no-style
python create_dataset.py --caveman-target answer  --out ./low-risk-style   # answers only
python create_dataset.py --caveman-mode pair      --out ./ablation-pairs
```

The script prints a per-source extraction report (`kept` plus every drop reason) and a mixture summary. Treat a large `quota_shortfall`, or any source reporting `kept=0`, as a failure — the previous pipeline silently yielded zero rows from three of its six sources because field names did not match.

Reference output from a scaled-down run:

```text
--- extraction report -------------------------------------------
  hermes-reasoning-tool-use   kept=142  invalid_structure=47, no_reasoning=18, unparsable=10, ...
  smoltalk2-smolagents        kept=44   called_tool_not_in_registry=47, unparsable=1
  mixture-of-thoughts         kept=24   too_long=27
  smoltalk2-systemchats       kept=12   no drops

caveman: 9347 -> 8214 completion tokens on compressed turns (12.1% cut, 106 rejected)

train: 212 rows
  tokens  min=298 p50=787 p95=2232 max=2923
  with tool call: 45.3%
  with <think>:   100.0%
  caveman style:  16.0%
```

### Acceptance checks

Run these against the produced JSONL before training:

- `all('<think>' in r['completion'] for r in rows)` — a thinking model must think on every supervised turn.
- `all(r['prompt'] + r['completion'] == r['text'] for r in rows)` — the completion boundary is exact.
- `all(r['prompt'].startswith('<|startoftext|>') for r in rows)` — BOS comes from the template.
- `not any('<tool_call>' in r['text'] or '<tools>' in r['text'] or '<tool_response>' in r['text'] for r in rows)` — no Hermes protocol leaked through.
- `0.40 <= mean(r['has_tool_call']) <= 0.55` — tool calls must not dominate, or restraint collapses.
- `all('caveman style' in r['prompt'] for r in rows if r['style'] == 'caveman')` and the converse for `plain` — the style must be conditioned on the directive, not learned unconditionally.

---

## 6. Fine-Tuning Configuration

### LoRA target modules

LFM2 is a hybrid conv/attention architecture and **does not use Llama module names**. `o_proj`, `gate_proj`, `up_proj` and `down_proj` do not exist in this model; PEFT will either raise or silently adapt nothing.

```yaml
lora:
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - q_proj        # attention (6 GQA blocks)
    - k_proj
    - v_proj
    - out_proj      # not o_proj
    - in_proj       # short-conv blocks (10 double-gated conv blocks)
    - w1            # MLP, not gate_proj
    - w2            # not down_proj
    - w3            # not up_proj
```

### Training

```yaml
base_model: LiquidAI/LFM2.5-1.2B-Thinking
sequence_len: 4096          # matches the dataset length budget
bf16: true

learning_rate: 1.0e-4       # 2e-4 for 3 epochs on 12k rows overfits a 1.2B model
lr_scheduler: cosine
warmup_ratio: 0.03
num_epochs: 2
micro_batch_size: 4
gradient_accumulation_steps: 4
weight_decay: 0.01
gradient_checkpointing: true

train_on_inputs: false      # loss on the completion only
sample_packing: true
group_by_length: false
```

Two things to verify in whichever trainer you use:

- **`add_special_tokens=False` when tokenizing.** The template already emits `<|startoftext|>`; adding another BOS shifts every position.
- **Completion-only masking is actually applied.** Either use the `prompt`/`completion` columns directly, or pass `return_assistant_tokens_mask=True` to `apply_chat_template` and rely on the template's `{% generation %}` markers. Do not mix both.

### Evaluation

Validation loss alone will not detect format collapse. Also track, on the held-out split:

- share of generations containing exactly one well-formed `<think>...</think>`;
- share of tool-calling generations that parse as a Python call list and name a tool actually present in the registry;
- argument-level exact match against the gold call;
- **false-call rate on the `relevance` subset** — the metric that catches tool hallucination;
- **the four metrics above split by `style`** — if `caveman` trails `plain` on argument exact match, the compressed register is damaging reasoning and `--caveman-target` should drop to `answer`;
- **style leakage**: rate at which the model answers in the compressed register *without* the directive present.

BFCLv3 is the external benchmark, but it needs a custom handler for the Liquid tool-use template; Liquid's own reported numbers use one.

---

## 7. Known Limitations

- **Reasoning distillation ceiling.** The `<think>` traces come from much larger teachers. A 1.2B model imitates their form more readily than their correctness; expect well-structured reasoning that is sometimes wrong, and evaluate answers, not reasoning aesthetics.
- **The compressed register is unvalidated on reasoning.** Answer-side compression has third-party evidence; reasoning-side compression does not. Treat the first training run as the experiment that decides it, and keep `--caveman-ratio 0` as the fallback.
- **12% is the deterministic ceiling.** Reaching the ~47% that an LLM rewriter achieves requires building the rewriter and its correctness gate.
- **`smolagents` yield is ~50%.** Roughly half of those traces call a tool absent from the recorded registry and are dropped. Harmless, but it means the quota draws from a larger pool than the row count suggests.
- **English-dominant.** LFM2.5 supports 8 languages; this mixture is essentially monolingual. If multilingual tool use matters, add `SFT/smoltalk_multilingual8_Qwen3_32B_think`. Note that the article-dropping rule is English-specific and must not be applied to languages where small markers carry case or role.
- **No preference stage.** This is SFT only. Restraint (call vs. don't call) responds well to a DPO pass afterwards; `Preference/tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think` is a starting point.

---

## 8. What Changed and Why

The previous revision is preserved as `LFM2.5-1.2B-Thinking-Agentic-Dataset-Analysis.legacy.md`, and its script as `create_dataset.legacy.py`.

**Format — the plan targeted the wrong protocol.** It specified Hermes `<tools>` and `<tool_call>` JSON in message content; LFM2.5 uses `List of tools:` plus Pythonic calls between `<|tool_call_start|>`/`<|tool_call_end|>`, rendered from structured fields. It omitted `<|startoftext|>`, stored reasoning inline instead of in `thinking`, wrapped tool results in `<tool_response>`, missed the `preserve_thinking` behaviour entirely, and recommended JSON grammar constraints that do not apply to the default output format.

**Sources — four of six could not deliver.** `Agent-FLAN` (2,500 planned) has no `train` split and uses `conversation`, not `conversations` — zero rows. `Fable-5.1` (2,500 planned) exposes only `messages`, while the script read `prompt`/`response`/`reasoning` — zero rows. `mondk/Deepseek-v4.1-CoT` (1,250 planned) has 23 rows. `xlam` (2,000 planned) is gated. `hermes-function-calling-v1` has no `default` config and would have raised on load. In total, **8,250 of 12,500 planned samples could not be produced**, and the pipeline had no reporting that would have surfaced it.

**Quality — the reasoning would have been actively damaged.** Boilerplate `<think>` stubs were injected into thousands of rows. `tools[:5]` could delete the tool the gold answer calls. Reasoning was truncated mid-sentence at 4,000 characters. `clean_think_tag` appended `</think>` at the *end* of a message, swallowing the answer into the reasoning block. `sanitize_json` replaced every apostrophe with a double quote (and was never called). There were no negative samples, no deduplication, no role whitelist, no tokenizer-based length check, and an ungrouped positional train/validation split that leaked near-duplicates.

**Training config.** LoRA targeted Llama module names that do not exist in LFM2; 3 epochs at 2e-4 was too aggressive for 12k rows on a 1.2B model.
