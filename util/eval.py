#!/usr/bin/env python3
"""util/eval.py -- scores a checkpoint on the public benchmarks, and draws the result.

The engine has no scoring harness and no opinion about what a benchmark is, so
the scoring happens on the reference side, the same way the tune does:
`lighteval`'s **eval backend** -- the inspect-ai one, `lighteval eval` on the
command line -- runs the tasks, and the model under test is this repository's
own binary. `build/app_main generate` is registered with inspect-ai as a model
provider called `inferliq`, so every sample of every benchmark is a real run of
the C engine over the checkpoint, not of `transformers` over the same weights.
Nothing about the engine changes to make this work; the provider is forty lines
that spawn a process and read its stdout.

    python3 util/eval.py --model model                  the nine benchmarks
    python3 util/eval.py --model model --samples 200    a longer run
    python3 util/eval.py --tasks ifeval,boolq --model model
    python3 util/eval.py --model model --rival hf-inference-providers/Qwen/Qwen3-4B
    python3 util/eval.py --chart                        redraw from the last results

It writes four things under `build/eval/`: `logs/` (the inspect-ai eval set,
which `inspect view --log-dir build/eval/logs` opens), `results.json`, a
markdown table in `results.md`, and `chart.svg`.

The nine benchmarks
-------------------

    IFBench     ifbench_test   held out instruction following constraints
    IFEval      ifeval         verifiable instruction following
    Long Horizon Execution     plan execution at growing step counts
    MMLU Pro    mmlu_pro       ten choice knowledge and reasoning
    MuSR        musr           multi step soft reasoning, three subsets
    SimpleQA    simpleqa       short fact seeking questions, model graded
    AGIEval     agieval        human exam questions, seventeen subsets
    bAbI QA     babi_qa        synthetic reading comprehension
    BoolQ       boolq          yes or no reading comprehension

Seven of the nine are lighteval tasks that the eval backend already runs. Two
are not, and this script supplies what is missing rather than dropping them:

  1. **`babi_qa` ships without its inspect-ai half.** Its lighteval config has
     a `prompt_function` and a metric, but no `sample_fields`, `solver` or
     `scorer`, and the eval backend refuses a task without them -- lighteval's
     own source marks it `TODO: clean dataset and convert to inspect-ai`. The
     three fields are filled in here, over the same `facebook/babi_qa` rows
     and the one word match the lighteval config asks for. The prompt function
     beside them could not be reused: it decodes every answer as a compass
     path before reading the question, which raises on the subset the config
     selects, so the unpacking is written out here with that decode where the
     question asks for it. If a later lighteval fills the fields in itself,
     its version wins and this one stands down.

  2. **Long Horizon Execution is not in lighteval at all**, at any version this
     script has seen, and the benchmark's rows are not on the Hub under a name
     worth guessing at. It is generated here instead, which is what this
     repository does for test media anyway: a seeded dictionary of key to value
     pairs, a plan of N lookups over it, and one answer that is the values
     concatenated in plan order. That is the shape the benchmark measures --
     execution, with the planning and the knowledge handed to the model in the
     prompt, so the only thing left to fail is carrying out the steps -- after
     "The Illusion of Diminishing Returns: Measuring Long Horizon Execution in
     LLMs". **It is not the paper's data**, so it is written
     `Long Horizon Execution (generated)` everywhere it is reported, and a run
     of it is comparable with another run of this script, not with a published
     number. Plan lengths are 2, 4, 8, 16 and 32 steps by default, which is
     what the second panel of the chart plots: the accuracy of one model
     against the length of the plan it was asked to execute.

What the numbers mean
---------------------

One score per benchmark, the metric the task itself reports: `accuracy` for the
multiple choice and short answer tasks, `prompt_level_strict_acc` for IFEval and
IFBench, and the mean over subtasks where a benchmark has them -- AGIEval is
seventeen exams, MuSR is three, BoolQ is two. `--samples N` caps how many rows
of each task are scored, and the default of 25 is a smoke test rather than a
result: a 2.6B model on a laptop CPU answers a few hundred samples an hour, and
a full AGIEval alone is eight thousand. Whatever the cap was is written into
`results.json`, printed under the table and drawn under the chart, because a
score without its sample count is not a number anyone can use.

Two of the tasks cost more than they look. SimpleQA is graded by a model rather
than by a string match, and with no `--grader` the grader is the model under
test, which is a 2.6B checkpoint marking its own homework -- pass
`--grader openai/gpt-4o-mini` or any other inspect-ai model string for a grade
worth quoting. IFEval and IFBench check the reply against verifiable
constraints, so they need `langdetect` and `nltk`, which `pyproject.toml`
carries.

Replies are cached, and the cache knows which weights answered
------------------------------------------------------------

Every task in the eval backend asks inspect-ai to cache what the model said,
so a run that stops halfway and a run that adds a benchmark both cost only the
samples they have not answered yet -- which matters when a sample is a minute
of CPU. The cache is keyed by the prompt and by the model name, and the model
name here is a folder: `build/tune/merged` is a different checkpoint after
every merge. So the provider hands inspect-ai a stamp of the shard sizes and
times, the engine binary and the quantisation as its base url, which is part
of that key. A retuned checkpoint at the same path is never served the replies
of the one before it.

Why a reply is split at `</think>`
----------------------------------

LFM2.5 is a thinking model whose chat template ends the generation prompt with
`<think>`, so the first thing the engine emits is reasoning and the answer comes
after `</think>`. A scorer that reads the whole completion would mark the
reasoning, which fails a multiple choice task that looks for `ANSWER: C` and
passes a fact task on a guess the model then talked itself out of. The provider
splits the reply and hands inspect-ai the answer as text and the reasoning as a
reasoning block, so the scorers see the answer and the logs keep both. A reply
that never closes its think block is scored whole -- that run hit the token cap,
and pretending otherwise would hide the failure.

Install
-------

    pip install -e .            or: pip install lighteval langdetect

`pyproject.toml` beside this script pins what the backend needs. The engine
itself still has no dependencies; this is a workflow helper like `util/tune.py`,
and `--chart` redraws the chart and the table from `results.json` with the
standard library alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

# This script lives in util/, one directory below the repository root; the
# engine, its build folder and the checkpoints are all rooted there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
HOUSE = os.path.join(BUILD, "eval")
BINARY = os.path.join(BUILD, "app_main" + (".exe" if os.name == "nt" else ""))


# ---------------------------------------------------------------------------
# the benchmarks
# ---------------------------------------------------------------------------

# key, the name printed, the lighteval task, and the metrics to read in the
# order they are preferred. A task reports what its own scorer computes:
# `accuracy` nearly everywhere, and the two instruction following tasks report
# a strict and a loose rate instead, of which the strict one is the number
# those benchmarks are quoted by.
BENCH_LIST = (
    ("ifbench",  "IFBench",                        "ifbench_test",
     ("prompt_level_strict_acc", "prompt_level_loose_acc", "accuracy")),
    ("ifeval",   "IFEval",                         "ifeval",
     ("prompt_level_strict_acc", "prompt_level_loose_acc", "accuracy")),
    ("horizon",  "Long Horizon Execution",         "long_horizon_execution",
     ("accuracy",)),
    ("mmlu_pro", "MMLU Pro",                       "mmlu_pro",   ("accuracy",)),
    ("musr",     "MuSR",                           "musr",       ("accuracy",)),
    ("simpleqa", "SimpleQA",                       "simpleqa",   ("accuracy",)),
    ("agieval",  "AGIEval",                        "agieval",    ("accuracy",)),
    ("babi_qa",  "bAbI QA",                        "babi_qa",    ("accuracy",)),
    ("boolq",    "BoolQ",                          "boolq",      ("accuracy",)),
)

BENCH_BY_KEY = {key: (key, label, task, metrics)
                for key, label, task, metrics in BENCH_LIST}

# The generated task says so wherever it is reported. See the header.
GENERATED_MARK = " (generated)"


# ---------------------------------------------------------------------------
# the engine as an inspect-ai model provider
# ---------------------------------------------------------------------------

class EngineSetup:
    """How to run `app_main`, filled in by main() before the backend starts.

    The provider is constructed by inspect-ai, not by this script, so there is
    no constructor to pass these through; they are read off the module instead.
    That is only sound because the eval backend runs in this process.
    """

    def __init__(self):
        self.binary = BINARY
        self.threads = 0
        self.quant = None
        self.context = 4096
        self.max_tokens = 512
        self.temperature = 0.0
        self.seconds = 900.0
        self.keep_thinking = False


ENGINE = EngineSetup()

THINK_END = "</think>"
STATS_LINE = re.compile(r"prefill (\d+) tok .* decode (\d+) tok")


def reply_split(text):
    """Splits a completion into (thought, answer) at the closing think tag.

    A reply that never closes the block is all answer: the run hit the token
    cap, and handing the scorer an empty string would report a formatting
    failure as a wrong answer.
    """
    mark = text.find(THINK_END)
    if mark < 0:
        return "", text.strip()
    thought = text[:mark].replace("<think>", "").strip()
    return thought, text[mark + len(THINK_END):].strip()


def prompt_flatten(messages):
    """Folds a chat message list into the (system, prompt) pair app_main takes.

    `app_main generate` applies the checkpoint's chat template to one system
    message and one user turn. Tasks that carry few shot examples or a graded
    conversation arrive as several turns, so the turns before the last are
    written into the prompt as a transcript. Nothing is dropped.
    """
    system = []
    turns = []
    for message in messages:
        role = getattr(message, "role", "user")
        text = getattr(message, "text", "") or ""
        if role == "system":
            system.append(text)
        elif role == "user":
            turns.append(("User", text))
        elif role == "assistant":
            turns.append(("Assistant", text))
    if len(turns) == 1:
        prompt = turns[0][1]
    else:
        body = "\n\n".join(f"{who}: {text}" for who, text in turns[:-1])
        prompt = f"{body}\n\n{turns[-1][1]}" if body else turns[-1][1]
    return "\n\n".join(part for part in system if part), prompt


def engine_teach():
    """Registers `inferliq/<checkpoint>` with inspect-ai as a model provider.

    Registration is a call rather than an import side effect so that `--chart`
    redraws without any of this installed.
    """
    from functools import partial

    import anyio
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ContentReasoning,
        ContentText,
        GenerateConfig,
        ModelAPI,
        ModelCall,
        ModelOutput,
        ModelUsage,
        modelapi,
    )

    @modelapi(name="inferliq")
    class EngineModel(ModelAPI):
        """One `app_main generate` per sample.

        The engine loads the checkpoint on every call. Memory mapped bf16
        weights cost about a tenth of a second to open, which is small beside
        a reply, and a process per sample is what keeps the provider this
        short -- there is no server in the engine to talk to.
        """

        def __init__(self, model_name, base_url=None, api_key=None,
                     config=GenerateConfig(), **model_args):
            super().__init__(model_name, base_url or checkpoint_mark(model_name),
                             api_key, [], config)
            self.checkpoint = model_name

        async def generate(self, input, tools, tool_choice, config):
            system, prompt = prompt_flatten(input)
            argv = [
                ENGINE.binary, "generate",
                "--model", self.checkpoint,
                "--prompt", prompt,
                "--ctx", str(ENGINE.context),
                "--max-tokens", str(config.max_tokens or ENGINE.max_tokens),
                "--temp", str(ENGINE.temperature if config.temperature is None
                              else config.temperature),
            ]
            if system:
                argv += ["--system", system]
            if ENGINE.threads:
                argv += ["--threads", str(ENGINE.threads)]
            if ENGINE.quant:
                argv += ["--quant", ENGINE.quant]
            if config.top_p is not None:
                argv += ["--top-p", str(config.top_p)]
            if config.top_k is not None:
                argv += ["--top-k", str(config.top_k)]
            if config.seed is not None:
                argv += ["--seed", str(config.seed)]

            text, notes, code = await anyio.to_thread.run_sync(
                partial(engine_call, argv))
            if code != 0 and not text.strip():
                # A checkpoint that will not load, or a flag the engine does
                # not take, must stop the run rather than score zero on every
                # sample: a failed load and a wrong answer are not the same
                # result, and only one of them is about the model.
                raise RuntimeError(
                    "app_main exited %s with nothing on stdout: %s"
                    % (code, " ".join((notes or "").split())[-300:]))
            thought, answer = reply_split(text)
            if ENGINE.keep_thinking:
                thought, answer = "", text.strip()

            content = []
            if thought:
                content.append(ContentReasoning(reasoning=thought))
            content.append(ContentText(text=answer))

            read, made = engine_count(notes)
            cap = config.max_tokens or ENGINE.max_tokens
            output = ModelOutput(
                model=self.model_name,
                choices=[ChatCompletionChoice(
                    message=ChatMessageAssistant(content=content,
                                                 model=self.model_name,
                                                 source="generate"),
                    stop_reason="max_tokens" if made >= cap else "stop")],
                usage=ModelUsage(input_tokens=read, output_tokens=made,
                                 total_tokens=read + made),
            )
            # The log keeps the command that produced this, with the prompt
            # and the system message named rather than repeated -- they are
            # already in the sample beside it, and a bAbI story twice over
            # makes a log nobody opens.
            shown = ["<prompt>" if part is prompt else
                     "<system>" if part is system else part for part in argv]
            call = ModelCall.create(
                {"argv": shown},
                {"exit": code, "completion": answer, "thinking": thought})
            return output, call

        def connection_key(self):
            return self.checkpoint

        def max_connections(self):
            # One process holds every worker thread, so a second one in
            # parallel takes the first one's cores. `--jobs` overrides this
            # from the command line, which is where a many core host wants it.
            return 1

    return EngineModel


def checkpoint_mark(path):
    """A stamp that moves when the weights, the engine or the load plan move.

    The eval backend's tasks ask inspect-ai to cache a reply, and the cache is
    keyed by the prompt and the model name -- which here is a folder. Two
    different checkpoints live at `build/tune/merged` before and after a
    merge, and without this they would share their replies. The stamp is
    handed to inspect-ai as the provider's base url, which is part of that
    key, so a retune is never served the last tune's answers.
    """
    marks = []
    if os.path.isdir(path):
        for leaf in sorted(os.listdir(path)):
            if leaf.endswith((".safetensors", ".json")):
                found = os.stat(os.path.join(path, leaf))
                marks.append(f"{leaf}:{found.st_size}:{int(found.st_mtime)}")
    if os.path.exists(ENGINE.binary):
        marks.append("engine:%d" % int(os.stat(ENGINE.binary).st_mtime))
    marks.append("quant:%s" % (ENGINE.quant or "none"))
    return "inferliq://" + hashlib.sha256("|".join(marks).encode()).hexdigest()[:16]


def engine_call(argv):
    """Runs app_main once and returns (stdout, stderr, exit code).

    A run that overruns `--seconds` is killed and whatever it printed by then
    is the reply. The alternative -- throwing the sample away -- would quietly
    raise the score by scoring only the samples the engine happened to finish.
    """
    process = subprocess.Popen(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               errors="replace")
    try:
        text, notes = process.communicate(timeout=ENGINE.seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        text, notes = process.communicate()
        notes = (notes or "") + "\n[killed at the --seconds wall]"
    return text or "", notes or "", process.returncode


def engine_count(notes):
    """Reads the prefill and decode token counts app_main prints on stderr."""
    found = STATS_LINE.search(notes or "")
    if not found:
        return 0, 0
    return int(found.group(1)), int(found.group(2))


def engine_build(skip):
    """Builds app_main through util/make.py unless it is already there."""
    if os.path.exists(ENGINE.binary):
        return True
    if skip:
        print(f"no engine at {ENGINE.binary}, and --no-build was asked for",
              file=sys.stderr)
        return False
    if ENGINE.binary != BINARY:
        print(f"no engine at {ENGINE.binary}", file=sys.stderr)
        return False
    print("> building the engine")
    code = subprocess.call([sys.executable, os.path.join(ROOT, "util", "make.py"), "build"])
    return code == 0 and os.path.exists(BINARY)


# ---------------------------------------------------------------------------
# Long Horizon Execution, generated
# ---------------------------------------------------------------------------

HORIZON_KEYS = ("bex", "dar", "fim", "gol", "hub", "jat", "kev", "lom",
                "mub", "nar", "pol", "qis", "rud", "sev", "tob", "urn",
                "vax", "web", "yol", "zad", "bim", "cor", "dun", "elk",
                "fap", "gir")
HORIZON_VALS = ("q7", "z3", "m8", "k2", "w5", "b9", "t4", "x6", "r1", "n0",
                "j5", "h2", "v8", "c3", "p6", "d9", "f4", "g7", "l1", "s0",
                "y2", "a5", "e8", "u3", "i6", "o9")

HORIZON_ASK = """You are executing a plan, one step at a time.

Dictionary:
{table}

Plan ({count} steps):
{plan}

Work through the plan in order. At each step look up that step's key in the
dictionary and append its two character value to a running string. Do not skip
a step and do not reorder them. When the last step is done, write the finished
string on a line of its own, like this:

ANSWER: <string>"""


def horizon_make(house, horizons, count, seed):
    """Writes the generated long horizon rows, and returns the folder.

    The folder name carries the settings that produced it, because inspect-ai
    caches a dataset by its path: a second run at other settings must land
    somewhere else or it would be served the first run's rows.

    Rows are written round robin across the plan lengths rather than in blocks,
    so that `--samples N` -- which takes the first N rows -- keeps every length
    in the sample rather than only the shortest ones.
    """
    tag = "h%s-n%d-s%d" % ("_".join(str(step) for step in horizons), count, seed)
    folder = os.path.join(house, "long-horizon", tag)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "test.jsonl")

    dice = random.Random(seed)
    rows = {step: [] for step in horizons}
    for step in horizons:
        for index in range(count):
            keys = list(HORIZON_KEYS)
            vals = list(HORIZON_VALS)
            dice.shuffle(vals)
            book = dict(zip(keys, vals))
            plan = [dice.choice(keys) for _ in range(step)]
            table = "\n".join(f"  {key} = {book[key]}" for key in keys)
            listing = "\n".join(f"  {place + 1}. look up {key}"
                                for place, key in enumerate(plan))
            rows[step].append({
                "id": f"h{step}-{index}",
                "horizon": step,
                "prompt": HORIZON_ASK.format(table=table, count=step,
                                             plan=listing),
                "answer": "".join(book[key] for key in plan),
            })

    with open(path, "w", encoding="utf-8") as fh:
        for index in range(count):
            for step in horizons:
                fh.write(json.dumps(rows[step][index]) + "\n")
    return folder


# ---------------------------------------------------------------------------
# the two task configs the eval backend is missing
# ---------------------------------------------------------------------------

def tasks_teach(folder, grader=None):
    """Adds what the backend cannot run, and returns every config it knows.

    lighteval builds its task registry by importing its own task modules, and
    the eval backend passes no custom task file through, so the way to add a
    task is to wrap the loader. Anything lighteval already has wins: when a
    later version ships either of these two, the version installed is used and
    what is written here is never reached.
    """
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import (
        CORRECT, INCORRECT, Score, accuracy, match, model_graded_fact, scorer,
        stderr,
    )
    from inspect_ai.solver import generate
    from lighteval.metrics.metrics import Metrics
    from lighteval.tasks.lighteval_task import LightevalTaskConfig
    from lighteval.tasks.registry import Registry

    answer_line = re.compile(r"ANSWER\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)

    @scorer(metrics=[accuracy(), stderr()])
    def horizon_scorer():
        """Exact match on the executed string, wherever the model put it.

        The reply is read for the last `ANSWER:` and, failing that, for the
        last word of it, so that a model which ignores the format but executes
        the plan is still marked on the execution. The colon is required
        rather than optional, because "the answer is" without one would hand
        the scorer the word "is". The comparison itself is exact: a plan of 32
        steps has one right answer and 32 ways to be one step wrong.
        """
        async def score(state, target):
            text = state.output.completion or ""
            found = answer_line.findall(text)
            if found:
                said = found[-1]
            else:
                lines = [line.split() for line in text.splitlines() if line.split()]
                said = lines[-1][-1] if lines else ""
            said = re.sub(r"[^A-Za-z0-9]", "", said)
            want = re.sub(r"[^A-Za-z0-9]", "", target.text)
            right = bool(want) and said.lower() == want.lower()
            return Score(
                value=CORRECT if right else INCORRECT,
                answer=said,
                explanation="horizon %s" % (state.metadata or {}).get("horizon"),
            )
        return score

    def horizon_sample(record):
        return Sample(input=record["prompt"], target=record["answer"],
                      id=record["id"], metadata={"horizon": record["horizon"]})

    horizon_task = LightevalTaskConfig(
        name="long_horizon_execution",
        # `prompt_function` is what the other backends build a prompt with and
        # is required by the config; the eval backend reads `sample_fields`
        # instead and never calls it.
        prompt_function=lambda line, task_name=None: None,
        hf_repo=folder,
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        metrics=[Metrics.exact_match],
        generation_size=1024,
        stop_sequence=[],
        version=0,
        sample_fields=horizon_sample,
        solver=[generate(cache=True)],
        scorer=horizon_scorer(),
    )

    def babi_sample(record):
        """One bAbI record is a story with several questions inside it.

        lighteval's own `babi_qa_prompt` cannot be reused for this: it decodes
        every answer as a compass path -- "n,w" to "north west" -- before it
        looks at what the question asks, so it raises a KeyError on every
        subtask whose answer is a word, `en-valid-qa1` among them, which is
        the subset the task config selects. What is below is that function
        with the decode moved to the questions that ask for a path, which are
        the ones carrying the `_` it fills in. inspect-ai takes a list of
        samples from a record mapper, so the stories need no flattening pass.

        Two things are kept as lighteval has them rather than improved, so
        that the rows are the rows lighteval would have scored. The story
        resets after each question, so a later question sees only the facts
        stated since the one before it. And filling `_` in with the decoded
        answer puts that answer into the question -- which is lighteval's
        line, and which the selected subset never reaches, because only the
        path finding subtask carries the placeholder.
        """
        story = record["story"]
        if not isinstance(story, dict):
            story = json.loads(story)
        compass = {"s": "south", "n": "north", "e": "east", "w": "west"}
        samples, told = [], []
        for kind, text, answer in zip(story["type"], story["text"],
                                      story["answer"]):
            if kind == "supporting fact":
                told.append(text)
            elif kind == "question":
                if "_" in text:
                    text = text.replace("_", " ".join(
                        compass.get(step, step) for step in answer.split(",")))
                query = "\n".join(told)
                samples.append(Sample(
                    input=f"{query}\nQuestion: {text}\nAnswer with one word.",
                    target=answer))
                told = []
        return samples

    def babi_fill(config):
        config.sample_fields = babi_sample
        config.solver = [generate(cache=True)]
        # bAbI answers are one word -- a room, a person, a direction -- and the
        # reply is a sentence around it, so the match is on the word appearing
        # rather than on the reply being it.
        config.scorer = match(location="any", ignore_case=True)
        return config

    plain = Registry.load_all_task_configs

    def load_all(custom_tasks=None, load_multilingual=False):
        configs = plain(custom_tasks=custom_tasks,
                        load_multilingual=load_multilingual)
        if not any(name.startswith("long_horizon") for name in configs):
            configs[horizon_task.name] = horizon_task
        found = configs.get("babi_qa")
        if found is not None and found.sample_fields is None:
            configs["babi_qa"] = babi_fill(found)
        if grader:
            marked = configs.get("simpleqa")
            if marked is not None:
                marked.scorer = model_graded_fact(model=grader)
        return configs

    Registry.load_all_task_configs = staticmethod(load_all)
    # A registry built with no task list loads every config and selects none,
    # which is how the names below are checked before a run starts.
    return Registry(tasks=None)._task_registry


# ---------------------------------------------------------------------------
# running the eval backend
# ---------------------------------------------------------------------------

def tasks_name(known, task):
    """Maps a benchmark's task name onto what the installed lighteval calls it."""
    if task in known:
        return task
    if any(name.split(":")[0] == task for name in known):
        return task
    if task == "long_horizon_execution":
        for name in sorted(known):
            if name.startswith("long_horizon"):
                return name
    return None


def backend_mend():
    """Drops the generate options lighteval sends that inspect-ai has dropped.

    lighteval 0.13 hands the backend three fields inspect-ai has since renamed
    or removed -- `frequence_penalty`, `log_probs` and `response_format` --
    and a recent inspect-ai refuses a field it does not know, so the two
    installed together will not start a run untouched. All three arrive as
    `None`, because nothing on this side sets them, and a `None` the backend
    has no field for is nothing: it is dropped here. An option that is
    actually set is passed on and allowed to fail, since a request quietly
    thrown away is worse than the error that says so.
    """
    import inspect as reflect

    import lighteval.main_inspect as backend
    from inspect_ai.model import GenerateConfig

    if getattr(backend.inspect_ai_eval_set, "_mended", False):
        return
    taken = set(reflect.signature(backend.inspect_ai_eval_set).parameters)
    known = set(GenerateConfig.model_fields)
    plain = backend.inspect_ai_eval_set

    def eval_set(*args, **kwargs):
        kept, gone = {}, []
        for name, value in kwargs.items():
            if name in taken or name in known or value is not None:
                kept[name] = value
            else:
                gone.append(name)
        if gone:
            print("> this inspect-ai has no " + ", ".join(sorted(gone))
                  + "; unset, so dropped", file=sys.stderr)
        return plain(*args, **kept)

    eval_set._mended = True
    backend.inspect_ai_eval_set = eval_set


def eval_run(models, benches, known, args, house):
    """Hands the tasks to `lighteval eval` and returns the log folder."""
    from lighteval.main_inspect import eval as lighteval_eval

    backend_mend()

    specs = []
    for key, label, task, _metrics in benches:
        name = tasks_name(known, task)
        if name is None:
            print(f"! {label}: no task named {task} in this lighteval, skipped",
                  file=sys.stderr)
            continue
        specs.append(name)
    if not specs:
        return None

    logs = os.path.join(house, "logs")
    display = args.display
    if display == "rich" and not sys.stdout.isatty():
        display = "plain"

    lighteval_eval(
        models=models,
        tasks=",".join(specs),
        max_samples=args.samples or None,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        max_connections=args.jobs,
        timeout=int(args.seconds),
        log_dir=logs,
        display=display,
        epochs=args.epochs,
    )
    return logs


# ---------------------------------------------------------------------------
# reading the eval set back
# ---------------------------------------------------------------------------

def logs_read(logs, models):
    """Collects (model, task) -> {metric: value} from an inspect-ai log folder.

    The folder is reused between runs so that a second run resumes rather than
    starts again, which means it also holds older logs; the newest log for a
    pair is the one that counts, and anything that did not finish is dropped
    rather than reported as a zero.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    wanted = set(models)
    found = {}
    for info in list_eval_logs(logs):
        try:
            log = read_eval_log(info, header_only=True)
        except Exception as bad:                        # a half written log
            print(f"! unreadable log {info}: {bad}", file=sys.stderr)
            continue
        if log.status != "success" or not log.results:
            continue
        if wanted and log.eval.model not in wanted:
            continue
        stamp = log.eval.created
        place = (log.eval.model, log.eval.task)
        if place in found and found[place][0] > stamp:
            continue
        metrics = {}
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                metrics[name] = metric.value
        metrics["_samples"] = log.results.completed_samples
        found[place] = (stamp, metrics, info)
    return {place: (metrics, info) for place, (stamp, metrics, info) in found.items()}


def scores_gather(read, models, benches):
    """Folds subtask scores into one number per benchmark per model.

    AGIEval is seventeen exams and MuSR is three, so the benchmark's score is
    the mean over its subtasks -- the same fold lighteval's own table does --
    and the parts are kept beside it in results.json.
    """
    scores = {}
    for model in models:
        scores[model] = {}
        for key, label, task, metrics in benches:
            parts = {}
            for (who, name), (found, _info) in read.items():
                if who != model:
                    continue
                stem = name.split(":")[0].split("|")[0]
                if stem != task and not stem.startswith(task):
                    continue
                for metric in metrics:
                    if metric in found:
                        parts[name] = {"metric": metric,
                                       "value": float(found[metric]),
                                       "stderr": float(found.get("stderr", 0.0)),
                                       "samples": found.get("_samples", 0)}
                        break
            if not parts:
                continue
            values = [part["value"] for part in parts.values()]
            first = parts[sorted(parts)[0]]
            scores[model][key] = {
                "value": sum(values) / len(values),
                "metric": first["metric"],
                "samples": sum(part["samples"] for part in parts.values()),
                "stderr": first["stderr"] if len(parts) == 1 else 0.0,
                "parts": {name: part["value"] for name, part in parts.items()},
            }
    return scores


def horizon_gather(read, models):
    """Accuracy against plan length, from the long horizon task's own samples."""
    from inspect_ai.log import read_eval_log

    curve = {}
    for (model, task), (_metrics, info) in read.items():
        if not task.split(":")[0].startswith("long_horizon") or model not in models:
            continue
        try:
            log = read_eval_log(info, resolve_attachments=False)
        except Exception as bad:
            print(f"! could not read the long horizon samples: {bad}",
                  file=sys.stderr)
            continue
        tally = {}
        for sample in log.samples or []:
            step = (sample.metadata or {}).get("horizon")
            if step is None:
                continue
            right = 0
            for score in (sample.scores or {}).values():
                right = 1 if score.value in (1, 1.0, True, "C") else 0
            made, seen = tally.get(step, (0, 0))
            tally[step] = (made + right, seen + 1)
        if tally:
            curve[model] = {str(step): made / seen
                            for step, (made, seen) in sorted(tally.items())}
    return curve


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def bench_label(key, label):
    return label + GENERATED_MARK if key == "horizon" else label


def table_write(path, book):
    """Writes the markdown table, and returns it."""
    models = book["models"]
    keys = [key for key in book["order"]
            if any(key in book["scores"].get(model, {}) for model in models)]

    head = "| benchmark | " + " | ".join(model_short(model) for model in models) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in models) + " |"
    lines = [head, rule]
    for key in keys:
        label = bench_label(key, BENCH_BY_KEY[key][1])
        cells = []
        for model in models:
            score = book["scores"].get(model, {}).get(key)
            cells.append("%.1f" % (100.0 * score["value"]) if score else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    cap = samples_say(book["samples"])
    note = (f"\n{cap}, {book['made'][:10]}. "
            "Long Horizon Execution is generated by `util/eval.py`, not the "
            "paper's rows; see the header of that file.\n")
    text = "\n".join(lines) + "\n" + note
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Benchmark scores\n\n" + text)
    return text


def samples_say(count):
    """`--samples 0` scores the whole task, and the report has to say which."""
    return f"{count} samples per task" if count else "every row of each task"


def model_short(model):
    """`inferliq/build/tune/merged` reads as `merged`; a hub path as its leaf."""
    if model.startswith("inferliq/"):
        leaf = model.split("/")[-1]
        return f"engine: {leaf}"
    return model.split("/")[-1]


# ---------------------------------------------------------------------------
# the chart
# ---------------------------------------------------------------------------

# Categorical slots 1 to 8 of the validated default palette, light and dark.
# The order is the palette's own and is the colour blindness mechanism, so it
# is taken as given rather than re-picked here.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767")

BAR_THICK = 18          # <= 24, and the leftover of the band is air
BAR_GAP = 2             # the surface gap between touching bars
GROUP_PAD = 20
PLOT_RIGHT = 54         # room for the value at the bar's tip
CHART_WIDE = 940


def svg_quote(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def text_span(text, size):
    """A workable width for a sans face at this size, for fitting labels."""
    return len(str(text)) * size * 0.56


def chart_write(path, book):
    """Draws the results as one SVG: bars per benchmark, then the horizon curve.

    Written by hand rather than with a plotting library because the engine's
    side of this repository has no dependencies and the chart does not need
    one: an SVG is text, and the file it writes opens in a browser, embeds in
    the README, and reads the same in a light or a dark theme.
    """
    models = [model for model in book["models"] if book["scores"].get(model)]
    keys = [key for key in book["order"]
            if any(key in book["scores"].get(model, {}) for model in models)]
    if not models or not keys:
        return None

    curve = {model: book.get("horizon", {}).get(model)
             for model in models
             if book.get("horizon", {}).get(model)}
    steps = sorted({int(step) for model in curve for step in curve[model]})

    # The left column is measured from the longest name rather than fixed:
    # `Long Horizon Execution (generated)` is half as wide again as `MuSR`,
    # and a label that does not fit is a label that gets cut in half.
    labels = {key: bench_label(key, BENCH_BY_KEY[key][1]) for key in keys}
    plot_x = int(min(300, max(150, max(text_span(name, 13)
                                       for name in labels.values()) + 28)))
    plot_w = CHART_WIDE - plot_x - PLOT_RIGHT

    band = len(models) * (BAR_THICK + BAR_GAP) - BAR_GAP + GROUP_PAD
    head = 104 + (26 if len(models) > 1 else 0)
    bars_high = band * len(keys)
    axis_high = 34
    curve_high = 250 if steps else 0
    high = head + bars_high + axis_high + curve_high + 58

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDE}" '
        f'height="{high}" viewBox="0 0 {CHART_WIDE} {high}" '
        f'font-family="ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    out.append(chart_style(models))
    out.append(f'<rect width="{CHART_WIDE}" height="{high}" fill="var(--surface)"/>')

    title = "Benchmark scores"
    if len(models) == 1:
        title = f"Benchmark scores — {model_short(models[0])}"
    out.append(f'<text x="24" y="40" class="title">{svg_quote(title)}</text>')
    when = book["made"][:10]
    under = f"per cent correct · {samples_say(book['samples'])} · {when}"
    out.append(f'<text x="24" y="64" class="note">{svg_quote(under)}</text>')

    top = head
    if len(models) > 1:
        out.append(chart_legend(models, 24, head - 20))

    # the gridlines, behind everything, one step off the surface
    axis_y = top + bars_high
    for mark in range(0, 101, 20):
        at = plot_x + plot_w * mark / 100.0
        out.append(f'<line x1="{at:.1f}" y1="{top - 6}" x2="{at:.1f}" '
                   f'y2="{axis_y}" class="grid"/>')
        out.append(f'<text x="{at:.1f}" y="{axis_y + 20}" class="tick" '
                   f'text-anchor="middle">{mark}</text>')

    for row, key in enumerate(keys):
        band_top = top + row * band
        middle = band_top + (band - GROUP_PAD) / 2.0
        out.append(label_draw(labels[key], plot_x - 16, middle))
        for index, model in enumerate(models):
            score = book["scores"].get(model, {}).get(key)
            y = band_top + index * (BAR_THICK + BAR_GAP)
            if not score:
                out.append(f'<text x="{plot_x + 6}" y="{y + BAR_THICK - 4:.1f}" '
                           f'class="tick">not run</text>')
                continue
            value = 100.0 * score["value"]
            width = max(0.0, plot_w * value / 100.0)
            out.append(bar_path(plot_x, y, width, BAR_THICK, index))
            out.append(f'<text x="{plot_x + width + 8:.1f}" '
                       f'y="{y + BAR_THICK - 4:.1f}" class="value">'
                       f'{value:.1f}</text>')

    out.append(f'<line x1="{plot_x}" y1="{axis_y}" x2="{plot_x + plot_w}" '
               f'y2="{axis_y}" class="axis"/>')

    if steps:
        out.append(chart_curve(curve, steps, models, plot_x, plot_w,
                               axis_y + axis_high + 30, curve_high - 60))

    foot = high - 20
    out.append(f'<text x="24" y="{foot}" class="note">'
               f'{svg_quote(chart_foot(book))}</text>')
    out.append("</svg>")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def label_draw(label, x, middle):
    """A benchmark's name, on two lines when one will not fit the column."""
    room = x - 24
    if text_span(label, 13) <= room or " " not in label:
        return (f'<text x="{x:.1f}" y="{middle + 5:.1f}" class="label" '
                f'text-anchor="end">{svg_quote(label)}</text>')
    words = label.split(" ")
    cut = len(words) // 2
    while cut > 1 and text_span(" ".join(words[:cut]), 13) > room:
        cut -= 1
    head, tail = " ".join(words[:cut]), " ".join(words[cut:])
    return (f'<text x="{x:.1f}" y="{middle - 2:.1f}" class="label" '
            f'text-anchor="end">{svg_quote(head)}</text>\n'
            f'<text x="{x:.1f}" y="{middle + 14:.1f}" class="label" '
            f'text-anchor="end">{svg_quote(tail)}</text>')


def chart_foot(book):
    engine = book.get("engine") or {}
    parts = []
    if engine.get("quant"):
        parts.append(f"weights {engine['quant']}")
    if engine.get("threads"):
        parts.append(f"{engine['threads']} threads")
    if engine.get("max_tokens"):
        parts.append(f"reply cap {engine['max_tokens']} tokens")
    where = " · ".join(parts)
    made = "Long Horizon Execution is generated by util/eval.py, not the paper's rows"
    return f"{made}{' · ' + where if where else ''}"


def chart_style(models):
    """Light and dark are two selected sets of the same palette, not a flip."""
    light = "\n".join(f"  --s{index + 1}: {SERIES_LIGHT[index % 8]};"
                      for index in range(len(models)))
    dark = "\n".join(f"    --s{index + 1}: {SERIES_DARK[index % 8]};"
                     for index in range(len(models)))
    return f"""<style>
svg {{
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --line: #e4e3df;
{light}
}}
@media (prefers-color-scheme: dark) {{
  svg {{
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --line: #34332f;
{dark}
  }}
}}
.title {{ fill: var(--ink); font-size: 19px; font-weight: 600; }}
.note  {{ fill: var(--ink-2); font-size: 12px; }}
.label {{ fill: var(--ink); font-size: 13px; }}
.value {{ fill: var(--ink); font-size: 12px; font-variant-numeric: tabular-nums; }}
.tick  {{ fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }}
.key   {{ fill: var(--ink); font-size: 12px; }}
.grid  {{ stroke: var(--line); stroke-width: 1; }}
.axis  {{ stroke: var(--line); stroke-width: 1; }}
.ring  {{ stroke: var(--surface); stroke-width: 2; }}
</style>"""


def bar_path(x, y, width, thick, index):
    """A bar with a 4px rounded tip and a square foot at the baseline."""
    fill = f"var(--s{index % 8 + 1})"
    radius = min(4.0, max(0.0, width))
    if width <= 0.6:
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="1.5" height="{thick}" '
                f'fill="{fill}" opacity="0.5"/>')
    return (f'<path d="M{x:.1f},{y:.1f} H{x + width - radius:.1f} '
            f'a{radius:.1f},{radius:.1f} 0 0 1 {radius:.1f},{radius:.1f} '
            f'V{y + thick - radius:.1f} '
            f'a{radius:.1f},{radius:.1f} 0 0 1 -{radius:.1f},{radius:.1f} '
            f'H{x:.1f} Z" fill="{fill}"/>')


def chart_legend(models, x, y):
    """Identity never rests on colour alone: every series is named here."""
    out = []
    at = x
    for index, model in enumerate(models):
        name = model_short(model)
        out.append(f'<rect x="{at:.1f}" y="{y - 9:.1f}" width="10" height="10" '
                   f'rx="2" fill="var(--s{index % 8 + 1})"/>')
        out.append(f'<text x="{at + 16:.1f}" y="{y:.1f}" class="key">'
                   f'{svg_quote(name)}</text>')
        at += 16 + text_span(name, 12) + 22
    return "\n".join(out)


def chart_curve(curve, steps, models, x, width, top, high):
    """Accuracy against plan length -- the one view the benchmark is about."""
    out = [f'<text x="24" y="{top - 22}" class="title" font-size="15">'
           f'Long Horizon Execution — accuracy by plan length</text>']
    # Each line is named at its end, so the panel gives back the room those
    # names need rather than running them off the edge of the drawing.
    if len(models) > 1:
        width -= max(text_span(model_short(model), 12)
                     for model in models) + 20
    spread = width / max(1, len(steps) - 1) if len(steps) > 1 else 0

    for mark in range(0, 101, 25):
        at = top + high - high * mark / 100.0
        out.append(f'<line x1="{x}" y1="{at:.1f}" x2="{x + width}" '
                   f'y2="{at:.1f}" class="grid"/>')
        out.append(f'<text x="{x - 10}" y="{at + 4:.1f}" class="tick" '
                   f'text-anchor="end">{mark}</text>')

    for index, step in enumerate(steps):
        at = x + index * spread
        out.append(f'<text x="{at:.1f}" y="{top + high + 20:.1f}" class="tick" '
                   f'text-anchor="middle">{step}</text>')
    out.append(f'<text x="{x + width / 2:.1f}" y="{top + high + 40:.1f}" '
               f'class="note" text-anchor="middle">steps in the plan</text>')

    for index, model in enumerate(models):
        walk = curve.get(model)
        if not walk:
            continue
        stroke = f"var(--s{index % 8 + 1})"
        points = []
        for place, step in enumerate(steps):
            value = walk.get(str(step))
            if value is None:
                continue
            points.append((x + place * spread,
                           top + high - high * min(1.0, max(0.0, value))))
        if not points:
            continue
        line = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        out.append(f'<polyline points="{line}" fill="none" stroke="{stroke}" '
                   f'stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round"/>')
        for px, py in points:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
                       f'fill="{stroke}" class="ring"/>')
        if len(models) > 1:
            px, py = points[-1]
            out.append(f'<text x="{px + 10:.1f}" y="{py + 4:.1f}" class="key">'
                       f'{svg_quote(model_short(model))}</text>')
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def benches_pick(names):
    if not names:
        return list(BENCH_LIST)
    picked = []
    for name in names.split(","):
        name = name.strip()
        if not name:
            continue
        if name in BENCH_BY_KEY:
            picked.append(BENCH_BY_KEY[name])
            continue
        found = [bench for bench in BENCH_LIST
                 if bench[2] == name or bench[1].lower() == name.lower()]
        if not found:
            raise SystemExit(f"unknown benchmark: {name}\n"
                             f"pick from: {', '.join(BENCH_BY_KEY)}")
        picked.append(found[0])
    return picked


def main(argv=None):
    parse = argparse.ArgumentParser(
        description="score a checkpoint on the public benchmarks and draw it",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("--model", action="append", default=[],
                       help="checkpoint folder, run through build/app_main "
                            "(repeat for more than one)")
    parse.add_argument("--rival", action="append", default=[],
                       help="an inspect-ai model string to score beside it, "
                            "e.g. hf-inference-providers/Qwen/Qwen3-4B")
    parse.add_argument("--tasks", default=None,
                       help="a subset of: " + ", ".join(BENCH_BY_KEY))
    parse.add_argument("--samples", type=int, default=25,
                       help="rows scored per task, 0 for all (default 25)")
    parse.add_argument("--epochs", type=int, default=1,
                       help="times each sample is answered (default 1)")
    parse.add_argument("--max-tokens", type=int, default=512,
                       help="reply cap in tokens (default 512)")
    parse.add_argument("--temp", type=float, default=0.0,
                       help="sampling temperature, 0 for greedy (default 0)")
    parse.add_argument("--threads", type=int, default=0,
                       help="engine worker threads (default: the host's cores)")
    parse.add_argument("--quant", choices=("q8", "none"), default="none",
                       help="repack the weights at load (default none)")
    parse.add_argument("--ctx", type=int, default=4096,
                       help="engine context window (default 4096)")
    parse.add_argument("--jobs", type=int, default=1,
                       help="engine processes in parallel (default 1)")
    parse.add_argument("--seconds", type=float, default=900.0,
                       help="wall clock cap on one reply (default 900)")
    parse.add_argument("--grader", default=None,
                       help="model that grades SimpleQA (default: the model "
                            "under test, which grades itself)")
    parse.add_argument("--binary", default=BINARY,
                       help="the engine to run (default build/app_main)")
    parse.add_argument("--keep-thinking", action="store_true",
                       help="score the reasoning span too, not just the answer")
    parse.add_argument("--horizons", default="2,4,8,16,32",
                       help="plan lengths for Long Horizon Execution")
    parse.add_argument("--horizon-samples", type=int, default=10,
                       help="generated rows per plan length (default 10)")
    parse.add_argument("--seed", type=int, default=1729,
                       help="seed for the generated task (default 1729)")
    parse.add_argument("--out", default=HOUSE,
                       help="where the logs, results and chart go")
    parse.add_argument("--display", default="rich",
                       choices=("rich", "full", "plain", "log", "none"),
                       help="how the backend reports progress")
    parse.add_argument("--no-build", action="store_true",
                       help="do not build the engine first")
    parse.add_argument("--chart", action="store_true",
                       help="redraw the chart and table from results.json "
                            "without running anything")
    args = parse.parse_args(argv)

    house = os.path.abspath(args.out)
    os.makedirs(house, exist_ok=True)
    kept = os.path.join(house, "results.json")

    if args.chart:
        if not os.path.exists(kept):
            raise SystemExit(f"no results to draw at {kept}")
        with open(kept, encoding="utf-8") as fh:
            book = json.load(fh)
        print(table_write(os.path.join(house, "results.md"), book))
        drawn = chart_write(os.path.join(house, "chart.svg"), book)
        print(f"chart  {drawn or 'not drawn, nothing scored'}")
        return 0

    models = [f"inferliq/{path}" for path in args.model] + list(args.rival)
    if not models:
        raise SystemExit("nothing to score: pass --model PATH or --rival NAME")

    ENGINE.binary = os.path.abspath(args.binary)
    ENGINE.threads = args.threads
    ENGINE.quant = None if args.quant == "none" else args.quant
    ENGINE.context = args.ctx
    ENGINE.max_tokens = args.max_tokens
    ENGINE.temperature = args.temp
    ENGINE.seconds = args.seconds
    ENGINE.keep_thinking = args.keep_thinking

    # The task list and the plan lengths are read before anything is built,
    # so that a typo in --tasks costs a message rather than a compile.
    benches = benches_pick(args.tasks)
    horizons = [int(step) for step in args.horizons.split(",") if step.strip()]

    if args.model and not engine_build(args.no_build):
        raise SystemExit("the engine did not build; run python3 util/make.py build")

    engine_teach()
    folder = horizon_make(house, horizons, args.horizon_samples, args.seed)
    known = tasks_teach(folder, grader=args.grader)

    logs = eval_run(models, benches, known, args, house)
    if logs is None:
        raise SystemExit("no task ran")

    read = logs_read(logs, models)
    book = {
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": args.samples,
        "models": models,
        "order": [bench[0] for bench in benches],
        "scores": scores_gather(read, models, benches),
        "horizon": horizon_gather(read, models),
        "engine": {
            "binary": ENGINE.binary,
            "threads": ENGINE.threads,
            "quant": args.quant,
            "ctx": args.ctx,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
        },
        "generated": {"horizons": horizons, "per_horizon": args.horizon_samples,
                      "seed": args.seed, "folder": folder},
    }
    with open(kept, "w", encoding="utf-8") as fh:
        json.dump(book, fh, indent=2, sort_keys=True)

    print()
    print(table_write(os.path.join(house, "results.md"), book))
    drawn = chart_write(os.path.join(house, "chart.svg"), book)
    print(f"results  {kept}")
    print(f"chart    {drawn or 'not drawn, nothing scored'}")
    print(f"logs     {logs}  (inspect view --log-dir {logs})")

    if not any(book["scores"].get(model) for model in models):
        print("\nnothing scored: every task the backend ran failed, and the "
              "log says why", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
