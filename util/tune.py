#!/usr/bin/env python3
"""util/tune.py - tunes the checkpoint into caveman speech, and repacks it.

The engine reads the Hugging Face checkpoint directly and has no trainer of
its own, so a tune happens on the reference side: a LoRA adapter trains over
the frozen base weights, and the adapter is then folded back into the float
checkpoint, which the engine already reads without conversion. The tune is a
workflow helper like the other `.py` files, not part of the engine.

What it tunes for
-----------------

Caveman is a compression register, not a personality: drop articles, filler,
hedging and pleasantries; keep every technical fact, every number, every
negation, and every byte of code. It is described in full by the `caveman`
skill at https://github.com/JuliusBrussee/caveman, and the rules below are
that skill's rules, mechanised.

LFM2.5 is a thinking model. Its chat template ends every generation prompt
with `<|im_start|>assistant\\n<think>`, so the model reasons before it
answers and **the reasoning is where most of the tokens are**. A tune that
compresses only the visible answer leaves the larger half untouched, so this
one trains both spans, at separate intensities: the thought at `ultra`, the
answer at `full`. The level is a system prompt - `Caveman mode: full.`,
`ultra`, `lite`, or `Normal mode.` - so the register stays a knob rather than
a permanent change of voice, and `Normal mode.` rows are carried in the
corpus so the knob has an off position that still works.

Six steps, each needing only what the one before it produced. `--uncensor` is
the one that is optional -- the tune runs with it or without it -- and every
step after it reads what it wrote:

    python3 util/tune.py --lint                                  audit the corpus
    python3 util/tune.py --make-data --source raw.jsonl          grow the corpus
    python3 util/tune.py --model ckpt/0.4b --uncensor            abliterate refusals
    python3 util/tune.py --model ckpt/0.4b --train
    python3 util/tune.py --model ckpt/0.4b --merge
    python3 util/tune.py --model ckpt/0.4b --check --tuned build/tune/merged

They also compose, which is the point of the chaining: one command takes the
published weights to a decensored, tuned checkpoint with nobody watching it.

    python3 util/tune.py --model ckpt/0.4b --uncensor --train --merge

`--train` writes `build/tune/adapter` (the PEFT adapter) and `build/tune/`
(its trainer state). `--merge` folds the adapter into the base weights and
writes `build/tune/merged`, a checkpoint in the same layout as `ckpt/0.4b`
that the engine and the reference both read directly:

    python3 util/make.py run -- generate --model build/tune/merged --prompt "Hello!"

`--check` is the number that says whether the tune worked: it runs the base
and the tuned checkpoint over the held out rows and reports how far the
output shrank beside how much of the answer survived.

The corpus
----------

`data/tune.jsonl` beside this script, one JSON object per line:

    prompt      required, the user turn
    thinking    required, the reasoning that goes inside <think>...</think>
    completion  required, the visible answer
    level       lite | full | ultra | off, default full
    system      overrides the level's system prompt when a row needs its own
    kind        reason | guard | prose | fact | mode, for the audit only
    check       a string the answer must contain, used by --check
    split       train | eval, default train

`prompt` and `completion` are the two keys the older corpus carried, so old
rows still parse; `thinking` is new and is required, because a row without it
trains the model on a shape it is never asked to produce. See "Why the think
block is not optional" below.

Two checks keep the corpus honest, and they run at different times.

`--make-data` holds every row it presses to three rules, and throws away the
row rather than repairing it when one is broken:

  1. **Deletion, not paraphrase.** Every pressed row's words are a subsequence
     of the source's, once a closed list of phrase rewrites has been applied.
     A transform that can only delete function words cannot introduce a claim
     the source did not make, which is the property that protects accuracy.
  2. **Guarded spans are copied byte for byte.** Fenced code, inline code,
     URLs, quoted error strings, anything with a digit in it, anything with an
     underscore, a slash or an interior full stop, and any word in capitals.
  3. **Negations and numbers survive.** `not`, `never`, `no`, `only`,
     `except`, `without`, `unless` and every numeral are counted on both sides
     and must match. Dropping a `not` costs more than every token it saves.

`--lint` audits what is already written - the hand written rows, which no
press produced and which those three rules therefore say nothing about -
against the register itself: no articles at `full` or `ultra`, no filler, no
hedging, no invented abbreviations, no arrows, one idea per sentence at twenty
words, and the boundary rule that keeps a warning or a commit message in full
sentences. It also checks that a row's `check` string is in its own answer.
`--lint` returning non-zero stops the tune, because a corpus that fails its
own audit teaches the audit's failures.

Why the think block is not optional
-----------------------------------

The chat template renders an assistant turn as
`<|im_start|>assistant\\n<think>THOUGHT</think>ANSWER<|im_end|>` when the turn
carries a `thinking` field, and as `<|im_start|>assistant\\nANSWER<|im_end|>`
when it does not. Inference always primes with `<think>`. Training on rows
with no thinking therefore teaches the model to answer in a position it never
occupies, and the first thing it does at inference is fall out of the shape it
was tuned into. Earlier revisions of this script did exactly that. Rows
without `thinking` are now an error rather than a warning.

What the adapter reaches
------------------------

The Liquid stack has two operators. An earlier note here said the tune could
only reach the attention layers because the convolution is not a linear layer
a LoRA adapter attaches to. That is true of the depthwise kernel
(`conv.conv.weight`) and false of everything around it: `conv.in_proj` and
`conv.out_proj` are ordinary linear layers, and there are twenty-two
convolution layers to eight attention ones. An adapter that skips them
reaches roughly a quarter of the stack and has to push a style change through
three frozen blocks out of four. The target set is therefore every linear
sheet in a block - attention's `q_proj`, `k_proj`, `v_proj`, `out_proj`, the
feed forward's `w1`, `w2`, `w3`, and the convolution's `in_proj` and
`out_proj` - with the depthwise kernel and every norm gain left frozen.

The embedding stays frozen too. `tie_word_embeddings` is true, so training it
would move the output head with it, and at 128000 by 2048 it is a third of
the parameters for a change of register that does not need new vocabulary.

Holding the reasoning still
---------------------------

A style tune erodes capability when the model learns the shape and forgets
the content. Three things push back, in increasing cost:

  - **The corpus.** Deletion-only compression means every tuned answer is
    still the source's answer. `Normal mode.` rows and prose rows keep the
    uncompressed register alive; `guard` rows keep the full-sentence warnings
    the skill demands for anything destructive.
  - **An eval split.** `--check` reports token reduction beside answer
    survival, so a tune that compressed by talking nonsense is visible as a
    number rather than as a feeling.
  - **`--anchor W`, off by default.** A second forward pass with the adapter
    switched off gives the frozen base's own distribution, and a KL term pulls
    the tuned model back towards it on the assistant tokens. Measured at
    **2.55x the step time** - 29.68 s against 75.67 s for one epoch over this
    corpus, taken as the minimum of two runs each, on four x86-64 cores with a
    six layer synthetic checkpoint over the real 128000 entry vocabulary. That
    is more than the second forward pass alone would cost, because the term
    also builds two float32 log-softmax planes that wide; on a stack where the
    layers rather than the vocabulary dominate, expect less, and measure it
    rather than taking this ratio. It is a flag and not the default for that
    reason. Use it when `--check` says the answers are drifting.

Uncensoring the weights first
-----------------------------

`--uncensor` runs heretic (<https://github.com/p-e-w/heretic>) over the
checkpoint before anything trains and writes the decensored weights to
`build/tune/uncensored`, in the same layout as `ckpt/0.4b`. It is not a tune and
shares no mechanism with one: no gradient step and no corpus, but a low rank
edit that subtracts the direction the residual stream moves in when the model
is about to refuse. That direction is measured over 400 harmless and 400
harmful prompts, applied per layer at a weight an Optuna search picks, and the
search is scored on two numbers at once -- how many of a hundred held out
harmful prompts still draw a refusal, and how far the first token distribution
has moved from the base on a hundred harmless ones.

Heretic already knows this architecture. It reaches `conv.out_proj`,
`self_attn.out_proj` and `feed_forward.w2` -- every sheet that writes back into
the residual stream on either operator, over all thirty blocks -- so nothing
here has to teach it where to cut. What is this checkpoint's rather than
heretic's is the settings, and two of them matter:

  - **The response prefix is `</think>`.** LFM2.5's template ends a generation
    prompt with `<think>`, so the first token of a reply is the first token of
    the reasoning. Heretic skips a think block by spotting a generated
    `<think>` and replacing it with a closed one, which cannot fire on a
    template that has already opened the block -- so without this, refusals are
    counted over thinking text, where `harmful`, `illegal` and `violat`, three
    of the refusal markers, are exactly what a thought reasoning about a
    request says, and the divergence is measured at the first token of a
    thought rather than of an answer. Closing the block in the prompt puts both
    back on the answer. The prefix carries no newline because this checkpoint
    writes none: greedy from `model/`, it produces
    `...concisely.</think>Here are three practical tips`.
  - **`--uncensor-kl 0.25` is a ceiling on what may be exported**, and the
    divergence heretic balances its two objectives at is set to the same
    number, so the search spends its trials in the band a trial can actually be
    taken from instead of treating 1.0 as typical. The cap is chosen rather
    than measured -- heretic's own note puts visible damage above 0.5, this
    checkpoint is 2.6B and has less to spare than the models that note came
    from, and an adapter trains over whatever comes out and spends accuracy of
    its own.

Nothing is asked of whoever starts it. Heretic ends at a menu -- which point of
the Pareto front, then what to do with it -- and 1.4.0 has a flag for neither,
so its prompts are answered for the length of the run from the shape of the
choices: the fewest refusals among the trials under the cap, save, exit. A run
that stops halfway leaves an optuna checkpoint under `build/tune/uncensor-study`
and the next `--uncensor` resumes it rather than starting again, which also
means a finished study re-exports in seconds; `--uncensor-fresh` throws it away.

Two things it does not do. It does not tell a warning from a refusal --
`disclaimer` and `harmful` are refusal markers, and the corpus' `guard` rows
are full sentences warning about destructive operations, so the register's own
warnings come back from the adapter that trains afterwards rather than
surviving the edit. And it has no number here: no abliteration has been run
over the 2.6B weights, because that needs a card and this repository's hosts
are processors. `TODO.md` carries the item.

When torch, transformers or peft are missing, the script reports what it
skipped and exits zero, so it stays usable inside a build pipeline; `heretic`
is reported the same way, and only `--uncensor` needs it. The rule engine, the
corpus loader and `--lint` need none of them and always run.
"""

import argparse
import inspect
import json
import os
import re
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNE_PATH = os.path.join(ROOT_PATH, "build", "tune")
ADAPTER_PATH = os.path.join(TUNE_PATH, "adapter")
MERGED_PATH = os.path.join(TUNE_PATH, "merged")
UNCENSOR_PATH = os.path.join(TUNE_PATH, "uncensored")
UNCENSOR_STUDY = os.path.join(TUNE_PATH, "uncensor-study")
DATA_PATH = os.path.join(ROOT_PATH, "data", "tune.jsonl")

# A style change is a small change, and the corpus is a few hundred rows of it.
# These defaults train it in one pass on one card; raise the epochs with the
# corpus, not instead of it, because a small corpus run long is memorisation.
TRAIN_EPOCHS = 3
TRAIN_BATCH = 2
TRAIN_STACK = 8          # gradient accumulation, so the effective batch is 16
TRAIN_RATE = 1e-4
TRAIN_WARM = 0.03
TRAIN_DECAY = 0.01
TRAIN_CUT = 1024         # a caveman row is short; a source row before pressing is not
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROP = 0.05
EVAL_SHARE = 0.1         # rows held out when a row does not name its own split

# The abliteration, which runs before the tune and is heretic's work rather
# than this script's. Two hundred trials at sixty random ones is heretic's own
# ratio and is left alone; the rest are this checkpoint's, and `uncensor_argv`
# says why each is what it is.
UNCENSOR_TRIALS = 200
UNCENSOR_START = 60
UNCENSOR_KL = 0.25       # most divergence from the base an exported trial may carry
UNCENSOR_PREFIX = "</think>"
UNCENSOR_SHARD = "1900MB"

# Every linear sheet in a block, and nothing else. `conv.conv` is the depthwise
# kernel and is not a linear layer; the norms are gains, not sheets. PEFT reads
# a string target as a regular expression.
TARGET_PATTERN = (r".*\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
                  r"|feed_forward\.(w1|w2|w3)"
                  r"|conv\.(in_proj|out_proj))")

# The level is carried as a system prompt so the register can be switched at
# serve time rather than baked in. Short on purpose: it is paid for on every
# request, and a long directive would spend the tokens the tune is saving.
LEVEL_SYSTEM = {
    "lite": "Caveman mode: lite.",
    "full": "Caveman mode: full.",
    "ultra": "Caveman mode: ultra.",
    "off": "Normal mode.",
}
LEVEL_LIST = tuple(LEVEL_SYSTEM)
KIND_LIST = ("reason", "guard", "prose", "fact", "mode")

# ---------------------------------------------------------------------------
# the rule engine
#
# Everything in this section is the `caveman` skill's rules written as code.
# It is used twice: to press a source corpus into caveman (`--make-data`), and
# to audit rows that were written by hand (`--lint`). One engine for both
# means a hand written row is held to the rules the generator obeys.
# ---------------------------------------------------------------------------

# Spans copied verbatim. Fenced code first so an inline backtick inside a fence
# is not mistaken for one of its own, then inline code, URLs, and double quoted
# strings - which is how an exact error message is carried through prose.
GUARD_PATTERN = re.compile(
    r"```.*?```"
    r"|`[^`\n]*`"
    r"|https?://\S+"
    r"|\"[^\"\n]*\"",
    re.DOTALL)

# Dropping one of these flips the meaning, which is worse than any token it
# saves. The multiset of them is compared across a press and must not move, and
# no deletion rule may touch one even inside a phrase it would otherwise cut.
KEEP_WORDS = {
    "not", "never", "no", "none", "nor", "only", "except", "without",
    "unless", "cannot", "wont", "dont", "doesnt", "isnt", "arent", "cant",
    "neither", "nothing", "nowhere", "must", "always",
}

# Adverbs that carry no fact. Deliberately short: a word earns a place here
# only when removing it from any sentence leaves the same claim standing.
FILLER_WORDS = {
    "just", "really", "basically", "actually", "simply", "essentially",
    "quite", "somewhat", "fairly", "literally", "definitely",
    "obviously", "clearly", "indeed", "truly", "very",
}

# Connectives that only signal a turn the reader can already see.
BRIDGE_WORDS = {
    "however", "furthermore", "additionally", "moreover",
    "nevertheless", "nonetheless", "meanwhile",
}

ARTICLE_WORDS = {"a", "an", "the"}

# Complementisers and discourse particles that `ultra` drops on top of `full`.
# `that` goes only where it introduces a clause, which is checked at the call
# site rather than here.
ULTRA_WORDS = {"that", "then", "thus", "hence", "overall", "also"}

# Whole phrases, matched before single words so the longer match wins. Each one
# is a deletion: the text either says the same thing without it, or the phrase
# was announcing the sentence rather than carrying it.
CUT_PHRASES = (
    "i would be happy to help", "i'd be happy to help", "i am happy to help",
    "i would be happy to", "i'd be happy to", "i am happy to", "happy to help",
    "do not hesitate to",
    "great question", "thanks for asking", "of course", "feel free to",
    "let me know if you need anything else", "i would recommend",
    "let me know if you have any questions", "as an ai", "i hope this helps",
    "i'd recommend", "i recommend that you", "keep in mind that",
    "it is important to note that", "it's important to note that",
    "one thing to note is that", "as you know", "needless to say",
    "to put it simply", "simply put", "generally speaking", "in general",
    "it might be worth", "it may be worth", "you could consider",
    "you might want to", "you may want to", "it would be good to",
    "it is worth noting that", "it's worth noting that", "note that",
    "i think", "i believe", "in my opinion", "if i understand correctly",
    "as far as i can tell", "it seems that", "it appears that",
    "let me", "i will now", "i'll start by", "i am going to", "i'm going to",
    "first, let me", "now let's", "let's go ahead and", "going forward",
    "at the end of the day", "when it comes to", "in terms of",
    "that being said", "with that said", "as you can see", "as mentioned",
    "in other words", "to be honest", "sort of", "kind of",
)

# Rewrites, applied before any deletion. Each is strictly shorter and says the
# same thing; the validator applies the same map to the source before checking
# that the pressed words are a subsequence of it, so this list is the only way
# a word may appear in a press that was not in the source.
REWRITE_PHRASES = (
    ("due to the fact that", "because"),
    ("the reason is because", "because"),
    ("for the purpose of", "to"),
    ("in the event that", "if"),
    ("at this point in time", "now"),
    ("has the ability to", "can"),
    ("have the ability to", "can"),
    ("is able to", "can"),
    ("are able to", "can"),
    ("is responsible for", "handles"),
    ("are responsible for", "handle"),
    ("a large number of", "many"),
    ("a small number of", "few"),
    ("take into account", "consider"),
    ("make sure that", "ensure"),
    ("in order to", "to"),
    ("prior to", "before"),
    ("subsequent to", "after"),
    ("with regard to", "for"),
    ("as a result", "so"),
    ("therefore", "so"),
    ("utilize", "use"),
    ("utilise", "use"),
    ("utilizes", "uses"),
    ("utilises", "uses"),
    ("extensive", "big"),
    ("approximately", "about"),
    ("additional", "more"),
    ("demonstrate", "show"),
    ("demonstrates", "shows"),
    ("attempt to", "try to"),
    ("in addition", "and"),
)

# Deletions that are only safe at the head of a sentence, where what follows
# is the imperative the phrase was announcing. Mid-sentence the same words are
# load bearing: "the first thing you should do" is not "the first thing do".
CUT_OPENERS = (
    "you should", "you need to", "you have to", "you will want to",
    "you might want to", "you may want to", "remember to", "make sure to",
    "make sure you", "be sure to", "what you want to do is",
)

# Abbreviations the skill bans by name, plus the few that keep being invented.
# They cost a token each, exactly as the full word does, so they buy nothing
# and read worse. Standard acronyms - DB, API, HTTP - are capitals and are
# guarded as such, so they never reach this check.
BAN_ABBREV = re.compile(
    r"\b(cfg|impl|impls|req|reqs|res|fn|fns|msg|msgs|ctx|obj|objs|val|vals|"
    r"defn|prev|curr|params?(?=\s)|btw|fyi)\b", re.IGNORECASE)

# Arrows are one token and replace one token, so they save nothing and cost the
# reader a decode. The skill says so; the audit enforces it.
BAN_ARROW = re.compile(r"(->|=>|-->|→|⇒|➔)")

BAN_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⬀-⯿]")

# "caveman mode on", "Caveman:", "me caveman think" - the skill forbids every
# announcement of the register, because the reply already shows it.
BAN_ANNOUNCE = re.compile(
    r"(caveman mode (on|engaged|active)|^caveman:|\bme caveman\b|\bugh\b|"
    r"\bcaveman (think|say|speak)\b)", re.IGNORECASE | re.MULTILINE)

# An opener is a whole word standing at the head of a sentence and carrying
# nothing: "Sure, ...", "Okay, so ...". The comma or full stop after it goes
# with it, which is why this is a pattern of its own rather than a word list.
OPENER_PATTERN = re.compile(
    r"(?<![^\s])(Sure|Certainly|Absolutely|Alright|Okay|OK|Well|Right|Great|"
    r"So|Now|Hmm|Ah)\s*[,.!:]\s+",
    re.IGNORECASE)

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
STE_WORD_LIMIT = 20


def span_list(text):
    """Splits text into (guarded, piece) pairs. Guarded pieces are never read
    for words and never changed; everything else is free prose."""
    piece_list = []
    at = 0
    for hit in GUARD_PATTERN.finditer(text):
        if hit.start() > at:
            piece_list.append((False, text[at:hit.start()]))
        piece_list.append((True, hit.group(0)))
        at = hit.end()
    if at < len(text):
        piece_list.append((False, text[at:]))
    return piece_list


def word_locked(word):
    """True when a bare word must survive a press untouched.

    A word is locked when it carries something a reader cannot reconstruct: a
    digit, an identifier's underscore or slash, an interior full stop, an at
    sign, capitals, or a negation. Everything else is ordinary prose and may
    be considered for deletion."""
    core = word.strip(".,;:!?()[]{}'\"`*_")
    if not core:
        return True
    if any(ch.isdigit() for ch in core):
        return True
    if "_" in core or "/" in core or "@" in core or "\\" in core:
        return True
    if "." in core[:-1]:
        return True
    if len(core) > 1 and core.upper() == core and core.lower() != core:
        return True
    return core.lower().replace("'", "") in KEEP_WORDS


def phrase_pattern(phrase):
    """A word-boundary pattern for a phrase, tolerant of runs of whitespace."""
    parts = [re.escape(bit) for bit in phrase.split(" ")]
    return re.compile(r"(?<![\w])" + r"\s+".join(parts) + r"(?![\w])",
                      re.IGNORECASE)


REWRITE_READY = tuple((phrase_pattern(was), now) for was, now in
                      sorted(REWRITE_PHRASES, key=lambda pair: -len(pair[0])))
CUT_READY = tuple(phrase_pattern(phrase) for phrase in
                  sorted(CUT_PHRASES, key=len, reverse=True))
OPEN_READY = tuple(
    re.compile(r"(^|(?<=[.!?:]\s))\s*" + r"\s+".join(re.escape(bit) for bit in phrase.split(" "))
               + r"\s+", re.IGNORECASE | re.MULTILINE)
    for phrase in sorted(CUT_OPENERS, key=len, reverse=True))


def rewrite_free(piece):
    """Applies the closed rewrite list to one free span."""
    for pattern, now in REWRITE_READY:
        piece = pattern.sub(now, piece)
    return piece


def phrase_cut(hit):
    """Deletes a matched phrase, unless it carries a word that must survive.

    "no problem at all" is a pleasantry and "not a problem" is a claim; the
    only difference a pattern can see is the negation inside it, so a phrase
    holding one is left exactly where it is."""
    body = hit.group(0)
    if any(word in KEEP_WORDS for word in free_words(body)):
        return body
    return " "


def press_free(piece, level):
    """Deletes what the level says to delete from one free span."""
    piece = OPENER_PATTERN.sub("", piece)
    for pattern in OPEN_READY:
        piece = pattern.sub(lambda hit: hit.group(1) or "", piece)
    for pattern in CUT_READY:
        piece = pattern.sub(phrase_cut, piece)

    drop = set(FILLER_WORDS)
    if level in ("full", "ultra"):
        drop |= ARTICLE_WORDS | BRIDGE_WORDS
    if level == "ultra":
        drop |= ULTRA_WORDS

    kept = []
    for token in re.split(r"(\s+)", piece):
        if not token or token.isspace():
            kept.append(token)
            continue
        if word_locked(token):
            kept.append(token)
            continue
        bare = token.strip(".,;:!?()[]{}'\"").lower()
        if bare in drop:
            # A dropped word leaves its punctuation behind, so a comma or a
            # full stop that followed it is kept and the tidy pass reattaches
            # it to the word before.
            tail = token[len(token.rstrip(".,;:!?)")):]
            kept.append(tail if tail else "")
            continue
        kept.append(token)
    return "".join(kept)


def tidy(piece, opens):
    """Puts the spacing and the capitals back after a deletion pass.

    This runs over one free span at a time and never over a guarded one. An
    earlier revision tidied the joined text, which collapsed the indentation
    inside every fenced code block it passed over; the validator caught that
    and threw those rows away, so the damage was a silently thinner corpus
    rather than a wrong one. `opens` says whether this span begins a sentence,
    so a span following a full stop in the span before it gets its capital.

    Indented code blocks are not guarded - only fenced ones and inline
    backticks are - so a source corpus that indents its code rather than
    fencing it will lose rows to the validator. Fence it before pressing."""
    # Runs of spaces collapse, but not the ones at the head of a line: those
    # are a list's indentation, and flattening them changes what the text is.
    piece = re.sub(r"(?<=\S)[ \t]+", " ", piece)
    # A dropped word leaves its punctuation behind so that "foo, and bar" keeps
    # its comma; at the head of a sentence that comma has nothing to attach to.
    piece = re.sub(r"([.!?])\s*[,;:]+\s*", r"\1 ", piece)
    piece = re.sub(r"(\n)[ \t]*[,;:!?.]+[ \t]*", r"\1", piece)
    piece = re.sub(r" +([.,;:!?)])", r"\1", piece)
    piece = re.sub(r"([(]) +", r"\1", piece)
    piece = re.sub(r"\n{3,}", "\n\n", piece)
    piece = re.sub(r"([.!?][ \t]+|\n[ \t]*)([a-z])",
                   lambda hit: hit.group(1) + hit.group(2).upper(), piece)
    if opens:
        # Only at the head of a sentence: mid-sentence this span follows an
        # inline code span, and the comma after it belongs to the sentence.
        piece = re.sub(r"^[ \t]*[,;:!?.]+[ \t]*", "", piece)
        piece = re.sub(r"^([ \t]*)([a-z])",
                       lambda hit: hit.group(1) + hit.group(2).upper(), piece)
    return piece


SENTENCE_END = re.compile(r"([.!?:][\s]*|\n\s*)$")


def press_text(text, level):
    """Presses prose into caveman at one level, leaving guarded spans alone."""
    if level == "off":
        return text
    out, opens = [], True
    for guarded, piece in span_list(text):
        if guarded:
            out.append(piece)
            # A fenced block ends a sentence; an inline span sits inside one.
            opens = piece.startswith("```")
            continue
        done = tidy(press_free(rewrite_free(piece), level), opens)
        if done.strip():
            opens = bool(SENTENCE_END.search(done))
        out.append(done)
    return "".join(out).strip()


def free_words(text):
    """The lower case words of the free spans, punctuation stripped."""
    found = []
    for guarded, piece in span_list(text):
        if not guarded:
            found += [word.lower() for word in WORD_PATTERN.findall(piece)]
    return found


def free_text(text):
    """Just the free spans, joined, for the checks that read whole sentences."""
    return " ".join(piece for guarded, piece in span_list(text) if not guarded)


def guard_list(text):
    return [piece for guarded, piece in span_list(text) if guarded]


def number_list(text):
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", free_text(text)))


def negation_list(text):
    return sorted(word for word in free_words(text)
                  if word.replace("'", "") in KEEP_WORDS)


def is_subsequence(small, large):
    """True when every word of `small` appears in `large` in the same order."""
    walk = iter(large)
    return all(word in walk for word in small)


def press_check(source, pressed, level):
    """Every way a press can be wrong, as a list of faults. Empty means sound.

    This is the argument that a pressed row is still true: the guarded spans
    are the same bytes, the numbers and negations are the same multisets, and
    the words that remain are a subsequence of the words that were there. A
    transform that can only delete cannot invent."""
    fault = []
    if guard_list(source) != guard_list(pressed):
        fault.append("a guarded span moved: code, a URL or a quoted string changed")
    if number_list(source) != number_list(pressed):
        fault.append("the numbers do not match")
    if negation_list(source) != negation_list(pressed):
        fault.append("a negation was added or lost")
    rewritten = "".join(
        piece if guarded else rewrite_free(piece)
        for guarded, piece in span_list(source))
    if not is_subsequence(free_words(pressed), free_words(rewritten)):
        fault.append("the press added or reordered a word; it may only delete")
    if level != "off" and len(free_words(pressed)) >= len(free_words(source)):
        fault.append("the press did not shorten anything")
    return fault


def style_check(text, level, kind):
    """Audits finished prose against the register's rules."""
    fault = []
    if not text.strip():
        return ["empty"]
    bare = free_text(text)
    if BAN_ARROW.search(bare):
        fault.append("an arrow: one token in, one token out, nothing saved")
    hit = BAN_ABBREV.search(bare)
    if hit:
        fault.append("an invented abbreviation, %r; the full word costs the same"
                     % hit.group(0))
    if BAN_EMOJI.search(bare):
        fault.append("an emoji")
    if BAN_ANNOUNCE.search(bare):
        fault.append("an announcement of the register; the reply already shows it")
    if "!" in bare:
        fault.append("an exclamation mark")
    if level == "off":
        return fault

    # A warning and a commit message are outside the register by the skill's
    # own boundary rule: one is read under pressure, the other is read by
    # somebody who never saw the chat. Both stay in full sentences, so the
    # article, connective and sentence-length checks do not apply to them.
    # Everything above this point still does: no arrows, no invented
    # abbreviations, no emoji anywhere.
    plain = kind in ("guard", "prose")

    word_list = free_words(text)
    if level in ("full", "ultra") and not plain:
        article = sorted(set(word_list) & ARTICLE_WORDS)
        if article:
            fault.append("an article survived at %s: %s" % (level, ", ".join(article)))
        bridge = sorted(set(word_list) & BRIDGE_WORDS)
        if bridge:
            fault.append("connective fluff: %s" % ", ".join(bridge))
    filler = sorted(set(word_list) & FILLER_WORDS)
    if filler:
        fault.append("filler: %s" % ", ".join(filler))
    for pattern, phrase in zip(() if kind == "prose" else CUT_READY,
                               sorted(CUT_PHRASES, key=len, reverse=True)):
        if pattern.search(bare):
            fault.append("a phrase the register drops: %r" % phrase)
            break
    for sentence in (() if plain else SENTENCE_SPLIT.split(bare)):
        count = len(WORD_PATTERN.findall(sentence))
        if count > STE_WORD_LIMIT:
            fault.append("a sentence of %d words; one idea per sentence, %d at most"
                         % (count, STE_WORD_LIMIT))
            break
    return fault


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

def row_read(row, where, number):
    """Normalises one corpus line, or says what is wrong with it."""
    for key in ("prompt", "completion"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("%s line %d: %s must be a non-empty string"
                             % (where, number, key))
    level = row.get("level", "full")
    if level not in LEVEL_SYSTEM:
        raise ValueError("%s line %d: level %r is not one of %s"
                         % (where, number, level, ", ".join(LEVEL_LIST)))
    kind = row.get("kind", "reason")
    if kind not in KIND_LIST:
        raise ValueError("%s line %d: kind %r is not one of %s"
                         % (where, number, kind, ", ".join(KIND_LIST)))
    split = row.get("split", "train")
    if split not in ("train", "eval"):
        raise ValueError("%s line %d: split %r is neither train nor eval"
                         % (where, number, split))
    return {
        "prompt": row["prompt"],
        "thinking": row.get("thinking", ""),
        "completion": row["completion"],
        "level": level,
        "system": row.get("system") or LEVEL_SYSTEM[level],
        "kind": kind,
        "check": row.get("check", ""),
        "split": split,
    }


def load_rows(data_path):
    """Reads the JSONL corpus. Raises on anything a tune must not train on."""
    row_list = []
    with open(data_path, "r", encoding="utf-8") as hand:
        for number, line in enumerate(hand, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except ValueError as why:
                raise ValueError("%s line %d: %s" % (data_path, number, why))
            row_list.append(row_read(row, data_path, number))
    if not row_list:
        raise ValueError("%s holds no rows" % data_path)

    # A row with no thinking renders without the <think> block the model is
    # primed with at inference, so training on it teaches a shape the model is
    # never asked for. This is a hard stop, not a warning; see the header.
    thinless = [at + 1 for at, row in enumerate(row_list) if not row["thinking"].strip()]
    if thinless:
        raise ValueError(
            "%s: %d row(s) carry no `thinking`, at line(s) %s. Every row needs "
            "one: the chat template primes generation with <think>, so a row "
            "without it trains a shape the model never produces."
            % (data_path, len(thinless),
               ", ".join(str(at) for at in thinless[:12])))
    return row_list


def split_rows(row_list, share=EVAL_SHARE):
    """Holds rows out for --check, honouring a row that names its own split.

    The choice is a hash of the prompt rather than a shuffle, so the same row
    lands on the same side on every machine and two runs are comparable."""
    train_list, eval_list = [], []
    for row in row_list:
        named = row["split"] == "eval"
        drawn = (hash_stable(row["prompt"]) % 1000) < int(share * 1000)
        (eval_list if named or drawn else train_list).append(row)
    return train_list, eval_list


def hash_stable(text):
    """FNV-1a. Python's own hash is salted per process and would move a split
    between runs, which would make two --check numbers incomparable."""
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


# ---------------------------------------------------------------------------
# growing the corpus from a source dataset
# ---------------------------------------------------------------------------

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# The field names the public reasoning corpora actually use, most specific
# first. A source that names its columns differently can be renamed with jq
# before it gets here; the point of the list is that the common shapes need
# no preparation at all.
FIELD_GUESS = (
    ("prompt", ("prompt", "question", "instruction", "problem", "query", "input")),
    ("thinking", ("thinking", "reasoning", "reasoning_content", "thought",
                  "chain_of_thought", "rationale")),
    ("completion", ("completion", "response", "answer", "output", "solution")),
)


def source_row(row):
    """Maps one row of a source dataset onto prompt, thinking and completion.

    Conversations are read as their last user turn and last assistant turn; a
    `<think>` block inside an answer is lifted out into `thinking`, which is
    how most published reasoning corpora carry it."""
    found = {}
    if isinstance(row.get("messages"), list):
        for turn in row["messages"]:
            if not isinstance(turn, dict):
                continue
            body = turn.get("content")
            if not isinstance(body, str):
                continue
            if turn.get("role") == "user":
                found["prompt"] = body
            elif turn.get("role") == "assistant":
                found["completion"] = body
                for key in ("thinking", "reasoning", "reasoning_content"):
                    if isinstance(turn.get(key), str) and turn[key].strip():
                        found["thinking"] = turn[key]
    for name, guess_list in FIELD_GUESS:
        if found.get(name):
            continue
        for guess in guess_list:
            if isinstance(row.get(guess), str) and row[guess].strip():
                found[name] = row[guess]
                break
    if not found.get("prompt") or not found.get("completion"):
        return None
    if not found.get("thinking"):
        hit = THINK_PATTERN.search(found["completion"])
        if not hit:
            return None
        found["thinking"] = hit.group(1).strip()
        found["completion"] = THINK_PATTERN.sub("", found["completion"]).strip()
    found["completion"] = THINK_PATTERN.sub("", found["completion"]).strip()
    if not found["completion"]:
        return None
    return found


def source_read(source, limit):
    """Yields raw rows from a JSONL file, or from a Hugging Face dataset id."""
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as hand:
            for number, line in enumerate(hand):
                if limit and number >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    try:
        from datasets import load_dataset
    except ImportError:
        raise ValueError(
            "%s is not a file, and reading it as a Hugging Face dataset needs "
            "`pip install datasets`" % source)
    stream = load_dataset(source, split="train", streaming=True)
    for number, row in enumerate(stream):
        if limit and number >= limit:
            return
        yield row


def make_data(source, out_path, think_level, reply_level, floor, limit):
    """Presses a source reasoning dataset into caveman rows, and drops the
    rows the press could not do honestly.

    The transform only ever deletes, so a surviving row still carries the
    source's own answer. Rows where a guarded span moved, where a number or a
    negation changed, or where the press saved less than the floor are thrown
    away rather than repaired: a corpus is worth more for what it refuses."""
    tally = {"read": 0, "shape": 0, "fault": 0, "thin": 0, "same": 0, "kept": 0}
    seen = set()
    kept_list = []
    before_count = after_count = 0

    for raw in source_read(source, limit):
        tally["read"] += 1
        found = source_row(raw)
        if not found:
            tally["shape"] += 1
            continue
        mark = hash_stable(" ".join(found["prompt"].lower().split()))
        if mark in seen:
            tally["same"] += 1
            continue
        seen.add(mark)

        thought = press_text(found["thinking"], think_level)
        answer = press_text(found["completion"], reply_level)
        fault = (press_check(found["thinking"], thought, think_level)
                 + press_check(found["completion"], answer, reply_level))
        if fault:
            tally["fault"] += 1
            continue

        was = len(free_words(found["thinking"]) + free_words(found["completion"]))
        now = len(free_words(thought) + free_words(answer))
        if not was or (was - now) / was < floor:
            tally["thin"] += 1
            continue
        before_count += was
        after_count += now
        tally["kept"] += 1
        kept_list.append({
            "prompt": found["prompt"],
            "thinking": thought,
            "completion": answer,
            "level": reply_level,
            "kind": "reason",
            "split": "eval" if (mark % 1000) < int(EVAL_SHARE * 1000) else "train",
        })

    with open(out_path, "w", encoding="utf-8") as hand:
        for row in kept_list:
            hand.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("read %d, kept %d" % (tally["read"], tally["kept"]))
    print("  dropped %d for shape, %d for a fault, %d for saving under the floor, "
          "%d as duplicates" % (tally["shape"], tally["fault"], tally["thin"], tally["same"]))
    if before_count:
        print("  words %d -> %d, %.1f%% off" %
              (before_count, after_count,
               100.0 * (before_count - after_count) / before_count))
    print("  written to %s" % out_path)
    return tally["kept"]


def lint_data(data_path):
    """Audits the corpus against the register's rules and reports the counts."""
    row_list = load_rows(data_path)
    train_list, eval_list = split_rows(row_list)
    bad_count = 0
    for at, row in enumerate(row_list, 1):
        fault = []
        # The thought is pressed harder than the answer, which is the whole
        # point of tuning both: the thought is the larger half and nobody reads
        # it. A row that names `ultra` for the answer gets `ultra` for the
        # thought too; there is nothing above it. The thought is also held to
        # the register on a `guard` or `prose` row, where the answer is not:
        # the model reasons in caveman and then writes the warning out in full.
        level = row["level"]
        deep = "ultra" if level == "full" else level
        for where, text, use, kind in (("thinking", row["thinking"], deep, "reason"),
                                       ("completion", row["completion"], level, row["kind"])):
            fault += ["%s: %s" % (where, note) for note in style_check(text, use, kind)]
        if row["check"] and row["check"] not in row["completion"]:
            fault.append("check: %r is not in the answer it is meant to hold"
                         % row["check"])
        if fault:
            bad_count += 1
            print("row %d (%s, %s):" % (at, row["kind"], row["level"]))
            for note in fault:
                print("    " + note)

    kind_tally, level_tally = {}, {}
    for row in row_list:
        kind_tally[row["kind"]] = kind_tally.get(row["kind"], 0) + 1
        level_tally[row["level"]] = level_tally.get(row["level"], 0) + 1
    print("%d rows: %d train, %d eval" % (len(row_list), len(train_list), len(eval_list)))
    print("  kinds:  " + ", ".join("%s %d" % pair for pair in sorted(kind_tally.items())))
    print("  levels: " + ", ".join("%s %d" % pair for pair in sorted(level_tally.items())))
    print("  %d row(s) with a style fault" % bad_count)
    return bad_count


# ---------------------------------------------------------------------------
# the abliteration
#
# Uncensoring is not a tune and does not share a mechanism with one: no
# gradient step, no corpus, a low rank edit of the weights that subtracts the
# direction the residual stream moves in when the model is about to refuse.
# Heretic does that work; everything in this section is about driving it with
# nobody at the keyboard.
# ---------------------------------------------------------------------------

def need_heretic():
    """Imports what an abliteration needs, or prints what is missing."""
    try:
        import heretic.main  # noqa: F401
    except ImportError as miss:
        print("skip: %s; pip install heretic-llm" % miss)
        return None
    return True


def carry_leaves(model_path, out_path):
    """Copies the template and the generation defaults across when a save did not.

    `save_pretrained` writes them only when the object it is called on holds
    them, so anything the base had and the copy lacks is carried across by
    hand. The template is what shapes a prompt; without it the reference falls
    back to a default that does not open the think block, and the shape the
    tune trains into goes with it."""
    import shutil
    for leaf in ("chat_template.jinja", "generation_config.json"):
        was = os.path.join(model_path, leaf)
        now = os.path.join(out_path, leaf)
        if os.path.isfile(was) and not os.path.isfile(now):
            shutil.copyfile(was, now)


def uncensor_argv(model_path, flag):
    """The heretic command line this checkpoint wants.

    Heretic takes its settings from, in falling precedence, the command line,
    `HERETIC_` environment variables, and a `config.toml` in the working
    directory. Everything that matters is passed as argv so that a stray
    config file beside the repository cannot quietly change what a run does.

    Four of these are this checkpoint's rather than heretic's defaults:

      - `--response-prefix </think>`. LFM2.5's template ends a generation
        prompt with `<think>`, so the first token of a reply is the first token
        of the reasoning. Heretic skips a think block by spotting a generated
        `<think>` and replacing it with a closed one, which cannot fire on a
        template that already opened it, so without this the refusal count is
        taken over thinking text -- where `harmful`, `illegal` and `violat`,
        three of the refusal markers, are what a thought reasoning about a
        request says -- and the divergence is measured at the first token of a
        thought. Closing the block in the prompt puts both back on the answer.
        No trailing newline: this checkpoint writes none, and greedy from
        `ckpt/2.6b` produces `...concisely.</think>Here are three practical tips`.
      - `--kl-divergence-scale` follows `--uncensor-kl`. The scale is the
        divergence heretic treats as typical when it balances its two
        objectives against each other; leaving it at 1.0 while exporting only
        trials at or under 0.25 spends the search on a band nothing can be
        taken from.
      - `--max-shard-size 1900MB`, the same cap `--merge` writes under, so the
        result can be committed the way `ckpt/` is: two gigabytes is the
        limit for one LFS object.
      - `--export-strategy merge`, because the engine reads a plain checkpoint
        and has no idea what a PEFT adapter is.

    The startup trials stay at heretic's ratio rather than its count. Sixty
    random trials of two hundred is thirty per cent; sixty of forty is every
    trial, and the TPE sampler would never get a turn. Where heretic wants a
    folder to save into it asks rather than reading a setting, so the output
    path is not here -- it is answered by the hand below."""
    return [
        "heretic",
        "--model", model_path,
        "--seed", str(flag.seed),
        "--response-prefix", UNCENSOR_PREFIX,
        "--kl-divergence-scale", str(flag.uncensor_kl),
        "--n-trials", str(flag.uncensor_trials),
        "--n-startup-trials", str(min(UNCENSOR_START,
                                      max(1, flag.uncensor_trials // 3))),
        "--batch-size", str(flag.uncensor_batch),
        "--quantization", flag.uncensor_quant,
        "--export-strategy", "merge",
        "--max-shard-size", UNCENSOR_SHARD,
        "--study-checkpoint-dir", UNCENSOR_STUDY,
    ]


def uncensor_pick(trial_list, cap):
    """The trial an unattended run exports.

    Heretic finishes with the Pareto front of refusal count against KL
    divergence from the base, and asks which point on it to keep: the one
    judgement in the process. The fewest refusals sit at the divergent end of
    that front, so taking the best refusal count alone hands back the most
    damaged model on offer. This takes the fewest refusals among the trials at
    or under `cap`, and the least divergent of those where they tie.

    `cap` is chosen, not measured. Heretic's own note puts visible damage
    above 0.5; this checkpoint is 2.6B and has less to spare than the models
    that note came from, and an adapter trains over whatever comes out and
    spends accuracy of its own. `--check` is where a cap set too high shows
    up, which is the reason that pass exists."""
    fit_list = [trial for trial in trial_list
                if trial.user_attrs["kl_divergence"] <= cap]
    if not fit_list:
        # Nothing on the front is exportable, so the run has not decensored
        # anything within budget. Take the least damaged rather than the
        # fewest refusals: raising the cap is a decision for whoever reads
        # these numbers, not one to make silently by taking the far end.
        print("  no trial came in under a divergence of %.3f; taking the least "
              "divergent of the %d on the front, and refusals with it"
              % (cap, len(trial_list)))
        return min(trial_list, key=lambda trial: trial.user_attrs["kl_divergence"])
    return min(fit_list, key=lambda trial: (trial.user_attrs["refusals"],
                                            trial.user_attrs["kl_divergence"]))


def uncensor_hand(out_path, cap, fresh):
    """Answers heretic's own prompts, so a run needs nobody at the keyboard.

    Heretic is interactive at both ends: it asks what to do about a previous
    run's checkpoint, which trial to keep, and what to do with the model once
    that trial is restored. Version 1.4.0 has a flag for none of the three, so
    its four prompt helpers are replaced for the length of the run.

    The answers are chosen by the **shape of the choices rather than the
    wording of the question**: the trial menu is the one whose choices carry
    optuna trials, the resume menu the one offering `continue` beside
    `restart`. A version that rewords a question still gets the right answer,
    where matching on the text would fall through to `exit` and throw away the
    run that had just finished. A question this does not recognise is answered
    with the empty string, which is cancel in every one of heretic's menus,
    and is printed on the way out: a silent exit after two hundred trials is
    not something a user should have to diagnose."""
    state = {"saved": False}

    def select(message, choices):
        value_list = [getattr(choice, "value", choice) for choice in choices]

        # The trial menu. An optuna trial is the only thing in any of heretic's
        # menus carrying user attributes.
        trial_list = [value for value in value_list
                      if hasattr(value, "user_attrs")]
        if trial_list:
            if state["saved"]:
                # The checkpoint is written; leave rather than export it twice.
                return ""
            trial = uncensor_pick(trial_list, cap)
            note = trial.user_attrs
            print("  taking trial %d from the %d on the front: %d refusals of %d "
                  "against the base's %d, divergence %.4f"
                  % (note["index"], len(trial_list), note["refusals"],
                     note["n_bad_prompts"], note["base_refusals"],
                     note["kl_divergence"]))
            return trial

        # The resume menu, shown when a previous run left a study checkpoint.
        # Continuing is the point of that checkpoint: a finished study exports
        # again in seconds, an interrupted one picks up where it stopped.
        if "continue" in value_list:
            return "restart" if fresh and "restart" in value_list else "continue"

        # The export menu, which --export-strategy answers before it is asked.
        # This is here for a version that asks anyway.
        if "merge" in value_list:
            return "merge"

        # The action menu. Save once, then leave; upload, chat and benchmark
        # are all things an unattended run has no business picking.
        for value in value_list:
            if isinstance(value, str) and "save" in value.lower():
                if state["saved"]:
                    return ""
                state["saved"] = True
                return value

        print("  heretic asked something this script cannot answer, and was "
              "told to cancel: %s" % message)
        return ""

    def path(message):
        _ = message
        return out_path

    def text(message, *spare, **more):
        # The only text prompt an unattended run can reach is the count of
        # additional trials, and empty means none of them.
        _ = (message, spare, more)
        return ""

    def secret(message):
        # Reached only by the upload action, which is never picked.
        _ = message
        return ""

    return {"select": select, "path": path, "text": text, "secret": secret}


def uncensor_step(model_path, out_path, flag):
    """Abliterates the refusal direction out of the checkpoint.

    Heretic's own `run` does the work -- loading the model, measuring the
    per layer refusal directions over 400 harmless and 400 harmful prompts,
    searching the ablation weights, restoring the chosen trial and writing the
    merged checkpoint -- because a copy of it here would be a second
    implementation of somebody else's arithmetic to keep in step with theirs.
    What this adds is the command line it is given, the hand that answers its
    prompts, and putting back the global state it changes on the way out.

    Heretic already knows this architecture: it reaches `conv.out_proj`,
    `self_attn.out_proj` and `feed_forward.w2`, which is every sheet that
    writes back into the residual stream on either operator, over all thirty of
    this checkpoint's blocks. Nothing here has to teach it where to cut."""
    import torch
    from heretic import main as heretic_main

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    hand = uncensor_hand(out_path, flag.uncensor_kl, flag.uncensor_fresh)
    argv_was = sys.argv
    # Heretic only ever runs a forward pass, so it turns gradients off for the
    # whole process. A --train in the same run would then take steps that
    # compute nothing, which is why this is restored rather than left.
    grad_was = torch.is_grad_enabled()
    kept = (heretic_main.prompt_select, heretic_main.prompt_path,
            heretic_main.prompt_text, heretic_main.prompt_password)
    sys.argv = uncensor_argv(model_path, flag)
    heretic_main.prompt_select = hand["select"]
    heretic_main.prompt_path = hand["path"]
    heretic_main.prompt_text = hand["text"]
    heretic_main.prompt_password = hand["secret"]
    try:
        heretic_main.run()
    finally:
        sys.argv = argv_was
        torch.set_grad_enabled(grad_was)
        (heretic_main.prompt_select, heretic_main.prompt_path,
         heretic_main.prompt_text, heretic_main.prompt_password) = kept

    if not os.path.isfile(os.path.join(out_path, "config.json")):
        return None
    carry_leaves(model_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# the tune
# ---------------------------------------------------------------------------

def need_modules():
    """Imports what a tune needs, or prints what is missing and returns None."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as miss:
        print("skip: %s; pip install torch transformers peft" % miss)
        return None
    return True


def frame_turns(row):
    """One row as the chat turns the template renders.

    The assistant turn carries `thinking`, which is what makes the template
    emit `<think>...</think>` ahead of the answer - the same shape the engine
    is primed into at serve time."""
    turn_list = []
    if row["system"]:
        turn_list.append({"role": "system", "content": row["system"]})
    turn_list.append({"role": "user", "content": row["prompt"]})
    turn_list.append({"role": "assistant",
                      "thinking": row["thinking"],
                      "content": row["completion"]})
    return turn_list


def frame_rows(book, row_list, cut):
    """Tokenises the corpus and masks the loss down to the assistant span.

    `return_assistant_tokens_mask` reads the `{% generation %}` markers in the
    checkpoint's own template, so the mask covers exactly
    `<think>THOUGHT</think>ANSWER<|im_end|>` and nothing before it. Training
    on the prompt as well - which is what a plain text field does - spends the
    gradient teaching the model to write the user's question back."""
    piece_list, long_count = [], 0
    for row in row_list:
        piece = book.apply_chat_template(
            frame_turns(row), tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True)
        ids = list(piece["input_ids"])
        mask = list(piece["assistant_masks"])
        if len(ids) > cut:
            # Truncating an answer would train the model to stop mid sentence,
            # so a long row is dropped and counted instead.
            long_count += 1
            continue
        if not any(mask):
            raise ValueError("the chat template produced no assistant span; "
                             "this checkpoint's template cannot be masked")
        piece_list.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [token if keep else -100 for token, keep in zip(ids, mask)],
        })
    if long_count:
        print("  dropped %d row(s) longer than the %d token cut" % (long_count, cut))
    if not piece_list:
        raise ValueError("every row was longer than the cut; raise --cut")
    return piece_list


def pad_batch(piece_list, pad_id):
    """Pads a batch to its longest row. Padding is masked out of the loss."""
    import torch
    wide = max(len(piece["input_ids"]) for piece in piece_list)
    stack = {"input_ids": [], "attention_mask": [], "labels": []}
    for piece in piece_list:
        room = wide - len(piece["input_ids"])
        stack["input_ids"].append(piece["input_ids"] + [pad_id] * room)
        stack["attention_mask"].append(piece["attention_mask"] + [0] * room)
        stack["labels"].append(piece["labels"] + [-100] * room)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in stack.items()}


def anchor_trainer(weight):
    """A Trainer that holds the tuned model near the frozen base.

    With the adapter switched off the same weights are the base model, so the
    anchor costs a second forward pass rather than a second checkpoint in
    memory. The term is KL(base || tuned) over the assistant positions only:
    the forward direction, which spreads the tuned distribution over
    everything the base thought possible rather than letting it collapse onto
    the caveman shape. It is off unless --anchor is passed, and it is not free:
    2.55x the step time on the host in the header note."""
    import torch
    import torch.nn.functional as tensor_fn
    from transformers import Trainer

    class AnchorTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **spare):
            # The cross entropy comes from Trainer's own path rather than from
            # `model(**inputs).loss`. Since transformers 4.46 the trainer hands
            # `num_items_in_batch` down and then does not divide by the
            # accumulation steps itself, so computing the loss here by hand
            # multiplies every gradient by those steps - an eightfold learning
            # rate at the default accumulation, silently. Deferring to `super`
            # keeps whichever contract the installed version uses.
            loss, out = super().compute_loss(model, inputs, return_outputs=True,
                                             **spare)
            if weight <= 0.0:
                return (loss, out) if return_outputs else loss

            core = self.accelerator.unwrap_model(model)
            pick = inputs["labels"][:, 1:].ne(-100)
            if pick.any():
                with torch.no_grad(), core.disable_adapter():
                    base = core(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"]).logits
                tuned_row = tensor_fn.log_softmax(out.logits[:, :-1][pick].float(), dim=-1)
                base_row = tensor_fn.log_softmax(base[:, :-1][pick].float(), dim=-1)
                # Normalised the same way the cross entropy just was, so that
                # --anchor 1.0 really does weigh the two terms equally.
                room = spare.get("num_items_in_batch")
                whole = tensor_fn.kl_div(tuned_row, base_row,
                                         log_target=True, reduction="sum")
                loss = loss + weight * (whole / (room if room else tuned_row.shape[0]))
            return (loss, out) if return_outputs else loss

    return AnchorTrainer


def train_step(model_path, out_path, data_path, flag):
    """Trains a LoRA adapter over the frozen base."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    book = AutoTokenizer.from_pretrained(model_path)
    row_list = load_rows(data_path)
    train_list, eval_list = split_rows(row_list)
    print("  corpus: %d rows, %d train, %d held out"
          % (len(row_list), len(train_list), len(eval_list)))

    on_card = torch.cuda.is_available()
    wide = torch.bfloat16 if on_card else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=wide, device_map="auto" if on_card else None)
    model.config.use_cache = False

    adapter = LoraConfig(
        r=flag.rank,
        lora_alpha=flag.rank * 2,
        lora_dropout=LORA_DROP,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_PATTERN,
    )
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()
    if flag.checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    train_piece = frame_rows(book, train_list, flag.cut)
    eval_piece = frame_rows(book, eval_list, flag.cut) if eval_list else None
    pad_id = book.pad_token_id if book.pad_token_id is not None else 0

    # Warmup is given in steps rather than as a ratio: `warmup_ratio` was
    # dropped from TrainingArguments in transformers 5, and `warmup_steps` is
    # in every version, so the ratio is turned into steps here instead.
    stride = max(1, flag.batch * flag.stack)
    whole = max(1, -(-len(train_piece) // stride)) * max(1, flag.epochs)
    want = dict(
        output_dir=out_path,
        num_train_epochs=flag.epochs,
        per_device_train_batch_size=flag.batch,
        per_device_eval_batch_size=flag.batch,
        gradient_accumulation_steps=flag.stack,
        gradient_checkpointing=flag.checkpointing,
        learning_rate=flag.rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(TRAIN_WARM * whole)),
        weight_decay=TRAIN_DECAY,
        max_grad_norm=1.0,
        bf16=on_card,
        logging_steps=1,
        eval_strategy="epoch" if eval_piece else "no",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=flag.seed,
    )
    # transformers renames arguments between major versions, and a tune that
    # dies on a keyword is worse than one that runs without a nicety. Anything
    # this version does not take is dropped and named, rather than guessed at.
    takes = set(inspect.signature(TrainingArguments.__init__).parameters)
    spare = sorted(key for key in want if key not in takes)
    if spare:
        print("  this transformers does not take %s; running without" % ", ".join(spare))
    setting = TrainingArguments(**{key: value for key, value in want.items()
                                   if key in takes})
    maker = anchor_trainer(flag.anchor)
    trainer = maker(
        model=model,
        args=setting,
        train_dataset=train_piece,
        eval_dataset=eval_piece,
        data_collator=lambda batch: pad_batch(batch, pad_id),
    )
    trainer.train()
    if eval_piece:
        print("  held out loss: %.4f" % trainer.evaluate().get("eval_loss", float("nan")))
    made = os.path.join(out_path, "adapter")
    trainer.save_model(made)
    book.save_pretrained(made)
    return made


def merge_step(model_path, adapter_path, out_path):
    """Folds the adapter into the base weights and writes a plain checkpoint.

    The merge arithmetic runs at float32 so the low rank product is not
    rounded twice, and the result is cast back to the checkpoint's own dtype
    before it is written: the engine reads bf16 directly, and saving float32
    would double a checkpoint nobody asked to grow. Shards are capped under
    two gigabytes, which is the limit a single LFS object may have, so a
    merged checkpoint can be committed the way `ckpt/` is."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    setting = AutoConfig.from_pretrained(model_path)
    kept = getattr(setting, "dtype", None) or getattr(setting, "torch_dtype", None)
    if isinstance(kept, str):
        kept = getattr(torch, kept, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    if kept is not None:
        model = model.to(kept)
    model.save_pretrained(out_path, safe_serialization=True, max_shard_size="1900MB")
    book = AutoTokenizer.from_pretrained(model_path)
    book.save_pretrained(out_path)
    carry_leaves(model_path, out_path)
    return out_path


# ---------------------------------------------------------------------------
# the number that says whether it worked
# ---------------------------------------------------------------------------

def speak_rows(model_path, row_list, room, quiet):
    """Greedy continuations for the held out prompts, split at `</think>`.

    Greedy on purpose: a sampled answer moves between runs and the comparison
    is between two checkpoints, not between two draws."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    book = AutoTokenizer.from_pretrained(model_path)
    on_card = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if on_card else torch.float32,
        device_map="auto" if on_card else None).eval()

    said_list = []
    for at, row in enumerate(row_list, 1):
        turn_list = frame_turns(row)[:-1]
        prime = book.apply_chat_template(
            turn_list, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            grown = model.generate(**prime, max_new_tokens=room, do_sample=False,
                                   pad_token_id=book.pad_token_id)
        tail = grown[0][prime["input_ids"].shape[1]:]
        text = book.decode(tail, skip_special_tokens=True)
        thought, _, answer = text.partition("</think>")
        if not answer:
            # The model ran out of room inside the thought. Counting that as a
            # zero length answer would flatter the compression number, so the
            # row is reported as unfinished instead.
            thought, answer = text, ""
        said_list.append({
            "row": row,
            "think_count": len(book(thought, add_special_tokens=False)["input_ids"]),
            "reply_count": len(book(answer, add_special_tokens=False)["input_ids"]),
            "answer": answer.strip(),
            "done": bool(answer),
        })
        if not quiet:
            print("    %d/%d" % (at, len(row_list)), end="\r", flush=True)
    del model
    return said_list


def check_step(model_path, tuned_path, data_path, room, quiet):
    """Reports how far the output shrank beside how much of it survived.

    Two numbers, and they are only worth reading together: a tune that
    compressed by dropping the answer scores well on the first and badly on
    the second, which is the failure this whole file is arranged to avoid."""
    row_list = load_rows(data_path)
    _, eval_list = split_rows(row_list)
    if not eval_list:
        print("no held out rows; mark some with \"split\": \"eval\"")
        return 1

    print("  base: %s" % model_path)
    base_list = speak_rows(model_path, eval_list, room, quiet)
    print("  tuned: %s" % tuned_path)
    tuned_list = speak_rows(tuned_path, eval_list, room, quiet)

    def sums(said_list):
        return (sum(said["think_count"] for said in said_list),
                sum(said["reply_count"] for said in said_list))

    base_think, base_reply = sums(base_list)
    tuned_think, tuned_reply = sums(tuned_list)

    def share(was, now):
        return "%d -> %d, %+.1f%%" % (was, now, -100.0 * (was - now) / was) if was else "none"

    print("\n  thought tokens: %s" % share(base_think, tuned_think))
    print("  answer tokens:  %s" % share(base_reply, tuned_reply))
    print("  whole reply:    %s" % share(base_think + base_reply, tuned_think + tuned_reply))

    def holds(said_list):
        want = [said for said in said_list if said["row"]["check"]]
        got = [said for said in want if said["row"]["check"] in said["answer"]]
        return len(got), len(want)

    base_hit, want_count = holds(base_list)
    tuned_hit, _ = holds(tuned_list)
    if want_count:
        print("  answers still right: base %d/%d, tuned %d/%d"
              % (base_hit, want_count, tuned_hit, want_count))
    else:
        print("  no row carries a \"check\" string, so nothing measured accuracy")

    def clean(said_list):
        return sum(1 for said in said_list
                   if not style_check(said["answer"], said["row"]["level"],
                                      said["row"]["kind"]))

    print("  answers clean against the register: base %d/%d, tuned %d/%d"
          % (clean(base_list), len(base_list), clean(tuned_list), len(tuned_list)))
    unfinished = sum(1 for said in tuned_list if not said["done"])
    if unfinished:
        print("  %d tuned reply/replies never left the thought inside %d tokens"
              % (unfinished, room))
    return 0


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="inferliqu caveman fine tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",
                        default=os.environ.get("INFERLIQU_MODEL",
                            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "ckpt", "0.4b")),
                        help="checkpoint folder in huggingface layout (default ckpt/0.4b)")
    parser.add_argument("--data", default=DATA_PATH,
                        help="JSONL corpus, one row per line")
    parser.add_argument("--lint", action="store_true",
                        help="audit the corpus against the register's rules")
    parser.add_argument("--make-data", action="store_true",
                        help="press a source reasoning dataset into caveman rows")
    parser.add_argument("--source", default=None,
                        help="JSONL file or Hugging Face dataset id to press")
    parser.add_argument("--out", default=None,
                        help="where --make-data writes; default beside --source")
    parser.add_argument("--think-level", default="ultra", choices=LEVEL_LIST,
                        help="how hard --make-data presses the thought")
    parser.add_argument("--reply-level", default="full", choices=LEVEL_LIST,
                        help="how hard --make-data presses the answer")
    parser.add_argument("--floor", type=float, default=0.15,
                        help="least a row must shrink to be worth keeping")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many source rows; 0 reads them all")
    parser.add_argument("--uncensor", action="store_true",
                        help="abliterate refusals out of the checkpoint into %s"
                             % UNCENSOR_PATH)
    parser.add_argument("--uncensor-out", default=UNCENSOR_PATH,
                        help="where --uncensor writes the decensored checkpoint")
    parser.add_argument("--uncensor-trials", type=int, default=UNCENSOR_TRIALS,
                        help="ablation trials --uncensor searches over")
    parser.add_argument("--uncensor-kl", type=float, default=UNCENSOR_KL,
                        help="most divergence from the base an exported trial may carry")
    parser.add_argument("--uncensor-batch", type=int, default=0,
                        help="sequences --uncensor scores in parallel; 0 measures it")
    parser.add_argument("--uncensor-quant", default="none",
                        choices=("none", "bnb_4bit"),
                        help="load the weights 4-bit, to fit --uncensor on a smaller card")
    parser.add_argument("--uncensor-fresh", action="store_true",
                        help="ignore an interrupted --uncensor run rather than resuming it")
    parser.add_argument("--train", action="store_true",
                        help="train a LoRA adapter into %s" % ADAPTER_PATH)
    parser.add_argument("--merge", action="store_true",
                        help="fold the adapter into %s" % MERGED_PATH)
    parser.add_argument("--check", action="store_true",
                        help="measure a tuned checkpoint against the base")
    parser.add_argument("--tuned", default=MERGED_PATH,
                        help="the merged checkpoint --check reads")
    parser.add_argument("--adapter", default=ADAPTER_PATH,
                        help="the adapter --merge reads")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--stack", type=int, default=TRAIN_STACK,
                        help="gradient accumulation steps")
    parser.add_argument("--rate", type=float, default=TRAIN_RATE)
    parser.add_argument("--cut", type=int, default=TRAIN_CUT,
                        help="longest row in tokens; longer rows are dropped")
    parser.add_argument("--rank", type=int, default=LORA_RANK,
                        help="LoRA rank; alpha follows at twice it")
    parser.add_argument("--anchor", type=float, default=0.0,
                        help="weight of the KL term holding the tune near the base")
    parser.add_argument("--no-checkpointing", dest="checkpointing",
                        action="store_false", default=True,
                        help="keep activations rather than recomputing them")
    parser.add_argument("--room", type=int, default=512,
                        help="tokens --check lets each reply run to")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    flag = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not any((flag.lint, flag.make_data, flag.uncensor, flag.train,
                flag.merge, flag.check)):
        print("nothing to do: pass --lint, --make-data, --uncensor, --train, "
              "--merge or --check")
        return 0

    # The rule engine needs nothing installed, so the audit and the press run
    # before the import check and report properly on a bare machine.
    if flag.lint:
        if not os.path.isfile(flag.data):
            print("skip: no corpus at %s" % flag.data)
            return 0
        if lint_data(flag.data):
            return 1
    if flag.make_data:
        if not flag.source:
            print("skip: --make-data needs --source, a JSONL file or a dataset id")
            return 0
        # Into build/, which util/make.py generates and git ignores: the source
        # tree's file list is closed, and a generated corpus is not part of it.
        # Merge what survives into data/tune.jsonl by hand, or point --data at it.
        out_path = flag.out or os.path.join(TUNE_PATH, "data_more.jsonl")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        make_data(flag.source, out_path, flag.think_level, flag.reply_level,
                  flag.floor, flag.limit)
    if not (flag.uncensor or flag.train or flag.merge or flag.check):
        return 0
    if not need_modules():
        return 0

    if flag.uncensor:
        if not flag.model or not os.path.isdir(flag.model):
            print("skip: no checkpoint folder given; pass --model or set INFERLIQU_MODEL")
            return 0
        if not need_heretic():
            return 0
        made = uncensor_step(flag.model, flag.uncensor_out, flag)
        if not made:
            print("skip: the abliteration wrote no checkpoint to %s"
                  % flag.uncensor_out)
            return 0
        print("uncensored checkpoint written to %s" % made)
        # Every step after this one reads the decensored weights as its base,
        # so `--uncensor --train --merge` is one command and the adapter trains
        # over what will actually be served. --check compares against the same
        # base the adapter saw, which is the comparison that means anything:
        # measured against the original it would be reporting the abliteration
        # and the register together as one number.
        flag.model = made

    if flag.train:
        if not flag.model or not os.path.isdir(flag.model):
            print("skip: no checkpoint folder given; pass --model or set INFERLIQU_MODEL")
            return 0
        if not os.path.isfile(flag.data):
            print("skip: no corpus at %s" % flag.data)
            return 0
        made = train_step(flag.model, TUNE_PATH, flag.data, flag)
        print("adapter written to %s" % made)
    if flag.merge:
        if not os.path.isdir(flag.adapter):
            print("skip: no adapter at %s; run --train first" % flag.adapter)
            return 0
        made = merge_step(flag.model, flag.adapter, MERGED_PATH)
        print("merged checkpoint written to %s" % made)
    if flag.check:
        if not os.path.isdir(flag.tuned):
            print("skip: no tuned checkpoint at %s; run --merge first" % flag.tuned)
            return 0
        return check_step(flag.model, flag.tuned, flag.data, flag.room, flag.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
