#!/usr/bin/env python3
"""
Builds a curated agentic + reasoning SFT dataset for LiquidAI/LFM2.5-1.2B-Thinking.

Everything is emitted in the *native* LFM2.5 format:
  - tools live in a structured `tools` list; the chat template renders them as
    `List of tools: [...]` in the system prompt,
  - reasoning lives in `message["thinking"]`, rendered as `<think>...</think>`,
  - actions live in `message["tool_calls"]`, rendered as Pythonic calls between
    `<|tool_call_start|>` and `<|tool_call_end|>`.

Two properties of the official chat template drive the whole design:
  1. `preserve_thinking` defaults to False, so only the *last* assistant turn keeps
     its `<think>` block. Training on a full multi-turn trace would therefore teach
     the model to answer without thinking on every non-final turn. We instead expand
     each trace into per-turn samples and put the loss on the final turn only.
  2. The template emits `bos_token` itself, so trainers must tokenize with
     `add_special_tokens=False`.

Usage:
    python create_dataset.py --out ./lfm25-agentic-12k
    python create_dataset.py --scale 0.01 --out /tmp/smoke   # fast smoke test
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

SEED = 42
TOKENIZER_ID = "LiquidAI/LFM2.5-1.2B-Thinking"

# Samples longer than this are dropped, never truncated: a truncated <think> block
# teaches the model to stop reasoning mid-sentence.
MAX_SEQ_LEN = 4096
MAX_PROMPT_LEN = 3072
MIN_COMPLETION_LEN = 16

# Real agent registries vary in size, so vary it here too instead of always
# pinning the same count. Tools the gold answer calls are never dropped.
MIN_TOOLS_PER_SAMPLE = 3
MAX_TOOLS_PER_SAMPLE = 8
MAX_TOOL_SCHEMA_CHARS = 6000

# Cap on how many training rows a single multi-turn trace may contribute, so that
# long traces don't flood the mixture with near-duplicate prefixes.
MAX_EXPANSIONS_PER_TRACE = 2

LFM_SYSTEM = "You are a helpful assistant trained by Liquid AI."

# Small amount of system-prompt variation so the model doesn't bind its behaviour
# to one exact string. No variant spells out the reasoning or tool-call tags: the
# template owns that protocol, and echoing tag text invites the model to emit it.
TOOL_SYSTEM_VARIANTS = [
    LFM_SYSTEM,
    LFM_SYSTEM + " Call a tool only when the request actually requires one.",
    LFM_SYSTEM
    + " Use the available tools when they are needed, and ask the user for any"
    " required argument they have not provided.",
]

VALID_ROLES = {"system", "user", "assistant", "tool"}

JSON_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "dict": "object",
    "object": "object",
    "any": "string",
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"</?tool_response>", re.IGNORECASE)
TOOLS_TAG_RE = re.compile(r"</?tools>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    source: str
    category: str
    trace_id: str = ""
    style: str = "plain"


class Stats:
    """Per-source accounting so silent zero-yield extractors are impossible to miss."""

    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = defaultdict(Counter)

    def bump(self, source: str, reason: str, n: int = 1) -> None:
        self.counters[source][reason] += n

    def report(self) -> None:
        print("\n--- extraction report -------------------------------------------")
        for source in sorted(self.counters):
            if source == "caveman":  # reported separately, it is not a source
                continue
            c = self.counters[source]
            kept = c.get("kept", 0)
            dropped = sorted(
                ((k, v) for k, v in c.items() if k != "kept"),
                key=lambda kv: -kv[1],
            )
            detail = ", ".join(f"{k}={v}" for k, v in dropped) or "no drops"
            print(f"  {source:<34} kept={kept:<6} {detail}")


STATS = Stats()


# ---------------------------------------------------------------------------
# Tool-schema normalisation
# ---------------------------------------------------------------------------


def _json_type(raw: Any) -> str:
    """Map loose annotations ('str, optional', 'List[int]') onto JSON Schema types."""
    token = str(raw).split(",")[0].split("[")[0].strip().lower()
    return JSON_TYPE_MAP.get(token, "string")


def _normalize_property(spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {"type": "string"}
    prop: Dict[str, Any] = {"type": _json_type(spec.get("type", "string"))}
    if spec.get("description"):
        prop["description"] = str(spec["description"])
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        prop["enum"] = spec["enum"]
    if prop["type"] == "array":
        items = spec.get("items")
        prop["items"] = (
            {"type": _json_type(items.get("type", "string"))}
            if isinstance(items, dict)
            else {"type": "string"}
        )
    return prop


def _is_optional(name: str, spec: Any, declared_required: Optional[List[str]]) -> bool:
    if declared_required is not None:
        return name not in declared_required
    if not isinstance(spec, dict):
        return True
    if "optional" in str(spec.get("type", "")).lower():
        return True
    return "default" in spec


def normalize_parameters(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "required": []}

    raw_props = params.get("properties")
    if not isinstance(raw_props, dict):
        # Some sources put the properties directly under `parameters`.
        raw_props = {
            k: v
            for k, v in params.items()
            if k not in {"type", "required", "additionalProperties", "$schema"}
        }

    declared = params.get("required")
    declared = declared if isinstance(declared, list) else None

    properties = {str(k): _normalize_property(v) for k, v in raw_props.items()}
    required = [k for k, v in raw_props.items() if not _is_optional(str(k), v, declared)]
    return {"type": "object", "properties": properties, "required": required}


def normalize_tool(tool: Any) -> Optional[Dict[str, Any]]:
    if isinstance(tool, str):
        tool = parse_loose_json(TOOLS_TAG_RE.sub("", tool).strip())
    if not isinstance(tool, dict):
        return None
    if "function" in tool and isinstance(tool["function"], dict):
        tool = tool["function"]
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    # The template renders calls as Python source, so the name must be an identifier.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name.strip()):
        return None
    return {
        "name": name.strip(),
        "description": str(tool.get("description") or "").strip(),
        "parameters": normalize_parameters(tool.get("parameters")),
    }


def normalize_tools(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = parse_json_documents(TOOLS_TAG_RE.sub("", raw))
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw:
        # smoltalk2 stores a whole registry as one newline-separated JSON string.
        candidates = (
            parse_json_documents(TOOLS_TAG_RE.sub("", item)) if isinstance(item, str) else [item]
        )
        for candidate in candidates:
            tool = normalize_tool(candidate)
            if tool and tool["name"] not in seen:
                seen.add(tool["name"])
                out.append(tool)
    return out


def select_tools(
    tools: List[Dict[str, Any]], required_names: Iterable[str], rng: random.Random
) -> Optional[List[Dict[str, Any]]]:
    """Prune the registry while guaranteeing every called tool stays available.

    Blindly slicing `tools[:N]` can delete the tool the gold answer calls, which
    trains the model to invent functions that are not in its registry.
    """
    by_name = {t["name"]: t for t in tools}
    required = [by_name[n] for n in dict.fromkeys(required_names) if n in by_name]
    if len(required) != len(set(required_names)):
        return None  # a called tool is missing from the registry -> unusable sample

    budget = rng.randint(MIN_TOOLS_PER_SAMPLE, MAX_TOOLS_PER_SAMPLE)
    budget = max(budget, len(required))

    distractors = [t for t in tools if t["name"] not in {r["name"] for r in required}]
    rng.shuffle(distractors)
    selected = required + distractors[: max(0, budget - len(required))]
    rng.shuffle(selected)

    while selected and len(json.dumps(selected)) > MAX_TOOL_SCHEMA_CHARS:
        removable = [t for t in selected if t["name"] not in {r["name"] for r in required}]
        if not removable:
            break
        selected.remove(removable[-1])
    return selected


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


def parse_loose_json(text: str) -> Any:
    """Parse JSON, falling back to Python literals (smoltalk2 uses single quotes)."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def parse_json_documents(text: str) -> List[Any]:
    """Parse one or more JSON values from a string.

    smoltalk2 packs an entire tool registry into a single newline-separated string
    of JSON objects, which is neither a JSON array nor a single JSON document.
    """
    single = parse_loose_json(text)
    if single is not None:
        return single if isinstance(single, list) else [single]

    decoder = json.JSONDecoder()
    out: List[Any] = []
    idx, length = 0, len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            value, idx = decoder.raw_decode(text, idx)
        except ValueError:
            break
        out.append(value)
    return out


def parse_assistant(content: str) -> Optional[Dict[str, Any]]:
    """Split a Hermes-style assistant turn into thinking / text / tool_calls.

    Returns None when a `<tool_call>` block cannot be parsed; a half-parsed action
    is worse than a dropped sample.
    """
    if not isinstance(content, str):
        return None

    thinking_parts = [m.group(1).strip() for m in THINK_RE.finditer(content)]
    body = THINK_RE.sub("", content)

    # An unterminated <think> means the reasoning ran into the answer: unrecoverable.
    if "<think>" in body or "</think>" in body:
        return None

    tool_calls: List[Dict[str, Any]] = []
    for raw in TOOL_CALL_RE.findall(body):
        parsed = parse_loose_json(raw)
        if not isinstance(parsed, dict):
            return None
        name = parsed.get("name")
        args = parsed.get("arguments", parsed.get("parameters", {}))
        if isinstance(args, str):
            args = parse_loose_json(args)
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        # The chat template raises if arguments are not a mapping.
        tool_calls.append({"type": "function", "function": {"name": name, "arguments": args}})

    if "<tool_call>" in body or "</tool_call>" in body:
        body = TOOL_CALL_RE.sub("", body)
        if "<tool_call>" in body or "</tool_call>" in body:
            return None
    else:
        body = TOOL_CALL_RE.sub("", body)

    return {
        "thinking": "\n\n".join(p for p in thinking_parts if p),
        "content": body.strip(),
        "tool_calls": tool_calls,
    }


def parse_tool_result(content: Any) -> Optional[str]:
    """Unwrap `<tool_response>` / `Observation:` envelopes down to the raw result."""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    if not isinstance(content, str):
        return None

    text = TOOL_RESPONSE_RE.sub("", content).strip()
    text = re.sub(r"^observation:\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return None

    parsed = parse_loose_json(text)
    if isinstance(parsed, dict) and "content" in parsed and {"name", "tool_call_id"} & parsed.keys():
        inner = parsed["content"]
        return inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
    return text


def canonical_role(raw: Any) -> Optional[str]:
    role = str(raw or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    if role in {"tool", "function", "observation", "tool_response", "ipython"}:
        return "tool"
    return None


def normalize_conversation(
    raw_messages: Iterable[Any], content_key: str = "content", role_key: str = "role"
) -> Optional[List[Dict[str, Any]]]:
    """Convert a heterogeneous conversation into LFM2.5 messages, or None if invalid."""
    messages: List[Dict[str, Any]] = []
    for turn in raw_messages:
        if not isinstance(turn, dict):
            return None
        role = canonical_role(turn.get(role_key) or turn.get("from") or turn.get("role"))
        if role is None:
            return None  # unknown roles must not be smuggled into the template
        raw_content = turn.get(content_key)
        if raw_content is None:
            raw_content = turn.get("value") if "value" in turn else turn.get("content")

        if role == "assistant":
            parsed = parse_assistant(raw_content or "")
            if parsed is None:
                return None
            if not parsed["content"] and not parsed["tool_calls"]:
                return None  # empty action AND empty answer -> nothing to learn
            messages.append({"role": "assistant", **parsed})
        elif role == "tool":
            text = parse_tool_result(raw_content)
            if text is None:
                return None
            messages.append({"role": "tool", "content": text})
        else:
            text = raw_content if isinstance(raw_content, str) else ""
            if role == "user" and not text.strip():
                return None
            messages.append({"role": role, "content": text.strip()})

    return messages or None


def called_tool_names(messages: List[Dict[str, Any]]) -> List[str]:
    return [
        call["function"]["name"]
        for m in messages
        if m["role"] == "assistant"
        for call in m.get("tool_calls", [])
    ]


def structurally_valid(messages: List[Dict[str, Any]]) -> bool:
    if not messages or messages[-1]["role"] != "assistant":
        return False
    if not any(m["role"] == "user" for m in messages):
        return False
    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            return False
        # A tool result must answer an assistant turn that actually called a tool.
        if m["role"] == "tool":
            if i == 0 or messages[i - 1]["role"] not in {"assistant", "tool"}:
                return False
            prior = [p for p in messages[:i] if p["role"] == "assistant"]
            if not prior or not prior[-1].get("tool_calls"):
                return False
    last = messages[-1]
    return bool(last.get("content") or last.get("tool_calls"))


def with_system(messages: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    """Replace any source-specific system prompt with the LFM2.5 one.

    Hermes/smolagents system prompts describe `<tools>`/`<tool_call>` XML, which is
    the wrong protocol for LFM2.5 and would fight the chat template.
    """
    body = [m for m in messages if m["role"] != "system"]
    return [{"role": "system", "content": prompt}] + body


def expand_turns(
    messages: List[Dict[str, Any]], rng: random.Random, max_expansions: int
) -> List[List[Dict[str, Any]]]:
    """Cut a trace into prefixes that each end on an assistant turn.

    Needed because the chat template keeps `<think>` only on the final assistant
    turn; each training row therefore supervises exactly one reasoning step.
    """
    indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not indices:
        return []
    if len(indices) > max_expansions:
        chosen = {indices[-1]} | set(rng.sample(indices[:-1], max_expansions - 1))
        indices = sorted(chosen)
    return [messages[: i + 1] for i in indices]


# ---------------------------------------------------------------------------
# Caveman compression
# ---------------------------------------------------------------------------
#
# Style rules follow the `caveman` skill, restricted to the transforms that are
# measurably cheaper under *this* model's tokenizer (65,536 vocab). Verified:
#   " the file" 2 tok -> " file" 1 tok      articles are a real saving
#   " configuration" 1 tok -> " cfg" 2 tok  invented abbreviations COST tokens
#   " A -> B" == " A then B" == 3 tok       arrows save nothing
#   " sees" == " see" == 1 tok              mangling grammar saves nothing
# So: drop filler and articles, never abbreviate, never mangle verbs, never
# insert arrows. Compression must only ever remove.

CAVEMAN_DIRECTIVE = (
    "Reply caveman style: no filler, no articles, short sentences. "
    "Keep code, paths, numbers, and negations exact."
)

# Dropping any of these inverts the meaning of a sentence, which is strictly
# worse than any number of tokens saved.
NEGATIONS = {
    "not", "no", "never", "none", "nor", "neither", "without", "except", "only",
    "cannot", "unless", "n't", "nothing", "nowhere", "fail", "fails", "failed",
}

PROTECTED_PATTERNS = [
    r"```.*?```",                        # fenced code
    r"`[^`\n]+`",                        # inline code
    r"https?://\S+",                     # URLs
    r"\"[^\"\n]{0,300}\"",               # quoted strings (often literal arguments)
    r"\$[^$\n]{1,200}\$",                # inline math
    r"\b[A-Za-z_][\w./\\-]*\.[A-Za-z]{1,6}\b",  # paths, filenames, dotted names
    r"\b[A-Za-z_]\w*\([^)\n]{0,200}\)",  # function calls
    r"\b\d[\d,._:/-]*(?:\s*%)?",         # numbers, versions, units
]
PROTECT_RE = re.compile("|".join(PROTECTED_PATTERNS), re.DOTALL)
SENTINEL = "\x00"

DROP_SENTENCE_RE = re.compile(
    r"(?:(?<=^)|(?<=[.!?]\s)|(?<=[.!?]\n))\s*"
    r"(?:sure|certainly|absolutely|of course|no problem|great question|"
    r"i(?:'?d| would| will| can) be happy to|i'?ll help|happy to help|i hope this helps|"
    r"let me know if|feel free to|does that help)"
    r"[^.!?\n]*[.!?]+\s*",
    re.IGNORECASE,
)

FILLER_RE = re.compile(
    r"\b(?:just|really|basically|actually|simply|essentially|literally|quite|very|"
    r"obviously|clearly|indeed|somewhat|rather|kind of|sort of)\s+",
    re.IGNORECASE,
)

PHRASE_RULES = [
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    # Bounded to the complete idiom: stripping a bare "the reason" strands the verb
    # ("The reason X re-renders is ..." -> "X re-renders is ...").
    (re.compile(r"\bthe reason (?:why )?(?:is|was) (?:that|because)\b", re.I), "because"),
    (re.compile(r"\bat this point in time\b", re.I), "now"),
    (re.compile(r"\bit (?:is|'s) important to note that\b", re.I), ""),
    (re.compile(r"\bit(?:'s| is) worth noting that\b", re.I), ""),
    (re.compile(r"\bit should be noted that\b", re.I), ""),
    (re.compile(r"\bplease note that\b", re.I), ""),
    (re.compile(r"\bas (?:you can|we) (?:see|can see),?\s*", re.I), ""),
    (re.compile(r"\bi (?:think|believe|suspect) (?:that )?", re.I), ""),
    (re.compile(r"\b(?:make sure|be sure) (?:to|that)\b", re.I), "ensure"),
    (re.compile(r"\b(?:is|are) able to\b", re.I), "can"),
    (re.compile(r"\bhas the ability to\b", re.I), "can"),
    (re.compile(r"\ba (?:large|great|significant) number of\b", re.I), "many"),
    (re.compile(r"\bin the event that\b", re.I), "if"),
    (re.compile(r"\bwith regard(?:s)? to\b", re.I), "for"),
    (re.compile(r"\bin addition,?\s*", re.I), ""),
    (re.compile(r"\bfurthermore,?\s*", re.I), ""),
    (re.compile(r"\badditionally,?\s*", re.I), ""),
    (re.compile(r"\butilize\b", re.I), "use"),
    (re.compile(r"\byou should\s+", re.I), ""),
    (re.compile(r"\bwe need to\s+", re.I), ""),
    (re.compile(r"\bwe can\s+", re.I), ""),
]

ARTICLE_RE = re.compile(r"\b(?:the|a|an)\s+", re.IGNORECASE)

# Caveman's Auto-Clarity carve-out: never compress a turn whose job is to warn.
HAZARD_RE = re.compile(
    r"\b(?:delete|deletes|deleting|drop table|truncate|rm -rf|force[- ]push|"
    r"irreversible|permanently|overwrite|credential|password|secret|token|"
    r"destructive|cannot be undone|data loss)\b",
    re.IGNORECASE,
)


def _mask_protected(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []

    def take(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"{SENTINEL}{len(spans) - 1}{SENTINEL}"

    return PROTECT_RE.sub(take, text), spans


def _unmask(text: str, spans: List[str]) -> Optional[str]:
    def put(match: re.Match) -> str:
        return spans[int(match.group(1))]

    restored = re.sub(f"{SENTINEL}(\\d+){SENTINEL}", put, text)
    return None if SENTINEL in restored else restored


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text.strip()


def compress_prose(text: str, drop_articles: bool) -> Optional[str]:
    """Remove filler from prose while leaving every protected span byte-identical."""
    if not text.strip():
        return None

    masked, spans = _mask_protected(text)
    masked = DROP_SENTENCE_RE.sub("", masked)
    for pattern, replacement in PHRASE_RULES:
        masked = pattern.sub(replacement, masked)
    masked = FILLER_RE.sub("", masked)
    if drop_articles:
        masked = ARTICLE_RE.sub("", masked)

    restored = _unmask(_tidy(masked), spans)
    return restored or None


def preserves_meaning(original: str, compressed: str) -> bool:
    """Reject any rewrite that moved a negation, a number, or a protected span."""
    if not compressed.strip():
        return False
    for word in NEGATIONS:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        if len(pattern.findall(original)) != len(pattern.findall(compressed)):
            return False
    if re.findall(r"\d", original) != re.findall(r"\d", compressed):
        return False
    return PROTECT_RE.findall(original) == PROTECT_RE.findall(compressed)


def compress_sample(
    sample: Sample, tokenizer, target: str, drop_articles: bool, min_saving: float
) -> Optional[Sample]:
    """Produce a caveman variant of `sample`, or None if it is not safely compressible.

    Tool call names and arguments are never reachable here: only `content` and
    `thinking` strings are rewritten, so actions survive structurally intact.
    """
    last = sample.messages[-1]
    if HAZARD_RE.search(last.get("content") or "") or HAZARD_RE.search(last.get("thinking") or ""):
        return None

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0

    before = after = 0
    messages: List[Dict[str, Any]] = []
    changed = False

    for i, message in enumerate(sample.messages):
        # User turns and tool results are left alone. Adobe's CAVEWOMAN result is
        # that compressing the *input* makes models answer longer and worse.
        if message["role"] != "assistant":
            messages.append(message)
            continue

        new = dict(message)
        is_last = i == len(sample.messages) - 1
        for field_name, enabled in (
            ("thinking", target in {"think", "both"} and is_last),
            ("content", target in {"answer", "both"}),
        ):
            original = message.get(field_name) or ""
            if not enabled or not original:
                continue
            compressed = compress_prose(original, drop_articles)
            if not compressed or not preserves_meaning(original, compressed):
                continue
            new[field_name] = compressed
            if is_last:
                before += count(original)
                after += count(compressed)
            changed = True
        messages.append(new)

    if not changed or before == 0:
        return None
    if (before - after) / before < min_saving:
        return None

    STATS.bump("caveman", "tokens_before", before)
    STATS.bump("caveman", "tokens_after", after)

    return Sample(
        messages=with_system(messages, f"{LFM_SYSTEM} {CAVEMAN_DIRECTIVE}"),
        tools=sample.tools,
        source=sample.source,
        category=f"{sample.category}+caveman",
        trace_id=sample.trace_id,  # keeps a converted/paired row on one side of the split
        style="caveman",
    )


def apply_caveman(
    samples: List[Sample],
    tokenizer,
    ratio: float,
    mode: str,
    target: str,
    drop_articles: bool,
    min_saving: float,
    rng: random.Random,
) -> List[Sample]:
    if ratio <= 0:
        return samples

    order = list(range(len(samples)))
    rng.shuffle(order)
    wanted = int(len(samples) * ratio)
    produced: Dict[int, Sample] = {}

    for index in order:
        if len(produced) >= wanted:
            break
        variant = compress_sample(samples[index], tokenizer, target, drop_articles, min_saving)
        if variant is None:
            STATS.bump("caveman", "not_compressible")
            continue
        produced[index] = variant

    if mode == "pair":
        return samples + list(produced.values())
    return [produced.get(i, s) for i, s in enumerate(samples)]


# ---------------------------------------------------------------------------
# Source iteration
# ---------------------------------------------------------------------------


def iter_source(
    name: str,
    config: Optional[str],
    split: str,
    streaming: bool,
    seed: int,
    buffer_size: int = 20_000,
) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(name, config, split=split, streaming=streaming)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size) if streaming else ds.shuffle(seed=seed)
    return iter(ds)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

HERMES_REASONING = "interstellarninja/hermes_reasoning_tool_use"


def build_hermes_reasoning(quotas: Dict[str, int], rng: random.Random) -> List[Sample]:
    """Reason-then-act traces with authentic <think> blocks and a real `tools` column.

    `mlx-community/hermes-reasoning-tool-use` is a reformatted copy of this dataset
    that drops the `tools` column and bakes the Hermes system prompt into the
    conversation, so we use the upstream version instead.
    """
    source = "hermes-reasoning-tool-use"
    print(f"[{source}] target={quotas}")
    remaining = dict(quotas)
    out: List[Sample] = []

    for idx, item in enumerate(iter_source(HERMES_REASONING, None, "train", False, SEED)):
        if not any(v > 0 for v in remaining.values()):
            break
        category = item.get("scenario_category") or "single"
        if remaining.get(category, 0) <= 0:
            continue

        messages = normalize_conversation(item.get("conversations") or [], content_key="value", role_key="from")
        if not messages:
            STATS.bump(source, "unparsable")
            continue

        tools = normalize_tools(item.get("tools"))
        if not tools:
            STATS.bump(source, "no_tools")
            continue

        messages = with_system(messages, rng.choice(TOOL_SYSTEM_VARIANTS))
        if not structurally_valid(messages):
            STATS.bump(source, "invalid_structure")
            continue

        selected = select_tools(tools, called_tool_names(messages), rng)
        if selected is None:
            STATS.bump(source, "called_tool_not_in_registry")
            continue

        max_exp = MAX_EXPANSIONS_PER_TRACE if category == "multiturn" else 1
        for prefix in expand_turns(messages, rng, max_exp):
            if remaining.get(category, 0) <= 0:
                break
            if not structurally_valid(prefix):
                continue
            out.append(
                Sample(
                    messages=prefix,
                    tools=selected,
                    source=source,
                    category=category,
                    trace_id=f"{source}:{idx}",
                )
            )
            remaining[category] -= 1

    for category, left in remaining.items():
        if left > 0:
            STATS.bump(source, f"quota_shortfall[{category}]", left)
    return out


def build_smolagents(num_samples: int, rng: random.Random) -> List[Sample]:
    """Multi-turn agentic traces (search / visit / final_answer) with real reasoning."""
    source = "smoltalk2-smolagents"
    print(f"[{source}] target={num_samples}")
    out: List[Sample] = []

    stream = iter_source(
        "HuggingFaceTB/smoltalk2", "SFT", "smolagents_toolcalling_traces_think", True, SEED
    )
    for idx, item in enumerate(stream):
        if len(out) >= num_samples:
            break
        kwargs = item.get("chat_template_kwargs") or {}
        tools = normalize_tools(kwargs.get("xml_tools")) + normalize_tools(kwargs.get("python_tools"))
        if not tools:
            STATS.bump(source, "no_tools")
            continue

        messages = normalize_conversation(item.get("messages") or [])
        if not messages:
            STATS.bump(source, "unparsable")
            continue

        messages = with_system(messages, rng.choice(TOOL_SYSTEM_VARIANTS))
        if not structurally_valid(messages):
            STATS.bump(source, "invalid_structure")
            continue

        selected = select_tools(tools, called_tool_names(messages), rng)
        if selected is None:
            STATS.bump(source, "called_tool_not_in_registry")
            continue

        for prefix in expand_turns(messages, rng, MAX_EXPANSIONS_PER_TRACE):
            if len(out) >= num_samples:
                break
            if not structurally_valid(prefix):
                continue
            out.append(
                Sample(
                    messages=prefix,
                    tools=selected,
                    source=source,
                    category="agentic-multistep",
                    trace_id=f"{source}:{idx}",
                )
            )
    return out


def build_mixture_of_thoughts(num_samples: int, rng: random.Random, max_tokens: int) -> List[Sample]:
    """Tool-free math/code/science reasoning, length-filtered to keep traces short.

    Prevents the tool-calling majority from eroding general reasoning, and teaches
    the model that `<think>` is for reasoning, not only for tool selection.
    """
    source = "mixture-of-thoughts"
    print(f"[{source}] target={num_samples}")
    out: List[Sample] = []

    for idx, item in enumerate(iter_source("open-r1/Mixture-of-Thoughts", "all", "train", True, SEED)):
        if len(out) >= num_samples:
            break
        # Pre-filter on the dataset's own token count before doing any work:
        # most rows are 10k-20k tokens and would be dropped later anyway.
        if int(item.get("num_tokens") or 0) > max_tokens:
            STATS.bump(source, "too_long")
            continue

        messages = normalize_conversation(item.get("messages") or [])
        if not messages:
            STATS.bump(source, "unparsable")
            continue

        messages = with_system(messages, LFM_SYSTEM)
        if not structurally_valid(messages) or not messages[-1].get("thinking"):
            STATS.bump(source, "no_reasoning")
            continue

        out.append(
            Sample(
                messages=messages,
                tools=None,
                source=source,
                category=f"reasoning-{item.get('source', 'unknown')}",
                trace_id=f"{source}:{idx}",
            )
        )
    return out


def build_systemchats(num_samples: int, rng: random.Random) -> List[Sample]:
    """Persona/system-following chat with thinking and *no* tools.

    Without tool-free chat in the mixture, a tool-heavy SFT run makes the model
    reach for `<|tool_call_start|>` even when no registry is present.
    """
    source = "smoltalk2-systemchats"
    print(f"[{source}] target={num_samples}")
    out: List[Sample] = []

    stream = iter_source(
        "HuggingFaceTB/smoltalk2", "SFT", "smoltalk_systemchats_Qwen3_32B_think", True, SEED
    )
    for idx, item in enumerate(stream):
        if len(out) >= num_samples:
            break
        messages = normalize_conversation(item.get("messages") or [])
        if not messages:
            STATS.bump(source, "unparsable")
            continue
        if called_tool_names(messages):
            STATS.bump(source, "unexpected_tool_call")
            continue

        instructions = (item.get("chat_template_kwargs") or {}).get("custom_instructions") or ""
        system = f"{LFM_SYSTEM} {instructions}".strip() if instructions else LFM_SYSTEM
        messages = with_system(messages, system)
        if not structurally_valid(messages) or not messages[-1].get("thinking"):
            STATS.bump(source, "no_reasoning")
            continue

        for prefix in expand_turns(messages, rng, 1):
            out.append(
                Sample(
                    messages=prefix,
                    tools=None,
                    source=source,
                    category="chat-no-tools",
                    trace_id=f"{source}:{idx}",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Rendering / validation
# ---------------------------------------------------------------------------

ASSISTANT_HEADER = "<|im_start|>assistant\n"


def render(sample: Sample, tokenizer) -> Optional[Dict[str, Any]]:
    """Render with the official template and split off the supervised completion.

    The split is done on the rendered string rather than by templating the prefix
    separately: dropping the final message would promote the previous assistant
    turn to "last" and re-insert its `<think>` block, desynchronising the prefix.
    """
    try:
        text = tokenizer.apply_chat_template(
            sample.messages,
            tools=sample.tools or None,
            tokenize=False,
            add_generation_prompt=False,
            preserve_thinking=False,
        )
    except Exception as exc:  # template raises on malformed tool_calls
        STATS.bump(sample.source, f"template_error:{type(exc).__name__}")
        return None

    head, sep, tail = text.rpartition(ASSISTANT_HEADER)
    if not sep or not tail.endswith("<|im_end|>\n"):
        STATS.bump(sample.source, "unsplittable")
        return None

    prompt, completion = head + sep, tail
    if prompt + completion != text:
        STATS.bump(sample.source, "split_mismatch")
        return None

    last = sample.messages[-1]
    # LFM2.5-Thinking reasons on every assistant turn; a supervised turn without a
    # <think> block teaches it to skip reasoning, so such rows are never kept.
    if not last.get("thinking"):
        STATS.bump(sample.source, "no_reasoning")
        return None
    if "<think>" not in completion:
        STATS.bump(sample.source, "thinking_lost")
        return None
    if last.get("tool_calls") and "<|tool_call_start|>" not in completion:
        STATS.bump(sample.source, "tool_call_lost")
        return None

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    if len(completion_ids) < MIN_COMPLETION_LEN:
        STATS.bump(sample.source, "completion_too_short")
        return None
    if len(prompt_ids) > MAX_PROMPT_LEN:
        STATS.bump(sample.source, "prompt_too_long")
        return None
    if len(prompt_ids) + len(completion_ids) > MAX_SEQ_LEN:
        STATS.bump(sample.source, "too_long")
        return None

    STATS.bump(sample.source, "kept")
    return {
        "text": text,
        "prompt": prompt,
        "completion": completion,
        "messages_json": json.dumps(sample.messages, ensure_ascii=False),
        "tools_json": json.dumps(sample.tools or [], ensure_ascii=False),
        "source": sample.source,
        "category": sample.category,
        "trace_id": sample.trace_id,
        "style": sample.style,
        "n_tokens": len(prompt_ids) + len(completion_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_completion_tokens": len(completion_ids),
        "n_turns": sum(1 for m in sample.messages if m["role"] in {"user", "assistant"}),
        "has_tool_call": bool(last.get("tool_calls")),
    }


def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = hashlib.sha1(row["text"].encode("utf-8")).hexdigest()
        if key in seen:
            STATS.bump(row["source"], "duplicate")
            continue
        seen.add(key)
        out.append(row)
    return out


def stratified_split(
    rows: List[Dict[str, Any]], eval_ratio: float, rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split by trace, not by row, so expanded prefixes cannot leak across splits."""
    by_source: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row["trace_id"])

    eval_traces = set()
    for source, traces in by_source.items():
        unique = sorted(set(traces))
        rng.shuffle(unique)
        n_eval = max(1, int(len(unique) * eval_ratio))
        eval_traces.update(unique[:n_eval])

    train = [r for r in rows if r["trace_id"] not in eval_traces]
    evaluation = [r for r in rows if r["trace_id"] in eval_traces]
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def summarize(name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"\n{name}: empty")
        return
    tokens = sorted(r["n_tokens"] for r in rows)
    by_source = Counter(r["source"] for r in rows)
    print(f"\n{name}: {len(rows)} rows")
    print(f"  tokens  min={tokens[0]} p50={tokens[len(tokens)//2]} p95={tokens[int(len(tokens)*0.95)]} max={tokens[-1]}")
    print(f"  with tool call: {sum(r['has_tool_call'] for r in rows) / len(rows):.1%}")
    print(f"  with <think>:   {sum('<think>' in r['completion'] for r in rows) / len(rows):.1%}")
    caveman = [r for r in rows if r["style"] == "caveman"]
    if caveman:
        print(f"  caveman style:  {len(caveman) / len(rows):.1%}")
    for source, count in by_source.most_common():
        print(f"    {source:<34} {count:>6} ({count / len(rows):.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global MAX_SEQ_LEN, MAX_PROMPT_LEN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./lfm25-agentic-12k")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale every quota (use e.g. 0.01 for a fast smoke test).",
    )
    parser.add_argument(
        "--caveman-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples rendered in caveman style. 0 disables.",
    )
    parser.add_argument(
        "--caveman-mode",
        choices=["convert", "pair"],
        default="convert",
        help="convert: replace the sample. pair: keep both, for a controlled ablation.",
    )
    parser.add_argument(
        "--caveman-target",
        choices=["think", "answer", "both"],
        default="both",
        help="Which spans to compress.",
    )
    parser.add_argument(
        "--caveman-level",
        choices=["lite", "full"],
        default="full",
        help="lite keeps articles; full drops them.",
    )
    parser.add_argument("--caveman-min-saving", type=float, default=0.10)
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    MAX_PROMPT_LEN = min(MAX_PROMPT_LEN, int(args.max_seq_len * 0.75))

    rng = random.Random(args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as exc:
        print(
            f"Could not load tokenizer '{args.tokenizer}': {exc}\n"
            "Accept the model licence and run `hf auth login` first.",
            file=sys.stderr,
        )
        return 1
    if not getattr(tokenizer, "chat_template", None):
        print(f"Tokenizer '{args.tokenizer}' has no chat template.", file=sys.stderr)
        return 1

    def q(n: int) -> int:
        return max(1, int(n * args.scale))

    samples: List[Sample] = []
    samples += build_hermes_reasoning(
        {
            "single": q(2400),
            "multistep": q(1600),
            "multiturn": q(2600),
            "relevance": q(1400),  # When2Call: answer directly / ask for missing args
        },
        rng,
    )
    samples += build_smolagents(q(2200), rng)
    samples += build_mixture_of_thoughts(q(1200), rng, max_tokens=int(args.max_seq_len * 0.7))
    samples += build_systemchats(q(600), rng)

    samples = apply_caveman(
        samples,
        tokenizer,
        ratio=args.caveman_ratio,
        mode=args.caveman_mode,
        target=args.caveman_target,
        drop_articles=args.caveman_level == "full",
        min_saving=args.caveman_min_saving,
        rng=rng,
    )

    print(f"\nExtracted {len(samples)} candidate samples; rendering...")
    rows = [r for r in (render(s, tokenizer) for s in samples) if r is not None]
    rows = dedup(rows)

    STATS.report()

    caveman_stats = STATS.counters.get("caveman")
    if caveman_stats and caveman_stats.get("tokens_before"):
        before = caveman_stats["tokens_before"]
        after = caveman_stats["tokens_after"]
        print(
            f"\ncaveman: {before} -> {after} completion tokens on compressed turns "
            f"({1 - after / before:.1%} cut, {caveman_stats.get('not_compressible', 0)} rejected)"
        )

    if not rows:
        print("\nNo samples survived rendering.", file=sys.stderr)
        return 1

    train, evaluation = stratified_split(rows, args.eval_ratio, rng)
    summarize("train", train)
    summarize("validation", evaluation)

    dataset = DatasetDict(
        {"train": Dataset.from_list(train), "validation": Dataset.from_list(evaluation)}
    )
    dataset.save_to_disk(args.out)
    for name, rows_ in (("train", train), ("validation", evaluation)):
        with open(f"{args.out.rstrip('/')}.{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows_:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to '{args.out}' (+ .train.jsonl / .validation.jsonl).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
