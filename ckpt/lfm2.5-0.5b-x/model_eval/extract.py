"""WDS data loading, schema-aware extraction prompts, local model inference,
and JSON-from-noise parsing — everything the trained-checkpoint stage needs.

Public entry: `run_extraction(samples, model_path, backend, ...)` returns a
list of records ready for the judge stage.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Iterator, Literal

import webdataset as wds

logger = logging.getLogger(__name__)

_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_EXTRACTION_TPL = Template((_PROMPT_DIR / "extraction_system.txt").read_text(encoding="utf-8"))


# ─── data loading ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalSample:
    key: str
    image_bytes: bytes
    schema: dict[str, str]
    ground_truth: dict[str, object]


def discover_tar_files(data_path: str) -> list[str]:
    """Resolve a path/glob/brace-expansion to a sorted list of `.tar` files."""
    if "{" in data_path and ".." in data_path:
        expanded = list(wds.shardlists.expand_urls(data_path))
        if expanded and Path(expanded[0]).is_dir():
            tars: list[str] = []
            for d in expanded:
                if Path(d).is_dir():
                    tars.extend(sorted(str(f) for f in Path(d).rglob("*.tar")))
            if not tars:
                raise FileNotFoundError(f"No .tar files found in: {data_path}")
            return tars
        return expanded

    p = Path(data_path)
    if p.is_file() and p.suffix == ".tar":
        return [str(p)]
    if p.is_dir():
        tars = sorted(str(f) for f in p.rglob("*.tar"))
        if not tars:
            raise FileNotFoundError(f"No .tar files found in {data_path}")
        return tars
    parent = p.parent
    tars = sorted(str(f) for f in parent.glob(p.name))
    if not tars:
        raise FileNotFoundError(f"No files matching pattern: {data_path}")
    return tars


def _first_image(sample: dict) -> bytes | None:
    """Return the first image, preferring `imgN.jpg` order then legacy keys."""
    multi: list[tuple[int, bytes]] = []
    for k, v in sample.items():
        if not isinstance(v, (bytes, bytearray)) or not k.startswith("img"):
            continue
        head, _, ext = k.partition(".")
        if ext.lower() not in _IMAGE_EXTS:
            continue
        idx_str = head[3:]
        if not idx_str.isdigit():
            continue
        multi.append((int(idx_str), bytes(v)))
    if multi:
        multi.sort(key=lambda x: x[0])
        return multi[0][1]
    for k in _IMAGE_EXTS:
        v = sample.get(k)
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
    return None


def _decode_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def iter_eval_samples(
    data_path: str,
    *,
    skip: int = 0,
    limit: int = 0,
) -> Iterator[EvalSample]:
    """Yield up to `limit` EvalSamples from WDS tars.

    Each sample carries `<key>.jpg`, `<key>.key_explanations` (schema with
    descriptions), and `<key>.structured_text` (ground-truth values).
    Samples missing image/schema/labels are silently skipped.
    """
    tar_files = discover_tar_files(data_path)
    logger.info("Discovered %d tar file(s) under %s", len(tar_files), data_path)

    dataset = wds.WebDataset(
        tar_files,
        shardshuffle=False,
        nodesplitter=None,
        handler=lambda e: logger.warning("WDS skip: %s", e) or True,
    )
    n_seen = 0
    n_yielded = 0
    for sample in dataset:
        img = _first_image(sample)
        ke = sample.get("key_explanations")
        st = sample.get("structured_text")
        if img is None or ke is None or st is None:
            continue
        try:
            schema = json.loads(_decode_text(ke))
            gt = json.loads(_decode_text(st))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Skip %s: bad JSON (%s)", sample.get("__key__", "?"), e)
            continue
        if not isinstance(schema, dict) or not isinstance(gt, dict):
            continue
        n_seen += 1
        if n_seen <= skip:
            continue
        yield EvalSample(
            key=str(sample.get("__key__", f"sample_{n_seen}")),
            image_bytes=img,
            schema=schema,
            ground_truth=gt,
        )
        n_yielded += 1
        if limit and n_yielded >= limit:
            break
    logger.info("Yielded %d eval sample(s) (skipped %d)", n_yielded, skip)


# ─── prompt rendering ──────────────────────────────────────────────────────


def schema_to_yaml(schema: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in schema.items())


def build_extraction_prompt(schema: dict[str, str]) -> str:
    return _EXTRACTION_TPL.substitute(schema=schema_to_yaml(schema))


# ─── JSON parsing ──────────────────────────────────────────────────────────


def sanitize_output(text: str) -> str:
    """Strip whitespace + markdown fences + bare `json` prefix."""
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = "" if nl == -1 else s[nl + 1 :]
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    head = s.split("\n", 1)
    if head and head[0].strip().lower() == "json":
        s = head[1] if len(head) > 1 else ""
        s = s.strip()
    return s


def _first_balanced(text: str, start: int) -> str | None:
    """Return `text[start:i+1]` when braces balance; None if never balances."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# Bare empty-string entries inside an object: ` "",` or `\n  ""\n}`.
# Some VLMs emit these as a runaway-collapse pattern.
_BARE_EMPTY_RE = re.compile(r',\s*""\s*(?=[,}])')
_BARE_EMPTY_BEFORE_CLOSE_RE = re.compile(r',\s*""\s*(?=\n*\s*})')


def extract_json_strict_first(text: str) -> tuple[dict, bool]:
    """Sanitize + parse. Returns `(dict, was_strict)`.

    `was_strict=True` if the strict parse succeeded — that's what
    `json_valid` reports. False covers repaired-success and total failure
    (caller distinguishes via `bool(dict)`).
    """
    sanitized = sanitize_output(text)
    if not sanitized:
        return {}, False
    start = sanitized.find("{")
    if start == -1:
        return {}, False

    candidate = _first_balanced(sanitized, start)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except (json.JSONDecodeError, ValueError):
            pass

    # Second-chance repair (ported from old bundle's `_repair_parse`):
    # try original `bal`, then progressively repaired versions, then the
    # last-`}` truncation with both repairs applied. First dict wins.
    candidates: list[str] = []
    bal = _first_balanced(sanitized[start:], 0)
    if bal is not None:
        candidates.append(bal)
        c2 = _BARE_EMPTY_RE.sub("", bal)
        c2 = _BARE_EMPTY_BEFORE_CLOSE_RE.sub("", c2)
        candidates.append(c2)
        candidates.append(_TRAILING_COMMA_RE.sub(r"\1", c2))
    last_close = sanitized.rfind("}")
    if last_close >= 0:
        tail = sanitized[: last_close + 1]
        candidates.append(tail)
        tail2 = _BARE_EMPTY_RE.sub("", tail)
        tail2 = _BARE_EMPTY_BEFORE_CLOSE_RE.sub("", tail2)
        tail2 = _TRAILING_COMMA_RE.sub(r"\1", tail2)
        candidates.append(tail2)

    for c in candidates:
        try:
            parsed = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed, False
    return {}, False


# ─── extraction backends ───────────────────────────────────────────────────


def _img_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_chat_messages(schema: dict[str, str], img_bytes: bytes) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_extraction_prompt(schema)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _img_to_data_url(img_bytes)}},
            ],
        },
    ]


def _extract_vllm(
    samples: list[EvalSample],
    *,
    model_path: str,
    max_model_len: int,
    gpu_mem_util: float,
    max_new_tokens: int,
) -> list[str]:
    """vLLM offline batch extraction. One shot, no retries — Ctrl+C if hung."""
    from vllm import LLM  # type: ignore

    logger.info("Initializing vLLM for %s …", model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enable_prefix_caching=True,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 1},
    )
    from vllm import SamplingParams  # type: ignore

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    conversations = [_build_chat_messages(s.schema, s.image_bytes) for s in samples]
    logger.info("vLLM.chat over %d samples …", len(samples))
    # Suppress reasoning for extraction-side reasoning models (Qwen3 family,
    # gpt-oss family). Without this they burn the token budget on internal
    # <think> blocks and emit no JSON. Non-reasoning models silently ignore.
    outputs = llm.chat(
        conversations,
        sampling_params=sp,
        use_tqdm=True,
        chat_template_kwargs={
            "enable_thinking": False,
            "reasoning_effort": "low",
        },
    )
    texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
    return texts


def _extract_hf(
    samples: list[EvalSample],
    *,
    model_path: str,
    max_new_tokens: int,
    batch: int,
) -> list[str]:
    """HF transformers fallback. Slower but works without vLLM (e.g. Mac)."""
    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

    logger.info("Loading HF model %s …", model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Decoder-only generation requires left padding so the model never sees
    # padding tokens in the middle of the sequence at decode time.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(samples), batch):
        chunk = samples[start : start + batch]
        msgs = [_build_chat_messages(s.schema, s.image_bytes) for s in chunk]
        # The processor strips the image_url data URIs and replaces with PIL.
        for m, s in zip(msgs, chunk):
            m[1]["content"][0] = {"type": "image", "image": Image.open(io.BytesIO(s.image_bytes))}
        inputs = processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            # Suppress reasoning blocks (Qwen3 family) — kwarg flows into the
            # model's Jinja chat template. Non-reasoning models ignore it.
            enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        outputs.extend(decoded)
        logger.info("HF extraction: %d/%d", min(start + batch, len(samples)), len(samples))
    return outputs


def _extract_smolvlm(
    samples: list[EvalSample],
    *,
    model_path: str,
    max_new_tokens: int,
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.85,
) -> list[str]:
    """SmolVLM / Idefics3-family extraction via vLLM with user-prompt format.

    Why a dedicated path:
      - SmolVLM was trained on user/assistant turns only; system messages
        carry weak signal and trigger generic image-captioning behavior
        rather than schema following. So we put the schema in the *user*
        prompt alongside the image.
      - vLLM natively supports the Idefics3 architecture (SmolVLM v1/v2),
        giving ~20× the throughput of single-sample HF generation. We use
        it directly here instead of going through the generic vLLM path
        (which would also work, but with a system-prompt template).
    """
    from vllm import LLM, SamplingParams  # type: ignore

    logger.info("Initializing vLLM for SmolVLM/Idefics3 model: %s …", model_path)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enable_prefix_caching=True,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    conversations: list[list[dict[str, Any]]] = []
    for s in samples:
        b64 = base64.b64encode(s.image_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        # User prompt (no system) — schema goes in the user turn alongside
        # the image. This is the format SmolVLM responds to.
        conversations.append([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": build_extraction_prompt(s.schema)},
            ]},
        ])

    logger.info("vLLM.chat over %d samples (SmolVLM) …", len(samples))
    outputs = llm.chat(conversations, sampling_params=sp, use_tqdm=True)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


def _is_smolvlm(model_path: str) -> bool:
    """Detect SmolVLM / Idefics3-family models from path."""
    p = model_path.lower()
    return "smolvlm" in p or "idefics" in p


def run_extraction(
    samples: list[EvalSample],
    *,
    model_path: str,
    backend: Literal["auto", "vllm", "hf"] = "auto",
    max_new_tokens: int = 1024,
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.85,
    batch: int = 8,
) -> list[dict[str, Any]]:
    """Run extraction; return one prediction record per input sample.

    `backend="auto"` tries vLLM first and falls back to HF on import error
    or init failure. `"vllm"` / `"hf"` force the choice.

    Special case: SmolVLM / Idefics3 family always uses a dedicated code
    path regardless of `backend` — vLLM doesn't support them well, and the
    standard `AutoModelForImageTextToText` invocation drops the chat
    template specifics they need.
    """
    if not samples:
        return []

    t0 = time.perf_counter()

    # SmolVLM / Idefics: dedicated path, bypass `backend` selection.
    if _is_smolvlm(model_path):
        logger.info("Detected SmolVLM/Idefics-family model — using dedicated extraction path.")
        raw_outputs = _extract_smolvlm(samples, model_path=model_path, max_new_tokens=max_new_tokens)
        backend_used = "smolvlm"
    elif backend == "hf":
        raw_outputs = _extract_hf(samples, model_path=model_path, max_new_tokens=max_new_tokens, batch=batch)
        backend_used = "hf"
    elif backend == "vllm":
        raw_outputs = _extract_vllm(
            samples,
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_mem_util=gpu_mem_util,
            max_new_tokens=max_new_tokens,
        )
        backend_used = "vllm"
    else:  # auto
        try:
            raw_outputs = _extract_vllm(
                samples,
                model_path=model_path,
                max_model_len=max_model_len,
                gpu_mem_util=gpu_mem_util,
                max_new_tokens=max_new_tokens,
            )
            backend_used = "vllm"
        except Exception as e:
            logger.warning("vLLM extraction failed (%s); falling back to HF transformers.", e)
            raw_outputs = _extract_hf(samples, model_path=model_path, max_new_tokens=max_new_tokens, batch=batch)
            backend_used = "hf"

    dt = time.perf_counter() - t0
    logger.info(
        "Extraction over %d samples took %.1fs (%.2f sample/s, backend=%s).",
        len(samples),
        dt,
        len(samples) / max(dt, 1e-9),
        backend_used,
    )

    if len(raw_outputs) != len(samples):
        raise RuntimeError(
            f"Backend returned {len(raw_outputs)} outputs for {len(samples)} samples"
        )

    records: list[dict[str, Any]] = []
    for s, raw in zip(samples, raw_outputs):
        parsed, strict = extract_json_strict_first(raw)
        records.append(
            {
                "key": s.key,
                "schema": s.schema,
                "ground_truth": s.ground_truth,
                "prediction_raw": raw,
                "prediction_json": parsed,
                "prediction_strict_valid": strict,
            }
        )
    return records
