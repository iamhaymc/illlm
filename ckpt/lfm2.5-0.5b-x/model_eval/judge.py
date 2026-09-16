"""OpenRouter-backed VLM judge.

One prompt per sample (batched-over-keys in one JSON answer), run
concurrently via asyncio. The VLM judge scores every key against the
image and is the only quality signal we ship.

Requires `OPENROUTER_API_KEY` in the environment. OpenRouter is
OpenAI-compatible, so we point the `openai` SDK's `base_url` at it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from string import Template
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_VLM_JUDGE_TPL = Template((_PROMPT_DIR / "vlm_judge_batch.txt").read_text(encoding="utf-8"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Reasoning suppression for the VLM judge. Default model
# (`qwen/qwen3.5-35b-a3b`) requires `enable_thinking=False` to avoid
# burning the full token budget on internal thought and returning
# `finish_reason=length` with no visible output. Both layers (top-level
# `reasoning` + `chat_template_kwargs`) are sent so whichever the
# underlying provider honors gets used; the other is ignored.
_VLM_JUDGE_REASONING = {
    "reasoning": {"enabled": False},
    "chat_template_kwargs": {"enable_thinking": False},
}


# ─── per-key pre-classification ────────────────────────────────────────────


def normalize_bool(gt_val: object, pred_val: object) -> object:
    """If GT is bool and pred is a yes/no/true/false string, coerce."""
    if isinstance(gt_val, bool) and isinstance(pred_val, str):
        lower = pred_val.strip().lower()
        if lower in ("yes", "true"):
            return True
        if lower in ("no", "false"):
            return False
    return pred_val


def initialize_per_key_evals(records: list[dict[str, Any]]) -> None:
    """Build a `per_key` mapping `{key: {gt, pred}}` for the VLM judge to score."""
    for rec in records:
        per_key: dict[str, dict[str, Any]] = {}
        gt = rec["ground_truth"]
        pred = rec["prediction_json"]
        for key, gt_val in gt.items():
            pred_val = pred.get(key)
            if pred_val is not None:
                pred_val = normalize_bool(gt_val, pred_val)
            per_key[key] = {"gt": gt_val, "pred": pred_val}
        rec["per_key"] = per_key


# ─── prompt building ───────────────────────────────────────────────────────


def _vlm_attr_block(entries: list[tuple[str, str, object]]) -> str:
    lines: list[str] = []
    for key, desc, pred in entries:
        lines.append(f'- "{key}": {desc}\n  predicted: {json.dumps(pred, ensure_ascii=False)}')
    return "\n".join(lines)


def build_vlm_judge_prompt(entries: list[tuple[str, str, object]]) -> str:
    return _VLM_JUDGE_TPL.substitute(attributes=_vlm_attr_block(entries))


# ─── response parsing ──────────────────────────────────────────────────────


_NUM_RE = re.compile(r"(\d+\.?\d*)")


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def parse_score(text: str | None) -> float:
    if not text:
        return 0.0
    text = text.strip()
    try:
        return _clamp(float(text))
    except ValueError:
        m = _NUM_RE.search(text)
        if not m:
            return 0.0
        try:
            return _clamp(float(m.group(1)))
        except ValueError:
            return 0.0


def parse_batch_scores(text: str | None, expected_keys: list[str]) -> dict[str, float]:
    """Parse `{key: score}` from judge output; missing keys default to 0.0."""
    result = {k: 0.0 for k in expected_keys}
    if not text:
        return result
    # Reuse the JSON extractor from extract.py — same logic.
    from extract import extract_json_strict_first

    parsed, _ = extract_json_strict_first(text)
    if not isinstance(parsed, dict):
        return result
    for k in expected_keys:
        v = parsed.get(k)
        if isinstance(v, bool):
            # Defensive: bool is a subclass of int. Don't accept it as a score.
            continue
        if isinstance(v, (int, float)):
            result[k] = _clamp(float(v))
        elif isinstance(v, str):
            # Judge sometimes returns string-formatted numbers like "0.8";
            # parse_score extracts the leading numeric.
            result[k] = parse_score(v)
    return result


def per_sample_judge_avg(per_key: dict[str, dict[str, Any]], score_field: str) -> float | None:
    """Average a judge score across all keys of one sample. None if no scores."""
    scores = [
        float(v.get(score_field))
        for v in per_key.values()
        if isinstance(v.get(score_field), (int, float)) and not isinstance(v.get(score_field), bool)
    ]
    return sum(scores) / len(scores) if scores else None


# ─── OpenRouter client + concurrent dispatch ───────────────────────────────


def _img_to_data_url(img_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('ascii')}"


def _make_client(api_key: str | None) -> AsyncOpenAI:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys "
            "and `export OPENROUTER_API_KEY=...` before running."
        )
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=3, timeout=120.0)


async def _one_chat(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user_text: str,
    image_bytes: bytes | None,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    extra_body: dict[str, Any] | None = None,
    max_retries_on_empty: int = 3,
) -> str:
    """Single OpenRouter chat call, rate-limited by `semaphore`.

    Some OpenRouter providers occasionally return HTTP 200 with empty content
    (no model output). Treating that as success silently fails the per-key
    JSON parse — so we retry up to `max_retries_on_empty` times before
    giving up.
    """
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_bytes:
        user_content.insert(0, {"type": "image_url", "image_url": {"url": _img_to_data_url(image_bytes)}})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    async with semaphore:
        for attempt in range(max_retries_on_empty + 1):
            try:
                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if text.strip():
                    return text
                finish = getattr(resp.choices[0], "finish_reason", None)
                if attempt < max_retries_on_empty:
                    logger.warning(
                        "Empty response from %s (finish_reason=%s); retrying %d/%d.",
                        model, finish, attempt + 1, max_retries_on_empty,
                    )
                    continue
                logger.warning(
                    "Empty response from %s after %d retries (finish_reason=%s); giving up.",
                    model, max_retries_on_empty, finish,
                )
                return ""
            except Exception as e:
                logger.warning("OpenRouter call failed (%s); returning empty string.", e)
                return ""
        return ""


async def _run_concurrent(
    coros: list[Any],
    *,
    desc: str,
) -> list[str]:
    """Run coroutines concurrently with a tqdm progress bar."""
    return await tqdm_asyncio.gather(*coros, desc=desc)


# ─── VLM judge ─────────────────────────────────────────────────────────────


def run_vlm_judge(
    records: list[dict[str, Any]],
    *,
    sample_images: dict[str, bytes],
    model: str,
    max_tokens: int = 1024,
    concurrency: int = 16,
    api_key: str | None = None,
) -> None:
    """Score every key against the image via an OpenRouter VLM."""
    plans: list[dict[str, Any]] = []
    prompts: list[str] = []
    imgs: list[bytes] = []
    for rec in records:
        per_key = rec.get("per_key", {})
        if not per_key:
            continue
        keys = list(per_key.keys())
        entries = [(k, rec["schema"].get(k, ""), per_key[k].get("pred")) for k in keys]
        img = sample_images.get(rec["key"], b"")
        if not img:
            logger.warning("VLM judge: missing image for sample %s", rec["key"])
            continue
        prompts.append(build_vlm_judge_prompt(entries))
        imgs.append(img)
        plans.append({"record": rec, "keys": keys})

    if not plans:
        for rec in records:
            rec["vlm_judge_avg"] = None
        return

    logger.info("VLM judge: scoring %d sample(s) × all keys via %s.", len(plans), model)

    client = _make_client(api_key)
    sem = asyncio.Semaphore(concurrency)
    coros = [
        _one_chat(
            client,
            model=model,
            system="You are a meticulous visual evaluator.",
            user_text=p,
            image_bytes=img,
            max_tokens=max_tokens,
            semaphore=sem,
            extra_body=_VLM_JUDGE_REASONING,
        )
        for p, img in zip(prompts, imgs)
    ]
    raw_outputs = asyncio.run(_run_concurrent(coros, desc="VLM judge"))

    for plan, text in zip(plans, raw_outputs):
        scores = parse_batch_scores(text, plan["keys"])
        rec = plan["record"]
        rec["vlm_judge_raw"] = text
        for k in plan["keys"]:
            rec["per_key"][k]["vlm_score"] = scores.get(k, 0.0)

    for rec in records:
        rec["vlm_judge_avg"] = per_sample_judge_avg(rec["per_key"], "vlm_score")
