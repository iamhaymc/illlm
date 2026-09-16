#!/usr/bin/env python
"""Eval pipeline driver: extraction → structural metrics → VLM judge → JSON.

Extraction runs locally on your GPU (vLLM/HF); the VLM judge runs remotely
via the OpenRouter API. One process, sequential stages, one JSON file out.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from extract import iter_eval_samples, run_extraction
from judge import initialize_per_key_evals, run_vlm_judge


# ─── metrics aggregation ───────────────────────────────────────────────────


def per_sample_structural(prediction_json: dict, ground_truth: dict, strict_valid: bool) -> dict[str, Any]:
    pred_keys = set(prediction_json.keys())
    gt_keys = set(ground_truth.keys())
    overlap = pred_keys & gt_keys
    p = len(overlap) / len(pred_keys) if pred_keys else 0.0
    r = len(overlap) / len(gt_keys) if gt_keys else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "json_valid": strict_valid,
        "total_keys": len(gt_keys),
        "total_pred_keys": len(pred_keys),
        "overlap_keys": len(overlap),
        "key_precision": p,
        "key_recall": r,
        "key_f1": f1,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"samples_evaluated": 0}

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    json_valid = sum(1 for r in records if r.get("json_valid"))
    vlm_scores = [r["vlm_judge_avg"] for r in records if r.get("vlm_judge_avg") is not None]

    return {
        "json_validity_rate": json_valid / n,
        "key_precision_macro": mean([r.get("key_precision", 0.0) for r in records]),
        "key_recall_macro": mean([r.get("key_recall", 0.0) for r in records]),
        "key_f1_macro": mean([r.get("key_f1", 0.0) for r in records]),
        "vlm_judge_score_avg": mean(vlm_scores) if vlm_scores else None,
        "samples_evaluated": n,
    }


def _strip_sample(rec: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy/internal fields before serialising to JSON."""
    return {
        "key": rec["key"],
        "schema": rec["schema"],
        "ground_truth": rec["ground_truth"],
        "prediction_raw": rec["prediction_raw"],
        "prediction_json": rec["prediction_json"],
        "json_valid": rec.get("json_valid", False),
        "total_keys": rec.get("total_keys", 0),
        "total_pred_keys": rec.get("total_pred_keys", 0),
        "key_precision": rec.get("key_precision", 0.0),
        "key_recall": rec.get("key_recall", 0.0),
        "key_f1": rec.get("key_f1", 0.0),
        "vlm_judge_avg": rec.get("vlm_judge_avg"),
        "vlm_judge_raw": rec.get("vlm_judge_raw"),
        "per_key": rec.get("per_key", {}),
    }


# ─── CLI ───────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="OpenRouter-judged structured-extraction eval.")
    p.add_argument("--checkpoint-path", required=True, help="HF id or local merged/LoRA dir.")
    p.add_argument("--data-path", default="./eval_data", help="WDS tar / dir / glob.")
    p.add_argument("--output-path", default="./eval_result.json")
    p.add_argument("--num-samples", type=int, default=0, help="Cap N samples (0 = all).")
    p.add_argument("--skip-samples", type=int, default=0)
    p.add_argument("--extraction-backend", choices=["auto", "vllm", "hf"], default="auto")
    p.add_argument("--extraction-batch", type=int, default=8)
    p.add_argument("--extraction-max-new-tokens", type=int, default=1024)
    p.add_argument("--extraction-gpu-mem-util", type=float, default=0.85)
    p.add_argument("--extraction-max-model-len", type=int, default=8192)
    p.add_argument("--vlm-judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vlm-judge-model", default="qwen/qwen3-vl-4b-instruct")
    p.add_argument("--vlm-judge-max-tokens", type=int, default=1024)
    p.add_argument("--judge-concurrency", type=int, default=16, help="Concurrent OpenRouter calls.")
    p.add_argument("--openrouter-api-key", default=None, help="Override $OPENROUTER_API_KEY.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.perf_counter()
    logger = logging.getLogger("run_eval")
    logger.info("=== OpenRouter-judged eval starting ===")

    # ── load samples ─────────────────────────────────────────────────────
    samples = list(
        iter_eval_samples(
            args.data_path,
            skip=args.skip_samples,
            limit=args.num_samples,
        )
    )
    if not samples:
        raise RuntimeError(
            f"No usable samples loaded from {args.data_path} — expected WDS tars "
            "with .jpg, .key_explanations, .structured_text per sample."
        )
    logger.info("Loaded %d sample(s).", len(samples))
    sample_images = {s.key: s.image_bytes for s in samples}

    # ── extraction ───────────────────────────────────────────────────────
    records = run_extraction(
        samples,
        model_path=args.checkpoint_path,
        backend=args.extraction_backend,
        max_new_tokens=args.extraction_max_new_tokens,
        max_model_len=args.extraction_max_model_len,
        gpu_mem_util=args.extraction_gpu_mem_util,
        batch=args.extraction_batch,
    )

    # ── structural metrics ───────────────────────────────────────────────
    for rec in records:
        rec.update(
            per_sample_structural(
                rec["prediction_json"],
                rec["ground_truth"],
                rec.get("prediction_strict_valid", bool(rec["prediction_json"])),
            )
        )

    initialize_per_key_evals(records)
    judge_errors: dict[str, str] = {}

    # ── VLM judge ────────────────────────────────────────────────────────
    if args.vlm_judge:
        try:
            run_vlm_judge(
                records,
                sample_images=sample_images,
                model=args.vlm_judge_model,
                max_tokens=args.vlm_judge_max_tokens,
                concurrency=args.judge_concurrency,
                api_key=args.openrouter_api_key,
            )
        except Exception as e:
            judge_errors["vlm_judge"] = repr(e)
            logger.warning("VLM judge failed (%s); continuing without VLM scores.", e)
            for rec in records:
                rec.setdefault("vlm_judge_avg", None)
    else:
        for rec in records:
            rec["vlm_judge_avg"] = None

    # ── write output ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    result = {
        "metadata": {
            "checkpoint_path": args.checkpoint_path,
            "data_path": args.data_path,
            "num_samples_evaluated": len(records),
            "extraction_backend": args.extraction_backend,
            "vlm_judge_model": args.vlm_judge_model if args.vlm_judge else None,
            "judge_errors": judge_errors or None,
            "elapsed_s": round(elapsed, 2),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "metrics": aggregate(records),
        "samples": [_strip_sample(rec) for rec in records],
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== JUDGING SUMMARY ===")
    print(f"output={out}")
    for k, v in result["metrics"].items():
        print(f"  {k}={v:.4f}" if isinstance(v, float) else f"  {k}={v}")
    print(f"  elapsed_s={elapsed:.1f}")
    print("=== JUDGING OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
