#!/bin/bash
# Example evaluation script — extraction on local GPU, judges via OpenRouter API.
#   bash run_eval.sh

set -euo pipefail

# --- vLLM env workarounds ----------------------------------------------------
# Needed when EXTRACTION_BACKEND=vllm or "auto". No-op on systems without
# vLLM / environment-modules.
#   1. CUDA toolkit on LD_LIBRARY_PATH so flashinfer's GDN/Mamba kernels can
#      dlopen libcudart.so.12 (LFM2.5-VL is a hybrid arch and uses these).
#   2. Skip flashinfer's sampling kernel — flashinfer 0.6 + CUDA 12.9 trigger
#      an NVCC stub bug (`__cudaLaunch` not declared). PyTorch-native sampler
#      is ~20% slower but works.
#   3. Skip vLLM 0.21's DeepGEMM autotune warmup (~18 min for MoE/FP8 models).
module load cuda12.9/toolkit/12.9.1 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# --- OpenRouter API key ------------------------------------------------------
# Required. Get one at https://openrouter.ai/keys, then either:
#   export OPENROUTER_API_KEY=...
# in your shell, OR uncomment and set it here.
#OPENROUTER_API_KEY="sk-or-v1-..."

# --- Checkpoint --------------------------------------------------------------
# HF id of the trained model, OR a local merged/LoRA checkpoint dir.
CHECKPOINT="LiquidAI/LFM2.5-VL-450M-Extract"

# --- Eval data ---------------------------------------------------------------
# WDS tar / dir of tars / brace-glob.
DATA_PATH="./eval_data"

# Output JSON path.
OUTPUT="./eval_result.json"

# --- Sample count ------------------------------------------------------------
# Number of samples to evaluate. Default 2000 runs the full shipped eval_data
# (~30 min). Set to 50 for a quick smoke test (~5 min).
NUM_SAMPLES=2000

# --- Extraction (local GPU) --------------------------------------------------
# "auto" tries vLLM first, falls back to HF transformers on init failure.
EXTRACTION_BACKEND="auto"
EXTRACTION_BATCH=8

# --- Judge model (OpenRouter) ------------------------------------------------
# Any image-capable OpenRouter model id works. Pricing:
# https://openrouter.ai/models
VLM_JUDGE_MODEL="qwen/qwen3.5-35b-a3b"

# Concurrent OpenRouter calls. Lower if you hit rate limits.
JUDGE_CONCURRENCY=16

# --- Run ---------------------------------------------------------------------
LOG_FILE="${LOG_FILE:-./eval_run.log}"
echo "Logging to: ${LOG_FILE}"
python run_eval.py \
  --checkpoint-path "${CHECKPOINT}" \
  --data-path "${DATA_PATH}" \
  --output-path "${OUTPUT}" \
  --num-samples "${NUM_SAMPLES}" \
  --extraction-backend "${EXTRACTION_BACKEND}" \
  --extraction-batch "${EXTRACTION_BATCH}" \
  --vlm-judge --vlm-judge-model "${VLM_JUDGE_MODEL}" \
  --judge-concurrency "${JUDGE_CONCURRENCY}" \
  2>&1 | tee "${LOG_FILE}"
