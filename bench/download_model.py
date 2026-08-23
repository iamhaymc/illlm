#!/usr/bin/env python3
"""Download the LFM2.5-2.6B checkpoint from Hugging Face into ./models/."""
import os
import sys

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "LFM2.5-2.6B")

# Keep every byte off the full C: drive.
os.environ["HF_HOME"] = os.path.join(ROOT, ".hf-cache")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT, ".hf-cache", "hub")
os.environ["TMP"] = os.path.join(ROOT, ".hf-cache", "tmp")
os.environ["TEMP"] = os.environ["TMP"]
os.makedirs(os.environ["TMP"], exist_ok=True)

print(f"downloading LiquidAI/LFM2.5-2.6B -> {DEST}", flush=True)
path = snapshot_download(
    repo_id="LiquidAI/LFM2.5-2.6B",
    local_dir=DEST,
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
    ],
)
print(f"done: {path}", flush=True)