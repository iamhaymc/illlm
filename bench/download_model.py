#!/usr/bin/env python3
"""Download the LFM2.5-2.6B checkpoint from Hugging Face into ./models/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "LFM2.5-2.6B")

# Keep every byte off the system drive.  huggingface_hub resolves these into
# module constants at import time, so they have to be set before it is
# imported -- setting them afterwards is silently ignored.
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(ROOT, ".hf-cache", "hub"))
os.environ.setdefault("TMP", os.path.join(ROOT, ".hf-cache", "tmp"))
os.environ.setdefault("TEMP", os.environ["TMP"])
os.makedirs(os.environ["TMP"], exist_ok=True)

from huggingface_hub import snapshot_download  # noqa: E402  (must follow the env)


def main():
    print(f"downloading LiquidAI/LFM2.5-2.6B -> {DEST}", flush=True)
    path = snapshot_download(
        repo_id="LiquidAI/LFM2.5-2.6B",
        local_dir=DEST,
        cache_dir=os.environ["HF_HUB_CACHE"],
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
