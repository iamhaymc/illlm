#!/usr/bin/env python3
"""Shared path and checkpoint handling for the bench/ scripts.

The engine binary carries a .exe suffix on Windows and none elsewhere, which
is exactly what run.py does when it builds; resolving it here keeps every
script in this directory runnable on any host the engine itself supports.
"""

import json
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "LFM2.5-2.6B")


def binary(name="app_main"):
    """Path to a built binary, named the way run.py names it."""
    suffix = ".exe" if os.name == "nt" else ""
    return os.path.join(ROOT, "build", name + suffix)


def require_binary(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run: python3 run.py build")
    return path


def require_model(path):
    if not os.path.isdir(path):
        raise SystemExit(f"{path} not found -- run: python3 bench/download_model.py")
    return path


def vocab_size(model_dir):
    """Reads the vocabulary width from the checkpoint rather than assuming it."""
    with open(os.path.join(model_dir, "config.json")) as fh:
        return int(json.load(fh)["vocab_size"])


def shard_header(path):
    """Returns the safetensors JSON header of one shard.

    The header is a length prefixed JSON blob at the head of the file, so
    this reads a few kilobytes rather than mapping several GiB of tensors.
    """
    with open(path, "rb") as fh:
        span = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(span))


def require_torch():
    """Imports torch, pointing at run.py install when it is missing."""
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed -- run: python3 run.py install")
    return torch
