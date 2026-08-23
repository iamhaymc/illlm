#!/usr/bin/env python3
"""make_index.py -- write a model.safetensors.index.json for a checkpoint whose
published shards lack an index (LiquidAI/LFM2.5-2.6B ships none)."""
import glob
import json
import os

import safetensors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "models", "LFM2.5-2.6B")

shards = sorted(glob.glob(os.path.join(DIR, "model-*.safetensors")))
assert len(shards) >= 2, shards
total = {}
for shard in shards:
    with safetensors.safe_open(shard, framework="pt", device="cpu") as fh:
        for name in fh.keys():
            total[name] = os.path.basename(shard)

index = {"metadata": {"total_size": sum(
    os.path.getsize(s) for s in shards)}, "weight_map": total}
out = os.path.join(DIR, "model.safetensors.index.json")
with open(out, "w") as fh:
    json.dump(index, fh, indent=2)
print(f"wrote {out}: {len(total)} tensors across {len(shards)} shards")