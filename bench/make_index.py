#!/usr/bin/env python3
"""make_index.py -- write a model.safetensors.index.json for a checkpoint whose
published shards lack an index (LiquidAI/LFM2.5-2.6B ships none).

Sizes come from each shard's safetensors header, so `total_size` counts tensor
bytes the way Hugging Face means it, and nothing is mapped into memory.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    args = ap.parse_args()
    folder = common.require_model(args.model)

    shards = sorted(glob.glob(os.path.join(folder, "model-*.safetensors")))
    if not shards:
        lone = os.path.join(folder, "model.safetensors")
        if os.path.exists(lone):
            shards = [lone]
    if not shards:
        raise SystemExit(f"no safetensors shards under {folder}")

    weight_map, total = {}, 0
    for shard in shards:
        header = common.shard_header(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            weight_map[name] = os.path.basename(shard)
            start, end = entry["data_offsets"]
            total += end - start

    out = os.path.join(folder, "model.safetensors.index.json")
    with open(out, "w") as fh:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, fh, indent=2)
    print(f"wrote {out}: {len(weight_map)} tensors across {len(shards)} shards, "
          f"{total / (1 << 30):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
