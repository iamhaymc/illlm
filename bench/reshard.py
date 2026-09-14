#!/usr/bin/env python3
"""reshard.py -- split oversized safetensors shards, byte exact.

GitHub caps a single LFS object at two gigabytes, and LFM2.5-2.6B's first
shard is 5.3 GB of it. This re-shards a checkpoint into the ordinary Hugging
Face layout -- model-0000N-of-0000M.safetensors plus the index that names
them -- moving no value and round-tripping no dtype: tensor bytes are copied
verbatim and only the offsets in each header are rebased to the new shard.

The result loads to the same weight total and reads back to the same bytes
per tensor as the original, which --verify checks rather than promises.
"""
import argparse
import glob
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# GitHub's LFS object cap is 2 GB; leave room for the header and the index.
TARGET_BYTES = 1750 * 1024 * 1024


def read_header(path):
    """Returns (header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as fh:
        span = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(span)), 8 + span


def tensor_names(header):
    return [k for k in header if k != "__metadata__"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--target", type=int, default=TARGET_BYTES,
                    help="max tensor bytes per output shard")
    ap.add_argument("--dest", default=None,
                    help="output folder (default: model/ at the repo root)")
    ap.add_argument("--verify", action="store_true",
                    help="hash every tensor in both copies and compare")
    args = ap.parse_args()

    folder = common.require_model(args.model)
    dest = args.dest or os.path.join(common.ROOT, "model")
    os.makedirs(dest, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(folder, "model-*.safetensors")))
    if not sources:
        lone = os.path.join(folder, "model.safetensors")
        if os.path.exists(lone):
            sources = [lone]
    if not sources:
        raise SystemExit(f"no safetensors shards under {folder}")

    # One pass over every source shard: which tensors, in what order, at what
    # offsets. The output keeps that global order, so the load order upstream
    # implies is preserved.
    jobs = []  # (name, entry, source path, source data start)
    for path in sources:
        header, data_start = read_header(path)
        for name in tensor_names(header):
            jobs.append((name, header[name], path, data_start))

    # Group into output shards under the target. A single tensor larger than
    # the target still gets its own shard -- the cap applies to the file, not
    # to the tensor, and splitting one is not possible without rewriting bytes.
    outputs, current, size = [], [], 0
    for name, entry, path, data_start in jobs:
        start, end = entry["data_offsets"]
        span = end - start
        if current and size + span > args.target:
            outputs.append(current)
            current, size = [], 0
        current.append((name, entry, path, data_start))
        size += span
    if current:
        outputs.append(current)
    count = len(outputs)

    weight_map, total = {}, 0
    for index, group in enumerate(outputs, 1):
        out_path = os.path.join(dest, f"model-{index:05d}-of-{count:05d}.safetensors")

        # Offsets are relative to the data region, which starts at zero in
        # every shard. A group can span two source shards, so the offsets are
        # renumbered against the group itself: each tensor lands at the sum
        # of the spans before it, and the data region is exactly the tensors'
        # bytes back to back. The bytes themselves are copied verbatim; only
        # the numbers in the header change.
        entries, pos = {}, 0
        for name, entry, _, _ in group:
            start, end = entry["data_offsets"]
            e = dict(entry)
            e["data_offsets"] = [pos, pos + (end - start)]
            pos += end - start
            entries[name] = e
        blob = json.dumps({"__metadata__": {"format": "pt"}, **entries},
                          separators=(",", ":")).encode()
        # The spec pads the header to a multiple of eight with spaces.
        padded = blob + b" " * (-len(blob) % 8)

        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(padded)))
            out.write(padded)
            for name, entry, path, data_start in group:
                start, end = entry["data_offsets"]
                with open(path, "rb") as src:
                    src.seek(data_start + start)
                    left = end - start
                    while left > 0:
                        chunk = src.read(min(left, 1 << 24))
                        if not chunk:
                            raise SystemExit(f"short read in {path}")
                        out.write(chunk)
                        left -= len(chunk)
                weight_map[name] = os.path.basename(out_path)
                total += end - start
        span = sum(e["data_offsets"][1] - e["data_offsets"][0]
                   for _, e, _, _ in group)
        print(f"wrote {out_path}: {len(group)} tensors, {span / (1 << 20):.1f} MiB")

    with open(os.path.join(dest, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, fh, indent=2)
    print(f"wrote index: {len(weight_map)} tensors, {total / (1 << 30):.2f} GiB, {count} shards")

    if args.verify:
        verify(jobs, dest)
    return 0


def tensor_digest(path, data_start, entry):
    """SHA-256 over one tensor's raw bytes, and its span."""
    start, end = entry["data_offsets"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        left = end - start
        while left > 0:
            chunk = fh.read(min(left, 1 << 24))
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest(), end - start


def verify(jobs, dest):
    """Hashes every tensor in the source and in the re-sharded copy.

    A copy is only vendored if this passes: the whole point of the split is
    that it changed nothing, and this is what checks that rather than the
    header arithmetic promising it.
    """
    index = json.load(open(os.path.join(dest, "model.safetensors.index.json")))
    shard_header = {}
    bad = 0
    for name, entry, path, data_start in jobs:
        shard = index["weight_map"].get(name)
        if shard is None:
            print(f"MISSING from copy: {name}")
            bad += 1
            continue
        out_path = os.path.join(dest, shard)
        if shard not in shard_header:
            shard_header[shard] = read_header(out_path)
        out_header, out_data_start = shard_header[shard]
        a, span_a = tensor_digest(path, data_start, entry)
        b, span_b = tensor_digest(out_path, out_data_start, out_header[name])
        if a != b or span_a != span_b:
            print(f"MISMATCH: {name}")
            bad += 1
    print(f"verified {len(jobs) - bad}/{len(jobs)} tensors byte identical")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
