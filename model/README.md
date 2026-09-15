# model

`LiquidAI/LFM2.5-2.6B`, vendored so the engine can be built and checked on a
machine that cannot reach Hugging Face.

    python3 run.py run -- generate --model model --prompt "Write a haiku about rivers."
    python3 run.py run -- chat --model model

## What is here

The upstream repository ships its weights as two shards, the first 5.3 GB.
That is over GitHub's two gigabyte limit for a single LFS object, so they are
re-sharded into three under the cap, with the index that names them — the
ordinary Hugging Face layout, which both this engine and `transformers` read
without being told. The re-shard is byte exact: every tensor's bytes were
copied verbatim and only the offsets in each header were renumbered, so no
value moved and no dtype was round-tripped. All 266 tensors were hashed in
both copies and confirmed to agree.

The upstream shards also ship no `model.safetensors.index.json` — the loader
in this repository does not need one, but `transformers` does, so it was
generated from the safetensors headers and committed here.

| file | what it holds |
| ---- | ------------- |
| `model-0000N-of-00003.safetensors` | the weights, 266 tensors over three shards |
| `model.safetensors.index.json` | which shard holds which tensor |
| `config.json` | 30 blocks — 22 short convolutions and 8 attentions over 32 query and 8 key-value heads |
| `tokenizer.json` | the vocabulary, 128000 entries |
| `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` | as upstream |

The shards and `tokenizer.json` are Git LFS objects; see `.gitattributes` at
the repository root. A clone without LFS installed will find text pointers
here rather than weights — a few hundred bytes of
`version https://git-lfs.github.com/spec/v1` where a checkpoint should be.
That needs the `git-lfs` program itself, which is a separate package on most
distributions (`apt install git-lfs`, `brew install git-lfs`) and not part of
git. With it present, run `git lfs install` before cloning, or `git lfs pull`
in an existing clone.

## Licence

The weights are LiquidAI's, released under the
[LiquidAI open licence](https://huggingface.co/LiquidAI/LFM2.5-2.6B) that
repository is published under. They are redistributed here unmodified apart
from the re-sharding described above. The engine in this repository is a
separate work and carries its own terms.
