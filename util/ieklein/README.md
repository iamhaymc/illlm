---
title: IEKLEIN
emoji: ⚡
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
license: apache-2.0
---

# IEKLEIN

Every component is a lazy property that loads on first touch and caches:

- _transformer_ — Denoise transformer, 4-bit NF4, double quant (~8 GB → ~2.5 GB)
- _text_encoder_ — Qwen3 text encoder, also 4-bit NF4
- _vae_ — Kept in bf16; quantizing the VAE visibly damages decoded output
- _scheduler_ — Assembled from the parts above on first use
- _tokenizer_ — Assembled from the parts above on first use
- _pipeline_ — Assembled from the parts above on first use

Optional flags if memory is limited:

```
--cpu-offload        (whole models offloaded between use)
--sequential-offload (layer by layer, slowest but least VRAM).
```

## API

### `KleinImageEditor`

(inherits `Flux2KleinPipeline` for `black-forest-labs/FLUX.2-klein-4B`)

##### `edit(image, prompt)`

- Handles a single image, a list for multi-reference editing or a file path;
- Recommended parameters: 4 steps, guidance 1.0.

##### `generate(prompt)`

- Plain text-to-image.
- Recommended parameters: 4 steps, guidance 1.0.

##### `preload()`

- Forces everything up front

##### `unload()`

- Frees VRAM between jobs

##### `describe()`

- Reports what's loaded.

## CLI

```bash
python ieklein.py --image input.png --prompt "make the sky sunset orange" --out edited.png
python ieklein.py --prompt "a cat holding a sign" --out created.png
python ieklein.py --info --device cpu # shows lazy state, downloads nothing
```
