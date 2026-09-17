# Copyright 2026 the authors of this workspace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FLUX.2 [klein] 4B image editing, quantized to 4-bit, lazy-loaded.

Wraps `diffusers.Flux2KleinPipeline` for the checkpoint
`black-forest-labs/FLUX.2-klein-4B` so that one class holds the whole
editing path: an input image plus a prompt in, an edited image out. The
transformer and the text encoder are quantized to 4-bit NF4 through
bitsandbytes, which brings the checkpoint from ~13 GB VRAM down to
roughly 4-5 GB; the VAE stays in bf16 because quantizing it visibly
damages the decoded image.

By default the text encoder and its tokenizer come from
`ponpoke/flux2-klein-4b-uncensored-text-encoder`, an abliterated copy
of the official Qwen3 encoder with the refusal direction orthogonalized
out of layers 14-35; the transformer, VAE and scheduler still come from
the official checkpoint. Pass `text_encoder_id=None` (or `--official`
on the command line) to use the stock encoder instead. The uncensored
repository is gated: accept its conditions on the Hub once, then log in
with `huggingface-cli login` before the first load.

Nothing is downloaded or loaded at import time or at construction time.
The first call to `edit()` or `generate()` (or an explicit `preload()`)
fetches the checkpoint from the Hub and builds the pipeline. Each
component has its own loader, so a caller that only wants the
transformer can ask for it without paying for the rest.

The klein 4B checkpoint is step-distilled: the model card recommends
4 inference steps at guidance scale 1.0, and those are the defaults
here. Editing passes the input image (or images, for multi-reference)
through the pipeline's `image` argument.

Requires: torch, diffusers (a version carrying Flux2KleinPipeline —
`pip install git+https://github.com/huggingface/diffusers.git` if the
installed one lacks it), transformers, accelerate, bitsandbytes,
safetensors, pillow, huggingface_hub.

Run with no arguments to see the lazy-loading behaviour: it builds the
editor object, prints what is and is not loaded, and exits having
downloaded nothing.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from typing import List, Optional, Union

import torch
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# The abliterated text encoder. Its weights live in a subfolder of the
# same name inside the repository, next to the GGUF exports; the
# architecture is unchanged Qwen3, so the pipeline takes it in place of
# the official `text_encoder` without any other change.
UNCENSORED_TEXT_ENCODER_ID = "ponpoke/flux2-klein-4b-uncensored-text-encoder"
UNCENSORED_TEXT_ENCODER_SUBFOLDER = "flux2-klein-4b-uncensored-text-encoder"

# The pipeline class moved into diffusers recently; fail with the fix in
# the message rather than an ImportError pointing at a missing symbol.
try:
    from diffusers import Flux2KleinPipeline
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "This script needs a diffusers version that ships Flux2KleinPipeline. "
        "Install or upgrade with: pip install -U git+https://github.com/huggingface/diffusers.git"
    ) from exc

try:
    from diffusers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - older diffusers
    from transformers import BitsAndBytesConfig


class KleinImageEditor:
    """The whole FLUX.2 [klein] 4B editing pipeline behind one lazy object.

    Every model is loaded on first use and cached. `edit()` and
    `generate()` are the two entry points; `preload()` forces everything
    up front for callers that would rather pay the cost once.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantize: bool = True,
        cpu_offload: bool = False,
        sequential_offload: bool = False,
        text_encoder_id: Optional[str] = UNCENSORED_TEXT_ENCODER_ID,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; pass device='cpu' (slow) or fix the driver."
            )
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantize = quantize
        self.cpu_offload = cpu_offload
        self.sequential_offload = sequential_offload
        # Where the text encoder and tokenizer come from. None means the
        # official checkpoint's own `text_encoder` subfolder; anything
        # else is `repo[:subfolder]`, defaulting to the abliterated copy.
        self.text_encoder_id = text_encoder_id

        # Lazy resources: each is None until its loader has run.
        self._quant_config: Optional[BitsAndBytesConfig] = None
        self._transformer = None
        self._text_encoder = None
        self._vae = None
        self._scheduler = None
        self._tokenizer = None
        self._pipe: Optional[Flux2KleinPipeline] = None

    # ------------------------------------------------------------------
    # Lazy loaders. Each one is idempotent and downloads only what it needs.
    # ------------------------------------------------------------------

    @property
    def quant_config(self) -> Optional[BitsAndBytesConfig]:
        """The 4-bit NF4 config, built once on first request."""
        if not self.quantize:
            return None
        if self._quant_config is None:
            self._quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        return self._quant_config

    @property
    def transformer(self):
        """The denoising transformer, 4-bit quantized, loaded on first use."""
        if self._transformer is None:
            from diffusers import Flux2Transformer2DModel

            print(f"[lazy] loading transformer from {self.model_id} (4-bit NF4)...")
            self._transformer = Flux2Transformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=self.dtype,
                quantization_config=self.quant_config,
            )
        return self._transformer

    def _text_encoder_source(self) -> tuple[str, str]:
        """(repo, subfolder) the text encoder and tokenizer load from."""
        if self.text_encoder_id is None:
            return self.model_id, "text_encoder"
        if ":" in self.text_encoder_id:
            repo, subfolder = self.text_encoder_id.split(":", 1)
            return repo, subfolder
        return self.text_encoder_id, UNCENSORED_TEXT_ENCODER_SUBFOLDER

    @property
    def text_encoder(self):
        """The Qwen3 text encoder, 4-bit quantized, loaded on first use.

        Defaults to the abliterated copy; the refusal vectors are
        orthogonalized out of the weights, the architecture is untouched,
        so it loads exactly like the official one.
        """
        if self._text_encoder is None:
            from transformers import Qwen3ForCausalLM

            repo, subfolder = self._text_encoder_source()
            print(f"[lazy] loading text encoder from {repo}/{subfolder} (4-bit NF4)...")
            self._text_encoder = Qwen3ForCausalLM.from_pretrained(
                repo,
                subfolder=subfolder,
                torch_dtype=self.dtype,
                quantization_config=self.quant_config,
            )
        return self._text_encoder

    @property
    def vae(self):
        """The Flux2 VAE, kept in bf16 on purpose: quantizing it costs image quality."""
        if self._vae is None:
            from diffusers import AutoencoderKLFlux2

            print(f"[lazy] loading VAE from {self.model_id} (bf16, not quantized)...")
            self._vae = AutoencoderKLFlux2.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=self.dtype,
            )
        return self._vae

    @property
    def scheduler(self):
        if self._scheduler is None:
            from diffusers import FlowMatchEulerDiscreteScheduler

            print(f"[lazy] loading scheduler from {self.model_id}...")
            self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
            )
        return self._scheduler

    @property
    def tokenizer(self):
        """The tokenizer, taken from the same source as the text encoder."""
        if self._tokenizer is None:
            from transformers import Qwen2Tokenizer

            repo, subfolder = self._text_encoder_source()
            print(f"[lazy] loading tokenizer from {repo}/{subfolder}...")
            self._tokenizer = Qwen2Tokenizer.from_pretrained(
                repo,
                subfolder=subfolder,
            )
        return self._tokenizer

    @property
    def pipe(self) -> Flux2KleinPipeline:
        """The assembled pipeline, built on first use from the parts above."""
        if self._pipe is None:
            print(f"[lazy] assembling pipeline {self.model_id}...")
            self._pipe = Flux2KleinPipeline.from_pretrained(
                self.model_id,
                transformer=self.transformer,
                text_encoder=self.text_encoder,
                vae=self.vae,
                scheduler=self.scheduler,
                tokenizer=self.tokenizer,
                torch_dtype=self.dtype,
            )
            if self.sequential_offload:
                self._pipe.enable_sequential_cpu_offload()
            elif self.cpu_offload:
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe.to(self.device)
            # Both savers cost nothing at quality and help small cards.
            self._pipe.enable_vae_slicing()
            self._pipe.enable_vae_tiling()
            print("[lazy] pipeline ready.")
        return self._pipe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Force every lazy resource to load now, so the first edit is fast."""
        _ = self.pipe

    @torch.no_grad()
    def edit(
        self,
        image: Union[Image.Image, List[Image.Image], str],
        prompt: str,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """Edit one image (or several, as multi-reference) with a prompt.

        `image` may be a PIL image, a list of them for multi-reference
        editing, or a path that is opened for you. Returns the edited
        PIL image.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = [image]
        image = [img.convert("RGB") for img in image]

        generator = self._make_generator(seed)
        result = self.pipe(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return result.images[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """Text-to-image with the same checkpoint, no reference image."""
        generator = self._make_generator(seed)
        result = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return result.images[0]

    def unload(self) -> None:
        """Drop every loaded model and free what VRAM can be freed."""
        self._pipe = None
        self._transformer = None
        self._text_encoder = None
        self._vae = None
        self._scheduler = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------

    def _make_generator(self, seed: Optional[int]) -> torch.Generator:
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        device = "cuda" if self.device.startswith("cuda") else "cpu"
        return torch.Generator(device=device).manual_seed(seed)

    def describe(self) -> str:
        """What is loaded right now, for the curious and for the CLI."""
        repo, subfolder = self._text_encoder_source()
        parts = {
            "transformer": self._transformer is not None,
            "text_encoder": self._text_encoder is not None,
            "vae": self._vae is not None,
            "scheduler": self._scheduler is not None,
            "tokenizer": self._tokenizer is not None,
            "pipeline": self._pipe is not None,
        }
        loaded = ", ".join(name for name, ok in parts.items() if ok) or "nothing"
        return (
            f"quantize={self.quantize}, device={self.device}, "
            f"text_encoder={repo}/{subfolder}, loaded: {loaded}"
        )


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Edit an image with FLUX.2 [klein] 4B, 4-bit quantized and lazy-loaded."
    )
    parser.add_argument("--image", help="input image to edit; omit for text-to-image")
    parser.add_argument(
        "--prompt", default=None, help="what to do to the image, or what to draw"
    )
    parser.add_argument("--out", default="flux_klein_out.png", help="output path")
    parser.add_argument(
        "--steps", type=int, default=4, help="inference steps (distilled model: 4)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="guidance scale (distilled model: 1.0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="output height; default follows the input",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="output width; default follows the input",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed; random when omitted"
    )
    parser.add_argument(
        "--no-quant", action="store_true", help="load in bf16 instead of 4-bit"
    )
    parser.add_argument(
        "--device", default="cuda", help="torch device; 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="offload whole models to CPU between use",
    )
    parser.add_argument(
        "--sequential-offload",
        action="store_true",
        help="offload layer by layer (slowest, least VRAM)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help=(
            "use the official text encoder from the base checkpoint "
            "instead of the abliterated one"
        ),
    )
    parser.add_argument(
        "--text-encoder",
        default=None,
        metavar="REPO[:SUBFOLDER]",
        help=(
            "text encoder repository to load from; default is the "
            f"abliterated {UNCENSORED_TEXT_ENCODER_ID}"
        ),
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="print what is loaded and exit without generating",
    )
    args = parser.parse_args(argv)

    if args.official and args.text_encoder:
        parser.error("--official and --text-encoder are mutually exclusive")
    text_encoder_id = None if args.official else (
        args.text_encoder or UNCENSORED_TEXT_ENCODER_ID
    )

    editor = KleinImageEditor(
        device=args.device,
        quantize=not args.no_quant,
        cpu_offload=args.cpu_offload,
        sequential_offload=args.sequential_offload,
        text_encoder_id=text_encoder_id,
    )

    if args.info:
        print(editor.describe())
        print("nothing was downloaded; resources load on first edit/generate/preload")
        return 0

    if not args.prompt:
        parser.error("--prompt is required unless --info is given")

    if args.image:
        out = editor.edit(
            args.image,
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
    else:
        out = editor.generate(
            args.prompt,
            height=args.height or 1024,
            width=args.width or 1024,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    out.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
