# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import torch
>>> from diffusers import QwenImageEditPlusPipeline
>>> from diffusers.utils import load_image

>>> pipe = QwenImageEditPlusPipeline.from_pretrained(
...     "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
... ).to("cuda")

>>> image = load_image(
... "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
... ).convert("RGB")

>>> prompt = "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"

>>> out = pipe(image=image, prompt=prompt, num_inference_steps=50).images[0]
>>> out.save("qwenimage_edit_plus.png")
```
"""

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def pad_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Pad (letterbox) to target aspect ratio without warping."""
    return ImageOps.pad(
        img.convert("RGB"),
        (int(target_w), int(target_h)),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def choose_condition_area(canvas_area: int, base_area: int = CONDITION_IMAGE_SIZE) -> int:
    """Choose a conditioning target area derived from canvas area with sensible bounds."""
    scaled = int(canvas_area * (base_area / (1024 * 1024)))
    return int(min(base_area, max(256 * 256, scaled)))


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area: int, ratio: float, multiple: int = 32):
    """
    Area-based sizing while snapping to a chosen lattice multiple.
    Used for canvas sizing AND conditioning sizing (anti-drift).
    """
    m = int(multiple) if multiple else 32
    m = max(1, m)

    width = math.sqrt(float(target_area) * float(ratio))
    height = width / float(ratio)

    width = round(width / m) * m
    height = round(height / m) * m
    return int(width), int(height)


# Optional: decoder VAE (Wan2x)
_ALT_VAE_WAN2X = None

# Track desired tiling state for the optional decoder VAE, so it stays consistent across lazy loads.
_ALT_VAE_WAN2X_TILING_ENABLED = False


def _set_vae_tiling(model: Any, enabled: bool) -> bool:
    """
    Best-effort tiling toggle for a VAE-like module.
    Returns True if a tiling method existed and was called, False otherwise.
    """
    if model is None:
        return False
    try:
        if enabled:
            if hasattr(model, "enable_tiling"):
                model.enable_tiling()
                return True
            if hasattr(model, "enable_vae_tiling"):
                model.enable_vae_tiling()
                return True
        else:
            if hasattr(model, "disable_tiling"):
                model.disable_tiling()
                return True
            if hasattr(model, "disable_vae_tiling"):
                model.disable_vae_tiling()
                return True
    except Exception as e:
        # Don't hard-fail inference if tiling toggle fails for an alt decoder.
        logger.warning(f"VAE tiling toggle failed on {type(model)}: {e}")
        return False
    return False



def _get_wan2x_vae(device: torch.device, dtype: torch.dtype):
    """
    Decoder-only finetune that outputs 2x resolution via pixel-shuffle.
    Lazy-loaded so it doesn't impact startup unless used.
    """
    global _ALT_VAE_WAN2X, _ALT_VAE_WAN2X_TILING_ENABLED
    if _ALT_VAE_WAN2X is None:
        from diffusers import AutoencoderKLWan

        _ALT_VAE_WAN2X = AutoencoderKLWan.from_pretrained(
            "spacepxl/Wan2.1-VAE-upscale2x",
            subfolder="diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1",
            torch_dtype=dtype,
        )
        _ALT_VAE_WAN2X.eval()

        # Apply last requested tiling immediately on first load (if supported).
        _set_vae_tiling(_ALT_VAE_WAN2X, _ALT_VAE_WAN2X_TILING_ENABLED)

    _ALT_VAE_WAN2X = _ALT_VAE_WAN2X.to(device=device, dtype=dtype)

    # Re-apply after moving to device, just in case.
    _set_vae_tiling(_ALT_VAE_WAN2X, _ALT_VAE_WAN2X_TILING_ENABLED)

    return _ALT_VAE_WAN2X


class QwenImageEditPlusPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    The Qwen-Image-Edit pipeline for image editing.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16

        # QwenImage latents are turned into 2x2 patches and packed; multiply scale-factor by patch size
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024


        # Track tiling state (applies to both primary VAE and optional decoder VAE)
        self._vae_tiling_enabled = False
        self.prompt_template_encode = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image.\n"
            "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128


        # ------------------------------------------------------------
        # VAE tiling control (applies to both primary VAE and optional decoder VAE)
        # ------------------------------------------------------------
        # Expose a stable API so app.py can call pipe.enable_vae_tiling()/disable_vae_tiling()
        # regardless of which decoder VAE is selected at runtime.
        
    def set_vae_tiling(self, enabled: bool) -> None:
        global _ALT_VAE_WAN2X_TILING_ENABLED, _ALT_VAE_WAN2X

        enabled = bool(enabled)
        self._vae_tiling_enabled = enabled

        # 1) Primary VAE (Qwen)
        _set_vae_tiling(getattr(self, "vae", None), enabled)

        # 2) Optional decoder VAE (Wan2x): store desired global state; apply now if already loaded.
        _ALT_VAE_WAN2X_TILING_ENABLED = enabled
        if _ALT_VAE_WAN2X is not None:
            _set_vae_tiling(_ALT_VAE_WAN2X, enabled)

    def enable_vae_tiling(self) -> None:
        self.set_vae_tiling(True)

    def disable_vae_tiling(self) -> None:
        self.set_vae_tiling(False)
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

        if isinstance(image, list):
            base_img_prompt = ""
            for i, _ in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.processor(text=txt, images=image, padding=True, return_tensors="pt").to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])

        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. "
                "Dimensions will be resized accordingly."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` must also be passed.")

        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` must also be passed.")

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, height, width)
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.latent_channels, 1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.latent_channels, 1, 1, 1).to(
            image_latents.device, image_latents.dtype
        )
        image_latents = (image_latents - latents_mean) / latents_std
        return image_latents

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []

            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image

                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)

            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        condition_area: Optional[int] = None,
        vae_image_indices: Optional[List[int]] = None,
        pad_to_canvas: bool = True,
        # NEW: lattice + VAE ref override
        resolution_multiple: Optional[int] = None,
        vae_ref_area: Optional[int] = None,
        vae_ref_start_index: int = 2,
        # Optional: decoder swap
        decoder_vae: str = "qwen",  # "qwen" | "wan2x"
        keep_decoder_2x: bool = False,
        # standard args
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """Run Qwen-Image-Edit inference.

        Examples:
        """
        # ---- determine input size ----
        if isinstance(image, list):
            image_size = image[0].size
        else:
            image_size = image.size

        # Lattice multiple used throughout (canvas sizing + condition sizing)
        multiple_of = int(resolution_multiple) if resolution_multiple is not None else (self.vae_scale_factor * 2)
        multiple_of = max(1, multiple_of)

        calculated_width, calculated_height = calculate_dimensions(
            1024 * 1024, float(image_size[0]) / float(image_size[1]), multiple=multiple_of
        )
        height = height or calculated_height
        width = width or calculated_width

        width = (int(width) // multiple_of) * multiple_of
        height = (int(height) // multiple_of) * multiple_of

        # ---- validate ----
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # ---- call params ----
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # ---- preprocess ----
        condition_images = None
        vae_images = None
        vae_image_sizes: List[tuple[int, int]] = []

        # support pre-latent tensors (rare, but keep compatibility)
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]

            canvas_area = int(width) * int(height)
            cond_area = int(condition_area) if condition_area is not None else choose_condition_area(canvas_area)

            cond_w, cond_h = calculate_dimensions(cond_area, float(width) / float(height), multiple=multiple_of)

            # Optional VAE ref override sizing (applied only to indices >= vae_ref_start_index)
            ref_w = ref_h = None
            if vae_ref_area is not None:
                try:
                    ref_w, ref_h = calculate_dimensions(
                        int(vae_ref_area),
                        float(width) / float(height),
                        multiple=multiple_of,
                    )
                except Exception:
                    ref_w = ref_h = None

            condition_images = []
            vae_images = []

            if vae_image_indices is None:
                vae_image_indices = list(range(len(image)))
            vae_set = set(int(i) for i in vae_image_indices)

            for idx, img in enumerate(image):
                pil = img.convert("RGB") if isinstance(img, Image.Image) else img

                if pad_to_canvas and isinstance(pil, Image.Image):
                    pil = pad_to_aspect(pil, int(width), int(height))

                # conditioning stream (always)
                condition_images.append(self.image_processor.resize(pil, cond_h, cond_w))

                # VAE stream (selective)
                if idx in vae_set:
                    if (ref_w is not None) and (ref_h is not None) and (int(idx) >= int(vae_ref_start_index)):
                        vw, vh = int(ref_w), int(ref_h)
                    else:
                        vw, vh = int(width), int(height)

                    vae_image_sizes.append((vw, vh))
                    vae_images.append(self.image_processor.preprocess(pil, int(vh), int(vw)).unsqueeze(2))

            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
            )
            if true_cfg_scale > 1 and not has_neg_prompt:
                logger.warning(
                    f"true_cfg_scale={true_cfg_scale} but CFG disabled because no negative prompt was provided."
                )
            if true_cfg_scale <= 1 and has_neg_prompt:
                logger.warning("negative_prompt provided but CFG disabled because true_cfg_scale <= 1")

            do_true_cfg = (true_cfg_scale > 1) and has_neg_prompt

            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            if do_true_cfg:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    image=condition_images,
                    prompt=negative_prompt,
                    prompt_embeds=negative_prompt_embeds,
                    prompt_embeds_mask=negative_prompt_embeds_mask,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )

            # ---- prepare latents ----
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, image_latents = self.prepare_latents(
                vae_images,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            img_shapes = [
                [
                    (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                    *[
                        (1, vae_h // self.vae_scale_factor // 2, vae_w // self.vae_scale_factor // 2)
                        for (vae_w, vae_h) in vae_image_sizes
                    ],
                ]
            ] * batch_size

        else:
            raise ValueError(
                "This Space pipeline expects `image` as PIL/np inputs (not pre-latents) in this setup."
            )

        # ---- timesteps ----
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # guidance-distilled models need explicit guidance input
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        else:
            if guidance_scale is not None:
                logger.warning("guidance_scale passed but ignored since model is not guidance-distilled.")
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        image_rotary_emb = self.transformer.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        do_true_cfg = (
            (true_cfg_scale > 1)
            and (negative_prompt_embeds is not None)
            and (negative_prompt_embeds_mask is not None)
        )
        if do_true_cfg:
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
            uncond_image_rotary_emb = self.transformer.pos_embed(img_shapes, negative_txt_seq_lens, device=latents.device)
        else:
            uncond_image_rotary_emb = None

        # ---- denoise ----
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            image_rotary_emb=uncond_image_rotary_emb,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / (noise_norm + 1e-8))

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # ---- decode ----
        if output_type == "latent":
            image_out = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)

            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            if decoder_vae == "wan2x":
                alt_vae = _get_wan2x_vae(latents.device, self.vae.dtype)
                decoder_out = alt_vae.decode(latents, return_dict=False)[0]  # [B, 12, F, H, W]
                img_2x = F.pixel_shuffle(decoder_out[:, :, 0], upscale_factor=2)  # [B, 3, 2H, 2W]
                if keep_decoder_2x:
                    decoded = img_2x
                else:
                    decoded = F.interpolate(img_2x, size=(int(height), int(width)), mode="area")
            else:
                decoded = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

            image_out = self.image_processor.postprocess(decoded, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return QwenImagePipelineOutput(images=image_out)