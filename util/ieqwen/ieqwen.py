# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Copyright 2025 the authors of the "Qwen-Image-Edit-Rapid-AIO-Loras-Experimental" Space.
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

"""Qwen-Image-Edit-2511 Rapid AIO: image + prompt -> edited image, in one file.

Extracted from the Hugging Face Space
https://huggingface.co/spaces/aet256/Qwen-Image-Edit-Rapid-AIO-Loras-Experimental
so that the default workflow (one input image, one prompt, one edited image)
runs without Gradio, without the `spaces` runtime, and without any LoRA.

Layout of this file, top to bottom:

  1. The AIO transformer, vendored verbatim from the Space's
     `qwenimage/transformer_qwenimage.py` (the Rapid AIO checkpoint
     prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19 is loaded into this class).
  2. The edit pipeline, vendored verbatim from the Space's
     `qwenimage/pipeline_qwenimage_edit_plus.py`.
  3. The extracted workflow: `QwenImageEditRapidAIO`, the app.py inference
     path reduced to the default configuration, with lazy model loading.

Nothing is downloaded at import time or at construction time. The first call
to `edit()` (or an explicit `preload()`) fetches the base pipeline
(Qwen/Qwen-Image-Edit-2511) and the AIO transformer. Run this file with no
arguments to see that: it builds the pipeline object and exits having
downloaded nothing.

Defaults are unchanged from the Space UI: guidance scale 1.0, 4 inference
steps, a 1.0-megapixel canvas, a 32-pixel resolution lattice, pad-to-canvas
on, highlight protection on at strength 0.35, VAE tiling off, seed 0 with
randomization on.

Dependencies (same as the Space's requirements.txt, minus gradio/spaces):
torch, torchvision, transformers (v4.57.3), diffusers, accelerate, peft,
safetensors, huggingface_hub, sentencepiece, numpy, av; `kernels` only if
you set ENABLE_FA3=1.
"""

import argparse
import functools
import gc
import inspect
import math
import os
import random
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(
                        frame, height, width, idx
                    )
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_pos[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(
                frame, height, width, -1
            )
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(
                frame, height, width, -1
            )
        else:
            freqs_height = (
                freqs_pos[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(
            seq_lens, -1
        )
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)"
            )

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(
            theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
        )

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                    )
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


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


def choose_condition_area(
    canvas_area: int, base_area: int = CONDITION_IMAGE_SIZE
) -> int:
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
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one."
        )

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
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

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.latent_channels = (
            self.vae.config.z_dim if getattr(self, "vae", None) else 16
        )

        # QwenImage latents are turned into 2x2 patches and packed; multiply scale-factor by patch size
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
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

        model_inputs = self.processor(
            text=txt, images=image, padding=True, return_tensors="pt"
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])

        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
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
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, image, device
            )

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_images_per_prompt, seq_len
        )

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
        if (
            height % (self.vae_scale_factor * 2) != 0
            or width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. "
                "Dimensions will be resized accordingly."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` must also be passed."
            )

        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` must also be passed."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}"
            )

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
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
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator, sample_mode="argmax"
            )

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
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
                    image_latents = self._encode_vae_image(
                        image=image, generator=generator
                    )
                else:
                    image_latents = image

                if (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] == 0
                ):
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat(
                        [image_latents] * additional_image_per_prompt, dim=0
                    )
                elif (
                    batch_size > image_latents.shape[0]
                    and batch_size % image_latents.shape[0] != 0
                ):
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents,
                    batch_size,
                    num_channels_latents,
                    image_latent_height,
                    image_latent_width,
                )
                all_image_latents.append(image_latents)

            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height, width
            )
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
        multiple_of = (
            int(resolution_multiple)
            if resolution_multiple is not None
            else (self.vae_scale_factor * 2)
        )
        multiple_of = max(1, multiple_of)

        calculated_width, calculated_height = calculate_dimensions(
            1024 * 1024,
            float(image_size[0]) / float(image_size[1]),
            multiple=multiple_of,
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
        if image is not None and not (
            isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
        ):
            if not isinstance(image, list):
                image = [image]

            canvas_area = int(width) * int(height)
            cond_area = (
                int(condition_area)
                if condition_area is not None
                else choose_condition_area(canvas_area)
            )

            cond_w, cond_h = calculate_dimensions(
                cond_area, float(width) / float(height), multiple=multiple_of
            )

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
                condition_images.append(
                    self.image_processor.resize(pil, cond_h, cond_w)
                )

                # VAE stream (selective)
                if idx in vae_set:
                    if (
                        (ref_w is not None)
                        and (ref_h is not None)
                        and (int(idx) >= int(vae_ref_start_index))
                    ):
                        vw, vh = int(ref_w), int(ref_h)
                    else:
                        vw, vh = int(width), int(height)

                    vae_image_sizes.append((vw, vh))
                    vae_images.append(
                        self.image_processor.preprocess(
                            pil, int(vh), int(vw)
                        ).unsqueeze(2)
                    )

            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None
                and negative_prompt_embeds_mask is not None
            )
            if true_cfg_scale > 1 and not has_neg_prompt:
                logger.warning(
                    f"true_cfg_scale={true_cfg_scale} but CFG disabled because no negative prompt was provided."
                )
            if true_cfg_scale <= 1 and has_neg_prompt:
                logger.warning(
                    "negative_prompt provided but CFG disabled because true_cfg_scale <= 1"
                )

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
                negative_prompt_embeds, negative_prompt_embeds_mask = (
                    self.encode_prompt(
                        image=condition_images,
                        prompt=negative_prompt,
                        prompt_embeds=negative_prompt_embeds,
                        prompt_embeds_mask=negative_prompt_embeds_mask,
                        device=device,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=max_sequence_length,
                    )
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
                    (
                        1,
                        height // self.vae_scale_factor // 2,
                        width // self.vae_scale_factor // 2,
                    ),
                    *[
                        (
                            1,
                            vae_h // self.vae_scale_factor // 2,
                            vae_w // self.vae_scale_factor // 2,
                        )
                        for (vae_w, vae_h) in vae_image_sizes
                    ],
                ]
            ] * batch_size

        else:
            raise ValueError(
                "This Space pipeline expects `image` as PIL/np inputs (not pre-latents) in this setup."
            )

        # ---- timesteps ----
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )

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

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # guidance-distilled models need explicit guidance input
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            ).expand(latents.shape[0])
        else:
            if guidance_scale is not None:
                logger.warning(
                    "guidance_scale passed but ignored since model is not guidance-distilled."
                )
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else None
        )
        image_rotary_emb = self.transformer.pos_embed(
            img_shapes, txt_seq_lens, device=latents.device
        )

        do_true_cfg = (
            (true_cfg_scale > 1)
            and (negative_prompt_embeds is not None)
            and (negative_prompt_embeds_mask is not None)
        )
        if do_true_cfg:
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
            uncond_image_rotary_emb = self.transformer.pos_embed(
                img_shapes, negative_txt_seq_lens, device=latents.device
            )
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

                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / (noise_norm + 1e-8))

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # ---- decode ----
        if output_type == "latent":
            image_out = latents
        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = latents.to(self.vae.dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            if decoder_vae == "wan2x":
                alt_vae = _get_wan2x_vae(latents.device, self.vae.dtype)
                decoder_out = alt_vae.decode(latents, return_dict=False)[
                    0
                ]  # [B, 12, F, H, W]
                img_2x = F.pixel_shuffle(
                    decoder_out[:, :, 0], upscale_factor=2
                )  # [B, 3, 2H, 2W]
                if keep_decoder_2x:
                    decoded = img_2x
                else:
                    decoded = F.interpolate(
                        img_2x, size=(int(height), int(width)), mode="area"
                    )
            else:
                decoded = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

            image_out = self.image_processor.postprocess(
                decoded, output_type=output_type
            )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)
        return QwenImagePipelineOutput(images=image_out)


# ============================================================
# Extracted workflow: image + prompt -> edited image
# ============================================================
# This is app.py's inference path reduced to the default configuration:
# no LoRA ("None"), no second image, no extra references, no derived
# conditioning. Every default matches the Space UI: guidance scale 1.0,
# 4 inference steps, a 1.0-megapixel canvas, a 32-pixel resolution
# lattice, pad-to-canvas on, highlight protection on at strength 0.35,
# VAE tiling off, seed 0 with randomization on.

AIO_REPO_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"
DEFAULT_AIO_VERSION = "v19"
FALLBACK_AIO_VERSION = "v19"
BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"

_VER_RE = re.compile(r"^v\d+$")
_DIGITS_RE = re.compile(r"^\d+$")

MAX_SEED = np.iinfo(np.int32).max

dtype = torch.bfloat16


def _normalize_version(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _VER_RE.fullmatch(s):
        return s
    # forgiving: allow "21" -> "v21"
    if _DIGITS_RE.fullmatch(s):
        return f"v{s}"
    return None


def _round_to_multiple(x: int, m: int) -> int:
    return max(m, (int(x) // m) * m)


def compute_canvas_dimensions_from_area(
    image: Image.Image,
    target_area: int,
    multiple_of: int,
) -> Tuple[int, int]:
    """Compute (width, height) that matches image aspect ratio and approximates target_area.

    The result is floored to be divisible by multiple_of (typically vae_scale_factor*2).
    """
    w, h = image.size
    aspect = w / h if h else 1.0

    # Use the pipeline's own area->(w,h) helper for consistency.
    width, height = calculate_dimensions(int(target_area), float(aspect))
    width = _round_to_multiple(int(width), int(multiple_of))
    height = _round_to_multiple(int(height), int(multiple_of))
    return width, height


def protect_highlights(
    image: Image.Image, strength: float = 0.35, threshold: float = 0.82
) -> Image.Image:
    """Apply a gentle shoulder curve to bright pixels to reduce clipped highlights."""
    try:
        amount = max(0.0, min(1.0, float(strength)))
    except Exception:
        amount = 0.0

    if amount <= 0:
        return image

    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    luminance = arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722

    knee = max(0.01, 1.0 - float(threshold))
    mask = np.clip((luminance - float(threshold)) / knee, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    compressed_luminance = float(threshold) + (luminance - float(threshold)) * (
        1.0 - amount * 0.72 * mask
    )
    ratio = np.divide(
        compressed_luminance,
        np.maximum(luminance, 1e-6),
        out=np.ones_like(luminance),
        where=luminance > 1e-6,
    )

    corrected = np.clip(arr * ratio[:, :, None], 0.0, 1.0)
    return Image.fromarray((corrected * 255.0 + 0.5).astype(np.uint8), mode="RGB")


# ---- FlashAttention-3 support (optional; off by default) ----
# The Space imports its FA3 processor at module load, which fetches the
# kernel from the `kernels` hub package immediately. Here the fetch and the
# custom-op registration are deferred until FA3 is actually requested, so
# importing this file never downloads anything.

_fa3_state: Dict[str, Any] = {
    "registered": False,
    "flash_attn_func": None,
    "error": None,
}


def _ensure_fa3_available() -> None:
    if _fa3_state["registered"]:
        return
    try:
        from kernels import get_kernel

        _k = get_kernel("kernels-community/vllm-flash-attn3")
        _flash = _k.flash_attn_func
    except Exception as e:
        _fa3_state["error"] = e
        raise ImportError(
            "FlashAttention-3 via Hugging Face `kernels` is required. "
            "Tried `get_kernel('kernels-community/vllm-flash-attn3')` and failed with:\n"
            f"{e}"
        )

    @torch.library.custom_op("flash::flash_attn_func", mutates_args=())
    def flash_attn_func(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        outputs, lse = _flash(q, k, v, causal=causal)
        return outputs

    @flash_attn_func.register_fake
    def _(q, k, v, **kwargs):
        meta_q = torch.empty_like(q).contiguous()
        return meta_q

    _fa3_state["flash_attn_func"] = flash_attn_func
    _fa3_state["registered"] = True


class QwenDoubleStreamAttnProcessorFA3:
    """
    FA3-based attention processor for Qwen double-stream architecture.
    Computes joint attention over concatenated [text, image] streams using vLLM FlashAttention-3
    accessed via Hugging Face `kernels`.

    Notes / limitations:
    - General attention masks are not supported here (FA3 path). `is_causal=False` and no arbitrary mask.
    - Expects an available `apply_rotary_emb_qwen` in scope (same as the SDPA processor).
    """

    _attention_backend = (
        "fa3"  # for parity with the other processors, not used internally
    )

    def __init__(self):
        _ensure_fa3_available()

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention module with to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, and .heads
        hidden_states: torch.FloatTensor,  # (B, S_img, D_model)  image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model)  text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,  # unused in FA3 path
        attention_mask: Optional[torch.FloatTensor] = None,  # unused in FA3 path
        image_rotary_emb: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (img_freqs, txt_freqs)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states (text stream)."
            )
        if attention_mask is not None:
            # FA3 kernel path here does not consume arbitrary masks; fail fast to avoid silent correctness issues.
            raise NotImplementedError(
                "attention_mask is not supported in this FA3 implementation."
            )

        _ensure_fa3_available()
        flash_attn_func = _fa3_state["flash_attn_func"]

        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # ---- QKV projections (image/sample stream) ----
        img_q = attn.to_q(hidden_states)  # (B, S_img, D)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        # ---- QKV projections (text/context stream) ----
        txt_q = attn.add_q_proj(encoder_hidden_states)  # (B, S_txt, D)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # ---- Reshape to (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # expects tensors shaped (B, S, H, D_h)
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # FlashAttention-3 path expects (B, S, H, D_h) and returns (out, softmax_lse)
        out = flash_attn_func(q, k, v, causal=False)  # out: (B, S_total, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3).to(q.dtype)

        # Split back to text / image segments
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out


def _is_flash_attention_kernel_error(exc: BaseException) -> bool:
    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).lower()
    return (
        "flash-attn" in text
        or "flash_attn" in text
        or "no kernel image is available for execution on the device" in text
    )


class QwenImageEditRapidAIO:
    """Image + prompt -> edited image, with the models loaded lazily.

    Constructing the class downloads nothing and touches no network. The
    base pipeline (Qwen/Qwen-Image-Edit-2511) and the Rapid AIO transformer
    (prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19) are fetched on the first
    call to `edit()` or `preload()`.

    Defaults match the Space UI exactly.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        aio_version: Optional[str] = None,
        enable_fa3: Optional[bool] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # AIO version: explicit argument wins, then the AIO_VERSION env var,
        # then the default v19 (same precedence as the Space).
        env_raw = os.environ.get("AIO_VERSION", "")
        env_norm = _normalize_version(env_raw)
        self.aio_version = aio_version or env_norm or DEFAULT_AIO_VERSION

        if enable_fa3 is None:
            enable_fa3 = os.environ.get("ENABLE_FA3", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.enable_fa3 = bool(enable_fa3)

        self.attention_backend = None
        self.pipe: Optional[QwenImageEditPlusPipeline] = None

    # ---- lazy loading ----

    def _load_pipe_with_version(self, version: str) -> QwenImageEditPlusPipeline:
        print(f"📦 Loading AIO transformer from: {AIO_REPO_ID}")
        p = QwenImageEditPlusPipeline.from_pretrained(
            BASE_REPO_ID,
            transformer=QwenImageTransformer2DModel.from_pretrained(
                AIO_REPO_ID,
                torch_dtype=dtype,
                device_map="cuda",
            ),
            torch_dtype=dtype,
        ).to(self.device)
        return p

    def _use_sdpa_attention(self) -> None:
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessor2_0())
        self.attention_backend = "sdpa"
        print("Using standard PyTorch SDPA attention processor.")

    def _use_fa3_attention(self) -> None:
        self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
        self.attention_backend = "fa3"
        print("Flash Attention 3 processor set successfully.")

    def _setup_attention(self) -> None:
        try:
            if self.enable_fa3:
                self._use_fa3_attention()
            else:
                self._use_sdpa_attention()
        except Exception as e:
            print(
                f"Warning: Could not set requested attention processor, falling back to SDPA: {e}"
            )
            self._use_sdpa_attention()

    def _ensure_pipe(self) -> QwenImageEditPlusPipeline:
        if self.pipe is not None:
            return self.pipe

        print(f"AIO_VERSION = {self.aio_version}")
        # Forgiving load: try the requested version, fall back to a known
        # working version if it fails (same as the Space).
        try:
            self.pipe = self._load_pipe_with_version(self.aio_version)
        except Exception:
            print(
                f"❌ Failed to load requested AIO_VERSION. Falling back to {FALLBACK_AIO_VERSION}."
            )
            print("---- exception ----")
            print(traceback.format_exc())
            print("-------------------")
            self.aio_version = FALLBACK_AIO_VERSION
            self.pipe = self._load_pipe_with_version(self.aio_version)

        self._setup_attention()
        return self.pipe

    def preload(self) -> "QwenImageEditRapidAIO":
        """Force the model download/load now instead of on first edit."""
        self._ensure_pipe()
        return self

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    # ---- VAE tiling toggle (OFF by default, applied per request) ----

    def _apply_vae_tiling(self, enabled: bool) -> None:
        pipe = self.pipe
        try:
            if enabled:
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
                    print("✅ VAE tiling ENABLED (per request).")
                elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                    print(
                        "✅ VAE tiling ENABLED via pipe.vae.enable_tiling() (per request)."
                    )
                else:
                    print(
                        "⚠️ No enable_vae_tiling()/vae.enable_tiling() found; cannot enable."
                    )
            else:
                if hasattr(pipe, "disable_vae_tiling"):
                    pipe.disable_vae_tiling()
                    print("🛑 VAE tiling DISABLED (per request).")
                elif hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                    pipe.vae.disable_tiling()
                    print(
                        "🛑 VAE tiling DISABLED via pipe.vae.disable_tiling() (per request)."
                    )
                else:
                    # If no disable method exists, leave current state unchanged.
                    print(
                        "⚠️ No disable_vae_tiling()/vae.disable_tiling() found; leaving current state unchanged."
                    )
        except Exception as e:
            print(f"⚠️ VAE tiling toggle failed: {e}")

    # ---- inference ----

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 0,
        randomize_seed: bool = True,
        guidance_scale: float = 1.0,
        steps: int = 4,
        target_megapixels: float = 1.0,
        resolution_multiple: int = 32,
        pad_to_canvas: bool = True,
        vae_tiling: bool = False,
        decoder_vae: str = "qwen",
        keep_decoder_2x: bool = False,
        highlight_protection: bool = True,
        highlight_protection_strength: float = 0.35,
    ) -> Tuple[Image.Image, int]:
        """Edit one image with one prompt; returns (edited image, seed used)."""
        if image is None:
            raise ValueError("Please provide an input image.")
        if not prompt or not str(prompt).strip():
            raise ValueError("Please provide a prompt.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pipe = self._ensure_pipe()

        # No LoRA in this workflow; clear any adapters defensively.
        try:
            pipe.set_adapters([], adapter_weights=[])
        except Exception:
            pass

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        negative_prompt = (
            "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, "
            "overexposed, blown highlights, clipped whites, washed out, harsh lighting"
        )
        true_cfg_scale = float(guidance_scale)
        active_negative_prompt = negative_prompt if true_cfg_scale > 1.0 else None

        img1 = image.convert("RGB")

        # Single image: passed to the pipeline as-is (not a list).
        pipe_images = img1

        # Resolution derived from the input image. Use target *area*
        # (≈ megapixels) rather than long-edge sizing to reduce FOV drift.
        try:
            mp = float(target_megapixels)
        except Exception:
            mp = 1.0
        # Treat 0 MP as "match input area"
        if mp <= 0:
            w, h = img1.size
            target_area = int(w * h)
        else:
            target_area = int(mp * 1024 * 1024)

        width, height = compute_canvas_dimensions_from_area(
            img1,
            target_area=target_area,
            multiple_of=int(resolution_multiple),
        )

        try:
            print("[infer] submitting request | " f"seed={seed} prompt={prompt!r}")
            print(
                f"[infer] canvas={width}x{height} (~{(width*height)/1_048_576:.3f} MP) vae_tiling={bool(vae_tiling)}"
            )

            # Lattice multiple passed to pipeline too (anti-drift / valid size grid)
            res_mult = (
                int(resolution_multiple)
                if resolution_multiple is not None
                else int(pipe.vae_scale_factor * 2)
            )

            self._apply_vae_tiling(bool(vae_tiling))

            pipe_kwargs = dict(
                image=pipe_images,
                prompt=prompt,
                negative_prompt=active_negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                pad_to_canvas=bool(pad_to_canvas),
                resolution_multiple=res_mult,
                decoder_vae=str(decoder_vae).lower(),
                keep_decoder_2x=bool(keep_decoder_2x),
            )

            try:
                result = pipe(**pipe_kwargs).images[0]
            except RuntimeError as e:
                if (
                    self.attention_backend == "fa3"
                    and _is_flash_attention_kernel_error(e)
                ):
                    print(
                        "⚠️ Flash Attention 3 is not compatible with this GPU/runtime. Retrying with SDPA attention."
                    )
                    self._use_sdpa_attention()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    result = pipe(**pipe_kwargs).images[0]
                else:
                    raise

            if bool(highlight_protection):
                result = protect_highlights(
                    result, strength=float(highlight_protection_strength)
                )

            return result, seed
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2511 Rapid AIO: put in an image and a prompt, get an edited image."
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to the input image."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Edit prompt.")
    parser.add_argument(
        "--output",
        type=str,
        default="edited.png",
        help="Where to save the edited image.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed (used only with --no-randomize)."
    )
    parser.add_argument(
        "--no-randomize", action="store_true", help="Disable seed randomization."
    )
    parser.add_argument(
        "--guidance", type=float, default=1.0, help="Guidance scale (default 1.0)."
    )
    parser.add_argument(
        "--steps", type=int, default=4, help="Inference steps (default 4)."
    )
    parser.add_argument(
        "--megapixels",
        type=float,
        default=1.0,
        help="Target canvas megapixels (default 1.0; 0 = match input area).",
    )
    parser.add_argument(
        "--lattice",
        type=int,
        default=32,
        choices=[32, 56, 112],
        help="Resolution lattice multiple (default 32).",
    )
    parser.add_argument("--no-pad", action="store_true", help="Disable pad-to-canvas.")
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling (lower VRAM, slower).",
    )
    parser.add_argument(
        "--no-highlight-protection",
        action="store_true",
        help="Disable highlight protection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the pipeline object and exit without loading or downloading any model.",
    )
    args = parser.parse_args()

    # Constructing the pipeline class downloads nothing; models load lazily
    # on the first edit() call.
    pipeline = QwenImageEditRapidAIO()
    print(f"Pipeline created (models loaded: {pipeline.is_loaded}).")

    if args.dry_run:
        print("Dry run: exiting without downloading or loading any model.")
        return

    if not args.image or not args.prompt:
        parser.error("--image and --prompt are required unless --dry-run is given.")

    image = Image.open(args.image)
    result, used_seed = pipeline.edit(
        image,
        args.prompt,
        seed=args.seed,
        randomize_seed=not args.no_randomize,
        guidance_scale=args.guidance,
        steps=args.steps,
        target_megapixels=args.megapixels,
        resolution_multiple=args.lattice,
        pad_to_canvas=not args.no_pad,
        vae_tiling=args.vae_tiling,
        highlight_protection=not args.no_highlight_protection,
    )
    result.save(args.output)
    print(f"Saved edited image (seed {used_seed}) to: {args.output}")


if __name__ == "__main__":
    main()
