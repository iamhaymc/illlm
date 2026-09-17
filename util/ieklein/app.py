import os
import re
import gc
import traceback
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image, ImageDraw
from typing import Iterable, Optional

from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# ============================================================
# Theme
# ============================================================

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)


class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"),
            "Arial",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


orange_red_theme = OrangeRedTheme()

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", device)

# ============================================================
# AIO version (Space variable)
# ============================================================

AIO_REPO_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"
DEFAULT_AIO_VERSION = "v19"
FALLBACK_AIO_VERSION = "v19"

_VER_RE = re.compile(r"^v\d+$")
_DIGITS_RE = re.compile(r"^\d+$")


def _normalize_version(raw: str) -> Optional[str]:
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


_AIO_ENV_RAW = os.environ.get("AIO_VERSION", "")
_AIO_ENV_NORM = _normalize_version(_AIO_ENV_RAW)

AIO_VERSION = _AIO_ENV_NORM or DEFAULT_AIO_VERSION
AIO_VERSION_SOURCE = "env" if _AIO_ENV_NORM else "default(v19)"

print(f"AIO_VERSION (env raw) = {_AIO_ENV_RAW!r}")
print(f"AIO_VERSION (normalized) = {_AIO_ENV_NORM!r}")
print(f"Using AIO_VERSION = {AIO_VERSION} ({AIO_VERSION_SOURCE})")

# ============================================================
# Pipeline
# ============================================================

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: F401
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
)
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16
ENABLE_FA3 = os.environ.get("ENABLE_FA3", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ATTENTION_BACKEND = "sdpa"


def _load_pipe_with_version(version: str) -> QwenImageEditPlusPipeline:
    print(f"📦 Loading AIO transformer from: {AIO_REPO_ID}")
    p = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        transformer=QwenImageTransformer2DModel.from_pretrained(
            AIO_REPO_ID,
            torch_dtype=dtype,
            device_map="cuda",
        ),
        torch_dtype=dtype,
    ).to(device)
    return p


# Forgiving load: try env/default version, fallback to a known working version if it fails.
try:
    pipe = _load_pipe_with_version(AIO_VERSION)
except Exception as e:
    print(
        f"❌ Failed to load requested AIO_VERSION. Falling back to {FALLBACK_AIO_VERSION}."
    )
    print("---- exception ----")
    print(traceback.format_exc())
    print("-------------------")
    AIO_VERSION = FALLBACK_AIO_VERSION
    AIO_VERSION_SOURCE = f"fallback_to_{FALLBACK_AIO_VERSION}"
    pipe = _load_pipe_with_version(AIO_VERSION)


def _use_sdpa_attention() -> None:
    global ATTENTION_BACKEND
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessor2_0())
    ATTENTION_BACKEND = "sdpa"
    print("Using standard PyTorch SDPA attention processor.")


def _use_fa3_attention() -> None:
    global ATTENTION_BACKEND
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    ATTENTION_BACKEND = "fa3"
    print("Flash Attention 3 processor set successfully.")


def _is_flash_attention_kernel_error(exc: BaseException) -> bool:
    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).lower()
    return (
        "flash-attn" in text
        or "flash_attn" in text
        or "no kernel image is available for execution on the device" in text
    )


try:
    if ENABLE_FA3:
        _use_fa3_attention()
    else:
        _use_sdpa_attention()
except Exception as e:
    print(
        f"Warning: Could not set requested attention processor, falling back to SDPA: {e}"
    )
    _use_sdpa_attention()

MAX_SEED = np.iinfo(np.int32).max

# ============================================================
# VAE tiling toggle (UI-controlled; OFF by default)
# ============================================================


def _apply_vae_tiling(enabled: bool):
    """
    Toggle VAE tiling on the global pipeline.

    This does NOT require a Space restart; it applies to the next pipe(...) call.
    Note: this is global process state, so concurrent users could flip it between runs.
    """
    try:
        if enabled:
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
                print("✅ VAE tiling ENABLED (per UI).")
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                print("✅ VAE tiling ENABLED via pipe.vae.enable_tiling() (per UI).")
            else:
                print(
                    "⚠️ No enable_vae_tiling()/vae.enable_tiling() found; cannot enable."
                )
        else:
            if hasattr(pipe, "disable_vae_tiling"):
                pipe.disable_vae_tiling()
                print("🛑 VAE tiling DISABLED (per UI).")
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()
                print("🛑 VAE tiling DISABLED via pipe.vae.disable_tiling() (per UI).")
            else:
                # If no disable method exists, we leave current state unchanged.
                print(
                    "⚠️ No disable_vae_tiling()/vae.disable_tiling() found; leaving current state unchanged."
                )
    except Exception as e:
        print(f"⚠️ VAE tiling toggle failed: {e}")


# ============================================================
# Derived conditioning (Transformers): Depth
# ============================================================
# Depth uses Depth Anything V2 Small (Transformers-compatible):
# https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Lazy cache keyed by device string ("cpu" / "cuda")
_DEPTH_CACHE = {}


def _derived_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")


def _load_depth_models(dev: torch.device):
    key = str(dev)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(dev)
    model.eval()
    _DEPTH_CACHE[key] = (proc, model)
    return _DEPTH_CACHE[key]


@torch.inference_mode()
def make_depth_map(img: Image.Image, *, use_gpu: bool) -> Image.Image:
    dev = _derived_device(use_gpu)
    proc, model = _load_depth_models(dev)

    w, h = img.size
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(dev)
    outputs = model(**inputs)
    predicted = outputs.predicted_depth  # [B, H, W]

    depth = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)[0]

    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    depth = (depth * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(depth).convert("RGB")


# ============================================================
# LoRA adapters + presets
# ============================================================

NONE_LORA = "None"

ADAPTER_SPECS = {
    "Consistance": {
        "type": "single",
        "repo": "sdfafdfsdf/QIE_2511_Consistency_Lora",
        "weights": "qe2511_consis_alpha_patched.safetensors",
        "adapter_name": "Consistency",
        "strength": 0.6,
    },
    "Semirealistic-photo-detailer": {
        "type": "single",
        "repo": "rzgar/Qwen-Image-Edit-semi-realistic-detailer",
        "weights": "Qwen-Image-Edit-Anime-Semi-Realistic-Detailer-v1.safetensors",
        "adapter_name": "semirealistic",
        "strength": 1.0,
    },
    "AnyPose": {
        "type": "package",
        "requires_two_images": True,
        "image2_label": "Upload Pose Reference (Image 2)",
        "parts": [
            {
                "repo": "lilylilith/AnyPose",
                "weights": "2511-AnyPose-base-000006250.safetensors",
                "adapter_name": "anypose-base",
                "strength": 0.7,
            },
            {
                "repo": "lilylilith/AnyPose",
                "weights": "2511-AnyPose-helper-00006000.safetensors",
                "adapter_name": "anypose-helper",
                "strength": 0.7,
            },
        ],
    },
    "Any2Real_2601": {
        "type": "single",
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601_A_final_patched.safetensors",
        "adapter_name": "photoreal",
        "strength": 1.0,
    },
    "Hyperrealistic-Portrait": {
        "type": "single",
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "HRPortrait",
        "strength": 1.0,
    },
    "Ultrarealistic-Portrait": {
        "type": "single",
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "URPortrait",
        "strength": 1.0,
    },
    "BFS-Best-FaceSwap": {
        "type": "single",
        "requires_two_images": True,
        "image2_label": "Upload Head/Face Donor (Image 2)",
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_original.safetensors",
        "adapter_name": "BFS-Best-Faceswap",
        "strength": 1.0,
        "needs_alpha_fix": True,  # <-- fixes KeyError 'img_in.alpha'
    },
    "BFS-Best-FaceSwap-merge": {
        "type": "single",
        "requires_two_images": True,
        "image2_label": "Upload Head/Face Donor (Image 2)",
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors",
        "adapter_name": "BFS-Best-Faceswap-merge",
        "strength": 1.1,
        "needs_alpha_fix": True,  # <-- fixes KeyError 'img_in.alpha'
    },
    "F2P": {
        "type": "single",
        "repo": "DiffSynth-Studio/Qwen-Image-Edit-F2P",
        "weights": "edit_0928_lora_step40000.safetensors",
        "adapter_name": "F2P",
        "strength": 1.0,
    },
    "Multiple-Angles": {
        "type": "single",
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
        "strength": 1.0,
    },
    "Light-Restoration": {
        "type": "single",
        "repo": "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "weights": "移除光影.safetensors",
        "adapter_name": "light-restoration",
        "strength": 1.0,
    },
    "Relight": {
        "type": "single",
        "repo": "dx8152/Qwen-Image-Edit-2509-Relight",
        "weights": "Qwen-Edit-Relight.safetensors",
        "adapter_name": "relight",
        "strength": 1.0,
    },
    "Multi-Angle-Lighting": {
        "type": "single",
        "repo": "dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        "weights": "多角度灯光-251116.safetensors",
        "adapter_name": "multi-angle-lighting",
        "strength": 1.0,
    },
    "Edit-Skin": {
        "type": "single",
        "repo": "tlennon-ie/qwen-edit-skin",
        "weights": "qwen-edit-skin_1.1_000002750.safetensors",
        "adapter_name": "edit-skin",
        "strength": 1.0,
    },
    "Next-Scene": {
        "type": "single",
        "repo": "lovis93/next-scene-qwen-image-lora-2509",
        "weights": "next-scene_lora-v2-3000.safetensors",
        "adapter_name": "next-scene",
        "strength": 1.0,
    },
    "Flat-Log": {
        "type": "single",
        "repo": "tlennon-ie/QwenEdit2509-FlatLogColor",
        "weights": "QwenEdit2509-FlatLogColor.safetensors",
        "adapter_name": "flat-log",
        "strength": 1.0,
    },
    "Upscale-Image": {
        "type": "single",
        "repo": "vafipas663/Qwen-Edit-2509-Upscale-LoRA",
        "weights": "qwen-edit-enhance_64-v3_000001000.safetensors",
        "adapter_name": "upscale-image",
        "strength": 1.0,
    },
    "Upscale2K": {
        "type": "single",
        "repo": "valiantcat/Qwen-Image-Edit-2509-Upscale2K",
        "weights": "qwen_image_edit_2509_upscale.safetensors",
        "adapter_name": "upscale-2k",
        "strength": 1.0,
        "target_long_edge": 2048,
    },
}

LORA_PRESET_PROMPTS = {
    "Any2Real_2601": "change the picture 1 to realistic photograph",
    "Semirealistic-photo-detailer": "transform the image to semi-realistic image",
    "AnyPose": "Make the person in image 1 do the exact same pose of the person in image 2. Changing the style and background of the image of the person in image 1 is undesirable, so don't do it. The new pose should be pixel accurate to the pose we are trying to copy. The position of the arms and head and legs should be the same as the pose we are trying to copy. Change the field of view and angle to match exactly image 2. Head tilt and eye gaze pose should match the person in image 2.",
    "Hyperrealistic-Portrait": "Transform the image into an ultra-realistic photorealistic portrait with strict identity preservation, facing straight to the camera. Enhance pore-level skin textures, realistic moisture effects, and natural wet hair clumping against the skin. Apply cool-toned soft-box lighting with subtle highlights and shadows, maintain realistic green-hazel eye catchlights without synthetic gloss, and preserve soft natural lip texture. Use shallow depth of field with a clean bokeh background, an 85mm macro photographic look, and raw photo grading without retouching to maintain realism and original details.",
    "Ultrarealistic-Portrait": "Transform the image into an ultra-realistic glamour portrait while strictly preserving the subject’s identity. Apply a close-up composition with a slight head tilt and a hand near the face, enhance cinematic directional lighting with dramatic fashion-style highlights, and refine makeup details including glowing skin, glossy lips, luminous highlighter, and defined eyes. Increase skin realism with detailed epidermal textures such as micropores, microhairs, subtle oil sheen, natural highlights, soft wrinkles, and subsurface scattering. Maintain a luxury fashion-magazine look in a 9:16 aspect ratio, preserving realism, facial structure, and original details without over-smoothing or retouching.",
    "Upscale2K": "Upscale this picture to 4K resolution.",
    "BFS-Best-FaceSwap": "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the eye direction, head rotation, and micro-expressions from Picture 1. high quality, sharp details, 4k",
    "BFS-Best-FaceSwap-merge": "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the eye direction, head rotation, and micro-expressions from Picture 1. high quality, sharp details, 4k",
}

# Track what is currently loaded in memory (adapter_name values)
LOADED_ADAPTERS = set()

# ============================================================
# Helpers: resolution
# ============================================================

# We prefer *area-based* sizing (≈ megapixels) over long-edge sizing.
# This aligns better with Qwen-Image-Edit's internal assumptions and reduces FOV drift.


def _round_to_multiple(x: int, m: int) -> int:
    return max(m, (int(x) // m) * m)


def compute_canvas_dimensions_from_area(
    image: Image.Image,
    target_area: int,
    multiple_of: int,
) -> tuple[int, int]:
    """Compute (width, height) that matches image aspect ratio and approximates target_area.

    The result is floored to be divisible by multiple_of (typically vae_scale_factor*2).
    """
    w, h = image.size
    aspect = w / h if h else 1.0

    # Use the pipeline's own area->(w,h) helper for consistency.
    from qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions

    width, height = calculate_dimensions(int(target_area), float(aspect))
    width = _round_to_multiple(int(width), int(multiple_of))
    height = _round_to_multiple(int(height), int(multiple_of))
    return width, height


def get_target_area_for_lora(
    image: Image.Image,
    lora_adapter: str,
    user_target_megapixels: float,
) -> int:
    """Return target pixel area for the canvas.

    Priority:
      1) Adapter spec: target_area (pixels) or target_megapixels
      2) Adapter spec: target_long_edge (legacy) -> converted to area using image aspect
      3) User slider target megapixels
    """
    spec = ADAPTER_SPECS.get(lora_adapter, {})

    if "target_area" in spec:
        try:
            return int(spec["target_area"])
        except Exception:
            pass

    if "target_megapixels" in spec:
        try:
            mp = float(spec["target_megapixels"])
            return int(mp * 1024 * 1024)
        except Exception:
            pass

    # Legacy support (e.g. Upscale2K)
    if "target_long_edge" in spec:
        try:
            long_edge = int(spec["target_long_edge"])
            w, h = image.size
            if w >= h:
                new_w = long_edge
                new_h = int(round(long_edge * (h / w)))
            else:
                new_h = long_edge
                new_w = int(round(long_edge * (w / h)))
            return int(new_w * new_h)
        except Exception:
            pass

    # User default
    try:
        mp = float(user_target_megapixels)
    except Exception:
        mp = 1.0

    # Treat 0 MP as "match input area"
    if mp <= 0:
        w, h = image.size
        return int(w * h)

    return int(mp * 1024 * 1024)


# ============================================================
# Helpers: multi-input routing + gallery normalization
# ============================================================


def lora_requires_two_images(lora_adapter: str) -> bool:
    return bool(ADAPTER_SPECS.get(lora_adapter, {}).get("requires_two_images", False))


def image2_label_for_lora(lora_adapter: str) -> str:
    return str(
        ADAPTER_SPECS.get(lora_adapter, {}).get(
            "image2_label", "Upload Reference (Image 2)"
        )
    )


def _to_pil_rgb(x) -> Optional[Image.Image]:
    """
    Accepts PIL / numpy / (image, caption) tuples from gr.Gallery and returns PIL RGB.
    Gradio Gallery commonly yields tuples like (image, caption).
    """
    if x is None:
        return None

    # Gallery often returns (image, caption)
    if isinstance(x, tuple) and len(x) >= 1:
        x = x[0]
        if x is None:
            return None

    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")

    # Best-effort fallback
    try:
        return Image.fromarray(np.array(x)).convert("RGB")
    except Exception:
        return None


def build_labeled_images(
    img1: Image.Image,
    img2: Optional[Image.Image],
    extra_imgs: Optional[list[Image.Image]],
) -> dict[str, Image.Image]:
    """
    Creates labels image_1, image_2, image_3... based on what is actually uploaded:
      - img1 is always image_1
      - img2 becomes image_2 only if present
      - extras start immediately after the last present base box
    The pipeline receives images in this exact order.
    """
    labeled: dict[str, Image.Image] = {}
    idx = 1

    labeled[f"image_{idx}"] = img1
    idx += 1

    if img2 is not None:
        labeled[f"image_{idx}"] = img2
        idx += 1

    if extra_imgs:
        for im in extra_imgs:
            if im is None:
                continue
            labeled[f"image_{idx}"] = im
            idx += 1

    return labeled


def _append_to_gallery(existing, new_img) -> list[Image.Image]:
    """Append new_img to an existing gallery list."""
    items: list[Image.Image] = []
    if existing:
        for item in existing:
            pil = _to_pil_rgb(item)
            if pil is not None:
                items.append(pil)
    if new_img is not None:
        pil = _to_pil_rgb(new_img)
        if pil is not None:
            items.append(pil)
    return items


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


# ============================================================
# Helpers: BFS alpha key fix
# ============================================================


def _inject_missing_alpha_keys(state_dict: dict) -> dict:
    """
    Diffusers' Qwen LoRA converter expects '<module>.alpha' keys.
    BFS safetensors omits them. We inject alpha = rank (neutral scaling).

    IMPORTANT: diffusers may strip 'diffusion_model.' before lookup, so we
    inject BOTH:
      - diffusion_model.xxx.alpha
      - xxx.alpha
    """
    bases = {}

    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".lora_down.weight") and v.ndim >= 1:
            base = k[: -len(".lora_down.weight")]
            rank = int(v.shape[0])
            bases[base] = rank

    for base, rank in bases.items():
        alpha_tensor = torch.tensor(float(rank), dtype=torch.float32)

        full_alpha = f"{base}.alpha"
        if full_alpha not in state_dict:
            state_dict[full_alpha] = alpha_tensor

        if base.startswith("diffusion_model."):
            stripped_base = base[len("diffusion_model.") :]
            stripped_alpha = f"{stripped_base}.alpha"
            if stripped_alpha not in state_dict:
                state_dict[stripped_alpha] = alpha_tensor

    return state_dict


def _filter_to_diffusers_lora_keys(state_dict: dict) -> tuple[dict, dict]:
    """Return (filtered_state_dict, stats).

    Some ComfyUI/Qwen safetensors (especially "merged" variants) include non-LoRA
    delta/patch keys like `*.diff` and `*.diff_b` alongside real LoRA tensors.
    Diffusers' internal Qwen LoRA converter is strict: any leftover keys cause an
    error (`state_dict should be empty...`).

    This helper keeps only the keys Diffusers can consume as a LoRA:
      - `*.lora_up.weight`
      - `*.lora_down.weight`
      - (rare) `*.lora_mid.weight`
      - alpha keys: `*.alpha` (or `*.lora_alpha` which we normalize to `*.alpha`)

    It also drops known patch keys (`*.diff`, `*.diff_b`) and everything else.
    """

    keep_suffixes = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_mid.weight",
        ".alpha",
        ".lora_alpha",
    )

    dropped_patch = 0
    dropped_other = 0
    kept = 0
    normalized_alpha = 0

    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            # Ignore non-tensor entries if any.
            dropped_other += 1
            continue

        # Drop ComfyUI "delta" keys that Diffusers' LoRA loader will never consume.
        if k.endswith(".diff") or k.endswith(".diff_b"):
            dropped_patch += 1
            continue

        if not k.endswith(keep_suffixes):
            dropped_other += 1
            continue

        if k.endswith(".lora_alpha"):
            # Normalize common alt name to what Diffusers expects.
            base = k[: -len(".lora_alpha")]
            k2 = f"{base}.alpha"
            out[k2] = v.float() if v.dtype != torch.float32 else v
            normalized_alpha += 1
            kept += 1
            continue

        out[k] = v
        kept += 1

    stats = {
        "kept": kept,
        "dropped_patch": dropped_patch,
        "dropped_other": dropped_other,
        "normalized_alpha": normalized_alpha,
    }
    return out, stats


def _duplicate_stripped_prefix_keys(
    state_dict: dict, prefix: str = "diffusion_model."
) -> dict:
    """Ensure both prefixed and unprefixed variants exist for LoRA-related keys.

    Diffusers' Qwen LoRA conversion may strip `diffusion_model.` when looking up
    modules. Some exports only include prefixed keys. To be maximally compatible,
    we duplicate LoRA keys (and alpha) in stripped form when missing.
    """

    out = dict(state_dict)
    for k, v in list(state_dict.items()):
        if not k.startswith(prefix):
            continue
        stripped = k[len(prefix) :]
        if stripped not in out:
            out[stripped] = v
    return out


def _load_lora_weights_with_fallback(
    repo: str, weight_name: str, adapter_name: str, needs_alpha_fix: bool = False
):
    """
    Normal path: pipe.load_lora_weights(repo, weight_name=..., adapter_name=...)
    BFS fallback: download safetensors, inject missing alpha keys, then load from dict.
    """
    try:
        pipe.load_lora_weights(repo, weight_name=weight_name, adapter_name=adapter_name)
        return
    except (KeyError, ValueError) as e:
        # KeyError: missing required alpha keys (common in BFS)
        # ValueError: Diffusers Qwen converter found leftover keys (e.g. .diff/.diff_b)
        if not needs_alpha_fix:
            raise

        print(
            "⚠️ LoRA load failed (will try safe dict fallback). "
            f"Adapter={adapter_name!r} file={weight_name!r} error={type(e).__name__}: {e}"
        )

        local_path = hf_hub_download(repo_id=repo, filename=weight_name)
        sd = safetensors_load_file(local_path)

        # 1) Inject required `<module>.alpha` keys (neutral scaling alpha=rank).
        sd = _inject_missing_alpha_keys(sd)

        # 2) Keep only LoRA + alpha keys; drop ComfyUI patch/delta keys.
        sd, stats = _filter_to_diffusers_lora_keys(sd)

        # 3) Duplicate stripped keys (remove `diffusion_model.`) for compatibility.
        sd = _duplicate_stripped_prefix_keys(sd)

        print(
            "🧹 LoRA dict cleanup stats: "
            f"kept={stats['kept']} dropped_patch={stats['dropped_patch']} "
            f"dropped_other={stats['dropped_other']} normalized_alpha={stats['normalized_alpha']}"
        )

        pipe.load_lora_weights(sd, adapter_name=adapter_name)
        return


# ============================================================
# LoRA loader: single/package + strengths
# ============================================================


def _ensure_loaded_and_get_active_adapters(selected_lora: str):
    spec = ADAPTER_SPECS.get(selected_lora)
    if not spec:
        raise gr.Error(f"Configuration not found for: {selected_lora}")

    adapter_names = []
    adapter_weights = []

    if spec.get("type") == "package":
        parts = spec.get("parts", [])
        if not parts:
            raise gr.Error(f"Package spec has no parts: {selected_lora}")

        for part in parts:
            repo = part["repo"]
            weights = part["weights"]
            adapter_name = part["adapter_name"]
            strength = float(part.get("strength", 1.0))
            needs_alpha_fix = bool(part.get("needs_alpha_fix", False))

            if adapter_name not in LOADED_ADAPTERS:
                print(
                    f"--- Downloading and Loading Adapter Part: {selected_lora} / {adapter_name} ---"
                )
                try:
                    _load_lora_weights_with_fallback(
                        repo=repo,
                        weight_name=weights,
                        adapter_name=adapter_name,
                        needs_alpha_fix=needs_alpha_fix,
                    )
                    LOADED_ADAPTERS.add(adapter_name)
                except Exception as e:
                    raise gr.Error(
                        f"Failed to load adapter part {selected_lora}/{adapter_name}: {e}"
                    )
            else:
                print(
                    f"--- Adapter part already loaded: {selected_lora} / {adapter_name} ---"
                )

            adapter_names.append(adapter_name)
            adapter_weights.append(strength)

    else:
        repo = spec["repo"]
        weights = spec["weights"]
        adapter_name = spec["adapter_name"]
        strength = float(spec.get("strength", 1.0))
        needs_alpha_fix = bool(spec.get("needs_alpha_fix", False))

        if adapter_name not in LOADED_ADAPTERS:
            print(f"--- Downloading and Loading Adapter: {selected_lora} ---")
            try:
                _load_lora_weights_with_fallback(
                    repo=repo,
                    weight_name=weights,
                    adapter_name=adapter_name,
                    needs_alpha_fix=needs_alpha_fix,
                )
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise gr.Error(f"Failed to load adapter {selected_lora}: {e}")
        else:
            print(f"--- Adapter {selected_lora} is already loaded. ---")

        adapter_names = [adapter_name]
        adapter_weights = [strength]

    return adapter_names, adapter_weights


# ============================================================
# UI handlers
# ============================================================


def on_lora_change_ui(selected_lora, current_prompt, current_extras_condition_only):
    # Preset prompt (fill only if empty)
    if selected_lora != NONE_LORA:
        preset = LORA_PRESET_PROMPTS.get(selected_lora, "")
        if preset and (current_prompt is None or str(current_prompt).strip() == ""):
            prompt_update = gr.update(value=preset)
        else:
            prompt_update = gr.update(value=current_prompt)
    else:
        prompt_update = gr.update(value=current_prompt)

    # Image2 visibility/label
    if lora_requires_two_images(selected_lora):
        img2_update = gr.update(
            visible=True, label=image2_label_for_lora(selected_lora)
        )
    else:
        img2_update = gr.update(
            visible=False, value=None, label="Upload Reference (Image 2)"
        )

    # Extra references routing default:
    # For BFS/AnyPose-like adapters, it's usually safer to keep extra refs as conditioning-only.
    if selected_lora in ("BFS-Best-FaceSwap", "BFS-Best-FaceSwap-merge", "AnyPose"):
        extras_update = gr.update(value=True)
    else:
        extras_update = gr.update(value=current_extras_condition_only)

    return prompt_update, img2_update, extras_update


# ============================================================
# UI helpers: output routing + derived conditioning
# ============================================================


def set_output_as_image1(last):
    if last is None:
        raise gr.Error("No output available yet.")
    return gr.update(value=last)


def set_output_as_image2(last):
    if last is None:
        raise gr.Error("No output available yet.")
    return gr.update(value=last)


def set_output_as_extra(last, existing_extra):
    if last is None:
        raise gr.Error("No output available yet.")
    return _append_to_gallery(existing_extra, last)


@spaces.GPU
def add_derived_ref(img1, existing_extra, derived_type, derived_use_gpu):
    if img1 is None:
        raise gr.Error("Please upload Image 1 first.")

    if derived_type == "None":
        return gr.update(value=existing_extra), gr.update(visible=False, value=None)

    base = img1.convert("RGB")

    if derived_type == "Depth (Depth Anything V2 Small)":
        derived = make_depth_map(base, use_gpu=bool(derived_use_gpu))
    else:
        raise gr.Error(f"Unknown derived type: {derived_type}")

    new_gallery = _append_to_gallery(existing_extra, derived)
    return gr.update(value=new_gallery), gr.update(visible=True, value=derived)


# ============================================================
# Inference
# ============================================================


@spaces.GPU
def infer(
    input_image_1,
    input_image_2,
    input_images_extra,  # gallery multi-image box
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    target_megapixels,
    extras_condition_only,
    pad_to_canvas,
    vae_tiling,  # VAE tiling toggle
    resolution_multiple,
    vae_ref_megapixels,
    decoder_vae,
    keep_decoder_2x,
    highlight_protection,
    highlight_protection_strength,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if input_image_1 is None:
        raise gr.Error("Please upload Image 1.")

    # Handle "None"
    if lora_adapter == NONE_LORA:
        try:
            pipe.set_adapters([], adapter_weights=[])
        except Exception:
            if LOADED_ADAPTERS:
                pipe.set_adapters(
                    list(LOADED_ADAPTERS), adapter_weights=[0.0] * len(LOADED_ADAPTERS)
                )
    else:
        adapter_names, adapter_weights = _ensure_loaded_and_get_active_adapters(
            lora_adapter
        )
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, "
        "overexposed, blown highlights, clipped whites, washed out, harsh lighting"
    )
    true_cfg_scale = float(guidance_scale)
    active_negative_prompt = negative_prompt if true_cfg_scale > 1.0 else None

    img1 = input_image_1.convert("RGB")
    img2 = input_image_2.convert("RGB") if input_image_2 is not None else None

    # Normalize extra images (Gallery) to PIL RGB (handles tuples from Gallery)
    extra_imgs: list[Image.Image] = []
    if input_images_extra:
        for item in input_images_extra:
            pil = _to_pil_rgb(item)
            if pil is not None:
                extra_imgs.append(pil)

    # Enforce existing 2-image LoRA behavior (image_1 + image_2 required)
    if lora_requires_two_images(lora_adapter) and img2 is None:
        raise gr.Error("This LoRA needs two images. Please upload Image 2 as well.")

    # Label images as image_1, image_2, image_3...
    labeled = build_labeled_images(img1, img2, extra_imgs)

    # Pass to pipeline in labeled order. Keep single-image call when only one is present.
    pipe_images = list(labeled.values())
    if len(pipe_images) == 1:
        pipe_images = pipe_images[0]

    # Resolution derived from Image 1 (base/body/target)
    # Use target *area* (≈ megapixels) rather than long-edge sizing to reduce FOV drift.
    target_area = get_target_area_for_lora(img1, lora_adapter, float(target_megapixels))
    width, height = compute_canvas_dimensions_from_area(
        img1,
        target_area=target_area,
        multiple_of=int(resolution_multiple),
    )

    # Decide which images participate in the VAE latent stream.
    # If enabled, extra references beyond (Img_1, Img_2) become conditioning-only.
    vae_image_indices = None
    if extras_condition_only:
        if isinstance(pipe_images, list) and len(pipe_images) > 2:
            vae_image_indices = [0, 1] if len(pipe_images) >= 2 else [0]

    try:
        print(
            "[DEBUG][infer] submitting request | "
            f"lora_adapter={lora_adapter!r} seed={seed} prompt={prompt!r}"
        )
        print(
            f"[DEBUG][infer] canvas={width}x{height} (~{(width*height)/1_048_576:.3f} MP) vae_tiling={bool(vae_tiling)}"
        )

        # ✅ Apply UI toggle per-request (OFF by default)
        # Lattice multiple passed to pipeline too (anti-drift / valid size grid)
        res_mult = (
            int(resolution_multiple)
            if resolution_multiple is not None
            else int(pipe.vae_scale_factor * 2)
        )

        # Optional: override VAE sizing for *extra* references (beyond Image 1 / Image 2)
        # Interpreted as megapixels; 0 disables override (uses canvas).
        try:
            mp_ref = float(vae_ref_megapixels)
        except Exception:
            mp_ref = 0.0

        vae_ref_area = int(mp_ref * 1024 * 1024) if mp_ref and mp_ref > 0 else None

        # Extras start index depends on whether Image 2 exists
        base_ref_count = 2 if img2 is not None else 1

        _apply_vae_tiling(bool(vae_tiling))

        pipe_kwargs = dict(
            image=pipe_images,
            prompt=prompt,
            negative_prompt=active_negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            vae_image_indices=vae_image_indices,
            pad_to_canvas=bool(pad_to_canvas),
            resolution_multiple=res_mult,
            vae_ref_area=vae_ref_area,
            vae_ref_start_index=base_ref_count,
            decoder_vae=str(decoder_vae).lower(),
            keep_decoder_2x=bool(keep_decoder_2x),
        )

        try:
            result = pipe(**pipe_kwargs).images[0]
        except RuntimeError as e:
            if ATTENTION_BACKEND == "fa3" and _is_flash_attention_kernel_error(e):
                print(
                    "⚠️ Flash Attention 3 is not compatible with this GPU/runtime. Retrying with SDPA attention."
                )
                _use_sdpa_attention()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = pipe(**pipe_kwargs).images[0]
            else:
                raise

        if bool(highlight_protection):
            result = protect_highlights(
                result, strength=float(highlight_protection_strength)
            )

        return result, seed, result
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@spaces.GPU
def infer_example(input_image, prompt, lora_adapter):
    if input_image is None:
        return None, 0, None
    input_pil = input_image.convert("RGB")
    guidance_scale = 1.0
    steps = 4
    # Examples don't supply Image 2 or extra images; and example list doesn't include AnyPose/BFS.
    # Keep VAE tiling OFF in examples (matches default).
    result, seed, last = infer(
        input_pil,
        None,
        None,
        prompt,
        lora_adapter,
        0,
        True,
        guidance_scale,
        steps,
        1.0,
        True,
        True,
        False,  # vae_tiling
        32,  # resolution_multiple
        0.0,  # vae_ref_megapixels
        "qwen",  # decoder_vae
        False,  # keep_decoder_2x
        True,  # highlight_protection
        0.35,  # highlight_protection_strength
    )
    return result, seed, last


# ============================================================
# UI
# ============================================================

css = """
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

aio_status_line = (
    f"**AIO transformer version:** `{AIO_VERSION}`  "
    f"({AIO_VERSION_SOURCE}; env `AIO_VERSION`={_AIO_ENV_RAW!r})"
)

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-Image-Edit-2511-LoRAs-Fast**", elem_id="main-title")
        gr.Markdown(
            f"""This **experimental** space for [QIE-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) utilizes [extracted transformers](https://huggingface.co/Pr0f3ssi0n4ln00b/Phr00t-Qwen-Rapid-AIO) of [Phr00t’s Rapid AIO merge](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO) and FA3-optimization with [LoRA](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image-Edit-2511) support and a couple of extra features:

- Optional conditioning-only routing for extra reference latents
- Uncapped canvas resolution
- Optional VAE tiling for high resolutions
- Optional depth mapping for conditioning
- Optional routing of output to input for further iterations
- Optional alternative decoder [VAE](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)

Current environment is running **{AIO_VERSION}** of the Rapid AIO. Duplicate the space and set the **AIO_VERSION** space variable to use a different version."""
        )
        gr.Markdown(aio_status_line)

        with gr.Row(equal_height=True):
            with gr.Column():
                input_image_1 = gr.Image(
                    label="Upload Image 1 (Base / Target)", type="pil", height=290
                )
                input_image_2 = gr.Image(
                    label="Upload Reference (Image 2)",
                    type="pil",
                    height=290,
                    visible=False,
                )

                input_images_extra = gr.Gallery(
                    label="Upload Additional Images (auto-indexed after Image 1/2)",
                    type="pil",
                    height=290,
                    columns=4,
                    rows=2,
                    interactive=True,
                )

                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., transform into photo..",
                )

                run_button = gr.Button("Edit Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(
                    label="Output Image", interactive=False, format="png", height=353
                )

                last_output = gr.State(value=None)

                with gr.Row():
                    btn_out_to_img1 = gr.Button(
                        "⬅️ Output → Image 1", variant="secondary"
                    )
                    btn_out_to_img2 = gr.Button(
                        "⬅️ Output → Image 2", variant="secondary"
                    )
                    btn_out_to_extra = gr.Button(
                        "➕ Output → Extra Ref", variant="secondary"
                    )

                derived_preview = gr.Image(
                    label="Derived Conditioning Preview",
                    interactive=False,
                    format="png",
                    height=200,
                    visible=False,
                )

                with gr.Row():
                    lora_choices = [NONE_LORA] + list(ADAPTER_SPECS.keys())
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=lora_choices,
                        value=NONE_LORA,
                    )

                with gr.Accordion("Advanced Settings", open=False, visible=True):
                    with gr.Accordion(
                        "Derived Conditioning (Pose / Depth)", open=False
                    ):
                        derived_type = gr.Dropdown(
                            label="Derived Type (from Image 1)",
                            choices=["None", "Depth (Depth Anything V2 Small)"],
                            value="None",
                        )
                        derived_use_gpu = gr.Checkbox(
                            label="Use GPU for derived model", value=False
                        )
                        add_derived_btn = gr.Button(
                            "➕ Add derived ref to Extras (conditioning-only recommended)"
                        )

                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                    )
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=1.0,
                    )
                    steps = gr.Slider(
                        label="Inference Steps", minimum=1, maximum=50, step=1, value=4
                    )
                    target_megapixels = gr.Slider(
                        label="Target Megapixels (canvas, 0 = match input area)",
                        minimum=0.0,
                        maximum=6.0,
                        step=0.1,
                        value=1.0,
                    )
                    resolution_multiple = gr.Dropdown(
                        label="Resolution lattice multiple (anti-drift)",
                        choices=[32, 56, 112],
                        value=32,
                        interactive=True,
                    )
                    vae_ref_megapixels = gr.Slider(
                        label="Extra refs VAE megapixels override (0 = use canvas)",
                        minimum=0.0,
                        maximum=6.0,
                        step=0.1,
                        value=0.0,
                    )
                    decoder_vae = gr.Dropdown(
                        label="Decoder VAE",
                        choices=["qwen", "wan2x"],
                        value="qwen",
                        interactive=True,
                    )
                    keep_decoder_2x = gr.Checkbox(
                        label="Keep 2× output (wan2x only)",
                        value=False,
                    )
                    highlight_protection = gr.Checkbox(
                        label="Highlight protection (reduce overexposure)",
                        value=True,
                    )
                    highlight_protection_strength = gr.Slider(
                        label="Highlight protection strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.35,
                    )
                    extras_condition_only = gr.Checkbox(
                        label="Extra references are conditioning-only (exclude from VAE)",
                        value=True,
                    )
                    pad_to_canvas = gr.Checkbox(
                        label="Pad images to canvas aspect (avoid warping)",
                        value=True,
                    )

                    # ✅ NEW: VAE tiling toggle (OFF by default)
                    vae_tiling = gr.Checkbox(
                        label="VAE tiling (lower VRAM, slower)",
                        value=False,
                    )

        # On LoRA selection: preset prompt + toggle Image 2
        lora_adapter.change(
            fn=on_lora_change_ui,
            inputs=[lora_adapter, prompt, extras_condition_only],
            outputs=[prompt, input_image_2, extras_condition_only],
        )

    run_button.click(
        fn=infer,
        inputs=[
            input_image_1,
            input_image_2,
            input_images_extra,
            prompt,
            lora_adapter,
            seed,
            randomize_seed,
            guidance_scale,
            steps,
            target_megapixels,
            extras_condition_only,
            pad_to_canvas,
            vae_tiling,
            resolution_multiple,
            vae_ref_megapixels,
            decoder_vae,
            keep_decoder_2x,
            highlight_protection,
            highlight_protection_strength,
        ],
        outputs=[output_image, seed, last_output],
    )

    # Output routing buttons
    btn_out_to_img1.click(
        fn=set_output_as_image1, inputs=[last_output], outputs=[input_image_1]
    )
    btn_out_to_img2.click(
        fn=set_output_as_image2, inputs=[last_output], outputs=[input_image_2]
    )
    btn_out_to_extra.click(
        fn=set_output_as_extra,
        inputs=[last_output, input_images_extra],
        outputs=[input_images_extra],
    )

    # Derived conditioning: append pose/depth map as extra ref (UI shows preview)
    add_derived_btn.click(
        fn=add_derived_ref,
        inputs=[input_image_1, input_images_extra, derived_type, derived_use_gpu],
        outputs=[input_images_extra, derived_preview],
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(
        css=css,
        theme=orange_red_theme,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
    )
