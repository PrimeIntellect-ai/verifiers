"""Image -> ascii/braille rendering for text-only vision.

Rendering an image to character art is lossy and hard for models (ASCII-Eval, arXiv
2410.01733: frontier models reach ~70% concept recognition). Textify is not a transparent
vision<->text equivalence; it turns vision tasks into deliberately hard text-space
perception tasks, and lets text-only models attempt vision environments at all.

`image_to_text` is the pure render core (numpy, deterministic, no fence). `textify_messages`
applies it to typed `Messages` (data-URI images only). The interception server applies the
same rendering to native request bodies via `Dialect.textify_body`, which is what makes
`--textify.enabled true` work for any environment without env cooperation.

Token economics: the default ascii ramp is single-byte characters that tokenize cheaply.
Braille cells pack 2x4 dots per character but each codepoint is 3 UTF-8 bytes and often
tokenizes to multiple tokens; measure before assuming braille is denser per token.
"""

import base64
import binascii
import io
import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import Field, field_validator
from pydantic_config import BaseConfig

from verifiers.v1.types import (
    ImageUrlContentPart,
    Message,
    Messages,
    TextContentPart,
)

logger = logging.getLogger(__name__)


_warned_passthrough = False


def _warn_passthrough(url: str) -> None:
    global _warned_passthrough
    if not _warned_passthrough:
        logger.warning(
            "textify: non-data image URLs pass through unrendered (first: %s)", url
        )
        _warned_passthrough = True


# Rec.601 luminance weights.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Braille dot bit positions: cell[row][col] -> bit index in U+2800..U+28FF.
_BRAILLE_BITS = np.array([[0, 3], [1, 4], [2, 5], [6, 7]], dtype=np.uint8)
_MAX_IMAGE_BYTES = 100_000_000


class TextifyConfig(BaseConfig):
    """How images on the model wire are rendered to text. Disabled by default; enabling
    uses ascii unless `mode="braille"` is selected. Images are replaced by fenced text
    renderings in place."""

    enabled: bool = False
    mode: Literal["ascii", "braille"] = "ascii"
    width: int = Field(160, ge=1, le=4096)
    """Output columns."""
    height: int | None = Field(None, ge=1, le=4096)
    """Output character rows; `None` derives it from image and character aspect ratios."""
    char_aspect: float = Field(0.5, gt=0)
    """Character cell height/width ratio correction; 0.5 suits typical monospace fonts."""
    gamma: float = Field(1.0, gt=0)
    """Brightness curve applied to luminance (`lum ** gamma`); >1 darkens midtones."""
    invert: bool | None = None
    """Map light pixels to dense glyphs (False), sparse glyphs (True), or auto (None):
    a predominantly light image — dark ink on white, the common diagram/document case —
    is inverted so ink gets the glyphs and background the blanks."""
    ramp: str = Field(" .:-=+*#%@", min_length=2)
    """Ascii glyphs from dark to light before optional inversion."""
    threshold: float | Literal["otsu"] = 0.5
    """Braille cutoff, or `otsu` for automatic global binarization of ASCII/Braille."""
    max_chars: int | None = Field(40_000, ge=1, le=1_000_000)
    """Hard character budget per image; width is clamped to fit."""

    @field_validator("char_aspect", "gamma")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold_range(cls, value: float | str) -> float | str:
        if isinstance(value, float) and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("threshold must be between 0 and 1")
        return value


def _grid_shape(h: int, w: int, cfg: TextifyConfig) -> tuple[int, int]:
    """The (rows, cols) character grid for source dimensions (>=1 each)."""
    if h <= 0 or w <= 0:
        raise ValueError("image dimensions must be nonzero")
    width = cfg.width
    height = cfg.height or max(1, round(width * (h / w) * cfg.char_aspect))
    output_chars = width * height + height - 1
    if cfg.max_chars is not None and output_chars > cfg.max_chars:
        scale = (cfg.max_chars / output_chars) ** 0.5
        width = max(1, int(width * scale))
        if cfg.height is None:
            height = max(1, round(width * (h / w) * cfg.char_aspect))
        else:
            height = max(1, int(height * scale))
        while width * height + height - 1 > cfg.max_chars:
            if width >= height and width > 1:
                width -= 1
            elif height > 1:
                height -= 1
            else:
                break
    if width * height + height - 1 > 1_000_000:
        raise ValueError(
            "textify output exceeds the one-million-character safety limit"
        )
    return height, width


def _decode(image: bytes | np.ndarray, cfg: TextifyConfig) -> np.ndarray:
    """Encoded image bytes or an (H, W[, C]) array -> sampled RGB uint8."""
    if isinstance(image, bytes):
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("textify needs pillow to decode encoded images") from e
        with Image.open(io.BytesIO(image)) as img:
            if img.width * img.height > 25_000_000:
                raise ValueError("textify image exceeds the 25-megapixel safety limit")
            rows, cols = _grid_shape(img.height, img.width, cfg)
            if cfg.mode == "braille":
                rows, cols = rows * 4, cols * 2
            # Resize before RGBA conversion/float work: a 4K source becomes only the output grid.
            img = img.resize((cols, rows), Image.Resampling.NEAREST).convert("RGBA")
            arr = np.asarray(img, dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3 or arr.shape[2] not in (1, 2, 3, 4):
            raise ValueError("image array must have shape (H, W), (H, W, 1..4)")
        if not np.issubdtype(arr.dtype, np.unsignedinteger):
            raise ValueError("image array must use an unsigned integer dtype")
        if arr.shape[0] * arr.shape[1] > 25_000_000:
            raise ValueError("textify image exceeds the 25-megapixel safety limit")
        rows, cols = _grid_shape(arr.shape[0], arr.shape[1], cfg)
        if cfg.mode == "braille":
            rows, cols = rows * 4, cols * 2
        ys = np.linspace(0, arr.shape[0], rows, endpoint=False).astype(np.intp)
        xs = np.linspace(0, arr.shape[1], cols, endpoint=False).astype(np.intp)
        arr = arr[np.ix_(ys, xs)]
        source_dtype = arr.dtype
        if source_dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr *= 255.0 / np.iinfo(source_dtype).max
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 2:
            arr = np.concatenate(
                [np.repeat(arr[..., :1], 3, axis=2), arr[..., 1:]], axis=2
            )
    if arr.shape[2] == 3:
        return arr.astype(np.uint8)
    # Transparent image backgrounds conventionally render white in documents/diagrams.
    rgba = arr.astype(np.float32)
    alpha = rgba[..., 3:4] / 255.0
    return (rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def _luminance(img: np.ndarray, cfg: TextifyConfig) -> np.ndarray:
    """Reduce the target-grid image to gamma-corrected luminance in [0, 1]."""
    lum = img.astype(np.float32) @ _LUMA / 255.0
    if cfg.gamma != 1.0:
        lum **= cfg.gamma
    if cfg.invert or (cfg.invert is None and float(lum.mean()) > 0.5):
        lum = 1.0 - lum
    return lum


def _otsu_threshold(lum: np.ndarray) -> float:
    """Classic global Otsu threshold over luminance in [0, 1]."""
    values = np.clip((lum * 255).round(), 0, 255).astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    if np.count_nonzero(hist) <= 1:
        return 0.5
    levels = np.arange(256, dtype=np.float64)
    left_weight = np.cumsum(hist)
    right_weight = hist.sum() - left_weight
    left_sum = np.cumsum(hist * levels)
    right_sum = left_sum[-1] - left_sum
    valid = (left_weight > 0) & (right_weight > 0)
    between = np.full(256, -1.0)
    delta = np.zeros(256)
    delta[valid] = (
        left_sum[valid] / left_weight[valid] - right_sum[valid] / right_weight[valid]
    )
    between[valid] = left_weight[valid] * right_weight[valid] * delta[valid] ** 2
    split = int(np.argmax(between))
    return (split + 0.5) / 255.0


def image_to_text(image: bytes | np.ndarray, cfg: TextifyConfig) -> str:
    """Render an image to multi-line character art (no fence). Deterministic in
    (image, cfg). Accepts encoded bytes (any Pillow format) or an RGB(A)/grayscale
    uint8 array."""
    # Encoded and array images are decoded/sampled directly at the target grid size.
    img = _decode(image, cfg)
    factor = (4, 2) if cfg.mode == "braille" else (1, 1)
    rows, cols = img.shape[0] // factor[0], img.shape[1] // factor[1]
    if cfg.mode == "braille":
        lum = _luminance(img, cfg)
        threshold = _otsu_threshold(lum) if cfg.threshold == "otsu" else cfg.threshold
        dots = lum >= threshold
        codes = np.zeros((rows, cols), dtype=np.uint16)
        for row in range(4):
            for col in range(2):
                bits = dots[row::4, col::2].astype(np.uint16)
                codes |= bits << _BRAILLE_BITS[row, col]
        return "\n".join("".join(chr(0x2800 + int(c)) for c in line) for line in codes)
    lum = _luminance(img, cfg)
    if cfg.threshold == "otsu":
        lum = (lum >= _otsu_threshold(lum)).astype(np.float32)
    idx = np.clip((lum * (len(cfg.ramp) - 1)).round(), 0, len(cfg.ramp) - 1)
    glyphs = np.array(list(cfg.ramp))
    return "\n".join("".join(line) for line in glyphs[idx.astype(np.intp)])


def data_url_bytes(url: str) -> bytes | None:
    """The decoded payload of a base64 `data:` URL, else None for non-data URLs.
    Malformed/unsupported data URLs raise so the rollout records the bad image."""
    if not url.startswith("data:"):
        return None
    head, sep, payload = url.partition(",")
    if not sep or not head.startswith("data:image/") or not head.endswith(";base64"):
        raise ValueError("textify supports base64 image data URLs only")
    if len(payload) > (_MAX_IMAGE_BYTES * 4 + 2) // 3:
        raise ValueError("textify image payload exceeds the byte safety limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except binascii.Error as e:
        raise ValueError("invalid base64 image data URL") from e
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("textify image payload exceeds the byte safety limit")
    return data


def render_url(url: str, cfg: TextifyConfig) -> str | None:
    """Render a data-URL image to fenced art, or None to leave the URL untouched.
    Plain http(s) URLs pass through: the interception hot path must not fetch."""
    data = data_url_bytes(url)
    if data is None:
        _warn_passthrough(url)
        return None
    art = image_to_text(data, cfg)
    return f"```image[{cfg.mode}]\n{art}\n```"


def textify_messages(
    messages: Messages,
    cfg: TextifyConfig,
    render: Callable[[str], str | None] | None = None,
) -> Messages:
    """Replace each data-URI image part with its fenced rendering, in place in the
    content structure. Identity when disabled. Non-image parts, `str` bodies, and
    non-data image URLs pass through unchanged. `render` lets interception reuse its
    per-rollout image cache; public callers default to direct rendering."""
    if not cfg.enabled:
        return messages
    render = render or (lambda url: render_url(url, cfg))
    out: list[Message] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list) or not any(
            isinstance(p, ImageUrlContentPart) for p in content
        ):
            out.append(message)
            continue
        parts = []
        for part in content:
            text = (
                render(part.image_url.url)
                if isinstance(part, ImageUrlContentPart)
                else None
            )
            parts.append(TextContentPart(text=text) if text is not None else part)
        out.append(message.model_copy(update={"content": parts}))
    return out
