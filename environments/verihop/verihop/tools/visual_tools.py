from __future__ import annotations

from typing import Any

from PIL import Image


def crop_region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    _pil_image: Image.Image | None = None,
) -> str:
    """
    Crop the rollout image to integer pixel bounds (inclusive-exclusive width/height
    semantics: from (x0,y0) to (x1,y1)). Returns a short text summary; the agent
    should use this to focus reasoning. ``_pil_image`` is injected by the environment.
    """
    if _pil_image is None:
        return "crop_region: no image available in this rollout."
    w, h = _pil_image.size
    x0c = max(0, min(w, min(x0, x1)))
    x1c = max(0, min(w, max(x0, x1)))
    y0c = max(0, min(h, min(y0, y1)))
    y1c = max(0, min(h, max(y0, y1)))
    if x1c <= x0c or y1c <= y0c:
        return "crop_region: empty region after clamping."
    crop = _pil_image.crop((x0c, y0c, x1c, y1c))
    stat = crop.resize((32, 32)).getcolors(maxcolors=1024)
    ncolors = len(stat) if stat else 0
    return (
        f"crop_region: size=({x1c - x0c}x{y1c - y0c}) "
        f"approx_distinct_colors={ncolors} (32x32 sample)."
    )


def zoom_center(
    factor_percent: int = 150,
    _pil_image: Image.Image | None = None,
) -> str:
    """
    Conceptually zoom the center of the image by ``factor_percent`` (100 = original).
    Returns a coarse summary of the central patch. ``_pil_image`` is injected.
    """
    if _pil_image is None:
        return "zoom_center: no image available."
    w, h = _pil_image.size
    f = max(50, min(300, int(factor_percent))) / 100.0
    cw, ch = int(w / f), int(h / f)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - cw // 2)
    y0 = max(0, cy - ch // 2)
    x1 = min(w, x0 + cw)
    y1 = min(h, y0 + ch)
    patch = _pil_image.crop((x0, y0, x1, y1))
    extrema = patch.getextrema()
    return f"zoom_center: patch bbox=({x0},{y0},{x1},{y1}) extrema={extrema}"


def count_color_blobs(
    color: str = "red",
    _pil_image: Image.Image | None = None,
) -> str:
    """
    Very rough blob count hint for red-ish or blue-ish pixels (debug-style heuristic,
    not ground truth). ``_pil_image`` is injected.
    """
    if _pil_image is None:
        return "count_color_blobs: no image."
    rgb = _pil_image.convert("RGB")
    w, h = rgb.size
    data = list(rgb.getdata())
    c = color.lower()
    n = 0
    for r, g, b in data:
        if c == "red" and r > g + 40 and r > b + 40:
            n += 1
        elif c == "blue" and b > r + 30 and b > g + 20:
            n += 1
    ratio = n / max(1, w * h)
    return f"count_color_blobs: ~{n} {c}-leaning pixels ({ratio:.3%} of image)."


def make_visual_tools() -> list[Any]:
    return [crop_region, zoom_center, count_color_blobs]
