"""Multimodal ingress: canonicalize image parts to
``{"type": "image_url", "image_url": {"url": ...}}`` with the URL offloaded
to a ``file://`` run image asset."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def _offload_image_url(url: str, image_dir: Path | None) -> str | None:
    try:
        offload_image_to_run_assets = import_module(
            "renderers.mm_store"
        ).offload_image_to_run_assets
    except (
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover - dependency-version guard
        raise RuntimeError(
            "Multimodal training requires a renderers version with raw image asset offload support."
        ) from exc

    return offload_image_to_run_assets(url, image_dir=image_dir)


def _prepare_image_part(part: dict[str, Any], *, image_dir: Path | None) -> None:
    """Rewrite one image part to the canonical shape with an offloaded URL."""
    if part.get("type") == "image":  # HF-style: URL directly under ``image``
        part["type"] = "image_url"
        part["image_url"] = {"url": part.pop("image", None)}
    image_url = part.get("image_url")
    if not isinstance(image_url, dict):
        raise TypeError(
            "v1 multimodal training requires the OpenAI image part shape "
            f"{{'image_url': {{'url': ...}}}}; got image_url of type {type(image_url).__name__}"
        )
    url = image_url.get("url")
    if not isinstance(url, str):
        raise TypeError(
            f"v1 multimodal training requires string image URLs; got {url!r}"
        )
    if url.startswith("file://"):
        return
    offloaded = _offload_image_url(url, image_dir)
    if offloaded is None:
        raise RuntimeError(
            "v1 multimodal training accepts data:image/...;base64 or file:// "
            f"image sources; got {url.split(',', 1)[0]!r}"
        )
    image_url["url"] = offloaded


def prepare_images_inplace(value: Any, *, image_dir: Path | None = None) -> None:
    """Rewrite every image part reachable from a request body to the
    canonical shape with a ``file://`` URL; reject unsupported sources."""
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url"):
            _prepare_image_part(value, image_dir=image_dir)
            return
        for child in value.values():
            prepare_images_inplace(child, image_dir=image_dir)
    elif isinstance(value, (list, tuple)):
        for child in value:
            prepare_images_inplace(child, image_dir=image_dir)
