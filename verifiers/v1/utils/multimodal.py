"""Multimodal ingress helpers for v1 training.

The env worker stores raw images as run assets before messages enter the trace.
Messages then carry cheap ``file://`` refs, and the renderer/vLLM path decides
which refs must be sent to inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageOffloadStats:
    images_rewritten: int = 0
    bytes_written: int = 0

    def add(self, other: "ImageOffloadStats") -> None:
        self.images_rewritten += other.images_rewritten
        self.bytes_written += other.bytes_written


def _offload_image_url(url: object, image_dir: Path | None) -> tuple[str, int] | None:
    try:
        from renderers.mm_store import offload_image_to_run_assets
    except ImportError as exc:  # pragma: no cover - dependency-version guard
        raise RuntimeError(
            "Multimodal training requires a renderers version with raw image "
            "asset offload support."
        ) from exc

    return offload_image_to_run_assets(url, image_dir=image_dir)


def _image_source_url(source: Any) -> object:
    if isinstance(source, dict):
        return source.get("url")
    return getattr(source, "url", None)


def _set_image_source_url(source: Any, url: str) -> None:
    if isinstance(source, dict):
        source["url"] = url
    else:
        source.url = url


def _require_file_image_url(source: Any) -> None:
    url = _image_source_url(source)
    if not isinstance(url, str) or not url.startswith("file://"):
        raise RuntimeError(
            "v1 multimodal training requires image_url entries to be offloaded "
            "to file:// run image assets"
        )


def _rewrite_image_source(source: Any, image_dir: Path | None) -> ImageOffloadStats:
    stats = ImageOffloadStats()

    result = _offload_image_url(_image_source_url(source), image_dir)
    if result is not None:
        new_url, nbytes = result
        _set_image_source_url(source, new_url)
        stats.images_rewritten += 1
        stats.bytes_written += nbytes
    _require_file_image_url(source)
    return stats


def offload_images_inplace(
    value: Any, *, image_dir: Path | None = None
) -> ImageOffloadStats:
    """Rewrite base64 image URLs reachable from ``value`` to run-asset refs.

    Handles OpenAI wire dicts/lists and the pydantic v1 message/content-part
    models used by the trace. Non-image values and already-file-backed URLs are
    left untouched.
    """
    stats = ImageOffloadStats()

    if isinstance(value, dict):
        if value.get("type") == "image_url":
            source = value.get("image_url")
            if source is not None:
                stats.add(_rewrite_image_source(source, image_dir))
        for child in value.values():
            stats.add(offload_images_inplace(child, image_dir=image_dir))
        return stats

    if isinstance(value, list):
        for child in value:
            stats.add(offload_images_inplace(child, image_dir=image_dir))
        return stats

    if isinstance(value, tuple):
        for child in value:
            stats.add(offload_images_inplace(child, image_dir=image_dir))
        return stats

    if getattr(value, "type", None) == "image_url":
        source = getattr(value, "image_url", None)
        if source is not None:
            stats.add(_rewrite_image_source(source, image_dir))
        return stats

    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        stats.add(offload_images_inplace(content, image_dir=image_dir))

    return stats
