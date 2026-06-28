"""Multimodal ingress helpers for v1 training."""

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


def _image_storage_mode(storage: str | None) -> str:
    if storage is not None:
        mode = storage
    else:
        from renderers.mm_store import image_storage_mode

        mode = image_storage_mode()
    if mode not in ("offload", "inline"):
        raise ValueError(
            f"multimodal image storage must be 'offload' or 'inline', got {mode!r}"
        )
    return mode


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


def _require_inline_image_url(source: Any) -> None:
    url = _image_source_url(source)
    if not isinstance(url, str):
        raise RuntimeError(
            "v1 inline multimodal training requires image_url.url strings"
        )
    if url.startswith("file://"):
        return
    if url.startswith("data:image/") and ";base64," in url:
        return
    raise RuntimeError(
        "v1 inline multimodal training requires image_url entries to be "
        "data:image/...;base64,... or file:// refs"
    )


def _prepare_image_source(
    source: Any, *, storage: str, image_dir: Path | None
) -> ImageOffloadStats:
    stats = ImageOffloadStats()

    if storage == "offload":
        result = _offload_image_url(_image_source_url(source), image_dir)
        if result is not None:
            new_url, nbytes = result
            _set_image_source_url(source, new_url)
            stats.images_rewritten += 1
            stats.bytes_written += nbytes
        _require_file_image_url(source)
    else:
        _require_inline_image_url(source)
    return stats


def prepare_images_inplace(
    value: Any, *, storage: str | None = None, image_dir: Path | None = None
) -> ImageOffloadStats:
    """Prepare image URLs reachable from ``value`` according to the storage mode.

    Handles OpenAI wire dicts/lists and the pydantic v1 message/content-part
    models used by the trace.
    """
    mode = _image_storage_mode(storage)
    stats = ImageOffloadStats()

    if isinstance(value, dict):
        if value.get("type") == "image_url":
            source = value.get("image_url")
            if source is not None:
                stats.add(
                    _prepare_image_source(source, storage=mode, image_dir=image_dir)
                )
        for child in value.values():
            stats.add(prepare_images_inplace(child, storage=mode, image_dir=image_dir))
        return stats

    if isinstance(value, list):
        for child in value:
            stats.add(prepare_images_inplace(child, storage=mode, image_dir=image_dir))
        return stats

    if isinstance(value, tuple):
        for child in value:
            stats.add(prepare_images_inplace(child, storage=mode, image_dir=image_dir))
        return stats

    if getattr(value, "type", None) == "image_url":
        source = getattr(value, "image_url", None)
        if source is not None:
            stats.add(_prepare_image_source(source, storage=mode, image_dir=image_dir))
        return stats

    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        stats.add(prepare_images_inplace(content, storage=mode, image_dir=image_dir))

    return stats


def offload_images_inplace(
    value: Any, *, image_dir: Path | None = None
) -> ImageOffloadStats:
    return prepare_images_inplace(value, storage="offload", image_dir=image_dir)
