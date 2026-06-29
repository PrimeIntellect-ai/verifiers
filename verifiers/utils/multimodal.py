"""Multimodal ingress helpers for renderer-backed training."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def _offload_image_url(url: object, image_dir: Path | None) -> str | None:
    try:
        offload_image_to_run_assets = getattr(
            import_module("renderers.mm_store"),
            "offload_image_to_run_assets",
        )
    except (
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover - dependency-version guard
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
            "multimodal training requires image_url entries to be offloaded "
            "to file:// run image assets"
        )


def _prepare_image_source(source: Any, *, image_dir: Path | None) -> None:
    result = _offload_image_url(_image_source_url(source), image_dir)
    if result is not None:
        _set_image_source_url(source, result)
    _require_file_image_url(source)


def prepare_images_inplace(value: Any, *, image_dir: Path | None = None) -> None:
    """Offload image URLs reachable from ``value`` to run image assets.

    Handles OpenAI wire dicts/lists and the pydantic v0/v1 message/content-part
    models used by trajectories and traces.
    """
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            source = value.get("image_url")
            if source is not None:
                _prepare_image_source(source, image_dir=image_dir)
        for child in value.values():
            prepare_images_inplace(child, image_dir=image_dir)
        return

    if isinstance(value, list):
        for child in value:
            prepare_images_inplace(child, image_dir=image_dir)
        return

    if isinstance(value, tuple):
        for child in value:
            prepare_images_inplace(child, image_dir=image_dir)
        return

    if getattr(value, "type", None) == "image_url":
        source = getattr(value, "image_url", None)
        if source is not None:
            _prepare_image_source(source, image_dir=image_dir)
        return

    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        prepare_images_inplace(content, image_dir=image_dir)
