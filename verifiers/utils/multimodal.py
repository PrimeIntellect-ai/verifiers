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
            "Multimodal training requires a renderers version with raw image asset offload support."
        ) from exc

    return offload_image_to_run_assets(url, image_dir=image_dir)


def _part_image_field(part_type: object) -> str | None:
    """The field carrying a content part's image source, keyed by ``type``.

    Mirrors the renderer-side part treaty (``renderers.qwen3_vl._image_source``):
    ``image_url`` parts nest the URL under ``image_url`` (``{"url": ...}`` or a
    direct string), ``image`` parts carry the URL string directly.
    """
    if part_type in ("image_url", "image"):
        return str(part_type)
    return None


def _get_field(container: Any, name: str) -> object:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _set_field(container: Any, name: str, value: str) -> None:
    if isinstance(container, dict):
        container[name] = value
    else:
        setattr(container, name, value)


def _prepare_image_part(part: Any, field: str, *, image_dir: Path | None) -> None:
    """Offload one image part's source to run assets and require ``file://``."""
    source = _get_field(part, field)
    if source is None:
        return
    if isinstance(source, str):
        container, key = part, field
    elif field == "image_url":
        container, key = source, "url"
    else:
        raise RuntimeError(
            f"multimodal training requires string image sources; got {type(source).__name__} under {field!r}"
        )

    url = _get_field(container, key)
    offloaded = _offload_image_url(url, image_dir)
    if offloaded is not None:
        _set_field(container, key, offloaded)
        url = offloaded
    if not isinstance(url, str) or not url.startswith("file://"):
        raise RuntimeError(
            "multimodal training requires image sources offloaded to file:// "
            f"run image assets; got {url!r} under {field!r}"
        )


def prepare_images_inplace(value: Any, *, image_dir: Path | None = None) -> None:
    """Offload image URLs reachable from ``value`` to run image assets.

    Handles OpenAI wire dicts/lists and the pydantic v0/v1 message/content-part
    models used by trajectories and traces.
    """
    if isinstance(value, dict):
        field = _part_image_field(value.get("type"))
        if field is not None:
            _prepare_image_part(value, field, image_dir=image_dir)
        for child in value.values():
            prepare_images_inplace(child, image_dir=image_dir)
        return

    if isinstance(value, (list, tuple)):
        for child in value:
            prepare_images_inplace(child, image_dir=image_dir)
        return

    field = _part_image_field(getattr(value, "type", None))
    if field is not None:
        _prepare_image_part(value, field, image_dir=image_dir)
        return

    content = getattr(value, "content", None)
    if isinstance(content, (list, tuple)):
        prepare_images_inplace(content, image_dir=image_dir)
