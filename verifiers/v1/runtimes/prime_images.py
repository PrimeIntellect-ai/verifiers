"""Push, find and delete Prime VM images in the configured team's registry.

Every call works in the configured team's registry (``$PRIME_TEAM_ID``, else the prime
config's ``team_id``; personal when neither is set), the scope sandboxes are created in:
an image pushed without it lands in the personal registry while a team listing searches
the team's, and never shows up. A sandbox boots from an image's VM_SANDBOX artifact, so
that artifact alone says whether the image is there and bootable.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import tarfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

_LIST_PAGE = 250
_FAILED = frozenset({"FAILED", "CANCELLED"})
MAX_TAG = 48
"""The registry cuts longer tags, so the image would not be found under its own."""
POLL = 10.0
"""Seconds between registry reads while an image builds; the registry offers no wait."""


class PrimeImage(BaseModel):
    """One image tag and the state of its VM artifact."""

    name: str
    tag: str
    ref: str
    """What ``PrimeConfig.image`` takes: ``prime/<owner>/<name>:<tag>``."""
    status: Literal[
        "PENDING", "UPLOADING", "BUILDING", "COMPLETED", "FAILED", "CANCELLED"
    ]
    error: str | None = None
    """What the registry reports about a failed build, when it reports anything."""


class ImageBuildError(RuntimeError):
    """An image did not become bootable."""


def configured_team_id() -> str | None:
    """The team the prime SDK creates sandboxes in: ``$PRIME_TEAM_ID`` (empty means
    personal), else the prime config's ``team_id``; None for the personal scope."""
    from prime_sandboxes.core import Config

    return Config().team_id


async def find_image(client, name: str, tag: str) -> PrimeImage | None:
    """The VM artifact of ``name:tag`` in the configured team's registry, or None when
    it was never pushed. Several rows (a re-push) resolve to the best."""
    team_id = configured_team_id()
    rows = []
    offset = 0
    while True:
        page = await client.list(
            # "" keeps the SDK from falling back to the configured team
            team_id=team_id or "",
            search=name,
            offset=offset,
            limit=_LIST_PAGE,
        )
        rows += [
            row
            for row in page.data
            if row.image_name == name
            and row.image_tag == tag
            and row.artifact_type.value == "VM_SANDBOX"
        ]
        offset += page.limit
        if page.total_count is None or offset >= page.total_count or not page.data:
            break
    if not rows:
        return None
    statuses = {row.status.value for row in rows}
    status = (
        "COMPLETED"
        if "COMPLETED" in statuses
        else next((s for s in statuses if s not in _FAILED), min(statuses))
    )
    ref = next(
        (row.display_ref for row in rows if row.display_ref),
        next(
            (row.full_image_path for row in rows if row.full_image_path),
            f"{name}:{tag}",
        ),
    )
    error = next(
        (row.error_message for row in rows if status in _FAILED and row.error_message),
        None,
    )
    return PrimeImage(name=name, tag=tag, ref=ref, status=status, error=error)


def context_digest(context: Path) -> str:
    """A digest of every file's path and bytes under `context`, for a tag that changes
    exactly when the build context does."""
    digest = hashlib.sha256()
    for path in sorted(p for p in context.rglob("*") if p.is_file()):
        digest.update(
            str(path.relative_to(context)).encode() + b"\0" + path.read_bytes()
        )
    return digest.hexdigest()


Upload = Callable[[str], Awaitable[None]]
"""Puts a gzipped tar of a build context to the presigned URL it is given."""


async def push_image(
    context: Path | Upload,
    name: str,
    tag: str,
    *,
    dockerfile: str = "Dockerfile",
    timeout: float = 3600,
    attempts: int = 2,
) -> PrimeImage:
    """Build `context` as the private image `name:tag` in the configured team's
    registry and wait until its VM artifact is bootable. A tag already built is reused
    and one building is waited for; a failed build is pushed again (`attempts` in
    all), since a build that fetches material can fail on a transient download.
    `context` is a local directory, or an `Upload` that puts the build context where
    it already is (a VM that built what the image ships). `dockerfile` is relative to
    the context, whose .dockerignore the build applies. `timeout` bounds each
    attempt's wait."""
    from prime_sandboxes import AsyncImageClient
    from prime_sandboxes.models import BuildImageRequest, ImageVisibility

    if len(tag) > MAX_TAG:
        raise ValueError(f"image tag {tag!r} is longer than {MAX_TAG} characters")
    if isinstance(context, Path):
        if not (context / dockerfile).is_file():
            raise ValueError(f"{context / dockerfile} is not a file")
        upload = _upload_directory(context)
    else:
        upload = context
    loop = asyncio.get_running_loop()
    async with AsyncImageClient() as client:
        image = await find_image(client, name, tag)
        for attempt in range(attempts):
            if image is None or image.status in _FAILED:
                await _upload_build(
                    client,
                    BuildImageRequest(
                        image_name=name,
                        image_tag=tag,
                        dockerfile_path=dockerfile,
                        team_id=configured_team_id(),
                        visibility=ImageVisibility.PRIVATE,
                    ),
                    upload,
                )
                image = None
            deadline = loop.time() + timeout
            # A new build's row can appear only after the build starts.
            while image is None or image.status not in ("COMPLETED", *_FAILED):
                if loop.time() > deadline:
                    raise ImageBuildError(
                        f"image {name}:{tag} not bootable after {timeout:.0f}s"
                    )
                await asyncio.sleep(POLL)
                image = await find_image(client, name, tag)
            if image.status == "COMPLETED":
                return image
    raise ImageBuildError(
        f"image {name}:{tag} failed to build {attempts} times: "
        f"{image.error or 'the registry reports no cause'}"
    )


async def _upload_build(client, request, upload: Upload) -> None:
    build = await client.initiate_build(request)
    await upload(build.upload_url)
    await client.start_build(build.build_id)


def _upload_directory(context: Path) -> Upload:
    async def upload(url: str) -> None:
        import httpx

        archive = await asyncio.to_thread(_pack, context)
        async with httpx.AsyncClient(timeout=600) as http:
            response = await http.put(
                url,
                content=archive,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()

    return upload


def _pack(context: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(context, arcname=".")
    return buffer.getvalue()


async def delete_image(name: str, tag: str) -> bool:
    """Delete ``name:tag`` from the configured team's registry; False when it was not
    there."""
    from prime_sandboxes import APIError, AsyncAPIClient

    team_id = configured_team_id()
    client = AsyncAPIClient()
    try:
        await client.request(
            "DELETE",
            f"/images/{name}/{tag}",
            params={"teamId": team_id} if team_id else None,
        )
    except APIError as e:
        if "HTTP 404" in str(e):
            return False
        raise
    finally:
        await client.aclose()
    return True
