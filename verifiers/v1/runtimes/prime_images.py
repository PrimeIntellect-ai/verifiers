"""Build a Docker context into a Prime VM image, cached by content digest.

``PrimeConfig.image`` boots any pullable ref, but a task whose environment is a
Dockerfile needs that Dockerfile built into the Prime registry first. ``ensure_image``
tags the image with a digest over the build context, so the same context is built once
and every later call (another episode, another launch, another host) reuses it: it returns
as soon as the registry lists the tag's VM artifact COMPLETED, waits on a build already
running, and otherwise uploads the context and waits for the platform's build.

A sandbox boots from the image's VM_SANDBOX artifact, so that artifact alone decides
reuse and readiness. Concurrent calls for one image in a process share one build.

Every call works in the configured team's registry (``$PRIME_TEAM_ID``, else the prime
config's ``team_id``; personal when neither is set), the scope sandboxes are created in.
The build, the listing and the delete all name it: a build pushed without it lands in the
personal registry while the listing searches the team's, and never shows up.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PACKAGED_DOCKERFILE = ".__prime_dockerfile__"
"""Where the Dockerfile sits in the uploaded context (the prime CLI's convention), so it
cannot collide with a file of the context."""

BUILD_TIMEOUT_SECONDS = 2_400
"""How long one image may take to become bootable."""

POLL_SECONDS = 5.0
_LIST_PAGE = 250
_UPLOAD_TIMEOUT_SECONDS = 600.0
_FAILED = frozenset({"FAILED", "CANCELLED"})
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")

_building: dict[
    tuple[asyncio.AbstractEventLoop, str | None, str, str], asyncio.Future[PrimeImage]
] = {}
"""The build each (loop, team, name, tag) is waiting on, shared by concurrent callers."""


class ImageBuildError(RuntimeError):
    """The platform failed the build, or it did not finish within its deadline."""


class PrimeImage(BaseModel):
    """One image tag and the state of its VM artifact."""

    name: str
    tag: str
    ref: str
    """What ``PrimeConfig.image`` takes: ``prime/<owner>/<name>:<tag>``."""
    status: Literal[
        "PENDING", "UPLOADING", "BUILDING", "COMPLETED", "FAILED", "CANCELLED"
    ]


def context_digest(
    context: Path, dockerfile: Path | None = None, salt: str = ""
) -> str:
    """A sha256 over everything a build reads: ``salt``, the Dockerfile, and every
    file of ``context`` (relative path, executable bit, bytes; a symlink's target).
    Refuses a context with a ``.dockerignore``: what it would exclude is not hashed
    the way the builder applies it, so assemble the context explicitly."""
    context = Path(context)
    dockerfile = Path(dockerfile) if dockerfile else context / "Dockerfile"
    if (context / ".dockerignore").exists():
        raise ValueError(
            f"{context} has a .dockerignore; assemble a context without one instead"
        )
    h = hashlib.sha256()
    h.update(salt.encode() + b"\0" + dockerfile.read_bytes() + b"\0")
    for path in sorted(
        context.rglob("*"), key=lambda p: p.relative_to(context).as_posix()
    ):
        rel = path.relative_to(context).as_posix()
        if path.is_symlink():
            h.update(f"L {rel} {os.readlink(path)}\0".encode())
        elif path.is_file():
            h.update(f"F {rel} {os.access(path, os.X_OK):d}\0".encode())
            h.update(path.read_bytes() + b"\0")
        elif path.is_dir():
            h.update(f"D {rel}\0".encode())
    return h.hexdigest()


def digest_tag(digest: str) -> str:
    return f"sha-{digest[:24]}"


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
    return PrimeImage(name=name, tag=tag, ref=ref, status=status)


def _package(context: Path, dockerfile: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        archive = Path(tmp.name)
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(context, arcname=".")
            tar.add(dockerfile, arcname=PACKAGED_DOCKERFILE)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive


async def _push(
    client,
    context: Path,
    dockerfile: Path,
    name: str,
    tag: str,
    visibility: str,
    team_id: str | None,
) -> None:
    import httpx
    from prime_sandboxes import BuildImageRequest

    archive = await asyncio.to_thread(_package, context, dockerfile)
    try:
        response = await client.initiate_build(
            BuildImageRequest(
                image_name=name,
                image_tag=tag,
                dockerfile_path=PACKAGED_DOCKERFILE,
                visibility=visibility,
                team_id=team_id,
            )
        )
        async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SECONDS) as http:
            upload = await http.put(
                response.upload_url,
                content=archive.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            upload.raise_for_status()
        await client.start_build(response.build_id)
    finally:
        archive.unlink(missing_ok=True)
    logger.info(
        "prime image %s:%s: build %s started (team %s)",
        name,
        tag,
        response.build_id,
        team_id or "personal",
    )


async def _ensure(
    context: Path,
    dockerfile: Path,
    name: str,
    tag: str,
    visibility: str,
    timeout: float,
    team_id: str | None,
) -> PrimeImage:
    from prime_sandboxes import AsyncImageClient

    async with AsyncImageClient() as client:
        image = await find_image(client, name, tag)
        if image is not None and image.status == "COMPLETED":
            logger.info("prime image %s reused", image.ref)
            return image
        if image is None or image.status in _FAILED:
            await _push(client, context, dockerfile, name, tag, visibility, team_id)
        else:
            logger.info(
                "prime image %s:%s already %s; waiting", name, tag, image.status
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            image = await find_image(client, name, tag)
            if image is not None and image.status == "COMPLETED":
                logger.info("prime image %s ready", image.ref)
                return image
            if image is not None and image.status in _FAILED:
                raise ImageBuildError(f"prime image {name}:{tag} build {image.status}")
            if loop.time() >= deadline:
                raise ImageBuildError(
                    f"prime image {name}:{tag} not bootable within {timeout:.0f}s "
                    f"(VM artifact: {image.status if image else 'absent'})"
                )
            await asyncio.sleep(POLL_SECONDS)


async def ensure_image(
    context: Path,
    name: str,
    *,
    dockerfile: Path | None = None,
    tag: str | None = None,
    salt: str = "",
    visibility: Literal["PRIVATE", "PUBLIC"] = "PRIVATE",
    timeout: float = BUILD_TIMEOUT_SECONDS,
) -> PrimeImage:
    """Build ``context`` (with ``dockerfile``, default ``context/Dockerfile``) into the
    Prime image ``name:tag`` unless the registry already has it bootable; return it once
    its VM artifact is COMPLETED. ``tag`` defaults to ``sha-<digest>`` over the context
    and ``salt`` (``context_digest``). It lives in the configured team's registry
    (``configured_team_id``). A failed earlier build of the tag is rebuilt; a
    build that fails now raises ``ImageBuildError``. The build runs to its end even when
    the caller is cancelled, so a later call finds it instead of starting another."""
    context = Path(context)
    dockerfile = Path(dockerfile) if dockerfile else context / "Dockerfile"
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid image name {name!r}: lowercase [a-z0-9._-]")
    tag = tag or digest_tag(
        await asyncio.to_thread(context_digest, context, dockerfile, salt)
    )
    if not _TAG.fullmatch(tag):
        raise ValueError(f"invalid image tag {tag!r}")
    team_id = configured_team_id()
    key = (asyncio.get_running_loop(), team_id, name, tag)
    task = _building.get(key)
    if task is None:
        task = asyncio.ensure_future(
            _ensure(context, dockerfile, name, tag, visibility, timeout, team_id)
        )
        _building[key] = task

        def settle(done: asyncio.Future[PrimeImage]) -> None:
            _building.pop(key, None)
            if not done.cancelled():
                done.exception()  # retrieved: every waiter may have been cancelled

        task.add_done_callback(settle)
    return await asyncio.shield(task)


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
