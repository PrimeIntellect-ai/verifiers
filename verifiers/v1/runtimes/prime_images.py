"""Find and delete Prime VM images in the configured team's registry.

Every call works in the configured team's registry (``$PRIME_TEAM_ID``, else the prime
config's ``team_id``; personal when neither is set), the scope sandboxes are created in:
an image pushed without it lands in the personal registry while a team listing searches
the team's, and never shows up. A sandbox boots from an image's VM_SANDBOX artifact, so
that artifact alone says whether the image is there and bootable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

_LIST_PAGE = 250
_FAILED = frozenset({"FAILED", "CANCELLED"})


class PrimeImage(BaseModel):
    """One image tag and the state of its VM artifact."""

    name: str
    tag: str
    ref: str
    """What ``PrimeConfig.image`` takes: ``prime/<owner>/<name>:<tag>``."""
    status: Literal[
        "PENDING", "UPLOADING", "BUILDING", "COMPLETED", "FAILED", "CANCELLED"
    ]


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
