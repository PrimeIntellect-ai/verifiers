"""Delete a run's leftover Prime sandboxes and tunnels by label.

Every Prime sandbox and every Prime tunnel a process opens carries its run labels
(``run_labels``: the process-wide base labels and the run scope, ``$VF_RUN_ID``). A run
that dies without its teardown (SIGKILL, OOM, a host reboot) leaves both behind: a
sandbox keeps billing until its idle timeout, a tunnel counts against the team's tunnel
cap until it expires. A launcher that owns a stable label, one every relaunch of the same
run reuses and no other live run carries, sweeps it before it creates anything; an
operator sweeps a dead run's label by hand.

A sweep never raises: a label whose sweep failed maps to -1 and logs the manual command.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TUNNEL_PAGE = 100
_TUNNEL_DELETES = 16
"""Concurrent single-tunnel deletes during a sweep."""


class SweepReport(BaseModel):
    """What one sweep deleted, per label; -1 means the sweep of that label failed."""

    sandboxes: dict[str, int] = {}
    tunnels: dict[str, int] = {}


async def sweep_sandboxes(labels: list[str]) -> dict[str, int]:
    """Bulk-delete this user's sandboxes carrying each label. One call per label: a
    bulk delete matches sandboxes carrying ALL of its labels."""
    from prime_sandboxes import AsyncSandboxClient

    deleted: dict[str, int] = {}
    client = AsyncSandboxClient()
    try:
        for label in dict.fromkeys(labels):
            try:
                response = await client.bulk_delete(labels=[label])
            except Exception:
                logger.exception(
                    "sandbox sweep for label %s failed; clean up with: "
                    "prime sandbox delete --label %s",
                    label,
                    label,
                )
                deleted[label] = -1
                continue
            if response.failed:
                logger.warning(
                    "sandbox sweep for label %s left %d sandboxes: %s",
                    label,
                    len(response.failed),
                    response.failed[:5],
                )
            deleted[label] = len(response.succeeded)
            if response.succeeded:
                logger.warning(
                    "swept %d leftover sandboxes labelled %s",
                    len(response.succeeded),
                    label,
                )
    finally:
        await client.aclose()
    return deleted


async def sweep_tunnels(labels: list[str]) -> dict[str, int]:
    """Delete this user's live tunnels carrying each label. The listing filter is
    re-checked client-side: a listing may include team members' tunnels."""
    from prime_tunnel.core.client import TunnelClient

    deleted: dict[str, int] = {}
    client = TunnelClient()
    gate = asyncio.Semaphore(_TUNNEL_DELETES)

    async def delete(tunnel_id: str) -> bool:
        async with gate:
            try:
                await client.delete_tunnel(tunnel_id)  # False: already gone
                return True
            except Exception:
                logger.warning("tunnel %s delete failed", tunnel_id, exc_info=True)
                return False

    try:
        user = client.config.user_id
        for label in dict.fromkeys(labels):
            try:
                found: list[str] = []
                page = 1
                while True:
                    result = await client.list_tunnels_page(
                        labels=[label], page=page, per_page=_TUNNEL_PAGE
                    )
                    found += [
                        t.tunnel_id
                        for t in result.tunnels
                        if label in t.labels
                        and t.user_id in (None, user)
                        and t.terminated_at is None
                    ]
                    if not result.has_next:
                        break
                    page += 1
            except Exception:
                logger.exception(
                    "tunnel sweep for label %s failed; clean up with: "
                    "prime tunnel stop --label %s",
                    label,
                    label,
                )
                deleted[label] = -1
                continue
            outcomes = await asyncio.gather(*(delete(t) for t in found))
            deleted[label] = sum(outcomes)
            if found:
                logger.warning(
                    "swept %d of %d leftover tunnels labelled %s",
                    deleted[label],
                    len(found),
                    label,
                )
    finally:
        await client.close()
    return deleted


async def sweep(labels: list[str]) -> SweepReport:
    """Delete this user's sandboxes and tunnels carrying any of ``labels``. Call it only
    for labels no live run carries: a sweep cannot tell a leftover from a box in use."""
    if not labels:
        return SweepReport()
    sandboxes, tunnels = await asyncio.gather(
        sweep_sandboxes(labels), sweep_tunnels(labels)
    )
    return SweepReport(sandboxes=sandboxes, tunnels=tunnels)
