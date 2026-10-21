"""Delete a run's leftover Prime sandboxes and tunnels by label.

Every Prime sandbox and every Prime tunnel a process opens carries its run labels
(``run_labels``: the process-wide base labels and the run scope, ``$VF_RUN_ID``). A run
that dies without its teardown (SIGKILL, OOM, a host reboot) leaves both behind: a
sandbox keeps billing until its idle timeout, a tunnel counts against the team's tunnel
cap until it expires. A launcher that reuses one run scope across relaunches sweeps it
when it starts; an operator sweeps a dead run's label by hand.

A sweep deletes only this user's resources, created before its `before` cutoff, and
never what this process holds open, so a launcher that takes `before` when it starts
cannot delete what it creates itself, whenever it sweeps (see `sweep`). A sweep never
raises: a label it could not sweep maps to -1 and logs why.

Prime volumes and Compose images are deliberately out of scope: volumes are named, shared
and persistent and carry no run labels, so a sweep never deletes them.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PAGE = 100
_TUNNEL_DELETES = 16
"""Concurrent single-tunnel deletes during a sweep."""


class SweepReport(BaseModel):
    """What one sweep deleted, per label; -1 means that label was not swept (the sweep
    failed, or this user's resources could not be told from teammates')."""

    sandboxes: dict[str, int] = {}
    tunnels: dict[str, int] = {}


def _check(labels: list[str], before: datetime) -> list[str]:
    from verifiers.v1.runtimes.backends.prime import BASE_LABELS
    from verifiers.v1.utils.scope import run_scope

    if before.tzinfo is None:
        raise ValueError("before must be timezone-aware, e.g. datetime.now(UTC)")
    shared = set(labels) & (set(BASE_LABELS) - {run_scope()})
    if shared:
        raise ValueError(
            f"refusing to sweep base labels {sorted(shared)}: every process that sets "
            "them carries them; sweep a run's own scope instead"
        )
    return list(dict.fromkeys(labels))


def _older(created: datetime | None, before: datetime) -> bool:
    """Whether `created` (the provider's time, UTC when it names no zone) is before
    `before`; an unknown creation time is not."""
    if created is None:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created < before


def _owned(owner: str | None, user: str | None, unique_labels: bool) -> bool:
    """Whether a resource a listing returned is this user's to delete. A listing covers
    the configured team, so it is only when it names this user as its owner. Without a
    user id, only a caller that vouches nobody else's resource carries the label may."""
    return owner == user if user is not None else unique_labels


def _unowned(kind: str, label: str) -> int:
    logger.warning(
        "%s sweep for label %s skipped: no Prime user id is configured "
        "(PRIME_USER_ID or `prime login`), so this user's resources can't be told "
        "from teammates'; pass unique_labels=True if the label is this launch's alone",
        kind,
        label,
    )
    return -1


def _failed(kind: str, label: str, command: str) -> int:
    logger.exception(
        "%s sweep for label %s failed; clean up with: %s --label %s",
        kind,
        label,
        command,
        label,
    )
    return -1


async def sweep_sandboxes(
    labels: list[str], *, before: datetime, unique_labels: bool = False
) -> dict[str, int]:
    """Delete this user's sandboxes that carry each label and were created before
    `before`, except those this process holds (see `sweep`)."""
    from verifiers.v1.runtimes.backends.prime import leased_client
    from verifiers.v1.runtimes.base import _LIVE

    labels = _check(labels, before)
    deleted: dict[str, int] = {}

    async def sweep_label(client, user: str | None, label: str) -> int:
        live = {backend.info.id for backend in list(_LIVE)}
        found: list[str] = []
        page = 1
        while True:
            result = await client.list(
                labels=[label],
                page=page,
                per_page=_PAGE,
                exclude_terminated=True,
                user_id=user,
            )
            found += [
                box.id
                for box in result.sandboxes
                if label in box.labels
                and _owned(box.user_id, user, unique_labels)
                and _older(box.created_at, before)
                and box.id not in live
            ]
            if not result.has_next:
                break
            page += 1
        succeeded = 0
        for start in range(0, len(found), _PAGE):
            response = await client.bulk_delete(
                sandbox_ids=found[start : start + _PAGE]
            )
            succeeded += len(response.succeeded)
            if response.failed:
                logger.warning(
                    "sandbox sweep for label %s left %d sandboxes: %s",
                    label,
                    len(response.failed),
                    response.failed[:5],
                )
        if succeeded:
            logger.warning("swept %d leftover sandboxes labelled %s", succeeded, label)
        return succeeded

    try:
        async with leased_client() as client:
            # An empty PRIME_USER_ID is no user id (the SDK returns "" for it).
            user = client.client.config.user_id or None
            for label in labels:
                if user is None and not unique_labels:
                    deleted[label] = _unowned("sandbox", label)
                    continue
                try:
                    deleted[label] = await sweep_label(client, user, label)
                except Exception:  # noqa: BLE001 - a sweep never raises; logged
                    deleted[label] = _failed("sandbox", label, "prime sandbox delete")
    except Exception:  # noqa: BLE001 - no client: every label left failed
        for label in labels:
            if label not in deleted:
                deleted[label] = _failed("sandbox", label, "prime sandbox delete")
    return deleted


async def sweep_tunnels(
    labels: list[str], *, before: datetime, unique_labels: bool = False
) -> dict[str, int]:
    """Delete this user's live tunnels that carry each label and were created before
    `before`, except those this process holds (see `sweep`)."""
    from verifiers.v1.interception.tunnel.prime import LIVE_TUNNELS

    labels = _check(labels, before)
    deleted: dict[str, int] = {}
    gate = asyncio.Semaphore(_TUNNEL_DELETES)

    async def delete(client, tunnel_id: str) -> bool:
        async with gate:
            try:
                return await client.delete_tunnel(tunnel_id)  # False: already gone
            except Exception:
                logger.warning("tunnel %s delete failed", tunnel_id, exc_info=True)
                return False

    async def sweep_label(client, user: str | None, label: str) -> int:
        live = {tunnel.tunnel_id for tunnel in list(LIVE_TUNNELS)}
        found: list[str] = []
        page = 1
        while True:
            result = await client.list_tunnels_page(
                labels=[label], page=page, per_page=_PAGE
            )
            found += [
                t.tunnel_id
                for t in result.tunnels
                if label in t.labels
                and _owned(t.user_id, user, unique_labels)
                and t.terminated_at is None
                and _older(t.created_at, before)
                and t.tunnel_id not in live
            ]
            if not result.has_next:
                break
            page += 1
        succeeded = sum(await asyncio.gather(*(delete(client, t) for t in found)))
        if found:
            logger.warning(
                "swept %d of %d leftover tunnels labelled %s",
                succeeded,
                len(found),
                label,
            )
        return succeeded

    try:
        from prime_tunnel.core.client import TunnelClient

        client = TunnelClient()
        try:
            user = client.config.user_id or None  # "" is no user id
            for label in labels:
                if user is None and not unique_labels:
                    deleted[label] = _unowned("tunnel", label)
                    continue
                try:
                    deleted[label] = await sweep_label(client, user, label)
                except Exception:  # noqa: BLE001 - a sweep never raises; logged
                    deleted[label] = _failed("tunnel", label, "prime tunnel stop")
        finally:
            await client.close()
    except Exception:  # noqa: BLE001 - no client: every label left failed
        for label in labels:
            if label not in deleted:
                deleted[label] = _failed("tunnel", label, "prime tunnel stop")
    return deleted


async def sweep(
    labels: list[str] | None = None,
    *,
    before: datetime,
    unique_labels: bool = False,
) -> SweepReport:
    """Delete this user's Prime sandboxes and tunnels that carry one of `labels` (by
    default the run scope, ``$VF_RUN_ID``) and were created before `before`, except
    those this process holds.

    - **The cutoff.** Take `before` when the launch starts, before anything creates a
      box: then nothing the launch creates is swept, whenever it sweeps. This holds up
      to the skew between this host's clock and the provider's, which stamps creation;
      only this process's own boxes and tunnels are spared regardless of the clock.
    - **Ownership.** A listing covers the configured team, so a resource is deleted
      only when the listing names this user (the configured Prime user id,
      ``PRIME_USER_ID`` or ``prime login``) as its owner. Without a user id nothing is
      deleted (the label reports -1), unless `unique_labels` says no one else's
      resource can carry the labels (a label minted for this launch). A known user id
      always decides, `unique_labels` or not.
    - **Shared labels.** Sweep only labels no other live run carries: a sweep cannot
      tell another run's box from a leftover. Base labels this process set
      (`set_base_sandbox_labels`) are refused, but labels shared some other way
      (another process's base labels, a `PrimeConfig.labels` entry such as a CI
      label) are not detected, so pass only labels you know to be one run's.

    Raises ValueError for a naive `before` or a base label; otherwise never raises: a
    label that was not swept (it failed, or ownership is unknown) reports -1."""
    from verifiers.v1.utils.scope import run_scope

    labels = _check([run_scope()] if labels is None else labels, before)
    if not labels:
        return SweepReport()
    sandboxes, tunnels = await asyncio.gather(
        sweep_sandboxes(labels, before=before, unique_labels=unique_labels),
        sweep_tunnels(labels, before=before, unique_labels=unique_labels),
    )
    return SweepReport(sandboxes=sandboxes, tunnels=tunnels)
