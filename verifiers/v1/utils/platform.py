"""The eval's run on the Prime Intellect platform (`--no-push` to keep it local)."""

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import prime_runs as pr
from prime_runs.projection import (
    build_samples,  # noqa: F401 - prime-rl imports it from here
)

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode

logger = logging.getLogger(__name__)


@dataclass
class PushState:
    """The dashboard's view of the run: reads through to it, owns no I/O."""

    run: pr.Run | None = None
    error: str | None = None
    incomplete: str | None = None
    """Set when the run closed out but the uploader lost records on the way: the run
    exists and holds what did land, so the footer says so rather than "failed"."""

    @property
    def url(self) -> str | None:
        return self.run.url if self.run is not None else None

    @property
    def finished(self) -> bool:
        return self.run is not None and self.run.finished

    @property
    def started(self) -> bool:
        """A live run, or a reason there isn't one."""
        return self.error is not None or self.url is not None


def open_run(config: EvalConfig, state: PushState, *, num_examples: int) -> pr.Run:
    """Open the run this eval streams into, before the first rollout, and give the
    config the run's id. A run that cannot be opened is logged and replaced by a
    disabled one; the eval goes on."""
    identity: dict[str, Any] = {
        "name": config.run.name,
        # Resolved by name via the hub's get-or-create; no taskset, nothing to attach to.
        "environments": [config.env.taskset.id] if config.env.taskset.id else [],
        "model": config.model,
        "framework": "verifiers",
        # The v0 keys the dashboard's lists read. The config itself is a follow-up:
        # a dump or the launched file can carry credentials and needs masking first.
        "config": {
            "model": config.model,
            "num_examples": num_examples,
            "rollouts_per_example": config.num_rollouts,
        },
    }
    if config.push and os.getenv(pr.MODE_ENV, "").strip().lower() == "disabled":
        # The SDK's own kill switch; the explicit `mode="online"` below would override it.
        logger.info("--push: %s=disabled; running without a platform run", pr.MODE_ENV)
    elif config.push:
        try:
            state.run = pr.init(mode="online", **identity)
        except Exception as e:  # noqa: BLE001 - a failed upload must not fail the eval
            logger.warning(
                "--push: could not open the run (%s: %s); running without it",
                type(e).__name__,
                e,
            )
            state.error = f"{type(e).__name__}: {e}"
    if state.run is None:
        state.run = pr.init(mode="disabled", **identity)
    # The run's one id: the platform's when online, the SDK's local one otherwise.
    # The SDK keys every upload to it regardless; this is for the local records.
    config.run.assign_id(state.run.id)
    return state.run


def log_episodes(run: pr.Run, episodes: list[Episode]) -> None:
    """Hand finished episodes to the run, best effort: the SDK already keeps upload
    failures on its own thread, so this only guards the hand-off itself. A problem
    here is the platform's, never the eval's."""
    if not episodes:
        return
    try:
        run.log_episodes(episodes)
    except Exception as e:  # noqa: BLE001 - the rollouts are on disk; report, don't raise
        logger.warning(
            "--push: could not queue %d episode(s) (%s: %s)",
            len(episodes),
            type(e).__name__,
            e,
        )


def finish_run(run: pr.Run, episodes: list[Episode], state: PushState) -> None:
    """Drain, write the run's aggregates, close it out. Blocking: call it off the loop."""
    try:
        summary = pr.metrics.from_episodes(episodes)
    except Exception as e:  # noqa: BLE001 - close the run even without its headline
        logger.warning(
            "--push: could not aggregate the run's metrics (%s: %s)",
            type(e).__name__,
            e,
        )
        summary = None
    _close(run, state, summary=summary)


def abort_run(run: pr.Run, error: BaseException, state: PushState) -> None:
    """Close the run out after the eval broke, so it doesn't sit at running."""
    if run.finished:
        return
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        status, message = pr.RunStatus.CANCELLED, "interrupted"
    else:
        status, message = pr.RunStatus.FAILED, f"{type(error).__name__}: {error}"
    _close(run, state, status=status, error=message)


def _close(
    run: pr.Run,
    state: PushState,
    summary: Mapping[str, Any] | None = None,
    status: pr.RunStatus = pr.RunStatus.COMPLETED,
    error: str | None = None,
) -> None:
    """`run.finish()`, best effort: the results are on disk, so nothing here may raise."""
    try:
        run.finish(summary, status=status, error=error)
    except Exception as e:  # noqa: BLE001 - the run is over; report, don't raise
        logger.warning(
            "--push: could not close out the run (%s: %s)", type(e).__name__, e
        )
        if state.error is None:
            state.error = f"{type(e).__name__}: {e}"
    else:
        state.incomplete = _losses(run)
        if state.incomplete:
            logger.warning("--push: %s, but %s", status.value, state.incomplete)
        if run.url:
            logger.info("--push: %s -> %s", status.value, run.url)


def _losses(run: pr.Run) -> str | None:
    """What the uploader could not store, or `None`. A sink that switched itself off
    quietly (Prime Traces outside the beta) is not a loss and is not counted here."""
    parts = [
        f"{count} record(s) not stored by the {sink} sink"
        for sink, count in sorted(run.failed_records.items())
        if count
    ]
    if run.dropped_records:
        parts.append(f"{run.dropped_records} record(s) never queued (uploader overrun)")
    return "; ".join(parts) or None
