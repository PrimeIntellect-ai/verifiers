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
    """Open the run this eval streams into, before the first rollout. A run that
    cannot be opened is logged and replaced by a disabled one; the eval goes on."""
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
            return state.run
        except Exception as e:  # noqa: BLE001 - a failed upload must not fail the eval
            logger.warning(
                "--push: could not open the run (%s: %s); running without it",
                type(e).__name__,
                e,
            )
            state.error = f"{type(e).__name__}: {e}"
    state.run = pr.init(mode="disabled", **identity)
    return state.run


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
        status, message = pr.RunStatus.CRASHED, "interrupted"
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
        if run.url:
            logger.info("--push: %s -> %s", status.value, run.url)
