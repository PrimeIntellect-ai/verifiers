"""The eval's run on the Prime Intellect platform (`--no-push` to keep it local)."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import prime_runs as pr
from prime_runs.projection import build_samples

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode

logger = logging.getLogger(__name__)

FRAMEWORK = "verifiers"

__all__ = [
    "PushState",
    "abort_run",
    "build_samples",
    "finish_run",
    "open_run",
]


@dataclass
class PushState:
    """The dashboard's view of the upload. Owns no I/O — the run does.

    Reads through to the live run, so the footer can show the run's URL from the
    moment it opens rather than only once everything has been uploaded."""

    run: "pr.Run | None" = None
    error: str | None = None

    @property
    def url(self) -> str | None:
        """Where to watch the run. `None` for a run that stays local."""
        return self.run.url if self.run is not None else None

    @property
    def finished(self) -> bool:
        """Whether the run has been closed out."""
        return self.run is not None and self.run.finished

    @property
    def started(self) -> bool:
        """Whether there is anything to report: a live run, or why there isn't one."""
        return self.error is not None or self.url is not None

    @property
    def warning(self) -> str | None:
        """What went wrong without sinking the upload, or `None` if nothing did.

        Records that were lost first, and how: dropped records reached no sink at
        all (the rollouts outran the uploader), while a sink failing says nothing
        about the others — with traces and samples both on, those records are
        usually still safe in the one that worked. Failing that, the first thing
        the SDK contained (`on_error="warn"`), which would otherwise be visible
        only in the run's log file."""
        if self.run is None:
            return None
        parts = (
            [f"{self.run.dropped_records} dropped"] if self.run.dropped_records else []
        )
        parts += [
            f"{count} failed via {sink}"
            for sink, count in sorted(self.run.failed_records.items())
            if count
        ]
        if parts:
            return ", ".join(parts)
        return self.run.errors[0] if self.run.errors else None


def open_run(config: EvalConfig, state: PushState | None = None) -> "pr.Run":
    """Open the run this eval streams into, before the first rollout."""
    identity: dict[str, Any] = {
        "name": config.run.name,
        # The environment is resolved by name through the hub's get-or-create, so
        # a local env uploads without a prior `prime env push`. A run with no
        # taskset has nothing to attach to and can only be a local run — say so
        # by passing none, rather than asking the hub to resolve an empty name.
        "environments": [config.env.taskset.id] if config.env.taskset.id else [],
        "model": config.model,
        "framework": FRAMEWORK,
        "config": run_config(config),
        # verifiers installs its own SIGINT/SIGTERM handler (`install_interrupt`)
        # so a killed eval still tears down its sandboxes; the runner reports the
        # terminal status from the unwind rather than letting the SDK take the
        # signal. The SDK's atexit hook still catches an exit that gets neither.
        "handle_signals": False,
    }
    if config.push:
        try:
            run = pr.init(mode="online", **identity)
            if state is not None:
                state.run = run
            return run
        except Exception as e:  # noqa: BLE001 - a failed upload must not fail the eval
            logger.warning(
                "--push: could not open the run (%s: %s); running without it",
                type(e).__name__,
                e,
            )
            if state is not None:
                state.error = f"{type(e).__name__}: {e}"
    run = pr.init(mode="disabled", **identity)
    if state is not None:
        state.run = run
    return run


def run_config(config: EvalConfig) -> dict[str, Any]:
    """What the run was configured with — the fields somebody actually set, plus
    the file it was launched from, kept byte for byte."""
    values: dict[str, Any] = config.model_dump(mode="json", exclude_unset=True)
    source = config.run.source
    if source is not None:
        try:
            values[pr.CONFIG_SOURCE_KEY] = pr.ConfigSource.from_file(source).to_dict()
        except pr.ConfigurationError as e:
            logger.warning("--push: not recording the run's config file (%s)", e)
    return values


def finish_run(
    run: "pr.Run", episodes: list[Episode], state: PushState | None = None
) -> None:
    """Drain the queued episodes, write the run's aggregates and close it out.

    Blocking — call it off the event loop (`asyncio.to_thread`) so the dashboard
    keeps refreshing while the last uploads land."""
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


def abort_run(
    run: "pr.Run", error: BaseException, state: PushState | None = None
) -> None:
    """Close the run out after the eval broke, so it doesn't sit at running."""
    if run.finished:
        return
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        status, message = pr.RunStatus.CRASHED, "interrupted"
    else:
        status, message = pr.RunStatus.FAILED, f"{type(error).__name__}: {error}"
    _close(run, state, status=status, error=message)


def _close(
    run: "pr.Run",
    state: PushState | None,
    summary: dict[str, Any] | None = None,
    status: "pr.RunStatus" = pr.RunStatus.COMPLETED,
    error: str | None = None,
) -> None:
    """`run.finish()` with the same best-effort contract as the rest of this
    module: the eval's results are already on disk, so nothing here may raise."""
    try:
        run.finish(summary, status=status, error=error)
    except Exception as e:  # noqa: BLE001 - the run is over; report, don't raise
        logger.warning(
            "--push: could not close out the run (%s: %s)", type(e).__name__, e
        )
        if state is not None and state.error is None:
            state.error = f"{type(e).__name__}: {e}"
    else:
        if run.url:
            logger.info("--push: %s -> %s", status.value, run.url)
