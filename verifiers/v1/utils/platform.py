"""The eval's run on the Prime Intellect platform (`--no-push` to keep it local)."""

import asyncio
import json
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import prime_runs as pr
from prime_runs.projection import build_samples

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.episode import Episode

logger = logging.getLogger(__name__)

FRAMEWORK = "verifiers"
REDACTED = "<redacted>"

__all__ = [
    "PushState",
    "abort_run",
    "build_samples",
    "credential_tables",
    "finish_run",
    "open_run",
    "run_config",
    "scrub_secrets",
]

# The two places a config can carry a credential. Neither is found by key
# name: both are free-form string tables, so everything under them is masked,
# whatever a value is called. The API key itself never enters the config
# (`api_key_var` names an environment variable), and the harness table has
# `forward_env` for exactly this reason.
#
# - `headers`: a client's extra request headers (`Authorization`, `x-api-key`
#   for a proxy or a non-Prime endpoint) — the top-level client and any seat's.
# - `harness.env`: a seat's program variables (`SERPER_API_KEY`).


def _is_credential_table(key: Any, parent: Any, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return key == "headers" or (key == "env" and parent == "harness")


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


def open_run(
    config: EvalConfig,
    state: PushState | None = None,
    *,
    num_examples: int | None = None,
) -> "pr.Run":
    """Open the run this eval streams into, before the first rollout.

    `num_examples` is how many tasks the run covers once selection has been
    applied (`-n`, the taskset's size); it is known to the runner, not the config.
    """
    identity: dict[str, Any] = {
        "name": config.run.name,
        # The environment is resolved by name through the hub's get-or-create, so
        # a local env uploads without a prior `prime env push`. A run with no
        # taskset has nothing to attach to and can only be a local run — say so
        # by passing none, rather than asking the hub to resolve an empty name.
        "environments": [config.env.taskset.id] if config.env.taskset.id else [],
        "model": config.model,
        "framework": FRAMEWORK,
        "config": run_config(config, num_examples=num_examples),
    }
    if config.push and os.getenv(pr.MODE_ENV, "").strip().lower() == "disabled":
        # The SDK's own kill switch, honoured here because the explicit
        # `mode="online"` below would otherwise take precedence over it.
        logger.info("--push: %s=disabled; running without a platform run", pr.MODE_ENV)
    elif config.push:
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


def run_config(
    config: EvalConfig, *, num_examples: int | None = None
) -> dict[str, Any]:
    """What the run was configured with, as the dashboard stores it.

    The fields somebody actually set, with the credential tables masked; the
    handful of v0 keys the dashboard reads for its lists (`model`,
    `num_examples`, `rollouts_per_example`); and, when the run was launched
    from one, the config file itself byte for byte — but only if it sets no
    credential table, since a file cannot be masked without rewriting it."""
    values: dict[str, Any] = scrub_secrets(
        config.model_dump(mode="json", exclude_unset=True)
    )
    # `model_dump(exclude_unset=True)` drops defaults; the dashboard reads these
    # unconditionally (the evals list, the reproduce command).
    values.setdefault("model", config.model)
    values["num_examples"] = (
        num_examples
        if num_examples is not None
        else (config.num_tasks if config.num_tasks is not None else -1)
    )
    values["rollouts_per_example"] = config.num_rollouts

    source = config.run.source
    if source is not None:
        try:
            tables = credential_tables(source)
        except pr.ConfigurationError as e:
            logger.warning("--push: not recording the run's config file (%s)", e)
        else:
            if tables:
                logger.warning(
                    "--push: not recording the run's config file: it sets %s. Keep "
                    "credentials in the environment (`api_key_var`, `forward_env`) "
                    "instead.",
                    ", ".join(tables),
                )
            else:
                try:
                    values[pr.CONFIG_SOURCE_KEY] = pr.ConfigSource.from_file(
                        source
                    ).to_dict()
                except pr.ConfigurationError as e:
                    logger.warning(
                        "--push: not recording the run's config file (%s)", e
                    )
    return values


def scrub_secrets(value: Any, _parent: Any = None) -> Any:
    """A copy of a config dump with every value in a credential table masked."""
    if isinstance(value, dict):
        return {
            key: {name: REDACTED for name in item}
            if _is_credential_table(key, _parent, item)
            else scrub_secrets(item, key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_secrets(item, _parent) for item in value]
    return value


def credential_tables(path: "str | os.PathLike[str]") -> list[str]:
    """Dotted paths of the non-empty credential tables in a config file.

    The file is parsed (TOML or JSON, by suffix) rather than scanned: a value's
    shape is then irrelevant. Raises `ConfigurationError` for a file that cannot
    be inspected, which the caller treats as "do not upload"."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    try:
        raw = resolved.read_bytes()
    except OSError as e:
        raise pr.ConfigurationError(f"{path}: {e}") from e
    try:
        if suffix == ".toml":
            document = tomllib.loads(raw.decode("utf-8"))
        elif suffix == ".json":
            document = json.loads(raw.decode("utf-8"))
        else:
            raise pr.ConfigurationError(
                f"{path}: cannot inspect a {suffix or 'suffix-less'} file for credentials"
            )
    except (ValueError, UnicodeDecodeError) as e:
        raise pr.ConfigurationError(f"{path}: could not parse it: {e}") from e
    found: list[str] = []
    _collect_credential_tables(document, "", None, found)
    return found


def _collect_credential_tables(
    value: Any, prefix: str, parent: Any, found: list[str]
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            if _is_credential_table(key, parent, item):
                if item:
                    found.append(here)
            else:
                _collect_credential_tables(item, here, key, found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_credential_tables(item, f"{prefix}[{index}]", parent, found)


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
    summary: Mapping[str, Any] | None = None,
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
