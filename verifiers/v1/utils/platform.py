"""The eval's run on the Prime Intellect platform (`--no-push` to keep it local)."""

import asyncio
import json
import logging
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import prime_runs as pr
from prime_runs.projection import (
    build_samples,  # noqa: F401 - prime-rl imports it from here
)

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.client import resolve_api_key
from verifiers.v1.episode import EPISODE_EXCLUDE_FIELDS, Episode, WireEpisode
from verifiers.v1.trace import EXCLUDE_FIELDS
from verifiers.v1.utils.prime import load_prime_config
from verifiers.v1.utils.redact import (
    MIN_SECRET_LENGTH,
    REDACTED,
    Redactor,
    env_credentials,
    url_credentials,
)

logger = logging.getLogger(__name__)


UPLOAD_EXCLUDE = {
    **EPISODE_EXCLUDE_FIELDS,
    "traces": {
        "__all__": {
            **EXCLUDE_FIELDS,
            "agent": {"config": {"client": {"headers"}, "harness": {"env"}}},
        }
    },
}
"""The episode projection uploaded: the disk record minus the config fields that carry
credentials (`harness.forward_env` names variables without their values and stays)."""

CREDENTIAL_MAPPING = re.compile(r"(?:^|_)(?:env|headers)$")
"""Config and task-data fields holding an environment or header mapping (a harness
`env`, Harbor's `verifier_env`, a client's or a task config's `headers`). Name-based
discovery stays inside these: applied to every field it would take `api_key_var`'s
value, the *name* of a variable, for a credential, and keeping it out would need the
reference-suffix lists this design avoids. A credential stored under a bare task-data
field (`api_key: ...`) is recognised only through its value's URL shape."""


def strings(value: Any) -> Iterator[str]:
    """Every string in a JSON tree."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def credential_mappings(value: Any, name: str = "") -> Iterator[dict]:
    """Environment and header mappings anywhere in a JSON tree, nested ones included
    (a harness config's `env`, Harbor's `verifier.env`, a task config's `headers`)."""
    if isinstance(value, dict):
        if CREDENTIAL_MAPPING.search(name):
            yield value
        for key, child in value.items():
            yield from credential_mappings(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from credential_mappings(child, name)


def redactable(secrets: Iterator[str] | list[str] | set[str]) -> set[str]:
    """Drop what cannot be redacted safely: a value shorter than `MIN_SECRET_LENGTH` would
    rewrite ordinary text, and one inside the `[REDACTED]` marker (`API_TOKEN=REDACTED`)
    is a sanitized placeholder."""
    return {s for s in secrets if len(s) >= MIN_SECRET_LENGTH and s not in REDACTED}


def known_secrets(
    episodes: list[Episode], config: EvalConfig, *values: str
) -> set[str]:
    """Every run-wide credential that could have reached a trace: the clients' API keys;
    the credentials in the host environment and in every environment or header mapping
    of the run's env config, the traced agent configs, and the trace and episode task
    data (`env_credentials`); URL credentials anywhere in those configs and task data (a
    client `base_url`, a harness endpoint, a task's connection string); and `values`.
    Rollout-specific credentials come from each episode's and trace's `upload_secrets`."""
    traces = [trace for episode in episodes for trace in episode.traces]
    clients = [
        config.client,
        *(t.agent.config.client for t in traces if t.agent.config.client is not None),
    ]
    dumps = [
        config.client.model_dump(mode="json"),
        config.env.model_dump(mode="json"),
        *(trace.agent.config.model_dump(mode="json") for trace in traces),
        *(trace.task.data.model_dump(mode="json") for trace in traces),
        *(episode.task.data.model_dump(mode="json") for episode in episodes),
    ]
    named = [
        os.environ,
        *(mapping for dump in dumps for mapping in credential_mappings(dump)),
    ]
    secrets = {
        *values,
        load_prime_config().get("api_key", ""),
        *(secret for episode in episodes for secret in episode.upload_secrets),
        *(secret for trace in traces for secret in trace.upload_secrets),
        *(resolve_api_key(client) for client in clients),
        *(credential for mapping in named for credential in env_credentials(mapping)),
        *(
            credential
            for dump in dumps
            for text in strings(dump)
            for credential in url_credentials(text)
        ),
    }
    return redactable(secrets)


@dataclass
class PushState:
    """The dashboard's view of the run: reads through to it, owns no I/O."""

    run: pr.Run | None = None
    error: str | None = None

    @property
    def incomplete(self) -> str | None:
        """Current upload errors and record losses, including while the run is active."""
        return _losses(self.run) if self.run is not None else None

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
    disabled one; the eval goes on. With `run.attach` the run already exists on the
    platform (a hosted evaluation's launcher created it and is waiting on it), so
    there is no local fallback: failing to attach fails the eval."""
    attach = config.run.attach
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
    identity = json.loads(
        Redactor(known_secrets([], config)).json(json.dumps(identity))
    )
    if config.push and os.getenv(pr.MODE_ENV, "").strip().lower() == "disabled":
        if attach:
            raise RuntimeError(
                f"run.attach={attach!r} names a run on the platform, but "
                f"{pr.MODE_ENV}=disabled would keep this eval local"
            )
        # The SDK's own kill switch; the explicit `mode="online"` below would override it.
        logger.info("--push: %s=disabled; running without a platform run", pr.MODE_ENV)
    elif config.push:
        try:
            state.run = pr.init(mode="online", id=attach, **identity)
        except Exception as e:
            if attach:
                # The launcher's run would sit at running until it times out; a local
                # run nobody reads is not a substitute.
                raise RuntimeError(
                    f"--run.attach: could not attach to run {attach!r} ({type(e).__name__}: {e})"
                ) from e
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


def log_episodes(run: pr.Run, episodes: list[Episode], config: EvalConfig) -> None:
    """Hand finished episodes to the run, best effort: the SDK already keeps upload
    failures on its own thread, so this only guards the hand-off itself. A problem
    here is the platform's, never the eval's."""
    if not episodes:
        return
    try:
        redactor = Redactor(known_secrets(episodes, config))
        sanitized = [
            WireEpisode.model_validate_json(
                redactor.json(
                    episode.model_dump_json(exclude=UPLOAD_EXCLUDE, exclude_none=True)
                )
            )
            for episode in episodes
        ]
        run.log_episodes(sanitized)
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
    """Close the run out after the eval broke, so it doesn't sit at running. Not
    for a break during `finish_run`: that close-out completes on its own thread
    and the SDK lets the first `finish()` decide the status — it sets `run.status`
    as it starts, so an in-flight one is visible here before it is `finished`."""
    if run.finished or run.status is not pr.RunStatus.RUNNING:
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
        if incomplete := state.incomplete:
            logger.warning("--push: %s, but %s", status.value, incomplete)
        if run.url:
            # The run's own status: `finish()` is a no-op once another caller closed it.
            logger.info("--push: %s -> %s", run.status.value, run.url)


def _losses(run: pr.Run) -> str | None:
    """What the run could not finish or store, or `None`."""
    parts = list(run.errors)
    parts.extend(
        f"{count} record(s) not stored by the {sink} sink"
        for sink, count in sorted(run.failed_records.items())
        if count
    )
    if run.dropped_records:
        parts.append(f"{run.dropped_records} record(s) never queued (uploader overrun)")
    return "; ".join(parts) or None
