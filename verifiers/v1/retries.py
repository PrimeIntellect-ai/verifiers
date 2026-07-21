"""Whole-rollout retries (per-call model/runtime retries are owned by the SDKs, not us).

Transient model-call and runtime faults are retried by the harness/runtime SDKs; the
framework adds targeted retries only where no SDK sits underneath (`retrying()`).
`run_episode_with_retry` reruns an entire env-rollout — the episode is the retry
atom, never one participant of a multi-agent interaction — when it ends with a
retryable error (matched by exception type name); off by default.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_config import BaseConfig
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import EpisodeInfo, Error, Trace

logger = logging.getLogger(__name__)


def retrying(
    *,
    on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    give_up: type[BaseException] | tuple[type[BaseException], ...] = (),
    retries: int,
    label: str | None = None,
) -> AsyncRetrying:
    """The shared retry policy: retry on `on` (minus `give_up`) up to `retries`
    times with exponential backoff + jitter, logging each retry. `label` names the
    operation in the log; omitted, it falls back to the retried callable's name, so
    the `async for attempt in retrying(...)` form should pass it."""

    def _log(state: RetryCallState) -> None:
        exc = state.outcome.exception()
        logger.warning(
            "retrying %s (retry %d/%d) after error: %s: %s",  # name too — some errors stringify empty
            label or getattr(state.fn, "__name__", "call"),
            state.attempt_number,
            retries,
            type(exc).__name__,
            exc,
        )

    return AsyncRetrying(
        stop=stop_after_attempt(retries + 1),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_exception_type(on) & retry_if_not_exception_type(give_up),
        before_sleep=_log,
        reraise=True,
    )


class RolloutRetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error. `include`/`exclude`
    name exception classes (e.g. ``ProviderError``, ``SandboxError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-rollout retries beyond the first attempt. Off by default — the SDKs
    already retry transient per-call faults; rerunning a whole trajectory is opt-in."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


class RetryConfig(BaseConfig):
    """A rollout's retries — only whole-`rollout` ones; per-call retries are the
    harness/runtime SDKs'."""

    rollout: RolloutRetryConfig = RolloutRetryConfig()
    """Retries of the whole rollout, on a captured retryable error."""


def _retryable(error: Error | None, retry: RolloutRetryConfig) -> bool:
    """Whether `error` matches the retry policy: its exception type is included (and
    not excluded)."""
    if error is None:
        return False
    if error.type in retry.exclude:
        return False
    if retry.include:
        return error.type in retry.include
    return True


def episode_should_retry(
    episode: EpisodeInfo, traces: list[Trace], retry: RolloutRetryConfig
) -> bool:
    """Whether a finished env-rollout should be retried: any captured error —
    episode-level or on any trace — is retryable. All captures count, not just the
    most recent: a retryable failure followed by a teardown error would otherwise
    never retry. Episode-atomic — a half-played sibling context isn't reproducible."""
    return any(_retryable(e, retry) for e in episode.errors) or any(
        _retryable(e, retry) for t in traces for e in t.errors
    )


async def run_episode_with_retry(
    run: Callable[[], Awaitable[tuple[EpisodeInfo, list[Trace]]]],
    retry: RolloutRetryConfig,
) -> list[Trace]:
    """Run one env-rollout (`run` must mint a fresh `EpisodeInfo` + traces per
    call), retrying while it ends with a retryable error; returns the final
    attempt's traces. When the final attempt fails too, the earlier attempts'
    errors are prepended onto its `EpisodeInfo` so the episode shows the full
    history; a final good attempt returns clean."""
    if retry.max_retries < 1:
        return (await run())[1]

    history: list = []

    def note(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        episode, traces = state.outcome.result()
        cause = episode.error or next((t.error for t in traces if t.error), None)
        logger.warning(
            "retrying env-rollout %s (retry %d/%d) after error: %s",
            episode.id,
            state.attempt_number,
            retry.max_retries,
            cause.type if cause else "?",
        )
        history.extend(episode.errors)
        for trace in traces:
            history.extend(trace.errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_retries + 1),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        retry=retry_if_result(lambda result: episode_should_retry(*result, retry)),
        before_sleep=note,
        retry_error_callback=lambda state: state.outcome.result(),
    )

    async def attempt() -> tuple[EpisodeInfo, list[Trace]]:
        # tenacity only awaits a coroutine *function*; adapt so a plain callable
        # returning an awaitable works too.
        return await run()

    episode, traces = await retrying(attempt)
    if history and (episode.errors or any(t.errors for t in traces)):
        # final attempt failed too → prepend the history
        episode.errors = history + episode.errors
    return traces
