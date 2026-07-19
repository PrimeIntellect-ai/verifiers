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
    from verifiers.v1.trace import Error, Episode

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


def episode_should_retry(episode: Episode, retry: RolloutRetryConfig) -> bool:
    """Whether a finished env-rollout should be retried: its episode-level error, or
    any trace's error, is retryable. Episode-atomic — a half-played sibling context
    isn't reproducible."""
    return _retryable(episode.error, retry) or any(
        _retryable(t.error, retry) for t in episode.traces
    )


async def run_episode_with_retry(
    run: Callable[[], Awaitable[Episode]],
    retry: RolloutRetryConfig,
) -> Episode:
    """Run one env-rollout (`run` must mint a fresh episode per call), retrying while
    it ends with a retryable error. When the final attempt fails too, the earlier
    attempts' errors are prepended so the episode shows the full history; a final
    good attempt returns clean."""
    if retry.max_retries < 1:
        return await run()

    history: list = []

    def note(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        attempt_episode = state.outcome.result()
        cause = attempt_episode.error or next(
            (t.error for t in attempt_episode.traces if t.error), None
        )
        logger.warning(
            "retrying env-rollout %s (retry %d/%d) after error: %s",
            attempt_episode.id,
            state.attempt_number,
            retry.max_retries,
            cause.type if cause else "?",
        )
        history.extend(attempt_episode.errors)
        for trace in attempt_episode.traces:
            history.extend(trace.errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_retries + 1),
        retry=retry_if_result(lambda rec: episode_should_retry(rec, retry)),
        before_sleep=note,
        retry_error_callback=lambda state: state.outcome.result(),
    )

    async def attempt() -> Episode:
        # tenacity only awaits a coroutine *function*; adapt so a plain callable
        # returning an awaitable works too.
        return await run()

    final = await retrying(attempt)
    if history and not final.ok:  # final attempt failed too → prepend the history
        final.errors = history + final.errors
    return final
