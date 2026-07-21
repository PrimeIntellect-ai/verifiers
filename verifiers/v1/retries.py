"""Whole-rollout retries (per-call model/runtime retries are owned by the SDKs, not us).

Transient model-call and runtime faults are retried by the harness/runtime SDKs; the
framework adds targeted retries only where no SDK sits underneath (`retrying()`).
Two opt-in whole-run retry atoms sit above that: `Agent.run` reruns ITS OWN rollout
while the trace ends with a retryable error (`--env.<agent>.retries` — a flaky
grader retries without re-burning the solver), and `run_episode_with_retry` reruns
the entire env-rollout (`--env.retries`) — the coarse fallback for faults no agent
owns: the env's own hooks, cross-agent state. Both match by exception type name;
both off by default.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_config import BaseConfig
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from verifiers.v1.episode import Episode
    from verifiers.v1.trace import Error

logger = logging.getLogger(__name__)


def backoff(attempt: int) -> float:
    """Exponential backoff with full jitter (same curve as `retrying()`'s)."""
    return min(0.5 * 2**attempt, 30.0) * (0.5 + random.random())


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


class RetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error. `include`/`exclude`
    name exception classes (e.g. ``ProviderError``, ``SandboxError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-rollout retries beyond the first attempt. Off by default — the SDKs
    already retry transient per-call faults; rerunning a whole trajectory is opt-in."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


def _retryable(error: Error | None, retry: RetryConfig) -> bool:
    """Whether `error` matches the retry policy: its exception type is included (and
    not excluded)."""
    if error is None:
        return False
    if error.type in retry.exclude:
        return False
    if retry.include:
        return error.type in retry.include
    return True


def trace_should_retry(trace, retry: RetryConfig) -> bool:
    """Whether a finished agent rollout should be retried: any captured error on
    its trace is retryable (all captures count, not just the most recent)."""
    return any(_retryable(e, retry) for e in trace.errors)


def episode_should_retry(episode: Episode, retry: RetryConfig) -> bool:
    """Whether a finished env-rollout should be retried: any captured error —
    episode-level or on any trace — is retryable. All captures count, not just the
    most recent: a retryable failure followed by a teardown error would otherwise
    never retry. Episode-atomic — a half-played sibling context isn't reproducible."""
    return any(_retryable(e, retry) for e in episode.errors) or any(
        _retryable(e, retry) for t in episode.traces for e in t.errors
    )


async def run_episode_with_retry(
    run: Callable[[], Awaitable[Episode]],
    retry: RetryConfig,
) -> Episode:
    """Run one env-rollout (`run` must mint a fresh episode per call), retrying while
    it ends with a retryable error. When the final attempt fails too, the earlier
    attempts' errors are prepended so the episode shows the full history; a final
    good attempt returns clean."""
    history: list = []
    for attempt in range(retry.max_retries + 1):
        final = await run()
        if attempt == retry.max_retries or not episode_should_retry(final, retry):
            break
        cause = final.error or next((t.error for t in final.traces if t.error), None)
        history.extend(final.errors)
        for trace in final.traces:
            history.extend(trace.errors)
        delay = backoff(attempt)
        logger.warning(
            "retrying env-rollout %s (retry %d/%d) in %.1fs after error: %s",
            final.id,
            attempt + 1,
            retry.max_retries,
            delay,
            cause.type if cause else "?",
        )
        await asyncio.sleep(delay)
    if history:
        # The full history rides the final episode either way; success is the
        # `ok` stamp, never errors-emptiness. In place: the envelope, the stamp,
        # and every consumer share this one list.
        final.errors[:0] = history
    return final
