"""Rollout-level retries: rerun a whole rollout while it ends with a retryable error.

Parity with v0's rollout retries. `RetryConfig` lives on `EnvConfig.retry`; `run_with_retry`
wraps a `Rollout` with tenacity, retrying on the trace's captured error (matched by exception
type name against include/exclude) and accumulating each failed attempt's error onto the
returned trace's `errors`, so the final trace shows the whole retry history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_config import BaseConfig
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_result,
    stop_after_attempt,
)

if TYPE_CHECKING:
    from verifiers.v1.rollout import Rollout
    from verifiers.v1.trace import Trace


class RetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error (parity with v0's
    rollout-level retries). Matching is by the error's exception type name, so
    `include`/`exclude` name exception classes (e.g. ``ModelError``, ``ProgramError``)."""

    attempts: int = 3
    """Total rollout attempts (1 = no retry)."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


def should_retry(trace: Trace, retry: RetryConfig) -> bool:
    """Whether a finished rollout should be retried: it ended with an error whose
    exception type is included (and not excluded)."""
    error = trace.error
    if error is None:
        return False
    if error.type in retry.exclude:
        return False
    if retry.include:
        return error.type in retry.include
    return True


async def run_with_retry(
    rollout: Rollout, shared_urls: dict[str, str] | None, retry: RetryConfig
) -> Trace:
    """Run the whole rollout, retrying it while its trace ends with a retryable error.
    Each retry-causing attempt's error is collected onto the returned trace's `errors`,
    so the final trace shows the full history; the last attempt's trace is returned
    as-is once attempts run out (or the rollout succeeds / hits a non-retryable error)."""
    if retry.attempts <= 1:
        return await rollout.run(shared_urls)

    history: list = []

    def record(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        history.extend(state.outcome.result().errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.attempts),
        retry=retry_if_result(lambda trace: should_retry(trace, retry)),
        before_sleep=record,
        retry_error_callback=lambda state: state.outcome.result(),
    )
    trace = await retrying(rollout.run, shared_urls)
    if trace.errors:  # final attempt errored too → prepend the earlier attempts'
        trace.errors = history + trace.errors
    return trace
