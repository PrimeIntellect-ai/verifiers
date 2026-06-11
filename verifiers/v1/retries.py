"""Retries at two granularities: per-call (model, runtime) and whole-rollout.

`RetryConfig` lives on `EnvConfig.retries`. Per-call retries wrap a single model or
runtime call (`RetryingClient` / `RetryingRuntime`), rerunning just the failed call so
the rest of the rollout's progress is kept — the cheap, default-on layer. Whole-rollout
retries (`run_with_retry`) rerun an entire trajectory when its trace ends with a retryable
error (matched by exception type name against include/exclude), accumulating each failed
attempt's error onto the returned trace's `errors`; off by default (parity with v0, but
superseded by the finer per-call retries).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic_config import BaseConfig
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_result,
    stop_after_attempt,
)

if TYPE_CHECKING:
    from verifiers.v1.interception import InterceptionPool
    from verifiers.v1.rollout import Rollout
    from verifiers.v1.trace import Trace


class CallRetryConfig(BaseConfig):
    """Retries around a single model or runtime call (tenacity). Reruns just the failed
    call, so the rest of the rollout's progress survives a transient failure."""

    max_attempts: int = Field(3, ge=1)
    """Total attempts for one call (1 = no retry)."""


class RolloutRetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error (parity with v0's
    rollout-level retries). Matching is by the error's exception type name, so
    `include`/`exclude` name exception classes (e.g. ``ModelError``, ``ProgramError``)."""

    max_attempts: int = Field(1, ge=1)
    """Total rollout attempts (1 = no retry)."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


class RetryConfig(BaseConfig):
    """All of a rollout's retries: per-call `model` + `runtime` (rerun a single failed
    call) and whole-`rollout` (rerun the whole trajectory on a captured error)."""

    model: CallRetryConfig = CallRetryConfig()
    """Retries around each model/provider call (the interception server's completion)."""
    runtime: CallRetryConfig = CallRetryConfig()
    """Retries around each runtime call (provision, exec, file read/write)."""
    rollout: RolloutRetryConfig = RolloutRetryConfig()
    """Retries of the whole rollout, on a captured retryable error."""


def should_retry(trace: Trace, retry: RolloutRetryConfig) -> bool:
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
    rollout: Rollout,
    shared_urls: dict[str, str] | None,
    interception: InterceptionPool | None,
    retry: RolloutRetryConfig,
) -> Trace:
    """Run the whole rollout, retrying it while its trace ends with a retryable error.
    Each retry-causing attempt's error is collected onto the returned trace's `errors`,
    so the final trace shows the full history; the last attempt's trace is returned
    as-is once attempts run out (or the rollout succeeds / hits a non-retryable error)."""
    if retry.max_attempts <= 1:
        return await rollout.run(shared_urls, interception)

    history: list = []

    def record(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        history.extend(state.outcome.result().errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_attempts),
        retry=retry_if_result(lambda trace: should_retry(trace, retry)),
        before_sleep=record,
        retry_error_callback=lambda state: state.outcome.result(),
    )
    trace = await retrying(rollout.run, shared_urls, interception)
    if trace.errors:  # final attempt errored too → prepend the earlier attempts'
        trace.errors = history + trace.errors
    return trace
