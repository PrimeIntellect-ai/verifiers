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

import logging
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
    from verifiers.v1.rollout import Rollout
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


def retrying(
    *,
    on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    give_up: type[BaseException] | tuple[type[BaseException], ...] = (),
    retries: int,
    label: str | None = None,
) -> AsyncRetrying:
    """The shared per-call retry policy (tenacity): retry on `on` (minus `give_up`) up to `retries`
    times with exponential backoff + jitter, logging each retry. Drives the model/runtime call
    wrappers (`RetryingClient` / `RetryingRuntime`) and `open_tunnel`. `label` names the operation
    in the log; omitted, it falls back to the retried callable's name (set by the `retrying(fn, ...)`
    call form; the `async for attempt in retrying(...)` form should pass `label`)."""

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


class CallRetryConfig(BaseConfig):
    """Retries around a single model or runtime call (tenacity). Reruns just the failed
    call, so the rest of the rollout's progress survives a transient failure."""

    max_retries: int = Field(3, ge=0)
    """Retries for one call beyond the first attempt (0 = no retry, N = up to N retries)."""


class RolloutRetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error (parity with v0's
    rollout-level retries). Matching is by the error's exception type name, so
    `include`/`exclude` name exception classes (e.g. ``ProviderError``, ``ProgramError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-rollout retries beyond the first attempt (0 = no retry, the default, N = up to N
    retries). Off by default — per-call `model`/`runtime` retries already cover transient faults;
    rerunning a whole trajectory is opt-in (set this, plus `include`/`exclude`)."""
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
    retry: RolloutRetryConfig,
) -> Trace:
    """Run the whole rollout, retrying it while its trace ends with a retryable error.
    Each retry-causing attempt's error is collected onto the returned trace's `errors`,
    so the final trace shows the full history; the last attempt's trace is returned
    as-is once attempts run out (or the rollout succeeds / hits a non-retryable error)."""
    if retry.max_retries < 1:
        return await rollout.run()

    history: list = []

    def record(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        trace = state.outcome.result()
        logger.warning(
            "retrying rollout %s (retry %d/%d) after error: %s",
            trace.id,
            state.attempt_number,
            retry.max_retries,
            trace.error.type if trace.error else "?",
        )
        history.extend(trace.errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_retries + 1),
        retry=retry_if_result(lambda trace: should_retry(trace, retry)),
        before_sleep=record,
        retry_error_callback=lambda state: state.outcome.result(),
    )
    trace = await retrying(rollout.run)
    if trace.errors:  # final attempt errored too → prepend the earlier attempts'
        trace.errors = history + trace.errors
    return trace
