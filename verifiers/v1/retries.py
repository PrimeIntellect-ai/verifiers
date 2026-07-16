"""Whole-rollout retries (per-call model/runtime retries are owned by the SDKs, not us).

Transient model-call faults are retried by the harness's own SDK (the interception server is a
faithful proxy — it relays the provider's status), and transient runtime faults by each runtime
SDK (prime/modal); the framework adds targeted retries only where there's no SDK underneath (e.g.
`open_tunnel`, via the shared `retrying()` policy). `RetryConfig` (on `EnvConfig.retries`) keeps one
knob: whole-rollout retries. `run_with_retry` reruns an entire trajectory when its trace ends with a
retryable error (matched by exception type name against include/exclude), accumulating each failed
attempt's error onto the returned trace's `errors`; off by default.
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
    from verifiers.v1.trace import Error, RolloutRecord, Trace

logger = logging.getLogger(__name__)


def retrying(
    *,
    on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    give_up: type[BaseException] | tuple[type[BaseException], ...] = (),
    retries: int,
    label: str | None = None,
) -> AsyncRetrying:
    """The shared retry policy (tenacity): retry on `on` (minus `give_up`) up to `retries` times with
    exponential backoff + jitter, logging each retry. For the framework's own targeted retries where
    no SDK retries underneath (e.g. `open_tunnel`). `label` names the operation in the log; omitted,
    it falls back to the retried callable's name (set by the `retrying(fn, ...)` call form; the
    `async for attempt in retrying(...)` form should pass `label`)."""

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
    """Retry a whole rollout when it ends with a captured error (parity with v0's
    rollout-level retries). Matching is by the error's exception type name, so
    `include`/`exclude` name exception classes (e.g. ``ProviderError``, ``SandboxError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-rollout retries beyond the first attempt (0 = no retry, the default, N = up to N
    retries). Off by default — the harness/runtime SDKs already retry transient per-call faults;
    rerunning a whole trajectory is opt-in (set this, plus `include`/`exclude`)."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


class RetryConfig(BaseConfig):
    """A rollout's retries. Per-call model/runtime retries are owned by the harness/runtime SDKs;
    the framework keeps only whole-`rollout` retries (rerun the whole trajectory on a captured
    retryable error)."""

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


def should_retry(trace: Trace, retry: RolloutRetryConfig) -> bool:
    """Whether a finished rollout should be retried: it ended with a retryable error."""
    return _retryable(trace.error, retry)


def record_should_retry(record: RolloutRecord, retry: RolloutRetryConfig) -> bool:
    """Whether a finished env-rollout should be retried: its record-level error, or
    any of its traces' errors, is retryable. Retries are record-atomic — the whole
    rollout reruns, never one participant of a multi-agent interaction (a half-played
    sibling context isn't reproducible)."""
    return _retryable(record.error, retry) or any(
        _retryable(t.error, retry) for t in record.traces
    )


async def run_with_retry(
    run: Callable[[], Awaitable[Trace]],
    retry: RolloutRetryConfig,
) -> Trace:
    """Run the whole rollout (`run` — e.g. `lambda: agent.run(task)`; each call must
    mint a fresh trajectory), retrying it while its trace ends with a retryable error.
    Each retry-causing attempt's error is collected onto the returned trace's `errors`,
    so the final trace shows the full history; the last attempt's trace is returned
    as-is once attempts run out (or the rollout succeeds / hits a non-retryable error)."""
    if retry.max_retries < 1:
        return await run()

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

    async def attempt() -> Trace:
        # tenacity only awaits a coroutine *function*; adapt so a plain callable
        # returning an awaitable (e.g. `lambda: agent.run(task)`) works too.
        return await run()

    trace = await retrying(attempt)
    if trace.errors:  # final attempt errored too → prepend the earlier attempts'
        trace.errors = history + trace.errors
    return trace


async def run_record_with_retry(
    run: Callable[[], Awaitable[RolloutRecord]],
    retry: RolloutRetryConfig,
) -> RolloutRecord:
    """Run one env-rollout (`run` — each call must mint a fresh record), retrying it
    while it ends with a retryable error, record-level or on any trace
    (`record_should_retry`). Each retry-causing attempt's errors are collected onto
    the returned record's `errors` when the final attempt fails too, so the record
    shows the full history; a final good attempt returns clean — the record-level
    twin of `run_with_retry`."""
    if retry.max_retries < 1:
        return await run()

    history: list = []

    def record(state: RetryCallState) -> None:
        # before_sleep fires only between attempts (a retry is imminent), so this
        # collects exactly the errors that caused a retry — never the final attempt's.
        attempt_record = state.outcome.result()
        cause = attempt_record.error or next(
            (t.error for t in attempt_record.traces if t.error), None
        )
        logger.warning(
            "retrying env-rollout %s (retry %d/%d) after error: %s",
            attempt_record.id,
            state.attempt_number,
            retry.max_retries,
            cause.type if cause else "?",
        )
        history.extend(attempt_record.errors)
        for trace in attempt_record.traces:
            history.extend(trace.errors)

    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_retries + 1),
        retry=retry_if_result(lambda rec: record_should_retry(rec, retry)),
        before_sleep=record,
        retry_error_callback=lambda state: state.outcome.result(),
    )

    async def attempt() -> RolloutRecord:
        # same adaptation as `run_with_retry`: tenacity only awaits coroutine functions
        return await run()

    final = await retrying(attempt)
    if history and not final.ok:  # final attempt failed too → prepend the history
        final.errors = history + final.errors
    return final
