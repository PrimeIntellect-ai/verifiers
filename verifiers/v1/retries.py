"""Whole-rollout retries (per-call model/runtime faults are retried by the
harness/runtime SDKs; the framework adds targeted retries only where no SDK sits
underneath — `retrying()`). The retry atom is one agent run: `Agent.run` reruns
its rollout while the trace ends with a retryable error; episodes are never
retried as a unit."""

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
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from verifiers.v1.trace import Error

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


class RetryConfig(BaseConfig):
    """Retry a whole agent run when its trace ends with a captured error.
    `include`/`exclude` name exception classes (e.g. ``ProviderError``,
    ``SandboxError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-run retries beyond the first attempt. Off by default — the SDKs
    already retry transient per-call faults; rerunning a whole trajectory is opt-in."""
    include: list[str] = []
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = []
    """Never retry errors whose type is listed (wins over `include`)."""


def retryable(error: Error | None, retry: RetryConfig) -> bool:
    """Whether `error` matches the retry policy: its exception type is included
    (and not excluded)."""
    if error is None:
        return False
    if error.type in retry.exclude:
        return False
    if retry.include:
        return error.type in retry.include
    return True
