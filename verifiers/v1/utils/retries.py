"""Whole-rollout retries (per-call retries are owned by the layer making the call).

A model call the provider fails transiently is retried in place by the interception
server (`ProviderRetryConfig`); transient runtime faults by the runtime SDKs; the
framework adds targeted retries only where no SDK sits underneath (`retrying()`).
Two opt-in whole-run retry atoms sit above that: `Agent.run` reruns ITS OWN rollout
while the trace ends with a retryable error (`--env.<agent>.retries` — a flaky
grader retries without re-burning the solver), and `run_episode_with_retry` reruns
the entire episode (`--env.retries`) — the coarse fallback for faults no agent
owns: the env's own hooks, cross-agent state. Both use ordered retry rules;
both off by default.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from verifiers.v1.configs.client import ProviderRetryConfig
from verifiers.v1.configs.retries import RetryConfig, RetryRule
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    SandboxError,
    TaskError,
    transient,
)

if TYPE_CHECKING:
    from verifiers.v1.episode import Episode
    from verifiers.v1.trace import Error

logger = logging.getLogger(__name__)


def backoff(attempt: int) -> float:
    """Exponential backoff with full jitter (same curve as `retrying()`'s)."""
    return min(0.5 * 2**attempt, 30.0) * (0.5 + random.random())


def provider_retry_wait(
    error: ProviderError,
    failures: int,
    failing_for: float,
    retry: ProviderRetryConfig,
    *,
    base: float,
) -> float | None:
    """How long to wait before sending a model call the provider failed again, or None
    when the failure is final: it is not `transient`, or the wait would end past
    `retry.budget`, counted from the call's first failure (`failing_for` seconds ago).
    Exponential backoff from `base` seconds with jitter, or the provider's
    `Retry-After` when longer, each capped at `retry.max_backoff`."""
    if not transient(error):
        return None
    wait = min(base * 2**failures, retry.max_backoff)
    wait *= random.uniform(0.5, 1.0)
    if error.retry_after is not None:
        wait = max(wait, min(error.retry_after, retry.max_backoff))
    if failing_for + wait > retry.budget:
        return None
    return wait


def retrying(
    *,
    on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    give_up: type[BaseException] | tuple[type[BaseException], ...] = (),
    when: Callable[[BaseException], bool] | None = None,
    retries: int,
    initial: float = 0.5,
    label: str | None = None,
) -> AsyncRetrying:
    """The shared retry policy: retry on `on` (minus `give_up`, and only errors `when`
    accepts) up to `retries` times with exponential backoff from `initial` seconds +
    jitter, logging each retry. `label` names the operation in the log; omitted, it
    falls back to the retried callable's name, so the `async for attempt in
    retrying(...)` form should pass it."""

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

    retry = retry_if_exception_type(on) & retry_if_not_exception_type(give_up)
    if when is not None:
        retry = retry & retry_if_exception(when)
    return AsyncRetrying(
        stop=stop_after_attempt(retries + 1),
        wait=wait_exponential_jitter(initial=initial, max=30),
        retry=retry,
        before_sleep=_log,
        reraise=True,
    )


class RetryState:
    """Per-run budgets, shared across attempts but never across concurrent runs."""

    def __init__(self, config: RetryConfig) -> None:
        self.rules = [*config.rules, RetryRule(max_retries=config.max_retries)]
        # Each retry consumes one finite budget, so their sum bounds the run.
        self.max_retries = sum(rule.max_retries for rule in self.rules)
        self.used = [0] * len(self.rules)
        self.patterns = [
            re.compile(rule.message) if rule.message is not None else None
            for rule in self.rules
        ]

    def next_error(self, errors: Iterable[Error]) -> Error | None:
        """Consume one retry for the first eligible error in capture order.

        A denied/exhausted first match shadows later rules for that error only.
        Return the triggering error so logs identify the actual retry cause.
        """
        for error in errors:
            for index, rule in enumerate(self.rules):
                if rule.type is not None and error.type != rule.type:
                    continue
                if rule.status_code is not None and (
                    error.status_code is None
                    or not any(
                        error.status_code == status
                        if isinstance(status, int)
                        else error.status_code // 100 == int(status[0])
                        for status in rule.status_code
                    )
                ):
                    continue
                pattern = self.patterns[index]
                if pattern is not None and pattern.search(error.message) is None:
                    continue
                if self.used[index] < rule.max_retries:
                    self.used[index] += 1
                    return error
                break
        return None


async def run_episode_with_retry(
    run: Callable[[], Awaitable[Episode]],
    retry: RetryConfig,
) -> Episode:
    """Run one episode (each attempt mints a fresh one), retrying while
    it ends with a retryable error. When the final attempt fails too, the earlier
    attempts' errors are prepended so the episode shows the full history; a final
    good attempt returns clean."""
    history: list = []
    state = RetryState(retry)
    for attempt in range(state.max_retries + 1):
        final = await run()
        if attempt == state.max_retries or final.ok:
            break
        # Successful traces may contain recovered failures from agent retries.
        errors = list(final.errors)
        errors.extend(e for trace in final.traces if not trace.ok for e in trace.errors)
        cause = state.next_error(errors)
        if cause is None:
            break
        history.extend(final.errors)
        for trace in final.traces:
            history.extend(trace.errors)
        delay = backoff(attempt)
        logger.warning(
            "retrying episode %s (retry %d/%d) in %.1fs after error: %s",
            final.id,
            attempt + 1,
            state.max_retries,
            delay,
            cause.type,
        )
        await asyncio.sleep(delay)
    if history:
        # The full history rides the final episode either way; success is the
        # `ok` stamp, never errors-emptiness. In place: the envelope, the stamp,
        # and every consumer share this one list.
        final.errors[:0] = history
    return final


# -- interaction retries (`AgentConfig.interaction_retries`) ---------------------

RETRY_NOTE = (
    "The host reopened this session after an infrastructure failure. The conversation "
    "above is intact. Files are as they were at the last saved point of this "
    "conversation; background processes you started are gone, so restart any you "
    "still need."
)
"""The user message an interaction retry continues with on a box restored from the
trace's latest checkpoint."""

KILLED_NOTE = (
    "The host reopened this session after your agent process was killed by a signal "
    "while the box stayed up. In this sandbox that almost always means your "
    "last command used more memory than the box has. Files are as they were at the last "
    "saved point of this conversation; background processes you started are gone. Do "
    "not run that command the same way again: find the step that needs the memory and "
    "make it smaller (fewer parallel jobs, streaming instead of loading whole files, a "
    "smaller build or data set), then continue."
)
"""The user message a run continues with after its agent process was killed
(`errors.HarnessKilled`)."""


def retry_note(error: Error | None) -> str:
    """The note a failed run continues with: `KILLED_NOTE` when its agent process was
    killed by a signal (`HarnessKilled`), else `RETRY_NOTE`."""
    if error is not None and error.type == "HarnessKilled":
        return KILLED_NOTE
    return RETRY_NOTE


MODEL_RETRY_NOTE = (
    "The host retried this turn after a model call failed. Nothing on the box "
    "changed; continue where you left off."
)
"""The user message an interaction retry continues with on the same box, after a
model-side failure that left part of the turn committed."""

HOST_NOTES = frozenset({RETRY_NOTE, KILLED_NOTE, MODEL_RETRY_NOTE})
"""The notes the host continues a run with: a continuation that opens on one no reply
answered drops it, so a run cut off again before the model replied gains no stack of
identical notes."""


def without_unanswered_notes(messages: list) -> list:
    """`messages` without the host notes at its end that no reply answered (a
    continuation cut off again before the model replied): the next continuation adds
    its own note, so the model sees one."""
    end = len(messages)
    while (
        end
        and messages[end - 1].role == "user"
        and isinstance(messages[end - 1].content, str)
        and messages[end - 1].content in HOST_NOTES
    ):
        end -= 1
    return messages[:end]


RETRYABLE_SANDBOX_KINDS = frozenset({"box_lost", "io"})
"""`SandboxError.kind`s an interaction retry recovers from on a fresh box."""


def retryable_error(error: BaseException) -> bool:
    """Whether an interaction retry recovers from `error`: an infrastructure failure a
    fresh box (or a fresh model call) can get past — a lost box or failed transfer, a
    provider, harness or interception failure. Never a task's own failure."""
    if isinstance(error, TaskError):
        return False
    if isinstance(error, SandboxError):
        return getattr(error, "kind", None) in RETRYABLE_SANDBOX_KINDS
    return isinstance(error, (ProviderError, HarnessError, InterceptionError))
