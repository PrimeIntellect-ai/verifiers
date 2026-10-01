"""Whole-rollout retries (per-call model/runtime retries are owned by the SDKs, not us).

Transient model-call and runtime faults are retried by the harness/runtime SDKs; the
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
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from verifiers.v1.configs.retries import RetryConfig, RetryRule

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


class RetryState:
    """Per-run budgets, shared across attempts but never across concurrent runs."""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config
        self.rules = (
            config.rules
            if config.rules is not None
            else [RetryRule(max_retries=config.max_retries)]
        )
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
        if sum(self.used) >= self.config.max_retries:
            return None
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
    for attempt in range(retry.max_retries + 1):
        final = await run()
        if attempt == retry.max_retries or final.ok:
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
            retry.max_retries,
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
