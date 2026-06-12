"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import Messages, Response, SamplingConfig, Tool

logger = logging.getLogger(__name__)


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        body: dict,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> tuple[dict, Response]:
        """Run one completion: the OpenAI chat.completion dict to hand the program, and the
        typed `Response` for the trace."""

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


class RetryingClient(Client):
    """Wraps a client to retry each completion on a transient `ModelError` (tenacity, up to
    `max_retries` retries). An `OverlongPromptError` is never retried — it's a budget limit
    the interception server turns into a clean truncation, not a transient fault."""

    def __init__(self, inner: Client, max_retries: int) -> None:
        self.inner = inner
        self.max_retries = max_retries
        # One Retrying, reused across (and concurrent within) calls: the control flow runs
        # off a per-call RetryCallState, so only its bookkeeping `.statistics` is shared.
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            retry=retry_if_exception_type(ModelError)
            & retry_if_not_exception_type(OverlongPromptError),
            before_sleep=self._log_retry,
            reraise=True,
        )

    def _log_retry(self, state: RetryCallState) -> None:
        logger.warning(
            "retrying model call (attempt %d/%d) after error: %s",
            state.attempt_number,
            self.max_retries + 1,
            state.outcome.exception(),
        )

    async def get_response(
        self,
        body: dict,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> tuple[dict, Response]:
        return await self._retrying(
            self.inner.get_response, body, prompt, model, sampling_args, tools
        )

    async def close(self) -> None:
        await self.inner.close()


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs (client + model + sampling), bundled
    so harnesses hold no rollout state. Built by the Environment."""

    client: Client
    model: str
    sampling: SamplingConfig
