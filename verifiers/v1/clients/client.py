"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import Messages, Response, SamplingConfig, Tool


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Run one completion, translating to/from this client's wire format."""

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


class RetryingClient(Client):
    """Wraps a client to retry each completion on a transient `ModelError` (tenacity,
    `max_attempts` total). An `OverlongPromptError` is never retried — it's a budget
    limit the interception server turns into a clean truncation, not a transient fault."""

    def __init__(self, inner: Client, max_attempts: int) -> None:
        self.inner = inner
        # One Retrying, reused across (and concurrent within) calls: the control flow runs
        # off a per-call RetryCallState, so only its bookkeeping `.statistics` is shared.
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            retry=retry_if_exception_type(ModelError)
            & retry_if_not_exception_type(OverlongPromptError),
            reraise=True,
        )

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingConfig,
        tools: list[Tool] | None = None,
    ) -> Response:
        return await self._retrying(
            self.inner.get_response, prompt, model, sampling_args, tools
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
