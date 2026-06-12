"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import ModelError, OverlongPromptError
from verifiers.v1.types import Response, SamplingConfig

logger = logging.getLogger(__name__)


@dataclass
class RelayReply:
    """A relayed upstream response streamed back: its content type + body chunks (one chunk for
    a JSON body, many for SSE). The connection closes when `chunks` is exhausted."""

    content_type: str
    chunks: AsyncIterator[bytes]


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> Response:
        """Run one completion -> a vf `Response`. The proxy client forwards `body` 1:1 and
        parses the provider response via `dialect` (carrying the raw on `Response.raw`); the
        renderer derives the typed prompt from `body` via `dialect` and tokenizes it."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> RelayReply:
        """Stream a (possibly SSE) response back, relaying the provider's bytes — the proxy's
        path for a streaming request. Only the relay (eval) client supports it; the renderer
        generates and cannot stream."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

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
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> Response:
        return await self._retrying(
            self.inner.get_response, dialect, body, model, sampling_args
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
    ) -> RelayReply:
        # Safe to retry: relay raises (and is retried) before any response byte is handed back;
        # once a `RelayReply` is returned, streaming is already underway.
        return await self._retrying(
            self.inner.relay, dialect, body, model, sampling_args
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
