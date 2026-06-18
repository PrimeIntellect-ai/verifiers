"""The client abstraction: turn a prompt into a `Response`.

Collapsed from v1's 4-typevar generic ABC with five conversion hooks to a single
abstract method. Each concrete client owns its own wire translation internally.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from verifiers.v1.dialects import Dialect
from verifiers.v1.errors import OverlongPromptError, ProviderError
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig

logger = logging.getLogger(__name__)

SESSION_ID_HEADER = "X-Session-ID"
"""Per-rollout routing header. Every turn of one rollout sends the same value (the trace id),
so a session-affinity router (e.g. vLLM's ``consistent_hash`` policy keyed on its
``request_id_headers``) pins all of a rollout's turns to the same engine — keeping the
growing cross-turn prefix warm in that engine's KV cache instead of re-prefilling it
cold on a random shard each turn."""


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Run one completion -> a vf `Response`. The eval client forwards the native JSON and
        eligible end-to-end headers, then parses a copy via `dialect`; the train client derives
        the typed prompt from `body` and tokenizes it.

        `session_id` is the rollout's stable id (the trace id); when set, the client sends it
        as the `SESSION_ID_HEADER` so a session-affinity router keeps the rollout's turns on
        one engine for cross-turn prefix-cache reuse. `turn` is the graph-resolved prompt
        prefix; train clients may use it for renderer bridging, while relay clients ignore it."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        """Stream a (possibly SSE) response back, relaying the provider's bytes — the proxy's
        path for a streaming request. Only the relay (eval) client supports it; the renderer
        generates and cannot stream."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        """Relay a non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`)
        verbatim to the provider and return its JSON. Only the relay (eval) client supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not relay aux routes")

    async def close(self) -> None:
        """Release any underlying resources. Default no-op."""


class RetryingClient(Client):
    """Wraps a client to retry each completion on a transient `ProviderError` (tenacity, up to
    `max_retries` retries) with exponential backoff + jitter between attempts — so retries don't
    fire back-to-back into the same brief endpoint outage, and concurrent rollouts hitting a shared
    endpoint stagger their retries instead of re-thundering it. An `OverlongPromptError` is never
    retried — it's a budget limit the interception server turns into a clean truncation, not a
    transient fault."""

    def __init__(self, inner: Client, max_retries: int) -> None:
        self.inner = inner
        self.max_retries = max_retries
        # One Retrying, reused across (and concurrent within) calls: the control flow runs
        # off a per-call RetryCallState, so only its bookkeeping `.statistics` is shared.
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=30),
            retry=retry_if_exception_type(ProviderError)
            & retry_if_not_exception_type(OverlongPromptError),
            before_sleep=self._log_retry,
            reraise=True,
        )

    def _log_retry(self, state: RetryCallState) -> None:
        # before_sleep fires after a failed attempt, before the imminent retry — so
        # attempt_number is the retry index (1 on the first retry); count out of max_retries.
        exc = state.outcome.exception()
        logger.warning(
            "retrying model call (retry %d/%d) after error: %s",
            state.attempt_number,
            self.max_retries,
            f"{type(exc).__name__}: {exc}",  # name too — some errors stringify empty
        )

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return await self._retrying(
            self.inner.get_response,
            dialect,
            body,
            model,
            sampling_args,
            session_id=session_id,
            turn=turn,
            headers=headers,
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        # Safe to retry: relay raises (and is retried) before any response byte is handed back;
        # once a `RelayReply` is returned, streaming is already underway.
        return await self._retrying(
            self.inner.relay,
            dialect,
            body,
            model,
            sampling_args,
            headers=headers,
            session_id=session_id,
        )

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        return await self._retrying(self.inner.relay_aux, dialect, route, body)

    async def close(self) -> None:
        await self.inner.close()


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs (client + model + sampling), bundled
    so harnesses hold no rollout state. Built by the Environment."""

    model: str
    client: Client
    sampling: Sampling
