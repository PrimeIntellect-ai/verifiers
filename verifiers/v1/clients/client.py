"""Client interfaces for model inference and relay."""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass

from verifiers.v1.dialects import Dialect
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

    async def relay_aux(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict:
        """Relay a non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`)
        as native JSON and return the provider JSON. Only the relay (eval) client supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not relay aux routes")

    async def close(self) -> None:
        pass


@dataclass(frozen=True)
class ModelContext:
    """Client, model, and sampling settings for one rollout."""

    model: str
    client: Client
    sampling: Sampling
