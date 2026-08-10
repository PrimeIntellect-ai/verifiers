"""Client interfaces for model inference and relay."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from verifiers.v1.configs.client import (
    BaseClientConfig,
    ClientConfig,
    TrainClientConfig,
)
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig

SESSION_ID_HEADER = "X-Session-ID"
"""Per-rollout routing header (the trace id, same value every turn), so a session-affinity
router pins a rollout's turns to one engine and its growing prefix stays KV-cached."""


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


class Client(ABC):
    async def prepare_request_body(self, dialect: Dialect, body: dict) -> dict:
        """Normalize a provider request before the interception server parses/traces it.

        Relay clients keep the request verbatim. Training clients may rewrite heavy
        in-process payloads (for example base64 images) into stable run-asset refs so the
        trace, renderer, and trainer all see the same cheap message content.
        """
        return body

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
        """Run one completion -> a vf `Response`. The eval client forwards the native JSON
        and parses a copy via `dialect`; the train client renders `body` to token ids.
        `session_id` is the rollout's trace id (sent as `SESSION_ID_HEADER`); `turn` is the
        graph-resolved prompt prefix, used by train clients for renderer bridging."""

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


def resolve_client(config: BaseClientConfig) -> Client:
    if isinstance(config, TrainClientConfig):
        from verifiers.v1.clients.train import TrainClient

        return TrainClient(config)
    from verifiers.v1.clients.eval import EvalClient

    return EvalClient(config)


@dataclass(frozen=True)
class ModelContext:
    """Model, endpoint config, and sampling for one rollout."""

    model: str
    client: ClientConfig
    sampling: Sampling = field(default_factory=Sampling)
