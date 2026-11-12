"""Client interfaces for model inference and relay."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from verifiers.v1.configs.client import (
    BaseClientConfig,
    ClientConfig,
    TrainClientConfig,
)
from verifiers.v1.configs.sampling import Sampling, SamplingConfig
from verifiers.v1.dialects import Dialect
from verifiers.v1.dialects.base import StreamParser, is_sse_done_event
from verifiers.v1.errors import ProviderError, RolloutError
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response

SESSION_ID_HEADER = "X-Session-ID"
"""Per-rollout routing header (the trace id, same value every turn), so a session-affinity
router pins a rollout's turns to one engine and its growing prefix stays KV-cached."""


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


async def collect_stream(
    dialect: Dialect,
    reply: RelayReply,
    *,
    parser: StreamParser | None = None,
    events: bytearray | None = None,
) -> Response:
    """Read a relayed provider stream whole into the assembled response, then close it.
    Comment-only events (the provider's own keepalives) are skipped; the rest are appended
    to `events` when given, for a caller that serves them on."""
    parser = parser or dialect.stream_parser()
    saw_terminal = False
    try:
        async for chunk in reply.chunks:
            if not any(line.startswith(b"data:") for line in chunk.splitlines()):
                continue
            if events is not None:
                events += chunk
            saw_terminal |= dialect.is_terminal_event(chunk)
            if parser.on_done is not None and is_sse_done_event(chunk):
                parser.on_done()
            parser.feed(chunk)
        if not saw_terminal:
            raise ProviderError("upstream stream ended before its terminal event")
        return parser.finish()
    except RolloutError:
        raise
    except Exception as e:  # a malformed provider stream
        raise ProviderError(str(e)) from e
    finally:
        await reply.close()


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        sampling: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Run one completion -> a vf `Response`. `body` is the final effective native
        request after overrides and policy mediation: the eval client forwards it unchanged,
        while the train client renders it to token ids using the resolved `sampling` config.
        `session_id` is the rollout's trace id (sent as `SESSION_ID_HEADER`); `turn` is the
        graph-resolved prompt prefix, used by train clients for renderer bridging."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        """Stream a response for the final effective `body`, relaying the provider's bytes.
        Only the relay (eval) client supports it; the renderer generates and cannot stream."""
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
