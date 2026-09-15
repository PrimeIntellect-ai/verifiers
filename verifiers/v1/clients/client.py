"""Client interfaces for model inference and relay."""

import asyncio
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


class LimitedClient(Client):
    """A client whose every request holds one permit of a shared semaphore while it
    is upstream; a streamed reply holds its permit until it closes."""

    def __init__(self, client: Client, gate: asyncio.Semaphore) -> None:
        self.client, self.gate = client, gate

    async def get_response(self, dialect: Dialect, *args, **kwargs) -> Response:
        async with self.gate:
            return await self.client.get_response(dialect, *args, **kwargs)

    async def relay(self, dialect: Dialect, *args, **kwargs) -> RelayReply:
        await self.gate.acquire()
        try:
            reply = await self.client.relay(dialect, *args, **kwargs)
        except BaseException:
            self.gate.release()
            raise
        closed = False

        async def close() -> None:
            nonlocal closed
            if not closed:
                closed = True
                try:
                    await reply.close()
                finally:
                    self.gate.release()

        return RelayReply(
            content_type=reply.content_type, chunks=reply.chunks, close=close
        )

    async def relay_aux(self, dialect: Dialect, *args, **kwargs) -> dict:
        return await self.client.relay_aux(dialect, *args, **kwargs)

    async def close(self) -> None:
        await self.client.close()


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
