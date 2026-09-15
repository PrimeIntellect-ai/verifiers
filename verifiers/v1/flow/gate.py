"""One inference budget for a flow run: every model request of every rollout,
nested harness calls included, holds a permit of one semaphore while upstream."""

import asyncio

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.configs.client import BaseClientConfig
from verifiers.v1.dialects import Dialect
from verifiers.v1.interception.server import InterceptionServer


class LimitedClient(Client):
    def __init__(self, client: Client, gate: asyncio.Semaphore):
        self.client, self.gate = client, gate

    async def get_response(self, dialect: Dialect, *args, **kwargs):
        async with self.gate:
            return await self.client.get_response(dialect, *args, **kwargs)

    async def relay(self, dialect: Dialect, *args, **kwargs):
        """A streamed reply holds its permit until it closes."""
        await self.gate.acquire()
        try:
            reply = await self.client.relay(dialect, *args, **kwargs)
        except BaseException:
            self.gate.release()
            raise
        return self._releasing(reply)

    async def relay_aux(self, dialect: Dialect, *args, **kwargs):
        return await self.client.relay_aux(dialect, *args, **kwargs)

    def _releasing(self, reply: RelayReply) -> RelayReply:
        closed = False

        async def close():
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


class Workers(InterceptionServer):
    """The run's one interception server; every session's client rides the gate."""

    def __init__(self, count: int, *, requires_tunnel: bool = True):
        self.gate = asyncio.Semaphore(count)
        super().__init__(requires_tunnel=requires_tunnel)

    def _client(self, config: BaseClientConfig) -> Client:
        return LimitedClient(super()._client(config), self.gate)
