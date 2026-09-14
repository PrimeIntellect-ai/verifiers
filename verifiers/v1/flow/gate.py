"""One inference budget for a flow run, including nested harness calls, and the one
place every model request passes: the gate that bounds them (`Capacity`, a resizable
FIFO semaphore).

Ported from the proven upstream data-flywheel `workers.py` (upstream-comparison-871ae9d8):
`Capacity` and `LimitedClient._releasing` are copied with their semantics intact — the
cancel-release-on-waiter logic and the close-AND-error permit lifetime are the parts
campaign evidence paid for.
"""

import asyncio
from collections import deque
from typing import Self

import httpx

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.dialects import Dialect
from verifiers.v1.interception.server import InterceptionServer

UPSTREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)
"""The gated client's timeouts: only the connect can fire (the eval client's default
5 s read expired mid-stream on connections the kernel had completed upstream, surfacing
as provider 504s)."""


class Capacity:
    """A resizable FIFO semaphore: `limit` permits over `held`; a grow admits waiters
    at once, a shrink cancels none (it takes effect as permits return)."""

    def __init__(self, name: str, limit: int):
        self.name = name
        self.held = 0
        self.taken = 0
        """Permits granted since the start: a run's status reads the gate's as its calls."""
        self._waiters: deque[asyncio.Future] = deque()
        self.resize(limit)

    def free(self) -> int:
        return max(0, self.limit - self.held)

    @property
    def waiting(self) -> int:
        return sum(not waiter.done() for waiter in self._waiters)

    async def acquire(self) -> None:
        if not self._waiters and self.held < self.limit:
            self.held += 1
            self.taken += 1
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                self.release()
            elif waiter in self._waiters:  # `_admit` may have popped it as the cancel came
                self._waiters.remove(waiter)
            raise

    def release(self) -> None:
        if self.held < 1:
            raise RuntimeError(f"capacity {self.name} released more than acquired")
        self.held -= 1
        self._admit()

    def resize(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{self.name} must be a positive integer")
        self.limit = limit
        self._admit()

    def _admit(self) -> None:
        while self._waiters and self.held < self.limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.held += 1
                self.taken += 1
                waiter.set_result(None)

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc) -> None:
        self.release()


class LimitedClient(Client):
    """Every model request, including nested harness calls, takes one permit of the
    shared gate while it is upstream; a failure is the seat's (the harness SDK retries
    a provider error), never absorbed here."""

    def __init__(self, client: Client, gate: Capacity):
        self.client, self.gate = client, gate

    async def get_response(self, dialect: Dialect, *args, **kwargs):
        async with self.gate:
            return await self.client.get_response(dialect, *args, **kwargs)

    async def relay(self, dialect: Dialect, *args, **kwargs):
        """A streamed request holds its permit until the reply closes: once open its
        bytes are the seat's."""
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

        return RelayReply(content_type=reply.content_type, chunks=reply.chunks, close=close)


class Workers(InterceptionServer):
    """The run's one interception server under the inference gate: every seat rollout
    (agent and expand nodes alike) rides it, so nested harness calls are bounded too."""

    def __init__(self, count: int, *, requires_tunnel: bool = True):
        self.gate = Capacity("inference", count)
        super().__init__(requires_tunnel=requires_tunnel)

    def resize(self, count: int) -> None:
        """Set the gate's size; a shrink takes effect as in-flight requests return."""
        self.gate.resize(count)

    def wrap_client(self, client, client_config):
        """Every session's client is the server-owned one under the gate."""
        if isinstance(getattr(client, "client", None), httpx.AsyncClient):
            client.client.timeout = UPSTREAM_TIMEOUT
        return LimitedClient(client, self.gate)

    def register(self, session) -> tuple[str, str]:
        model_secret, state_secret = super().register(session)
        session.client = self.wrap_client(session.client, session.ctx.client)
        return model_secret, state_secret
