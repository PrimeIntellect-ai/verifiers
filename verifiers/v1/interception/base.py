"""The interception contract: hand each rollout a slot on a host interception server.

Three shapes, picked by `InterceptionConfig` type (see `make_interception`): a single
`InterceptionServer`, a fixed `StaticInterceptionPool`, or an on-demand
`ElasticInterceptionPool`. From the outside they behave the same: start/stop (or the async
context manager wrapping them) bound the lifecycle — one eval, one topology run, or one
agent `.run` — and each rollout `acquire`s a slot and frees it. An `Interception` can be
shared: whoever entered it owns the lifecycle; borrowers only `acquire`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Collection
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from verifiers.v1.interception.server import InterceptionServer
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.trace import LiveBox


# (base_url, model_secret, state_secret): model inference and task state deliberately use
# separate capabilities. `base_url` is universally reachable — the interception is exposed
# whenever any consumer is remote.
Slot = tuple[str, str, str]


class Interception(ABC):
    """How rollouts reach the host interception server. `start` brings the servers up (or
    arms lazy growth), `stop` tears every server (+ its tunnel) down via `stack` — LIFO,
    even if one teardown fails; `async with` wraps the two. Each rollout `acquire`s a slot
    and frees it on exit."""

    state_service_secrets: Collection[str] = ()

    def __init__(self) -> None:
        self.stack = AsyncExitStack()
        self._adopted: dict[str, tuple[InterceptionServer, int]] = {}
        """Per adopted tunnel, the server holding it and how many rollouts use it."""
        self._adopting = asyncio.Lock()

    @abstractmethod
    async def start(self) -> None:
        """Bring the interception up; resources land on `stack` so `stop` frees them."""

    async def stop(self) -> None:
        await self.stack.aclose()

    async def __aenter__(self) -> Self:
        try:
            await self.start()
        except BaseException:
            # unwind whatever `start` already put on the stack
            await self.stop()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @abstractmethod
    def acquire(self, session: RolloutSession) -> AbstractAsyncContextManager[Slot]:
        """Register `session` on a server (bringing one up if needed) and yield its `Slot`;
        free it on exit."""

    @asynccontextmanager
    async def adopt(
        self, session: RolloutSession, live: LiveBox
    ) -> AsyncIterator[Slot]:
        """A slot for a rollout taken over in its live box (`Trace.live`): under the
        secrets its program holds, on a server at the URL the program reaches — its
        tunnel, adopted from the process that registered it. Rollouts behind one tunnel
        share one server (one frpc can hold a tunnel; a second would fight it), up while
        any of them runs."""
        from verifiers.v1.errors import TunnelError
        from verifiers.v1.interception.server import InterceptionServer
        from verifiers.v1.interception.tunnel.prime import AdoptedPrimeTunnel

        if live.tunnel_id is None or live.tunnel is None:
            raise TunnelError("the live box's program reaches no tunnel to adopt")
        tunnel_id = live.tunnel_id
        async with self._adopting:
            held = self._adopted.get(tunnel_id)
            if held is None:
                server = InterceptionServer(
                    state_service_secrets=self.state_service_secrets,
                    tunnel=AdoptedPrimeTunnel(live.tunnel),
                )
                try:
                    await server.start()
                except BaseException:
                    await server.stop()
                    raise
                held = (server, 0)
            server, users = held
            self._adopted[tunnel_id] = (server, users + 1)
            keys = server.register(session, (live.model_secret, live.state_secret))
        try:
            yield server.base_url, *keys
        finally:
            server.unregister(*keys)
            async with self._adopting:
                server, users = self._adopted[tunnel_id]
                if users > 1:
                    self._adopted[tunnel_id] = (server, users - 1)
                else:
                    del self._adopted[tunnel_id]
                    await server.stop()
