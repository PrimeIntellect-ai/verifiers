"""A single interception server: the custom (bring-your-own-endpoint) shape.

No pool — one `InterceptionServer` bound to the configured endpoint, shared by every rollout (each
registers under its own secret, which is what the server routes by). The BYO endpoint is one URL,
so there's nothing to grow or multiplex; `acquire` just hands out a slot on the one server.
"""

import logging
from contextlib import asynccontextmanager
from typing import Self

from verifiers.v1.interception.base import Interception
from verifiers.v1.interception.config import CustomInterceptionConfig
from verifiers.v1.interception.server import InterceptionServer, RolloutSession
from verifiers.v1.interception.tunnel import CustomTunnel
from verifiers.v1.runtimes import RuntimeConfig, runtime_is_local

logger = logging.getLogger(__name__)


class SingleInterception(Interception):
    """One interception server at the custom BYO endpoint, shared by every rollout (no pool, no
    multiplex). The server binds the configured `port` on all interfaces; the harness reaches it at
    the configured `url` (or localhost, for a local harness runtime)."""

    def __init__(
        self, runtime_config: RuntimeConfig, config: CustomInterceptionConfig
    ) -> None:
        super().__init__()
        self.is_local = runtime_is_local(runtime_config)
        self.config = config
        self._server: InterceptionServer  # both set in __aenter__, live for the run
        self._base_url: str

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self._server = InterceptionServer(CustomTunnel(self.config))
        await self._stack.enter_async_context(self._server)
        self._base_url = await self._stack.enter_async_context(
            self._server.reachable(is_local=self.is_local)
        )
        logger.info("interception: single custom server at %s", self._base_url)
        return self

    @asynccontextmanager
    async def acquire(self, session: RolloutSession):
        """Register `session` on the one server and yield its `(endpoint, secret, port, base_url)`;
        free the slot on exit. See `InterceptionPool.acquire` for the tuple's meaning."""
        secret = self._server.register(session)
        try:
            yield (
                f"{self._base_url}/v1",
                secret,
                self._server.port,
                self._base_url,
            )
        finally:
            self._server.unregister(secret)
