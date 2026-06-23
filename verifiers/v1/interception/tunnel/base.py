"""The Tunnel contract: make the host interception server reachable from a remote harness.

The interception server runs on the host; a harness in a remote runtime reaches it over a tunnel.
`Tunnel` is that contract — how to build the reachable URL, and what address/port the server must
bind. Concrete tunnels live alongside this base; the interception pool picks the one matching its
config's type. The host-side counterpart to a `Runtime`: a `Runtime.expose` publishes a port
*inside* a sandbox; a `Tunnel.expose` publishes a *host* port outward.
"""

import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar, Generic, TypeVar

from verifiers.v1.interception.config import BaseInterceptionConfig

ConfigT = TypeVar("ConfigT", bound=BaseInterceptionConfig)


class Tunnel(ABC, Generic[ConfigT]):
    """Makes the host interception server reachable from a remote harness runtime. One per
    interception type (the pool picks it by config); lightweight and stateless beyond the config it
    holds (generic over its config type, so `self.config` is typed per subclass)."""

    bind_host: ClassVar[str] = "127.0.0.1"
    """Interface the interception server binds for this tunnel to reach it. Loopback by default —
    frpc reaches it over localhost; `CustomTunnel` binds all interfaces so a remote harness can reach
    it directly (or via a proxy)."""

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = config

    @property
    def bind_port(self) -> int:
        """Fixed local port the interception server must bind (0 = an ephemeral port)."""
        return 0

    @contextlib.asynccontextmanager
    async def reachable(self, port: int, *, is_local: bool) -> AsyncIterator[str]:
        """Yield a URL the harness's runtime uses to reach the host interception server on `port`. A
        local runtime shares the host network → localhost (no tunnel, whatever the type); a remote
        one is bridged by `expose`."""
        if is_local:
            yield f"http://127.0.0.1:{port}"
        else:
            async with self.expose(port) as url:
                yield url

    @abstractmethod
    def expose(self, port: int) -> contextlib.AbstractAsyncContextManager[str]:
        """An async context manager yielding a public URL that reaches the host's `port` from a
        remote runtime. Entered only when the harness runtime is remote; torn down on exit."""
