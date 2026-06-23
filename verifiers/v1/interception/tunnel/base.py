"""The Tunnel contract: make the host interception server reachable from a remote harness.

The interception server runs on the host; a harness in a remote runtime reaches it over a tunnel.
`Tunnel` is that contract — how to build the reachable URL, and what address/port the server must
bind. Concrete tunnels live alongside this base; `make_tunnel` builds the one matching a config's
type. The host-side counterpart to a `Runtime`: a `Runtime.expose` publishes a port *inside* a
sandbox; a `Tunnel.expose` publishes a *host* port outward.
"""

import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from verifiers.v1.interception.config import BaseInterceptionConfig


class Tunnel(ABC):
    """Makes the host interception server reachable from a remote harness runtime. One per
    interception type, built from its config by `make_tunnel`; lightweight and stateless beyond
    the config it holds."""

    bind_host: ClassVar[str] = "127.0.0.1"
    """Address the interception server must bind for this tunnel to reach it. Loopback by default —
    frpc / a BYO proxy connect over localhost on the same host."""

    def __init__(self, config: "BaseInterceptionConfig") -> None:
        self.config = config

    @property
    def bind_port(self) -> int:
        """Fixed local port the interception server must bind (0 = an ephemeral port)."""
        return 0

    @property
    def single_server(self) -> bool:
        """One server shared by every rollout (the pool never grows past it). True iff the server
        binds a fixed port — two servers can't share one, so a fixed port is a single endpoint
        (only `CustomTunnel`, a single BYO URL)."""
        return self.bind_port != 0

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
