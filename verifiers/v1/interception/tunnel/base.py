"""The Tunnel contract: expose a host interception port to a remote harness runtime.

The interception server runs on the host. A *local* harness (host network) reaches it at localhost —
no tunnel, whatever the config. A *remote* harness needs the port published outward: that's a
`Tunnel`. It says where the server must bind for the tunnel to reach it (`bind_host`/`bind_port`) and
`expose`s the bound port as a public URL. So a tunnel knows nothing about locality — the caller (the
interception server / `reachable_url`) uses it only in the remote case. The host-side counterpart to
a `Runtime`: `Runtime.expose` publishes a port *inside* a sandbox; `Tunnel.expose` a *host* port.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from verifiers.v1.interception.config import BaseInterceptionConfig

ConfigT = TypeVar("ConfigT", bound=BaseInterceptionConfig)


class Tunnel(ABC, Generic[ConfigT]):
    """Exposes a host interception port to a remote harness runtime. One per interception type (the
    pool picks it by config); lightweight and stateless beyond the config it holds (generic over its
    config type, so `self.config` is typed per subclass)."""

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

    @abstractmethod
    def expose(self, port: int) -> contextlib.AbstractAsyncContextManager[str]:
        """An async context manager yielding a public URL that reaches the host's `port` from a
        remote runtime. Entered only when the harness runtime is remote; torn down on exit."""
