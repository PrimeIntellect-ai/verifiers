"""The Tunnel contract: expose a host interception port to a remote consumer.

The interception server runs on the host. Local runtimes use a host-local URL directly or
through their runtime translation; remote runtimes need the port published outward. A
`Tunnel` supplies the bind address and exposes that host port as a public URL. It is the
host-side counterpart to `Runtime.expose`, which publishes a port inside a sandbox.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic_config import BaseConfig


class BaseTunnelConfig(BaseConfig):
    """Base for the tunnel types — the discriminated union's common type. Per-type fields
    live on the subclasses (custom's `url`/`port`)."""


ConfigT = TypeVar("ConfigT", bound=BaseTunnelConfig)


class Tunnel(ABC, Generic[ConfigT]):
    """Exposes a host interception port to a remote consumer. Lightweight and stateless
    beyond the config it holds (generic over its config type, so `self.config` is typed
    per subclass)."""

    bind_host: ClassVar[str] = "127.0.0.1"
    """Interface the interception server binds for this tunnel to reach it. Loopback by
    default — frpc reaches it over localhost; `CustomTunnel` binds all interfaces so a
    remote consumer can reach it directly (or via a proxy)."""

    def __init__(self, config: ConfigT | None = None) -> None:
        self.config = config

    @property
    def bind_port(self) -> int:
        """Fixed local port the interception server must bind (0 = an ephemeral port)."""
        return 0

    @abstractmethod
    def expose(self, port: int) -> contextlib.AbstractAsyncContextManager[str]:
        """An async context manager yielding a public URL that reaches the host's `port`
        from a remote runtime. Entered only when a consumer is remote; torn down on exit.
        A setup failure raises `TunnelError`; an error raised while the URL is held (the
        caller's body) propagates unchanged."""
