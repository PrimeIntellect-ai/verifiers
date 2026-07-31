"""The Tunnel contract: expose a host interception port to a remote consumer.

The interception server runs on the host. Local runtimes use a host-local URL directly or
through their runtime translation; remote runtimes need the port published outward. A
`Tunnel` supplies the bind address and exposes that host port as a public URL. It is the
host-side counterpart to `Runtime.expose`, which publishes a port inside a sandbox.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Protocol, TypeVar

from pydantic_config import BaseConfig


class BaseTunnelConfig(BaseConfig):
    """Base for the tunnel types — the discriminated union's common type. Per-type fields
    live on the subclasses (custom's `url`/`port`)."""


ConfigT = TypeVar("ConfigT", bound=BaseTunnelConfig)


class Endpoint(Protocol):
    """A live exposed endpoint, valid while its `Tunnel.expose` is entered. A healing
    tunnel re-mints at a NEW URL when it finds itself dead, so consumers ask `url()` per
    acquire instead of caching a snapshot."""

    async def url(self) -> str:
        """The current public URL, healing a dead tunnel first (so it may differ across
        calls). Raises `TunnelError` when it can't be re-established."""
        ...

    async def healthy(self, url: str) -> bool:
        """Whether the endpoint a consumer reached at `url` is still up — the
        failure-attribution hook: `False` means that tunnel died, so the consumer's
        concurrent failure was the tunnel's fault. Anchored to `url` rather than bare
        liveness because a heal re-mints at a new URL: a stale `url` proves its tunnel
        died even when a replacement is already up. Must not raise; an inconclusive
        probe reads as healthy."""
        ...


class FixedEndpoint:
    """An `Endpoint` at a URL the framework doesn't manage: nothing to probe or heal —
    the operator owns it."""

    def __init__(self, url: str) -> None:
        self._url = url

    async def url(self) -> str:
        return self._url

    async def healthy(self, url: str) -> bool:
        return True


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
    def expose(self, port: int) -> contextlib.AbstractAsyncContextManager[Endpoint]:
        """An async context manager yielding an `Endpoint` whose `url()` reaches the
        host's `port` from a remote runtime. Entered only when a consumer is remote; torn
        down on exit. A setup failure raises `TunnelError`; an error raised while the
        endpoint is held (the caller's body) propagates unchanged."""
