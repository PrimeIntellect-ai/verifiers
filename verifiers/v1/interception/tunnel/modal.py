"""Modal tunnel: expose the host interception port via Modal's own forwarding (`modal.forward`).

Requires the framework to run inside a Modal container — `modal.forward` raises `InvalidError`
otherwise. Binds `0.0.0.0`: Modal forwards to the container's routable interface, not its loopback,
so a loopback-only server is unreachable. Safe — that server runs inside an isolated container whose
only ingress is the tunnel itself.
"""

import contextlib
from collections.abc import AsyncIterator
from typing import ClassVar

from verifiers.v1.interception.tunnel.base import Tunnel


class ModalTunnel(Tunnel):
    bind_host: ClassVar[str] = "0.0.0.0"

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        try:
            import modal
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "modal interception requires the Modal SDK; install `verifiers[modal]`."
            ) from e
        async with modal.forward(port) as tunnel:
            yield str(tunnel.url).rstrip("/")
