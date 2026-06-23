"""Prime tunnel: expose the host interception port via prime_tunnel (frpc) — the shared host
endpoint. The default; works from any host with prime credentials, for harnesses in prime *or*
modal sandboxes alike."""

import contextlib
from collections.abc import AsyncIterator

from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.runtimes.base import host_endpoint


class PrimeTunnel(Tunnel):
    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        async with host_endpoint(port, is_local=False) as url:
            yield url
