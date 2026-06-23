"""Prime tunnel: expose the host interception port via prime_tunnel (frpc) — the shared host
endpoint. The default; works from any host with prime credentials, for harnesses in prime *or*
modal sandboxes alike."""

import contextlib
from collections.abc import AsyncIterator

from verifiers.v1.interception.tunnel.base import Tunnel


class PrimeTunnel(Tunnel):
    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation is
        network-bound and globally rate-capped (`prime_tunnel` — 512/min, host-wide via the shared
        `TUNNEL_LIMITER`), so transient failures are retried; the tunnel is torn down on exit. A
        terminal failure propagates raw — the call site classifies it as `TunnelError`."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.retries import retrying
        from verifiers.v1.runtimes.limiters import TUNNEL_LIMITER

        client = None
        async for attempt in retrying(retries=3, label=f"host tunnel (port {port})"):
            with attempt:
                client = TunnelClient(local_port=port)
                async with TUNNEL_LIMITER:
                    url = str(await client.start()).rstrip("/")
        try:
            yield url
        finally:
            with contextlib.suppress(Exception):
                client.sync_stop()
