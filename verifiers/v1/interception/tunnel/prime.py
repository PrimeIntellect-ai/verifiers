"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in prime *or* modal sandboxes
alike — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Literal

import httpx

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Tunnel
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process for the user that opens one. One user-global
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512
TUNNEL_LIMITER = creation_limiter(_TUNNELS_PER_MIN / 60, "prime-tunnel")

logger = logging.getLogger(__name__)


class PrimeTunnelConfig(BaseTunnelConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). No fields — the tunnel
    service mints a fresh public URL per exposed port."""

    type: Literal["prime"] = "prime"


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    def __init__(self, config: PrimeTunnelConfig | None = None) -> None:
        ensure_prime_auth()
        super().__init__(config)
        self._client = None
        self._lock = asyncio.Lock()

    async def _start(self, port: int):
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        label = f"host tunnel (port {port})"
        client = None
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=port)
                    async with TUNNEL_LIMITER:
                        url = str(await client.start()).rstrip("/")
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        return client, url

    async def healthy(self) -> bool | None:
        """Check both the local frpc process and its backend registration."""
        client = self._client
        if client is None or not client.is_running:
            return False
        try:
            if not await client.check_registered():
                return False
        except Exception as e:  # noqa: BLE001 - backend failure means unknown health
            logger.debug("prime tunnel health check unavailable: %s", e)
            return None
        url = str(client.url or "").rstrip("/")
        if not url:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as probe:
                response = await probe.get(f"{url}/state")
        except httpx.HTTPError as e:
            logger.debug("prime tunnel endpoint probe unavailable: %s", e)
            return None
        # The state route requires a bearer. Its expected 401 proves that the request
        # crossed the public route and reached this interception server.
        return response.status_code != 404 and response.status_code < 500

    async def reconnect(self, port: int) -> str | None:
        """Replace an idle stale registration. A replacement gets a new public URL."""
        async with self._lock:
            old, self._client = self._client, None
            if old is not None:
                with contextlib.suppress(Exception):
                    await run_shielded(old.stop())
            client, url = await self._start(port)
            self._client = client
            return url

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and globally rate-capped (512/min, user-wide via the shared
        `TUNNEL_LIMITER`), so transient failures are retried; a terminal one raises
        `TunnelError`. The tunnel is torn down on exit."""
        async with self._lock:
            client, url = await self._start(port)
            self._client = client
        try:
            yield url
        finally:
            # Run the synchronous stop to completion even under cancellation (`run_shielded`
            # re-raises the cancellation after); tunnel-stop failures are best-effort.
            async with self._lock:
                client, self._client = self._client, None
                if client is not None:
                    with contextlib.suppress(Exception):
                        await run_shielded(asyncio.to_thread(client.sync_stop))
