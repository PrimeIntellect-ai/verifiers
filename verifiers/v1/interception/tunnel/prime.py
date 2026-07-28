"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in prime *or* modal sandboxes
alike — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Endpoint, Tunnel
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from prime_tunnel import Tunnel as TunnelClient

logger = logging.getLogger(__name__)

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process on the host that opens one. One host-global
# limiter, not a per-runtime config knob.
TUNNELS_PER_MIN = 512
TUNNEL_LIMITER = creation_limiter(TUNNELS_PER_MIN / 60, "prime-tunnel")

# A registration can lapse server-side while frpc is still running locally, so `url()`
# also probes the tunnel service — but that's a network call, so at most this often.
# (`is_running`, a local process poll, guards every call.)
REGISTRATION_CHECK_INTERVAL = 60.0


class PrimeTunnelConfig(BaseTunnelConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). No fields — the tunnel
    service mints a fresh public URL per exposed port."""

    type: Literal["prime"] = "prime"


class PrimeEndpoint:
    """A self-healing prime tunnel: `url()` verifies the frpc process (and, throttled, the
    server-side registration) and re-mints the tunnel when either is gone — at a NEW public
    URL. Healing serves the *next* acquire; a consumer holding the old URL isn't saved (its
    URL is baked in) — `healthy()` is how its failure gets attributed to the tunnel."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._client: TunnelClient | None = None
        self._url = ""
        self._lock = asyncio.Lock()
        self._last_registration_check = 0.0

    async def url(self) -> str:
        async with self._lock:
            if self._client is not None and not await self._alive():
                await self.close()
            if self._client is None:
                await self._mint()
            return self._url

    async def healthy(self, url: str) -> bool:
        async with self._lock:
            # A re-mint changes the URL, so a stale `url` proves the consumer's tunnel
            # died — even when a concurrent acquire already healed past it. (Without the
            # anchor, probing the replacement would mask the death.)
            if url != self._url:
                return False
            # Torn down (or a heal already failed): it was dead.
            if self._client is None:
                return False
            if await self._alive(fresh=True):
                return True
            await self.close()  # observed dead: tear down so the next `url()` re-mints
            return False

    async def _alive(self, fresh: bool = False) -> bool:
        """Whether the tunnel is up: frpc running locally and (throttled — or forced with
        `fresh`, the attribution path) still registered server-side. An unreachable tunnel
        API is not a dead tunnel: the probe reads as alive."""
        assert self._client is not None
        client = self._client
        if not client.is_running:
            output = await asyncio.to_thread(lambda: "\n".join(client.recent_output))
            logger.warning("prime tunnel %s: frpc died\n%s", self._url, output)
            return False
        now = time.monotonic()
        if fresh or now - self._last_registration_check > REGISTRATION_CHECK_INTERVAL:
            self._last_registration_check = now
            try:
                # Not `client.check_registered()`: deletion is soft server-side (the
                # record survives as `status="terminated"`, and GET keeps returning it),
                # so existence alone reads a terminated tunnel as registered.
                info = await client._client.get_tunnel(client.tunnel_id)
                if info is None or info.status == "terminated":
                    logger.warning(
                        "prime tunnel %s: registration %s server-side",
                        self._url,
                        "terminated" if info else "gone",
                    )
                    return False
            except Exception as e:  # noqa: BLE001 - API unreachable is not a dead tunnel
                logger.warning(
                    "prime tunnel %s: liveness probe failed (%s), assuming alive",
                    self._url,
                    e,
                )
        return True

    async def _mint(self) -> None:
        """Register a fresh tunnel (network-bound and globally rate-capped — 512/min,
        host-wide via the shared `TUNNEL_LIMITER` — so transient failures are retried);
        a terminal one raises `TunnelError`."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.retries import retrying

        label = f"host tunnel (port {self.port})"
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=self.port)
                    async with TUNNEL_LIMITER:
                        self._url = str(await client.start()).rstrip("/")
                    self._client = client
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        logger.info("prime tunnel up: %s -> 127.0.0.1:%d", self._url, self.port)

    async def close(self) -> None:
        """Stop the *current* frpc client (it changes across heals). Runs the synchronous
        stop to completion even under cancellation (`run_shielded` re-raises the
        cancellation after); tunnel-stop failures are best-effort."""
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await run_shielded(asyncio.to_thread(client.sync_stop))


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[Endpoint]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc), self-healing
        for the lifetime of the context: each `url()` re-mints a dead tunnel. The tunnel
        is torn down on exit."""
        endpoint = PrimeEndpoint(port)
        try:
            await endpoint.url()  # mint eagerly so a setup failure raises here, typed
            yield endpoint
        finally:
            await endpoint.close()
