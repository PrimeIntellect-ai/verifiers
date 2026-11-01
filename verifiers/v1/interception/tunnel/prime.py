"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in any remote sandbox — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from verifiers.v1.configs.interception import PrimeTunnelConfig
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.runtimes import suspend
from verifiers.v1.runtimes.limiters import CreationLimiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

if TYPE_CHECKING:
    from prime_tunnel import Tunnel as TunnelClient

logger = logging.getLogger(__name__)


class TunnelLogHandler(logging.Handler):
    """Forward tunnel warnings and errors to this module's logger — and, when the app
    enables INFO on `prime_tunnel`, the login after a reconnect."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().removeprefix("frpc ")
        if record.levelno >= logging.WARNING or "login to server success" in message:
            logger.log(record.levelno, "tunnel %s", message)


logging.getLogger("prime_tunnel.frpc").addHandler(TunnelLogHandler())

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process of a run that opens one. One run-scoped
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512


def tunnel_limiter() -> CreationLimiter:
    return CreationLimiter("prime-tunnel", run_scope(), _TUNNELS_PER_MIN / 60)


# How often a held tunnel's state is checked with the tunnel service. The service marks a
# tunnel disconnected about a minute after its frpc stops answering; two such checks in a
# row restart frpc.
CHECK_SECONDS = 20


async def _watch(client: "TunnelClient", url: str) -> None:
    """Keep a held tunnel up. A tunnel service hiccup can leave frpc running while the
    service has it disconnected for good, and every request to the URL then hangs or
    404s; frpc can also exit. Either way, restart it. A tunnel the service has deleted
    or expired can't be restarted, so stop watching it."""
    from prime_tunnel import TunnelGoneError

    strikes = 0
    while True:
        await asyncio.sleep(CHECK_SECONDS * random.uniform(0.75, 1.25))
        if client.is_running and not client.is_gone:
            try:
                status = await client.status()
            except Exception as e:  # noqa: BLE001 - the service is unreachable: no verdict
                logger.debug("tunnel %s: state check failed: %s", url, e)
                continue
            strikes = strikes + 1 if status.registration == "disconnected" else 0
        if client.is_gone:
            logger.error("tunnel %s: registration gone; cannot repair", url)
            return
        if client.is_running and strikes < 2:
            continue
        logger.warning(
            "tunnel %s: %s; restarting frpc",
            url,
            "disconnected" if client.is_running else "frpc exited",
        )
        try:
            await client.restart()
        except TunnelGoneError:
            logger.error("tunnel %s: registration gone; cannot repair", url)
            return
        except Exception as e:  # noqa: BLE001 - tried again on the next check
            logger.warning("tunnel %s: frpc restart failed: %s", url, e)
        else:
            logger.warning("tunnel %s: back up", url)
            strikes = 0


_held: dict[str, "TunnelClient"] = {}
"""The tunnels this process holds open, by URL (`held_tunnel`)."""


def held_tunnel(url: str) -> dict | None:
    """The registration (prime_tunnel's `TunnelInfo`) of the tunnel this process holds
    at `url`, for another process to serve the same URL (`AdoptedPrimeTunnel`)."""
    client = _held.get(url.rstrip("/"))
    info = client._tunnel_info if client is not None else None
    return info.model_dump(mode="json") if info is not None else None


@contextlib.asynccontextmanager
async def _hold(client: "TunnelClient", url: str) -> AsyncIterator[str]:
    """Keep a started tunnel up while held (`_watch`), then tear it down: stop frpc and
    delete the registration."""
    _held[url] = client
    watch = asyncio.create_task(_watch(client, url))

    async def close() -> None:
        watch.cancel()  # first, so it can't start a new frpc after the stop
        await asyncio.wait({watch})
        if suspend.keeps_tunnel(client.tunnel_id):
            # A suspended box's program reaches it: stop frpc, keep the registration
            # for the process that adopts the box (`sync_stop` deletes a held one).
            client._tunnel_info = None
        await asyncio.to_thread(client.sync_stop)

    try:
        yield url
    finally:
        _held.pop(url, None)
        # Run the stop to completion even under cancellation (`run_shielded` re-raises
        # the cancellation after); tunnel-stop failures are best-effort.
        with contextlib.suppress(Exception):
            await run_shielded(close())


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    def __init__(self, config: PrimeTunnelConfig | None = None) -> None:
        ensure_prime_auth()
        super().__init__(config)

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and rate-capped (512/min, run-wide via the shared
        `tunnel_limiter`), so transient failures are retried; a terminal one raises
        `TunnelError`. While held, frpc is restarted if the tunnel goes down. The tunnel is
        torn down on exit."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        label = f"host tunnel (port {port})"
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=port)
                    async with tunnel_limiter():
                        url = str(await client.start()).rstrip("/")
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        async with _hold(client, url):
            yield url


class AdoptedPrimeTunnel(Tunnel[PrimeTunnelConfig]):
    """A tunnel another process registered (`held_tunnel`), served from here: its URL
    reaches this process's port once frpc runs with the registration. A process that
    died left it registered; the harness programs behind it reach the model at its URL
    and see the interception again once it is back (`Trace.live`)."""

    def __init__(self, info: dict) -> None:
        ensure_prime_auth()
        super().__init__(PrimeTunnelConfig())
        self.info = info

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        from prime_tunnel import Tunnel as TunnelClient
        from prime_tunnel.models import TunnelInfo

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        info = TunnelInfo.model_validate(self.info)
        label = f"adopted tunnel {info.tunnel_id} (port {port})"
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=port)
                    # prime_tunnel restarts frpc for a registration it holds, keeping
                    # the URL; it has no public way to hold one it did not register.
                    client._tunnel_info = info
                    client._config_file = client._write_frpc_config()
                    client._started = True
                    url = str(await client.restart()).rstrip("/")
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        logger.info("tunnel %s: adopted", url)
        async with _hold(client, url):
            yield url
