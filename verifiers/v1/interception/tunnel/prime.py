"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in any remote sandbox — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import atexit
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from verifiers.v1.configs.interception import PrimeTunnelConfig
from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.runtimes.backends.prime import run_labels
from verifiers.v1.runtimes.limiters import CreationLimiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

if TYPE_CHECKING:
    from prime_tunnel import Tunnel as TunnelClient

    from verifiers.v1.errors import OutageKind

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


_LIVE: set["TunnelClient"] = set()
"""Tunnels held open: the atexit backstop stops them if a signal cut their `finally`
short (a Ctrl-C or SIGTERM that unwound the loop mid-teardown). A stopped tunnel's frpc
would otherwise outlive the process, and its registration stays connected."""
_atexit_armed = False


QUOTA_WAIT_SECONDS = 60.0
"""How long every tunnel creation in the process waits after the tunnel service refused
one for the account's creation cap (hourly, or per token): retrying sooner is refused
again and spends the cap other runs share."""
_quota_until = 0.0


def creation_outage(error: BaseException) -> "OutageKind | None":
    """The outage a failed tunnel creation was: `quota` for the service's creation caps
    (prime_tunnel raises them as a plain `TunnelError` with the service's detail, its
    status dropped, so the detail is the evidence), `account` for a refused key."""
    from prime_tunnel.exceptions import TunnelAuthError, TunnelLimitReachedError

    from verifiers.v1.errors import sandbox_error_kind

    if isinstance(error, TunnelLimitReachedError):
        return "quota"
    if isinstance(error, TunnelAuthError):
        return "account"
    kind = sandbox_error_kind(error)
    return kind if kind in ("account", "quota", "io") else None


def _stop_at_exit() -> None:
    for client in list(_LIVE):
        with contextlib.suppress(Exception):
            client.sync_stop()


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    def __init__(self, config: PrimeTunnelConfig | None = None) -> None:
        ensure_prime_auth()
        super().__init__(config)

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and rate-capped (512/min, run-wide via the shared
        `tunnel_limiter`), so transient failures are retried; a terminal one raises
        `TunnelError`. While held, frpc is restarted if the tunnel goes down. The tunnel
        carries the run labels (`run_labels`), so a crashed run's tunnels can be swept
        (`runtimes.sweep`), and is torn down on exit."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        global _quota_until
        label = f"host tunnel (port {port})"
        loop = asyncio.get_running_loop()
        try:
            async for attempt in retrying(
                retries=3,
                label=label,
                when=lambda e: creation_outage(e) not in ("quota", "account"),
            ):
                with attempt:
                    # A capped account refuses every creation for a while: wait out
                    # the hold the last refusal set instead of asking again now.
                    if (wait := _quota_until - loop.time()) > 0:
                        await asyncio.sleep(wait)
                    client = TunnelClient(local_port=port, labels=run_labels())
                    async with tunnel_limiter():
                        url = str(await client.start()).rstrip("/")
        except Exception as e:
            kind = creation_outage(e)
            if kind == "quota":
                _quota_until = max(_quota_until, loop.time() + QUOTA_WAIT_SECONDS)
            raise TunnelError(f"{label} failed: {e}", kind=kind) from e
        global _atexit_armed
        if not _atexit_armed:
            _atexit_armed = True
            atexit.register(_stop_at_exit)
        _LIVE.add(client)
        watch = asyncio.create_task(_watch(client, url))

        async def close() -> None:
            watch.cancel()  # first, so it can't start a new frpc after the stop
            await asyncio.wait({watch})
            await asyncio.to_thread(client.sync_stop)
            _LIVE.discard(client)

        try:
            yield url
        finally:
            # Run the stop to completion even under cancellation (`run_shielded` re-raises
            # the cancellation after); tunnel-stop failures are best-effort.
            with contextlib.suppress(Exception):
                await run_shielded(close())
