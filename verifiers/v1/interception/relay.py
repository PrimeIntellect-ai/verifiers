"""Opt-in loopback relay in a remote runtime that rides out transient network failures.

A remote runtime reaches the host's interception server over the network (a tunnel, a
proxy, a direct bind). When that path blips, clients see refused or dropped connections,
or an error from whatever sits in between (a tunnel's "not found", a proxy's 502), and
most fail the rollout. The harness and colocated tool servers reach the host through the
relay (`relay_server.py`, stdlib only) instead: it forwards requests unchanged and, for up
to `relay_seconds` after a failure, retries them. The host stamps every response it sends,
so any transient-looking error without the stamp is a network failure, whatever its
wording; repeats are safe because the host dedupes them. Without a usable `python3` the
rollout connects directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import shlex
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from verifiers.v1.interception.server import INTERCEPTION_HEADER, RETRY_COUNT_HEADER
from verifiers.v1.runtimes.base import Runtime

logger = logging.getLogger(__name__)

RELAY_SOURCE = Path(__file__).with_name("relay_server.py").read_bytes()
STATS = ("relay_retried_requests", "relay_rescued_requests", "relay_retry_seconds")
# Fails when a configured proxy would catch the harness's traffic to the relay.
LOOPBACK_DIRECT = (
    "import urllib.request as u; assert not u.getproxies_environment() "
    "or u.proxy_bypass_environment('127.0.0.1')"
)
FIND_PYTHON = (
    "for p in python3 python; do $p -c "
    + shlex.quote(f"import sys; assert sys.version_info >= (3, 8); {LOOPBACK_DIRECT}")
    + " 2>/dev/null && command -v $p && exit; done; exit 1"
)


class Relay:
    def __init__(self, runtime: Runtime, python: str, files: str, url: str) -> None:
        self.runtime, self.python, self.files, self.url = runtime, python, files, url

    async def adopt_policy(self) -> bool:
        """Hand the relay the proxy settings programs get once the runtime's network
        policy applies (it started before the policy did). False if those settings
        would send the harness's loopback traffic to a proxy: connect it directly."""
        env = f"{self.files}.env"
        save = f"umask 077; env | grep -i '_proxy=' > {env}.tmp; mv {env}.tmp {env}"
        check = shlex.join([self.python, "-c", LOOPBACK_DIRECT])
        with contextlib.suppress(Exception):
            async with asyncio.timeout(30):
                done = await self.runtime.run(["sh", "-c", f"{save}; {check}"], {})
                return done.exit_code == 0
        return False


@contextlib.asynccontextmanager
async def serve_relay(
    runtime: Runtime,
    upstream: str,
    window: float,
    record: Callable[[dict[str, float]], None],
) -> AsyncIterator[Relay | None]:
    """Yield the started relay, or None if it didn't start."""
    files = f"/tmp/vf-relay-{uuid.uuid4().hex[:12]}"
    relay, launched = None, False
    with contextlib.suppress(Exception):
        async with asyncio.timeout(60):
            found = await runtime.run(["sh", "-c", FIND_PYTHON], {})
            if found.exit_code == 0:
                await runtime.write(f"{files}.py", RELAY_SOURCE)
                python = found.stdout.split()[-1]
                argv = shlex.join(
                    [python, f"{files}.py", upstream, files, str(window)]
                    + [RETRY_COUNT_HEADER, INTERCEPTION_HEADER]
                )
                # Restarted if it exits; a restart binds the same port.
                script = f"echo $$ > {files}.pid; while :; do {argv}; sleep 1; done"
                launched = True
                await runtime.run_background(["sh", "-c", script], {}, f"{files}.log")
                for _ in range(60):
                    with contextlib.suppress(Exception):
                        port = await runtime.read(f"{files}.port", max_bytes=64)
                        port = int(port.split()[0])
                        url = f"http://127.0.0.1:{port}"
                        relay = Relay(runtime, python, files, url)
                        break
                    await asyncio.sleep(0.25)
    if relay is None:
        logger.warning("interception relay did not start; connecting directly")
    try:
        yield relay
    finally:
        if relay is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(30):
                    # The agent can write this file: keep only the relay's own counters.
                    stats = json.loads(
                        await runtime.read(f"{files}.json", max_bytes=4096)
                    )
                    record(
                        {
                            key: float(stats[key])
                            for key in STATS
                            if math.isfinite(float(stats.get(key, "nan")))
                        }
                    )
        if launched:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(30):
                    # The supervising loop first, so it can't restart the relay. Both
                    # files are agent-writable: only plain pids reach `kill`.
                    pids = []
                    for path, field in ((f"{files}.pid", 0), (f"{files}.port", 1)):
                        with contextlib.suppress(Exception):
                            pid = (await runtime.read(path, max_bytes=64)).split()[
                                field
                            ]
                            if pid.isdigit() and int(pid) > 1:
                                pids.append(pid.decode())
                    if pids:
                        await runtime.run(["kill", *pids], {})
