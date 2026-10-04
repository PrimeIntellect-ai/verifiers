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
# Prints a python3 >= 3.8; exits 1 if there is none, 2 if a proxy would catch loopback.
FIND_PYTHON = (
    "for p in python3 python; do "
    "$p -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null || continue; "
    f"$p -c {shlex.quote(LOOPBACK_DIRECT)} 2>/dev/null || exit 2; "
    "command -v $p; exit; done; exit 1"
)
PROXIED = "a configured proxy would catch the harness's loopback traffic"


class Relay:
    def __init__(
        self,
        runtime: Runtime,
        python: str,
        files: str,
        url: str,
        record: Callable[[dict[str, float]], None],
    ) -> None:
        self.runtime, self.python, self.files, self.url = runtime, python, files, url
        self.record = record

    async def adopt_policy(self) -> bool:
        """Hand the relay the proxy settings programs get once the runtime's network
        policy applies (it started before the policy did). False if those settings
        would send the harness's loopback traffic to a proxy: connect it directly."""
        env = f"{self.files}.env"
        save = f"umask 077; env | grep -i '_proxy=' > {env}.tmp; mv {env}.tmp {env}"
        check = shlex.join([self.python, "-c", LOOPBACK_DIRECT])
        try:
            async with asyncio.timeout(120):
                done = await self.runtime.run(["sh", "-c", f"{save}; {check}"], {})
            active, reason = done.exit_code == 0, PROXIED
        except Exception as e:  # noqa: BLE001 - connect directly instead
            active, reason = False, f"applying the network policy failed: {e!r}"
        self.record({"relay_active": float(active)})
        if not active:
            logger.warning(
                "interception relay unused (%s); connecting directly", reason
            )
        return active


@contextlib.asynccontextmanager
async def serve_relay(
    runtime: Runtime,
    upstream: str,
    window: float,
    record: Callable[[dict[str, float]], None],
) -> AsyncIterator[Relay | None]:
    """Yield the started relay, or None if it didn't start."""
    files = f"/tmp/vf-relay-{uuid.uuid4().hex[:12]}"
    relay, launched, reason = None, False, ""
    try:
        # Generous: sandbox commands slow down when thousands of rollouts start at once.
        async with asyncio.timeout(300):
            found = await runtime.run(["sh", "-c", FIND_PYTHON], {})
            if found.exit_code == 2:
                reason = PROXIED
            elif found.exit_code != 0:
                reason = "no python3 >= 3.8 in the runtime"
            else:
                await runtime.write(f"{files}.py", RELAY_SOURCE)
                python = found.stdout.split()[-1]
                args = shlex.join(
                    [f"{files}.py", upstream, files, str(window)]
                    + [RETRY_COUNT_HEADER, INTERCEPTION_HEADER]
                )
                # Restarted if it exits; a restart binds the same port. The interpreter
                # comes from the environment, so `pkill -f python3` spares the loop.
                script = f'echo $$ > {files}.pid; while :; do "$VF_RELAY_BIN" {args}; sleep 1; done'
                launched = True
                await runtime.run_background(
                    ["sh", "-c", script], {"VF_RELAY_BIN": python}, f"{files}.log"
                )
                poll = 0.25
                while relay is None:
                    with contextlib.suppress(Exception):
                        port = await runtime.read(f"{files}.port", max_bytes=64)
                        url = f"http://127.0.0.1:{int(port.split()[0])}"
                        relay = Relay(runtime, python, files, url, record)
                    if relay is None:
                        await asyncio.sleep(poll)
                        poll = min(poll * 2, 2)
    except TimeoutError:
        reason = "it did not start within 300 s"
    except Exception as e:  # noqa: BLE001 - connect directly instead
        reason = repr(e)
    if relay is None:
        record({"relay_active": 0.0})
        logger.warning(
            "interception relay did not start (%s); connecting directly", reason
        )
    try:
        yield relay
    finally:
        if relay is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(120):
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
                async with asyncio.timeout(120):
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
            # Before the task's finalize and scoring: leave nothing behind for a grader to see.
            leftovers = [
                f"{files}.{ext}" for ext in ("py", "port", "pid", "json", "env", "log")
            ]
            with contextlib.suppress(Exception):
                async with asyncio.timeout(120):
                    await runtime.run(
                        ["rm", "-f", *leftovers, *(f"{f}.tmp" for f in leftovers)], {}
                    )
