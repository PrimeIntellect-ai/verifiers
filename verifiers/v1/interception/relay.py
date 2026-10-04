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
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

RELAY_SOURCE = Path(__file__).with_name("relay_server.py").read_bytes()
STATS = ("relay_retried_requests", "relay_rescued_requests", "relay_retry_seconds")
# Fails when a configured proxy would catch the harness's traffic to the relay, or needs
# TLS to reach (the relay speaks plain HTTP to proxies).
LOOPBACK_DIRECT = (
    "import urllib.request as u; p = u.getproxies_environment(); "
    "assert not p or u.proxy_bypass_environment('127.0.0.1'); "
    "assert not any(v.lower().startswith('https://') for v in p.values())"
)
# Prints a python3 >= 3.8 after creating the relay's private directory $1; exits 1 if
# there is no python, 2 if a proxy would catch loopback, 3 if $1 can't be created.
FIND_PYTHON = (
    "for p in python3 python; do "
    "$p -c 'import sys, ssl, http.server; assert sys.version_info >= (3, 8)' "
    "2>/dev/null || continue; "
    f"$p -c {shlex.quote(LOOPBACK_DIRECT)} 2>/dev/null || exit 2; "
    'mkdir -m 700 "$1" || exit 3; '
    "command -v $p; exit; done; exit 1"
)
PROXIED = "a configured proxy would catch loopback traffic, or needs TLS"
# One request through the relay: fails unless the host itself answers (any status), or
# nothing answers in time (the network is down, not the relay broken).
PROBE = """import socket, sys, urllib.error, urllib.request
try:
    r = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=30)
except urllib.error.HTTPError as e:
    r = e
except (TimeoutError, socket.timeout):
    sys.exit(0)
sys.exit(0 if r.headers.get(sys.argv[2]) else 4)
"""
# Kills every process running the relay under directory argv[1], then removes it.
STOP = """import os, shutil, signal, sys
mark = (sys.argv[1] + "/relay.py").encode()
for _ in range(2):  # the second pass catches a relay the loop restarted meanwhile
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if int(pid) != os.getpid() and mark in f.read():
                    os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
shutil.rmtree(sys.argv[1], ignore_errors=True)
"""


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
    # All of the relay's files live in a directory only the runtime's user can enter, so
    # an agent running as another user can't plant links there or read the proxy settings.
    home = f"/tmp/vf-relay-{uuid.uuid4().hex}"
    files = f"{home}/relay"
    relay, found, python, reason = None, None, None, ""
    try:
        try:
            # Generous: sandbox commands slow down when thousands of rollouts start at once.
            async with asyncio.timeout(300):
                found = await runtime.run(["sh", "-c", FIND_PYTHON, "sh", home], {})
                if found.exit_code == 2:
                    reason = PROXIED
                elif found.exit_code == 3:
                    reason = f"could not create {home}"
                elif found.exit_code != 0:
                    reason = "no python3 >= 3.8 with ssl in the runtime"
                else:
                    python = found.stdout.split()[-1]
                    await runtime.write(f"{files}.py", RELAY_SOURCE)
                    args = shlex.join(
                        [f"{files}.py", upstream, files, str(window)]
                        + [RETRY_COUNT_HEADER, INTERCEPTION_HEADER]
                    )
                    # Restarted if it exits; a restart binds the same port. The
                    # interpreter comes from the environment, so `pkill -f python3`
                    # spares the loop.
                    script = f'while :; do "$VF_RELAY_BIN" {args}; sleep 1; done'
                    await runtime.run_background(
                        ["sh", "-c", script], {"VF_RELAY_BIN": python}, f"{files}.log"
                    )
                    url, poll = None, 0.25
                    while url is None:
                        with contextlib.suppress(Exception):
                            port = await runtime.read(f"{files}.port", max_bytes=64)
                            url = f"http://127.0.0.1:{int(port)}"
                        if url is None:
                            await asyncio.sleep(poll)
                            poll = min(poll * 2, 2)
                    # A relay that can't reach the host (say, a Python without the CA
                    # certificates the harness ships) would be worse than going direct.
                    probe = await runtime.run(
                        [python, "-c", PROBE, f"{url}/v1/models", INTERCEPTION_HEADER],
                        {},
                    )
                    if probe.exit_code == 0:
                        relay = Relay(runtime, python, files, url, record)
                    else:
                        error = probe.stderr.strip().splitlines()[-1:] or ["no stamp"]
                        reason = f"it could not reach the host ({error[0][:200]})"
        except TimeoutError:
            reason = "it did not start within 300 s"
        except Exception as e:  # noqa: BLE001 - connect directly instead
            reason = repr(e)
        if relay is None:
            record({"relay_active": 0.0})
            logger.warning(
                "interception relay did not start (%s); connecting directly", reason
            )
        yield relay
    finally:
        # Even when cancelled, and before the task's finalize and scoring: leave
        # nothing behind for a grader to see.
        if python is not None or found is None:
            await run_shielded(_stop(runtime, python, home, relay, record))


async def _stop(
    runtime: Runtime,
    python: str | None,
    home: str,
    relay: Relay | None,
    record: Callable[[dict[str, float]], None],
) -> None:
    if relay is not None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(120):
                # An agent running as this user can write it: keep only the counters.
                stats = json.loads(
                    await runtime.read(f"{relay.files}.json", max_bytes=4096)
                )
                record(
                    {
                        key: float(stats[key])
                        for key in STATS
                        if math.isfinite(float(stats.get(key, "nan")))
                    }
                )
    with contextlib.suppress(Exception):
        async with asyncio.timeout(120):
            stop = [python, "-c", STOP, home] if python else ["rm", "-rf", "--", home]
            await runtime.run(stop, {})
