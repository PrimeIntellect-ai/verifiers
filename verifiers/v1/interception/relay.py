"""Opt-in loopback relay in a remote runtime that rides out transient network failures.

A remote runtime reaches the host's interception server over the network (a tunnel, a
proxy, a direct bind). When that path blips, clients see refused or dropped connections,
or an error from whatever sits in between (a tunnel's "not found", a proxy's 502), and
most fail the rollout. The harness reaches the host through the relay
(`relay_server.py`, stdlib only) instead: it forwards requests unchanged and, for up to
`relay_seconds` after a failure, retries them. Tool servers keep the direct URL: they may
run as a more privileged user than the agent, and their state secret shouldn't pass
through a port an agent could take over. The host stamps every response it sends,
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
# Python runs isolated (-I -S): never importing from the environment, the working
# directory or site-packages, which an agent may control.
ISOLATED = ("-I", "-S")
# Prints the real path of an interpreter fit to run the relay, which runs it again on
# every restart: Python >= 3.8 with all it imports, whose files (the executable, the
# modules and the libraries it loads) and their directories only root or this user can
# change, and a sticky /tmp. Exits 10 if a configured proxy would catch the harness's
# loopback traffic, or is one the relay can't use (it speaks plain HTTP to http://
# proxies only).
CHECK = """import os, sys
assert sys.version_info >= (3, 8)
import base64, hashlib, http.client, http.server, json, math, random, re, select
import socket, ssl, threading, time, types, urllib.request as u
p = u.getproxies_environment()
if p and not u.proxy_bypass_environment("127.0.0.1"):
    sys.exit(10)
if "all" in p or any(not p[k].lower().startswith("http://") for k in ("http", "https") if k in p):
    sys.exit(10)
assert sys.executable and os.stat("/tmp").st_mode & 0o1000
files = {sys.executable} | {getattr(m, "__file__", None) for m in list(sys.modules.values())}
with open("/proc/self/maps") as maps:
    files |= {line.split()[5] for line in maps if line.count(" ") >= 5 and " /" in line}
checked = set()
for path in filter(None, files):
    path = os.path.realpath(path)
    if not os.path.isfile(path):  # a deleted mapping, a device: nothing to change
        continue
    while path not in checked:
        checked.add(path)
        s = os.stat(path)
        assert s.st_uid in (0, os.getuid()) and not s.st_mode & 0o022
        path = os.path.dirname(path)
print(os.path.realpath(sys.executable))
"""
# Prints such an interpreter after creating the relay's private directory $1; exits 1 if
# there is none, 2 if a proxy would catch loopback, 3 if $1 can't be created.
FIND_PYTHON = (
    "for p in python3 python /usr/local/bin/python3 /usr/bin/python3; do "
    f"exe=$($p -I -S -c {shlex.quote(CHECK)} 2>/dev/null); c=$?; "
    "[ $c = 10 ] && exit 2; [ $c = 0 ] || continue; "
    '(umask 077 && mkdir -- "$1") || exit 3; '
    'echo "$exe"; exit; done; exit 1'
)
PROXIED = "a configured proxy would catch loopback traffic, or isn't plain HTTP"
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

    async def adopt_policy(self, limit: float | None = None) -> bool:
        """Hand the relay the proxy settings programs get once the runtime's network
        policy applies (it started before the policy did). False if those settings
        would send the harness's loopback traffic to a proxy: connect it directly.
        `limit` caps the seconds this may take."""
        env, tmp = (
            shlex.quote(f"{self.files}.env"),
            shlex.quote(f"{self.files}.env.tmp"),
        )
        save = f"umask 077; env | grep -i '_proxy=' > {tmp}; mv {tmp} {env}"
        check = shlex.join([self.python, *ISOLATED, "-c", CHECK])
        try:
            async with asyncio.timeout(120 if limit is None else min(120, limit)):
                done = await self.runtime.run(
                    ["sh", "-c", f"{check} > /dev/null && {save}"], {}
                )
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
    limit: float | None = None,
) -> AsyncIterator[Relay | None]:
    """Yield the started relay, or None if it didn't start within `limit` seconds
    (at most 300)."""
    # All of the relay's files live in a directory only the runtime's user can enter, so
    # an agent running as another user can't plant links there or read the proxy settings.
    home = f"/tmp/vf-relay-{uuid.uuid4().hex}"
    files = f"{home}/relay"
    relay, found, python, reason, stopped = None, None, None, "", False
    # Generous: sandbox commands slow down when thousands of rollouts start at once.
    budget = 300 if limit is None else min(300, limit)
    try:
        try:
            async with asyncio.timeout(budget):
                found = await runtime.run(["sh", "-c", FIND_PYTHON, "sh", home], {})
                if found.exit_code == 2:
                    reason = PROXIED
                elif found.exit_code == 3:
                    reason = f"could not create {home}"
                elif found.exit_code != 0 or not found.stdout.strip():
                    reason = "no python3 >= 3.8 with ssl that only root or this user can change"
                else:
                    python = found.stdout.strip().splitlines()[-1]
                    await runtime.write(f"{files}.py", RELAY_SOURCE)
                    args = shlex.join(
                        [*ISOLATED, f"{files}.py", upstream, files, str(window)]
                        + [RETRY_COUNT_HEADER, INTERCEPTION_HEADER]
                    )
                    # Restarted at once if it exits (after a pause if it keeps
                    # crashing), on the same port. The interpreter comes from the
                    # environment, so `pkill -f python3` spares the loop, and so does
                    # a fixed PATH, so an agent's directories can't shadow `date`.
                    script = (
                        f'while :; do t=$(date +%s); "$VF_RELAY_BIN" {args}; '
                        "[ $(($(date +%s) - t)) -lt 5 ] && sleep 1; done"
                    )
                    await runtime.run_background(
                        ["sh", "-c", script],
                        {"VF_RELAY_BIN": python, "PATH": "/usr/bin:/bin"},
                        f"{files}.log",
                    )
                    # The relay first checks its way to the host: one that can't reach
                    # it (say, a Python without the CA certificates the harness ships)
                    # would be worse than going direct.
                    poll = 0.25
                    while relay is None and not reason:
                        with contextlib.suppress(Exception):
                            port = await runtime.read(f"{files}.port", max_bytes=512)
                            if port.startswith(b"refused: "):
                                why = port[9:].decode(errors="replace")
                                reason = f"it can't reach the host: {why!r}"
                            elif 0 < int(port) < 65536 and port.strip().isdigit():
                                url = f"http://127.0.0.1:{int(port)}"
                                relay = Relay(runtime, python, files, url, record)
                        if relay is None and not reason:
                            await asyncio.sleep(poll)
                            poll = min(poll * 2, 2)
        except TimeoutError:
            reason = f"it did not start within {budget:.0f} s"
        except Exception as e:  # noqa: BLE001 - connect directly instead
            reason = repr(e)
        if relay is None:
            record({"relay_active": 0.0})
            logger.warning(
                "interception relay did not start (%s); connecting directly", reason
            )
            # Stopped now, so a relay that keeps crashing doesn't run on meanwhile.
            if found is None or found.exit_code == 0:
                await run_shielded(_stop(runtime, python, home, None, record))
            stopped = True
        yield relay
    finally:
        # Even when cancelled, and before the task's finalize and scoring: leave
        # nothing behind for a grader to see.
        if not stopped and (found is None or found.exit_code == 0):
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
                        key: float(value)
                        for key, value in stats.items()
                        if key in STATS
                        and type(value) in (int, float)
                        and 0 <= value < math.inf
                    }
                )
    with contextlib.suppress(Exception):
        async with asyncio.timeout(120):
            stop = [python, *ISOLATED, "-c", STOP, home]
            await runtime.run(stop if python else ["rm", "-rf", "--", home], {})
