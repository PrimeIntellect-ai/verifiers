"""Opt-in loopback relay that carries a remote runtime's model calls through network blips.

When the path from a remote runtime to the host blips, harnesses see refused or dropped
connections, or an error from whatever sits in between (a tunnel's "not found", a proxy's
502), and most fail the rollout: their clients retry briefly at best, and never retry a
404. With `relay_seconds` set, the harness reaches the model through a relay on the
runtime's loopback instead (`relay_server.py`), which retries those for up to that long.
It runs as the agent, so it holds nothing the agent can't already reach. The rollout
connects directly if the runtime has no usable python3 or sets a proxy, or if the relay
can't reach the host.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import shlex
import uuid
import zipfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import h11

from verifiers.v1.runtimes.base import Runtime
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)


def _bundle() -> bytes:
    """The relay and the h11 it uses, as one zip that `python3 relay.pyz` runs. Stored,
    not compressed, so importing from it needs no zlib."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as bundle:
        bundle.write(Path(__file__).with_name("relay_server.py"), "__main__.py")
        for path in sorted(Path(h11.__file__).parent.glob("*.py")):
            bundle.write(path, f"h11/{path.name}")
    return buffer.getvalue()


RELAY_BUNDLE = _bundle()
# Python runs isolated (-I -S), so a task's PYTHONPATH or working directory can't shadow
# the standard library.
ISOLATED = ("-I", "-S")
# Prints the real path of an interpreter fit to run the relay; exits 10 if a proxy is
# configured (the relay doesn't use one, and the harness's loopback calls might).
CHECK = """import os, ssl, sys, urllib.request
assert sys.version_info >= (3, 8) and sys.executable
if set(urllib.request.getproxies_environment()) - {"no"}:  # NO_PROXY alone is fine
    sys.exit(10)
print(os.path.realpath(sys.executable))
"""
# Prints such an interpreter after creating the relay's private directory $1.
FIND_PYTHON = (
    "for p in python3 python /usr/local/bin/python3 /usr/bin/python3; do "
    f"exe=$($p -I -S -c {shlex.quote(CHECK)} 2>/dev/null); c=$?; "
    "[ $c = 10 ] && exit 2; [ $c = 0 ] || continue; "
    '(umask 077 && mkdir -- "$1") || exit 3; '
    'echo "$exe"; exit; done; exit 1'
)
UNUSABLE = {
    1: "no python3 >= 3.8 with ssl in the runtime",
    2: "the runtime sets a proxy",
    3: "could not create its directory",
}
# Kills every process running the relay under directory argv[1], then removes it.
STOP = """import os, shutil, signal, sys
mark = (sys.argv[1] + "/relay.pyz").encode()
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


@contextlib.asynccontextmanager
async def serve_relay(
    runtime: Runtime,
    upstream: str,
    window: float,
    record: Callable[[dict[str, float]], None],
    limit: float | None = None,
) -> AsyncIterator[str | None]:
    """Yield the started relay's URL, or None if it isn't usable or didn't start within
    `limit` seconds (at most 300)."""
    home = f"/tmp/vf-relay-{uuid.uuid4().hex}"
    files = f"{home}/relay"
    # Generous: sandbox commands slow down when thousands of rollouts start at once.
    budget = 300 if limit is None else min(300, limit)
    deadline = asyncio.get_running_loop().time() + budget
    url = python = None
    reason = f"only {budget:.0f} s of setup time to spare"
    dirty = False  # its directory may exist
    try:
        if budget >= 10:
            try:
                async with asyncio.timeout_at(deadline):
                    dirty = True
                    found = await runtime.run(["sh", "-c", FIND_PYTHON, "sh", home], {})
                    if found.exit_code or not found.stdout.strip():
                        dirty = found.exit_code == 0
                        reason = UNUSABLE.get(found.exit_code, UNUSABLE[1])
                    else:
                        python = found.stdout.strip().splitlines()[-1]
                        url, reason = await _start(
                            runtime, python, files, upstream, window
                        )
            except TimeoutError:
                reason = f"it did not start within {budget:.0f} s"
            except Exception as e:  # noqa: BLE001 - connect directly instead
                reason = repr(e)
        record({"relay_active": float(url is not None)})
        if url is None:
            logger.warning(
                "interception relay unused (%s); connecting directly", reason
            )
            # Stopped now, within the budget, so a relay that keeps crashing doesn't run
            # on meanwhile; teardown tries again if this can't.
            if dirty:
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout_at(deadline):
                        dirty = not await _stop(runtime, python, home)
        yield url
    finally:
        # Even when cancelled, and before the task's finalize and scoring: leave nothing
        # behind for a grader to see.
        if dirty:
            await run_shielded(_finish(runtime, python, home, url, record))


async def _start(
    runtime: Runtime, python: str, files: str, upstream: str, window: float
) -> tuple[str | None, str]:
    await runtime.write(f"{files}.pyz", RELAY_BUNDLE)
    bundle, port = shlex.quote(f"{files}.pyz"), shlex.quote(f"{files}.port")
    args = shlex.join([*ISOLATED, f"{files}.pyz", upstream, files, str(window)])
    # Restarted at once if it exits (after a pause if it keeps crashing), on the same
    # port, until its files are gone. One that dies before it can say why is reported as
    # refusing. The interpreter comes from the environment, so `pkill -f python3`
    # spares the loop.
    script = (
        f"while [ -e {bundle} ]; do t=$(date +%s); "
        f'"$VF_RELAY_BIN" {args}; c=$?; '
        f'[ -e {port} ] || echo "refused: relay exited with status $c" > {port}; '
        "[ $(($(date +%s) - t)) -lt 5 ] && sleep 1; done"
    )
    await runtime.run_background(
        ["sh", "-c", script], {"VF_RELAY_BIN": python}, f"{files}.log"
    )
    # The relay first checks its way to the host: one that can't reach it (say, a
    # Python without CA certificates) would be worse than going direct.
    poll = 0.25
    while True:
        with contextlib.suppress(Exception):
            written = await runtime.read(f"{files}.port", max_bytes=512)
            if written.startswith(b"refused: "):
                why = written[9:].decode(errors="replace").strip()
                return None, f"it refused: {why!r}"
            if written.strip().isdigit() and 0 < int(written) < 65536:
                return f"http://127.0.0.1:{int(written)}", ""
        await asyncio.sleep(poll)
        poll = min(poll * 2, 2)


async def _finish(
    runtime: Runtime,
    python: str | None,
    home: str,
    url: str | None,
    record: Callable[[dict[str, float]], None],
) -> None:
    if url is not None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(120):
                rescued = await runtime.read(f"{home}/relay.rescued", max_bytes=1 << 20)
                record({"relay_rescued_requests": float(len(rescued))})
    await _stop(runtime, python, home)


async def _stop(runtime: Runtime, python: str | None, home: str) -> bool:
    # Removing the directory also ends the restart loop, should the interpreter fail.
    script = f'"$0" {" ".join(ISOLATED)} -c "$1" "$2" || rm -rf -- "$2"'
    stop = (
        ["sh", "-c", script, python, STOP, home]
        if python
        else ["rm", "-rf", "--", home]
    )
    with contextlib.suppress(Exception):
        async with asyncio.timeout(120):
            await runtime.run(stop, {})
            return True
    return False
