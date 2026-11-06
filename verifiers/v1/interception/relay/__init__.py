"""Host side of the in-box relay (`server.py`).

A harness in a remote box reaches the interception server through a relay that runs in
its box, at `127.0.0.1`. The relay forwards every request to its current upstream and
swaps the harness's bearer tokens for the upstream's, so the host can move the harness
to a new URL or new secrets without the harness noticing: `Relay.reroute`.

Each rollout runs its own relay, under `RELAY_ROOT/<name>` and on a port of its own, so
rollouts that share a box never share one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from verifiers.v1.errors import RelayError, SandboxError, TunnelError

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime

RELAY_SOURCE = (Path(__file__).resolve().parent / "server.py").read_text()
RELAY_ROOT = "/tmp/vf-relay"
RELAY_START_SECONDS = 60
"""How long a launched relay gets to accept connections."""


def relay_config(upstream: str, tokens: dict[str, str]) -> bytes:
    """A relay's config file: the upstream base URL, and the harness's tokens mapped to
    the upstream's."""
    return json.dumps({"upstream": upstream, "tokens": tokens}).encode()


class Relay:
    """A rollout's relay in its box. It launches when created, before its upstream is
    known, so its start overlaps the harness setup; `connect` points it at the
    interception server, `reroute` moves it, `stop` ends it."""

    def __init__(self, runtime: Runtime, name: str) -> None:
        self.runtime = runtime
        self.dir = f"{RELAY_ROOT}/{name}"
        self.url = ""
        """The base URL the harness reaches the interception server at, once launched."""
        self.upstream = ""
        self.tokens: dict[str, str] = {}
        self._connected = asyncio.Event()
        self._launch = asyncio.create_task(self._start())

    @property
    def config_path(self) -> str:
        return f"{self.dir}/config.json"

    @property
    def pid_path(self) -> str:
        return f"{self.dir}/pid"

    @property
    def failures_path(self) -> str:
        """Where the relay appends the reason for each request it gives up on."""
        return f"{self.dir}/failures"

    async def _start(self) -> None:
        # Each launch reads its port from a file of its own: a relay an earlier attempt
        # left in the box can never answer for this one.
        ready = f"{self.dir}/port-{uuid.uuid4().hex}"
        # The gateway creates the directory. Nothing listens on port 9 (discard).
        await self.runtime.write(
            self.config_path, relay_config("http://127.0.0.1:9", {})
        )
        program = await self.runtime.prepare_uv_script(RELAY_SOURCE, activate=False)
        args = ["--config", self.config_path, "--port", "0", "--ready", ready]
        args += ["--failures", self.failures_path]
        await self.runtime.run_background(
            [*program, *args, "--pid", self.pid_path], {}, log=f"{self.dir}/relay.log"
        )
        deadline = time.monotonic() + RELAY_START_SECONDS
        while True:
            port = (await self.runtime.run(["cat", ready], {})).stdout.strip()
            if port.isdigit():
                self.url = f"http://127.0.0.1:{port}"
                return
            if time.monotonic() > deadline:
                raise SandboxError(
                    f"the relay did not start within {RELAY_START_SECONDS}s"
                )
            await asyncio.sleep(0.1)

    async def connect(self, upstream: str, tokens: dict[str, str]) -> str:
        """Point the relay at the interception server once it runs, and return the base
        URL the harness reaches the server at."""
        try:
            await self._launch
            await self._configure(upstream, tokens)
        finally:
            self._connected.set()
        return self.url

    async def wait_connected(self) -> None:
        """Wait until `connect` finished; raise if the relay did not start."""
        await self._connected.wait()
        if not self.upstream:
            raise TunnelError("the relay did not start")

    async def reroute(
        self, upstream: str, tokens: dict[str, str] | None = None
    ) -> None:
        """Move the relay to `upstream`, with `tokens` if the secrets changed. Its
        requests in flight retry against the new upstream."""
        await self.wait_connected()
        await self._configure(upstream, self.tokens if tokens is None else tokens)

    async def _configure(self, upstream: str, tokens: dict[str, str]) -> None:
        # Written whole, then moved: the relay must never read a partial config.
        staged = f"{self.config_path}.next"
        await self.runtime.write(staged, relay_config(upstream, tokens))
        config, pid = shlex.quote(self.config_path), shlex.quote(self.pid_path)
        reload = f'mv {shlex.quote(staged)} {config} && kill -HUP "$(cat {pid})"'
        result = await self.runtime.run(["sh", "-c", reload], {})
        if result.exit_code:
            raise SandboxError(
                f"the relay did not reload: {result.stderr.strip()[-500:]}"
            )
        self.upstream, self.tokens = upstream, tokens

    async def failure(self) -> RelayError | None:
        """The relay's failure, if it gave up on a request or exited; for a harness that
        died, so the trace names the relay rather than the harness."""
        if not self.upstream:
            return None
        failures, pid = shlex.quote(self.failures_path), shlex.quote(self.pid_path)
        check = (
            f"tail -n 1 {failures} 2>/dev/null; "
            f'kill -0 "$(cat {pid})" 2>/dev/null || echo "the relay exited"'
        )
        with contextlib.suppress(Exception):
            result = await self.runtime.run(["sh", "-c", check], {})
            reasons = result.stdout.strip().splitlines()
            if reasons:
                return RelayError("; ".join(reasons))
        return None

    async def stop(self) -> None:
        """End the relay and remove its files; best effort, the box may be gone."""
        self._connected.set()
        if not self._launch.done():
            self._launch.cancel()
        await asyncio.wait({self._launch})
        if self._launch.cancelled() or self._launch.exception() is not None:
            return
        pid, directory = shlex.quote(self.pid_path), shlex.quote(self.dir)
        # Best effort: the box may be gone already.
        with contextlib.suppress(Exception):
            await self.runtime.run(
                ["sh", "-c", f'kill "$(cat {pid})" 2>/dev/null; rm -rf {directory}'],
                {},
            )
