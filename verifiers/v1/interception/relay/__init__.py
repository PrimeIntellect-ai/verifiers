"""Host side of the in-box relay (`server.py`).

A harness in a remote box reaches the interception server through a loopback relay in
its box, with the box tokens of the relay config as its secrets. The relay forwards to
the current upstream and swaps each box token for its upstream token. When the upstream
URL changes (the server replaced a lost tunnel), the host points it at the new one.

Each rollout runs its own relay, under `RELAY_ROOT/<name>` and on a port of its own, so
rollouts that share a box never share a relay: `start` launches it, `reroute` rewrites
its config and makes it reload (SIGHUP), and `stop` ends it. On a reload the relay aborts
every in-flight upstream request and retries each one that has sent its client only the
status, headers and SSE comments.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from verifiers.v1.errors import SandboxError

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime

RELAY_SOURCE = (Path(__file__).resolve().parent / "server.py").read_text()
RELAY_ROOT = "/tmp/vf-relay"
"""Where each rollout's relay keeps its files (`Relay.dir`)."""
RELAY_START_SECONDS = 60
"""How long a started relay gets to accept connections."""


class RelayConfig(BaseModel):
    """The relay's config file."""

    model_config = ConfigDict(extra="forbid")

    upstream: str
    """Base URL that request paths are appended to; it may carry a path prefix."""
    tokens: dict[str, str] = {}
    """Box token to upstream token, swapped in `Authorization: Bearer` and `x-api-key`."""
    retry_seconds: float = 300.0
    """How long the relay retries a request while the upstream cannot be reached."""

    @field_validator("upstream")
    @classmethod
    def _http_url(cls, upstream: str) -> str:
        parts = urlsplit(upstream)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"upstream must be an http(s) URL: {upstream!r}")
        if parts.query or parts.fragment:
            raise ValueError(f"upstream must not have a query: {upstream!r}")
        return upstream


def relay_config(upstream: str, tokens: dict[str, str], **options: Any) -> bytes:
    """The contents of a relay's config file; `options` sets the other `RelayConfig`
    fields."""
    return (
        RelayConfig(upstream=upstream, tokens=tokens, **options)
        .model_dump_json()
        .encode()
    )


@dataclass(frozen=True)
class Relay:
    """A rollout's relay in its box."""

    dir: str
    url: str
    """The base URL the harness reaches the interception server at."""

    @property
    def config_path(self) -> str:
        """`RelayConfig` as JSON. It holds the upstream secret."""
        return f"{self.dir}/config.json"

    @property
    def pid_path(self) -> str:
        return f"{self.dir}/pid"


async def start(
    runtime: Runtime, name: str, upstream: str, tokens: dict[str, str]
) -> Relay:
    """Start a relay in `runtime`'s box under `RELAY_ROOT/name`, forwarding to
    `upstream` with `tokens`, on a free port."""
    directory = f"{RELAY_ROOT}/{name}"
    ready = f"{directory}/port"
    await runtime.run(["sh", "-c", f"rm -rf {shlex.quote(directory)}"], {})
    relay = Relay(dir=directory, url="")
    await _write_config(runtime, relay, upstream, tokens)
    program = await runtime.prepare_uv_script(RELAY_SOURCE, activate=False)
    args = ["--config", relay.config_path, "--port", "0", "--ready", ready]
    await runtime.run_background(
        [*program, *args, "--pid", relay.pid_path], {}, log=f"{directory}/relay.log"
    )
    deadline = time.monotonic() + RELAY_START_SECONDS
    while True:
        port = (await runtime.run(["cat", ready], {})).stdout.strip()
        if port.isdigit():
            return Relay(dir=directory, url=f"http://127.0.0.1:{port}")
        if time.monotonic() > deadline:
            raise SandboxError(f"the relay did not start within {RELAY_START_SECONDS}s")
        await asyncio.sleep(0.2)


async def reroute(
    runtime: Runtime, relay: Relay, upstream: str, tokens: dict[str, str]
) -> None:
    """Point a running relay at `upstream`, swapping its box tokens for `tokens`'
    values. Its requests in flight retry against the new upstream."""
    await _write_config(runtime, relay, upstream, tokens)
    pid = shlex.quote(relay.pid_path)
    result = await runtime.run(["sh", "-c", f'kill -HUP "$(cat {pid})"'], {})
    if result.exit_code:
        raise SandboxError(f"the relay did not reload: {result.stderr.strip()[-500:]}")


async def stop(runtime: Runtime, relay: Relay) -> None:
    """End a relay and remove its files."""
    pid = shlex.quote(relay.pid_path)
    await runtime.run(
        [
            "sh",
            "-c",
            f'kill "$(cat {pid})" 2>/dev/null; rm -rf {shlex.quote(relay.dir)}',
        ],
        {},
    )


async def _write_config(
    runtime: Runtime, relay: Relay, upstream: str, tokens: dict[str, str]
) -> None:
    await runtime.run(["mkdir", "-p", relay.dir], {})
    # Written whole, then moved: the relay must never read a partial config.
    staged = f"{relay.config_path}.next"
    await runtime.write(staged, relay_config(upstream, tokens))
    result = await runtime.run(["mv", staged, relay.config_path], {})
    if result.exit_code:
        raise SandboxError(f"writing the relay config failed: {result.stderr[-500:]}")
