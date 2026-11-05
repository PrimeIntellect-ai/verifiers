"""Host side of the in-box relay (`server.py`).

A harness in a remote box can reach the interception server through a loopback relay in
the box, at `RELAY_URL`, with the box tokens of the relay config as its secrets. The relay
forwards to the current upstream and swaps each box token for its upstream token. When
the upstream URL changes (the server replaced a lost tunnel), the host points it at the
new one:

1. Write `relay_config(upstream, tokens)` to `RELAY_CONFIG_PATH`, creating `RELAY_DIR`.
2. Start `[*await runtime.prepare_uv_script(RELAY_SOURCE, activate=False),
   *relay_args()]` in the background, and wait for `RELAY_READY_PATH` to exist. It
   holds the listening port; `RELAY_PID_PATH` holds the process id.
3. To reroute, rewrite `RELAY_CONFIG_PATH` and run `RELAY_RELOAD_COMMAND`. The relay
   re-reads the config, aborts every in-flight upstream request and retries each one
   that has sent its client only the status, headers and SSE comments. A request that
   already streamed data downstream has its client connection closed instead.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from verifiers.v1.errors import SandboxError

if TYPE_CHECKING:
    from verifiers.v1.runtimes.runtime import Runtime

RELAY_SOURCE = (Path(__file__).resolve().parent / "server.py").read_text()
RELAY_DIR = "/tmp/vf-relay"
RELAY_CONFIG_PATH = f"{RELAY_DIR}/config.json"
"""The relay's config: `RelayConfig` as JSON. It holds upstream secrets."""
RELAY_READY_PATH = f"{RELAY_DIR}/port"
"""Written with the listening port (decimal, newline-terminated) once the relay accepts
connections."""
RELAY_PID_PATH = f"{RELAY_DIR}/pid"
"""Written with the relay's process id before `RELAY_READY_PATH`."""
RELAY_PORT = 47821
"""A fixed port, so the harness URL is known before the relay starts. Port 0 picks a
free one instead, read back from `RELAY_READY_PATH`."""
RELAY_URL = f"http://127.0.0.1:{RELAY_PORT}"
RELAY_RELOAD_COMMAND = ["sh", "-c", f'kill -HUP "$(cat {RELAY_PID_PATH})"']
"""Make the relay re-read `RELAY_CONFIG_PATH` (SIGHUP). An invalid config is logged and
ignored, keeping the old upstream."""


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
    """The contents of `RELAY_CONFIG_PATH`; `options` sets the other `RelayConfig` fields."""
    return (
        RelayConfig(upstream=upstream, tokens=tokens, **options)
        .model_dump_json()
        .encode()
    )


def relay_args(
    port: int = RELAY_PORT,
    config: str = RELAY_CONFIG_PATH,
    ready: str = RELAY_READY_PATH,
    pid: str = RELAY_PID_PATH,
) -> list[str]:
    """The relay's arguments, after the argv of its prepared script."""
    return ["--config", config, "--port", str(port), "--ready", ready, "--pid", pid]


RELAY_START_SECONDS = 60
"""How long a started relay gets to accept connections."""


async def start(
    runtime: Runtime, name: str, upstream: str, tokens: dict[str, str]
) -> dict[str, Any]:
    """Start the relay in `runtime`'s box, forwarding to `upstream` with `tokens`, and
    return what a later process needs to reroute it (`reroute`). `name` labels its log."""
    await _write_config(runtime, upstream, tokens)
    program = await runtime.prepare_uv_script(RELAY_SOURCE, activate=False)
    await runtime.run_background(
        [*program, *relay_args()], {}, log=f"{RELAY_DIR}/{name}.log"
    )
    deadline = time.monotonic() + RELAY_START_SECONDS
    while (await runtime.run(["test", "-s", RELAY_READY_PATH], {})).exit_code:
        if time.monotonic() > deadline:
            raise SandboxError(f"the relay did not start within {RELAY_START_SECONDS}s")
        await asyncio.sleep(0.2)
    return {"url": RELAY_URL}


async def reroute(
    runtime: Runtime, relay: dict[str, Any], upstream: str, tokens: dict[str, str]
) -> None:
    """Point a running relay (`start`) at `upstream`, swapping its box tokens for
    `tokens`' values. Its requests in flight retry against the new upstream."""
    await _write_config(runtime, upstream, tokens)
    result = await runtime.run(RELAY_RELOAD_COMMAND, {})
    if result.exit_code:
        raise SandboxError(f"the relay did not reload: {result.stderr.strip()[-500:]}")


async def _write_config(
    runtime: Runtime, upstream: str, tokens: dict[str, str]
) -> None:
    await runtime.run(["mkdir", "-p", RELAY_DIR], {})
    # Written whole, then moved: the relay must never read a partial config.
    staged = f"{RELAY_CONFIG_PATH}.next"
    await runtime.write(staged, relay_config(upstream, tokens))
    result = await runtime.run(
        ["sh", "-c", f"mv {shlex.quote(staged)} {shlex.quote(RELAY_CONFIG_PATH)}"], {}
    )
    if result.exit_code:
        raise SandboxError(f"writing the relay config failed: {result.stderr[-500:]}")
