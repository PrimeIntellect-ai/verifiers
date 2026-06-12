"""The user simulator: a first-class conversation partner, served like a tool server.

A taskset registers a `User` via `Taskset.user` — structurally a `Tools`
(an MCP server with a runtime), exposing a single `respond` tool. Unlike a tool server,
the user simulator is never handed to the model: the framework drives it. After each model
turn the interception server calls `respond` with the model's last message, appends the
simulated user message(s) to the conversation, and re-prompts the model — so a multi-turn
game (e.g. TextArena) plays out as alternating assistant/user turns recorded in the trace,
with the harness and its program none the wiser.

`serve_user` brings the user server up for a rollout (colocated, like a tool server) and
yields the async `respond` the interception server drives; `connect_user` is the MCP client
side. `respond(message)` takes the model's last assistant text and returns the next user
messages plus whether the conversation is done.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from verifiers.v1.errors import ProgramError, RolloutError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.tools import Tools, serve_tools
from verifiers.v1.types import Messages

logger = logging.getLogger(__name__)

# The colocated user server is up once its in-runtime probe passes, but under high concurrency
# it can still momentarily refuse a host connection. Retry the connect before giving up so a
# transient refusal doesn't fail the rollout.
_USER_CONNECT_ATTEMPTS = 12
_USER_CONNECT_BACKOFF = 0.2  # seconds, exponential up to the cap
_USER_CONNECT_MAX_BACKOFF = 2.0


@dataclass(frozen=True)
class User(Tools):
    """A user simulator — structurally a tool server (an MCP server with a runtime), but
    consumed by the framework, not the model. Its single `respond` tool maps the model's
    last message to the next user message(s) and a done flag."""


# The model's last assistant text in; the next user messages + a done flag out.
Respond = Callable[[str], Awaitable[tuple[Messages, bool]]]


@contextlib.asynccontextmanager
async def connect_user(url: str) -> AsyncIterator[Respond]:
    """Open an MCP client session to a user server at `url` and yield an async
    `respond(message)` that calls its `respond` tool, parsing the JSON it returns
    (`{"messages": [...], "done": bool}`) into typed `(messages, done)`.

    Retries the connect — under high concurrency the colocated user server can be slow to
    accept (or briefly refuse) a connection. A server that stays unreachable raises
    `ProgramError` (a captured, retryable rollout error), so a transport failure never escapes
    as a raw `ExceptionGroup`/`ConnectError` that would bypass rollout error handling and crash
    the batch. The connect is entered and exited in this one frame so anyio's cancel scopes stay
    correctly nested."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from verifiers.v1.interception import parse_message

    last_exc: Exception | None = None
    for attempt in range(_USER_CONNECT_ATTEMPTS):
        connected = False
        try:
            async with (
                streamable_http_client(url) as (read, write, *_),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                connected = True

                async def respond(message: str) -> tuple[Messages, bool]:
                    result = await session.call_tool("respond", {"message": message})
                    texts = [
                        b.text
                        for b in result.content
                        if getattr(b, "type", None) == "text"
                    ]
                    data = json.loads("\n".join(texts))
                    messages = [parse_message(m) for m in data["messages"]]
                    return messages, bool(data["done"])

                yield respond
            return
        except RolloutError:
            raise  # a real rollout error surfaced after connecting: propagate as-is
        except Exception as e:
            if connected:
                # the user-sim connection broke mid-rollout/teardown (e.g. the colocated server
                # was killed under memory pressure): capture it as a retryable rollout error
                # instead of letting the raw transport ExceptionGroup escape and crash the batch
                raise ProgramError(
                    f"user server at {url} connection lost: {e!r}"
                ) from e
            last_exc = e  # the connect itself failed: back off and retry
            await asyncio.sleep(
                min(_USER_CONNECT_BACKOFF * 2**attempt, _USER_CONNECT_MAX_BACKOFF)
            )
    raise ProgramError(
        f"user server at {url} unreachable after {_USER_CONNECT_ATTEMPTS} attempts: {last_exc!r}"
    )


@contextlib.asynccontextmanager
async def serve_user(
    user: Tools | None,
    runtime_config: RuntimeConfig,
    agent_runtime: "Runtime | None" = None,
    colocated: bool = True,
) -> AsyncIterator[Respond | None]:
    """Bring a rollout's user server up and yield the async `respond` the interception server
    drives — or `None` when the taskset has no user server. The framework always drives the user
    from the host. When `colocated` (and an `agent_runtime` is given) the server runs inside the
    agent's already-started runtime — reusing it, so there's no extra runtime/subprocess setup
    per rollout — with its port published back to the host (`host_reachable`). Otherwise it runs
    in its OWN runtime (`runtime_config`), for a remote sandbox that can't publish the colocated
    port back. Either way it's host-reachable: a remote sandbox's `public_url`, else localhost."""
    if user is None:
        yield None
        return
    if colocated and agent_runtime is not None:
        async with serve_tools(
            [user], agent_runtime, colocated=True, host_reachable=True
        ) as urls:
            async with connect_user(next(iter(urls.values()))) as respond:
                yield respond
        return
    runtime = make_runtime(runtime_config)
    await runtime.start()
    try:
        async with serve_tools(
            [user], runtime, colocated=True, host_reachable=True
        ) as urls:
            async with connect_user(next(iter(urls.values()))) as respond:
                yield respond
    finally:
        await runtime.stop()
